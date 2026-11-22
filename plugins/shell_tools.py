"""受控白名单 Shell 命令插件（档位 1）：shlex 解析 + shell=False + 程序/路径/子命令白名单。"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from personal_assistant.agent.tools import tool
from personal_assistant.assistant.common import truncate_or_offload

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ALLOWED_EXECUTABLES = (
    "ls",
    "find",
    "grep",
    "rg",
    "wc",
    "head",
    "tail",
    "cat",
    "pwd",
    "date",
    "df",
    "git",
    "pytest",
)
DEFAULT_GIT_SUBCOMMANDS = ("status", "log", "diff", "show", "branch", "rev-parse")
DEFAULT_REJECT_SEGMENTS = (".env", ".ssh", "state", "uploads", "models.dsl", "models.json")
DANGEROUS_FIND_FLAGS = frozenset(
    {"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fls", "-fprint", "-fprint0", "-fprintf"}
)
DANGEROUS_RG_FLAGS = frozenset({"--no-ignore", "--no-ignore-dot", "--hidden", "-.", "-u", "-uu", "-uuu", "--unrestricted"})
DANGEROUS_GIT_PREFIXES = (
    "-c",
    "--config-env",
    "--git-dir",
    "--work-tree",
    "--exec-path",
    "--upload-pack",
    "--receive-pack",
    "--ext-cmd",
)
SHELL_OPERATORS = frozenset({";", "|", "||", "&", "&&", ">", ">>", "<", "<<", "(", ")"})


# 受控命令执行器的白名单策略配置（从 config/tools.json 的 shell 段读取）
@dataclass(frozen=True)
class ShellPolicy:
    project_root: Path
    cwd: Path
    allowed_executables: tuple[str, ...] = DEFAULT_ALLOWED_EXECUTABLES
    allowed_git_subcommands: tuple[str, ...] = DEFAULT_GIT_SUBCOMMANDS
    reject_segments: tuple[str, ...] = DEFAULT_REJECT_SEGMENTS
    timeout: float = 10.0
    max_output_bytes: int = 8192
    offload_dir: Path | None = None

    # 从 config/tools.json 读取 shell 白名单配置，缺失时回退到安全默认值
    @classmethod
    def from_config(cls, config_path: Path | None = None) -> "ShellPolicy":
        cfg_file = (
            config_path
            if config_path is not None
            else Path(os.getenv("MY_HELPER_CONFIG", str(REPO_ROOT / "config"))) / "tools.json"
        ).resolve()
        project_root = cfg_file.parent.parent if cfg_file.is_file() else REPO_ROOT
        shell_cfg: dict[str, object] = {}
        if cfg_file.is_file():
            raw = json.loads(cfg_file.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("shell"), dict):
                shell_cfg = raw["shell"]

        cwd_raw = str(shell_cfg.get("cwd") or ".").strip()
        cwd_rel = Path(cwd_raw)
        if cwd_rel.is_absolute() or ".." in cwd_rel.parts:
            raise ValueError(f"shell.cwd 必须是项目内相对路径：{cwd_raw}")
        cwd = (project_root / cwd_rel).resolve()
        if not cwd.is_relative_to(project_root.resolve()) or not cwd.is_dir():
            raise ValueError(f"shell.cwd 不存在或超出项目范围：{cwd_raw}")

        allowed_exec = shell_cfg.get("allowed_executables", list(DEFAULT_ALLOWED_EXECUTABLES))
        allowed_git = shell_cfg.get("allowed_git_subcommands", list(DEFAULT_GIT_SUBCOMMANDS))
        rejects = shell_cfg.get("reject_segments", list(DEFAULT_REJECT_SEGMENTS))
        timeout = float(shell_cfg.get("timeout", 10.0))
        max_output = int(shell_cfg.get("max_output_bytes", 8192))

        return cls(
            project_root=project_root.resolve(),
            cwd=cwd,
            allowed_executables=tuple(str(x) for x in allowed_exec),
            allowed_git_subcommands=tuple(str(x) for x in allowed_git),
            reject_segments=tuple(str(x) for x in rejects),
            timeout=timeout,
            max_output_bytes=max_output,
        )


# 词法解析单条命令并拦截未引用的管道/分号/重定向/命令替换等壳元字符
def _parse_single_command(command: str) -> list[str]:
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command 不能为空。")
    if any(ch in command for ch in ("\x00", "\n", "\r")):
        raise ValueError("只允许执行单行命令，不能包含换行或 NUL。")
    if "$(" in command or "`" in command:
        raise ValueError("禁止使用命令替换语法（$() 或反引号）。")

    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    tokens = list(lexer)
    if not tokens:
        raise ValueError("command 不能为空。")
    for tok in tokens:
        if tok in SHELL_OPERATORS:
            raise ValueError(f"不支持壳操作符「{tok}」（管道/拼接/重定向）；请拆分为单条命令调用。")
    return tokens


# 校验单个路径参数：拒绝绝对路径、.. 穿越、敏感目录段以及指向外部或敏感文件的软链接
def _validate_path_candidate(raw_value: str, policy: ShellPolicy) -> None:
    if not raw_value:
        return
    if raw_value.startswith("~") or Path(raw_value).is_absolute():
        raise ValueError(f"禁止访问绝对路径或家目录路径：{raw_value}")
    rel = Path(raw_value.replace("\\", "/"))
    if ".." in rel.parts:
        raise ValueError(f"禁止路径穿越（..）：{raw_value}")
    for part in rel.parts:
        if part in policy.reject_segments:
            raise ValueError(f"禁止访问受保护路径：{part}")

    candidate = policy.cwd / rel
    if candidate.exists() or candidate.is_symlink():
        resolved = candidate.resolve(strict=False)
        if not resolved.is_relative_to(policy.project_root):
            raise ValueError(f"目标路径超出项目根目录：{raw_value}")
        for part in resolved.relative_to(policy.project_root).parts:
            if part in policy.reject_segments:
                raise ValueError(f"禁止访问受保护目标：{part}")


# 对 git / find 等高能力命令做子命令与危险选项收口检查
def _validate_program_rules(argv: list[str], policy: ShellPolicy) -> None:
    exe, args = argv[0], argv[1:]
    if exe == "git":
        subcmd: str | None = None
        for arg in args:
            for bad in DANGEROUS_GIT_PREFIXES:
                if arg == bad or arg.startswith(f"{bad}="):
                    raise ValueError(f"禁止使用高危 git 选项：{arg}")
            if subcmd is None and not arg.startswith("-"):
                subcmd = arg
        if subcmd is None or subcmd not in policy.allowed_git_subcommands:
            allowed = ", ".join(policy.allowed_git_subcommands)
            raise ValueError(f"git 子命令不在白名单内（允许：{allowed}）。")
    elif exe == "find":
        for arg in args:
            if arg in DANGEROUS_FIND_FLAGS:
                raise ValueError(f"find 禁止使用危险选项：{arg}")
    elif exe == "rg":
        for arg in args:
            if arg in DANGEROUS_RG_FLAGS or arg.startswith("--no-ignore"):
                raise ValueError(f"rg 禁止绕过忽略规则或扫描隐藏文件：{arg}")


# 为 grep 自动追加敏感文件/目录排除项，防止递归搜索（grep -r）泄漏 .env 等受保护文件
def _augment_safe_argv(argv: list[str], policy: ShellPolicy) -> list[str]:
    if argv[0] != "grep":
        return argv
    excludes: list[str] = []
    for seg in policy.reject_segments:
        excludes.append(f"--exclude={seg}")
        excludes.append(f"--exclude-dir={seg}")
    excludes.append("--exclude-dir=.git")
    return [argv[0], *excludes, *argv[1:]]


# 合并 stdout/stderr 并接入 Head+Tail TOAST 输出卸载，保持退出码始终位于首行
def _format_command_output(
    returncode: int,
    stdout: str,
    stderr: str,
    *,
    max_output_bytes: int,
    tag: str,
    offload_dir: Path | None,
) -> str:
    clean_out = (stdout or "").strip()
    clean_err = (stderr or "").strip()
    sections: list[str] = []
    if clean_out:
        sections.append(clean_out)
    if clean_err:
        sections.append(f"stderr:\n{clean_err}")
    if not sections:
        return f"退出码：{returncode}"
    body = truncate_or_offload(
        "\n".join(sections),
        max_output_bytes,
        tag=tag,
        mode="head_tail",
        offload_dir=offload_dir,
    )
    return f"退出码：{returncode}\n{body}"


# 在安全策略护栏内执行单条受控命令（shell=False、脱敏环境、超时与 Head+Tail TOAST 卸载）
def execute_safe_command(command: str, policy: ShellPolicy) -> str:
    try:
        argv = _parse_single_command(command)
        exe = argv[0]
        if "/" in exe or "\\" in exe:
            raise ValueError("命令名不能包含路径前缀，只能使用白名单程序名。")
        if exe not in policy.allowed_executables:
            allowed = ", ".join(policy.allowed_executables)
            raise ValueError(f"命令 {exe} 不在白名单内；允许程序：{allowed}")

        _validate_program_rules(argv, policy)
        for arg in argv[1:]:
            if arg.startswith("-"):
                if "=" in arg:
                    _, rhs = arg.split("=", 1)
                    _validate_path_candidate(rhs, policy)
                continue
            _validate_path_candidate(arg, policy)
    except ValueError as error:
        return f"错误：{error}"

    final_argv = _augment_safe_argv(argv, policy)
    # 使用 absolute() 而非 resolve()，保留 .venv/bin/python 软链所在目录以便定位 .venv/bin/pytest
    venv_bin = str(Path(sys.executable).absolute().parent)
    safe_path = f"{venv_bin}:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
    executable = shutil.which(final_argv[0], path=safe_path)
    if not executable:
        return f"错误：找不到可执行程序 {final_argv[0]}。"

    env = {
        "PATH": safe_path,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HOME": str(policy.cwd),
        "TMPDIR": tempfile.gettempdir(),
    }
    try:
        done = subprocess.run(
            (executable, *final_argv[1:]),
            shell=False,
            cwd=policy.cwd,
            timeout=policy.timeout,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return f"错误：命令执行超时（>{policy.timeout:g}s）。"
    except OSError as error:
        return f"错误：命令启动失败（{error}）。"

    return _format_command_output(
        done.returncode,
        done.stdout or "",
        done.stderr or "",
        max_output_bytes=policy.max_output_bytes,
        tag=f"shell-{final_argv[0]}",
        offload_dir=policy.offload_dir,
    )


# 插件工具入口：受 config/tools.json 的 shell 段白名单管控执行单条只读/受控命令
@tool(
    name="run_command",
    description=(
        "在项目目录内执行单条受白名单保护的本地命令（如 ls、find、grep、rg、wc、head、tail、"
        "cat、pwd、date、df、git 只读子命令、pytest）。不支持管道或命令拼接，禁止访问 .env 等敏感路径；"
        "超长输出保留头尾预览并自动保存至 data/workspace/tool_outputs/ 供分页读取。"
    ),
    parameters={"command": "要执行的单条命令字符串，例如 'ls -la' 或 'git log -n 5 --oneline'"},
)
def run_command(command: str) -> str:
    policy = ShellPolicy.from_config()
    return execute_safe_command(command, policy)
