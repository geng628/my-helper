"""受控白名单 Shell 命令插件（档位 1）：shlex 解析 + shell=False + 程序/路径/子命令白名单。"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from personal_assistant.agent.tools import tool
from personal_assistant.assistant.common import truncate_or_offload
from personal_assistant.assistant.files_policy import (
    DEFAULT_REJECT_SEGMENTS,
    collect_protected_inodes,
    hits_protected_inode,
    is_sensitive_segment,
    normalize_segment,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_ALLOWED_EXECUTABLES = ("ls", "find", "grep", "rg", "wc", "head", "tail", "cat", "pwd", "date", "df", "git", "pytest")
DEFAULT_GIT_SUBCOMMANDS = ("status", "log", "diff", "show", "branch", "rev-parse")
_MAX_CAPTURE_BYTES = 1024 * 1024

DANGEROUS_FIND_FLAGS = frozenset(
    "-exec -execdir -ok -okdir -delete -fls -fprint -fprint0 -fprintf -files0-from -f".split()
)
GIT_STATUS_FLAGS = frozenset({"-s", "--short", "-b", "--branch", "--porcelain"})
GIT_BRANCH_FLAGS = frozenset(
    {"--list", "-l", "-a", "--all", "-r", "--remotes", "-v", "-vv", "--verbose", "--show-current"}
)
GIT_LOG_BOOL_FLAGS = frozenset({"--oneline", "--stat", "--name-only", "--name-status", "--graph", "--decorate"})
GIT_DIFF_FLAGS = frozenset({"--stat", "--name-only", "--name-status", "--cached", "--staged"})
GIT_SHOW_SUMMARY_FLAGS = frozenset({"--stat", "--name-only", "--name-status", "-s", "--no-patch"})
GIT_SHOW_FLAGS = GIT_SHOW_SUMMARY_FLAGS | frozenset({"--oneline"})
GIT_REV_PARSE_FLAGS = frozenset(
    {"--short", "--abbrev-ref", "--show-toplevel", "--is-inside-work-tree", "--verify", "--symbolic-full-name"}
)
GIT_SUPPORTED_SUBCOMMANDS = frozenset({"status", "branch", "log", "diff", "show", "rev-parse"})

PYTEST_SHORT_BOOL, PYTEST_SHORT_VAL = frozenset("vqxsl"), frozenset("km")
PYTEST_LONG_BOOL = frozenset(
    "--ff --lf --nf --collect-only --no-header --verbose --quiet --exitfirst --co --no-summary --disable-warnings".split()
)
PYTEST_LONG_VAL: dict[str, frozenset[str] | str] = {
    "--tb": frozenset({"auto", "long", "short", "line", "native", "no"}), "--maxfail": "int", "--durations": "int",
}
_COMMON_LONG_BOOL = frozenset(
    "--ignore-case --invert-match --word-regexp --line-regexp --count --files-with-matches "
    "--line-number --no-filename --only-matching --quiet --version".split()
)
_COMMON_LONG_VAL = {
    "--regexp": "pattern", "--max-count": "int", "--after-context": "int",
    "--before-context": "int", "--context": "int", "--color": "word",
}
_COMMON_SHORT_VAL = {"e": "pattern", "m": "int", "A": "int", "B": "int", "C": "int"}
SEARCH_SPECS = {
    "grep": {
        "short_bool": frozenset("EFGPivwxcLlnhHoqsaIrV"),
        "short_val": {**_COMMON_SHORT_VAL, "d": "action"},
        "forbidden_short": frozenset("fR"),
        "long_bool": _COMMON_LONG_BOOL | frozenset({"--extended-regexp", "--fixed-strings", "--with-filename", "--recursive"}),
        "long_val": {**_COMMON_LONG_VAL, "--include": "glob", "--exclude": "path", "--exclude-dir": "path"},
        "forbidden_long": frozenset({"--file", "--exclude-from", "--dereference-recursive"}),
    },
    "rg": {
        "short_bool": frozenset("ivwxclnHhoqsaFSIV"),
        "short_val": {**_COMMON_SHORT_VAL, "t": "type", "T": "type", "g": "glob"},
        "forbidden_short": frozenset("f.uL"),
        "long_bool": _COMMON_LONG_BOOL | frozenset({"--smart-case", "--fixed-strings", "--multiline", "--files", "--json"}),
        "long_val": {**_COMMON_LONG_VAL, "--type": "type", "--type-not": "type", "--glob": "glob", "--iglob": "glob"},
        "forbidden_long": frozenset("--file --hidden --unrestricted --follow --ignore-file --pre --pre-glob --hostname-bin".split()),
    },
}
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

    # 从已解析的 shell 配置字典与上下文路径直接构建 ShellPolicy（支持绑定实例专属 offload_dir）
    @classmethod
    def from_dict(
        cls, shell_cfg: Mapping[str, object], *, project_root: Path = REPO_ROOT, offload_dir: Path | None = None
    ) -> "ShellPolicy":
        cwd_raw = str(shell_cfg.get("cwd") or ".").strip()
        cwd_rel = Path(cwd_raw)
        if cwd_rel.is_absolute() or ".." in cwd_rel.parts:
            raise ValueError(f"shell.cwd 必须是项目内相对路径：{cwd_raw}")
        cwd = (project_root / cwd_rel).resolve()
        if not cwd.is_relative_to(project_root.resolve()) or not cwd.is_dir():
            raise ValueError(f"shell.cwd 不存在或超出项目范围：{cwd_raw}")
        return cls(
            project_root=project_root.resolve(),
            cwd=cwd,
            allowed_executables=tuple(str(x) for x in shell_cfg.get("allowed_executables", list(DEFAULT_ALLOWED_EXECUTABLES))),
            allowed_git_subcommands=tuple(str(x) for x in shell_cfg.get("allowed_git_subcommands", list(DEFAULT_GIT_SUBCOMMANDS))),
            reject_segments=tuple(str(x) for x in shell_cfg.get("reject_segments", list(DEFAULT_REJECT_SEGMENTS))),
            timeout=float(shell_cfg.get("timeout", 10.0)),
            max_output_bytes=int(shell_cfg.get("max_output_bytes", 8192)),
            offload_dir=offload_dir,
        )

    # 从 config/tools.json 读取 shell 白名单配置，缺失时回退到安全默认值
    @classmethod
    def from_config(cls, config_path: Path | None = None, *, offload_dir: Path | None = None) -> "ShellPolicy":
        cfg_file = (
            config_path if config_path is not None
            else Path(os.getenv("MY_HELPER_CONFIG", str(REPO_ROOT / "config"))) / "tools.json"
        ).resolve()
        project_root = cfg_file.parent.parent if cfg_file.is_file() else REPO_ROOT
        shell_cfg: dict[str, object] = {}
        if cfg_file.is_file():
            raw = json.loads(cfg_file.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("shell"), dict):
                shell_cfg = raw["shell"]
        return cls.from_dict(shell_cfg, project_root=project_root, offload_dir=offload_dir)


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


# 判断单个路径段是否命中受保护敏感名称（复用 files_policy 的 Unicode 折叠检测，仅放行 .env.example）
def _is_rejected_segment(part: str, reject_segments: tuple[str, ...]) -> bool:
    return is_sensitive_segment(part, reject_segments)


# 检查带通配符表达式是否可能经过 Unicode 折叠后命中敏感路径段或 .env.* / .tmp-models-* 变体
def _matches_sensitive_glob(glob_expr: str, reject_segments: tuple[str, ...]) -> bool:
    if not any(ch in glob_expr for ch in "*?["):
        return False
    samples = {normalize_segment(seg) for seg in reject_segments} | {
        ".env", ".env.local", ".env.production", ".env.staging", ".env.development", ".tmp-models-x",
    }
    norm = normalize_segment(glob_expr.replace("\\", "/"))
    cands = {norm, *(p for p in Path(norm).parts if any(ch in p for ch in "*?["))}
    return any(fnmatch.fnmatchcase(sample, cand) for cand in cands for sample in samples)


# 收集当前 ShellPolicy 对应的停止边界目录与受保护内核 Inode 集合
def _protected_inode_context(policy: ShellPolicy) -> tuple[list[Path], set[tuple[int, int]]]:
    root_res = policy.project_root.resolve(strict=False)
    scan_roots: list[Path] = [root_res, root_res / "config", root_res / "data"]
    stop_roots: list[Path] = [root_res]
    if policy.offload_dir is not None:
        off_res = policy.offload_dir.resolve(strict=False)
        scan_roots.extend([off_res, off_res.parent])
        stop_roots.append(off_res.parent)
        if off_res.parent.parent != off_res.parent:
            scan_roots.append(off_res.parent.parent)
            stop_roots.append(off_res.parent.parent)
    return stop_roots, collect_protected_inodes(scan_roots, policy.reject_segments)


# 校验单个路径参数：拒绝绝对路径、.. 穿越、Unicode 敏感目录段、内核 Inode 命中及越权软链接
def _validate_path_candidate(
    raw_value: str,
    policy: ShellPolicy,
    *,
    protected_ctx: tuple[list[Path], set[tuple[int, int]]] | None = None,
) -> None:
    if not raw_value:
        return
    if raw_value.startswith("~") or Path(raw_value).is_absolute():
        raise ValueError(f"禁止访问绝对路径或家目录路径：{raw_value}")
    rel = Path(raw_value.replace("\\", "/"))
    if ".." in rel.parts:
        raise ValueError(f"禁止路径穿越（..）：{raw_value}")
    for part in rel.parts:
        if _is_rejected_segment(part, policy.reject_segments):
            raise ValueError(f"禁止访问受保护路径：{part}")

    stop_roots, protected_inodes = protected_ctx or _protected_inode_context(policy)
    root_res = policy.project_root.resolve(strict=False)
    candidate = policy.cwd / rel
    if hits_protected_inode(candidate, stop_roots, protected_inodes):
        raise ValueError(f"禁止访问受保护路径：{raw_value}")

    # 逐级排查软链接（即使末端文件不存在，中间目录软链接也不得逃出项目或落入受保护目标）
    cur = policy.cwd
    for part in rel.parts:
        cur = cur / part
        if cur.is_symlink():
            cur_res = cur.resolve(strict=False)
            if not cur_res.is_relative_to(root_res):
                raise ValueError(f"目标路径超出项目根目录：{raw_value}")
            for rpart in cur_res.relative_to(root_res).parts:
                if _is_rejected_segment(rpart, policy.reject_segments):
                    raise ValueError(f"禁止访问受保护目标：{rpart}")
            if hits_protected_inode(cur_res, stop_roots, protected_inodes):
                raise ValueError(f"禁止访问受保护目标：{raw_value}")

    if candidate.exists() or candidate.is_symlink():
        resolved = candidate.resolve(strict=False)
        if not resolved.is_relative_to(root_res):
            raise ValueError(f"目标路径超出项目根目录：{raw_value}")
        for part in resolved.relative_to(root_res).parts:
            if _is_rejected_segment(part, policy.reject_segments):
                raise ValueError(f"禁止访问受保护目标：{part}")
        if hits_protected_inode(resolved, stop_roots, protected_inodes):
            raise ValueError(f"禁止访问受保护目标：{raw_value}")


# 校验搜索/文件过滤 glob（禁止正向匹配隐藏路径或经 Unicode 折叠命中敏感段，放行排除式 !glob）
def _validate_safe_glob(raw_glob: str, policy: ShellPolicy) -> None:
    if not raw_glob:
        raise ValueError("glob 参数不能为空。")
    negated = raw_glob.startswith("!")
    cleaned = raw_glob[1:] if negated else raw_glob
    _validate_path_candidate(cleaned, policy)
    if negated:
        return
    norm = normalize_segment(cleaned.replace("\\", "/"))
    if cleaned != ".env.example" and (
        norm.startswith(".") or "/." in norm or any(p.startswith(".") for p in Path(norm).parts)
    ):
        raise ValueError(f"禁止通过 glob 正向匹配隐藏路径：{raw_glob}")
    if _matches_sensitive_glob(cleaned, policy.reject_segments) or (".env" in norm and cleaned != ".env.example"):
        raise ValueError(f"禁止通过 glob 匹配受保护路径：{raw_glob}")


# 从内联参数或下一位置提取带值选项的取值并推进扫描游标
def _consume_opt_value(args: list[str], idx: int, inline: str | None, label: str) -> tuple[str, int]:
    if inline is not None:
        return inline, idx + 1
    if idx + 1 >= len(args):
        raise ValueError(f"{label} 缺少参数值。")
    return args[idx + 1], idx + 2


# 校验 pytest 参数并返回是否已显式传入测试目标（仅纯 --version/-V 可跳过 tests/ 检查）
def _parse_pytest_args(args: list[str], policy: ShellPolicy) -> bool:
    if args in (["--version"], ["-V"]):
        return True
    root_res = policy.project_root.resolve(strict=False)
    tests_entry = root_res / "tests"
    tests_root = tests_entry.resolve(strict=False)
    if not tests_entry.exists() or tests_entry.is_symlink() or not tests_entry.is_dir() or not tests_root.is_relative_to(root_res):
        raise ValueError("pytest 要求项目根目录存在真实的 tests/ 目录。")

    idx, has_target, after_dd = 0, False, False
    while idx < len(args):
        arg = args[idx]
        if after_dd or not arg.startswith("-") or arg == "-":
            file_part = arg.split("::", 1)[0] if "::" in arg else arg
            if not file_part:
                raise ValueError(f"pytest 测试路径无效：{arg}")
            _validate_path_candidate(file_part, policy)
            if not (policy.cwd / Path(file_part.replace("\\", "/"))).resolve(strict=False).is_relative_to(tests_root):
                raise ValueError(f"pytest 只能运行 tests/ 目录内的测试：{arg}")
            has_target, idx = True, idx + 1
            continue
        if arg == "--":
            after_dd, idx = True, idx + 1
            continue
        if arg.startswith("--"):
            opt, has_eq, raw_val = arg.partition("=")
            if opt in PYTEST_LONG_BOOL:
                if has_eq:
                    raise ValueError(f"pytest 布尔选项不接受赋值：{arg}")
                idx += 1
                continue
            if opt in PYTEST_LONG_VAL:
                val, idx = _consume_opt_value(args, idx, raw_val if has_eq else None, f"pytest 选项 {opt}")
                spec = PYTEST_LONG_VAL[opt]
                if (isinstance(spec, frozenset) and val not in spec) or (spec == "int" and not val.isdigit()):
                    raise ValueError(f"pytest 选项 {opt} 取值非法：{val}")
                continue
            raise ValueError(f"pytest 仅允许安全只读运行标志，已拒绝：{arg}")

        next_idx = idx + 1
        for pos, ch in enumerate(arg[1:]):
            if ch in PYTEST_SHORT_VAL:
                rest = arg[pos + 2 :]
                inline = (rest[1:] if rest.startswith("=") else rest) if rest else None
                val, next_idx = _consume_opt_value(args, idx, inline, f"pytest 选项 -{ch}")
                if (inline is None and val.startswith("-")) or not val.strip():
                    raise ValueError(f"pytest 选项 -{ch} 参数值非法：{val}")
                break
            if ch not in PYTEST_SHORT_BOOL:
                raise ValueError(f"pytest 禁止使用高危或未授权短选项：-{ch}（来自 {arg}）")
        idx = next_idx
    return has_target


# 校验 git branch --list/-l 的分支名称匹配模式（允许 * 通配符，拦截路径穿越、魔法语法与敏感段）
def _validate_git_branch_pattern(arg: str, policy: ShellPolicy) -> None:
    if not arg or arg.startswith("~") or ":(" in arg or ":^" in arg or ":!" in arg:
        raise ValueError(f"git branch 模式参数非法：{arg}")
    rel = Path(arg.replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"git branch 模式不能包含绝对路径或 ..：{arg}")
    for part in rel.parts:
        if _is_rejected_segment(part, policy.reject_segments):
            raise ValueError(f"git branch 模式禁止引用受保护名称：{part}")


# 校验 git 位置参数（路径/revision/冒号引用），阻断 pathspec 魔法语法、通配符绕过与敏感路径
def _validate_git_pathspec(
    arg: str,
    policy: ShellPolicy,
    *,
    protected_ctx: tuple[list[Path], set[tuple[int, int]]],
) -> None:
    if ":(" in arg or ":^" in arg or ":!" in arg:
        raise ValueError(f"git 禁止使用 pathspec 魔法语法：{arg}")
    _validate_path_candidate(arg, policy, protected_ctx=protected_ctx)
    if _matches_sensitive_glob(arg, policy.reject_segments):
        raise ValueError(f"git 禁止通配符匹配受保护路径：{arg}")
    if ":" in arg:
        parts = arg.split(":")
        if not parts[-1]:
            raise ValueError(f"git 冒号引用缺少路径目标：{arg}")
        for i in range(1, len(parts)):
            for cand in (parts[i], ":".join(parts[i:]), *parts[i].split("..")):
                if cand:
                    _validate_path_candidate(cand, policy, protected_ctx=protected_ctx)
                    if _matches_sensitive_glob(cand, policy.reject_segments):
                        raise ValueError(f"git 禁止通配符匹配受保护路径：{cand}")


# 解析并严格校验单个 git 只读子命令及选项，返回 (子命令名, 子命令后参数列表)
def _parse_git_command(args: list[str], policy: ShellPolicy) -> tuple[str, list[str]]:
    allowed = ", ".join(policy.allowed_git_subcommands)
    sub_idx = 1 if args[:1] == ["--no-pager"] else 0
    if (
        sub_idx >= len(args)
        or args[sub_idx] not in policy.allowed_git_subcommands
        or args[sub_idx] not in GIT_SUPPORTED_SUBCOMMANDS
    ):
        raise ValueError(f"git 仅允许直接调用白名单只读子命令（允许：{allowed}），禁止前置全局选项。")

    sub = args[sub_idx]
    rest = args[sub_idx + 1 :]
    protected_ctx = _protected_inode_context(policy)

    idx = 0
    after_dd = False
    seen_flags: list[str] = []
    positionals: list[str] = []

    while idx < len(rest):
        arg = rest[idx]
        if arg == "--" and not after_dd:
            after_dd = True
            idx += 1
            continue
        if after_dd or not arg.startswith("-") or arg == "-":
            if sub == "branch":
                _validate_git_branch_pattern(arg, policy)
            else:
                _validate_git_pathspec(arg, policy, protected_ctx=protected_ctx)
            positionals.append(arg)
            idx += 1
            continue

        if sub == "status":
            if arg not in GIT_STATUS_FLAGS:
                raise ValueError(f"git status 不允许选项：{arg}")
            seen_flags.append(arg)
            idx += 1
        elif sub == "branch":
            if arg not in GIT_BRANCH_FLAGS:
                raise ValueError(f"git branch 仅允许只读列举标志，禁止：{arg}")
            seen_flags.append(arg)
            idx += 1
        elif sub == "diff":
            if arg not in GIT_DIFF_FLAGS:
                raise ValueError(f"git diff 不允许选项：{arg}")
            seen_flags.append(arg)
            idx += 1
        elif sub == "show":
            if arg not in GIT_SHOW_FLAGS:
                raise ValueError(f"git show 不允许选项：{arg}")
            seen_flags.append(arg)
            idx += 1
        elif sub == "rev-parse":
            if arg not in GIT_REV_PARSE_FLAGS:
                raise ValueError(f"git rev-parse 不允许选项：{arg}")
            seen_flags.append(arg)
            idx += 1
        elif sub == "log":
            if arg in GIT_LOG_BOOL_FLAGS:
                seen_flags.append(arg)
                idx += 1
            elif arg == "-n":
                val, idx = _consume_opt_value(rest, idx, None, "git log -n")
                if not re.fullmatch(r"[1-9]\d*", val):
                    raise ValueError(f"git log -n 需要正整数：{val}")
                seen_flags.append(arg)
            elif re.fullmatch(r"-n[1-9]\d*", arg) or re.fullmatch(r"-[1-9]\d*", arg):
                seen_flags.append(arg)
                idx += 1
            elif re.fullmatch(r"--max-count=[1-9]\d*", arg):
                seen_flags.append(arg)
                idx += 1
            else:
                raise ValueError(f"git log 不允许选项：{arg}")

    if sub == "branch" and positionals:
        has_list = any(flag in {"--list", "-l"} for flag in seen_flags)
        if not has_list or "--show-current" in seen_flags:
            raise ValueError("git branch 仅允许纯列举只读标志，严禁传入位置参数创建分支。")

    return sub, rest


# 校验 git 参数的统一入口（供验证与外部测试调用）
def _validate_git_args(args: list[str], policy: ShellPolicy) -> None:
    _parse_git_command(args, policy)


# 校验 grep / rg 带值选项参数，阻断非法整数、类型与敏感 glob
def _validate_search_opt_value(exe: str, opt: str, kind: str, val: str, policy: ShellPolicy) -> None:
    if kind == "int" and not val.isdigit():
        raise ValueError(f"{exe} 选项 {opt} 需要非负整数：{val}")
    elif kind == "action" and val not in {"read", "skip", "recurse"}:
        raise ValueError(f"{exe} 选项 {opt} 取值非法：{val}")
    elif kind in {"type", "word"} and not re.fullmatch(r"[A-Za-z0-9_+-]+", val):
        raise ValueError(f"{exe} 选项 {opt} 取值非法：{val}")
    elif kind == "glob":
        _validate_safe_glob(val, policy)
    elif kind == "path":
        _validate_path_candidate(val, policy)


# 解析并校验 grep / rg 参数（全量扫描后再判定路径位置参数，彻底封堵后置 -e/--files 绕过）
def _parse_search_args(exe: str, args: list[str], policy: ShellPolicy) -> tuple[int, bool, bool]:
    spec = SEARCH_SPECS[exe]
    protected_ctx = _protected_inode_context(policy)
    idx, dd_idx, is_recursive, explicit_pattern, after_dd = 0, None, exe == "rg", False, False
    positionals: list[str] = []
    while idx < len(args):
        arg = args[idx]
        if after_dd or not arg.startswith("-") or arg == "-":
            positionals.append(arg)
            idx += 1
            continue
        if arg == "--":
            dd_idx = idx if dd_idx is None else dd_idx
            after_dd, idx = True, idx + 1
            continue
        if arg.startswith("--"):
            opt, has_eq, raw_val = arg.partition("=")
            if opt in spec["forbidden_long"] or (exe == "rg" and opt.startswith("--no-ignore")):
                raise ValueError(f"{exe} 禁止使用高危或绕过忽略规则选项：{arg}")
            if opt in spec["long_bool"]:
                if has_eq:
                    raise ValueError(f"{exe} 布尔选项不接受赋值：{arg}")
                is_recursive = is_recursive or opt == "--recursive"
                explicit_pattern = explicit_pattern or opt == "--files"
                idx += 1
                continue
            if opt in spec["long_val"]:
                val, idx = _consume_opt_value(args, idx, raw_val if has_eq else None, f"{exe} 选项 {opt}")
                kind = spec["long_val"][opt]
                if kind == "pattern":
                    explicit_pattern = True
                else:
                    _validate_search_opt_value(exe, opt, kind, val, policy)
                continue
            raise ValueError(f"{exe} 不支持的长选项：{arg}")

        if exe == "rg" and arg in {"-u", "-uu", "-uuu", "-."}:
            raise ValueError(f"rg 禁止绕过忽略规则或扫描隐藏文件：{arg}")
        next_idx = idx + 1
        for pos, ch in enumerate(arg[1:]):
            if ch in spec["forbidden_short"]:
                raise ValueError(f"{exe} 禁止使用高危或绕过保护短选项：-{ch}（来自 {arg}）")
            if ch == "r":
                is_recursive = True
            if ch in spec["short_val"]:
                rest = arg[pos + 2 :]
                inline = (rest[1:] if rest.startswith("=") else rest) if rest else None
                val, next_idx = _consume_opt_value(args, idx, inline, f"{exe} 选项 -{ch}")
                if ch == "d" and val == "recurse":
                    is_recursive = True
                kind = spec["short_val"][ch]
                if kind == "pattern":
                    explicit_pattern = True
                else:
                    _validate_search_opt_value(exe, f"-{ch}", kind, val, policy)
                break
            if ch not in spec["short_bool"]:
                raise ValueError(f"{exe} 不支持的短选项或连写路径参数：-{ch}（来自 {arg}）")
        idx = next_idx
    targets = positionals if explicit_pattern else positionals[1:]
    for target in targets:
        _validate_path_candidate(target, policy, protected_ctx=protected_ctx)
    only_example = bool(targets) and all(
        (t == ".env.example" or t.endswith("/.env.example")) and not (policy.cwd / Path(t.replace("\\", "/"))).is_dir()
        for t in targets
    )
    return (len(args) if dd_idx is None else dd_idx), is_recursive, only_example


# 按程序名分派专用安全规则与全量参数校验
def _validate_program_rules(argv: list[str], policy: ShellPolicy) -> None:
    exe, args = argv[0], argv[1:]
    if exe == "pytest":
        _parse_pytest_args(args, policy)
    elif exe == "git":
        _validate_git_args(args, policy)
    elif exe in SEARCH_SPECS:
        _parse_search_args(exe, args, policy)
    else:
        protected_ctx = _protected_inode_context(policy)
        after_dd = False
        for arg in args:
            if arg == "--" and not after_dd:
                after_dd = True
                continue
            if not after_dd and arg.startswith("-") and arg != "-":
                if exe == "find" and (
                    arg in DANGEROUS_FIND_FLAGS or arg.startswith("-files0-from") or (arg.startswith("-f") and arg != "-false")
                ):
                    raise ValueError(f"find 禁止使用危险选项：{arg}")
                if exe == "date" and (
                    (not arg.startswith("--") and "f" in arg[1:]) or arg == "--file" or arg.startswith("--file=")
                ):
                    raise ValueError(f"date 禁止从文件读取输入：{arg}")
                if arg.startswith("--"):
                    if arg == "--files0-from" or arg.startswith("--files0-from="):
                        raise ValueError(f"{exe} 禁止从文件读取路径列表：{arg}")
                    if "=" in arg:
                        _validate_path_candidate(arg.split("=", 1)[1], policy, protected_ctx=protected_ctx)
                    continue
                if not re.fullmatch(r"-[A-Za-z0-9]+", arg):
                    raise ValueError(f"{exe} 禁止使用未知连写路径短选项：{arg}")
                continue
            _validate_path_candidate(arg, policy, protected_ctx=protected_ctx)


# 将路径转换为相对 cwd 的字符串以简化 --ignore 目标参数
def _rel_to_cwd(path: Path, cwd: Path) -> str:
    return str(path.relative_to(cwd)) if path.is_relative_to(cwd) else str(path)


# 为 grep/rg 注入敏感排除项，为 pytest 注入 --ignore 与 tests 目标，为 git show/diff 补齐安全限制与负向排除
def _augment_safe_argv(argv: list[str], policy: ShellPolicy) -> list[str]:
    exe, args = argv[0], argv[1:]
    if exe in SEARCH_SPECS:
        insert_at, is_recursive, only_example = _parse_search_args(exe, args, policy)
        add_env = (not only_example) if exe == "rg" else (is_recursive or not only_example)
        pats = (
            *policy.reject_segments,
            *((".git",) if ".git" not in policy.reject_segments else ()),
            *((".env.*",) if add_env else ()),
            ".tmp-models-*",
        )
        fmts = ("--exclude={}", "--exclude-dir={}") if exe == "grep" else ("--glob=!{}", "--glob=!**/{}/**")
        excludes = [fmt.format(seg) for seg in pats for fmt in fmts]
        return [exe, *args[:insert_at], *excludes, *args[insert_at:]]

    if exe == "pytest":
        if args in (["--version"], ["-V"]):
            return argv
        root_res, cwd_res = policy.project_root.resolve(strict=False), policy.cwd.resolve(strict=False)
        ignores = [f"--ignore={_rel_to_cwd(root_res / 'data', cwd_res)}"]
        if policy.offload_dir is not None:
            off_res = policy.offload_dir.resolve(strict=False)
            if off_res.is_relative_to(root_res) and (rel_off := off_res.relative_to(root_res)).parts and rel_off.parts[0] != "data":
                ignores.append(f"--ignore={_rel_to_cwd(root_res / rel_off.parts[0], cwd_res)}")
        has_target = _parse_pytest_args(args, policy)
        tail = [] if has_target else [_rel_to_cwd((root_res / "tests").resolve(strict=False), cwd_res)]
        return [exe, *ignores, *args, *tail]

    if exe == "git":
        sub, rest = _parse_git_command(args, policy)
        if sub == "show":
            before_dd = rest[: rest.index("--")] if "--" in rest else rest
            if not any(tok in GIT_SHOW_SUMMARY_FLAGS for tok in before_dd):
                sub_pos = args.index("show") + 1
                return [exe, *args[:sub_pos], "--stat", *args[sub_pos:]]
        if sub == "diff":
            diff_pats = (*policy.reject_segments, ".env.*", ".tmp-models-*")
            neg_specs: list[str] = []
            for seg in diff_pats:
                if seg == ".git":
                    continue
                neg_specs.extend(
                    (
                        f":(exclude,glob,icase){seg}",
                        f":(exclude,glob,icase)**/{seg}",
                        f":(exclude,glob,icase)**/{seg}/**",
                    )
                )
            if "--" in args:
                dd_pos = args.index("--")
                has_pos_path = dd_pos + 1 < len(args)
                return [exe, *args, *(() if has_pos_path else (".",)), *neg_specs]
            # 区分无 -- 时的修订号与工作区文件路径，确保不把文件名塞到 -- 左侧报 bad revision
            sub_pos = args.index("diff") + 1
            rev_part: list[str] = []
            path_part: list[str] = []
            in_paths = False
            for tok in args[sub_pos:]:
                if not in_paths and (tok.startswith("-") or not (policy.cwd / tok).exists() or tok == "HEAD"):
                    rev_part.append(tok)
                else:
                    in_paths = True
                    path_part.append(tok)
            return [exe, *args[:sub_pos], *rev_part, "--", *(path_part or ["."]), *neg_specs]

    return argv


# 合并 stdout/stderr 并接入 Head+Tail TOAST 输出卸载，保持退出码始终位于首行
def _format_command_output(
    returncode: int, stdout: str, stderr: str, *, max_output_bytes: int, tag: str, offload_dir: Path | None
) -> str:
    clean_out, clean_err = (stdout or "").strip(), (stderr or "").strip()
    sections = ([clean_out] if clean_out else []) + ([f"stderr:\n{clean_err}"] if clean_err else [])
    if not sections:
        return f"退出码：{returncode}"
    body = truncate_or_offload(
        "\n".join(sections), max_output_bytes, tag=tag, mode="head_tail", offload_dir=offload_dir
    )
    return f"退出码：{returncode}\n{body}"


# 构建隔离的子进程环境变量：空 HOME + 禁用全局/系统 git config 与交互提示
def _build_isolated_env(safe_path: str, isolated_home: str) -> dict[str, str]:
    return {
        "PATH": safe_path,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HOME": isolated_home,
        "TMPDIR": tempfile.gettempdir(),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_PAGER": "cat",
        "GIT_EDITOR": "true",
    }


# 在守护线程中流式读取管道字节，超过硬上限即停止缓冲并通知终止整个进程组
def _read_pipe_limited(
    pipe: IO[bytes] | None,
    limit: int,
    chunks: list[bytes],
    overflow: threading.Event,
    wake: threading.Event,
) -> None:
    if pipe is None:
        return
    total = 0
    try:
        while True:
            block = pipe.read(16384)
            if not block:
                break
            remaining = limit - total
            if remaining > 0:
                kept = block[:remaining]
                chunks.append(kept)
                total += len(kept)
            if len(block) > remaining:
                overflow.set()
                wake.set()
                break
    except OSError:
        pass
    finally:
        try:
            pipe.close()
        except OSError:
            pass


# 强制向子进程所属整个进程组发送 SIGKILL 并回收进程，杜绝超时或超限后孙进程残留
def _kill_process_group(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        pass


# 以独立进程组、限长流式管道读取和超时强杀机制执行单个子进程命令
def _run_subprocess_limited(
    cmd_argv: tuple[str, ...],
    *,
    cwd: Path,
    timeout: float,
    env: Mapping[str, str],
    max_capture_bytes: int,
) -> tuple[int, str, str, str | None]:
    proc = subprocess.Popen(
        cmd_argv,
        shell=False,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(env),
        start_new_session=True,
    )
    out_chunks: list[bytes] = []
    err_chunks: list[bytes] = []
    overflow = threading.Event()
    wake = threading.Event()

    # 等待子进程结束并立即唤醒主线程
    def _wait_proc() -> None:
        try:
            proc.wait()
        finally:
            wake.set()

    t_out = threading.Thread(
        target=_read_pipe_limited,
        args=(proc.stdout, max_capture_bytes, out_chunks, overflow, wake),
        daemon=True,
    )
    t_err = threading.Thread(
        target=_read_pipe_limited,
        args=(proc.stderr, max_capture_bytes, err_chunks, overflow, wake),
        daemon=True,
    )
    t_wait = threading.Thread(target=_wait_proc, daemon=True)
    t_out.start()
    t_err.start()
    t_wait.start()

    finished = wake.wait(timeout=timeout)
    if not finished:
        _kill_process_group(proc)
        t_out.join(timeout=1.0)
        t_err.join(timeout=1.0)
        t_wait.join(timeout=1.0)
        return -1, "", "", f"错误：命令执行超时（>{timeout:g}s）。"

    t_wait.join(timeout=0.5)
    t_out.join(timeout=0.5)
    t_err.join(timeout=0.5)
    if t_out.is_alive() or t_err.is_alive() or proc.poll() is None:
        _kill_process_group(proc)
        t_out.join(timeout=1.0)
        t_err.join(timeout=1.0)
        t_wait.join(timeout=1.0)

    if overflow.is_set():
        return -1, "", "", f"错误：命令输出超过单流硬上限（>{max_capture_bytes} 字节），已强制终止。"

    stdout_text = b"".join(out_chunks).decode("utf-8", errors="replace")
    stderr_text = b"".join(err_chunks).decode("utf-8", errors="replace")
    return proc.returncode if proc.returncode is not None else 0, stdout_text, stderr_text, None


# 核验单个 git 修订引用（及其解引用 ^{}）是否指向裸 blob 或 tree 对象
def _check_git_rev_not_blob_or_tree(
    executable: str,
    rev: str,
    *,
    sub: str,
    cwd: Path,
    timeout: float,
    env: Mapping[str, str],
) -> str | None:
    for probe in (rev, f"{rev}^{{}}"):
        code, obj_type, _, err_msg = _run_subprocess_limited(
            (executable, "cat-file", "-t", probe),
            cwd=cwd,
            timeout=min(timeout, 3.0),
            env=env,
            max_capture_bytes=4096,
        )
        if err_msg is not None:
            return err_msg
        kind = obj_type.strip()
        if code == 0 and kind in {"blob", "tree"}:
            return f"错误：git {sub} 禁止直接读取 {kind} 对象（{rev}），仅允许基于 commit 摘要或安全路径查看。"
    return None


# 在执行 git show / diff 前核验范围端点与冒号左侧引用，严禁利用裸 blob 或 tree SHA 绕过敏感路径过滤
def _reject_bare_git_blob_or_tree(
    executable: str,
    sub: str,
    git_args: list[str],
    *,
    cwd: Path,
    timeout: float,
    env: Mapping[str, str],
) -> str | None:
    sub_idx = 1 if git_args[:1] == ["--no-pager"] else 0
    rest = git_args[sub_idx + 1 :]
    before_dd = rest[: rest.index("--")] if "--" in rest else rest
    for tok in before_dd:
        if tok.startswith("-"):
            continue
        for ep in re.split(r"\.{2,3}", tok):
            clean_ep = re.sub(r"\^[!@]$", "", ep.strip())
            if not clean_ep:
                continue
            if ":" in clean_ep:
                lhs, _ = clean_ep.split(":", 1)
                if not lhs or lhs in {"0", "1", "2", "3"}:
                    continue
                err = _check_git_rev_not_blob_or_tree(
                    executable, lhs, sub=sub, cwd=cwd, timeout=timeout, env=env
                )
                if err is not None:
                    return err
                continue
            err = _check_git_rev_not_blob_or_tree(
                executable, clean_ep, sub=sub, cwd=cwd, timeout=timeout, env=env
            )
            if err is not None:
                return err
    return None


# 在安全策略护栏内执行单条受控命令（shell=False、隔离 HOME、进程组限长与 Head+Tail TOAST 卸载）
def execute_safe_command(command: str, policy: ShellPolicy) -> str:
    try:
        argv = _parse_single_command(command)
        exe = argv[0]
        if "/" in exe or "\\" in exe:
            raise ValueError("命令名不能包含路径前缀，只能使用白名单程序名。")
        if exe not in policy.allowed_executables:
            raise ValueError(f"命令 {exe} 不在白名单内；允许程序：{', '.join(policy.allowed_executables)}")
        _validate_program_rules(argv, policy)
    except ValueError as error:
        return f"错误：{error}"

    orig_git_sub = _parse_git_command(argv[1:], policy)[0] if argv[0] == "git" else ""
    final_argv = _augment_safe_argv(argv, policy)
    # 使用 absolute() 而非 resolve()，保留 .venv/bin/python 软链所在目录以便定位 .venv/bin/pytest
    venv_bin = str(Path(sys.executable).absolute().parent)
    safe_path = f"{venv_bin}:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
    executable = shutil.which(final_argv[0], path=safe_path)
    if not executable:
        return f"错误：找不到可执行程序 {final_argv[0]}。"

    with tempfile.TemporaryDirectory(prefix="my_helper_home_") as isolated_home:
        env = _build_isolated_env(safe_path, isolated_home)
        try:
            if orig_git_sub in {"show", "diff"}:
                err_msg = _reject_bare_git_blob_or_tree(
                    executable,
                    orig_git_sub,
                    argv[1:],
                    cwd=policy.cwd,
                    timeout=policy.timeout,
                    env=env,
                )
                if err_msg is not None:
                    return err_msg

            returncode, stdout_text, stderr_text, run_err = _run_subprocess_limited(
                (executable, *final_argv[1:]),
                cwd=policy.cwd,
                timeout=policy.timeout,
                env=env,
                max_capture_bytes=_MAX_CAPTURE_BYTES,
            )
        except OSError as error:
            return f"错误：命令启动失败（{error}）。"

    if run_err is not None:
        return run_err

    return _format_command_output(
        returncode,
        stdout_text,
        stderr_text,
        max_output_bytes=policy.max_output_bytes,
        tag=f"shell-{final_argv[0]}",
        offload_dir=policy.offload_dir,
    )


# 插件工具入口：受 config/tools.json 的 shell 段白名单管控执行单条只读/受控命令（运行时上下文参数由 ToolsPlugin / build_tool 自动绑定）
@tool(
    name="run_command",
    description=(
        "在项目目录内执行单条受白名单保护的本地命令（如 ls、find、grep、rg、wc、head、tail、"
        "cat、pwd、date、df、git 只读子命令、pytest）。不支持管道或命令拼接，禁止访问 .env 等敏感路径；"
        "超长输出保留头尾预览并自动保存至 data/workspace/tool_outputs/ 供分页读取。"
    ),
    parameters={"command": "要执行的单条命令字符串，例如 'ls -la' 或 'git log -n 5 --oneline'"},
    risk="exec",
)
def run_command(
    command: str,
    *,
    project_root: Path | None = None,
    config_path: Path | None = None,
    offload_dir: Path | None = None,
    shell_config: Mapping[str, object] | None = None,
) -> str:
    if isinstance(shell_config, Mapping):
        policy = ShellPolicy.from_dict(shell_config, project_root=project_root or REPO_ROOT, offload_dir=offload_dir)
    else:
        policy = ShellPolicy.from_config(config_path, offload_dir=offload_dir)
    return execute_safe_command(command, policy)
