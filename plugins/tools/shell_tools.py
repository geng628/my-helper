"""受控白名单 Shell 命令插件（档位 1）：shlex 解析 + shell=False + 程序/路径/子命令白名单。"""
from __future__ import annotations

import fnmatch
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

DANGEROUS_FIND_FLAGS = frozenset("-exec -execdir -ok -okdir -delete -fls -fprint -fprint0 -fprintf -files0-from -f".split())
GIT_DIFF_SUMMARY_FLAGS = frozenset({"--stat", "--name-only", "--name-status"})
GIT_SHOW_SUMMARY_FLAGS = GIT_DIFF_SUMMARY_FLAGS | frozenset({"-s", "--no-patch"})
GIT_SIMPLE_FLAGS: dict[str, frozenset[str]] = {
    "status": frozenset({"-s", "--short", "-b", "--branch", "--porcelain"}),
    "branch": frozenset({"--list", "-l", "-a", "--all", "-r", "--remotes", "-v", "-vv", "--verbose", "--show-current"}),
    "diff": GIT_DIFF_SUMMARY_FLAGS | frozenset({"--cached", "--staged"}),
    "show": GIT_SHOW_SUMMARY_FLAGS | frozenset({"--oneline"}),
    "rev-parse": frozenset({"--short", "--abbrev-ref", "--show-toplevel", "--is-inside-work-tree", "--verify", "--symbolic-full-name"}),
}
GIT_LOG_BOOL_FLAGS = frozenset({"--oneline", "--stat", "--name-only", "--name-status", "--graph", "--decorate"})
GIT_SUPPORTED_SUBCOMMANDS = frozenset({*GIT_SIMPLE_FLAGS, "log"})

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
ProtectedCtx = tuple[list[Path], set[tuple[int, int]]]


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

    # 从主板注入的 shell 配置字典与上下文路径直接构建 ShellPolicy（缺省使用安全默认值）
    @classmethod
    def from_dict(
        cls,
        shell_cfg: Mapping[str, object] | None = None,
        *,
        project_root: Path = REPO_ROOT,
        offload_dir: Path | None = None,
    ) -> "ShellPolicy":
        cfg = shell_cfg or {}
        cwd_raw = str(cfg.get("cwd") or ".").strip()
        cwd_rel = Path(cwd_raw)
        if cwd_rel.is_absolute() or ".." in cwd_rel.parts:
            raise ValueError(f"shell.cwd 必须是项目内相对路径：{cwd_raw}")
        cwd = (project_root / cwd_rel).resolve()
        if not cwd.is_relative_to(project_root.resolve()) or not cwd.is_dir():
            raise ValueError(f"shell.cwd 不存在或超出项目范围：{cwd_raw}")
        raw_exes = cfg.get("allowed_executables")
        raw_git = cfg.get("allowed_git_subcommands")
        raw_reject = cfg.get("reject_segments")
        raw_timeout = cfg.get("timeout", 10.0)
        raw_max_out = cfg.get("max_output_bytes", 8192)
        timeout = float(raw_timeout) if isinstance(raw_timeout, (int, float, str)) and not isinstance(raw_timeout, bool) else 10.0
        max_output_bytes = int(raw_max_out) if isinstance(raw_max_out, (int, float, str)) and not isinstance(raw_max_out, bool) else 8192
        return cls(
            project_root=project_root.resolve(),
            cwd=cwd,
            allowed_executables=(
                tuple(str(x) for x in raw_exes)
                if isinstance(raw_exes, (list, tuple))
                else DEFAULT_ALLOWED_EXECUTABLES
            ),
            allowed_git_subcommands=(
                tuple(str(x) for x in raw_git)
                if isinstance(raw_git, (list, tuple))
                else DEFAULT_GIT_SUBCOMMANDS
            ),
            reject_segments=(
                tuple(str(x) for x in raw_reject)
                if isinstance(raw_reject, (list, tuple))
                else DEFAULT_REJECT_SEGMENTS
            ),
            timeout=timeout,
            max_output_bytes=max_output_bytes,
            offload_dir=offload_dir,
        )


# 词法解析单条命令并拦截未引用的管道、拼接、重定向与命令替换语法
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


# 检查通配符表达式是否可能经 Unicode 折叠后命中敏感路径段、.env.* 或 .tmp-* 暂存文件
def _matches_sensitive_glob(glob_expr: str, reject_segments: tuple[str, ...]) -> bool:
    if not any(ch in glob_expr for ch in "*?["):
        return False
    samples = {normalize_segment(seg) for seg in reject_segments} | {
        ".env", ".env.local", ".env.production", ".env.staging", ".env.development", ".tmp-x",
    }
    norm = normalize_segment(glob_expr.replace("\\", "/"))
    cands = {norm, *(p for p in Path(norm).parts if any(ch in p for ch in "*?["))}
    return any(fnmatch.fnmatchcase(sample, cand) for cand in cands for sample in samples)


# 收集当前 ShellPolicy 对应的停止边界目录与受保护内核 Inode 集合
def _protected_inode_context(policy: ShellPolicy) -> ProtectedCtx:
    root_res = policy.project_root.resolve(strict=False)
    scan_roots: list[Path] = [root_res, root_res / "config", root_res / "data"]
    stop_roots: list[Path] = [root_res]
    if policy.offload_dir is not None:
        off_res = policy.offload_dir.resolve(strict=False)
        scan_roots.extend([off_res, off_res.parent, off_res.parent.parent])
        stop_roots.extend([off_res.parent, off_res.parent.parent])
    return stop_roots, collect_protected_inodes(scan_roots, policy.reject_segments)


# 校验单个路径参数：拒绝绝对路径、.. 穿越、Unicode 敏感段、受保护 Inode 与越界软链
def _validate_path_candidate(raw_value: str, policy: ShellPolicy, *, protected_ctx: ProtectedCtx | None = None) -> None:
    if not raw_value:
        return
    if raw_value.startswith("~") or Path(raw_value).is_absolute():
        raise ValueError(f"禁止访问绝对路径或家目录路径：{raw_value}")
    rel = Path(raw_value.replace("\\", "/"))
    if ".." in rel.parts:
        raise ValueError(f"禁止路径穿越（..）：{raw_value}")
    for part in rel.parts:
        if is_sensitive_segment(part, policy.reject_segments):
            raise ValueError(f"禁止访问受保护路径：{part}")

    stop_roots, protected_inodes = protected_ctx or _protected_inode_context(policy)
    root_res = policy.project_root.resolve(strict=False)
    candidate = policy.cwd / rel
    if hits_protected_inode(candidate, stop_roots, protected_inodes):
        raise ValueError(f"禁止访问受保护路径：{raw_value}")

    targets: list[Path] = []
    cur = policy.cwd
    for part in rel.parts:
        cur = cur / part
        if cur.is_symlink():
            targets.append(cur.resolve(strict=False))
    if candidate.exists():
        targets.append(candidate.resolve(strict=False))
    for target in targets:
        if not target.is_relative_to(root_res):
            raise ValueError(f"目标路径超出项目根目录：{raw_value}")
        for part in target.relative_to(root_res).parts:
            if is_sensitive_segment(part, policy.reject_segments):
                raise ValueError(f"禁止访问受保护目标：{part}")
        if hits_protected_inode(target, stop_roots, protected_inodes):
            raise ValueError(f"禁止访问受保护目标：{raw_value}")


# 校验搜索与文件过滤 glob（禁止正向匹配隐藏路径或敏感段，放行排除式 !glob）
def _validate_safe_glob(raw_glob: str, policy: ShellPolicy, *, protected_ctx: ProtectedCtx | None = None) -> None:
    if not raw_glob:
        raise ValueError("glob 参数不能为空。")
    negated = raw_glob.startswith("!")
    cleaned = raw_glob[1:] if negated else raw_glob
    _validate_path_candidate(cleaned, policy, protected_ctx=protected_ctx)
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


# 提取短选项簇中带参短选项的内联值或下一参数值
def _consume_short_opt(args: list[str], idx: int, arg: str, pos: int, label: str) -> tuple[str, str | None, int]:
    rest = arg[pos + 2 :]
    inline = (rest[1:] if rest.startswith("=") else rest) if rest else None
    val, next_idx = _consume_opt_value(args, idx, inline, label)
    return val, inline, next_idx


# 校验 pytest 参数并返回是否已显式传入测试目标（仅纯 --version/-V 可跳过 tests/ 检查）
def _parse_pytest_args(args: list[str], policy: ShellPolicy) -> bool:
    if args in (["--version"], ["-V"]):
        return True
    root_res = policy.project_root.resolve(strict=False)
    tests_entry = root_res / "tests"
    tests_root = tests_entry.resolve(strict=False)
    if not tests_entry.is_dir() or tests_entry.is_symlink() or not tests_root.is_relative_to(root_res):
        raise ValueError("pytest 要求项目根目录存在真实的 tests/ 目录。")

    protected_ctx = _protected_inode_context(policy)
    idx, has_target, after_dd = 0, False, False
    while idx < len(args):
        arg = args[idx]
        if after_dd or not arg.startswith("-") or arg == "-":
            file_part = arg.split("::", 1)[0]
            if not file_part:
                raise ValueError(f"pytest 测试路径无效：{arg}")
            _validate_path_candidate(file_part, policy, protected_ctx=protected_ctx)
            if not (policy.cwd / Path(file_part.replace("\\", "/"))).resolve(strict=False).is_relative_to(tests_root):
                raise ValueError(f"pytest 只能运行 tests/ 目录内的测试：{arg}")
            has_target, idx = True, idx + 1
        elif arg == "--":
            after_dd, idx = True, idx + 1
        elif arg.startswith("--"):
            opt, has_eq, raw_val = arg.partition("=")
            if opt in PYTEST_LONG_BOOL and not has_eq:
                idx += 1
            elif opt in PYTEST_LONG_VAL:
                val, idx = _consume_opt_value(args, idx, raw_val if has_eq else None, f"pytest 选项 {opt}")
                spec = PYTEST_LONG_VAL[opt]
                if (isinstance(spec, frozenset) and val not in spec) or (spec == "int" and not val.isdigit()):
                    raise ValueError(f"pytest 选项 {opt} 取值非法：{val}")
            else:
                raise ValueError(f"pytest 仅允许安全只读运行标志，已拒绝：{arg}")
        else:
            next_idx = idx + 1
            for pos, ch in enumerate(arg[1:]):
                if ch in PYTEST_SHORT_VAL:
                    val, inline, next_idx = _consume_short_opt(args, idx, arg, pos, f"pytest 选项 -{ch}")
                    if (inline is None and val.startswith("-")) or not val.strip():
                        raise ValueError(f"pytest 选项 -{ch} 参数值非法：{val}")
                    break
                if ch not in PYTEST_SHORT_BOOL:
                    raise ValueError(f"pytest 禁止使用高危或未授权短选项：-{ch}（来自 {arg}）")
            idx = next_idx
    return has_target


# 校验 git branch --list/-l 的分支名称模式（拦截路径穿越、魔法语法与敏感段）
def _validate_git_branch_pattern(arg: str, policy: ShellPolicy) -> None:
    if not arg or arg.startswith("~") or ":" in arg:
        raise ValueError(f"git branch 模式参数非法：{arg}")
    rel = Path(arg.replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"git branch 模式不能包含绝对路径或 ..：{arg}")
    for part in rel.parts:
        if is_sensitive_segment(part, policy.reject_segments):
            raise ValueError(f"git branch 模式禁止引用受保护名称：{part}")


# 校验 git 位置参数：禁止冒号对象引用、解引用、pathspec 魔法、敏感路径及裸哈希对象
def _validate_git_pathspec(
    arg: str, policy: ShellPolicy, *, sub: str, after_dd: bool, protected_ctx: ProtectedCtx
) -> None:
    if ":" in arg or "^{" in arg or "@{" in arg:
        raise ValueError(f"git 禁止使用冒号对象引用、解引用或 pathspec 魔法语法：{arg}")
    if sub in {"show", "diff"} and not after_dd and any(ch in arg for ch in "*?["):
        raise ValueError(f"git {sub} 通配符路径必须位于 -- 之后：{arg}")
    _validate_path_candidate(arg, policy, protected_ctx=protected_ctx)
    if _matches_sensitive_glob(arg, policy.reject_segments):
        raise ValueError(f"git 禁止通配符匹配受保护路径：{arg}")
    for ep in re.split(r"\.{2,3}", arg):
        if ep:
            _validate_path_candidate(ep, policy, protected_ctx=protected_ctx)
            if _matches_sensitive_glob(ep, policy.reject_segments):
                raise ValueError(f"git 禁止通配符匹配受保护路径：{ep}")
            if sub in {"show", "diff"} and re.fullmatch(r"(?:.*-\d+-g)?[0-9a-fA-F]{4,64}(?:[~^].*)?", ep):
                raise ValueError(f"git {sub} 禁止直接传入裸对象哈希引用：{arg}")


# 解析并严格校验单个 git 只读子命令及选项，返回 (子命令名, 子命令后参数列表)
def _parse_git_command(args: list[str], policy: ShellPolicy) -> tuple[str, list[str]]:
    sub_idx = 1 if args[:1] == ["--no-pager"] else 0
    if (
        sub_idx >= len(args)
        or args[sub_idx] not in policy.allowed_git_subcommands
        or args[sub_idx] not in GIT_SUPPORTED_SUBCOMMANDS
    ):
        allowed = ", ".join(policy.allowed_git_subcommands)
        raise ValueError(f"git 仅允许直接调用白名单只读子命令（允许：{allowed}），禁止前置全局选项。")

    sub, rest = args[sub_idx], args[sub_idx + 1 :]
    protected_ctx = _protected_inode_context(policy)
    idx, after_dd = 0, False
    seen_flags: list[str] = []
    positionals: list[str] = []

    while idx < len(rest):
        arg = rest[idx]
        if arg == "--" and not after_dd:
            after_dd, idx = True, idx + 1
        elif after_dd or not arg.startswith("-") or arg == "-":
            if sub == "branch":
                _validate_git_branch_pattern(arg, policy)
            else:
                _validate_git_pathspec(arg, policy, sub=sub, after_dd=after_dd, protected_ctx=protected_ctx)
            positionals.append(arg)
            idx += 1
        elif sub in GIT_SIMPLE_FLAGS:
            if arg not in GIT_SIMPLE_FLAGS[sub]:
                raise ValueError(f"git {sub} 不允许选项：{arg}")
            seen_flags.append(arg)
            idx += 1
        else:
            if arg in GIT_LOG_BOOL_FLAGS or re.fullmatch(r"(?:-n?[1-9]\d*|--max-count=[1-9]\d*)", arg):
                idx += 1
            elif arg == "-n":
                val, idx = _consume_opt_value(rest, idx, None, "git log -n")
                if not re.fullmatch(r"[1-9]\d*", val):
                    raise ValueError(f"git log -n 需要正整数：{val}")
            else:
                raise ValueError(f"git log 不允许选项：{arg}")
            seen_flags.append(arg)

    if sub == "branch" and positionals:
        if not any(f in {"--list", "-l"} for f in seen_flags) or "--show-current" in seen_flags:
            raise ValueError("git branch 仅允许纯列举只读标志，严禁传入位置参数创建分支。")
    return sub, rest


# 校验 grep / rg 带值选项参数，阻断非法整数、类型与敏感 glob
def _validate_search_opt_value(exe: str, opt: str, kind: str, val: str, policy: ShellPolicy, protected_ctx: ProtectedCtx) -> None:
    if kind == "int" and not val.isdigit():
        raise ValueError(f"{exe} 选项 {opt} 需要非负整数：{val}")
    elif kind == "action" and val not in {"read", "skip", "recurse"}:
        raise ValueError(f"{exe} 选项 {opt} 取值非法：{val}")
    elif kind in {"type", "word"} and not re.fullmatch(r"[A-Za-z0-9_+-]+", val):
        raise ValueError(f"{exe} 选项 {opt} 取值非法：{val}")
    elif kind == "glob":
        _validate_safe_glob(val, policy, protected_ctx=protected_ctx)
    elif kind == "path":
        _validate_path_candidate(val, policy, protected_ctx=protected_ctx)


# 全量扫描并校验 grep / rg 参数（封堵后置 -e/--files 绕过并返回排除项插入点）
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
        elif arg == "--":
            dd_idx = idx if dd_idx is None else dd_idx
            after_dd, idx = True, idx + 1
        elif arg.startswith("--"):
            opt, has_eq, raw_val = arg.partition("=")
            if opt in spec["forbidden_long"] or (exe == "rg" and opt.startswith("--no-ignore")):
                raise ValueError(f"{exe} 禁止使用高危或绕过忽略规则选项：{arg}")
            if opt in spec["long_bool"]:
                if has_eq:
                    raise ValueError(f"{exe} 布尔选项不接受赋值：{arg}")
                is_recursive = is_recursive or opt == "--recursive"
                explicit_pattern = explicit_pattern or opt == "--files"
                idx += 1
            elif opt in spec["long_val"]:
                val, idx = _consume_opt_value(args, idx, raw_val if has_eq else None, f"{exe} 选项 {opt}")
                kind = spec["long_val"][opt]
                if kind == "pattern":
                    explicit_pattern = True
                else:
                    _validate_search_opt_value(exe, opt, kind, val, policy, protected_ctx)
            else:
                raise ValueError(f"{exe} 不支持的长选项：{arg}")
        else:
            if exe == "rg" and arg in {"-u", "-uu", "-uuu", "-."}:
                raise ValueError(f"rg 禁止绕过忽略规则或扫描隐藏文件：{arg}")
            next_idx = idx + 1
            for pos, ch in enumerate(arg[1:]):
                if ch in spec["forbidden_short"]:
                    raise ValueError(f"{exe} 禁止使用高危或绕过保护短选项：-{ch}（来自 {arg}）")
                is_recursive = is_recursive or ch == "r"
                if ch in spec["short_val"]:
                    val, _, next_idx = _consume_short_opt(args, idx, arg, pos, f"{exe} 选项 -{ch}")
                    is_recursive = is_recursive or (ch == "d" and val == "recurse")
                    kind = spec["short_val"][ch]
                    if kind == "pattern":
                        explicit_pattern = True
                    else:
                        _validate_search_opt_value(exe, f"-{ch}", kind, val, policy, protected_ctx)
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


# 校验通用白名单命令（ls/find/cat/wc/date 等）的选项与路径参数
def _validate_generic_args(exe: str, args: list[str], policy: ShellPolicy) -> None:
    protected_ctx = _protected_inode_context(policy)
    after_dd = False
    for arg in args:
        if arg == "--" and not after_dd:
            after_dd = True
        elif not after_dd and arg.startswith("-") and arg != "-":
            if exe == "find" and (
                arg in DANGEROUS_FIND_FLAGS or arg.startswith("-files0-from") or (arg.startswith("-f") and arg != "-false")
            ):
                raise ValueError(f"find 禁止使用危险选项：{arg}")
            if exe == "date" and ((not arg.startswith("--") and "f" in arg[1:]) or arg == "--file" or arg.startswith("--file=")):
                raise ValueError(f"date 禁止从文件读取输入：{arg}")
            if arg.startswith("--"):
                if arg == "--files0-from" or arg.startswith("--files0-from="):
                    raise ValueError(f"{exe} 禁止从文件读取路径列表：{arg}")
                if "=" in arg:
                    _validate_path_candidate(arg.split("=", 1)[1], policy, protected_ctx=protected_ctx)
            elif not re.fullmatch(r"-[A-Za-z0-9]+", arg):
                raise ValueError(f"{exe} 禁止使用未知连写路径短选项：{arg}")
        else:
            _validate_path_candidate(arg, policy, protected_ctx=protected_ctx)


# 将路径转换为相对 cwd 的字符串以简化 --ignore 与默认测试目录参数
def _rel_to_cwd(path: Path, cwd: Path) -> str:
    return str(path.relative_to(cwd)) if path.is_relative_to(cwd) else str(path)


# 统一完成程序参数校验，并为 grep/rg/pytest/git 自动补齐安全隔离参数
def _augment_safe_argv(argv: list[str], policy: ShellPolicy) -> list[str]:
    exe, args = argv[0], argv[1:]
    if exe in SEARCH_SPECS:
        insert_at, is_recursive, only_example = _parse_search_args(exe, args, policy)
        add_env = (not only_example) if exe == "rg" else (is_recursive or not only_example)
        pats = (
            *policy.reject_segments,
            ".tmp-*",
            *((".git",) if ".git" not in policy.reject_segments else ()),
            *((".env.*",) if add_env else ()),
        )
        fmts = ("--exclude={}", "--exclude-dir={}") if exe == "grep" else ("--glob=!{}", "--glob=!**/{}/**")
        return [exe, *args[:insert_at], *(fmt.format(seg) for seg in pats for fmt in fmts), *args[insert_at:]]

    if exe == "pytest":
        has_target = _parse_pytest_args(args, policy)
        if args in (["--version"], ["-V"]):
            return argv
        root_res, cwd_res = policy.project_root.resolve(strict=False), policy.cwd.resolve(strict=False)
        ignores = [f"--ignore={_rel_to_cwd(root_res / 'data', cwd_res)}"]
        if policy.offload_dir is not None:
            off_res = policy.offload_dir.resolve(strict=False)
            if off_res.is_relative_to(root_res) and (rel_off := off_res.relative_to(root_res)).parts and rel_off.parts[0] != "data":
                ignores.append(f"--ignore={_rel_to_cwd(root_res / rel_off.parts[0], cwd_res)}")
        tail = [] if has_target else [_rel_to_cwd((root_res / "tests").resolve(strict=False), cwd_res)]
        return [exe, *ignores, *args, *tail]

    if exe == "git":
        sub, rest = _parse_git_command(args, policy)
        insert_pos = (1 if args[:1] == ["--no-pager"] else 0) + 1
        before_dd = rest[: rest.index("--")] if "--" in rest else rest
        if sub == "show" and not any(tok in GIT_SHOW_SUMMARY_FLAGS for tok in before_dd):
            return [exe, *args[:insert_pos], "--stat", *args[insert_pos:]]
        if sub == "diff" and not any(tok in GIT_DIFF_SUMMARY_FLAGS for tok in before_dd):
            after_dd = rest[rest.index("--") + 1 :] if "--" in rest else []
            explicit_files = (
                after_dd if "--" in rest else [t for t in before_dd if not t.startswith("-") and (policy.cwd / t).is_file()]
            )
            if not explicit_files or any((policy.cwd / p).is_dir() for p in explicit_files):
                return [exe, *args[:insert_pos], "--stat", *args[insert_pos:]]
            if any(not (policy.cwd / p).is_file() for p in explicit_files):
                pats = (s for s in (*policy.reject_segments, ".env.*", ".tmp-*") if s != ".git")
                neg = [f":(exclude,glob,icase){p}" for s in pats for p in (s, f"**/{s}", f"**/{s}/**")]
                return [*argv, *neg]
        return argv

    _validate_generic_args(exe, args, policy)
    return argv


# 合并 stdout/stderr 并接入 Head+Tail TOAST 输出卸载，保持退出码位于首行
def _format_command_output(
    returncode: int, stdout: str, stderr: str, *, max_output_bytes: int, tag: str, offload_dir: Path | None
) -> str:
    clean_out, clean_err = (stdout or "").strip(), (stderr or "").strip()
    sections = ([clean_out] if clean_out else []) + ([f"stderr:\n{clean_err}"] if clean_err else [])
    if not sections:
        return f"退出码：{returncode}"
    body = truncate_or_offload("\n".join(sections), max_output_bytes, tag=tag, mode="head_tail", offload_dir=offload_dir)
    return f"退出码：{returncode}\n{body}"


# 构建隔离的子进程环境变量：空 HOME + 禁用全局/系统 git 配置与交互提示
def _build_isolated_env(safe_path: str, isolated_home: str) -> dict[str, str]:
    return {
        "PATH": safe_path, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "HOME": isolated_home,
        "TMPDIR": tempfile.gettempdir(), "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_ATTR_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0", "GIT_PAGER": "cat", "GIT_EDITOR": "true",
    }


# 在守护线程中流式读取管道字节，超过硬上限即停止缓冲并唤醒主线程强杀进程组
def _read_pipe_limited(
    pipe: IO[bytes] | None, limit: int, chunks: list[bytes], overflow: threading.Event, wake: threading.Event
) -> None:
    if pipe is None:
        return
    total = 0
    try:
        while block := pipe.read(16384):
            remaining = limit - total
            if remaining > 0:
                chunks.append(block[:remaining])
                total += min(len(block), remaining)
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


# 强制向子进程所属进程组发送 SIGKILL 并回收进程，杜绝超时或超限后孙进程残留
def _kill_process_group(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
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
    cmd_argv: tuple[str, ...], *, cwd: Path, timeout: float, env: Mapping[str, str], max_capture_bytes: int
) -> tuple[int, str, str, str | None]:
    proc = subprocess.Popen(
        cmd_argv, shell=False, cwd=cwd, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=dict(env), start_new_session=True,
    )
    out_chunks: list[bytes] = []
    err_chunks: list[bytes] = []
    overflow, wake = threading.Event(), threading.Event()

    # 等待子进程退出并唤醒主线程
    def _wait_proc() -> None:
        try:
            proc.wait()
        finally:
            wake.set()

    threads = (
        threading.Thread(target=_read_pipe_limited, args=(proc.stdout, max_capture_bytes, out_chunks, overflow, wake), daemon=True),
        threading.Thread(target=_read_pipe_limited, args=(proc.stderr, max_capture_bytes, err_chunks, overflow, wake), daemon=True),
        threading.Thread(target=_wait_proc, daemon=True),
    )
    for th in threads:
        th.start()

    if not wake.wait(timeout=timeout):
        _kill_process_group(proc)
        for th in threads:
            th.join(timeout=1.0)
        return -1, "", "", f"错误：命令执行超时（>{timeout:g}s）。"

    for th in threads:
        th.join(timeout=0.5)
    if any(th.is_alive() for th in threads[:2]) or proc.poll() is None:
        _kill_process_group(proc)
        for th in threads:
            th.join(timeout=1.0)
    if overflow.is_set():
        return -1, "", "", f"错误：命令输出超过单流硬上限（>{max_capture_bytes} 字节），已强制终止。"
    return (
        proc.returncode if proc.returncode is not None else 0,
        b"".join(out_chunks).decode("utf-8", errors="replace"),
        b"".join(err_chunks).decode("utf-8", errors="replace"),
        None,
    )


# 在安全策略护栏内执行单条受控命令（shell=False、隔离 HOME、进程组限长与输出卸载）
def execute_safe_command(command: str, policy: ShellPolicy) -> str:
    try:
        argv = _parse_single_command(command)
        if "/" in argv[0] or "\\" in argv[0]:
            raise ValueError("命令名不能包含路径前缀，只能使用白名单程序名。")
        if argv[0] not in policy.allowed_executables:
            raise ValueError(f"命令 {argv[0]} 不在白名单内；允许程序：{', '.join(policy.allowed_executables)}")
        final_argv = _augment_safe_argv(argv, policy)
    except ValueError as error:
        return f"错误：{error}"

    safe_path = f"{Path(sys.executable).absolute().parent}:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
    executable = shutil.which(final_argv[0], path=safe_path)
    if not executable:
        return f"错误：找不到可执行程序 {final_argv[0]}。"

    with tempfile.TemporaryDirectory(prefix="my_helper_home_") as isolated_home:
        try:
            returncode, stdout_text, stderr_text, run_err = _run_subprocess_limited(
                (executable, *final_argv[1:]),
                cwd=policy.cwd,
                timeout=policy.timeout,
                env=_build_isolated_env(safe_path, isolated_home),
                max_capture_bytes=_MAX_CAPTURE_BYTES,
            )
        except OSError as error:
            return f"错误：命令启动失败（{error}）。"

    if run_err is not None:
        return run_err
    return _format_command_output(
        returncode, stdout_text, stderr_text,
        max_output_bytes=policy.max_output_bytes, tag=f"shell-{final_argv[0]}", offload_dir=policy.offload_dir,
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
    risk="exec",
)
def run_command(
    command: str,
    *,
    project_root: Path | None = None,
    offload_dir: Path | None = None,
    shell_config: Mapping[str, object] | None = None,
) -> str:
    policy = ShellPolicy.from_dict(
        shell_config,
        project_root=project_root or REPO_ROOT,
        offload_dir=offload_dir,
    )
    return execute_safe_command(command, policy)
