"""纯 Markdown 记忆、规则存储以及受目录白名单保护的文件读写工具。"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from personal_assistant.commands import truncate_utf8
from personal_assistant.memory_lifecycle import (
    apply_topic_operations,
    archive_expired_topics,
    is_line_expired,
    read_topic_snapshots as _read_topic_snapshots,
    today_shanghai as _today_shanghai,
)
from personal_assistant.tools import Tool

DEFAULT_REJECT_SEGMENTS = (".env", ".git", "state", ".ssh", "uploads")
MAX_SEARCH_FILE_BYTES = 1024 * 1024


# 校验配置中的整数项必须为正整数，缺省时返回默认值
def _positive_int(value: object, default: int, name: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须是正整数。")
    return value


# 将配置里的相对目录安全解析为绝对路径，data/* 前缀自动映射到自定义 data_root
def _resolve_config_dir(entry: object, project_root: Path, data_root: Path | None) -> Path:
    if not isinstance(entry, str) or not entry.strip():
        raise ValueError("目录配置必须是非空字符串。")
    rel = Path(entry.strip())
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"目录配置必须是相对路径且不能包含 ..：{entry}")
    if data_root is not None and rel.parts and rel.parts[0] == "data":
        resolved_data = data_root.resolve()
        sub = Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path(".")
        target = resolved_data / sub
        if not target.resolve().is_relative_to(resolved_data):
            raise ValueError(f"目录配置超出 data_root 范围：{entry}")
        target.mkdir(parents=True, exist_ok=True)
        return target
    target = project_root / rel
    if not target.resolve().is_relative_to(project_root.resolve()):
        raise ValueError(f"目录配置超出项目范围：{entry}")
    return target


# 文件访问白名单策略（允许读写根目录、行数/字节限制与禁止访问目录段）
@dataclass(frozen=True)
class FilesPolicy:
    project_root: Path
    read_roots: tuple[Path, ...]
    write_root: Path
    max_read_bytes: int = 65536
    max_write_bytes: int = 65536
    max_lines: int = 400
    max_results: int = 100
    reject_segments: tuple[str, ...] = DEFAULT_REJECT_SEGMENTS

    # 从 tools.json 的 files 字段字典构造并校验 FilesPolicy
    @classmethod
    def from_dict(
        cls,
        raw: dict[str, object] | None,
        project_root: Path,
        data_root: Path | None = None,
    ) -> "FilesPolicy":
        root = project_root.resolve()
        cfg = raw or {}
        if not isinstance(cfg, dict):
            raise ValueError("files 配置必须是 JSON 对象。")
        raw_reads = cfg.get("read_roots", ["data/memory", "data/workspace", "docs"])
        if not isinstance(raw_reads, list) or not raw_reads:
            raise ValueError("files.read_roots 必须是非空数组。")
        read_roots = tuple(_resolve_config_dir(item, root, data_root) for item in raw_reads)
        write_root = _resolve_config_dir(cfg.get("write_root", "data/workspace"), root, data_root)
        raw_rejects = cfg.get("reject_segments", list(DEFAULT_REJECT_SEGMENTS))
        if not isinstance(raw_rejects, list) or not all(isinstance(x, str) and x for x in raw_rejects):
            raise ValueError("files.reject_segments 必须是字符串数组。")
        return cls(
            project_root=root,
            read_roots=read_roots,
            write_root=write_root,
            max_read_bytes=_positive_int(cfg.get("max_read_bytes"), 65536, "max_read_bytes"),
            max_write_bytes=_positive_int(cfg.get("max_write_bytes"), 65536, "max_write_bytes"),
            max_lines=_positive_int(cfg.get("max_lines"), 400, "max_lines"),
            max_results=_positive_int(cfg.get("max_results"), 100, "max_results"),
            reject_segments=tuple(raw_rejects),
        )


# 通过同目录临时文件 + os.replace 实现原子写入，防止写到一半断电损坏文件
def _atomic_write(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


# 从中英文查询语句提取主检索词（primary）与中文二元组兜底词（secondary）
def _extract_terms(query: str) -> tuple[list[str], list[str]]:
    normalized = " ".join(query.strip().lower().split())
    if not normalized:
        return [], []
    primary = list(dict.fromkeys(normalized.split(" ")))
    for token in re.findall(r"[a-z0-9_+-]+", normalized):
        if len(token) >= 2 and token not in primary:
            primary.append(token)
    secondary: list[str] = []
    for run in re.findall(r"[\u4e00-\u9fff]+", normalized):
        if len(run) >= 2 and run not in primary:
            primary.append(run)
        if len(run) >= 4:
            for idx in range(len(run) - 1):
                bg = run[idx : idx + 2]
                if bg not in primary and bg not in secondary:
                    secondary.append(bg)
    return primary, secondary


# 单条 Markdown 搜索命中结果（路径、行号、所属章节、上下文片段与得分）
@dataclass(frozen=True)
class SearchHit:
    path: str
    line: int
    section: str
    snippet: str
    score: float


# 纯 Markdown 记忆与本地文件管理类：直接读写磁盘 .md/.txt 文件
class MemoryStore:
    # 初始化数据目录与安全策略；create_dirs=False 时以只读方式打开（供 eval 使用）
    def __init__(
        self,
        data_root: Path | str,
        policy: FilesPolicy | None = None,
        *,
        create_dirs: bool = True,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        if policy is None:
            proj = self.data_root.parent
            self.policy = FilesPolicy(
                project_root=proj,
                read_roots=(
                    self.data_root / "memory",
                    self.data_root / "workspace",
                    proj / "docs",
                ),
                write_root=self.data_root / "workspace",
            )
        else:
            self.policy = policy
        self.project_root = self.policy.project_root.resolve()
        self.memory_dir = self.data_root / "memory"
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.facts_file = self.memory_dir / "facts.md"
        self.rules_dir = self.memory_dir / "rules"
        self.archive_dir = self.memory_dir / "archive"
        self.audit_path = self.memory_dir / "audit.jsonl"
        self.workspace_dir = self.policy.write_root

        if create_dirs:
            for directory in (
                self.rules_dir,
                self.workspace_dir,
                self.workspace_dir / ".backups",
                self.workspace_dir / ".tmp",
            ):
                directory.mkdir(parents=True, exist_ok=True)
        self._roots = self._build_roots()

    # 预先整理所有允许根目录的（逻辑路径、真实路径、展示前缀、短别名）映射表
    def _build_roots(self) -> list[tuple[Path, Path, str, str]]:
        seen: set[Path] = set()
        entries: list[tuple[Path, Path, str, str]] = []
        for raw in (*self.policy.read_roots, self.policy.write_root):
            resolved = raw.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            entries.append((raw, resolved, *self._root_labels(raw)))
        return entries

    # 计算单个允许根目录的标准展示路径（如 data/memory）和简写别名（如 memory）
    def _root_labels(self, raw_root: Path) -> tuple[str, str]:
        for base, prefix in ((self.data_root, "data"), (self.project_root, "")):
            try:
                rel = raw_root.relative_to(base)
                if prefix:
                    display = ("data" / rel).as_posix() if rel.parts else "data"
                    alias = rel.as_posix() if rel.parts else ""
                    return display, alias
                display = rel.as_posix() if rel.parts else "."
                return display, display
            except ValueError:
                continue
        return raw_root.name, raw_root.name

    # 将磁盘绝对路径转换为统一的相对展示路径（供外部模块如 eval 调用）
    def display_path(self, resolved: Path) -> str:
        return self._to_display_path(resolved)

    # 内部路径格式化：按允许根目录把真实路径转回统一的相对展示路径
    def _to_display_path(self, resolved: Path) -> str:
        for _, root_resolved, display_prefix, _ in self._roots:
            if resolved.is_relative_to(root_resolved):
                rel = resolved.relative_to(root_resolved)
                return f"{display_prefix}/{rel.as_posix()}" if rel.parts else display_prefix
        if resolved.is_relative_to(self.data_root):
            return ("data" / resolved.relative_to(self.data_root)).as_posix()
        return resolved.name

    # 检查路径中是否包含受保护的敏感目录段（如 .env、.git、state）
    def _check_segments(self, parts: tuple[str, ...]) -> None:
        for part in parts:
            if part in self.policy.reject_segments:
                raise ValueError(f"禁止访问受保护路径：{part}")

    # 安全解析相对路径：校验读写白名单、拒绝 .. 穿越、敏感目录与越权软链接
    def resolve_path(self, rel_path: str, *, write: bool = False) -> Path:
        if not isinstance(rel_path, str) or not rel_path.strip() or "\x00" in rel_path:
            raise ValueError("路径必须是非空字符串。")
        rel = Path(rel_path.strip().replace("\\", "/"))
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("路径必须是项目内相对路径，且不能包含 ..。")
        self._check_segments(rel.parts)

        if write:
            return self._resolve_write_path(rel)

        unresolved = self._map_read_candidate(rel)
        resolved = unresolved.resolve(strict=False)
        for raw_root in self.policy.read_roots:
            root_resolved = raw_root.resolve()
            if resolved.is_relative_to(root_resolved):
                self._check_segments(resolved.relative_to(root_resolved).parts)
                return resolved
        allowed = "、".join(self._root_labels(r)[0] for r in self.policy.read_roots)
        raise ValueError(f"路径不在允许读取的目录内；允许范围：{allowed}")

    # 解析并校验写入路径：限制在 write_root（data/workspace）内且禁止任何软链接组件
    def _resolve_write_path(self, rel: Path) -> Path:
        parts = rel.parts
        write_root = self.policy.write_root
        write_resolved = write_root.resolve()
        write_display, write_alias = self._root_labels(write_root)
        display_parts = Path(write_display).parts

        if parts[: len(display_parts)] == display_parts:
            unresolved = write_root / Path(*parts[len(display_parts) :])
        elif write_alias and parts[0] == write_alias:
            unresolved = write_root / Path(*parts[1:])
        elif self._matches_other_root(parts):
            raise ValueError(f"只能写入 {write_display} 目录。")
        else:
            unresolved = write_root / rel

        current = write_root
        for part in unresolved.relative_to(write_root).parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("写入路径不能包含符号链接。")
        resolved = unresolved.resolve(strict=False)
        if not resolved.is_relative_to(write_resolved):
            raise ValueError(f"只能写入 {write_display} 目录。")
        self._check_segments(resolved.relative_to(write_resolved).parts)
        return resolved

    # 判断写入路径是否误指向其他只读根目录（如 data/memory、docs 等）
    def _matches_other_root(self, parts: tuple[str, ...]) -> bool:
        if not parts:
            return False
        if parts[0] == "data":
            return True
        for raw_root in self.policy.read_roots:
            display, alias = self._root_labels(raw_root)
            d_parts = Path(display).parts
            if parts[: len(d_parts)] == d_parts or (alias and parts[0] == Path(alias).parts[0]):
                return True
        return False

    # 将用户传入的读取路径（支持 data/memory/... 或简写 memory/...）映射到磁盘路径
    def _map_read_candidate(self, rel: Path) -> Path:
        parts = rel.parts
        for raw_root, _, display_prefix, short_alias in self._roots:
            d_parts = Path(display_prefix).parts
            if parts[: len(d_parts)] == d_parts:
                return raw_root / Path(*parts[len(d_parts) :])
            if short_alias:
                a_parts = Path(short_alias).parts
                if parts[: len(a_parts)] == a_parts:
                    return raw_root / Path(*parts[len(a_parts) :])
        if parts and parts[0] == "data":
            return self.data_root / Path(*parts[1:])
        return self.project_root / rel

    # ---- 常驻提示词直读：MEMORY.md 目录索引与 rules/*.md ----

    # 读取 data/memory/MEMORY.md 记忆索引（严格限制前 max_lines 行与 max_chars 字符，防止提示词膨胀）
    def read_memory_index(self, max_lines: int = 100, max_chars: int = 4000) -> str:
        if not self.memory_file.is_file():
            return ""
        text = self.memory_file.read_text(encoding="utf-8").strip()
        if not text:
            return ""
        lines = text.splitlines()
        truncated = False
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            truncated = True
        body = "\n".join(lines)
        if len(body) > max_chars:
            body = body[:max_chars]
            if "\n" in body:
                body = body.rsplit("\n", 1)[0]
            truncated = True
        if truncated:
            return f'{body}\n（索引已截断，完整目录请调用 read_markdown("data/memory/MEMORY.md", ...)）'
        return body

    # 读取 data/memory/facts.md 中最近 limit 条事实记录（仅供 /memory 展示，不注入系统提示词）
    def read_recent_facts(self, limit: int = 20) -> str:
        if not self.facts_file.is_file() or limit <= 0:
            return ""
        items = [
            line
            for line in self.facts_file.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("- ")
        ]
        if not items:
            return ""
        shown = items[-limit:]
        prefix = f"（仅展示最近 {len(shown)}/{len(items)} 条）\n" if len(items) > limit else ""
        return prefix + "\n".join(shown)

    # 倒序读取 data/memory/rules/*.md，提取所有行为规则摘要列表
    def read_rules_summary(self, limit: int = 30) -> str:
        if not self.rules_dir.is_dir():
            return ""
        rules = sorted(self.rules_dir.glob("*.md"), reverse=True)
        if not rules:
            return ""
        lines = [f"- data/memory/rules/{r.name} — {self._rule_statement(r)}" for r in rules[:limit]]
        if len(rules) > limit:
            lines.append(f"- 另有 {len(rules) - limit} 条较早规则，可用 search_markdown 查看 data/memory/rules。")
        return "\n".join(lines)

    # 从单篇规则 Markdown 文件中提取 Statement 或首行标题
    @staticmethod
    def _rule_statement(rule: Path) -> str:
        content = rule.read_text(encoding="utf-8")
        for line in content.splitlines():
            if line.startswith("- Statement:"):
                return line.removeprefix("- Statement:").strip()
        for line in content.splitlines():
            if line.strip():
                return line.strip().lstrip("# ").removeprefix("Rule:").strip() or "（无标题）"
        return "（空规则文件）"

    # 读取全部活跃主题记忆文件快照（排除 MEMORY.md 与 archive/），供沉淀器读旧改旧消冲突
    def read_topic_snapshots(self, max_total_chars: int = 6000) -> str:
        return _read_topic_snapshots(self.memory_dir, max_total_chars=max_total_chars)

    # ---- 按需浏览与搜索 Markdown 文件 ----

    # 过滤目录遍历结果：跳过非文件、隐藏目录、敏感路径段及越界软链接
    def _safe_files(self, target: Path, iterator) -> list[Path]:
        files: list[Path] = []
        for item in sorted(iterator):
            if not item.is_file():
                continue
            try:
                resolved = item.resolve()
            except OSError:
                continue
            if not resolved.is_relative_to(target):
                continue
            parts = resolved.relative_to(target).parts
            if any(p in self.policy.reject_segments or p.startswith(".") for p in parts):
                continue
            files.append(resolved)
        return files

    # 按 glob 模式列出允许读取目录内的所有文件路径（最多返回 max_results 条）
    def list_files(self, root: str = "data/memory", pattern: str = "*.md") -> str:
        if not isinstance(pattern, str) or not pattern.strip() or len(pattern) > 200:
            raise ValueError("pattern 必须是 200 字符以内的 glob，例如 *.md。")
        if ".." in pattern or "\x00" in pattern:
            raise ValueError("pattern 不能包含 .. 或 NUL。")
        target = self.resolve_path(root, write=False)
        if not target.is_dir():
            raise ValueError(f"目录不存在：{root}")
        pat = pattern.strip()
        iterator = target.glob(pat) if ("/" in pat or "**" in pat) else target.rglob(pat)
        found = [self._to_display_path(p) for p in self._safe_files(target, iterator)[: self.policy.max_results]]
        if not found:
            return "没有匹配的文件。"
        suffix = f"\n（仅展示前 {self.policy.max_results} 个文件）" if len(found) >= self.policy.max_results else ""
        return "\n".join(found) + suffix

    # 在允许目录下扫描 Markdown/文本文件并返回排序后的结构化 SearchHit 列表（默认跳过归档与已过期记忆）
    def search_hits(
        self,
        query: str,
        root: str = "data/memory",
        limit: int = 10,
        *,
        today: date | None = None,
    ) -> list[SearchHit]:
        primary, secondary = _extract_terms(query)
        if not primary and not secondary:
            raise ValueError("搜索关键词不能为空。")
        target = self.resolve_path(root, write=False)
        if not target.is_dir():
            raise ValueError(f"目录不存在：{root}")

        ref_day = today or _today_shanghai()
        archive_resolved = self.archive_dir.resolve()
        memory_resolved = self.memory_dir.resolve()
        searching_archive = target.is_relative_to(archive_resolved)

        candidates: list[tuple[float, str, int, str, str]] = []
        for resolved in self._safe_files(target, target.rglob("*")):
            if resolved.suffix.lower() not in {".md", ".txt"}:
                continue
            if not searching_archive and resolved.is_relative_to(archive_resolved):
                continue
            try:
                if resolved.stat().st_size > MAX_SEARCH_FILE_BYTES:
                    continue
                text = resolved.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            skip_expired = ref_day if (not searching_archive and resolved.is_relative_to(memory_resolved)) else None
            candidates.extend(
                self._score_file(
                    self._to_display_path(resolved),
                    text,
                    primary,
                    secondary,
                    skip_expired_today=skip_expired,
                )
            )

        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [
            SearchHit(path=path, line=line_no, section=section, snippet=snippet, score=score)
            for score, path, line_no, section, snippet in candidates[:limit]
        ]

    # 对外搜索接口：将 search_hits 命中结果格式化为带路径、行号与章节的文本片段
    def search_markdown(
        self,
        query: str,
        root: str = "data/memory",
        limit: int = 10,
        *,
        today: date | None = None,
    ) -> str:
        hits = self.search_hits(query, root=root, limit=limit, today=today)
        if not hits:
            return f"未在 {root} 下找到匹配「{query}」的 Markdown 内容。"
        return "\n\n".join(
            f"{hit.path}:{hit.line} [章节: {hit.section or '正文'}]\n{hit.snippet}"
            for hit in hits
        )

    # 对单篇笔记按文件名、章节标题与正文行打分，并提取不重叠的上下文片段（支持跳过已过期记忆行）
    def _score_file(
        self,
        display_path: str,
        text: str,
        primary: list[str],
        secondary: list[str],
        *,
        skip_expired_today: date | None = None,
    ) -> list[tuple[float, str, int, str, str]]:
        lines = text.splitlines()
        if not lines:
            return []
        expired_flags = [
            (skip_expired_today is not None and is_line_expired(line, today=skip_expired_today))
            for line in lines
        ]
        path_bonus = sum(4.0 for t in primary if t in display_path.lower())
        section = ""
        hits: list[tuple[float, int, str]] = []
        for idx, line in enumerate(lines, start=1):
            if expired_flags[idx - 1]:
                continue
            stripped = line.strip()
            if stripped.startswith("#"):
                section = stripped.lstrip("# ").strip() or section
            line_lower, sec_lower = line.lower(), section.lower()
            score, matched = 0.0, 0
            for term in primary:
                if term in line_lower:
                    score += 5.0 if stripped.startswith("#") else 3.0
                    matched += 1
                elif term in sec_lower:
                    score += 0.6
            if score == 0.0 and secondary:
                sec_hits = sum(1 for t in secondary if t in line_lower)
                if sec_hits >= 2 or (len(secondary) == 1 and sec_hits == 1):
                    score += sec_hits * 0.9
            if score > 0.0:
                hits.append((score + max(0, matched - 1) * 2.0 + (path_bonus * 0.5 if matched else 0.0), idx, section))
        if not hits and path_bonus > 0.0:
            hits.append((path_bonus, 1, ""))
        hits.sort(key=lambda item: (-item[0], item[1]))

        picked: list[tuple[float, str, int, str, str]] = []
        used: list[int] = []
        for score, line_no, sec in hits:
            if any(abs(line_no - prev) <= 4 for prev in used):
                continue
            used.append(line_no)
            snippet = "\n".join(
                f"  {n}: {lines[n - 1]}"
                for n in range(max(1, line_no - 2), min(len(lines), line_no + 2) + 1)
                if not expired_flags[n - 1]
            )
            picked.append((score, display_path, line_no, sec, snippet))
            if len(picked) >= 3:
                break
        return picked

    # 按行号分页读取允许目录下的 Markdown/文本文件，附带总行数和字节截断保护
    def read_markdown(self, path: str, start_line: int = 1, end_line: int = 120) -> str:
        if start_line < 1 or end_line < start_line:
            raise ValueError("行号范围无效：start_line >= 1 且 end_line >= start_line。")
        if end_line - start_line + 1 > self.policy.max_lines:
            raise ValueError(f"单次最多读取 {self.policy.max_lines} 行。")
        target = self.resolve_path(path, write=False)
        if not target.is_file():
            raise ValueError(f"文件不存在：{path}")
        try:
            lines = target.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise ValueError("目标文件不是有效的 UTF-8 文本。") from error
        display_path, total = self._to_display_path(target), len(lines)
        if total == 0:
            return f"文件：{display_path}（空文件，共 0 行）"
        if start_line > total:
            return f"文件：{display_path} 共 {total} 行，没有第 {start_line} 行。"
        actual_end = min(total, end_line)
        body = "\n".join(f"{n}: {lines[n - 1]}" for n in range(start_line, actual_end + 1))
        return truncate_utf8(f"文件：{display_path}（共 {total} 行，显示第 {start_line}-{actual_end} 行）\n{body}", self.policy.max_read_bytes)

    # ---- 白名单受控写入（workspace / MEMORY.md / rules） ----

    # 在 data/workspace 中创建或覆盖文本文件，覆盖已有文件前自动备份至 .backups/
    def write_workspace_file(self, path: str, content: str, mode: str = "create") -> str:
        if mode not in {"create", "overwrite"}:
            raise ValueError("mode 只能是 create 或 overwrite。")
        if not isinstance(content, str) or "\x00" in content:
            raise ValueError("内容必须是有效文本且不能包含 NUL。")
        if len(content.encode("utf-8")) > self.policy.max_write_bytes:
            raise ValueError(f"内容超过 {self.policy.max_write_bytes} 字节写入上限。")
        target = self.resolve_path(path, write=True)
        if target.exists() and target.is_dir():
            raise ValueError("目标路径是目录，不能写入文件。")
        if mode == "create" and target.exists():
            raise ValueError("目标文件已存在；create 模式不会覆盖，请换路径或使用 overwrite。")

        backup_rel: str | None = None
        if mode == "overwrite" and target.exists():
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            backup = self.workspace_dir / ".backups" / f"{stamp}-{target.name}"
            counter = 2
            while backup.exists():
                backup = self.workspace_dir / ".backups" / f"{stamp}-{counter}-{target.name}"
                counter += 1
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
            backup_rel = self._to_display_path(backup)

        _atomic_write(target, content)
        rel_path = self._to_display_path(target)
        self.record_audit(action="write_file", actor="agent", target=rel_path, old_value=backup_rel, new_value=f"{len(content)} chars ({mode})")
        return rel_path

    # 确保指定记忆文件已在 MEMORY.md 索引中登记一次，避免重复改写索引破坏 KV Cache
    def _ensure_index_entry(self, target_path: str, entry_line: str) -> None:
        existing = (
            self.memory_file.read_text(encoding="utf-8")
            if self.memory_file.is_file()
            else "# 记忆索引 (MEMORY.md)\n\n"
        )
        if target_path in existing or Path(target_path).name in existing:
            return
        sep = "" if not existing or existing.endswith("\n") else "\n"
        _atomic_write(self.memory_file, f"{existing}{sep}{entry_line}\n")

    # 将用户原话按时间戳与消息编号幂等追加到 data/memory/facts.md，不撑大 MEMORY.md 索引
    def remember_user_statement(self, user_text: str, message_id: int | None = None) -> str:
        clean = " ".join(user_text.split())
        if not clean:
            raise ValueError("要记住的内容不能为空。")
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        existing = (
            self.facts_file.read_text(encoding="utf-8")
            if self.facts_file.is_file()
            else "# 事实备忘 (facts.md)\n\n"
        )
        if message_id is not None and f"(msg#{message_id})" in existing:
            return "data/memory/facts.md"
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M")
        tag = f" (msg#{message_id})" if message_id is not None else ""
        sep = "" if not existing or existing.endswith("\n") else "\n"
        _atomic_write(self.facts_file, f"{existing}{sep}- [{stamp}]{tag} {clean}\n")
        self._ensure_index_entry(
            "data/memory/facts.md",
            "- [事实备忘](data/memory/facts.md) — 通过 remember_fact 记录的长期事实，需要时请按需检索",
        )
        self.record_audit(action="remember_fact", actor="user", target="data/memory/facts.md", old_value=None, new_value=clean)
        return "data/memory/facts.md"

    # 对指定主题记忆文件执行 add / replace / remove 操作，写入期直接消解冲突并登记 MEMORY.md 索引
    def apply_memory_operations(
        self,
        topic: str,
        title: str,
        index_summary: str,
        operations: list[dict[str, object]],
        *,
        today: date | None = None,
    ) -> str | None:
        return apply_topic_operations(
            self.memory_dir,
            topic=topic,
            title=title,
            index_summary=index_summary,
            operations=operations,
            write_fn=_atomic_write,
            ensure_index_fn=self._ensure_index_entry,
            audit_fn=self.record_audit,
            today=today,
        )

    # 将沉淀提炼的长期记忆按主题去重追加到 data/memory/<topic>.md（兼容旧接口）
    def save_memory_entries(
        self,
        topic: str,
        title: str,
        index_summary: str,
        items: list[str],
        *,
        today: date | None = None,
    ) -> str | None:
        ops = [{"op": "add", "text": str(item)} for item in items if str(item).strip()]
        return self.apply_memory_operations(topic, title, index_summary, ops, today=today)

    # 扫描活跃主题记忆文件，将 (until: YYYY-MM-DD) 已过期的条目原子归档至 data/memory/archive/YYYY-MM.md
    def archive_expired_memories(self, today: date | None = None) -> dict[str, int]:
        return archive_expired_topics(
            self.memory_dir,
            self.archive_dir,
            write_fn=_atomic_write,
            audit_fn=self.record_audit,
            today=today,
        )

    # 将一条行为规则保存为 data/memory/rules/ 下独立可读的 Markdown 文件
    def save_rule(self, statement: str, source: str = "agent") -> Path:
        clean = " ".join(statement.split())
        if not clean:
            raise ValueError("规则内容不能为空。")
        self.rules_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        slug = re.sub(r"[^\w]+", "-", clean).strip("-").lower() or "rule"
        path = self.rules_dir / f"{timestamp}-{slug[:40]}.md"
        counter = 2
        while path.exists():
            path = self.rules_dir / f"{timestamp}-{slug[:36]}-{counter}.md"
            counter += 1
        body = f"# Rule: {clean}\n\n- Statement: {clean}\n- Source: {source}\n- Created: {datetime.now(UTC).isoformat()}\n"
        _atomic_write(path, body)
        self.record_audit(action="save_rule", actor=source, target=f"data/memory/rules/{path.name}", old_value=None, new_value=clean)
        return path

    # 向 data/memory/audit.jsonl 追加一条写入或记忆变更审计日志
    def record_audit(
        self, *, action: str, actor: str, target: str, old_value: str | None, new_value: str | None
    ) -> dict[str, object]:
        entry: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "trace_id": uuid.uuid4().hex[:16],
            "actor": actor,
            "action": action,
            "target": target,
            "old_value": old_value,
            "new_value": new_value,
        }
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    # 读取最近 limit 条记忆与文件操作审计记录
    def audit_entries(self, limit: int = 50) -> list[dict[str, object]]:
        if not self.audit_path.is_file():
            return []
        lines = [line for line in self.audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [json.loads(line) for line in lines[-limit:]]


# 将 MemoryStore 的核心能力打包成 6 个基础内置 Tool（受 config/tools.json 白名单开关控制）
def memory_tools(
    memory: MemoryStore,
    get_user_message: Callable[[], tuple[str, int | None]] | None = None,
) -> list[Tool]:
    # 适配 read_markdown 工具参数（将模型传入的字符串行号转为整数）
    def read_markdown(path: str, start_line: str, end_line: str) -> str:
        try:
            start, end = int(start_line), int(end_line)
        except ValueError as error:
            raise ValueError("start_line 和 end_line 必须是整数。") from error
        return memory.read_markdown(path, start, end)

    # 适配 write_file 工具：写入工作区并返回相对路径提示
    def write_file(path: str, content: str, mode: str) -> str:
        saved = memory.write_workspace_file(path, content, mode=mode)
        return f"已写入：{saved}"

    # 适配 remember_fact 工具：直接保存当前轮用户原话，防止模型篡改事实
    def remember_fact() -> str:
        if get_user_message is None:
            raise RuntimeError("缺少当前用户消息上下文，无法记录用户原话。")
        user_text, message_id = get_user_message()
        target = memory.remember_user_statement(user_text, message_id=message_id)
        return f"已记录用户原话到 {target}"

    # 适配 save_rule 工具：将用户要求遵守的规则落盘到 data/memory/rules/
    def save_rule(statement: str) -> str:
        path = memory.save_rule(statement, source="agent")
        return f"已保存行为规则：data/memory/rules/{path.name}"

    return [
        Tool(
            "list_files",
            "在允许读取的目录内按 glob 列出文件路径，先看结构再决定读哪篇。",
            {"root": "目录起点，如 data/memory、data/workspace、docs", "pattern": "glob 模式，如 *.md"},
            memory.list_files,
        ),
        Tool(
            "search_markdown",
            "按关键词搜索允许目录内 Markdown 文件名、标题与正文上下文。",
            {"query": "搜索关键词", "root": "搜索目录，如 data/memory、data/workspace、docs"},
            memory.search_markdown,
        ),
        Tool(
            "read_markdown",
            "按路径和行号读取允许目录内的 Markdown/文本文件。",
            {"path": "相对路径，如 data/memory/MEMORY.md 或 data/workspace/note.md", "start_line": "起始行号（从1开始）", "end_line": "结束行号"},
            read_markdown,
        ),
        Tool(
            "write_file",
            "在 data/workspace 目录内创建或覆盖文本文件（覆盖前自动备份）。",
            {"path": "data/workspace 内的相对路径", "content": "UTF-8 完整文件内容", "mode": "create 或 overwrite"},
            write_file,
            param_enums={"mode": ("create", "overwrite")},
        ),
        Tool(
            "remember_fact",
            "仅在用户明确要求记住信息时调用，将当前用户原话追加到 data/memory/facts.md（按需检索，不膨胀索引）。",
            {},
            remember_fact,
        ),
        Tool(
            "save_rule",
            "将用户要求遵守的行为规则保存到 data/memory/rules/。",
            {"statement": "规则内容"},
            save_rule,
        ),
    ]
