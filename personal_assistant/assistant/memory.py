"""纯 Markdown 记忆、规则存储以及受目录白名单保护的文件读写核心存储层。"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from personal_assistant.agent.plugins import RetrieverProtocol
from personal_assistant.assistant.common import atomic_write_text, now_shanghai, today_shanghai, truncate_utf8
from personal_assistant.assistant.files_policy import FilesPolicy, is_sensitive_segment
from personal_assistant.assistant.memory_lifecycle import (
    apply_topic_operations, archive_expired_topics, is_line_expired as _is_line_expired,
    prune_raw_facts_for_batch, read_topic_snapshots as _read_topic_snapshots,
)

_atomic_write = atomic_write_text
MAX_SEARCH_FILE_BYTES = 1024 * 1024
__all__ = ["LinearRetriever", "MAX_SEARCH_FILE_BYTES", "MemoryStore", "SearchHit"]


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


# 内置零索引线性扫描检索器（实现 RetrieverProtocol，作为默认无外挂索引时的基准实现）
class LinearRetriever:
    name = "linear"

    # 扫描允许目录下的 Markdown/文本文件并按文件名、标题与正文命中权重排序返回 SearchHit
    def search_hits(
        self, memory: "MemoryStore", query: str, root: str = "data/memory", limit: int = 10, today: date | None = None
    ) -> list[SearchHit]:
        primary, secondary = _extract_terms(query)
        if not primary and not secondary:
            raise ValueError("搜索关键词不能为空。")
        ref_day = today or today_shanghai()
        searchable, _ = memory.iter_searchable_files(root)
        candidates: list[tuple[float, str, int, str, str]] = []
        for resolved, display_path, is_memory in searchable:
            try:
                text = resolved.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            candidates.extend(
                memory._score_file(display_path, text, primary, secondary, skip_expired_today=ref_day if is_memory else None)
            )
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [
            SearchHit(path=path, line=line_no, section=section, snippet=snippet, score=score)
            for score, path, line_no, section, snippet in candidates[:limit]
        ]


# 纯 Markdown 记忆与本地文件管理类：直接读写磁盘 .md/.txt 文件
class MemoryStore:
    # 初始化数据目录、安全策略与检索协议实例；create_dirs=False 时以只读方式打开（供 eval 使用）
    def __init__(
        self,
        data_root: Path | str,
        policy: FilesPolicy | None = None,
        *,
        create_dirs: bool = True,
        retriever: RetrieverProtocol | None = None,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.retriever: RetrieverProtocol = retriever if retriever is not None else LinearRetriever()
        self._write_lock = threading.RLock()
        if policy is None:
            proj = self.data_root.parent
            self.policy = FilesPolicy(
                project_root=proj,
                read_roots=(self.data_root / "memory", self.data_root / "workspace", proj / "docs"),
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
            ):
                directory.mkdir(parents=True, exist_ok=True)
        self._roots = self.policy.build_roots(self.data_root)

    # 将磁盘绝对路径按允许根目录转换为统一的相对展示路径
    def display_path(self, resolved: Path) -> str:
        return self.policy.to_display_path(resolved, self.data_root, roots=self._roots)

    # 安全解析相对路径：委托 FilesPolicy 校验读写白名单、拒绝 .. 穿越、敏感目录与越权软链接
    def resolve_path(self, rel_path: str, *, write: bool = False) -> Path:
        return self.policy.resolve_path(rel_path, self.data_root, write=write, roots=self._roots)

    # 判断目录遍历路径段是否为隐藏文件/目录或受保护敏感段（含 Unicode 折叠检测）
    def _is_ignored_part(self, part: str) -> bool:
        return part.startswith(".") or is_sensitive_segment(part, self.policy.reject_segments)

    # 统计 data/memory 下非隐藏、非敏感段的 Markdown 与纯文本记忆文件数量
    def count_files(self) -> int:
        if not self.memory_dir.is_dir():
            return 0
        return sum(
            1
            for p in self.memory_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in {".md", ".txt"}
            and not any(self._is_ignored_part(part) for part in p.relative_to(self.memory_dir).parts)
        )

    # 读取 data/memory/MEMORY.md 记忆索引（严格限制前 max_lines 行与 max_chars 字符，防止提示词膨胀）
    def read_memory_index(self, max_lines: int = 100, max_chars: int = 4000) -> str:
        if not self.memory_file.is_file():
            return ""
        text = self.memory_file.read_text(encoding="utf-8").strip()
        if not text:
            return ""
        lines = text.splitlines()
        truncated = len(lines) > max_lines
        body = "\n".join(lines[:max_lines])
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
        items = [ln for ln in self.facts_file.read_text(encoding="utf-8").splitlines() if ln.strip().startswith("- ")]
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
        lines = rule.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if line.startswith("- Statement:"):
                return line.removeprefix("- Statement:").strip()
        for line in lines:
            if line.strip():
                return line.strip().lstrip("# ").removeprefix("Rule:").strip() or "（无标题）"
        return "（空规则文件）"

    # 读取全部活跃主题记忆文件快照（排除 MEMORY.md 与 archive/），供沉淀器读旧改旧消冲突
    def read_topic_snapshots(self, max_total_chars: int = 6000) -> str:
        return _read_topic_snapshots(self.memory_dir, max_total_chars=max_total_chars)

    # 判断指定路径（及其软链接自身与目标）是否命中受保护 Inode 集合
    @staticmethod
    def _hits_protected(path: Path, protected: set[tuple[int, int]], *, resolved: Path | None = None) -> bool:
        if not protected:
            return False
        for candidate, stat_fn in ((path, os.lstat), (resolved or path, os.stat)):
            try:
                st = stat_fn(candidate)
                if (st.st_dev, st.st_ino) in protected:
                    return True
            except OSError:
                continue
        return False

    # 过滤目录遍历结果：跳过非文件、隐藏目录、敏感路径段、受保护 Inode 硬链接及越界软链接
    def _safe_files(self, target: Path, iterator: Iterable[Path]) -> list[Path]:
        protected = self.policy.protected_inodes(self.data_root)
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
            if self._hits_protected(item, protected, resolved=resolved):
                continue
            parts = resolved.relative_to(target).parts
            if not any(self._is_ignored_part(p) for p in parts):
                files.append(resolved)
        return files

    # 按 glob 模式列出允许读取目录内的所有文件路径（默认排除 data/memory/archive/ 归档目录）
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
        archive_resolved = self.archive_dir.resolve()
        listing_archive = target.is_relative_to(archive_resolved)
        candidates = [
            p for p in self._safe_files(target, iterator)
            if listing_archive or not p.is_relative_to(archive_resolved)
        ]
        found = [self.display_path(p) for p in candidates[: self.policy.max_results]]
        if not found:
            return "没有匹配的文件。"
        suffix = f"\n（仅展示前 {self.policy.max_results} 个文件）" if len(found) >= self.policy.max_results else ""
        return "\n".join(found) + suffix

    # 安全枚举允许检索的 Markdown/文本文件列表（os.scandir 快速遍历，严格守住白名单、敏感段、软链接与归档隔离）
    def iter_searchable_files(self, root: str = "data/memory") -> tuple[list[tuple[Path, str, bool]], bool]:
        target = self.resolve_path(root, write=False)
        if not target.is_dir():
            raise ValueError(f"目录不存在：{root}")
        archive_resolved, memory_resolved = self.archive_dir.resolve(), self.memory_dir.resolve()
        searching_archive = target.is_relative_to(archive_resolved)
        protected = self.policy.protected_inodes(self.data_root)

        files: list[tuple[Path, str, bool]] = []
        stack: list[Path] = [target]
        while stack:
            current = stack.pop()
            try:
                with os.scandir(current) as it:
                    entries = sorted(it, key=lambda e: e.name)
            except OSError:
                continue
            for entry in entries:
                if self._is_ignored_part(entry.name):
                    continue
                is_link = entry.is_symlink()
                entry_path = Path(entry.path)
                if is_link:
                    try:
                        resolved = entry_path.resolve()
                    except OSError:
                        continue
                    if not resolved.is_relative_to(target) or any(
                        self._is_ignored_part(p) for p in resolved.relative_to(target).parts
                    ):
                        continue
                else:
                    resolved = entry_path

                if not searching_archive and resolved.is_relative_to(archive_resolved):
                    continue
                try:
                    st = resolved.stat() if is_link else entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if protected and (
                    (st.st_dev, st.st_ino) in protected
                    or (is_link and self._hits_protected(entry_path, protected, resolved=resolved))
                ):
                    continue
                if resolved.is_dir():
                    if not is_link:
                        stack.append(resolved)
                    continue
                if resolved.suffix.lower() not in {".md", ".txt"}:
                    continue
                if st.st_size > MAX_SEARCH_FILE_BYTES:
                    continue
                is_memory = not searching_archive and resolved.is_relative_to(memory_resolved)
                files.append((resolved, self.display_path(resolved), is_memory))

        files.sort(key=lambda item: item[1])
        return files, searching_archive

    # 在允许目录下检索 Markdown/文本文件并返回排序后的结构化 SearchHit 列表（统一委派 self.retriever）
    def search_hits(
        self, query: str, root: str = "data/memory", limit: int = 10, *, today: date | None = None
    ) -> list[SearchHit]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("搜索关键词不能为空。")
        ref_day = today or today_shanghai()
        return list(self.retriever.search_hits(self, query, root=root, limit=limit, today=ref_day))

    # 对外搜索接口：将 search_hits 命中结果格式化为带路径、行号与章节的文本片段
    def search_markdown(
        self, query: str, root: str = "data/memory", limit: int = 10, *, today: date | None = None
    ) -> str:
        hits = self.search_hits(query, root=root, limit=limit, today=today)
        if not hits:
            return f"未在 {root} 下找到匹配「{query}」的 Markdown 内容。"
        return "\n\n".join(
            f"{hit.path}:{hit.line} [章节: {hit.section or '正文'}]\n{hit.snippet}" for hit in hits
        )

    # 对单篇笔记按文件名、章节标题与正文行打分，并提取不重叠的上下文片段（支持跳过已过期记忆行）
    def _score_file(
        self, display_path: str, text: str, primary: list[str], secondary: list[str],
        *, skip_expired_today: date | None = None,
    ) -> list[tuple[float, str, int, str, str]]:
        lines = text.splitlines()
        if not lines:
            return []
        expired_flags = [
            skip_expired_today is not None and _is_line_expired(line, today=skip_expired_today)
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
        display_path, total = self.display_path(target), len(lines)
        if total == 0:
            return f"文件：{display_path}（空文件，共 0 行）"
        if start_line > total:
            return f"文件：{display_path} 共 {total} 行，没有第 {start_line} 行。"
        actual_end = min(total, end_line)
        body = "\n".join(f"{n}: {lines[n - 1]}" for n in range(start_line, actual_end + 1))
        return truncate_utf8(
            f"文件：{display_path}（共 {total} 行，显示第 {start_line}-{actual_end} 行）\n{body}",
            self.policy.max_read_bytes,
        )

    # 在 data/workspace 中创建或覆盖文本文件，覆盖已有文件前自动备份至 .backups/
    def write_workspace_file(self, path: str, content: str, mode: str = "create") -> str:
        if mode not in {"create", "overwrite"}:
            raise ValueError("mode 只能是 create 或 overwrite。")
        if not isinstance(content, str) or "\x00" in content:
            raise ValueError("内容必须是有效文本且不能包含 NUL。")
        if len(content.encode("utf-8")) > self.policy.max_write_bytes:
            raise ValueError(f"内容超过 {self.policy.max_write_bytes} 字节写入上限。")
        target = self.resolve_path(path, write=True)

        with self._write_lock:
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
                backup_rel = self.display_path(backup)
            _atomic_write(target, content)
            rel_path = self.display_path(target)
            self.record_audit(
                action="write_file", actor="agent", target=rel_path,
                old_value=backup_rel, new_value=f"{len(content)} chars ({mode})",
            )
            return rel_path

    # 读取已有 UTF-8 文本，文件不存在时返回默认表头
    @staticmethod
    def _read_or_default(path: Path, default: str) -> str:
        return path.read_text(encoding="utf-8") if path.is_file() else default

    # 确保指定记忆文件已在 MEMORY.md 索引中登记一次，避免重复改写索引破坏 KV Cache
    def _ensure_index_entry(self, target_path: str, entry_line: str) -> None:
        with self._write_lock:
            existing = self._read_or_default(self.memory_file, "# 记忆索引 (MEMORY.md)\n\n")
            if target_path in existing or f"/{Path(target_path).name})" in existing:
                return
            sep = "" if not existing or existing.endswith("\n") else "\n"
            _atomic_write(self.memory_file, f"{existing}{sep}{entry_line}\n")

    # 将用户原话按时间戳与消息/回合编号幂等追加到 data/memory/facts.md，不撑大 MEMORY.md 索引
    def remember_user_statement(
        self, user_text: str, message_id: int | None = None, *, turn_tag: str | None = None
    ) -> str:
        clean = " ".join(user_text.split())
        if not clean:
            raise ValueError("要记住的内容不能为空。")
        with self._write_lock:
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            existing = self._read_or_default(self.facts_file, "# 事实备忘 (facts.md)\n\n")
            if message_id is not None and f"(msg#{message_id})" in existing:
                return "data/memory/facts.md"
            if message_id is None and turn_tag and f"(turn#{turn_tag})" in existing:
                return "data/memory/facts.md"
            stamp = now_shanghai().strftime("%Y-%m-%d %H:%M")
            tag = f" (msg#{message_id})" if message_id is not None else (f" (turn#{turn_tag})" if turn_tag else "")
            sep = "" if not existing or existing.endswith("\n") else "\n"
            _atomic_write(self.facts_file, f"{existing}{sep}- [{stamp}]{tag} {clean}\n")
            self._ensure_index_entry(
                "data/memory/facts.md",
                "- [事实备忘](data/memory/facts.md) — 通过 remember_fact 记录的长期事实，需要时请按需检索",
            )
            self.record_audit(action="remember_fact", actor="user", target="data/memory/facts.md", old_value=None, new_value=clean)
            return "data/memory/facts.md"

    # 提交或回滚流内暂存的 (turn#...) 事实标记：成功时绑定正式 (msg#<id>)，失败中断时移除未提交行
    def finalize_turn_fact(self, turn_tag: str, *, message_id: int | None) -> None:
        if not turn_tag:
            return
        marker = f"(turn#{turn_tag})"
        with self._write_lock:
            if not self.facts_file.is_file():
                return
            content = self.facts_file.read_text(encoding="utf-8")
            if marker not in content:
                return
            if message_id is not None:
                _atomic_write(self.facts_file, content.replace(marker, f"(msg#{message_id})"))
                return
            rebuilt = "\n".join(line for line in content.splitlines() if marker not in line).rstrip()
            _atomic_write(self.facts_file, f"{rebuilt}\n" if rebuilt else "")

    # 对指定主题记忆文件执行 add / merge / replace / remove 操作，写入期直接消解冲突、软退休旧事实并登记 MEMORY.md 索引
    def apply_memory_operations(
        self,
        topic: str,
        title: str,
        index_summary: str,
        operations: list[dict[str, object]],
        *,
        today: date | None = None,
        applied_texts: list[str] | None = None,
    ) -> str | None:
        with self._write_lock:
            return apply_topic_operations(
                self.memory_dir,
                topic=topic,
                title=title,
                index_summary=index_summary,
                operations=operations,
                write_fn=_atomic_write,
                ensure_index_fn=self._ensure_index_entry,
                audit_fn=self.record_audit,
                archive_dir=self.archive_dir,
                today=today,
                applied_texts=applied_texts,
            )

    # 扫描活跃主题记忆文件，将 (until: YYYY-MM-DD) 已过期的条目原子归档至 data/memory/archive/YYYY-MM.md
    def archive_expired_memories(self, today: date | None = None) -> dict[str, int]:
        with self._write_lock:
            return archive_expired_topics(
                self.memory_dir, self.archive_dir, write_fn=_atomic_write,
                audit_fn=self.record_audit, ensure_index_fn=self._ensure_index_entry, today=today,
            )

    # 清理 facts.md 中已被本次沉淀规范化吸收的原始 (msg#...) 裸行，防止残留无有效期副本
    def prune_consolidated_raw_facts(self, batch_msg_ids: set[int], written_texts: list[str]) -> int:
        with self._write_lock:
            return prune_raw_facts_for_batch(
                self.memory_dir, batch_msg_ids, written_texts,
                write_fn=_atomic_write, audit_fn=self.record_audit,
            )

    # 将一条行为规则保存为 data/memory/rules/ 下独立可读的 Markdown 文件
    def save_rule(self, statement: str, source: str = "agent") -> Path:
        clean = " ".join(statement.split())
        if not clean:
            raise ValueError("规则内容不能为空。")
        with self._write_lock:
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
            "timestamp": datetime.now(UTC).isoformat(), "trace_id": uuid.uuid4().hex[:16],
            "actor": actor, "action": action, "target": target,
            "old_value": old_value, "new_value": new_value,
        }
        with self._write_lock:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    # 读取最近 limit 条记忆与文件操作审计记录
    def audit_entries(self, limit: int = 50) -> list[dict[str, object]]:
        with self._write_lock:
            if not self.audit_path.is_file():
                return []
            lines = [line for line in self.audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [json.loads(line) for line in lines[-limit:]]
