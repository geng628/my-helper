"""持久化结构化 RAG 检索插件（零第三方依赖）：Markdown 结构分块、独立派生 SQLite 索引库、双语 FTS5 BM25 与 Small-to-Big 上下文扩展。"""
from __future__ import annotations

import sqlite3
import threading
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from personal_assistant.agent.plugins import Plugin, PluginContext
from personal_assistant.assistant.chunker import DEFAULT_CHUNK_MAX_CHARS, DocumentChunk, chunk_markdown
from personal_assistant.assistant.common import require_positive_int, today_shanghai
from personal_assistant.assistant.memory import SearchHit
from personal_assistant.assistant.text_search import (
    parse_retrieval_thresholds,
    tokenize_text,
    validate_retrieval_thresholds,
)

if TYPE_CHECKING:
    from personal_assistant.assistant.memory import MemoryStore

RETRIEVER_NAME = "rag"
_SCHEMA_VERSION = "1"
_CHUNKER_VERSION = "1"
_MAX_SNIPPET_CHARS = 4000
_DEFAULT_CONTEXT_MAX_CHARS = 1200


# 校验派生索引文件路径与父目录不含符号链接，阻断借派生库路径向越界位置写文件
def _assert_safe_index_path(db_path: Path, data_root: Path | None = None) -> Path:
    if db_path.is_symlink() or (db_path.exists() and not db_path.is_file()):
        raise ValueError(f"安全校验失败：派生索引路径不能是符号链接或非普通文件（{db_path.name}）。")
    parent = db_path.parent
    if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
        raise ValueError(f"安全校验失败：派生索引目录不能是符号链接或非目录（{parent.name}）。")
    if data_root is not None:
        base = data_root.resolve()
        if not db_path.resolve(strict=False).is_relative_to(base):
            raise ValueError("安全校验失败：派生索引路径越出数据目录边界。")
    return db_path


# 清理指定磁盘 SQLite 派生库及其 WAL/SHM 附属文件（仅在损坏重建时调用）
def _unlink_sqlite_files(db_path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        candidate = db_path.parent / f"{db_path.name}{suffix}"
        if candidate.is_symlink():
            raise ValueError(f"安全校验失败：派生索引附属路径不能是符号链接（{candidate.name}）。")
        if candidate.exists():
            candidate.unlink()


# 判断两块相邻分块是否同章节、均无过期标签且中间仅隔空行（确保扩展上下文绝不跨越标题或跳过时效行）
def _is_contiguous_peer(lines: list[str], prev: DocumentChunk, cur: DocumentChunk) -> bool:
    if prev.until_date or cur.until_date or prev.section != cur.section:
        return False
    if cur.start_line <= prev.end_line:
        return False
    return all(not lines[ln - 1].strip() for ln in range(prev.end_line + 1, cur.start_line))


# 持久化/内存双用 SQLite 派生索引容器（与权威源文件及 assistant.db 物理隔离，可随时删除无损重建）
class _DerivedRAGIndex:
    # 打开指定磁盘路径（或 :memory:）并校验 Schema 与分块配置版本，损坏或版本变更时自动重建表结构
    def __init__(self, db_path: Path | None, *, chunk_max_chars: int) -> None:
        self.db_path = db_path
        self.chunk_max_chars = chunk_max_chars
        self.conn = self._open_and_initialize()

    # 打开 SQLite 连接并在遇到磁盘派生库损坏时安全清理并重建
    def _open_and_initialize(self) -> sqlite3.Connection:
        if self.db_path is None:
            conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._ensure_schema(conn)
            return conn

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        _assert_safe_index_path(self.db_path)
        try:
            conn = self._connect_disk(self.db_path)
            self._ensure_schema(conn)
            return conn
        except sqlite3.DatabaseError:
            _unlink_sqlite_files(self.db_path)
            conn = self._connect_disk(self.db_path)
            self._ensure_schema(conn)
            return conn

    # 以 WAL + NORMAL 同步模式打开磁盘派生 SQLite 连接（派生缓存优先低延迟与崩溃一致性）
    @staticmethod
    def _connect_disk(db_path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA busy_timeout = 5000")
            return conn
        except BaseException:
            conn.close()
            raise

    # 建表并校验 index_meta 版本指纹；若 Schema 或分块参数发生变化则清空派生表待增量重填
    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        expected_meta = {
            "schema_version": _SCHEMA_VERSION,
            "chunker_version": _CHUNKER_VERSION,
            "chunk_max_chars": str(self.chunk_max_chars),
        }
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS index_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            existing_meta = dict(conn.execute("SELECT key, value FROM index_meta").fetchall())
            if existing_meta and existing_meta != expected_meta:
                conn.execute("DROP TABLE IF EXISTS chunks_fts")
                conn.execute("DROP TABLE IF EXISTS chunks")
                conn.execute("DROP TABLE IF EXISTS indexed_files")
                conn.execute("DELETE FROM index_meta")
                existing_meta = {}

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS indexed_files (
                    root_key TEXT NOT NULL,
                    path TEXT NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    size INTEGER NOT NULL,
                    ino INTEGER NOT NULL,
                    is_archive INTEGER NOT NULL,
                    PRIMARY KEY (root_key, path)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    root_key TEXT NOT NULL,
                    path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    section TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    text TEXT NOT NULL,
                    until_date TEXT NOT NULL,
                    is_archive INTEGER NOT NULL,
                    join_prev INTEGER NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_root_path_idx ON chunks(root_key, path, chunk_index)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash)")
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    path_tokens,
                    section_tokens,
                    body_tokens,
                    context_tokens,
                    tokenize='unicode61 tokenchars ''_'''
                )
                """
            )
            if not existing_meta:
                conn.executemany(
                    "INSERT OR REPLACE INTO index_meta (key, value) VALUES (?, ?)",
                    list(expected_meta.items()),
                )

    # 按文件指纹 (mtime_ns, size, ino) 增量同步目标根目录：未修改文件零重读、删改文件原子更新
    def sync(
        self,
        root_key: str,
        searchable: list[tuple[Path, str, bool]],
        *,
        is_archive_root: bool,
    ) -> None:
        existing_fps: dict[str, tuple[int, int, int, int]] = {
            str(path): (int(mtime_ns), int(size), int(ino), int(is_arch))
            for path, mtime_ns, size, ino, is_arch in self.conn.execute(
                "SELECT path, mtime_ns, size, ino, is_archive FROM indexed_files WHERE root_key = ?",
                (root_key,),
            ).fetchall()
        }

        archive_flag = 1 if is_archive_root else 0
        current_paths: set[str] = set()
        to_reindex: list[tuple[Path, str, bool, tuple[int, int, int, int]]] = []

        for resolved, display_path, is_memory in searchable:
            try:
                st = resolved.stat()
            except OSError:
                continue
            fp = (st.st_mtime_ns, st.st_size, st.st_ino, archive_flag)
            current_paths.add(display_path)
            if existing_fps.get(display_path) != fp:
                to_reindex.append((resolved, display_path, is_memory, fp))

        removed_paths = [p for p in existing_fps if p not in current_paths]
        if not removed_paths and not to_reindex:
            return

        with self.conn:
            for old_path in removed_paths:
                self._delete_path(root_key, old_path)

            for resolved, display_path, is_memory, fp in to_reindex:
                try:
                    raw_text = resolved.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    self._delete_path(root_key, display_path)
                    continue
                self._delete_path(root_key, display_path)
                self._index_file(
                    root_key,
                    display_path,
                    raw_text,
                    track_expiry=is_memory,
                    is_archive=archive_flag,
                )
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO indexed_files
                        (root_key, path, mtime_ns, size, ino, is_archive)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (root_key, display_path, fp[0], fp[1], fp[2], fp[3]),
                )

    # 删除指定根目录与文件路径对应的全部 chunks、FTS5 倒排条目及文件指纹记录
    def _delete_path(self, root_key: str, display_path: str) -> None:
        rows = self.conn.execute(
            "SELECT id FROM chunks WHERE root_key = ? AND path = ?",
            (root_key, display_path),
        ).fetchall()
        if rows:
            ids = [(r[0],) for r in rows]
            self.conn.executemany("DELETE FROM chunks_fts WHERE rowid = ?", ids)
            self.conn.execute(
                "DELETE FROM chunks WHERE root_key = ? AND path = ?",
                (root_key, display_path),
            )
        self.conn.execute(
            "DELETE FROM indexed_files WHERE root_key = ? AND path = ?",
            (root_key, display_path),
        )

    # 对单篇文档执行结构分块并写入 chunks 与双语 FTS5（低权 context_tokens 仅采同章节无过期相邻块边界行）
    def _index_file(
        self,
        root_key: str,
        display_path: str,
        text: str,
        *,
        track_expiry: bool,
        is_archive: int,
    ) -> None:
        chunks = chunk_markdown(
            display_path,
            text,
            max_chars=self.chunk_max_chars,
            track_expiry=track_expiry,
        )
        if not chunks:
            return
        lines = text.splitlines()
        path_tok = " ".join(tokenize_text(display_path, for_query=False))
        join_prev_flags = [
            _is_contiguous_peer(lines, chunks[i - 1], chunks[i]) if i > 0 else False
            for i in range(len(chunks))
        ]
        # 预取每个块首尾单行分词，供紧邻常驻同章节块填入低权 context_tokens（避免块边界割裂跨行多词查询）
        first_line_toks = [
            " ".join(tokenize_text(chunk.text.splitlines()[0], for_query=False))
            if chunk.text
            else ""
            for chunk in chunks
        ]
        last_line_toks = [
            " ".join(tokenize_text(chunk.text.splitlines()[-1], for_query=False))
            if chunk.text
            else ""
            for chunk in chunks
        ]

        for idx, chunk in enumerate(chunks):
            sec_tok = " ".join(tokenize_text(chunk.section, for_query=False))
            body_tok = " ".join(tokenize_text(chunk.text, for_query=False))
            ctx_parts: list[str] = []
            if join_prev_flags[idx] and last_line_toks[idx - 1]:
                ctx_parts.append(last_line_toks[idx - 1])
            if idx + 1 < len(chunks) and join_prev_flags[idx + 1] and first_line_toks[idx + 1]:
                ctx_parts.append(first_line_toks[idx + 1])
            ctx_tok = " ".join(ctx_parts)

            cur = self.conn.execute(
                """
                INSERT INTO chunks (
                    root_key, path, chunk_index, start_line, end_line,
                    section, content_hash, text, until_date, is_archive, join_prev
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    root_key,
                    chunk.path,
                    chunk.chunk_index,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.section,
                    chunk.content_hash,
                    chunk.text,
                    chunk.until_date,
                    is_archive,
                    1 if join_prev_flags[idx] else 0,
                ),
            )
            rowid = cur.lastrowid
            self.conn.execute(
                """
                INSERT INTO chunks_fts (rowid, path_tokens, section_tokens, body_tokens, context_tokens)
                VALUES (?, ?, ?, ?, ?)
                """,
                (rowid, path_tok, sec_tok, body_tok, ctx_tok),
            )

    # Small-to-Big 上下文扩展：在严格同章节、无过期且仅隔空行的边界内向左右各扩展最多 1 块，绝不越界污染或重复已展示行
    def expand_chunk_window(
        self,
        root_key: str,
        path: str,
        chunk_index: int,
        start_line: int,
        end_line: int,
        text: str,
        until_date: str,
        join_prev: int,
        *,
        max_context_chars: int,
        blocked_spans: list[tuple[int, int]] | None = None,
    ) -> tuple[int, int, str]:
        if until_date or max_context_chars <= len(text):
            return start_line, end_line, text

        peers = {
            int(c_idx): (int(s_ln), int(e_ln), str(c_txt), str(u_date), int(j_prev))
            for c_idx, s_ln, e_ln, c_txt, u_date, j_prev in self.conn.execute(
                """
                SELECT chunk_index, start_line, end_line, text, until_date, join_prev
                FROM chunks
                WHERE root_key = ? AND path = ? AND chunk_index IN (?, ?)
                """,
                (root_key, path, chunk_index - 1, chunk_index + 1),
            ).fetchall()
        }

        blocked = blocked_spans or []

        # 判断候选邻居块行区间是否已与同文件更高分命中重叠，避免向模型重复展示同一段正文
        def _overlaps_blocked(span_start: int, span_end: int) -> bool:
            return any(span_start <= p_end and span_end >= p_start for p_start, p_end in blocked)

        cur_start, cur_end, cur_text = start_line, end_line, text

        # 优先尝试拼接左侧紧邻同章节块（必须由当前块 join_prev == 1 证明无标题/无过期行阻隔）
        left = peers.get(chunk_index - 1)
        if join_prev == 1 and left is not None and not left[3] and not _overlaps_blocked(left[0], left[1]):
            gap = cur_start - left[1] - 1
            if gap >= 0:
                sep = "\n" * (gap + 1)
                candidate = f"{left[2]}{sep}{cur_text}"
                if len(candidate) <= max_context_chars:
                    cur_start = left[0]
                    cur_text = candidate

        # 再尝试拼接右侧紧邻同章节块（必须由右块 join_prev == 1 证明无标题/无过期行阻隔）
        right = peers.get(chunk_index + 1)
        if right is not None and right[4] == 1 and not right[3] and not _overlaps_blocked(right[0], right[1]):
            gap = right[0] - cur_end - 1
            if gap >= 0:
                sep = "\n" * (gap + 1)
                candidate = f"{cur_text}{sep}{right[2]}"
                if len(candidate) <= max_context_chars:
                    cur_end = right[1]
                    cur_text = candidate

        return cur_start, cur_end, cur_text

    # 关闭底层 SQLite 连接
    def close(self) -> None:
        self.conn.close()


# 持久化结构化 RAG 检索插件配置项（从 config/tools.json 的 retrieval 段读取）
@dataclass(frozen=True)
class RAGConfig:
    min_term_coverage: float = 0.22
    min_bm25_score: float = 0.1
    max_hits_per_file: int = 2
    chunk_max_chars: int = DEFAULT_CHUNK_MAX_CHARS
    context_max_chars: int = _DEFAULT_CONTEXT_MAX_CHARS
    index_path: str | None = None

    # 校验覆盖率、BM25 下限、单文件命中上限、分块长度与扩展窗口长度的合法范围
    def __post_init__(self) -> None:
        validate_retrieval_thresholds(self.min_term_coverage, self.min_bm25_score)
        require_positive_int(self.max_hits_per_file, "retrieval.max_hits_per_file")
        require_positive_int(self.chunk_max_chars, "retrieval.chunk_max_chars")
        require_positive_int(self.context_max_chars, "retrieval.context_max_chars")
        if self.index_path is not None and (not isinstance(self.index_path, str) or not self.index_path.strip()):
            raise ValueError("retrieval.index_path 必须是非空路径字符串或省略。")

    # 从配置字典构造 RAGConfig 并执行严格数值校验
    @classmethod
    def from_dict(cls, raw: Mapping[str, object] | None) -> "RAGConfig":
        cfg = raw or {}
        coverage, min_bm25 = parse_retrieval_thresholds(cfg)

        # 解析并校验可选正整数配置项，显式拒绝布尔值
        def _parse_int(key: str, default: int) -> int:
            val = cfg.get(key, default)
            if isinstance(val, bool):
                raise ValueError(f"retrieval.{key} 必须是正整数。")
            if val is not None and not isinstance(val, (int, float, str)):
                raise TypeError(f"retrieval.{key} 必须是整数。")
            return default if val is None else int(val)

        per_file = _parse_int("max_hits_per_file", 2)
        chunk_chars = _parse_int("chunk_max_chars", DEFAULT_CHUNK_MAX_CHARS)
        context_chars = _parse_int("context_max_chars", _DEFAULT_CONTEXT_MAX_CHARS)

        raw_index_path = cfg.get("index_path")
        if raw_index_path is not None and not isinstance(raw_index_path, str):
            raise TypeError("retrieval.index_path 必须是字符串。")
        index_path = raw_index_path.strip() if isinstance(raw_index_path, str) and raw_index_path.strip() else None

        return cls(
            min_term_coverage=coverage,
            min_bm25_score=min_bm25,
            max_hits_per_file=per_file,
            chunk_max_chars=chunk_chars,
            context_max_chars=context_chars,
            index_path=index_path,
        )


# 持久化结构化 RAG 检索引擎（实现 RetrieverProtocol，线程安全，读写模式落盘到 data/state/rag_index.db，只读模式零写盘）
class RAGRetriever:
    name = RETRIEVER_NAME

    # 初始化检索配置、线程锁与按数据根目录隔离的派生 SQLite 索引句柄缓存
    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig()
        self._lock = threading.RLock()
        self._indices: dict[Path | None, _DerivedRAGIndex] = {}

    # 解析当前 MemoryStore 对应的派生索引路径：读写模式落盘到 data/state/rag_index.db，只读模式返回 None 使用内存库
    def _resolve_db_path(self, memory: "MemoryStore") -> Path | None:
        if self.config.index_path is not None:
            candidate = Path(self.config.index_path)
            if not candidate.is_absolute():
                candidate = memory.data_root / candidate
            return _assert_safe_index_path(candidate)
        if memory.read_only:
            return None
        db_path = memory.data_root / "state" / "rag_index.db"
        return _assert_safe_index_path(db_path, data_root=memory.data_root)

    # 获取或懒初始化指定派生库连接；若磁盘派生库在运行期被外部删除则自动重开重建
    def _get_index(self, memory: "MemoryStore") -> _DerivedRAGIndex:
        db_path = self._resolve_db_path(memory)
        cache_key = db_path if db_path is not None else memory.data_root
        existing = self._indices.get(cache_key)
        if existing is not None:
            if existing.db_path is not None and not existing.db_path.is_file():
                existing.close()
                self._indices.pop(cache_key, None)
            else:
                return existing
        created = _DerivedRAGIndex(db_path, chunk_max_chars=self.config.chunk_max_chars)
        self._indices[cache_key] = created
        return created

    # 将命中分块原文渲染为带精确行号前缀的片段，并对超大块执行友好截断提示
    @staticmethod
    def _format_chunk_snippet(text: str, start_line: int) -> str:
        numbered = "\n".join(
            f"  {n}: {line}"
            for n, line in enumerate(text.splitlines(), start=start_line)
        )
        if len(numbered) <= _MAX_SNIPPET_CHARS:
            return numbered
        cut = numbered[:_MAX_SNIPPET_CHARS].rsplit("\n", 1)[0]
        return f"{cut}\n  …（分块内容较长，完整内容请调用 read_markdown 按行查看）"

    # 执行增量同步 + SQLite FTS5 BM25 检索 + 覆盖率门控 + Small-to-Big 上下文扩展与多样性重排
    def search_hits(
        self,
        memory: "MemoryStore",
        query: str,
        root: str = "data/memory",
        limit: int = 10,
        today: date | None = None,
    ) -> list[SearchHit]:
        query_terms = list(dict.fromkeys(tokenize_text(query, for_query=True)))
        if not query_terms:
            raise ValueError("搜索关键词不能为空。")

        ref_day_iso = (today or today_shanghai()).isoformat()
        target_dir = memory.resolve_path(root, write=False)
        searchable, searching_archive = memory.iter_searchable_files(root)
        if not searchable:
            return []

        root_key = str(target_dir.resolve())
        fts_expr = " OR ".join(f'"{term}"' for term in query_terms)
        query_set = set(query_terms)
        min_matched = 2 if len(query_set) >= 4 else 1
        candidate_limit = max(limit * 16, 80)

        with self._lock:
            index = self._get_index(memory)
            index.sync(root_key, searchable, is_archive_root=searching_archive)

            rows = index.conn.execute(
                """
                SELECT
                    c.path,
                    c.chunk_index,
                    c.start_line,
                    c.end_line,
                    c.section,
                    c.text,
                    c.until_date,
                    c.join_prev,
                    -bm25(chunks_fts, 0.4, 1.5, 3.0, 0.4) AS bm25_score,
                    f.path_tokens,
                    f.section_tokens,
                    f.body_tokens,
                    f.context_tokens
                FROM chunks_fts AS f
                JOIN chunks AS c ON c.id = f.rowid
                WHERE chunks_fts MATCH ?
                  AND c.root_key = ?
                  AND (? = 1 OR c.is_archive = 0)
                  AND (c.until_date = '' OR c.until_date >= ?)
                ORDER BY bm25(chunks_fts, 0.4, 1.5, 3.0, 0.4)
                LIMIT ?
                """,
                (fts_expr, root_key, 1 if searching_archive else 0, ref_day_iso, candidate_limit),
            ).fetchall()

            total_chunks = int(
                index.conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE root_key = ?",
                    (root_key,),
                ).fetchone()[0]
            )
            scored: list[tuple[float, str, int, int, int, str, str, str, int]] = []
            for (
                path,
                chunk_index,
                start_line,
                end_line,
                section,
                body_text,
                until_date,
                join_prev,
                bm25_score,
                p_tok,
                s_tok,
                b_tok,
                c_tok,
            ) in rows:
                bm25_val = max(0.0, float(bm25_score))
                core_terms = set(f"{p_tok} {s_tok} {b_tok}".split())
                all_terms = core_terms | set(str(c_tok).split())
                matched_core = query_set & core_terms
                matched_all = query_set & all_terms
                if not matched_core or len(matched_all) < min_matched:
                    continue
                coverage = len(matched_all) / len(query_set)
                high_coverage = coverage >= 0.6
                if total_chunks >= 10 and bm25_val < self.config.min_bm25_score and not high_coverage:
                    continue
                high_conf_bm25 = len(matched_all) >= 3 and bm25_val >= 10.0
                if coverage < self.config.min_term_coverage and not high_conf_bm25:
                    continue
                body_terms = set(str(b_tok).split())
                body_ratio = len(query_set & body_terms) / len(query_set)
                base_score = max(bm25_val, self.config.min_bm25_score, 0.1)
                final_score = round(base_score * (1.0 + 0.6 * coverage + 0.4 * body_ratio), 4)
                scored.append(
                    (
                        final_score,
                        str(path),
                        int(chunk_index),
                        int(start_line),
                        int(end_line),
                        str(section),
                        str(body_text),
                        str(until_date),
                        int(join_prev),
                    )
                )

            scored.sort(key=lambda item: (-item[0], item[1], item[3], item[4]))

            # 1. Small-to-Big 上下文扩展 + 同文件区间去重（防止同一上下文窗口被重复展示）
            expanded: list[tuple[float, str, int, int, str, str]] = []
            used_spans_by_file: dict[str, list[tuple[int, int]]] = {}
            for score, path, c_idx, s_ln, e_ln, section, body_text, until_date, join_prev in scored:
                prev_spans = used_spans_by_file.setdefault(path, [])
                if any(s_ln <= p_end and e_ln >= p_start for p_start, p_end in prev_spans):
                    continue
                exp_start, exp_end, exp_text = index.expand_chunk_window(
                    root_key,
                    path,
                    c_idx,
                    s_ln,
                    e_ln,
                    body_text,
                    until_date,
                    join_prev,
                    max_context_chars=self.config.context_max_chars,
                    blocked_spans=prev_spans,
                )
                prev_spans.append((exp_start, exp_end))
                expanded.append((score, path, exp_start, exp_end, section, exp_text))

            # 2. 按 max_hits_per_file 做多文件多样性优先挑选，剩余名额从溢出候选中补齐
            primary_picks: list[tuple[float, str, int, int, str, str]] = []
            overflow: list[tuple[float, str, int, int, str, str]] = []
            per_file_counts: dict[str, int] = {}
            cap = self.config.max_hits_per_file if limit >= 4 else limit
            for item in expanded:
                path = item[1]
                count = per_file_counts.get(path, 0)
                if count < cap:
                    primary_picks.append(item)
                    per_file_counts[path] = count + 1
                else:
                    overflow.append(item)

            chosen = (primary_picks + overflow)[:limit]
            return [
                SearchHit(
                    path=path,
                    start_line=start_line,
                    end_line=end_line,
                    section=section,
                    snippet=self._format_chunk_snippet(body_text, start_line),
                    score=score,
                )
                for score, path, start_line, end_line, section, body_text in chosen
            ]

    # 关闭本实例持有的全部派生 SQLite 连接并清空缓存
    def close(self) -> None:
        with self._lock:
            for index in self._indices.values():
                index.close()
            self._indices.clear()


# 检索插件工厂函数：按 config/tools.json 的 retrieval 段装配实现 RetrieverProtocol 的 RAGRetriever
def create_retriever(config: dict[str, object]) -> RAGRetriever:
    return RAGRetriever(RAGConfig.from_dict(config))


# 持久化结构化 RAG 主板插件（外层继承 Plugin ABC，内层向 retrievers 插槽注册 rag 检索引擎并托管连接回收）
class RAGRetrieverPlugin(Plugin):
    plugin_id = RETRIEVER_NAME

    # 初始化活跃检索实例弱引用集合，确保 terminate() 可统一释放 SQLite 句柄而不阻碍垃圾回收
    def __init__(self, context: PluginContext) -> None:
        super().__init__(context)
        self._instances: weakref.WeakSet[RAGRetriever] = weakref.WeakSet()

    # 统一初始化生命周期：向 self.context.register_retriever 插槽登记 rag 检索引擎工厂
    def initialize(self) -> None:
        # 创建并登记 RAGRetriever 实例，便于插件停用时统一关闭 SQLite 连接
        def _tracked_factory(cfg: dict[str, object]) -> RAGRetriever:
            instance = create_retriever(cfg)
            self._instances.add(instance)
            return instance

        self.context.register_retriever(self.plugin_id, _tracked_factory)

    # 统一停用清理生命周期：关闭本插件创建的所有 RAGRetriever SQLite 派生库连接
    def terminate(self) -> None:
        for instance in list(self._instances):
            instance.close()
        self._instances.clear()
