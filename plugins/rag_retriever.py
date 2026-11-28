"""持久化两层父子混合 RAG 检索插件（零第三方依赖）：父子表分离、整句 Overlap、上下文摘要缓存、float32 向量缓存与 BM25+Dense RRF 融合。"""
from __future__ import annotations

import hashlib
import math
import os
import sqlite3
import threading
import time
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from personal_assistant.agent.plugins import Plugin, PluginContext
from personal_assistant.ai import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    AbortSignal,
    EmbeddingInvoker,
    LLMNotConfigured,
    ModelInvoker,
    StreamAbortedError,
    cosine_similarity,
    normalize_vector,
    pack_vector,
    unpack_vector,
)
from personal_assistant.assistant.chunker import (
    DEFAULT_CHILD_MAX_CHARS,
    DEFAULT_CHILD_OVERLAP_CHARS,
    DEFAULT_PARENT_MAX_CHARS,
    ParentChunk,
    chunk_markdown_hierarchy,
    format_context_header,
    strip_expiring_lines,
)
from personal_assistant.assistant.common import require_positive_int, today_shanghai
from personal_assistant.assistant.memory import SearchHit
from personal_assistant.assistant.rag_contextualizer import generate_document_contexts
from personal_assistant.assistant.text_search import (
    parse_retrieval_thresholds,
    tokenize_text,
    validate_retrieval_thresholds,
)

if TYPE_CHECKING:
    from personal_assistant.assistant.memory import MemoryStore

_T = TypeVar("_T")

RETRIEVER_NAME = "rag"
_SCHEMA_VERSION = "3"
_CHUNKER_VERSION = "2"
_MAX_SNIPPET_CHARS = 4000
_MAX_VECTOR_CACHE_ITEMS = 20_000
_SQLITE_SIDECAR_SUFFIXES = ("", "-wal", "-shm", "-journal")
_REQUIRED_TABLES = frozenset(
    {
        "index_meta",
        "indexed_files",
        "parent_chunks",
        "child_chunks",
        "child_chunks_fts",
        "chunk_contexts",
        "chunk_embeddings",
    }
)
_CORRUPT_ERROR_CODES = frozenset(
    code
    for code in (
        getattr(sqlite3, "SQLITE_CORRUPT", None),
        getattr(sqlite3, "SQLITE_NOTADB", None),
        getattr(sqlite3, "SQLITE_FORMAT", None),
    )
    if isinstance(code, int)
)

_ParentHitPayload = tuple[float, str, int, int, str, str, str, str]


# 精确判断异常是否属于 SQLite 文件物理损坏（严禁把锁竞争 SQLITE_BUSY/LOCKED/READONLY/IOERR 误判为损坏而删库）
def _is_corruption_error(error: sqlite3.DatabaseError) -> bool:
    code = getattr(error, "sqlite_errorcode", None)
    if isinstance(code, int):
        primary = code & 0xFF
        if code in _CORRUPT_ERROR_CODES or primary in _CORRUPT_ERROR_CODES:
            return True
        return False
    msg = str(error).lower()
    if any(tok in msg for tok in ("locked", "busy", "readonly", "permission", "disk i/o")):
        return False
    return any(tok in msg for tok in ("malformed", "not a database", "file is encrypted", "corrupt"))


# 逐级校验 data_root 到目标派生库及其 WAL/SHM/Journal 附属路径零符号链接且严格位于 data_root 边界内
def _assert_safe_index_path(db_path: Path, *, data_root: Path) -> Path:
    if data_root.is_symlink() or (data_root.exists() and not data_root.is_dir()):
        raise ValueError(f"安全校验失败：数据目录不能是符号链接或非目录（{data_root.name}）。")
    base = data_root.resolve()
    try:
        rel_parts = db_path.relative_to(base).parts
    except ValueError as error:
        raise ValueError("安全校验失败：派生索引路径越出数据目录边界。") from error
    if not rel_parts or ".." in rel_parts:
        raise ValueError("安全校验失败：派生索引路径越出数据目录边界。")

    # 逐级校验中间父目录（在 mkdir 与打开连接前即拦截祖先符号链接）
    cur = base
    for part in rel_parts[:-1]:
        cur = cur / part
        if cur.is_symlink() or (cur.exists() and not cur.is_dir()):
            raise ValueError(f"安全校验失败：派生索引祖先目录不能是符号链接或非目录（{cur.name}）。")

    # 校验主库文件与全部 SQLite 附属文件（-wal / -shm / -journal）不含符号链接或非普通文件
    for suffix in _SQLITE_SIDECAR_SUFFIXES:
        candidate = db_path.parent / f"{db_path.name}{suffix}"
        if candidate.is_symlink() or (candidate.exists() and not candidate.is_file()):
            raise ValueError(f"安全校验失败：派生索引路径不能是符号链接或非普通文件（{candidate.name}）。")

    if not db_path.resolve(strict=False).is_relative_to(base):
        raise ValueError("安全校验失败：派生索引路径越出数据目录边界。")
    return db_path


# 清理指定磁盘 SQLite 派生库及其 WAL/SHM/Journal 附属文件（仅在确认物理损坏重建前调用）
def _unlink_sqlite_files(db_path: Path, *, data_root: Path) -> None:
    _assert_safe_index_path(db_path, data_root=data_root)
    for suffix in _SQLITE_SIDECAR_SUFFIXES:
        candidate = db_path.parent / f"{db_path.name}{suffix}"
        candidate.unlink(missing_ok=True)


# 读取磁盘路径的 (st_dev, st_ino) 身份（不跟随符号链接），不存在时返回 None
def _file_identity(path: Path) -> tuple[int, int] | None:
    try:
        st = os.lstat(path)
    except OSError:
        return None
    return (st.st_dev, st.st_ino)


# 校验非负整数参数（严格拒绝布尔值与浮点数）
def _require_non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} 必须是非负整数。")
    return value


# 校验有限浮点参数位于指定闭区间内（严格拒绝布尔值、NaN 与 Inf）
def _require_bounded_float(value: object, name: str, *, low: float, high: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是有限数字。")
    val = float(value)
    if not math.isfinite(val) or val < low or (high is not None and val > high):
        bounds = f"[{low:g}, {high:g}]" if high is not None else f">= {low:g}"
        raise ValueError(f"{name} 必须是有限数字且满足 {bounds}。")
    return val


# 构造单个子块的统一向量化文本（含来源/章节/背景摘要头）及其 SHA-256 缓存键 embed_hash
def _child_embed_payload(path: str, section: str, context_summary: str, child_text: str) -> tuple[str, str]:
    header = format_context_header(path, section, context_summary)
    embed_text = f"{header}\n{child_text}"
    digest = hashlib.sha256(embed_text.encode("utf-8")).hexdigest()
    return embed_text, digest


# 统一提取并去空白向量客户端的规范模型标识（未配置或为空时返回 None）
def _clean_embedder_model(embedder: EmbeddingInvoker | None) -> str | None:
    if embedder is None:
        return None
    clean = str(embedder.model_id).strip()
    return clean or None


# 读取向量客户端显式声明的目标维度（未声明或非正整数时返回 None）
def _embedder_target_dims(embedder: EmbeddingInvoker | None) -> int | None:
    if embedder is None:
        return None
    raw = getattr(embedder, "dimensions", None)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        return None
    return raw


# 持久化/内存双用两层父子 SQLite 派生索引容器（与权威源文件物理隔离，chunk_contexts 与 chunk_embeddings 跨配置重建保留）
class _DerivedRAGIndex:
    # 打开指定磁盘路径（或 :memory:）并校验 Schema 与父子分块配置版本，确认物理损坏时安全自愈重建
    def __init__(
        self,
        db_path: Path | None,
        *,
        data_root: Path | None,
        parent_max_chars: int,
        child_max_chars: int,
        child_overlap_chars: int,
        context_model: str,
    ) -> None:
        self.db_path = db_path
        self.data_root = data_root
        self.parent_max_chars = parent_max_chars
        self.child_max_chars = child_max_chars
        self.child_overlap_chars = child_overlap_chars
        self.context_model = context_model
        self._vec_cache: dict[tuple[str, str, int], tuple[float, ...]] = {}
        self.conn = self._open_and_initialize()
        self.file_identity = _file_identity(self.db_path) if self.db_path is not None else None

    # 打开 SQLite 连接并在确认物理损坏时安全清理并重建（非损坏类错误如 BUSY/LOCKED 绝不删库）
    def _open_and_initialize(self) -> sqlite3.Connection:
        if self.db_path is None:
            conn = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
            try:
                self._ensure_schema(conn)
                return conn
            except BaseException:
                conn.close()
                raise

        if self.data_root is None:
            raise ValueError("磁盘派生索引必须绑定 data_root 安全边界。")
        _assert_safe_index_path(self.db_path, data_root=self.data_root)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        _assert_safe_index_path(self.db_path, data_root=self.data_root)
        try:
            conn = self._connect_disk(self.db_path)
            try:
                self._ensure_schema(conn)
                return conn
            except BaseException:
                conn.close()
                raise
        except sqlite3.DatabaseError as error:
            if not _is_corruption_error(error):
                raise
            _unlink_sqlite_files(self.db_path, data_root=self.data_root)
            conn = self._connect_disk(self.db_path)
            try:
                self._ensure_schema(conn)
                return conn
            except BaseException:
                conn.close()
                raise

    # 以显式事务模式（isolation_level=None）与 WAL + NORMAL 打开磁盘派生 SQLite 连接（安全处理并发首次切换 WAL 竞争）
    @staticmethod
    def _connect_disk(db_path: Path, *, busy_timeout_ms: int = 5000) -> sqlite3.Connection:
        timeout_sec = max(0.01, busy_timeout_ms / 1000.0)
        conn = sqlite3.connect(str(db_path), timeout=timeout_sec, check_same_thread=False, isolation_level=None)
        try:
            conn.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
            deadline = time.monotonic() + timeout_sec
            while True:
                try:
                    cur_mode_row = conn.execute("PRAGMA journal_mode").fetchone()
                    cur_mode = str(cur_mode_row[0]).lower() if cur_mode_row else ""
                    if cur_mode != "wal":
                        conn.execute("PRAGMA journal_mode = WAL")
                    break
                except sqlite3.OperationalError as error:
                    if "locked" not in str(error).lower() and "busy" not in str(error).lower():
                        raise
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.02)
            conn.execute("PRAGMA synchronous = NORMAL")
            return conn
        except BaseException:
            conn.close()
            raise

    # 检查并初始化两层父子表、chunk_contexts 与 chunk_embeddings 缓存表（配置变更重建时双缓存表均保留不丢失）
    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        expected_meta = {
            "schema_version": _SCHEMA_VERSION,
            "chunker_version": _CHUNKER_VERSION,
            "parent_max_chars": str(self.parent_max_chars),
            "child_max_chars": str(self.child_max_chars),
            "child_overlap_chars": str(self.child_overlap_chars),
        }
        # 只读快路径：若派生库表结构与元数据指纹均已匹配，无需申请写锁
        tables = {
            str(r[0])
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name IN ("
                "'index_meta', 'indexed_files', 'parent_chunks', 'child_chunks', "
                "'child_chunks_fts', 'chunk_contexts', 'chunk_embeddings')"
            ).fetchall()
        }
        if tables == _REQUIRED_TABLES:
            existing_ro = dict(conn.execute("SELECT key, value FROM index_meta").fetchall())
            if existing_ro == expected_meta:
                return

        conn.execute("BEGIN IMMEDIATE")
        try:
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
                # 配置或版本变更时仅重建派生分块与倒排表，绝不清空已付费生成的 chunk_contexts 与 chunk_embeddings
                conn.execute("DROP TABLE IF EXISTS child_chunks_fts")
                conn.execute("DROP TABLE IF EXISTS child_chunks")
                conn.execute("DROP TABLE IF EXISTS parent_chunks")
                conn.execute("DROP TABLE IF EXISTS chunks_fts")
                conn.execute("DROP TABLE IF EXISTS chunks")
                conn.execute("DROP TABLE IF EXISTS indexed_files")
                conn.execute("DELETE FROM index_meta")
                self._vec_cache.clear()
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
                CREATE TABLE IF NOT EXISTS parent_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    root_key TEXT NOT NULL,
                    path TEXT NOT NULL,
                    parent_index INTEGER NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    section TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    text TEXT NOT NULL,
                    table_header TEXT NOT NULL,
                    context_summary TEXT NOT NULL DEFAULT '',
                    until_date TEXT NOT NULL,
                    is_archive INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_parent_root_path_idx ON parent_chunks(root_key, path, parent_index)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_parent_content_hash ON parent_chunks(content_hash)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS child_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_id INTEGER NOT NULL,
                    root_key TEXT NOT NULL,
                    path TEXT NOT NULL,
                    child_index INTEGER NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    section TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    embed_hash TEXT NOT NULL,
                    text TEXT NOT NULL,
                    until_date TEXT NOT NULL,
                    is_archive INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_child_parent_id ON child_chunks(parent_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_child_root_path ON child_chunks(root_key, path)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_child_embed_hash ON child_chunks(embed_hash)"
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS child_chunks_fts USING fts5(
                    path_tokens,
                    section_tokens,
                    context_tokens,
                    body_tokens,
                    tokenize='unicode61 tokenchars ''_'''
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunk_contexts (
                    content_hash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (content_hash, model)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    content_hash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dims INTEGER NOT NULL,
                    vector_blob BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (content_hash, model, dims)
                )
                """
            )
            if not existing_meta:
                conn.executemany(
                    "INSERT OR REPLACE INTO index_meta (key, value) VALUES (?, ?)",
                    list(expected_meta.items()),
                )
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    # 读取指定 root_key 已落盘的文件指纹表
    def _load_fingerprints(self, root_key: str) -> dict[str, tuple[int, int, int, int]]:
        return {
            str(path): (int(mtime_ns), int(size), int(ino), int(is_arch))
            for path, mtime_ns, size, ino, is_arch in self.conn.execute(
                "SELECT path, mtime_ns, size, ino, is_archive FROM indexed_files WHERE root_key = ?",
                (root_key,),
            ).fetchall()
        }

    # 按文件指纹 (mtime_ns, size, ino) 增量同步目标根目录：无变更快路径零写锁，有变更在 BEGIN IMMEDIATE 中原子提交
    def sync(
        self,
        root_key: str,
        searchable: list[tuple[Path, str, bool]],
        *,
        is_archive_root: bool,
    ) -> None:
        archive_flag = 1 if is_archive_root else 0
        scanned: list[tuple[Path, str, bool, tuple[int, int, int, int]]] = []
        current_paths: set[str] = set()
        for resolved, display_path, is_memory in searchable:
            try:
                st = resolved.stat()
            except OSError:
                continue
            fp = (st.st_mtime_ns, st.st_size, st.st_ino, archive_flag)
            current_paths.add(display_path)
            scanned.append((resolved, display_path, is_memory, fp))

        # 无变更快路径：不申请写锁，未修改文件零重读
        existing_fps = self._load_fingerprints(root_key)
        needs_write = any(p not in current_paths for p in existing_fps) or any(
            existing_fps.get(display_path) != fp for _, display_path, _, fp in scanned
        )
        if not needs_write:
            return

        # 写路径持 BEGIN IMMEDIATE 排他写锁后重新核对指纹，彻底消除跨进程删插 FTS5 孤儿行竞态
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            locked_fps = self._load_fingerprints(root_key)
            removed_paths = [p for p in locked_fps if p not in current_paths]
            to_reindex = [
                (resolved, display_path, is_memory, fp)
                for resolved, display_path, is_memory, fp in scanned
                if locked_fps.get(display_path) != fp
            ]
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
            self.conn.execute("COMMIT")
        except BaseException:
            if self.conn.in_transaction:
                self.conn.execute("ROLLBACK")
            raise

    # 按指纹增量同步单个文件（供 enrich_contexts 在每文件准备期消除并发修改与损坏重建后的失步；指纹未变零写锁）
    def sync_single_file(
        self,
        root_key: str,
        display_path: str,
        raw_text: str,
        *,
        fp: tuple[int, int, int, int],
        track_expiry: bool,
    ) -> None:
        row = self.conn.execute(
            "SELECT mtime_ns, size, ino, is_archive FROM indexed_files WHERE root_key = ? AND path = ?",
            (root_key, display_path),
        ).fetchone()
        if row is not None and (int(row[0]), int(row[1]), int(row[2]), int(row[3])) == fp:
            return
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            locked_row = self.conn.execute(
                "SELECT mtime_ns, size, ino, is_archive FROM indexed_files WHERE root_key = ? AND path = ?",
                (root_key, display_path),
            ).fetchone()
            if locked_row is None or (
                int(locked_row[0]),
                int(locked_row[1]),
                int(locked_row[2]),
                int(locked_row[3]),
            ) != fp:
                self._delete_path(root_key, display_path)
                self._index_file(
                    root_key,
                    display_path,
                    raw_text,
                    track_expiry=track_expiry,
                    is_archive=fp[3],
                )
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO indexed_files
                        (root_key, path, mtime_ns, size, ino, is_archive)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (root_key, display_path, fp[0], fp[1], fp[2], fp[3]),
                )
            self.conn.execute("COMMIT")
        except BaseException:
            if self.conn.in_transaction:
                self.conn.execute("ROLLBACK")
            raise

    # 删除指定根目录与文件路径对应的全部父子分块、FTS5 倒排条目及文件指纹记录（必须在活跃写事务内调用）
    def _delete_path(self, root_key: str, display_path: str) -> None:
        child_rows = self.conn.execute(
            "SELECT id FROM child_chunks WHERE root_key = ? AND path = ?",
            (root_key, display_path),
        ).fetchall()
        if child_rows:
            ids = [(r[0],) for r in child_rows]
            self.conn.executemany("DELETE FROM child_chunks_fts WHERE rowid = ?", ids)
            self.conn.execute(
                "DELETE FROM child_chunks WHERE root_key = ? AND path = ?",
                (root_key, display_path),
            )
        self.conn.execute(
            "DELETE FROM parent_chunks WHERE root_key = ? AND path = ?",
            (root_key, display_path),
        )
        self.conn.execute(
            "DELETE FROM indexed_files WHERE root_key = ? AND path = ?",
            (root_key, display_path),
        )

    # 按父块内容哈希从 chunk_contexts 查找已缓存的离线背景摘要（优先匹配当前模型，否则取最新缓存）
    def _lookup_cached_context(self, content_hash: str) -> str:
        row = self.conn.execute(
            """
            SELECT summary
            FROM chunk_contexts
            WHERE content_hash = ?
            ORDER BY CASE WHEN model = ? THEN 0 ELSE 1 END, created_at DESC
            LIMIT 1
            """,
            (content_hash, self.context_model),
        ).fetchone()
        return str(row[0]) if row is not None else ""

    # 对单篇文档执行两层父子分块并写入 parent_chunks、child_chunks（含 embed_hash）与 child_chunks_fts
    def _index_file(
        self,
        root_key: str,
        display_path: str,
        text: str,
        *,
        track_expiry: bool,
        is_archive: int,
    ) -> None:
        parents, children = chunk_markdown_hierarchy(
            display_path,
            text,
            parent_max_chars=self.parent_max_chars,
            child_max_chars=self.child_max_chars,
            child_overlap_chars=self.child_overlap_chars,
            track_expiry=track_expiry,
        )
        if not parents:
            return

        path_tok = " ".join(tokenize_text(display_path, for_query=False))
        parent_id_by_idx: dict[int, int] = {}
        parent_summary_by_idx: dict[int, str] = {}
        parent_ctx_tok_by_idx: dict[int, str] = {}

        for parent in parents:
            summary = "" if parent.until_date else self._lookup_cached_context(parent.content_hash)
            cur = self.conn.execute(
                """
                INSERT INTO parent_chunks (
                    root_key, path, parent_index, start_line, end_line,
                    section, kind, content_hash, text, table_header,
                    context_summary, until_date, is_archive
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    root_key,
                    parent.path,
                    parent.parent_index,
                    parent.start_line,
                    parent.end_line,
                    parent.section,
                    parent.kind,
                    parent.content_hash,
                    parent.text,
                    parent.table_header,
                    summary,
                    parent.until_date,
                    is_archive,
                ),
            )
            parent_id = cur.lastrowid
            if parent_id is None:
                raise RuntimeError("写入 parent_chunks 失败，未返回有效主键。")
            parent_id_by_idx[parent.parent_index] = int(parent_id)
            parent_summary_by_idx[parent.parent_index] = summary
            parent_ctx_tok_by_idx[parent.parent_index] = (
                " ".join(tokenize_text(summary, for_query=False)) if summary else ""
            )

        for child in children:
            parent_id = parent_id_by_idx[child.parent_index]
            summary = parent_summary_by_idx.get(child.parent_index, "")
            _, embed_hash = _child_embed_payload(child.path, child.section, summary, child.text)
            sec_tok = " ".join(tokenize_text(child.section, for_query=False))
            ctx_tok = parent_ctx_tok_by_idx.get(child.parent_index, "")
            body_tok = " ".join(tokenize_text(child.text, for_query=False))

            cur = self.conn.execute(
                """
                INSERT INTO child_chunks (
                    parent_id, root_key, path, child_index, start_line, end_line,
                    section, content_hash, embed_hash, text, until_date, is_archive
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    parent_id,
                    root_key,
                    child.path,
                    child.child_index,
                    child.start_line,
                    child.end_line,
                    child.section,
                    child.content_hash,
                    embed_hash,
                    child.text,
                    child.until_date,
                    is_archive,
                ),
            )
            child_rowid = cur.lastrowid
            self.conn.execute(
                """
                INSERT INTO child_chunks_fts (rowid, path_tokens, section_tokens, context_tokens, body_tokens)
                VALUES (?, ?, ?, ?, ?)
                """,
                (child_rowid, path_tok, sec_tok, ctx_tok, body_tok),
            )

    # 原子落盘单条背景摘要至 chunk_contexts，并同步刷新对应父块及其全部子块的 FTS5 context_tokens 与 embed_hash
    def upsert_context_summary(
        self,
        *,
        content_hash: str,
        model_name: str,
        summary: str,
        write_cache: bool = True,
    ) -> None:
        ctx_tok = " ".join(tokenize_text(summary, for_query=False)) if summary else ""
        created_at = datetime.now(UTC).isoformat()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            if write_cache:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO chunk_contexts (content_hash, model, summary, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (content_hash, model_name, summary, created_at),
                )
            self.conn.execute(
                "UPDATE parent_chunks SET context_summary = ? WHERE content_hash = ? AND until_date = ''",
                (summary, content_hash),
            )
            child_rows = self.conn.execute(
                """
                SELECT c.id, c.path, c.section, c.text
                FROM child_chunks AS c
                JOIN parent_chunks AS p ON p.id = c.parent_id
                WHERE p.content_hash = ? AND p.until_date = ''
                """,
                (content_hash,),
            ).fetchall()
            if child_rows:
                fts_updates: list[tuple[str, int]] = []
                hash_updates: list[tuple[str, int]] = []
                for cid, c_path, c_sec, c_text in child_rows:
                    cid_int = int(cid)
                    _, new_embed_hash = _child_embed_payload(str(c_path), str(c_sec), summary, str(c_text))
                    fts_updates.append((ctx_tok, cid_int))
                    hash_updates.append((new_embed_hash, cid_int))
                self.conn.executemany(
                    "UPDATE child_chunks_fts SET context_tokens = ? WHERE rowid = ?",
                    fts_updates,
                )
                self.conn.executemany(
                    "UPDATE child_chunks SET embed_hash = ? WHERE id = ?",
                    hash_updates,
                )
            self.conn.execute("COMMIT")
        except BaseException:
            if self.conn.in_transaction:
                self.conn.execute("ROLLBACK")
            raise

    # 将一批 L2 单位向量原子写入 chunk_embeddings（小端 float32 BLOB）并预热有界内存向量缓存
    def store_embeddings(
        self,
        model: str,
        items: Sequence[tuple[str, tuple[float, ...]]],
    ) -> None:
        if not items:
            return
        created_at = datetime.now(UTC).isoformat()
        packed_rows = [
            (content_hash, model, len(vec), pack_vector(vec), created_at)
            for content_hash, vec in items
        ]
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO chunk_embeddings
                    (content_hash, model, dims, vector_blob, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                packed_rows,
            )
            self.conn.execute("COMMIT")
        except BaseException:
            if self.conn.in_transaction:
                self.conn.execute("ROLLBACK")
            raise

        for content_hash, vec in items:
            self._put_cached_vector(content_hash, model, vec)

    # 将已解包向量写入有界内存缓存（按 (content_hash, model, dims) 三元组隔离不同维度），超限时按 FIFO 驱逐
    def _put_cached_vector(self, content_hash: str, model: str, vec: tuple[float, ...]) -> None:
        key = (content_hash, model, len(vec))
        self._vec_cache[key] = vec
        while len(self._vec_cache) > _MAX_VECTOR_CACHE_ITEMS:
            oldest = next(iter(self._vec_cache))
            self._vec_cache.pop(oldest, None)

    # 读取指定 (embed_hash, model, dims) 的解包浮点向量（优先命中内存缓存，未命中则校验类型与长度解码 BLOB 并回填）
    def get_cached_vector(
        self,
        embed_hash: str,
        model: str,
        dims: int,
        blob: object,
    ) -> tuple[float, ...] | None:
        key = (embed_hash, model, dims)
        cached = self._vec_cache.get(key)
        if cached is not None:
            return cached
        try:
            vec = unpack_vector(blob, dims)
        except ValueError:
            return None
        self._put_cached_vector(embed_hash, model, vec)
        return vec

    # 读取指定文档的全部常驻父块（排除过期条目）及其当前 context_summary，供离线摘要器增量补全
    def load_enrichable_parents(
        self,
        root_key: str,
        display_path: str,
    ) -> list[tuple[ParentChunk, str]]:
        rows = self.conn.execute(
            """
            SELECT
                parent_index, start_line, end_line, section, kind,
                content_hash, text, table_header, context_summary, until_date
            FROM parent_chunks
            WHERE root_key = ? AND path = ? AND until_date = '' AND kind != 'expiring'
            ORDER BY parent_index
            """,
            (root_key, display_path),
        ).fetchall()
        result: list[tuple[ParentChunk, str]] = []
        for (
            parent_index,
            start_line,
            end_line,
            section,
            kind,
            content_hash,
            body_text,
            table_header,
            context_summary,
            until_date,
        ) in rows:
            header = format_context_header(display_path, str(section))
            tbl_hdr = str(table_header)
            body = str(body_text)
            contextual = f"{header}\n{tbl_hdr}\n{body}" if tbl_hdr else f"{header}\n{body}"
            result.append(
                (
                    ParentChunk(
                        path=display_path,
                        parent_index=int(parent_index),
                        start_line=int(start_line),
                        end_line=int(end_line),
                        section=str(section),
                        kind=str(kind),
                        text=body,
                        contextual_text=contextual,
                        content_hash=str(content_hash),
                        until_date=str(until_date),
                        table_header=tbl_hdr,
                    ),
                    str(context_summary),
                )
            )
        return result

    # 关闭底层 SQLite 连接并清空内存向量缓存
    def close(self) -> None:
        self._vec_cache.clear()
        self.conn.close()


# 持久化两层父子混合 RAG 检索插件配置项（从 config/tools.json 的 retrieval 段读取）
@dataclass(frozen=True)
class RAGConfig:
    min_term_coverage: float = 0.22
    min_bm25_score: float = 0.1
    max_hits_per_file: int = 2
    parent_max_chars: int = DEFAULT_PARENT_MAX_CHARS
    child_max_chars: int = DEFAULT_CHILD_MAX_CHARS
    child_overlap_chars: int = DEFAULT_CHILD_OVERLAP_CHARS
    index_path: str | None = None
    context_model: str = "default"
    min_vector_score: float = 0.45
    rrf_k: int = 60
    bm25_weight: float = 1.0
    vector_weight: float = 1.0

    # 校验覆盖率、BM25 下限、父子分块、Overlap、混合 RRF 权重、抗噪余弦阈值及派生索引相对路径边界
    def __post_init__(self) -> None:
        validate_retrieval_thresholds(self.min_term_coverage, self.min_bm25_score)
        require_positive_int(self.max_hits_per_file, "retrieval.max_hits_per_file")
        require_positive_int(self.parent_max_chars, "retrieval.parent_max_chars")
        require_positive_int(self.child_max_chars, "retrieval.child_max_chars")
        _require_non_negative_int(self.child_overlap_chars, "retrieval.child_overlap_chars")
        if self.child_overlap_chars >= self.child_max_chars:
            raise ValueError("retrieval.child_overlap_chars 必须严格小于 retrieval.child_max_chars。")
        _require_bounded_float(self.min_vector_score, "retrieval.min_vector_score", low=0.0, high=1.0)
        require_positive_int(self.rrf_k, "retrieval.rrf_k")
        w_bm25 = _require_bounded_float(self.bm25_weight, "retrieval.bm25_weight", low=0.0)
        w_vec = _require_bounded_float(self.vector_weight, "retrieval.vector_weight", low=0.0)
        if w_bm25 <= 0.0 and w_vec <= 0.0:
            raise ValueError("retrieval.bm25_weight 与 retrieval.vector_weight 不能同时为 0。")
        if self.index_path is not None:
            if not isinstance(self.index_path, str) or not self.index_path.strip() or "\x00" in self.index_path:
                raise ValueError("retrieval.index_path 必须是非空相对路径字符串或省略。")
            rel = Path(self.index_path.strip())
            if not rel.parts or rel.is_absolute() or ".." in rel.parts:
                raise ValueError("retrieval.index_path 必须是数据目录内的相对路径，不能为绝对路径或包含 ..。")
        if not isinstance(self.context_model, str) or not self.context_model.strip() or "\x00" in self.context_model:
            raise ValueError("retrieval.context_model 必须是非空字符串。")

    # 从配置字典构造 RAGConfig 并执行严格类型与数值校验（拒绝布尔值与非整型浮点数）
    @classmethod
    def from_dict(cls, raw: Mapping[str, object] | None) -> "RAGConfig":
        cfg = raw or {}
        coverage, min_bm25 = parse_retrieval_thresholds(cfg)

        # 解析并校验可选整数配置项，显式拒绝布尔值与浮点数隐式截断
        def _parse_int(key: str, default: int) -> int:
            val = cfg.get(key, default)
            if val is None:
                return default
            if isinstance(val, bool):
                raise ValueError(f"retrieval.{key} 必须是整数。")
            if isinstance(val, int):
                return val
            if isinstance(val, str):
                try:
                    return int(val.strip())
                except ValueError as error:
                    raise ValueError(f"retrieval.{key} 必须是整数。") from error
            raise ValueError(f"retrieval.{key} 必须是整数，不支持浮点或非整型值。")

        # 解析并校验可选有限浮点配置项，显式拒绝布尔值与非数值
        def _parse_float(key: str, default: float) -> float:
            val = cfg.get(key, default)
            if val is None:
                return default
            if isinstance(val, bool):
                raise ValueError(f"retrieval.{key} 必须是有限数字。")
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                try:
                    return float(val.strip())
                except ValueError as error:
                    raise ValueError(f"retrieval.{key} 必须是有限数字。") from error
            raise ValueError(f"retrieval.{key} 必须是有限数字。")

        per_file = _parse_int("max_hits_per_file", 2)
        parent_chars = _parse_int("parent_max_chars", DEFAULT_PARENT_MAX_CHARS)
        child_chars = _parse_int("child_max_chars", DEFAULT_CHILD_MAX_CHARS)
        overlap_chars = _parse_int("child_overlap_chars", DEFAULT_CHILD_OVERLAP_CHARS)
        min_vec_score = _parse_float("min_vector_score", 0.45)
        rrf_k = _parse_int("rrf_k", 60)
        bm25_w = _parse_float("bm25_weight", 1.0)
        vec_w = _parse_float("vector_weight", 1.0)

        raw_index_path = cfg.get("index_path")
        if raw_index_path is not None and not isinstance(raw_index_path, str):
            raise TypeError("retrieval.index_path 必须是字符串。")
        index_path = raw_index_path.strip() if isinstance(raw_index_path, str) and raw_index_path.strip() else None

        raw_ctx_model = cfg.get("context_model")
        if raw_ctx_model is not None and not isinstance(raw_ctx_model, str):
            raise TypeError("retrieval.context_model 必须是字符串。")
        context_model = (
            raw_ctx_model.strip() if isinstance(raw_ctx_model, str) and raw_ctx_model.strip() else "default"
        )

        return cls(
            min_term_coverage=coverage,
            min_bm25_score=min_bm25,
            max_hits_per_file=per_file,
            parent_max_chars=parent_chars,
            child_max_chars=child_chars,
            child_overlap_chars=overlap_chars,
            index_path=index_path,
            context_model=context_model,
            min_vector_score=min_vec_score,
            rrf_k=rrf_k,
            bm25_weight=bm25_w,
            vector_weight=vec_w,
        )


# 持久化两层父子混合 RAG 检索引擎（实现 RetrieverProtocol，线程安全，未配 embedder 时零开销走纯 BM25，配置后自动激活 BM25+Dense RRF）
class RAGRetriever:
    name = RETRIEVER_NAME

    # 初始化检索配置、可选向量客户端、线程锁与按数据根目录隔离的派生 SQLite 索引句柄缓存
    def __init__(
        self,
        config: RAGConfig | None = None,
        *,
        embedder: EmbeddingInvoker | None = None,
    ) -> None:
        self.config = config or RAGConfig()
        self._embedder: EmbeddingInvoker | None = embedder
        self._lock = threading.RLock()
        self._indices: dict[Path, _DerivedRAGIndex] = {}

    # 线程安全读取当前绑定的可选向量嵌入客户端
    @property
    def embedder(self) -> EmbeddingInvoker | None:
        with self._lock:
            return self._embedder

    # 线程安全热切换可选向量嵌入客户端（传入 None 即平滑退回纯词面 BM25 模式）
    def set_embedder(self, embedder: EmbeddingInvoker | None) -> None:
        with self._lock:
            self._embedder = embedder

    # 解析当前 MemoryStore 对应的派生索引路径：只读模式恒返回 None 零写盘，读写模式严格收口于 data_root 内
    def _resolve_db_path(self, memory: "MemoryStore") -> Path | None:
        if memory.read_only:
            return None
        base = memory.data_root
        rel = Path(self.config.index_path.strip()) if self.config.index_path is not None else Path("state") / "rag_index.db"
        db_path = base / rel
        return _assert_safe_index_path(db_path, data_root=base)

    # 获取或懒初始化指定派生库连接；若磁盘派生库被删除、换 Inode 或变为符号链接则立即重新校验并重建
    def _get_index(self, memory: "MemoryStore") -> tuple[Path, _DerivedRAGIndex]:
        db_path = self._resolve_db_path(memory)
        cache_key = db_path if db_path is not None else memory.data_root
        existing = self._indices.get(cache_key)
        if existing is not None:
            if existing.db_path is not None:
                cur_id = _file_identity(existing.db_path)
                if cur_id is None or cur_id != existing.file_identity:
                    try:
                        existing.close()
                    finally:
                        self._indices.pop(cache_key, None)
                else:
                    return cache_key, existing
            else:
                return cache_key, existing
        created = _DerivedRAGIndex(
            db_path,
            data_root=memory.data_root if db_path is not None else None,
            parent_max_chars=self.config.parent_max_chars,
            child_max_chars=self.config.child_max_chars,
            child_overlap_chars=self.config.child_overlap_chars,
            context_model=self.config.context_model,
        )
        self._indices[cache_key] = created
        return cache_key, created

    # 在持锁状态下执行依赖派生库的操作，遇到确认物理损坏的磁盘库时自动清理重建并安全重试一次
    def _with_index_recovery(
        self,
        memory: "MemoryStore",
        operation: Callable[[_DerivedRAGIndex], _T],
    ) -> _T:
        with self._lock:
            for attempt in range(2):
                cache_key, index = self._get_index(memory)
                try:
                    return operation(index)
                except sqlite3.DatabaseError as error:
                    if (
                        attempt == 0
                        and index.db_path is not None
                        and index.data_root is not None
                        and _is_corruption_error(error)
                    ):
                        try:
                            index.close()
                        finally:
                            self._indices.pop(cache_key, None)
                        _unlink_sqlite_files(index.db_path, data_root=index.data_root)
                        continue
                    raise
        raise RuntimeError("派生索引操作异常终止。")

    # 将命中的完整父块原文渲染为带可选背景摘要、续片表头与精确行号前缀的片段
    @staticmethod
    def _format_parent_snippet(
        text: str,
        start_line: int,
        *,
        table_header: str = "",
        context_summary: str = "",
    ) -> str:
        prefix_lines: list[str] = []
        if context_summary:
            prefix_lines.append(f"  [背景: {context_summary}]")
        if table_header:
            for hdr_line in table_header.split("\n"):
                prefix_lines.append(f"  (表头) {hdr_line}")
        body_lines = [
            f"  {n}: {line}"
            for n, line in enumerate(text.split("\n"), start=start_line)
        ]
        rendered = "\n".join(prefix_lines + body_lines)
        if len(rendered) <= _MAX_SNIPPET_CHARS:
            return rendered
        cut = rendered[:_MAX_SNIPPET_CHARS].rsplit("\n", 1)[0]
        return f"{cut}\n  …（父块内容较长，完整内容请调用 read_markdown 按行查看）"

    # 执行词面 BM25 子块检索并按 parent_id 聚合去重（取最高子块得分）
    def _query_bm25_parents(
        self,
        index: _DerivedRAGIndex,
        root_key: str,
        *,
        searching_archive: bool,
        query_terms: list[str],
        ref_day_iso: str,
        limit: int,
    ) -> dict[int, _ParentHitPayload]:
        fts_expr = " OR ".join(f'"{term}"' for term in query_terms)
        query_set = set(query_terms)
        min_matched = 2 if len(query_set) >= 4 else 1
        candidate_limit = max(limit * 20, 100)

        rows = index.conn.execute(
            """
            SELECT
                p.id AS parent_id,
                p.path,
                p.start_line,
                p.end_line,
                p.section,
                p.text AS parent_text,
                p.table_header,
                p.context_summary,
                -bm25(child_chunks_fts, 0.4, 1.5, 1.8, 3.0) AS bm25_score,
                f.path_tokens,
                f.section_tokens,
                f.context_tokens,
                f.body_tokens
            FROM child_chunks_fts AS f
            JOIN child_chunks AS c ON c.id = f.rowid
            JOIN parent_chunks AS p ON p.id = c.parent_id
            WHERE child_chunks_fts MATCH ?
              AND c.root_key = ?
              AND (? = 1 OR c.is_archive = 0)
              AND (c.until_date = '' OR c.until_date >= ?)
            ORDER BY bm25(child_chunks_fts, 0.4, 1.5, 1.8, 3.0)
            LIMIT ?
            """,
            (fts_expr, root_key, 1 if searching_archive else 0, ref_day_iso, candidate_limit),
        ).fetchall()

        total_children = int(
            index.conn.execute(
                "SELECT COUNT(*) FROM child_chunks WHERE root_key = ?",
                (root_key,),
            ).fetchone()[0]
        )

        best_by_parent: dict[int, _ParentHitPayload] = {}
        for (
            parent_id,
            path,
            start_line,
            end_line,
            section,
            parent_text,
            table_header,
            context_summary,
            bm25_score,
            p_tok,
            s_tok,
            c_tok,
            b_tok,
        ) in rows:
            bm25_val = max(0.0, float(bm25_score))
            all_terms = set(f"{p_tok} {s_tok} {c_tok} {b_tok}".split())
            matched_all = query_set & all_terms
            if len(matched_all) < min_matched:
                continue
            coverage = len(matched_all) / len(query_set)
            high_coverage = coverage >= 0.6
            if total_children >= 10 and bm25_val < self.config.min_bm25_score and not high_coverage:
                continue
            high_conf_bm25 = len(matched_all) >= 3 and bm25_val >= 10.0
            if coverage < self.config.min_term_coverage and not high_conf_bm25:
                continue

            body_and_ctx = set(f"{b_tok} {c_tok}".split())
            body_ratio = len(query_set & body_and_ctx) / len(query_set)
            base_score = max(bm25_val, self.config.min_bm25_score, 0.1)
            final_score = round(base_score * (1.0 + 0.6 * coverage + 0.4 * body_ratio), 4)

            pid = int(parent_id)
            prev = best_by_parent.get(pid)
            if prev is None or final_score > prev[0]:
                best_by_parent[pid] = (
                    final_score,
                    str(path),
                    int(start_line),
                    int(end_line),
                    str(section),
                    str(parent_text),
                    str(table_header),
                    str(context_summary),
                )
        return best_by_parent

    # 执行稠密向量余弦相似度检索（math.sumprod），过 min_vector_score 抗噪门控并按 parent_id 聚合取最高相似度
    def _query_dense_parents(
        self,
        index: _DerivedRAGIndex,
        root_key: str,
        *,
        searching_archive: bool,
        query_vec: tuple[float, ...],
        embedder_model: str,
        ref_day_iso: str,
    ) -> dict[int, _ParentHitPayload]:
        q_dims = len(query_vec)
        rows = index.conn.execute(
            """
            SELECT
                p.id AS parent_id,
                p.path,
                p.start_line,
                p.end_line,
                p.section,
                p.text AS parent_text,
                p.table_header,
                p.context_summary,
                c.embed_hash,
                e.vector_blob
            FROM child_chunks AS c
            JOIN parent_chunks AS p ON p.id = c.parent_id
            JOIN chunk_embeddings AS e
              ON e.content_hash = c.embed_hash AND e.model = ? AND e.dims = ?
            WHERE c.root_key = ?
              AND (? = 1 OR c.is_archive = 0)
              AND (c.until_date = '' OR c.until_date >= ?)
            """,
            (embedder_model, q_dims, root_key, 1 if searching_archive else 0, ref_day_iso),
        ).fetchall()

        min_sim = self.config.min_vector_score
        best_by_parent: dict[int, _ParentHitPayload] = {}
        for (
            parent_id,
            path,
            start_line,
            end_line,
            section,
            parent_text,
            table_header,
            context_summary,
            embed_hash,
            vector_blob,
        ) in rows:
            vec = index.get_cached_vector(str(embed_hash), embedder_model, q_dims, vector_blob)
            if vec is None:
                continue
            sim = cosine_similarity(query_vec, vec)
            if sim < min_sim:
                continue
            pid = int(parent_id)
            prev = best_by_parent.get(pid)
            if prev is None or sim > prev[0]:
                best_by_parent[pid] = (
                    round(sim, 6),
                    str(path),
                    int(start_line),
                    int(end_line),
                    str(section),
                    str(parent_text),
                    str(table_header),
                    str(context_summary),
                )
        return best_by_parent

    # 将父级候选按行区间去重与 max_hits_per_file 多样性重排后渲染为 SearchHit 列表
    def _finalize_hits(self, ranked: list[_ParentHitPayload], *, limit: int) -> list[SearchHit]:
        deduped: list[_ParentHitPayload] = []
        used_spans_by_file: dict[str, list[tuple[int, int]]] = {}
        for item in ranked:
            _, path, s_ln, e_ln, _, _, _, _ = item
            prev_spans = used_spans_by_file.setdefault(path, [])
            if any(s_ln <= p_end and e_ln >= p_start for p_start, p_end in prev_spans):
                continue
            prev_spans.append((s_ln, e_ln))
            deduped.append(item)

        primary_picks: list[_ParentHitPayload] = []
        overflow: list[_ParentHitPayload] = []
        per_file_counts: dict[str, int] = {}
        cap = self.config.max_hits_per_file if limit >= 4 else limit
        for item in deduped:
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
                snippet=self._format_parent_snippet(
                    parent_text,
                    start_line,
                    table_header=table_header,
                    context_summary=context_summary,
                ),
                score=score,
            )
            for score, path, start_line, end_line, section, parent_text, table_header, context_summary in chosen
        ]

    # 在已打开的派生索引上执行增量同步、BM25（及可选 Dense + RRF 融合）、父级去重与多样性重排
    def _sync_and_query(
        self,
        index: _DerivedRAGIndex,
        root_key: str,
        searchable: list[tuple[Path, str, bool]],
        *,
        searching_archive: bool,
        query_terms: list[str],
        ref_day_iso: str,
        limit: int,
        query_vec: tuple[float, ...] | None = None,
        embedder_model: str | None = None,
    ) -> list[SearchHit]:
        # 无论 searchable 是否为空均先 sync，确保目录内所有文件被删后派生库旧行也被彻底清除
        index.sync(root_key, searchable, is_archive_root=searching_archive)
        if not searchable:
            return []

        dense_active = query_vec is not None and bool(embedder_model) and self.config.vector_weight > 0.0
        bm25_by_parent: dict[int, _ParentHitPayload] = {}
        if query_terms and (not dense_active or self.config.bm25_weight > 0.0):
            bm25_by_parent = self._query_bm25_parents(
                index,
                root_key,
                searching_archive=searching_archive,
                query_terms=query_terms,
                ref_day_iso=ref_day_iso,
                limit=limit,
            )

        # 未激活稠密通道时直接走纯净 BM25 排序，零额外开销
        if not dense_active or query_vec is None or not embedder_model:
            ranked = sorted(bm25_by_parent.values(), key=lambda item: (-item[0], item[1], item[2], item[3]))
            return self._finalize_hits(ranked, limit=limit)

        dense_by_parent = self._query_dense_parents(
            index,
            root_key,
            searching_archive=searching_archive,
            query_vec=query_vec,
            embedder_model=embedder_model,
            ref_day_iso=ref_day_iso,
        )
        if not bm25_by_parent and not dense_by_parent:
            return []

        # 加权 Reciprocal Rank Fusion (RRF)：双路同时命中的父块自然排在单路命中之前
        bm25_order = sorted(
            bm25_by_parent.items(),
            key=lambda kv: (-kv[1][0], kv[1][1], kv[1][2], kv[1][3]),
        )
        dense_order = sorted(
            dense_by_parent.items(),
            key=lambda kv: (-kv[1][0], kv[1][1], kv[1][2], kv[1][3]),
        )
        bm25_rank = {pid: rank for rank, (pid, _) in enumerate(bm25_order, start=1)}
        dense_rank = {pid: rank for rank, (pid, _) in enumerate(dense_order, start=1)}

        k = self.config.rrf_k
        w_bm25 = self.config.bm25_weight
        w_vec = self.config.vector_weight
        fused_entries: list[tuple[float, float, _ParentHitPayload]] = []
        for pid in set(bm25_by_parent) | set(dense_by_parent):
            rrf_score = 0.0
            tie_signal = 0.0
            if pid in bm25_rank and w_bm25 > 0.0:
                rrf_score += w_bm25 / (k + bm25_rank[pid])
                tie_signal = max(tie_signal, bm25_by_parent[pid][0])
            if pid in dense_rank and w_vec > 0.0:
                rrf_score += w_vec / (k + dense_rank[pid])
                tie_signal = max(tie_signal, dense_by_parent[pid][0])
            if rrf_score <= 0.0:
                continue
            meta = bm25_by_parent.get(pid) or dense_by_parent[pid]
            fused_payload: _ParentHitPayload = (
                round(rrf_score, 6),
                meta[1],
                meta[2],
                meta[3],
                meta[4],
                meta[5],
                meta[6],
                meta[7],
            )
            fused_entries.append((fused_payload[0], tie_signal, fused_payload))

        fused_entries.sort(key=lambda item: (-item[0], -item[1], item[2][1], item[2][2], item[2][3]))
        return self._finalize_hits([entry[2] for entry in fused_entries], limit=limit)

    # 在锁外按批次为活跃子块增量补齐目标维度向量缓存（严守已过期不外发、按 (hash, model, dims) 精确对齐并支持断点续跑）
    def _ensure_active_embeddings(
        self,
        memory: "MemoryStore",
        root_key: str,
        searchable: list[tuple[Path, str, bool]],
        *,
        searching_archive: bool,
        embedder: EmbeddingInvoker,
        ref_day_iso: str,
        batch_size: int,
        expected_dims: int | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> dict[str, int]:
        model_id = _clean_embedder_model(embedder)
        if not model_id:
            raise ValueError("embedder.model_id 不能为空。")

        target_dims = expected_dims if expected_dims is not None else _embedder_target_dims(embedder)
        embedded = 0
        final_total = 0
        final_cached = 0
        final_dims = target_dims or 0

        # 最多两轮：若未显式声明维度且首轮新批返回的真实维度与历史推断维度不同，自动切至真实维度补齐其余子块
        for _pass_idx in range(2):
            # 持锁同步并一次性提取当前根目录全部活跃（未过期）子块及目标维度的已有向量缓存键集合
            def _collect_missing(
                index: _DerivedRAGIndex,
                *,
                req_dims: int | None = target_dims,
            ) -> tuple[int, int, int | None, list[tuple[str, str]], dict[str, int]]:
                index.sync(root_key, searchable, is_archive_root=searching_archive)
                rows = index.conn.execute(
                    """
                    SELECT c.embed_hash, c.path, c.section, p.context_summary, c.text
                    FROM child_chunks AS c
                    JOIN parent_chunks AS p ON p.id = c.parent_id
                    WHERE c.root_key = ?
                      AND (? = 1 OR c.is_archive = 0)
                      AND (c.until_date = '' OR c.until_date >= ?)
                    ORDER BY c.path, c.id
                    """,
                    (root_key, 1 if searching_archive else 0, ref_day_iso),
                ).fetchall()
                total_children = len(rows)
                if not rows:
                    return 0, 0, req_dims, [], {}

                effective_dims = req_dims
                if effective_dims is None:
                    latest = index.conn.execute(
                        "SELECT dims FROM chunk_embeddings WHERE model = ? ORDER BY created_at DESC LIMIT 1",
                        (model_id,),
                    ).fetchone()
                    if latest is not None:
                        effective_dims = int(latest[0])

                if effective_dims is not None:
                    cached_hashes = {
                        str(r[0])
                        for r in index.conn.execute(
                            """
                            SELECT content_hash
                            FROM chunk_embeddings
                            WHERE model = ?
                              AND dims = ?
                              AND typeof(vector_blob) = 'blob'
                              AND length(vector_blob) = dims * 4
                            """,
                            (model_id, effective_dims),
                        ).fetchall()
                    }
                else:
                    cached_hashes = set()

                missing_unique: list[tuple[str, str]] = []
                missing_counts: dict[str, int] = {}
                cached_cnt = 0
                for embed_hash, c_path, c_sec, ctx_sum, c_text in rows:
                    h_str = str(embed_hash)
                    if h_str in cached_hashes:
                        cached_cnt += 1
                    else:
                        if h_str not in missing_counts:
                            embed_text, _ = _child_embed_payload(str(c_path), str(c_sec), str(ctx_sum), str(c_text))
                            missing_unique.append((h_str, embed_text))
                        missing_counts[h_str] = missing_counts.get(h_str, 0) + 1
                return total_children, cached_cnt, effective_dims, missing_unique, missing_counts

            total_children, cached_cnt, inferred_dims, missing_unique, missing_counts = self._with_index_recovery(
                memory,
                _collect_missing,
            )
            final_total = total_children
            final_cached = cached_cnt
            if inferred_dims is not None and final_dims == 0:
                final_dims = inferred_dims

            if not missing_unique:
                break

            dims_changed = False
            observed_dims: int | None = target_dims
            # 锁外按 batch_size 分批请求 Embedding API，统一执行单位归一化与维度一致性校验并原子落盘
            for start in range(0, len(missing_unique), batch_size):
                if abort_signal is not None and abort_signal.aborted:
                    raise StreamAbortedError(abort_signal.reason or "批量向量化已取消")
                batch = missing_unique[start : start + batch_size]
                raw_vectors = embedder.embed([txt for _, txt in batch], abort_signal=abort_signal)
                if len(raw_vectors) != len(batch):
                    raise ValueError(f"向量化返回数量不匹配：期望 {len(batch)}，实际 {len(raw_vectors)}。")
                norm_vectors = [normalize_vector(vec) for vec in raw_vectors]
                for vec in norm_vectors:
                    if observed_dims is None:
                        observed_dims = len(vec)
                    elif len(vec) != observed_dims:
                        raise ValueError(
                            f"向量化返回维度不一致：期望 {observed_dims}，实际 {len(vec)}。"
                        )
                if observed_dims is not None:
                    final_dims = observed_dims
                pairs = [(h, vec) for (h, _), vec in zip(batch, norm_vectors)]

                # 写回前确保若锁外期间磁盘派生库被外部重建，分块表与向量缓存可一并恢复同步
                def _persist_batch(
                    idx: _DerivedRAGIndex,
                    *,
                    items: list[tuple[str, tuple[float, ...]]] = pairs,
                ) -> None:
                    idx.sync(root_key, searchable, is_archive_root=searching_archive)
                    idx.store_embeddings(model_id, items)

                self._with_index_recovery(memory, _persist_batch)
                for h, _ in pairs:
                    embedded += 1
                    occ = missing_counts.get(h, 1)
                    if occ > 1:
                        final_cached += occ - 1

            if (
                target_dims is None
                and inferred_dims is not None
                and observed_dims is not None
                and observed_dims != inferred_dims
                and final_cached > 0
            ):
                target_dims = observed_dims
                dims_changed = True
            if not dims_changed:
                break

        return {
            "files": len(searchable),
            "children_total": final_total,
            "embedded": embedded,
            "cached_hits": final_cached,
            "skipped": 0,
            "dims": final_dims,
        }

    # 对外检索入口：未配 embedder 或 vector_weight=0 时直通纯 BM25；否则先编码 query 获取真实维度再对齐向量缓存并执行 RRF
    def search_hits(
        self,
        memory: "MemoryStore",
        query: str,
        root: str = "data/memory",
        limit: int = 10,
        today: date | None = None,
    ) -> list[SearchHit]:
        clean_query = query.strip() if isinstance(query, str) else ""
        query_terms = list(dict.fromkeys(tokenize_text(clean_query, for_query=True)))
        active_embedder = self.embedder
        active_model = _clean_embedder_model(active_embedder)
        dense_enabled = active_embedder is not None and active_model is not None and self.config.vector_weight > 0.0

        if not query_terms and (not clean_query or not dense_enabled):
            raise ValueError("搜索关键词不能为空。")

        ref_day_iso = (today or today_shanghai()).isoformat()
        target_dir = memory.resolve_path(root, write=False)
        searchable, searching_archive = memory.iter_searchable_files(root)
        root_key = memory.display_path(target_dir)

        query_vec: tuple[float, ...] | None = None
        embedder_model: str | None = None
        if dense_enabled and active_embedder is not None and active_model is not None and searchable:
            try:
                # 先编码 query 获取真实 q_dims，确保维度变更时 _ensure_active_embeddings 精确按 (hash, model, q_dims) 补齐
                q_vecs = active_embedder.embed([clean_query])
                if len(q_vecs) != 1:
                    raise ValueError(f"查询向量返回数量异常：期望 1，实际 {len(q_vecs)}。")
                norm_q = normalize_vector(q_vecs[0])
                q_dims = len(norm_q)
                self._ensure_active_embeddings(
                    memory,
                    root_key,
                    searchable,
                    searching_archive=searching_archive,
                    embedder=active_embedder,
                    ref_day_iso=ref_day_iso,
                    batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
                    expected_dims=q_dims,
                )
                query_vec = norm_q
                embedder_model = active_model
            except StreamAbortedError:
                raise
            except (OSError, RuntimeError, ValueError, TypeError, LLMNotConfigured):
                # 在线检索阶段遇 Embedding 接口异常自动优雅降级为纯 BM25，保障前台对话不中断
                query_vec = None
                embedder_model = None

        if not query_terms and query_vec is None:
            return []

        return self._with_index_recovery(
            memory,
            lambda index: self._sync_and_query(
                index,
                root_key,
                searchable,
                searching_archive=searching_archive,
                query_terms=query_terms,
                ref_day_iso=ref_day_iso,
                limit=limit,
                query_vec=query_vec,
                embedder_model=embedder_model,
            ),
        )

    # 离线批量为目标根目录的活跃子块计算并持久化向量缓存（供 /rag embed 调用，支持 AbortSignal 断点续跑且暴露真实错误）
    def embed_chunks(
        self,
        memory: "MemoryStore",
        *,
        root: str = "data/memory",
        embedder: EmbeddingInvoker | None = None,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        today: date | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> dict[str, int]:
        if memory.read_only:
            raise ValueError("只读模式下不支持持久化写入向量缓存。")
        require_positive_int(batch_size, "batch_size")
        active = embedder if embedder is not None else self.embedder
        if active is None:
            raise ValueError("未配置向量嵌入模型（embedder），请先在 config/models.dsl 中添加 embedder 或设置 EMBEDDING_MODEL。")

        ref_day_iso = (today or today_shanghai()).isoformat()
        target_dir = memory.resolve_path(root, write=False)
        searchable, searching_archive = memory.iter_searchable_files(root)
        root_key = memory.display_path(target_dir)

        return self._ensure_active_embeddings(
            memory,
            root_key,
            searchable,
            searching_archive=searching_archive,
            embedder=active,
            ref_day_iso=ref_day_iso,
            batch_size=batch_size,
            expected_dims=_embedder_target_dims(active),
            abort_signal=abort_signal,
        )

    # 同步目标根目录并返回当前派生索引的详细只读状态（含父子块、摘要与活跃子块目标维度向量覆盖统计，供 /rag status 展示）
    def index_status(
        self,
        memory: "MemoryStore",
        root: str = "data/memory",
        *,
        today: date | None = None,
    ) -> dict[str, object]:
        target_dir = memory.resolve_path(root, write=False)
        searchable, searching_archive = memory.iter_searchable_files(root)
        root_key = memory.display_path(target_dir)
        ref_day_iso = (today or today_shanghai()).isoformat()
        active_emb = self.embedder
        active_model = _clean_embedder_model(active_emb)
        target_dims = _embedder_target_dims(active_emb)

        # 同步后汇总目标根目录下的文件数、父子块数、上下文摘要及活跃子块向量覆盖情况
        def _collect(index: _DerivedRAGIndex) -> dict[str, object]:
            index.sync(root_key, searchable, is_archive_root=searching_archive)
            indexed_files = int(
                index.conn.execute(
                    "SELECT COUNT(*) FROM indexed_files WHERE root_key = ?",
                    (root_key,),
                ).fetchone()[0]
            )
            parent_chunks = int(
                index.conn.execute(
                    "SELECT COUNT(*) FROM parent_chunks WHERE root_key = ?",
                    (root_key,),
                ).fetchone()[0]
            )
            child_chunks = int(
                index.conn.execute(
                    "SELECT COUNT(*) FROM child_chunks WHERE root_key = ?",
                    (root_key,),
                ).fetchone()[0]
            )
            enriched_parents = int(
                index.conn.execute(
                    "SELECT COUNT(*) FROM parent_chunks WHERE root_key = ? AND context_summary != ''",
                    (root_key,),
                ).fetchone()[0]
            )
            cached_contexts = int(
                index.conn.execute("SELECT COUNT(*) FROM chunk_contexts").fetchone()[0]
            )
            cached_embeddings = int(
                index.conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0]
            )
            if active_model is not None:
                if target_dims is not None:
                    embedded_children = int(
                        index.conn.execute(
                            """
                            SELECT COUNT(DISTINCT c.id)
                            FROM child_chunks AS c
                            JOIN chunk_embeddings AS e
                              ON e.content_hash = c.embed_hash AND e.model = ? AND e.dims = ?
                            WHERE c.root_key = ?
                              AND (? = 1 OR c.is_archive = 0)
                              AND (c.until_date = '' OR c.until_date >= ?)
                            """,
                            (active_model, target_dims, root_key, 1 if searching_archive else 0, ref_day_iso),
                        ).fetchone()[0]
                    )
                else:
                    embedded_children = int(
                        index.conn.execute(
                            """
                            SELECT COUNT(DISTINCT c.id)
                            FROM child_chunks AS c
                            JOIN chunk_embeddings AS e
                              ON e.content_hash = c.embed_hash AND e.model = ?
                            WHERE c.root_key = ?
                              AND (? = 1 OR c.is_archive = 0)
                              AND (c.until_date = '' OR c.until_date >= ?)
                            """,
                            (active_model, root_key, 1 if searching_archive else 0, ref_day_iso),
                        ).fetchone()[0]
                    )
            else:
                embedded_children = 0
            if index.db_path is not None and index.data_root is not None:
                rel_db = index.db_path.relative_to(index.data_root).as_posix()
                db_label = f"data/{rel_db}"
            else:
                db_label = ":memory:"
            return {
                "retriever": RETRIEVER_NAME,
                "root": root_key,
                "db_path": db_label,
                "schema_version": _SCHEMA_VERSION,
                "parent_max_chars": self.config.parent_max_chars,
                "child_max_chars": self.config.child_max_chars,
                "child_overlap_chars": self.config.child_overlap_chars,
                "indexed_files": indexed_files,
                "parent_chunks": parent_chunks,
                "child_chunks": child_chunks,
                "enriched_parents": enriched_parents,
                "cached_contexts": cached_contexts,
                "embedder_model": active_model,
                "embedded_children": embedded_children,
                "cached_embeddings": cached_embeddings,
            }

        return self._with_index_recovery(memory, _collect)

    # 离线为目标根目录的常驻父块生成 Prompt-Caching 背景摘要，逐条立即落盘并支持 AbortSignal 断点续跑
    def enrich_contexts(
        self,
        memory: "MemoryStore",
        llm: ModelInvoker,
        *,
        root: str = "data/memory",
        model_name: str = "default",
        min_doc_chars: int = 0,
        abort_signal: AbortSignal | None = None,
    ) -> dict[str, int]:
        if memory.read_only:
            raise ValueError("只读模式下不支持写入离线上下文摘要。")
        if not isinstance(model_name, str) or not model_name.strip() or "\x00" in model_name:
            raise ValueError("model_name 必须是非空字符串。")
        clean_model = model_name.strip()
        min_chars = _require_non_negative_int(min_doc_chars, "min_doc_chars")

        target_dir = memory.resolve_path(root, write=False)
        searchable, searching_archive = memory.iter_searchable_files(root)
        root_key = memory.display_path(target_dir)

        self._with_index_recovery(
            memory,
            lambda index: index.sync(root_key, searchable, is_archive_root=searching_archive),
        )

        stats = {
            "files": 0,
            "parents_total": 0,
            "enriched": 0,
            "cached_hits": 0,
            "skipped": 0,
        }
        archive_flag = 1 if searching_archive else 0

        for resolved, display_path, is_memory in searchable:
            if abort_signal is not None and abort_signal.aborted:
                raise StreamAbortedError(abort_signal.reason or "离线上下文摘要已取消")
            try:
                st_before = resolved.stat()
                raw_text = resolved.read_text(encoding="utf-8")
                st_after = resolved.stat()
                if (st_before.st_mtime_ns, st_before.st_size, st_before.st_ino) != (
                    st_after.st_mtime_ns,
                    st_after.st_size,
                    st_after.st_ino,
                ):
                    raw_text = resolved.read_text(encoding="utf-8")
                    st_after = resolved.stat()
            except (OSError, UnicodeDecodeError):
                continue
            fp = (st_after.st_mtime_ns, st_after.st_size, st_after.st_ino, archive_flag)

            # 在持锁状态下同步当前文件快照（抵御损坏重建与并发修改失步）并读取缺失父块与重复哈希计数
            def _prepare_file(
                index: _DerivedRAGIndex,
                *,
                d_path: str = display_path,
                doc_text: str = raw_text,
                file_fp: tuple[int, int, int, int] = fp,
                mem_flag: bool = is_memory,
            ) -> tuple[list[ParentChunk], dict[str, int], int, int, int]:
                index.sync_single_file(
                    root_key,
                    d_path,
                    doc_text,
                    fp=file_fp,
                    track_expiry=mem_flag,
                )
                parent_rows = index.load_enrichable_parents(root_key, d_path)
                total_parents = len(parent_rows)
                if not parent_rows:
                    return [], {}, 0, 0, 0
                if len(doc_text) < min_chars:
                    return [], {}, 0, total_parents, total_parents
                missing: list[ParentChunk] = []
                missing_counts: dict[str, int] = {}
                cached_cnt = 0
                for parent, current_summary in parent_rows:
                    row = index.conn.execute(
                        "SELECT summary FROM chunk_contexts WHERE content_hash = ? AND model = ?",
                        (parent.content_hash, clean_model),
                    ).fetchone()
                    if row is not None:
                        cached_summary = str(row[0])
                        cached_cnt += 1
                        if current_summary != cached_summary:
                            index.upsert_context_summary(
                                content_hash=parent.content_hash,
                                model_name=clean_model,
                                summary=cached_summary,
                                write_cache=False,
                            )
                    else:
                        if parent.content_hash not in missing_counts:
                            missing.append(parent)
                        missing_counts[parent.content_hash] = missing_counts.get(parent.content_hash, 0) + 1
                return missing, missing_counts, cached_cnt, 0, total_parents

            missing_parents, missing_counts, cached_cnt, skipped_cnt, total_parents = self._with_index_recovery(
                memory,
                _prepare_file,
            )
            stats["files"] += 1
            stats["parents_total"] += total_parents
            stats["cached_hits"] += cached_cnt
            stats["skipped"] += skipped_cnt

            if not missing_parents:
                continue

            # 记忆文件送入 <document> 前剔除全部带 (until: ...) 的过期块，杜绝过期敏感内容外发或泄漏进常驻摘要缓存
            safe_doc_text = strip_expiring_lines(raw_text) if is_memory else raw_text

            # 在锁外流式调用模型生成背景摘要，每产出 1 条立即持锁原子提交至 chunk_contexts 与 FTS5
            produced_hashes: set[str] = set()
            for content_hash, summary in generate_document_contexts(
                display_path,
                safe_doc_text,
                missing_parents,
                llm,
                abort_signal=abort_signal,
            ):
                produced_hashes.add(content_hash)
                self._with_index_recovery(
                    memory,
                    lambda index, h=content_hash, s=summary: index.upsert_context_summary(
                        content_hash=h,
                        model_name=clean_model,
                        summary=s,
                        write_cache=True,
                    ),
                )
                occ = missing_counts.get(content_hash, 1)
                stats["enriched"] += 1
                if occ > 1:
                    stats["cached_hits"] += occ - 1

            # 统计模型返回空白摘要的父块数（保持 parents_total == enriched + cached_hits + skipped 恒等）
            for h, occ in missing_counts.items():
                if h not in produced_hashes:
                    stats["skipped"] += occ

        return stats

    # 安全关闭本实例持有的全部派生 SQLite 连接（单个关闭异常不阻断其余句柄回收）
    def close(self) -> None:
        with self._lock:
            indices = list(self._indices.values())
            self._indices.clear()
        first_error: Exception | None = None
        for index in indices:
            try:
                index.close()
            except Exception as error:  # noqa: BLE001
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


# 检索插件工厂函数：按 config/tools.json 的 retrieval 段装配实现 RetrieverProtocol 的 RAGRetriever
def create_retriever(config: dict[str, object]) -> RAGRetriever:
    return RAGRetriever(RAGConfig.from_dict(config))


# 持久化两层父子混合 RAG 主板插件（外层继承 Plugin ABC，内层向 retrievers 插槽注册 rag 检索引擎并线程安全托管连接回收）
class RAGRetrieverPlugin(Plugin):
    plugin_id = RETRIEVER_NAME

    # 初始化线程锁与活跃检索实例弱引用集合，确保并发创建与 terminate() 均线程安全
    def __init__(self, context: PluginContext) -> None:
        super().__init__(context)
        self._lock = threading.Lock()
        self._instances: weakref.WeakSet[RAGRetriever] = weakref.WeakSet()

    # 统一初始化生命周期：向 self.context.register_retriever 插槽登记 rag 检索引擎工厂
    def initialize(self) -> None:
        # 创建并登记 RAGRetriever 实例，便于插件停用时统一关闭 SQLite 连接
        def _tracked_factory(cfg: dict[str, object]) -> RAGRetriever:
            instance = create_retriever(cfg)
            with self._lock:
                self._instances.add(instance)
            return instance

        self.context.register_retriever(self.plugin_id, _tracked_factory)

    # 统一停用清理生命周期：线程安全关闭本插件创建的所有 RAGRetriever SQLite 派生库连接（单实例异常不阻断其余回收）
    def terminate(self) -> None:
        with self._lock:
            instances = list(self._instances)
        first_error: Exception | None = None
        for instance in instances:
            try:
                instance.close()
            except Exception as error:  # noqa: BLE001
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
