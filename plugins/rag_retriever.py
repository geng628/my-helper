"""持久化两层父子 RAG 检索插件（零第三方依赖）：父子表分离、整句 Overlap、Prompt-Caching 上下文摘要缓存与双语 FTS5 BM25。"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from personal_assistant.agent.plugins import Plugin, PluginContext
from personal_assistant.ai import AbortSignal, ModelInvoker, StreamAbortedError
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
_SCHEMA_VERSION = "2"
_CHUNKER_VERSION = "2"
_MAX_SNIPPET_CHARS = 4000
_SQLITE_SIDECAR_SUFFIXES = ("", "-wal", "-shm", "-journal")
_REQUIRED_TABLES = frozenset(
    {
        "index_meta",
        "indexed_files",
        "parent_chunks",
        "child_chunks",
        "child_chunks_fts",
        "chunk_contexts",
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


# 持久化/内存双用两层父子 SQLite 派生索引容器（与权威源文件物理隔离，chunk_contexts 跨配置重建保留）
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

    # 检查并初始化两层父子表与 chunk_contexts 缓存表（已就绪库走只读快路径零写锁，配置变更重建时保留 chunk_contexts）
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
                "AND name IN ('index_meta', 'indexed_files', 'parent_chunks', 'child_chunks', 'child_chunks_fts', 'chunk_contexts')"
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
                # 配置或版本变更时仅重建派生分块与倒排表，绝不清空已付费生成的 chunk_contexts 缓存表
                conn.execute("DROP TABLE IF EXISTS child_chunks_fts")
                conn.execute("DROP TABLE IF EXISTS child_chunks")
                conn.execute("DROP TABLE IF EXISTS parent_chunks")
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

    # 对单篇文档执行两层父子分块并写入 parent_chunks、child_chunks 与 child_chunks_fts
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
            parent_ctx_tok_by_idx[parent.parent_index] = (
                " ".join(tokenize_text(summary, for_query=False)) if summary else ""
            )

        for child in children:
            parent_id = parent_id_by_idx[child.parent_index]
            sec_tok = " ".join(tokenize_text(child.section, for_query=False))
            ctx_tok = parent_ctx_tok_by_idx.get(child.parent_index, "")
            body_tok = " ".join(tokenize_text(child.text, for_query=False))

            cur = self.conn.execute(
                """
                INSERT INTO child_chunks (
                    parent_id, root_key, path, child_index, start_line, end_line,
                    section, content_hash, text, until_date, is_archive
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    # 原子落盘单条背景摘要至 chunk_contexts 并同步刷新对应父块及其全部子块的 FTS5 context_tokens
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
            child_ids = self.conn.execute(
                """
                SELECT c.id
                FROM child_chunks AS c
                JOIN parent_chunks AS p ON p.id = c.parent_id
                WHERE p.content_hash = ? AND p.until_date = ''
                """,
                (content_hash,),
            ).fetchall()
            if child_ids:
                self.conn.executemany(
                    "UPDATE child_chunks_fts SET context_tokens = ? WHERE rowid = ?",
                    [(ctx_tok, int(row[0])) for row in child_ids],
                )
            self.conn.execute("COMMIT")
        except BaseException:
            if self.conn.in_transaction:
                self.conn.execute("ROLLBACK")
            raise

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

    # 关闭底层 SQLite 连接
    def close(self) -> None:
        self.conn.close()


# 持久化两层父子 RAG 检索插件配置项（从 config/tools.json 的 retrieval 段读取）
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

    # 校验覆盖率、BM25 下限、父子分块字数、Overlap 上界、模型标识及派生索引相对路径边界
    def __post_init__(self) -> None:
        validate_retrieval_thresholds(self.min_term_coverage, self.min_bm25_score)
        require_positive_int(self.max_hits_per_file, "retrieval.max_hits_per_file")
        require_positive_int(self.parent_max_chars, "retrieval.parent_max_chars")
        require_positive_int(self.child_max_chars, "retrieval.child_max_chars")
        _require_non_negative_int(self.child_overlap_chars, "retrieval.child_overlap_chars")
        if self.child_overlap_chars >= self.child_max_chars:
            raise ValueError("retrieval.child_overlap_chars 必须严格小于 retrieval.child_max_chars。")
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

        per_file = _parse_int("max_hits_per_file", 2)
        parent_chars = _parse_int("parent_max_chars", DEFAULT_PARENT_MAX_CHARS)
        child_chars = _parse_int("child_max_chars", DEFAULT_CHILD_MAX_CHARS)
        overlap_chars = _parse_int("child_overlap_chars", DEFAULT_CHILD_OVERLAP_CHARS)

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
        )


# 持久化两层父子 RAG 检索引擎（实现 RetrieverProtocol，线程安全，读写落盘 data/state/rag_index.db，只读零写盘）
class RAGRetriever:
    name = RETRIEVER_NAME

    # 初始化检索配置、线程锁与按数据根目录隔离的派生 SQLite 索引句柄缓存
    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig()
        self._lock = threading.RLock()
        self._indices: dict[Path, _DerivedRAGIndex] = {}

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

    # 在已打开的派生索引上执行增量同步、子块 BM25 检索、覆盖率门控、按 parent_id 聚合去重与多样性重排
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
    ) -> list[SearchHit]:
        # 无论 searchable 是否为空均先 sync，确保目录内所有文件被删后派生库旧行也被彻底清除
        index.sync(root_key, searchable, is_archive_root=searching_archive)
        if not searchable:
            return []

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

        # 按 parent_id 聚合去重：同一父块下多个子块同时命中时取最高子块得分并只返回 1 条完整父块
        best_by_parent: dict[int, tuple[float, str, int, int, str, str, str, str]] = {}
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

        ranked = sorted(best_by_parent.values(), key=lambda item: (-item[0], item[1], item[2], item[3]))

        # 同文件行区间去重（防御性保障同一行区间不重复展示）
        deduped: list[tuple[float, str, int, int, str, str, str, str]] = []
        used_spans_by_file: dict[str, list[tuple[int, int]]] = {}
        for item in ranked:
            _, path, s_ln, e_ln, _, _, _, _ = item
            prev_spans = used_spans_by_file.setdefault(path, [])
            if any(s_ln <= p_end and e_ln >= p_start for p_start, p_end in prev_spans):
                continue
            prev_spans.append((s_ln, e_ln))
            deduped.append(item)

        # 按 max_hits_per_file 做多文件多样性优先挑选，剩余名额从溢出候选中补齐
        primary_picks: list[tuple[float, str, int, int, str, str, str, str]] = []
        overflow: list[tuple[float, str, int, int, str, str, str, str]] = []
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

    # 对外检索入口：使用规范相对 root_key，遇到运行期确认物理损坏的磁盘派生库时自动清理重建并重试一次
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
        root_key = memory.display_path(target_dir)

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
            ),
        )

    # 同步目标根目录并返回当前派生索引的详细只读状态（供 /rag status 展示）
    def index_status(self, memory: "MemoryStore", root: str = "data/memory") -> dict[str, object]:
        target_dir = memory.resolve_path(root, write=False)
        searchable, searching_archive = memory.iter_searchable_files(root)
        root_key = memory.display_path(target_dir)

        # 同步后汇总目标根目录下的文件数、父子块数及上下文摘要覆盖情况
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


# 持久化两层父子 RAG 主板插件（外层继承 Plugin ABC，内层向 retrievers 插槽注册 rag 检索引擎并线程安全托管连接回收）
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
