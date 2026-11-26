"""极简 SQLite 状态存储：维护对话历史 messages 与上下文压缩状态 session_compactions。"""
from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path


# 打开 SQLite 连接并开启 WAL 日志模式与行字典映射，支持安全并发读写
def connect_sqlite(path: Path | str, timeout: float = 5.0) -> sqlite3.Connection:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path_obj, timeout=timeout, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}").close()
    conn.execute("PRAGMA journal_mode=WAL").close()
    return conn


# 幂等初始化 messages 与 session_compactions 表结构及查询索引
def init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages (chat_id, id)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS session_compactions (
            chat_id TEXT PRIMARY KEY,
            summary TEXT NOT NULL,
            compacted_until_id INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


# 线程安全的对话历史与压缩游标存储类
class StateStore:
    # 初始化数据库连接、可重入锁并确保表结构存在
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self.conn = connect_sqlite(self.db_path)
        init_schema(self.conn)

    # 关闭底层 SQLite 连接
    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # 将数据库查询行转换为标准消息字典
    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> dict[str, object]:
        return {
            "id": int(row["id"]),
            "chat_id": str(row["chat_id"]),
            "role": str(row["role"]),
            "content": str(row["content"]),
            "created_at": str(row["created_at"]),
        }

    # 追加一条对话消息（user 或 assistant），返回自增消息编号
    def append_message(self, chat_id: str, role: str, content: str) -> int:
        created_at = datetime.now(UTC).isoformat()
        with self._lock:
            cursor = self.conn.execute(
                "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (chat_id, role, content, created_at),
            )
            self.conn.commit()
            return int(cursor.lastrowid)

    # 在同一事务内原子成对写入本轮 user 与 assistant 消息，返回 (user_msg_id, assistant_msg_id)
    def append_turn(
        self, chat_id: str, user_content: str, assistant_content: str
    ) -> tuple[int, int]:
        created_at = datetime.now(UTC).isoformat()
        with self._lock:
            try:
                user_cursor = self.conn.execute(
                    "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                    (chat_id, "user", user_content, created_at),
                )
                user_id = int(user_cursor.lastrowid)
                assistant_cursor = self.conn.execute(
                    "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                    (chat_id, "assistant", assistant_content, created_at),
                )
                assistant_id = int(assistant_cursor.lastrowid)
                self.conn.commit()
            except BaseException:
                self.conn.rollback()
                raise
            return user_id, assistant_id

    # 返回当前 messages 表的最大自增编号（空表返回 0），供流前锁定历史检索快照边界
    def max_message_id(self) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT COALESCE(MAX(id), 0) AS max_id FROM messages"
            ).fetchone()
        return int(row["max_id"]) if row is not None else 0

    # 按时间正序返回指定会话最近 limit 条对话记录（不区分是否已压缩）
    def recent_messages(self, chat_id: str, limit: int = 12) -> list[dict[str, object]]:
        if limit <= 0:
            return []
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT id, chat_id, role, content, created_at
                FROM messages
                WHERE chat_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (chat_id, limit),
            ).fetchall()
        return [self._row_to_message(row) for row in reversed(rows)]

    # 读取指定会话的上下文压缩状态（前情摘要与已沉淀截止消息编号）
    def get_compaction(self, chat_id: str) -> dict[str, object]:
        with self._lock:
            row = self.conn.execute(
                """
                SELECT summary, compacted_until_id, updated_at
                FROM session_compactions
                WHERE chat_id = ?
                """,
                (chat_id,),
            ).fetchone()
        if row is None:
            return {"summary": "", "compacted_until_id": 0, "updated_at": ""}
        return {
            "summary": str(row["summary"]),
            "compacted_until_id": int(row["compacted_until_id"]),
            "updated_at": str(row["updated_at"]),
        }

    # 原子保存会话前情摘要并单调推进已沉淀游标（不会回退游标，也不删除原始消息）
    def save_compaction(self, chat_id: str, summary: str, compacted_until_id: int) -> None:
        updated_at = datetime.now(UTC).isoformat()
        with self._lock:
            current = self.get_compaction(chat_id)
            if compacted_until_id < int(current["compacted_until_id"]):
                return
            self.conn.execute(
                """
                INSERT INTO session_compactions (chat_id, summary, compacted_until_id, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    summary = excluded.summary,
                    compacted_until_id = excluded.compacted_until_id,
                    updated_at = excluded.updated_at
                """,
                (chat_id, summary.strip(), int(compacted_until_id), updated_at),
            )
            self.conn.commit()

    # 在单个互斥锁临界区内原子读取会话压缩状态与活跃消息列表，避免后台推进游标导致空洞
    def get_context_snapshot(
        self,
        chat_id: str,
        *,
        limit: int | None = None,
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        with self._lock:
            compaction = self.get_compaction(chat_id)
            compacted_until = int(compaction["compacted_until_id"])
            if limit is None:
                rows = self.conn.execute(
                    """
                    SELECT id, chat_id, role, content, created_at
                    FROM messages
                    WHERE chat_id = ? AND id > ?
                    ORDER BY id ASC
                    """,
                    (chat_id, compacted_until),
                ).fetchall()
                return compaction, [self._row_to_message(row) for row in rows]
            if limit <= 0:
                return compaction, []
            rows = self.conn.execute(
                """
                SELECT id, chat_id, role, content, created_at
                FROM messages
                WHERE chat_id = ? AND id > ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (chat_id, compacted_until, limit),
            ).fetchall()
            return compaction, [self._row_to_message(row) for row in reversed(rows)]

    # 按时间正序返回指定会话尚未被压缩的活跃消息（id > compacted_until_id）
    def active_messages(
        self, chat_id: str, *, limit: int | None = None
    ) -> list[dict[str, object]]:
        _, messages = self.get_context_snapshot(chat_id, limit=limit)
        return messages

    # 统计全部或单个会话已保存的消息总数
    def count_messages(self, chat_id: str | None = None) -> int:
        with self._lock:
            if chat_id is None:
                row = self.conn.execute("SELECT COUNT(*) AS total FROM messages").fetchone()
            else:
                row = self.conn.execute(
                    "SELECT COUNT(*) AS total FROM messages WHERE chat_id = ?",
                    (chat_id,),
                ).fetchone()
        return int(row["total"]) if row is not None else 0
