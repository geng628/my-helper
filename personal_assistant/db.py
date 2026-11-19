"""极简 SQLite 状态存储：只维护一张 messages 对话历史表。"""
from __future__ import annotations

import sqlite3
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


# 幂等初始化 messages 表结构与按会话查询的联合索引
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
    conn.commit()


# 对话历史存储类：负责写入、按会话查询最近消息与统计消息数
class StateStore:
    # 初始化数据库连接并确保表结构存在
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.conn = connect_sqlite(self.db_path)
        init_schema(self.conn)

    # 关闭底层 SQLite 连接
    def close(self) -> None:
        self.conn.close()

    # 追加一条对话消息（user 或 assistant），返回自增消息编号
    def append_message(self, chat_id: str, role: str, content: str) -> int:
        created_at = datetime.now(UTC).isoformat()
        cursor = self.conn.execute(
            "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (chat_id, role, content, created_at),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    # 按时间正序返回指定会话最近 limit 条对话记录
    def recent_messages(self, chat_id: str, limit: int = 12) -> list[dict[str, object]]:
        if limit <= 0:
            return []
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
        return [
            {
                "id": int(row["id"]),
                "chat_id": str(row["chat_id"]),
                "role": str(row["role"]),
                "content": str(row["content"]),
                "created_at": str(row["created_at"]),
            }
            for row in reversed(rows)
        ]

    # 统计全部或单个会话已保存的消息总数
    def count_messages(self, chat_id: str | None = None) -> int:
        if chat_id is None:
            row = self.conn.execute("SELECT COUNT(*) AS total FROM messages").fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) AS total FROM messages WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
        return int(row["total"]) if row is not None else 0
