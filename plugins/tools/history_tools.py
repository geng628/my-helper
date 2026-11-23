"""只读增量 FTS5 历史对话检索插件：免改写 chat_history.sqlite3 结构，直接复用双语分词与 BM25 门控检索过往原文。"""
from __future__ import annotations

import sqlite3
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from personal_assistant.agent.tools import tool
from personal_assistant.assistant.common import truncate_utf8
from plugins.fts5_retriever import FTS5Config, tokenize_text

DEFAULT_HISTORY_LIMIT = 5
DEFAULT_CONTEXT_WINDOW = 1
MAX_MATCHED_MSG_CHARS = 1800
MAX_NEIGHBOR_MSG_CHARS = 600
MAX_OUTPUT_BYTES = 10000
MAX_CACHED_DBS = 16


# 单条历史对话搜索命中结果（含同会话相邻上下文窗口）
@dataclass(frozen=True)
class HistoryHit:
    msg_id: int
    chat_id: str
    role: str
    created_at: str
    score: float
    context_messages: tuple[tuple[int, str, str, str], ...]


# 单个 chat_history.sqlite3 对应的内存增量 FTS5 索引容器（只读同步，零表结构侵入）
class _HistoryFTSIndex:
    # 初始化内存 SQLite FTS5 表结构、线程锁与增量同步游标
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.execute(
            """
            CREATE TABLE msg_meta (
                id INTEGER PRIMARY KEY,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self.conn.execute("CREATE INDEX idx_msg_meta_chat ON msg_meta(chat_id, id)")
        self.conn.execute(
            """
            CREATE VIRTUAL TABLE msg_fts USING fts5(
                body_tokens,
                prev_tokens,
                tokenize='unicode61 tokenchars ''_'''
            )
            """
        )
        self.db_ino = -1
        self.last_indexed_id = 0
        self.indexed_count = 0
        self.last_tokens_by_chat: dict[str, str] = {}

    # 线程安全地关闭内存 FTS5 连接
    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # 清空内存索引与游标状态，用于库文件重建或切换时的干净重置
    def _reset(self) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM msg_fts")
            self.conn.execute("DELETE FROM msg_meta")
        self.last_indexed_id = 0
        self.indexed_count = 0
        self.last_tokens_by_chat.clear()

    # 以只读 URI 模式打开源库并按 id > last_indexed_id 增量同步新消息（检测到删库重建时自动重置）
    def sync(self, db_path: Path) -> bool:
        resolved = db_path.resolve()
        try:
            st = resolved.stat()
        except OSError:
            self._reset()
            self.db_ino = -1
            return False

        ro_uri = f"{resolved.as_uri()}?mode=ro"
        ro_conn: sqlite3.Connection | None = None
        try:
            ro_conn = sqlite3.connect(ro_uri, uri=True, timeout=5.0)
            try:
                stat_row = ro_conn.execute(
                    "SELECT COALESCE(MAX(id), 0), COUNT(*) FROM messages"
                ).fetchone()
            except sqlite3.OperationalError:
                return False
            max_id = int(stat_row[0]) if stat_row else 0
            total_count = int(stat_row[1]) if stat_row else 0

            if (
                st.st_ino != self.db_ino
                or max_id < self.last_indexed_id
                or total_count < self.indexed_count
            ):
                self._reset()
                self.db_ino = st.st_ino

            if max_id == self.last_indexed_id and total_count == self.indexed_count:
                return True

            rows = ro_conn.execute(
                """
                SELECT id, chat_id, role, content, created_at
                FROM messages
                WHERE id > ?
                ORDER BY id ASC
                """,
                (self.last_indexed_id,),
            ).fetchall()
        except sqlite3.Error:
            return False
        finally:
            if ro_conn is not None:
                ro_conn.close()

        if not rows:
            return True

        with self.conn:
            for msg_id, chat_id, role, content, created_at in rows:
                mid = int(msg_id)
                cid = str(chat_id)
                text = str(content)
                body_tok = " ".join(tokenize_text(text, for_query=False))
                # prev_tokens 仅引用同一会话前一条消息，严格保证单向追加与 before_msg_id 隔离
                prev_tok = self.last_tokens_by_chat.get(cid, "")
                self.conn.execute(
                    """
                    INSERT INTO msg_meta (id, chat_id, role, content, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (mid, cid, str(role), text, str(created_at)),
                )
                self.conn.execute(
                    """
                    INSERT INTO msg_fts (rowid, body_tokens, prev_tokens)
                    VALUES (?, ?, ?)
                    """,
                    (mid, body_tok, prev_tok),
                )
                self.last_tokens_by_chat[cid] = body_tok
                self.last_indexed_id = mid
                self.indexed_count += 1
        return True

    # 同步只读快照后执行 BM25 + 词覆盖率双门控检索，并拼装同会话相邻上下文窗口
    def search(
        self,
        db_path: Path,
        query: str,
        *,
        chat_id: str | None = None,
        before_msg_id: int | None = None,
        limit: int = DEFAULT_HISTORY_LIMIT,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        config: FTS5Config | None = None,
    ) -> list[HistoryHit]:
        cfg = config or FTS5Config()
        query_terms = list(dict.fromkeys(tokenize_text(query, for_query=True)))
        if not query_terms:
            return []

        clean_chat = chat_id.strip() if isinstance(chat_id, str) and chat_id.strip() else None
        limit_val = max(1, int(limit))
        window_val = max(0, int(context_window))
        fts_expr = " OR ".join(f'"{term}"' for term in query_terms)
        query_set = set(query_terms)
        min_matched = 2 if len(query_set) >= 4 else 1
        candidate_limit = max(limit_val * 12, 60)

        with self._lock:
            if not self.sync(db_path):
                return []

            rows = self.conn.execute(
                """
                SELECT
                    m.id,
                    m.chat_id,
                    m.role,
                    m.created_at,
                    -bm25(msg_fts, 3.0, 0.8) AS bm25_score,
                    f.body_tokens,
                    f.prev_tokens
                FROM msg_fts AS f
                JOIN msg_meta AS m ON m.id = f.rowid
                WHERE msg_fts MATCH ?
                  AND (? IS NULL OR m.chat_id = ?)
                  AND (? IS NULL OR m.id < ?)
                ORDER BY bm25(msg_fts, 3.0, 0.8)
                LIMIT ?
                """,
                (fts_expr, clean_chat, clean_chat, before_msg_id, before_msg_id, candidate_limit),
            ).fetchall()

            total_msgs = int(
                self.conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM msg_meta
                    WHERE (? IS NULL OR chat_id = ?)
                      AND (? IS NULL OR id < ?)
                    """,
                    (clean_chat, clean_chat, before_msg_id, before_msg_id),
                ).fetchone()[0]
            )

            scored: list[tuple[float, int, str, str, str]] = []
            for msg_id, m_chat_id, role, created_at, bm25_score, b_tok, p_tok in rows:
                bm25_val = max(0.0, float(bm25_score))
                if total_msgs >= 10 and bm25_val < cfg.min_bm25_score:
                    continue
                body_terms = set(str(b_tok).split())
                all_terms = body_terms | set(str(p_tok).split())
                matched_body = query_set & body_terms
                matched_all = query_set & all_terms
                if not matched_body:
                    continue
                if len(matched_all) < min_matched:
                    continue
                coverage = len(matched_all) / len(query_set)
                high_conf_bm25 = len(matched_all) >= 3 and bm25_val >= 10.0
                if coverage < cfg.min_term_coverage and not high_conf_bm25:
                    continue
                body_ratio = len(matched_body) / len(query_set)
                base_score = bm25_val if bm25_val > 0.0 else 0.1
                final_score = round(base_score * (1.0 + 0.6 * coverage + 0.4 * body_ratio), 4)
                scored.append((final_score, int(msg_id), str(m_chat_id), str(role), str(created_at)))

            scored.sort(key=lambda item: (-item[0], -item[1]))

            chosen: list[tuple[float, int, str, str, str]] = []
            chosen_by_chat: dict[str, list[int]] = {}
            for item in scored:
                _, hit_id, m_chat_id, _, _ = item
                prev_ids = chosen_by_chat.setdefault(m_chat_id, [])
                if any(abs(hit_id - prev_id) <= window_val for prev_id in prev_ids):
                    continue
                prev_ids.append(hit_id)
                chosen.append(item)
                if len(chosen) >= limit_val:
                    break

            hits: list[HistoryHit] = []
            for score, hit_id, m_chat_id, role, created_at in chosen:
                prev_rows = self.conn.execute(
                    """
                    SELECT id, role, content, created_at
                    FROM msg_meta
                    WHERE chat_id = ? AND id < ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (m_chat_id, hit_id, window_val),
                ).fetchall()
                cur_row = self.conn.execute(
                    "SELECT id, role, content, created_at FROM msg_meta WHERE id = ?",
                    (hit_id,),
                ).fetchone()
                next_rows = self.conn.execute(
                    """
                    SELECT id, role, content, created_at
                    FROM msg_meta
                    WHERE chat_id = ? AND id > ? AND (? IS NULL OR id < ?)
                    ORDER BY id ASC
                    LIMIT ?
                    """,
                    (m_chat_id, hit_id, before_msg_id, before_msg_id, window_val),
                ).fetchall()
                ctx_rows = [
                    *((int(r[0]), str(r[1]), str(r[2]), str(r[3])) for r in reversed(prev_rows)),
                    *((int(cur_row[0]), str(cur_row[1]), str(cur_row[2]), str(cur_row[3])) for _ in (0,) if cur_row),
                    *((int(r[0]), str(r[1]), str(r[2]), str(r[3])) for r in next_rows),
                ]
                hits.append(
                    HistoryHit(
                        msg_id=hit_id,
                        chat_id=m_chat_id,
                        role=role,
                        created_at=created_at,
                        score=score,
                        context_messages=tuple(ctx_rows),
                    )
                )
            return hits


_CACHE_LOCK = threading.RLock()
_HISTORY_INDICES: OrderedDict[Path, _HistoryFTSIndex] = OrderedDict()


# 获取或创建指定数据库路径对应的有界 LRU 增量索引实例
def _get_history_index(db_path: Path) -> _HistoryFTSIndex:
    key = db_path.resolve()
    with _CACHE_LOCK:
        index = _HISTORY_INDICES.get(key)
        if index is not None:
            _HISTORY_INDICES.move_to_end(key)
            return index
        index = _HistoryFTSIndex()
        _HISTORY_INDICES[key] = index
        while len(_HISTORY_INDICES) > MAX_CACHED_DBS:
            _, oldest = _HISTORY_INDICES.popitem(last=False)
            oldest.close()
        return index


# 按字符上限裁剪单条消息正文，保留换行结构便于阅读列表
def _clip_message(text: str, max_chars: int) -> str:
    clean = text.strip()
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + "…(已截断)"


# 将结构化历史命中列表格式化为紧凑可读的一问一答文本块（超长受 MAX_OUTPUT_BYTES 限制）
def format_history_hits(hits: list[HistoryHit]) -> str:
    if not hits:
        return "未找到匹配的历史对话记录。"
    blocks: list[str] = []
    for idx, hit in enumerate(hits, start=1):
        lines = [f"[命中 {idx} | chat={hit.chat_id} | score={hit.score:.2f}]"]
        matched_pos = next(
            (pos for pos, item in enumerate(hit.context_messages) if item[0] == hit.msg_id),
            -1,
        )
        for pos, (cid, role, content, created_at) in enumerate(hit.context_messages):
            is_matched = cid == hit.msg_id
            # 当命中用户提问时，紧随其后的助手答复也保留完整长度，避免长推荐列表被腰斩
            is_paired_reply = (
                matched_pos != -1
                and pos == matched_pos + 1
                and hit.context_messages[matched_pos][1] == "user"
                and role == "assistant"
            )
            cap = MAX_MATCHED_MSG_CHARS if (is_matched or is_paired_reply) else MAX_NEIGHBOR_MSG_CHARS
            marker = ">>> " if is_matched else "    "
            day_part = created_at[:10] if created_at else ""
            lines.append(f"{marker}[{role} #{cid} | {day_part}] {_clip_message(content, cap)}")
        blocks.append("\n".join(lines))
    return truncate_utf8("\n\n".join(blocks), MAX_OUTPUT_BYTES)


# 对指定 chat_history.sqlite3 执行只读增量 FTS5 历史检索并返回格式化结果
def execute_search_chat_history(
    db_path: Path,
    query: str,
    *,
    chat_id: str | None = None,
    before_msg_id: int | None = None,
    limit: int = DEFAULT_HISTORY_LIMIT,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    config: FTS5Config | None = None,
) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("搜索关键词不能为空。")
    index = _get_history_index(db_path)
    hits = index.search(
        db_path,
        query,
        chat_id=chat_id,
        before_msg_id=before_msg_id,
        limit=limit,
        context_window=context_window,
        config=config,
    )
    return format_history_hits(hits)


# 插件工具入口：只读检索当前轮之前的历史对话原文（运行时上下文由 PluginContext 自动绑定）
@tool(
    name="search_chat_history",
    description=(
        "按关键词检索当前轮之前的历史对话原文（含同会话相邻一问一答），"
        "用于找回未整理进 data/memory/ 的过往推荐、数字或原话细节；"
        "当前有效状态与偏好以 data/memory/ 为准。"
    ),
    parameters={"query": "搜索关键词（中英文皆可），用于匹配历史对话原文"},
    risk="read",
)
def search_chat_history(
    query: str,
    *,
    data_root: Path | None = None,
    history_db_path: Path | None = None,
    before_msg_id: int | None = None,
    retrieval_config: Mapping[str, object] | None = None,
) -> str:
    db_path = (
        Path(history_db_path)
        if history_db_path is not None
        else ((data_root / "state" / "chat_history.sqlite3") if data_root is not None else None)
    )
    if db_path is None:
        return "未找到匹配的历史对话记录。"
    cfg = (
        FTS5Config.from_dict(dict(retrieval_config))
        if isinstance(retrieval_config, Mapping)
        else FTS5Config()
    )
    return execute_search_chat_history(
        db_path,
        query,
        before_msg_id=before_msg_id,
        config=cfg,
    )
