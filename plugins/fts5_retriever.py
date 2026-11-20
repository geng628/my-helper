"""SQLite FTS5 + BM25 检索插件（零第三方依赖）：中英双语分词、停用词与词干归一化、SQL 级过期过滤及增量同步。"""
from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from personal_assistant.assistant.memory import SearchHit
from personal_assistant.assistant.memory_lifecycle import UNTIL_PATTERN, today_shanghai

if TYPE_CHECKING:
    from personal_assistant.assistant.memory import MemoryStore

RETRIEVER_NAME = "fts5_bm25"

_ASCII_WORD_RE = re.compile(r"[a-z0-9]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")

# 高频英文功能词/疑问词停用表，避免长对话检索被 what/did/the 等噪声淹没
EN_STOPWORDS = frozenset(
    {
        "a", "about", "above", "after", "again", "all", "also", "am", "an", "and",
        "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
        "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
        "done", "down", "during", "each", "else", "few", "for", "from", "further",
        "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
        "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
        "just", "may", "me", "might", "more", "most", "must", "my", "myself", "no",
        "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our",
        "ours", "out", "over", "own", "s", "same", "shall", "she", "should", "so",
        "some", "such", "t", "than", "that", "the", "their", "theirs", "them", "then",
        "there", "these", "they", "this", "those", "through", "to", "too", "under",
        "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
        "while", "who", "whom", "whose", "why", "will", "with", "would", "you", "your",
        "yours", "tell", "like", "know", "think", "really", "actually", "still",
    }
)

# 常见英文不规则动词词形还原映射，提升英文多会话问答召回
EN_IRREGULAR_LEMMAS: dict[str, str] = {
    "went": "go",
    "gone": "go",
    "bought": "buy",
    "ran": "run",
    "made": "make",
    "gave": "give",
    "given": "give",
    "took": "take",
    "taken": "take",
    "saw": "see",
    "seen": "see",
    "told": "tell",
    "found": "find",
    "won": "win",
    "met": "meet",
    "wrote": "write",
    "written": "write",
    "ate": "eat",
    "eaten": "eat",
    "drove": "drive",
    "driven": "drive",
    "spoke": "speak",
    "spoken": "speak",
    "chose": "choose",
    "chosen": "choose",
    "built": "build",
    "sent": "send",
    "spent": "spend",
    "left": "leave",
    "kept": "keep",
    "felt": "feel",
    "heard": "hear",
    "held": "hold",
    "read": "read",
    "children": "child",
    "people": "person",
    "women": "woman",
    "men": "man",
}

# 中文疑问词/连接虚词切分正则（用于查询侧去噪与双侧轻度虚词切分）
_CJK_QUERY_NOISE_RE = re.compile(
    r"(?:有什么|是什么|是多少|为什么|怎么样|是怎么|怎么|怎样|如何|哪些|哪里|哪个|哪天|哪一|"
    r"什么|是否|能否|可以|请问|一下|以及|或者|还是|但是|如果|的话|之后|之前|里面|有关|关于|"
    r"的|了|吗|呢|啊|吧|和|与|及)"
)
_CJK_DOC_PARTICLE_RE = re.compile(r"[的了吗呢啊吧]")


# 对单个英文词执行轻量词干与不规则动词归一化（零依赖，保护 macos 等非复数词尾）
def _normalize_en_word(word: str) -> str:
    if word in EN_IRREGULAR_LEMMAS:
        return EN_IRREGULAR_LEMMAS[word]
    if len(word) <= 3 or word.isdigit():
        return word
    if word.endswith("ies") and len(word) >= 5:
        return word[:-3] + "y"
    if word.endswith("ied") and len(word) >= 5:
        return word[:-3] + "y"
    if word.endswith("ing") and len(word) >= 6:
        stem = word[:-3]
        if len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] not in {"s", "l", "z"}:
            stem = stem[:-1]
        return stem
    if word.endswith("ed") and len(word) >= 5:
        stem = word[:-2]
        if len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] not in {"s", "l", "z"}:
            stem = stem[:-1]
        return stem
    if word.endswith(("ches", "shes", "xes", "zes", "sses")) and len(word) >= 6:
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is", "os", "as", "js", "ts", "rs")):
        return word[:-1]
    return word


# 对中英文混合文本做统一分词：英文去停用词+词干归一化，中文按虚词切段并提取双字 bigram
def tokenize_text(text: str, *, for_query: bool = False) -> list[str]:
    lowered = text.lower()
    tokens: list[str] = []

    for word in _ASCII_WORD_RE.findall(lowered):
        if len(word) < 2 and not word.isdigit():
            continue
        if word in EN_STOPWORDS:
            continue
        norm = _normalize_en_word(word)
        if norm in EN_STOPWORDS:
            continue
        tokens.append(norm)

    splitter = _CJK_QUERY_NOISE_RE if for_query else _CJK_DOC_PARTICLE_RE
    for run in _CJK_RUN_RE.findall(lowered):
        for seg in splitter.split(run):
            if len(seg) >= 2:
                for idx in range(len(seg) - 1):
                    tokens.append(seg[idx : idx + 2])
            elif len(seg) == 1 and not for_query:
                tokens.append(seg)

    # 兜底：若查询全部由停用词或单字构成，保留原始有效字符防止空查询报错
    if for_query and not tokens:
        for word in _ASCII_WORD_RE.findall(lowered):
            if word:
                tokens.append(word)
        for run in _CJK_RUN_RE.findall(lowered):
            if len(run) >= 2:
                for idx in range(len(run) - 1):
                    tokens.append(run[idx : idx + 2])
            elif run:
                tokens.append(run)
    return tokens


# 解析单行文本中的 (until: YYYY-MM-DD) 截止日期，无截止日期时返回空字符串
def _extract_until_date(line: str) -> str:
    match = UNTIL_PATTERN.search(line)
    return match.group(1) if match else ""


# 单个搜索根目录对应的内存 SQLite FTS5 增量索引容器
class _RootFTSIndex:
    # 初始化内存 SQLite FTS5 表结构与文件指纹缓存
    def __init__(self) -> None:
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.execute(
            """
            CREATE TABLE chunks_meta (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                line_no INTEGER NOT NULL,
                section TEXT NOT NULL,
                until_date TEXT NOT NULL,
                is_archive INTEGER NOT NULL
            )
            """
        )
        self.conn.execute("CREATE INDEX idx_chunks_meta_path ON chunks_meta(path)")
        self.conn.execute(
            """
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                path_tokens,
                section_tokens,
                body_tokens,
                context_tokens,
                tokenize='unicode61 tokenchars ''_'''
            )
            """
        )
        self.fingerprints: dict[str, tuple[int, int, int]] = {}
        self.file_lines: dict[str, tuple[list[str], list[str]]] = {}

    # 增量同步目录中的 Markdown/文本文件，仅重建新增、变更或被删除的文件索引
    def sync(self, searchable: list[tuple[Path, str, bool]], *, is_archive_root: bool) -> None:
        current_paths: set[str] = set()
        to_reindex: list[tuple[Path, str, bool, tuple[int, int, int]]] = []

        for resolved, display_path, is_memory in searchable:
            try:
                st = resolved.stat()
            except OSError:
                continue
            fp = (st.st_mtime_ns, st.st_size, st.st_ino)
            current_paths.add(display_path)
            if self.fingerprints.get(display_path) != fp:
                to_reindex.append((resolved, display_path, is_memory, fp))

        removed_paths = [p for p in self.fingerprints if p not in current_paths]
        if not removed_paths and not to_reindex:
            return

        with self.conn:
            for old_path in removed_paths:
                self._delete_path(old_path)
                self.fingerprints.pop(old_path, None)
                self.file_lines.pop(old_path, None)

            for resolved, display_path, is_memory, fp in to_reindex:
                try:
                    text = resolved.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    self._delete_path(display_path)
                    self.fingerprints.pop(display_path, None)
                    self.file_lines.pop(display_path, None)
                    continue
                self._delete_path(display_path)
                self._index_file(
                    display_path,
                    text,
                    track_expiry=is_memory,
                    is_archive=1 if is_archive_root else 0,
                )
                self.fingerprints[display_path] = fp

    # 删除指定文件路径对应的全部元数据与 FTS5 倒排条目
    def _delete_path(self, display_path: str) -> None:
        rows = self.conn.execute(
            "SELECT id FROM chunks_meta WHERE path = ?",
            (display_path,),
        ).fetchall()
        if not rows:
            return
        ids = [(r[0],) for r in rows]
        self.conn.executemany("DELETE FROM chunks_fts WHERE rowid = ?", ids)
        self.conn.execute("DELETE FROM chunks_meta WHERE path = ?", (display_path,))

    # 按行索引单篇笔记：标题归入所属正文行的 section_tokens，常驻相邻行归入低权 context_tokens
    def _index_file(
        self,
        display_path: str,
        text: str,
        *,
        track_expiry: bool,
        is_archive: int,
    ) -> None:
        lines = text.splitlines()
        until_dates = [_extract_until_date(line) if track_expiry else "" for line in lines]
        self.file_lines[display_path] = (lines, until_dates)
        if not lines:
            return

        path_tok = " ".join(tokenize_text(display_path, for_query=False))
        sections: list[str] = []
        current_sec = ""
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                current_sec = stripped.lstrip("# ").strip() or current_sec
            sections.append(current_sec)

        body_toks_per_line = [
            " ".join(tokenize_text(line, for_query=False))
            if (line.strip() and not line.strip().startswith("#"))
            else ""
            for line in lines
        ]
        has_content_line = any(body_toks_per_line)

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            is_heading = stripped.startswith("#")
            if is_heading and has_content_line:
                continue

            sec_tok = " ".join(tokenize_text(sections[idx - 1], for_query=False))
            body_tok = body_toks_per_line[idx - 1] if not is_heading else sec_tok

            # 仅把同文件内无过期标签的相邻行并入低权 context_tokens，避免已过期行借邻居行泄漏
            ctx_parts: list[str] = []
            for neighbor in (idx - 2, idx):
                if 0 <= neighbor < len(lines) and not until_dates[neighbor]:
                    ntok = body_toks_per_line[neighbor]
                    if ntok:
                        ctx_parts.append(ntok)
            ctx_tok = " ".join(ctx_parts)

            cur = self.conn.execute(
                """
                INSERT INTO chunks_meta (path, line_no, section, until_date, is_archive)
                VALUES (?, ?, ?, ?, ?)
                """,
                (display_path, idx, sections[idx - 1], until_dates[idx - 1], is_archive),
            )
            rowid = cur.lastrowid
            self.conn.execute(
                """
                INSERT INTO chunks_fts (rowid, path_tokens, section_tokens, body_tokens, context_tokens)
                VALUES (?, ?, ?, ?, ?)
                """,
                (rowid, path_tok, sec_tok, body_tok, ctx_tok),
            )


# SQLite FTS5 + BM25 检索插件配置项（从 config/tools.json 的 retrieval 段读取）
@dataclass(frozen=True)
class FTS5Config:
    min_term_coverage: float = 0.22
    min_bm25_score: float = 0.1
    max_hits_per_file: int = 2

    # 从字典构造 FTS5Config 并校验数值范围
    @classmethod
    def from_dict(cls, raw: dict[str, object] | None) -> "FTS5Config":
        cfg = raw or {}
        coverage = float(cfg.get("min_term_coverage", 0.22))
        min_bm25 = float(cfg.get("min_bm25_score", 0.1))
        per_file = int(cfg.get("max_hits_per_file", 2))
        if not (0.0 <= coverage <= 1.0):
            raise ValueError("retrieval.min_term_coverage 必须在 [0, 1] 之间。")
        if min_bm25 < 0.0:
            raise ValueError("retrieval.min_bm25_score 必须 >= 0。")
        if per_file < 1:
            raise ValueError("retrieval.max_hits_per_file 必须 >= 1。")
        return cls(
            min_term_coverage=coverage,
            min_bm25_score=min_bm25,
            max_hits_per_file=per_file,
        )


# SQLite FTS5 + BM25 检索后端插件（线程安全，按根目录隔离并支持只读内存索引）
class FTS5Retriever:
    name = RETRIEVER_NAME

    # 初始化检索配置、线程锁与按目录划分的 FTS5 索引缓存
    def __init__(self, config: FTS5Config | None = None) -> None:
        self.config = config or FTS5Config()
        self._lock = threading.RLock()
        self._indices: dict[tuple[Path, Path], _RootFTSIndex] = {}

    # 生成当前命中行上下各 2 行的上下文片段，并自动隐藏在 ref_day 已过期的相邻行
    @staticmethod
    def _build_snippet(
        lines: list[str],
        until_dates: list[str],
        line_no: int,
        ref_day_iso: str,
    ) -> str:
        start = max(1, line_no - 2)
        end = min(len(lines), line_no + 2)
        return "\n".join(
            f"  {n}: {lines[n - 1]}"
            for n in range(start, end + 1)
            if not (until_dates[n - 1] and until_dates[n - 1] < ref_day_iso)
        )

    # 执行增量同步 + SQLite FTS5 BM25 检索 + 覆盖率阈值门控 + 每文件多样性重排
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

        fts_expr = " OR ".join(f'"{term}"' for term in query_terms)
        query_set = set(query_terms)
        # 较长查询（>=4 个有效词）要求至少命中 2 个词元，避免单个高频实体词触发假阳性
        min_matched = 2 if len(query_set) >= 4 else 1
        candidate_limit = max(limit * 16, 80)

        with self._lock:
            cache_key = (memory.data_root, target_dir)
            index = self._indices.get(cache_key)
            if index is None:
                index = _RootFTSIndex()
                self._indices[cache_key] = index
            index.sync(searchable, is_archive_root=searching_archive)

            rows = index.conn.execute(
                """
                SELECT
                    m.path,
                    m.line_no,
                    m.section,
                    -bm25(chunks_fts, 0.4, 1.5, 3.0, 0.4) AS bm25_score,
                    f.path_tokens,
                    f.section_tokens,
                    f.body_tokens,
                    f.context_tokens
                FROM chunks_fts AS f
                JOIN chunks_meta AS m ON m.id = f.rowid
                WHERE chunks_fts MATCH ?
                  AND (? = 1 OR m.is_archive = 0)
                  AND (m.until_date = '' OR m.until_date >= ?)
                ORDER BY bm25(chunks_fts, 0.4, 1.5, 3.0, 0.4)
                LIMIT ?
                """,
                (fts_expr, 1 if searching_archive else 0, ref_day_iso, candidate_limit),
            ).fetchall()

            total_chunks = int(
                index.conn.execute("SELECT COUNT(*) FROM chunks_meta").fetchone()[0]
            )
            scored: list[tuple[float, str, int, str]] = []
            for path, line_no, section, bm25_score, p_tok, s_tok, b_tok, c_tok in rows:
                bm25_val = max(0.0, float(bm25_score))
                # SQLite FTS5 在极小语料（<10 行）下 ln((N-n+0.5)/(n+0.5)) 可能为 0，仅在语料达规模后启用 BM25 下限
                if total_chunks >= 10 and bm25_val < self.config.min_bm25_score:
                    continue
                core_terms = set(f"{p_tok} {s_tok} {b_tok}".split())
                all_terms = core_terms | set(c_tok.split())
                matched_all = query_set & all_terms
                matched_core = query_set & core_terms
                if not matched_core:
                    continue
                if len(matched_all) < min_matched:
                    continue
                coverage = len(matched_all) / len(query_set)
                # 长查询上若同时命中 >=3 个词且 BM25 显著高（>=10），即使中文 bigram 分母偏大也应放行
                high_conf_bm25 = len(matched_all) >= 3 and bm25_val >= 10.0
                if coverage < self.config.min_term_coverage and not high_conf_bm25:
                    continue
                body_terms = set(b_tok.split())
                body_ratio = len(query_set & body_terms) / len(query_set)
                base_score = bm25_val if bm25_val > 0.0 else 0.1
                final_score = round(base_score * (1.0 + 0.6 * coverage + 0.4 * body_ratio), 4)
                scored.append((final_score, str(path), int(line_no), str(section)))

            scored.sort(key=lambda item: (-item[0], item[1], item[2]))

            # 1. 同文件内仅跳过已落在前一条 5 行摘要窗口 [prev-2, prev+2] 内的重复命中
            deduped: list[tuple[float, str, int, str]] = []
            used_lines_by_file: dict[str, list[int]] = {}
            for item in scored:
                _, path, line_no, _ = item
                prev_lines = used_lines_by_file.setdefault(path, [])
                if any(abs(line_no - prev) <= 2 for prev in prev_lines):
                    continue
                prev_lines.append(line_no)
                deduped.append(item)

            # 2. 按 max_hits_per_file 做多文件多样性优先挑选，剩余名额再从溢出候选中补齐
            primary_picks: list[tuple[float, str, int, str]] = []
            overflow: list[tuple[float, str, int, str]] = []
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
            hits: list[SearchHit] = []
            for score, path, line_no, section in chosen:
                lines, until_dates = index.file_lines.get(path, ([], []))
                if not lines or line_no > len(lines):
                    continue
                snippet = self._build_snippet(lines, until_dates, line_no, ref_day_iso)
                hits.append(
                    SearchHit(
                        path=path,
                        line=line_no,
                        section=section,
                        snippet=snippet,
                        score=score,
                    )
                )
            return hits


# 检索插件工厂函数：由 personal_assistant.plugins 自动发现并按 config/tools.json 的 retrieval 段装配
def create_retriever(config: dict[str, object]) -> FTS5Retriever:
    return FTS5Retriever(FTS5Config.from_dict(config))


RETRIEVERS = {
    RETRIEVER_NAME: create_retriever,
}
