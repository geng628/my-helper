"""检索后增强引擎（零第三方依赖）：跨文档去重、句/行级上下文压缩、Lost-in-the-Middle 重排与 CRAG 相关性校验兜底。"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, Protocol, TypeVar

from personal_assistant.ai import (
    AbortSignal,
    Context,
    ModelInvoker,
    NoopModelClient,
    StreamAbortedError,
    StreamOptions,
    UserMessage,
)
from personal_assistant.assistant.common import require_positive_int
from personal_assistant.assistant.text_search import tokenize_text

CRAGVerdict = Literal["correct", "ambiguous", "incorrect"]
CRAGAction = Literal["answer", "rewrite_retry", "switch_source", "abstain"]

_LINE_PREFIX_RE = re.compile(r"^\s*(\d+:\s*)(.*)$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])|(?<=\.)\s+")
_FALLBACK_TOOL_ORDER: tuple[str, ...] = ("web_search", "search_chat_history", "list_due_reviews")
_FALLBACK_TOOL_LABELS: dict[str, str] = {
    "web_search": "联网搜索（web_search）",
    "search_chat_history": "历史对话记录库（search_chat_history）",
    "list_due_reviews": "学习复习卡片（list_due_reviews）",
}


# 检索命中对象最小结构协议（零依赖 memory.py，兼容 SearchHit 不可变 dataclass）
class _HitLike(Protocol):
    path: str
    start_line: int
    end_line: int
    section: str
    snippet: str
    score: float


_THit = TypeVar("_THit", bound=_HitLike)


# 严格校验布尔配置项（拒绝字符串/整数隐式转换）
def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} 必须是布尔值（true/false）。")
    return value


# 严格校验有限浮点范围配置项（拒绝布尔值、NaN 与 Inf）
def _require_float_range(
    value: object,
    name: str,
    *,
    low: float,
    high: float,
    low_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是有限数字。")
    num = float(value)
    if not math.isfinite(num):
        raise ValueError(f"{name} 必须是有限数字。")
    too_low = num < low if low_inclusive else num <= low
    if too_low or num > high:
        left = "[" if low_inclusive else "("
        raise ValueError(f"{name} 必须在 {left}{low}, {high}] 范围内，实际：{num}")
    return num


# 检索后增强统一配置项（对应 config/tools.json 的 retrieval.post_retrieval 段）
@dataclass(frozen=True)
class PostRetrievalConfig:
    dedup_similarity: float = 0.82
    max_context_chars: int = 6000
    max_chars_per_hit: int = 1200
    compress_long_hits: bool = True
    lost_in_middle_reorder: bool = False
    crag_enabled: bool = True
    crag_high_threshold: float = 0.65
    crag_low_threshold: float = 0.25
    max_corrective_rewrites: int = 1

    # 校验全部去重阈值、预算上限、头尾重排与 CRAG 分档参数合法性
    def __post_init__(self) -> None:
        _require_float_range(
            self.dedup_similarity,
            "post_retrieval.dedup_similarity",
            low=0.0,
            high=1.0,
            low_inclusive=False,
        )
        require_positive_int(self.max_context_chars, "post_retrieval.max_context_chars")
        require_positive_int(self.max_chars_per_hit, "post_retrieval.max_chars_per_hit")
        _require_bool(self.compress_long_hits, "post_retrieval.compress_long_hits")
        _require_bool(self.lost_in_middle_reorder, "post_retrieval.lost_in_middle_reorder")
        _require_bool(self.crag_enabled, "post_retrieval.crag_enabled")
        high = _require_float_range(
            self.crag_high_threshold, "post_retrieval.crag_high_threshold", low=0.0, high=1.0
        )
        low = _require_float_range(
            self.crag_low_threshold, "post_retrieval.crag_low_threshold", low=0.0, high=1.0
        )
        if low > high:
            raise ValueError(
                f"post_retrieval.crag_low_threshold ({low}) 不能大于 crag_high_threshold ({high})。"
            )
        if (
            isinstance(self.max_corrective_rewrites, bool)
            or not isinstance(self.max_corrective_rewrites, int)
            or not (0 <= self.max_corrective_rewrites <= 2)
        ):
            raise ValueError("post_retrieval.max_corrective_rewrites 必须是 0 到 2 之间的整数。")

    # 从配置字典安全解析 PostRetrievalConfig
    @classmethod
    def from_dict(cls, raw: Mapping[str, object] | None) -> "PostRetrievalConfig":
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError("retrieval.post_retrieval 必须是 JSON 对象。")
        return cls(
            dedup_similarity=_require_float_range(
                raw.get("dedup_similarity", 0.82),
                "post_retrieval.dedup_similarity",
                low=0.0,
                high=1.0,
                low_inclusive=False,
            ),
            max_context_chars=require_positive_int(
                raw.get("max_context_chars", 6000),
                "post_retrieval.max_context_chars",
            ),
            max_chars_per_hit=require_positive_int(
                raw.get("max_chars_per_hit", 1200),
                "post_retrieval.max_chars_per_hit",
            ),
            compress_long_hits=_require_bool(
                raw.get("compress_long_hits", True),
                "post_retrieval.compress_long_hits",
            ),
            lost_in_middle_reorder=_require_bool(
                raw.get("lost_in_middle_reorder", False),
                "post_retrieval.lost_in_middle_reorder",
            ),
            crag_enabled=_require_bool(
                raw.get("crag_enabled", True),
                "post_retrieval.crag_enabled",
            ),
            crag_high_threshold=_require_float_range(
                raw.get("crag_high_threshold", 0.65),
                "post_retrieval.crag_high_threshold",
                low=0.0,
                high=1.0,
            ),
            crag_low_threshold=_require_float_range(
                raw.get("crag_low_threshold", 0.25),
                "post_retrieval.crag_low_threshold",
                low=0.0,
                high=1.0,
            ),
            max_corrective_rewrites=_parse_non_neg_int(
                raw.get("max_corrective_rewrites", 1),
                "post_retrieval.max_corrective_rewrites",
            ),
        )


# 解析非负整数配置项（严格拒绝布尔值与浮点数）
def _parse_non_neg_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} 必须是非负整数。")
    return value


# 剥离渲染片段中的 [背景:]、(表头) 与行号前缀，还原纯净正文供相似度去重与相关性覆盖度计算
def _strip_snippet_body(snippet: str) -> str:
    body_lines: list[str] = []
    for raw_line in snippet.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("[背景:") or stripped.startswith("(表头)") or stripped == "…":
            continue
        m = _LINE_PREFIX_RE.match(raw_line)
        body_lines.append(m.group(2).strip() if m else stripped)
    return " ".join(" ".join(body_lines).split())


# 跨文档/跨分块近重复去重：合并同文件重叠区间及词元集合 Jaccard 相似度超阈值的重复片段（保留高分位）
def deduplicate_hits(
    hits: Sequence[_THit],
    *,
    similarity_threshold: float = 0.82,
) -> list[_THit]:
    thresh = _require_float_range(
        similarity_threshold,
        "similarity_threshold",
        low=0.0,
        high=1.0,
        low_inclusive=False,
    )
    accepted: list[_THit] = []
    accepted_sigs: list[tuple[str, frozenset[str]]] = []
    spans_by_file: dict[str, list[tuple[int, int]]] = {}

    for hit in hits:
        prev_spans = spans_by_file.setdefault(hit.path, [])
        if any(hit.start_line <= p_end and hit.end_line >= p_start for p_start, p_end in prev_spans):
            continue

        clean_body = _strip_snippet_body(hit.snippet).lower()
        tok_set = frozenset(tokenize_text(clean_body, for_query=False))
        is_dup = False
        for prev_body, prev_toks in accepted_sigs:
            if clean_body and clean_body == prev_body:
                is_dup = True
                break
            if len(tok_set) >= 3 and len(prev_toks) >= 3:
                inter = len(tok_set & prev_toks)
                if inter > 0:
                    union = len(tok_set | prev_toks)
                    if (inter / union) >= thresh:
                        is_dup = True
                        break
        if is_dup:
            continue

        prev_spans.append((hit.start_line, hit.end_line))
        accepted.append(hit)
        accepted_sigs.append((clean_body, tok_set))
    return accepted


# 对单行超长段落按句末标点执行句级抽取压缩，保留命中核心句及邻接上下文并维持原行号前缀
def _compress_long_line(
    raw_line: str,
    query_terms: set[str],
    max_chars: int,
) -> str:
    if len(raw_line) <= max_chars:
        return raw_line
    m = _LINE_PREFIX_RE.match(raw_line)
    prefix = f"  {m.group(1)}" if m else "  "
    content = m.group(2).strip() if m else raw_line.strip()
    body_budget = max(40, max_chars - len(prefix) - 4)
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(content) if s.strip()]
    if len(sentences) <= 1:
        return f"{prefix}{content[:body_budget].rstrip()}…"

    scored: list[tuple[float, int]] = []
    for idx, sent in enumerate(sentences):
        s_low = sent.lower()
        s_toks = set(tokenize_text(s_low, for_query=False))
        score = float(len(query_terms & s_toks))
        if score == 0.0:
            score += sum(0.8 for t in query_terms if len(t) >= 2 and t in s_low)
        scored.append((score, idx))

    if any(sc > 0.0 for sc, _ in scored):
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        picked: set[int] = set()
        used = 0
        for sc, idx in scored:
            if sc <= 0.0 and picked:
                break
            for cand in (idx, idx - 1, idx + 1):
                if 0 <= cand < len(sentences) and cand not in picked:
                    cand_len = len(sentences[cand]) + (3 if picked else 0)
                    if used + cand_len <= body_budget or not picked:
                        picked.add(cand)
                        used += cand_len
        ordered_idx = sorted(picked)
        parts: list[str] = []
        for pos, s_idx in enumerate(ordered_idx):
            if pos > 0 and s_idx > ordered_idx[pos - 1] + 1:
                parts.append(" … ")
            elif pos > 0:
                parts.append(" ")
            parts.append(sentences[s_idx])
        joined = "".join(parts)
        if len(joined) > body_budget:
            joined = joined[:body_budget].rstrip() + "…"
        return f"{prefix}{joined}"

    # 纯语义同义命中无字面重叠时回退保留句首若干完整句子
    head_parts: list[str] = []
    used = 0
    for sent in sentences:
        if used + len(sent) > body_budget and head_parts:
            break
        head_parts.append(sent)
        used += len(sent) + 1
    rendered = " ".join(head_parts)
    if len(rendered) > body_budget:
        rendered = rendered[:body_budget].rstrip()
    return f"{prefix}{rendered}…"


# 对单条超长命中片段执行行/句级抽取压缩：保底保留 [背景:] 与 (表头)，按查询词锚点选取核心行及 ±1 邻行
def _compress_single_snippet(
    snippet: str,
    query_terms: set[str],
    max_chars: int,
) -> str:
    if len(snippet) <= max_chars:
        return snippet
    lines = snippet.splitlines()
    if not lines:
        return snippet[:max_chars]

    sticky: list[str] = []
    body_lines: list[str] = []
    in_header = True
    sticky_cap = max(80, max_chars // 3)
    sticky_used = 0
    for line in lines:
        s = line.strip()
        if in_header and (s.startswith("[背景:") or s.startswith("(表头)")):
            if sticky_used + len(line) + 1 <= sticky_cap:
                sticky.append(line)
                sticky_used += len(line) + 1
            continue
        in_header = False
        body_lines.append(line)

    if not body_lines:
        return "\n".join(sticky)[:max_chars]

    body_budget = max(80, max_chars - sticky_used - 8)

    # 给每一行正文按查询词匹配度打分（支持中英文词元与数字/错误码子串）
    scored_lines: list[tuple[float, int]] = []
    for idx, line in enumerate(body_lines):
        m = _LINE_PREFIX_RE.match(line)
        content = m.group(2) if m else line
        c_low = content.lower()
        c_toks = set(tokenize_text(c_low, for_query=False))
        score = float(len(query_terms & c_toks))
        score += sum(0.8 for t in query_terms if len(t) >= 2 and t in c_low)
        scored_lines.append((score, idx))

    picked_indices: set[int] = set()
    used_chars = 0
    if any(sc > 0.0 for sc, _ in scored_lines):
        scored_lines.sort(key=lambda pair: (-pair[0], pair[1]))
        for sc, idx in scored_lines:
            if sc <= 0.0 and picked_indices:
                break
            for cand in (idx, idx - 1, idx + 1):
                if 0 <= cand < len(body_lines) and cand not in picked_indices:
                    line_cost = min(len(body_lines[cand]), body_budget) + 1
                    if used_chars + line_cost <= body_budget or not picked_indices:
                        picked_indices.add(cand)
                        used_chars += line_cost
    else:
        # 零字面重合（例如纯稠密语义同义命中）时平滑保留开头行
        for idx, line in enumerate(body_lines):
            line_cost = min(len(line), body_budget) + 1
            if used_chars + line_cost > body_budget and picked_indices:
                break
            picked_indices.add(idx)
            used_chars += line_cost

    ordered = sorted(picked_indices)
    out_lines: list[str] = list(sticky)
    per_line_cap = max(80, body_budget // max(1, len(ordered)))
    for pos, idx in enumerate(ordered):
        if pos > 0 and idx > ordered[pos - 1] + 1:
            out_lines.append("  …")
        raw_line = body_lines[idx]
        if len(raw_line) > per_line_cap and len(raw_line) > body_budget // 2:
            raw_line = _compress_long_line(raw_line, query_terms, max(per_line_cap, body_budget // 2))
        out_lines.append(raw_line)
    if ordered and ordered[-1] < len(body_lines) - 1:
        out_lines.append("  …")

    rendered = "\n".join(out_lines)
    if len(rendered) <= max_chars:
        return rendered
    cut = rendered[: max_chars - 4].rsplit("\n", 1)[0]
    return f"{cut}\n  …" if cut else f"{rendered[: max_chars - 1].rstrip()}…"


# 对命中列表执行单条句/行级上下文压缩与整体字符预算收口，去除无关冗余噪声
def compress_hits(
    hits: Sequence[_THit],
    query: str,
    *,
    max_context_chars: int = 6000,
    max_chars_per_hit: int = 1200,
    compress_long_hits: bool = True,
) -> list[_THit]:
    total_budget = require_positive_int(max_context_chars, "max_context_chars")
    per_hit_budget = require_positive_int(max_chars_per_hit, "max_chars_per_hit")
    if not hits:
        return []

    query_terms = set(tokenize_text(query.strip(), for_query=True))
    compressed: list[_THit] = []
    used_chars = 0

    for idx, hit in enumerate(hits):
        remaining = total_budget - used_chars
        if idx > 0 and remaining < 160:
            break
        cap = min(per_hit_budget, remaining) if compress_long_hits else remaining
        snippet = hit.snippet
        if len(snippet) > cap:
            if compress_long_hits:
                snippet = _compress_single_snippet(snippet, query_terms, cap)
            else:
                cut = snippet[: max(40, cap - 4)].rsplit("\n", 1)[0]
                snippet = f"{cut}\n  …" if cut else f"{snippet[: max(40, cap - 1)].rstrip()}…"
        updated = hit if snippet == hit.snippet else replace(hit, snippet=snippet)
        compressed.append(updated)
        used_chars += len(updated.snippet) + len(updated.path) + 32
    return compressed


# Lost-in-the-Middle 头尾重排：将最高分证据置于开头与末尾、低分证据置于中间（[0, 2, 4, ..., 3, 1]）
def reorder_lost_in_middle(hits: Sequence[_THit]) -> list[_THit]:
    items = list(hits)
    if len(items) <= 2:
        return items
    head = items[0::2]
    tail = items[1::2][::-1]
    return head + tail


# 组合执行去重、上下文压缩与可选 Lost-in-the-Middle 头尾重排的单一路径流水线
def apply_post_retrieval(
    hits: Sequence[_THit],
    query: str,
    *,
    config: PostRetrievalConfig | None = None,
    compress: bool = True,
    reorder: bool | None = None,
) -> list[_THit]:
    cfg = config or PostRetrievalConfig()
    refined = deduplicate_hits(hits, similarity_threshold=cfg.dedup_similarity)
    if compress:
        refined = compress_hits(
            refined,
            query,
            max_context_chars=cfg.max_context_chars,
            max_chars_per_hit=cfg.max_chars_per_hit,
            compress_long_hits=cfg.compress_long_hits,
        )
    should_reorder = cfg.lost_in_middle_reorder if reorder is None else bool(reorder)
    if should_reorder:
        refined = reorder_lost_in_middle(refined)
    return refined


# CRAG / Self-RAG 相关性自检与兜底决策结果
@dataclass(frozen=True)
class CRAGAssessment:
    verdict: CRAGVerdict
    action: CRAGAction
    max_score: float
    rewritten_query: str = ""
    suggested_sources: tuple[str, ...] = ()
    attempts: int = 1
    reason: str = ""

    # 导出 JSON 友好字典（供 smart_search 返回与 /rag query 展示）
    def to_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "action": self.action,
            "max_score": round(self.max_score, 6),
            "rewritten_query": self.rewritten_query,
            "suggested_sources": list(self.suggested_sources),
            "attempts": self.attempts,
            "reason": self.reason,
        }

    # 渲染面向模型与用户的可读中文兜底提示（仅在拒答、切换源或低置信度时返回非空文本）
    def format_notice(self) -> str:
        if self.action == "switch_source" and self.suggested_sources:
            labels = "、".join(_FALLBACK_TOOL_LABELS.get(src, src) for src in self.suggested_sources)
            return (
                f"【CRAG 相关性校验：本地知识库证据不足（verdict={self.verdict}，{self.reason}）】\n"
                f"为避免幻觉，已丢弃低相关片段；建议改用以下已启用数据源重新检索：{labels}。"
            )
        if self.action == "abstain":
            return (
                f"【CRAG 相关性校验：证据不足，已触发拒答保护（verdict={self.verdict}，{self.reason}）】\n"
                "当前知识库中未检索到足以回答该问题的可靠依据，请勿编造结论，应直接向用户说明未找到相关记录。"
            )
        if self.verdict == "ambiguous":
            return f"【CRAG 提示】召回片段相关性中等（{self.reason}），回答时请谨慎核对原文细节，不确定处明确说明。"
        return ""


# 按当前白名单工具列表筛选允许推荐的备用检索数据源（严禁推荐已禁用工具）
def _allowed_fallback_sources(available_tools: Sequence[str] | None) -> tuple[str, ...]:
    if available_tools is None:
        return ()
    enabled = {str(t).strip() for t in available_tools if str(t).strip()}
    return tuple(tool for tool in _FALLBACK_TOOL_ORDER if tool in enabled)


# CRAG 三级相关性评估：有 Cross-Encoder 时按精确阈值判定，无重排器时按词面覆盖度启发式（稠密同义命中防误杀）
def evaluate_retrieval_sufficiency(
    query: str,
    hits: Sequence[_HitLike],
    *,
    has_reranker: bool,
    has_dense: bool = False,
    config: PostRetrievalConfig | None = None,
    available_tools: Sequence[str] | None = None,
    allow_rewrite: bool = True,
    attempts: int = 1,
    rewritten_query: str = "",
) -> CRAGAssessment:
    cfg = config or PostRetrievalConfig()
    fallbacks = _allowed_fallback_sources(available_tools)
    max_score = max((float(h.score) for h in hits), default=0.0)

    if not hits:
        action: CRAGAction = (
            "rewrite_retry"
            if allow_rewrite
            else ("switch_source" if fallbacks else "abstain")
        )
        return CRAGAssessment(
            verdict="incorrect",
            action=action,
            max_score=0.0,
            rewritten_query=rewritten_query,
            suggested_sources=fallbacks if action == "switch_source" else (),
            attempts=attempts,
            reason="本地知识库未召回任何匹配片段",
        )

    if has_reranker:
        if max_score >= cfg.crag_high_threshold:
            verdict: CRAGVerdict = "correct"
            reason = f"精排最高分 {max_score:.3f} >= 高置信阈值 {cfg.crag_high_threshold:.2f}"
        elif max_score >= cfg.crag_low_threshold:
            verdict = "ambiguous"
            reason = (
                f"精排最高分 {max_score:.3f} 处于模糊区间 "
                f"[{cfg.crag_low_threshold:.2f}, {cfg.crag_high_threshold:.2f})"
            )
        else:
            verdict = "incorrect"
            reason = f"精排最高分 {max_score:.3f} < 最低相关性阈值 {cfg.crag_low_threshold:.2f}"
    else:
        q_tokens = set(tokenize_text(query.strip(), for_query=True))
        if q_tokens:
            best_cov = 0.0
            for hit in hits[:3]:
                body_tokens = set(tokenize_text(_strip_snippet_body(hit.snippet), for_query=False))
                sec_tokens = set(tokenize_text(hit.section, for_query=False))
                cov = len(q_tokens & (body_tokens | sec_tokens)) / len(q_tokens)
                if cov > best_cov:
                    best_cov = cov
            if best_cov >= 0.60:
                verdict = "correct"
                reason = f"词面关键覆盖度 {best_cov:.2f} >= 0.60"
            elif best_cov >= 0.30 or has_dense:
                verdict = "ambiguous"
                reason = (
                    f"稠密语义召回且词面覆盖度 {best_cov:.2f}（无 Cross-Encoder 精排）"
                    if has_dense and best_cov < 0.30
                    else f"词面关键覆盖度 {best_cov:.2f} 处于中等区间"
                )
            else:
                verdict = "incorrect"
                reason = f"词面关键覆盖度 {best_cov:.2f} < 0.30"
        else:
            verdict = "ambiguous"
            reason = "查询不含可分词关键词，按中等置信度放行"

    if verdict == "correct":
        final_action: CRAGAction = "answer"
    elif verdict == "ambiguous":
        final_action = "rewrite_retry" if allow_rewrite else "answer"
    else:
        final_action = (
            "rewrite_retry"
            if allow_rewrite
            else ("switch_source" if fallbacks else "abstain")
        )

    return CRAGAssessment(
        verdict=verdict,
        action=final_action,
        max_score=max_score,
        rewritten_query=rewritten_query,
        suggested_sources=fallbacks if final_action == "switch_source" else (),
        attempts=attempts,
        reason=reason,
    )


# CRAG 纠错改写：当首轮召回相关性不足时，调用轻量模型（reasoning="off"）生成 1 条更贴近文档措辞的替代检索词
def corrective_rewrite_query(
    query: str,
    hits: Sequence[_HitLike],
    llm: ModelInvoker,
    *,
    abort_signal: AbortSignal | None = None,
) -> str:
    clean_q = " ".join(query.split())
    if not clean_q or isinstance(llm, NoopModelClient):
        return ""
    if abort_signal is not None and abort_signal.aborted:
        raise StreamAbortedError(abort_signal.reason or "CRAG 改写已取消")

    preview_lines: list[str] = []
    for hit in hits[:3]:
        body = _strip_snippet_body(hit.snippet)[:140]
        preview_lines.append(f"- {hit.path} [{hit.section or '正文'}]: {body}")
    preview = "\n".join(preview_lines) if preview_lines else "（无任何命中片段）"

    sys_prompt = (
        "你是检索纠错改写助手（CRAG）。用户上一轮本地知识库检索结果相关性不足或为空。\n"
        "请根据原问题，换用更贴近知识库文档正文的同义词、正式术语或核心关键词组合，生成 1 条更易命中的单行检索查询。\n"
        "规则：\n"
        "1. 绝不编造问题中不存在的日期、编号或实体。\n"
        "2. 用空格分隔 2-6 个核心检索词或输出一条简明短句。\n"
        "3. 只输出改写后的单行查询本身，不要加引号、前缀或解释。"
    )
    user_prompt = f"【原问题】\n{clean_q}\n\n【首轮召回摘要（相关性不足）】\n{preview}"
    provider_api = str(getattr(getattr(llm, "profile", None), "api", "") or "").strip().lower()
    budget = 1024 if provider_api == "openai-reasoning" else 140
    msg = llm.complete(
        Context(system_prompt=sys_prompt, messages=(UserMessage(content=user_prompt),)),
        StreamOptions(temperature=0.0, max_tokens=budget, reasoning="off", abort_signal=abort_signal),
    )
    rewritten = " ".join(msg.text.split()).strip(" \"'“”‘’")[:240].rstrip()
    if not rewritten or rewritten.lower() == clean_q.lower():
        return ""
    return rewritten
