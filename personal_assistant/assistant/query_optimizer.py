"""检索前 Query 侧优化引擎（零第三方依赖）：路由（Routing）、多轮指代改写、多查询扩展/拆解与 HyDE 假设文档嵌入。"""
from __future__ import annotations

import json
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

RouteTarget = Literal["direct", "docs", "chat_history", "web", "study"]
VALID_ROUTES: tuple[RouteTarget, ...] = ("direct", "docs", "chat_history", "web", "study")
_ROUTE_BY_NAME: dict[str, RouteTarget] = {
    "direct": "direct",
    "docs": "docs",
    "chat_history": "chat_history",
    "web": "web",
    "study": "study",
}

ExpansionStrategy = Literal["single", "multi_query", "decompose"]
VALID_STRATEGIES: tuple[ExpansionStrategy, ...] = ("single", "multi_query", "decompose")

DEFAULT_MAX_SUB_QUERIES = 3
MAX_ALLOWED_SUB_QUERIES = 6
MAX_HISTORY_TURNS_FOR_REWRITE = 6
MAX_HISTORY_CHARS_FOR_REWRITE = 2400
MAX_INPUT_QUERY_CHARS = 500
MAX_REWRITTEN_QUERY_CHARS = 300
MAX_HYDE_CHARS = 600

_ROUTE_DESCRIPTIONS: dict[RouteTarget, str] = {
    "direct": "无需检索（如寒暄、纯算术/翻译/通用常识问答）",
    "docs": "本地 Markdown 知识库与长期记忆文档（search_markdown / read_markdown）",
    "chat_history": "本地 SQLite 历史对话记录库（search_chat_history，查找过往聊天原话）",
    "web": "联网实时搜索与外部公开信息（web_search / fetch_webpage）",
    "study": "个人间隔重复学习卡片与到期复习计划（list_due_reviews）",
}


# 校验布尔配置项，严格拒绝字符串或数字隐式转换
def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} 必须是布尔值（true/false）。")
    return value


# 检索前 Query 侧优化开关配置（对应 config/tools.json 的 retrieval.query_optimization 段）
@dataclass(frozen=True)
class QueryOptimizationConfig:
    enabled: bool = False
    routing: bool = True
    rewrite: bool = True
    multi_query: bool = True
    hyde: bool = True
    max_sub_queries: int = DEFAULT_MAX_SUB_QUERIES

    # 校验全部布尔开关与子查询上限范围（1..6）
    def __post_init__(self) -> None:
        _require_bool(self.enabled, "query_optimization.enabled")
        _require_bool(self.routing, "query_optimization.routing")
        _require_bool(self.rewrite, "query_optimization.rewrite")
        _require_bool(self.multi_query, "query_optimization.multi_query")
        _require_bool(self.hyde, "query_optimization.hyde")
        val = require_positive_int(self.max_sub_queries, "query_optimization.max_sub_queries")
        if val > MAX_ALLOWED_SUB_QUERIES:
            raise ValueError(f"query_optimization.max_sub_queries 不能超过 {MAX_ALLOWED_SUB_QUERIES}。")

    # 从配置字典安全解析 QueryOptimizationConfig
    @classmethod
    def from_dict(cls, raw: Mapping[str, object] | None) -> "QueryOptimizationConfig":
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError("retrieval.query_optimization 必须是 JSON 对象。")
        raw_max = raw.get("max_sub_queries", DEFAULT_MAX_SUB_QUERIES)
        if raw_max is None:
            max_sub = DEFAULT_MAX_SUB_QUERIES
        elif isinstance(raw_max, bool) or not isinstance(raw_max, int):
            raise ValueError("query_optimization.max_sub_queries 必须是正整数。")
        else:
            max_sub = raw_max
        return cls(
            enabled=_require_bool(raw.get("enabled", False), "query_optimization.enabled"),
            routing=_require_bool(raw.get("routing", True), "query_optimization.routing"),
            rewrite=_require_bool(raw.get("rewrite", True), "query_optimization.rewrite"),
            multi_query=_require_bool(raw.get("multi_query", True), "query_optimization.multi_query"),
            hyde=_require_bool(raw.get("hyde", True), "query_optimization.hyde"),
            max_sub_queries=max_sub,
        )


# 路由判定结果（标记是否需要检索、目标路由元组及简短理由）
@dataclass(frozen=True)
class RouteDecision:
    routes: tuple[RouteTarget, ...]
    needs_retrieval: bool
    reason: str = ""

    # 导出 JSON 友好字典
    def to_dict(self) -> dict[str, object]:
        return {
            "routes": list(self.routes),
            "needs_retrieval": self.needs_retrieval,
            "reason": self.reason,
        }


# 查询扩展/拆解结果（记录原始查询、去重后的子查询元组及策略）
@dataclass(frozen=True)
class QueryExpansion:
    original: str
    sub_queries: tuple[str, ...]
    strategy: ExpansionStrategy

    # 导出 JSON 友好字典
    def to_dict(self) -> dict[str, object]:
        return {
            "original": self.original,
            "sub_queries": list(self.sub_queries),
            "strategy": self.strategy,
        }


# 完整的检索前 Query 优化计划（聚合改写、路由、扩展/拆解与可选 HyDE 假设文档）
@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    rewritten_query: str
    route: RouteDecision
    expansion: QueryExpansion
    hyde_document: str = ""
    applied: bool = True
    routing_applied: bool = True

    # 导出 JSON 友好字典（供 /rag query 调试与接口返回）
    def to_dict(self) -> dict[str, object]:
        return {
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "route": self.route.to_dict(),
            "expansion": self.expansion.to_dict(),
            "hyde_document": self.hyde_document,
            "applied": self.applied,
            "routing_applied": self.routing_applied,
        }

    # 生成注入当前轮用户消息的紧凑检索建议（透传/降级返回空串；关闭路由时不伪造默认路由行；绝不包含 HyDE 假设正文）
    def format_agent_hint(self) -> str:
        if not self.applied:
            return ""
        lines: list[str] = []
        if self.routing_applied:
            routes_label = "、".join(self.route.routes)
            reason_part = f"（{self.route.reason}）" if self.route.reason else ""
            if not self.route.needs_retrieval:
                lines.append(f"- 检索路由建议：{routes_label} — 无需检索，可直接回答{reason_part}")
            else:
                lines.append(f"- 检索路由建议：{routes_label}{reason_part}")
        if self.route.needs_retrieval:
            if self.rewritten_query != self.original_query:
                lines.append(f"- 指代补全后的独立查询：{self.rewritten_query}")
            if len(self.expansion.sub_queries) > 1:
                subs = "；".join(self.expansion.sub_queries)
                lines.append(
                    f"- 建议子查询（{self.expansion.strategy}，可传入 search_markdown 的 sub_queries 参数或并行检索）：{subs}"
                )
        return "\n".join(lines)


# 根据当前白名单工具名推断本次允许路由的目标集合（direct 始终允许）
def _allowed_routes_for_tools(available_tools: Sequence[str] | None) -> set[RouteTarget]:
    if available_tools is None:
        return set(VALID_ROUTES)
    names = {str(t).strip() for t in available_tools if str(t).strip()}
    allowed: set[RouteTarget] = {"direct"}
    if names & {"search_markdown", "read_markdown", "list_files"}:
        allowed.add("docs")
    if "search_chat_history" in names:
        allowed.add("chat_history")
    if names & {"web_search", "fetch_webpage"}:
        allowed.add("web")
    if names & {"list_due_reviews", "save_study_card"}:
        allowed.add("study")
    return allowed


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


# 从任意文本中按 raw_decode 扫描提取首个合法 JSON 字典对象
def _scan_first_json_dict(text: str) -> dict[str, object] | None:
    decoder = json.JSONDecoder()
    pos = 0
    length = len(text)
    while pos < length:
        brace = text.find("{", pos)
        if brace == -1:
            return None
        try:
            parsed, _ = decoder.raw_decode(text, brace)
        except json.JSONDecodeError:
            pos = brace + 1
            continue
        if isinstance(parsed, dict):
            return parsed
        pos = brace + 1
    return None


# 从模型可能带围栏或前后说明文字的回复中稳健提取 JSON 对象（优先解析 ```json 围栏块，防止前文出现 {} 示例误导）
def _extract_json_object(raw_text: str) -> dict[str, object] | None:
    text = raw_text.strip()
    if not text:
        return None
    for match in _FENCED_JSON_RE.finditer(text):
        fenced = _scan_first_json_dict(match.group(1))
        if fenced is not None:
            return fenced
    return _scan_first_json_dict(text)


# 清洗原始用户查询输入（仅折叠空白并限长，绝不剥离引号以免纯符号消息被洗空）
def _clean_query_input(text: str, *, max_chars: int = MAX_INPUT_QUERY_CHARS) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[:max_chars].rstrip()


# 清洗模型生成的单行查询字符串（折叠空白、剥离首尾装饰性引号并限制最大长度）
def _clean_single_line(text: str, *, max_chars: int) -> str:
    collapsed = " ".join(text.split()).strip(" \"'“”‘’")
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[:max_chars].rstrip()


# 检索命中对象最小结构协议（零依赖 memory.py，兼容 SearchHit 等不可变 dataclass）
class _HitLike(Protocol):
    path: str
    start_line: int
    end_line: int
    section: str
    snippet: str
    score: float


_THit = TypeVar("_THit", bound=_HitLike)


# 将结构化 SearchHit 列表格式化为与 MemoryStore.search_markdown 一致的带行号区间与章节面包屑文本
def format_search_hits(hits: Sequence[_HitLike], *, query: str, root: str = "data/memory") -> str:
    if not hits:
        return f"未在 {root} 下找到匹配「{query}」的 Markdown 内容。"
    return "\n\n".join(
        f"{hit.path}:{hit.start_line if hit.start_line == hit.end_line else f'{hit.start_line}-{hit.end_line}'} "
        f"[章节: {hit.section or '正文'}]\n{hit.snippet}"
        for hit in hits
    )


# 对多条子查询各自的 SearchHit 列表执行通用 RRF 去重融合（供未实现 search_multi_hits 的线性/自定义检索器复用）
def fuse_multi_query_hits(
    hit_lists: Sequence[Sequence[_THit]],
    *,
    limit: int = 10,
    rrf_k: int = 60,
) -> list[_THit]:
    k = require_positive_int(rrf_k, "rrf_k")
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("limit 必须是整数。")
    if limit <= 0:
        return []
    fused: dict[tuple[str, int, int], tuple[float, float, _THit]] = {}
    for hits in hit_lists:
        for rank, hit in enumerate(hits, start=1):
            key = (hit.path, hit.start_line, hit.end_line)
            contrib = 1.0 / (k + rank)
            prev = fused.get(key)
            if prev is None:
                fused[key] = (contrib, hit.score, hit)
            else:
                fused[key] = (prev[0] + contrib, max(prev[1], hit.score), prev[2])
    ordered = sorted(
        fused.values(),
        key=lambda item: (-round(item[0], 6), -item[1], item[2].path, item[2].start_line, item[2].end_line),
    )
    deduped: list[_THit] = []
    used_spans_by_file: dict[str, list[tuple[int, int]]] = {}
    for rrf_val, _, hit in ordered:
        prev_spans = used_spans_by_file.setdefault(hit.path, [])
        if any(hit.start_line <= p_end and hit.end_line >= p_start for p_start, p_end in prev_spans):
            continue
        prev_spans.append((hit.start_line, hit.end_line))
        deduped.append(replace(hit, score=round(rrf_val, 6)))
        if len(deduped) >= limit:
            break
    return deduped


# 将近期对话历史格式化为有界紧凑文本（仅保留最近若干轮 user/assistant，无可用历史返回空串）
def _format_recent_history(
    history: Sequence[Mapping[str, object]] | None,
    *,
    max_turns: int = MAX_HISTORY_TURNS_FOR_REWRITE,
    max_chars: int = MAX_HISTORY_CHARS_FOR_REWRITE,
) -> str:
    if not history:
        return ""
    lines: list[str] = []
    for item in history[-max_turns:]:
        role = str(item.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = " ".join(str(item.get("content") or "").split())
        if not content:
            continue
        if len(content) > 400:
            content = content[:400].rstrip() + "…"
        label = "用户" if role == "user" else "助手"
        lines.append(f"{label}：{content}")
    if not lines:
        return ""
    rendered = "\n".join(lines)
    if len(rendered) <= max_chars:
        return rendered
    return rendered[-max_chars:]


# 统一执行轻量级 LLM 文本调用：显式关闭 reasoning、对 openai-reasoning 放宽输出预算并响应 AbortSignal
def _complete_text(
    llm: ModelInvoker,
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int,
    temperature: float = 0.0,
    abort_signal: AbortSignal | None = None,
) -> str:
    if abort_signal is not None and abort_signal.aborted:
        raise StreamAbortedError(abort_signal.reason or "查询优化已取消")
    provider_api = str(getattr(getattr(llm, "profile", None), "api", "") or "").strip().lower()
    budget = max(max_tokens, 1024) if provider_api == "openai-reasoning" else max_tokens
    ctx = Context(
        system_prompt=system_prompt,
        messages=(UserMessage(content=user_prompt),),
    )
    opts = StreamOptions(
        temperature=temperature,
        max_tokens=budget,
        reasoning="off",
        abort_signal=abort_signal,
    )
    msg = llm.complete(ctx, opts)
    return msg.text


# 规范化并过滤模型输出的路由目标列表，确保结果非空且严格落在当前允许集合内
def _normalize_route_decision(
    parsed: dict[str, object] | None,
    *,
    allowed: set[RouteTarget],
) -> RouteDecision:
    default_target: RouteTarget = "docs" if "docs" in allowed else (
        next(iter(sorted(t for t in allowed if t != "direct")), "direct")
    )
    if parsed is None:
        return RouteDecision(
            routes=(default_target,),
            needs_retrieval=default_target != "direct",
            reason="默认路由",
        )

    raw_reason = parsed.get("reason")
    reason = _clean_single_line(raw_reason, max_chars=160) if isinstance(raw_reason, str) else ""
    raw_needs = parsed.get("needs_retrieval")
    raw_routes = parsed.get("routes")
    picked: list[RouteTarget] = []
    if isinstance(raw_routes, list):
        for item in raw_routes:
            norm = str(item or "").strip().lower()
            typed = _ROUTE_BY_NAME.get(norm)
            if typed is not None and typed in allowed and typed not in picked:
                picked.append(typed)

    # 若同时出现 direct 与实际检索目标，剔除 direct 保留实际检索目标
    non_direct = [r for r in picked if r != "direct"]
    if non_direct:
        return RouteDecision(routes=tuple(non_direct), needs_retrieval=True, reason=reason)
    if raw_needs is False or ("direct" in picked and raw_needs is not True):
        return RouteDecision(routes=("direct",), needs_retrieval=False, reason=reason)
    return RouteDecision(
        routes=(default_target,),
        needs_retrieval=default_target != "direct",
        reason=reason or "默认回退路由",
    )


# 规范化并去重模型输出的扩展/拆解子查询列表，保证策略合法且至少包含 1 条有效查询
def _normalize_expansion(
    base_query: str,
    parsed: dict[str, object] | None,
    *,
    max_sub_queries: int,
) -> QueryExpansion:
    if parsed is None or max_sub_queries <= 1:
        return QueryExpansion(original=base_query, sub_queries=(base_query,), strategy="single")

    raw_strategy = str(parsed.get("strategy") or "single").strip().lower()
    if raw_strategy == "decompose":
        strategy: ExpansionStrategy = "decompose"
    elif raw_strategy == "multi_query":
        strategy = "multi_query"
    else:
        strategy = "single"

    raw_queries = parsed.get("queries")
    cleaned_list: list[str] = []
    seen_lower: set[str] = set()
    if isinstance(raw_queries, list):
        for item in raw_queries:
            if not isinstance(item, str):
                continue
            q = _clean_single_line(item, max_chars=MAX_REWRITTEN_QUERY_CHARS)
            if q and q.lower() not in seen_lower:
                seen_lower.add(q.lower())
                cleaned_list.append(q)

    if strategy == "single" or not cleaned_list:
        return QueryExpansion(original=base_query, sub_queries=(base_query,), strategy="single")

    if strategy == "multi_query":
        # 同义扩展始终把原句放首位，防止原专有名词或编号被改写丢掉
        final_list: list[str] = [base_query]
        base_low = base_query.lower()
        for q in cleaned_list:
            if q.lower() != base_low:
                final_list.append(q)
            if len(final_list) >= max_sub_queries:
                break
        final_strategy: ExpansionStrategy = "multi_query" if len(final_list) > 1 else "single"
        return QueryExpansion(original=base_query, sub_queries=tuple(final_list), strategy=final_strategy)

    # decompose：仅在确实拆出 >=2 条独立子问题时采纳；若只返回 1 条则回退完整原句，防止丢弃复合问题的后半部分
    trimmed = cleaned_list[:max_sub_queries]
    if len(trimmed) <= 1:
        return QueryExpansion(original=base_query, sub_queries=(base_query,), strategy="single")
    return QueryExpansion(original=base_query, sub_queries=tuple(trimmed), strategy="decompose")


# 原子能力 1（Query Rewriting）：结合近期多轮对话历史，将省略与代词指代补全为独立完整的检索句（无历史零调用）
def rewrite_query(
    query: str,
    history: Sequence[Mapping[str, object]] | None,
    llm: ModelInvoker,
    *,
    abort_signal: AbortSignal | None = None,
) -> str:
    clean_q = _clean_query_input(query)
    if not clean_q:
        raise ValueError("待改写的查询不能为空。")
    hist_text = _format_recent_history(history)
    if not hist_text or isinstance(llm, NoopModelClient):
        return clean_q

    sys_prompt = (
        "你是检索查询改写助手。请结合下方近期对话历史，将用户最新提问中的省略、代词指代"
        "（如“它”“这个”“那个方案”“上次说的”）补全为一条独立、完整、可直接用于检索的简明查询。\n"
        "规则：\n"
        "1. 严格忠实于历史上下文，绝不编造历史中未出现的实体、数字或事实。\n"
        "2. 保留原问题中的所有专有名词、版本号、端口号、文件名与日期。\n"
        "3. 如果原问题本身已经完整独立，直接原样输出。\n"
        "4. 只输出改写后的单行查询本身，不要加引号、前缀或解释。"
    )
    user_prompt = f"【近期对话历史】\n{hist_text}\n\n【用户最新提问】\n{clean_q}"
    raw = _complete_text(llm, sys_prompt, user_prompt, max_tokens=160, temperature=0.0, abort_signal=abort_signal)
    rewritten = _clean_single_line(raw, max_chars=MAX_REWRITTEN_QUERY_CHARS)
    return rewritten or clean_q


# 原子能力 2（Routing）：判断问题是否需要检索，并路由至 direct / docs / chat_history / web / study 或多路组合
def route_query(
    query: str,
    llm: ModelInvoker,
    *,
    available_tools: Sequence[str] | None = None,
    history: Sequence[Mapping[str, object]] | None = None,
    abort_signal: AbortSignal | None = None,
) -> RouteDecision:
    clean_q = _clean_query_input(query)
    if not clean_q:
        raise ValueError("待路由的查询不能为空。")
    allowed = _allowed_routes_for_tools(available_tools)
    if isinstance(llm, NoopModelClient):
        return _normalize_route_decision(None, allowed=allowed)

    route_menu = "\n".join(f'- "{r}": {_ROUTE_DESCRIPTIONS[r]}' for r in VALID_ROUTES if r in allowed)
    sys_prompt = (
        "你是个人助手的检索路由器。请根据用户问题判断：是否需要检索，以及应路由到哪些数据源。\n"
        f"可选路由：\n{route_menu}\n"
        "仅输出单个 JSON 对象，格式严格为：\n"
        '{"routes": ["docs"], "needs_retrieval": true, "reason": "简短中文原因"}\n'
        "若不需要任何检索，输出 routes=[\"direct\"] 且 needs_retrieval=false；若需跨源对比可输出多个目标。"
    )
    hist_text = _format_recent_history(history, max_turns=4, max_chars=1200)
    user_prompt = (
        f"【近期对话】\n{hist_text}\n\n【当前问题】\n{clean_q}"
        if hist_text
        else f"【当前问题】\n{clean_q}"
    )
    raw = _complete_text(llm, sys_prompt, user_prompt, max_tokens=180, temperature=0.0, abort_signal=abort_signal)
    return _normalize_route_decision(_extract_json_object(raw), allowed=allowed)


# 原子能力 3（Multi-Query / Decomposition）：识别复合问题并拆解为子问题，或为单一问题扩写同义表述
def expand_and_decompose_query(
    query: str,
    llm: ModelInvoker,
    *,
    max_sub_queries: int = DEFAULT_MAX_SUB_QUERIES,
    abort_signal: AbortSignal | None = None,
) -> QueryExpansion:
    clean_q = _clean_query_input(query)
    if not clean_q:
        raise ValueError("待扩展的查询不能为空。")
    limit = require_positive_int(max_sub_queries, "max_sub_queries")
    if limit > MAX_ALLOWED_SUB_QUERIES:
        raise ValueError(f"max_sub_queries 不能超过 {MAX_ALLOWED_SUB_QUERIES}。")
    if limit == 1 or isinstance(llm, NoopModelClient):
        return QueryExpansion(original=clean_q, sub_queries=(clean_q,), strategy="single")

    sys_prompt = (
        f"你是检索查询优化专家。请分析用户问题，并最多生成 {limit} 条子查询，严格输出单个 JSON 对象：\n"
        '{"strategy": "single" | "multi_query" | "decompose", "queries": ["..."]}\n'
        "策略判定规则：\n"
        "- decompose：问题包含多个并列实体、对比项或多跳子问题时，拆解为各自独立的子查询。\n"
        "- multi_query：单一主题但可能存在同义词、中英文术语或不同表达方式时，扩写 2-3 种同义检索表述。\n"
        "- single：已经非常精确的单个短查询无需扩展。"
    )
    raw = _complete_text(
        llm,
        sys_prompt,
        f"【待优化查询】\n{clean_q}",
        max_tokens=220,
        temperature=0.0,
        abort_signal=abort_signal,
    )
    return _normalize_expansion(clean_q, _extract_json_object(raw), max_sub_queries=limit)


# 原子能力 4（HyDE 假设性文档嵌入）：生成一段简洁的假设性答案片段，供语义向量检索对齐“答案-答案”空间
def generate_hyde_document(
    query: str,
    llm: ModelInvoker,
    *,
    abort_signal: AbortSignal | None = None,
) -> str:
    clean_q = _clean_query_input(query)
    if not clean_q:
        raise ValueError("生成 HyDE 假设文档的查询不能为空。")
    if isinstance(llm, NoopModelClient):
        return ""

    sys_prompt = (
        "你是检索增强生成的 HyDE（假设性文档嵌入）辅助生成器。\n"
        "请针对用户问题，直接用陈述句撰写一段 2-4 句、150 字以内的假设性知识库正文片段，"
        "模拟“真正记录答案的文档段落”的措辞与专业术语（仅供向量相似度检索，不会直接展示给用户）。\n"
        "要求：单段输出，不要加标题、免责声明或问答格式。"
    )
    raw = _complete_text(
        llm,
        sys_prompt,
        f"【用户问题】\n{clean_q}",
        max_tokens=220,
        temperature=0.2,
        abort_signal=abort_signal,
    )
    return _clean_single_line(raw, max_chars=MAX_HYDE_CHARS)


# 统一 Query 优化编排器：单次合并调用完成路由 + 改写 + 扩展/拆解，仅在 docs + embedder 就绪时按需执行 HyDE
class QueryOptimizer:
    # 绑定 QueryOptimizationConfig 开关配置
    def __init__(self, config: QueryOptimizationConfig | None = None) -> None:
        self.config = config or QueryOptimizationConfig()

    # 构造零调用透传计划（用于关闭开关、未配置模型或非取消故障优雅降级）
    @staticmethod
    def _passthrough(query: str, *, allowed: set[RouteTarget], reason: str) -> QueryPlan:
        default_route: RouteTarget = "docs" if "docs" in allowed else (
            next(iter(sorted(t for t in allowed if t != "direct")), "direct")
        )
        return QueryPlan(
            original_query=query,
            rewritten_query=query,
            route=RouteDecision(routes=(default_route,), needs_retrieval=default_route != "direct", reason=reason),
            expansion=QueryExpansion(original=query, sub_queries=(query,), strategy="single"),
            hyde_document="",
            applied=False,
            routing_applied=False,
        )

    # 执行完整检索前优化计划（单次合并 JSON 调用降低延迟；AbortSignal 必穿透，其余瞬时故障优雅回退透传）
    def plan(
        self,
        query: str,
        llm: ModelInvoker,
        *,
        history: Sequence[Mapping[str, object]] | None = None,
        available_tools: Sequence[str] | None = None,
        has_embedder: bool = False,
        force_enable: bool = False,
        abort_signal: AbortSignal | None = None,
    ) -> QueryPlan:
        clean_q = _clean_query_input(query)
        if not clean_q:
            raise ValueError("待优化的查询不能为空。")
        if abort_signal is not None and abort_signal.aborted:
            raise StreamAbortedError(abort_signal.reason or "查询优化已取消")

        allowed = _allowed_routes_for_tools(available_tools)
        if (not self.config.enabled and not force_enable) or isinstance(llm, NoopModelClient):
            return self._passthrough(clean_q, allowed=allowed, reason="passthrough")

        hist_text = _format_recent_history(history) if self.config.rewrite else ""
        do_routing = self.config.routing
        do_rewrite = self.config.rewrite and bool(hist_text)
        do_expand = self.config.multi_query and self.config.max_sub_queries > 1

        rewritten = clean_q
        route_dec = _normalize_route_decision(None, allowed=allowed)
        expansion = QueryExpansion(original=clean_q, sub_queries=(clean_q,), strategy="single")
        plan_called = False
        routing_applied = False

        if do_routing or do_rewrite or do_expand:
            plan_called = True
            route_menu = "\n".join(f'- "{r}": {_ROUTE_DESCRIPTIONS[r]}' for r in VALID_ROUTES if r in allowed)
            tasks: list[str] = []
            schema_fields: list[str] = []
            if do_routing:
                tasks.append(f"1. 路由判定：从以下可用目标中选出所需路由：\n{route_menu}")
                schema_fields.append('"routes": ["docs"], "needs_retrieval": true, "reason": "简短原因"')
            if do_rewrite:
                tasks.append("2. 多轮指代改写：结合对话历史，把代词与省略补全为独立完整的检索句（严禁编造）。")
                schema_fields.append('"rewritten_query": "补全后的独立查询"')
            if do_expand:
                tasks.append(
                    f"3. 扩展/拆解（最多 {self.config.max_sub_queries} 条）：复合对比用 decompose，同义改写用 multi_query，精确单查用 single；"
                    "若无需检索（direct）则 strategy 为 single。"
                )
                schema_fields.append('"strategy": "single"|"multi_query"|"decompose", "queries": ["..."]')

            sys_prompt = (
                "你是检索前查询优化器。请一次性完成以下分析，仅输出单个严格 JSON 对象：\n"
                + "\n".join(tasks)
                + "\nJSON 模板：{"
                + ", ".join(schema_fields)
                + "}"
            )
            user_prompt = (
                f"【近期对话历史】\n{hist_text}\n\n【用户最新提问】\n{clean_q}"
                if hist_text
                else f"【用户提问】\n{clean_q}"
            )
            try:
                raw = _complete_text(
                    llm,
                    sys_prompt,
                    user_prompt,
                    max_tokens=320,
                    temperature=0.0,
                    abort_signal=abort_signal,
                )
            except StreamAbortedError:
                raise
            except Exception:  # noqa: BLE001
                return self._passthrough(clean_q, allowed=allowed, reason="fallback")

            parsed = _extract_json_object(raw)
            if parsed is None:
                return self._passthrough(clean_q, allowed=allowed, reason="fallback")
            if do_routing:
                route_dec = _normalize_route_decision(parsed, allowed=allowed)
                routing_applied = True
            if do_rewrite and isinstance(parsed.get("rewritten_query"), str):
                cand = _clean_single_line(str(parsed["rewritten_query"]), max_chars=MAX_REWRITTEN_QUERY_CHARS)
                if cand:
                    rewritten = cand
            if not route_dec.needs_retrieval:
                expansion = QueryExpansion(original=rewritten, sub_queries=(rewritten,), strategy="single")
            elif do_expand:
                expansion = _normalize_expansion(rewritten, parsed, max_sub_queries=self.config.max_sub_queries)
            else:
                expansion = QueryExpansion(original=rewritten, sub_queries=(rewritten,), strategy="single")

        # 仅在路由命中 docs、配置开启 hyde、系统激活稠密向量且非复合拆解（避免单一混合 HyDE 稀释各独立子问题）时生成 HyDE
        hyde_doc = ""
        if (
            self.config.hyde
            and has_embedder
            and route_dec.needs_retrieval
            and "docs" in route_dec.routes
            and expansion.strategy != "decompose"
        ):
            try:
                hyde_doc = generate_hyde_document(rewritten, llm, abort_signal=abort_signal)
            except StreamAbortedError:
                raise
            except Exception:  # noqa: BLE001
                hyde_doc = ""

        if not plan_called and not hyde_doc:
            return self._passthrough(clean_q, allowed=allowed, reason="passthrough")

        return QueryPlan(
            original_query=clean_q,
            rewritten_query=rewritten,
            route=route_dec,
            expansion=expansion,
            hyde_document=hyde_doc,
            applied=True,
            routing_applied=routing_applied,
        )
