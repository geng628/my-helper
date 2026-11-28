"""离线 RAG 上下文补全器（Contextual Retrieval）：基于同文档逐字节一致的 Prompt Caching 前缀为父块生成简短背景说明。"""
from __future__ import annotations

from collections.abc import Iterator, Sequence

from personal_assistant.ai import (
    AbortSignal,
    Context,
    ModelInvoker,
    StreamAbortedError,
    StreamOptions,
    UserMessage,
)
from personal_assistant.assistant.chunker import ParentChunk

DEFAULT_MAX_DOC_CHARS = 24_000
MAX_CONTEXT_SUMMARY_CHARS = 240

_SYSTEM_INSTRUCTION = (
    "你是一个严谨的知识库索引助手。下方给出了一篇完整文档（<document>）。\n"
    "你的唯一任务是：针对用户提供的某一个文档片段（<chunk>），结合整篇文档背景，"
    "用 1 到 2 句简洁客观的中文说明该片段在整篇文档中的上下文定位（如所属主体、时间范围、核心主题），"
    "消除“该公司”“该方案”“本季度”“如上表所示”等脱离原文后的指代不明。\n"
    "要求：\n"
    "1. 只输出 1-2 句背景说明本身，不要添加标题、编号、引号或多余解释。\n"
    "2. 严格忠实于给定文档，不要编造文档中不存在的事实。\n"
    "3. 控制在 120 字以内，单行输出。"
)


# 清洗模型生成的单条背景摘要：折叠换行为单行并截断至安全上限
def _clean_summary(raw_text: str) -> str:
    collapsed = " ".join(raw_text.split()).strip(" \"'“”‘’")
    if len(collapsed) <= MAX_CONTEXT_SUMMARY_CHARS:
        return collapsed
    return collapsed[:MAX_CONTEXT_SUMMARY_CHARS].rstrip()


# 构造单篇文档共享的静态系统提示词（同文档所有父块逐字节完全一致，最大化 KV Cache / Prompt Caching 命中）
def _build_document_system_prompt(path: str, document_text: str, *, max_doc_chars: int) -> str:
    clipped = document_text
    if len(clipped) > max_doc_chars:
        clipped = f"{clipped[:max_doc_chars].rstrip()}\n…（文档较长，后文已截断）"
    return f'{_SYSTEM_INSTRUCTION}\n\n<document path="{path}">\n{clipped}\n</document>'


# 构造单个父块的最小用户提示词（仅携带章节、行号与片段正文，不污染静态前缀）
def _build_chunk_user_prompt(parent: ParentChunk) -> str:
    section_attr = f' section="{parent.section}"' if parent.section else ""
    body = f"{parent.table_header}\n{parent.text}" if parent.table_header else parent.text
    return (
        f"请结合完整文档，为下方片段生成 1-2 句单行背景说明：\n"
        f'<chunk lines="{parent.start_line}-{parent.end_line}"{section_attr}>\n'
        f"{body}\n"
        "</chunk>"
    )


# 为单篇文档的候选常驻父块逐条流式产出 (content_hash, summary)，支持协同取消与 Prompt Caching 复用
def generate_document_contexts(
    path: str,
    document_text: str,
    parents: Sequence[ParentChunk],
    llm: ModelInvoker,
    *,
    max_doc_chars: int = DEFAULT_MAX_DOC_CHARS,
    abort_signal: AbortSignal | None = None,
) -> Iterator[tuple[str, str]]:
    if isinstance(max_doc_chars, bool) or not isinstance(max_doc_chars, int) or max_doc_chars < 1:
        raise ValueError("max_doc_chars 必须是正整数。")

    targets = [p for p in parents if not p.until_date and p.kind != "expiring"]
    if not targets:
        return

    system_prompt = _build_document_system_prompt(path, document_text, max_doc_chars=max_doc_chars)
    session_id = f"rag-ctx:{path}"

    for parent in targets:
        if abort_signal is not None and abort_signal.aborted:
            raise StreamAbortedError(abort_signal.reason or "离线上下文摘要已取消")
        ctx = Context(
            system_prompt=system_prompt,
            messages=(UserMessage(content=_build_chunk_user_prompt(parent)),),
        )
        opts = StreamOptions(
            temperature=0.0,
            max_tokens=160,
            cache_retention="short",
            session_id=session_id,
            abort_signal=abort_signal,
        )
        response = llm.complete(ctx, opts)
        summary = _clean_summary(response.text)
        if summary:
            yield parent.content_hash, summary
