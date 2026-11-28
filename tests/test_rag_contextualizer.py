"""离线 LLM 上下文摘要器与父子索引联调测试（TDD）：验证 Prompt Caching 前缀一致、缓存复用、取消保断点与 CLI /rag 命令。"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from personal_assistant.ai import (
    AbortController,
    AssistantMessage,
    Context,
    StreamAbortedError,
    StreamDone,
    StreamOptions,
    TokenUsage,
    UserMessage,
)
from personal_assistant.assistant.app import AssistantApp
from personal_assistant.assistant.chunker import chunk_markdown_hierarchy
from personal_assistant.assistant.cli import _handle_repl_command
from personal_assistant.assistant.memory import MemoryStore
from personal_assistant.assistant.rag_contextualizer import generate_document_contexts


# 录制每次 LLM 调用的 Context 与 StreamOptions，支持按调用序号注入预设摘要或触发协同取消
class _SpyContextLLM:
    last_usage: TokenUsage | None = None

    # 初始化预设回复序列与可选取消钩子
    def __init__(self, replies: list[str], *, abort_after: int | None = None, controller: AbortController | None = None) -> None:
        self._replies = list(replies)
        self._abort_after = abort_after
        self._controller = controller
        self.calls: list[tuple[Context, StreamOptions | None]] = []

    # 流式接口（满足 ModelInvoker 协议）
    def stream(self, context: Context, options: StreamOptions | None = None):
        msg = self.complete(context, options)
        yield StreamDone(reason="stop", message=msg)

    # 同步接口：检查取消信号、记录调用入参并返回预设背景摘要
    def complete(self, context: Context, options: StreamOptions | None = None) -> AssistantMessage:
        if options is not None and options.abort_signal is not None and options.abort_signal.aborted:
            raise StreamAbortedError(options.abort_signal.reason or "aborted")
        self.calls.append((context, options))
        reply = self._replies.pop(0) if self._replies else "默认背景说明。"
        if self._abort_after is not None and len(self.calls) == self._abort_after and self._controller is not None:
            self._controller.abort("用户中止离线摘要")
        return AssistantMessage.from_text(reply, stop_reason="stop")


# 测试同篇文档的多个父块共享逐字节一致的 system_prompt（最大化 Prompt Caching 命中）并清洗输出格式
def test_generate_document_contexts_preserves_stable_prompt_cache_prefix() -> None:
    doc_text = (
        "# A 公司 2025 Q3 财报\n\n"
        "## 财务业绩\n"
        "本季度营收同比增长 3%，主要受新产品线拉动。\n\n"
        "## 研发投入\n"
        "该部门本季度研发费用较去年同期增加 12%。\n"
    )
    parents, _ = chunk_markdown_hierarchy("data/memory/q3.md", doc_text, parent_max_chars=500)
    assert len(parents) == 2

    spy = _SpyContextLLM(
        [
            "  这段内容摘自 A 公司 2025 年 Q3 财报财务业绩部分，\n说明营收增长原因。  ",
            "这段内容摘自 A 公司 2025 年 Q3 财报研发投入部分，说明研发费用变化。",
        ]
    )
    summaries = list(generate_document_contexts("data/memory/q3.md", doc_text, parents, spy))
    assert len(summaries) == 2
    assert len(spy.calls) == 2

    # 核心 Prompt Caching 不变量：同文档所有父块的 system_prompt 必须逐字节完全一致，且携带 cache_retention
    sys_0 = spy.calls[0][0].system_prompt
    sys_1 = spy.calls[1][0].system_prompt
    assert sys_0 == sys_1
    assert "A 公司 2025 Q3 财报" in sys_0
    assert spy.calls[0][1] is not None and spy.calls[0][1].cache_retention == "short"
    assert spy.calls[0][1].session_id == "rag-ctx:data/memory/q3.md"

    # 输出自动折叠换行并去除多余空白
    assert summaries[0][0] == parents[0].content_hash
    assert "\n" not in summaries[0][1]
    assert "A 公司 2025 年 Q3 财报" in summaries[0][1]


# 测试 RAGRetriever.enrich_contexts：通过 LLM 背景摘要消除代词指代盲区、按 (content_hash, model) 缓存零重算、AbortSignal 取消保断点
def test_rag_enrich_contexts_pronoun_recall_caching_and_abort_resume(tmp_path: Path) -> None:
    from plugins.rag_retriever import RAGConfig, RAGRetriever

    data_dir = tmp_path / "data"
    mem_dir = data_dir / "memory"
    mem_dir.mkdir(parents=True)
    report = mem_dir / "review.md"
    report.write_text(
        "# 季度复盘\n\n"
        "## 业绩\n"
        "该公司本季度营收同比增长 3%，主要受新产品线拉动。\n\n"
        "## 展望\n"
        "预计该公司下季度毛利率保持稳定。\n",
        encoding="utf-8",
    )

    retriever = RAGRetriever(RAGConfig(parent_max_chars=500, child_max_chars=240, child_overlap_chars=40))
    memory = MemoryStore(data_dir, create_dirs=True, retriever=retriever)
    try:
        # 1. 补全上下文前：原文仅写“该公司”，不含“星河科技”“2025 财报”，直接按主体检索无法命中
        assert memory.search_hits("星河科技 2025 财报") == []

        # 2. 离线补全第 1 段后中途触发 AbortSignal：验证第 1 段已持久化落盘，第 2 段留待续跑
        ctrl = AbortController()
        spy_abort = _SpyContextLLM(
            ["该段来自星河科技 2025 年财报业绩章节，说明星河科技营收增长原因。"],
            abort_after=1,
            controller=ctrl,
        )
        with pytest.raises(StreamAbortedError):
            retriever.enrich_contexts(
                memory,
                spy_abort,
                root="data/memory",
                model_name="mini-fast",
                abort_signal=ctrl.signal,
            )
        assert len(spy_abort.calls) == 1

        # 第 1 段的背景摘要已写入 chunk_contexts 并立刻可用于检索命中！
        hits_after_partial = memory.search_hits("星河科技 2025 财报")
        assert hits_after_partial
        assert "营收同比增长 3%" in hits_after_partial[0].snippet
        assert "[背景:" in hits_after_partial[0].snippet
        assert "星河科技" in hits_after_partial[0].snippet

        # 3. 再次执行 enrich_contexts：第 1 段直接命中缓存，只为剩余第 2 段调 1 次模型
        spy_resume = _SpyContextLLM(["该段来自星河科技 2025 年财报展望章节，预测下季度毛利率稳定。"])
        stats = retriever.enrich_contexts(memory, spy_resume, root="data/memory", model_name="mini-fast")
        assert stats["enriched"] == 1
        assert stats["cached_hits"] == 1
        assert len(spy_resume.calls) == 1

        # 4. 第三次执行：全部命中缓存，0 次 LLM 调用
        spy_zero = _SpyContextLLM([])
        stats_zero = retriever.enrich_contexts(memory, spy_zero, root="data/memory", model_name="mini-fast")
        assert stats_zero["enriched"] == 0 and stats_zero["cached_hits"] == 2
        assert len(spy_zero.calls) == 0
    finally:
        memory.close()

    # 5. 调整 child_overlap_chars 触发派生索引重建时，chunk_contexts 缓存表必须保留不丢失
    retriever_v2 = RAGRetriever(RAGConfig(parent_max_chars=500, child_max_chars=240, child_overlap_chars=80))
    memory_v2 = MemoryStore(data_dir, create_dirs=True, retriever=retriever_v2)
    try:
        db_path = data_dir / "state" / "rag_index.db"
        # 触发重建后立刻检查 chunk_contexts 仍保留已付费缓存
        memory_v2.search_hits("毛利率")
        with sqlite3.connect(db_path) as conn:
            cnt = conn.execute("SELECT COUNT(*) FROM chunk_contexts").fetchone()[0]
            assert cnt == 2
    finally:
        memory_v2.close()


# 测试 CLI /rag status 与 /rag enrich 命令接入（复用后台 consolidator 模型，未配置模型时友好提示）
def test_cli_rag_status_and_enrich_commands(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = tmp_path / "app_data"
    (data_dir / "memory").mkdir(parents=True)
    (data_dir / "memory" / "note.md").write_text(
        "# 笔记\n\n## 摘要测试\n该方案将延迟从 200ms 降至 15ms。\n",
        encoding="utf-8",
    )

    spy = _SpyContextLLM(["本段记录了 MyHelper 检索优化方案将平均延迟从 200ms 降至 15ms 的结果。"])
    app = AssistantApp(
        data_root=data_dir,
        config_path=repo_root / "config" / "tools.json",
        project_root=repo_root,
        llm=spy,
        consolidator_llm=spy,
        auto_consolidate=False,
    )
    try:
        st_out = _handle_repl_command(app, "/rag status")
        assert st_out is not None
        assert '"parent_chunks": 1' in st_out
        assert '"child_chunks": 1' in st_out

        en_out = _handle_repl_command(app, "/rag enrich data/memory")
        assert en_out is not None
        assert "新生成 1" in en_out
        assert len(spy.calls) == 1

        # 离线补全后立即能通过背景实体词搜中
        search_out = app.search("MyHelper 检索优化 延迟")
        assert "200ms 降至 15ms" in search_out
    finally:
        app.close()
