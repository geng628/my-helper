"""离线 LLM 上下文摘要器与父子索引联调测试（TDD）：验证 Prompt Caching 前缀一致、缓存复用、取消保断点与 CLI /rag 命令。"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from personal_assistant.ai import (
    AbortController,
    AssistantMessage,
    Context,
    NoopModelClient,
    StreamAbortedError,
    StreamDone,
    StreamOptions,
    TokenUsage,
)
from personal_assistant.assistant.app import AssistantApp
from personal_assistant.assistant.chunker import chunk_markdown_hierarchy
from personal_assistant.assistant.cli import _handle_repl_command, run_repl
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

    # 核心 Prompt Caching 不变量：同文档所有父块的 system_prompt 必须逐字节完全一致，且显式关闭 reasoning 并携带 cache_retention
    sys_0 = spy.calls[0][0].system_prompt
    sys_1 = spy.calls[1][0].system_prompt
    assert sys_0 == sys_1
    assert "A 公司 2025 Q3 财报" in sys_0
    assert spy.calls[0][1] is not None and spy.calls[0][1].cache_retention == "short"
    assert spy.calls[0][1].reasoning == "off"
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

    # 5. 调整 child_overlap_chars 触发派生索引重建时，chunk_contexts 不仅保留且新检索立即可复用该背景摘要
    retriever_v2 = RAGRetriever(RAGConfig(parent_max_chars=500, child_max_chars=240, child_overlap_chars=80))
    memory_v2 = MemoryStore(data_dir, create_dirs=True, retriever=retriever_v2)
    try:
        db_path = data_dir / "state" / "rag_index.db"
        rebuilt_hits = memory_v2.search_hits("星河科技 2025 财报")
        assert rebuilt_hits and "营收同比增长 3%" in rebuilt_hits[0].snippet and "[背景:" in rebuilt_hits[0].snippet
        with sqlite3.connect(db_path) as conn:
            cnt = conn.execute("SELECT COUNT(*) FROM chunk_contexts").fetchone()[0]
            assert cnt == 2
    finally:
        memory_v2.close()


# 测试 enrich_contexts 防过期泄漏、重复 content_hash 统计恒等、空白摘要、min_doc_chars、只读拦截、并发修改与损坏自愈续跑
def test_rag_enrich_contexts_expiry_isolation_duplicate_hashes_and_resilience(tmp_path: Path) -> None:
    from plugins.rag_retriever import RAGConfig, RAGRetriever

    data_dir = tmp_path / "data"
    mem_dir = data_dir / "memory"
    mem_dir.mkdir(parents=True)

    # 1. 记忆文档含常驻正文 + 多行过期敏感项，验证过期项绝不进入 <document> 提示词也绝不写入 chunk_contexts
    (mem_dir / "a_mem.md").write_text(
        "# 架构\n\n"
        "## 正文\n"
        "该服务通过派生索引加速检索。\n\n"
        "- [2026-10-01] 临时故障口令 EXPIRED_SECRET_777 (until: 2026-10-15)\n"
        "  内部令牌 SECRET_SUBLINE_888\n",
        encoding="utf-8",
    )
    # 2. 同文件内同名标题下两段完全相同的正文（content_hash 相同），以及短文档
    (mem_dir / "b_dup.md").write_text(
        "# 重复测试\n\n"
        "## 小节\n"
        "完全相同的常驻正文用于校验重复哈希统计恒等。\n\n"
        "## 小节\n"
        "完全相同的常驻正文用于校验重复哈希统计恒等。\n",
        encoding="utf-8",
    )
    (mem_dir / "c_short.md").write_text("# 短\n短句。\n", encoding="utf-8")

    # 只读模式显式拒绝 enrich_contexts
    ro_ret = RAGRetriever(RAGConfig(parent_max_chars=500, child_max_chars=240, child_overlap_chars=40))
    ro_mem = MemoryStore(data_dir, create_dirs=False, retriever=ro_ret)
    try:
        with pytest.raises(ValueError, match="只读"):
            ro_ret.enrich_contexts(ro_mem, _SpyContextLLM([]), root="data/memory")
    finally:
        ro_mem.close()

    retriever = RAGRetriever(RAGConfig(parent_max_chars=500, child_max_chars=240, child_overlap_chars=40))
    memory = MemoryStore(data_dir, create_dirs=True, retriever=retriever)
    try:
        # 触发 min_doc_chars 过滤短文档 c_short.md，同时验证 a_mem 过期敏感项绝不外发至 LLM、b_dup 重复哈希仅调 1 次 LLM 且统计恒等
        spy = _SpyContextLLM(["该段来自架构检索优化章节。", ""])
        stats1 = retriever.enrich_contexts(memory, spy, root="data/memory", model_name="m1", min_doc_chars=20)
        assert stats1["files"] == 3
        assert stats1["parents_total"] == stats1["enriched"] + stats1["cached_hits"] + stats1["skipped"]
        assert stats1["enriched"] == 1
        assert stats1["skipped"] == 3  # b_dup 2 个父块（空摘要） + c_short 1 个父块（短文档跳过）
        assert len(spy.calls) == 2
        assert "EXPIRED_SECRET_777" not in spy.calls[0][0].system_prompt
        assert "SECRET_SUBLINE_888" not in spy.calls[0][0].system_prompt

        # 第二次为 b_dup 提供有效摘要：2 个同哈希父块只调 1 次 LLM，enriched=1 且同文件复用 cached_hits=1，总和仍恒等
        spy2 = _SpyContextLLM(["该段验证同文件重复父块共享缓存摘要。"])
        stats2 = retriever.enrich_contexts(memory, spy2, root="data/memory", model_name="m1", min_doc_chars=20)
        assert len(spy2.calls) == 1
        assert stats2["parents_total"] == 4
        assert stats2["enriched"] == 1
        assert stats2["cached_hits"] == 2  # a_mem 1 条已有缓存 + b_dup 第 2 块同哈希复用
        assert stats2["skipped"] == 1  # c_short
        assert stats2["parents_total"] == stats2["enriched"] + stats2["cached_hits"] + stats2["skipped"]

        # 第三次模拟锁外期间文件 b_dup.md 发生并发修改 + 外部删库重建，验证不静默漏跑且新内容立即被摘要
        db_path = data_dir / "state" / "rag_index.db"
        (mem_dir / "b_dup.md").write_text("# 重复测试\n\n## 小节\n并发更新后的全新正文内容 V2_FRESH。\n", encoding="utf-8")
        db_path.unlink()
        spy3 = _SpyContextLLM(["并发更新后 V2_FRESH 的新背景摘要。", "架构检索优化重跑摘要。"])
        stats3 = retriever.enrich_contexts(memory, spy3, root="data/memory", model_name="m1", min_doc_chars=20)
        assert stats3["enriched"] >= 1
        assert any("V2_FRESH" in call[0].system_prompt for call in spy3.calls)
        hits_v2 = memory.search_hits("V2_FRESH 背景摘要")
        assert hits_v2 and "V2_FRESH" in hits_v2[0].snippet
    finally:
        memory.close()


# 测试 CLI /rag status 与 /rag enrich 命令接入与全面容错（未配模型、越界目录、多余参数、只读、中断与 REPL 永不崩溃）
def test_cli_rag_status_and_enrich_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
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
        # 支持默认 /rag、空格与制表符分隔
        st_out = _handle_repl_command(app, "/rag\tstatus")
        assert st_out is not None
        assert '"parent_chunks": 1' in st_out
        assert '"child_chunks": 1' in st_out

        # 非法参数/越界路径/不存在目录均返回友好提示而不抛未捕获异常
        bad_extra = _handle_repl_command(app, "/rag status data/memory extra")
        assert bad_extra is not None and "用法" in bad_extra
        bad_outside = _handle_repl_command(app, "/rag status ../outside")
        assert bad_outside is not None and "失败" in bad_outside
        bad_enrich_path = _handle_repl_command(app, "/rag enrich ../outside")
        assert bad_enrich_path is not None and "失败" in bad_enrich_path

        en_out = _handle_repl_command(app, "/rag enrich data/memory")
        assert en_out is not None
        assert "新生成 1" in en_out
        assert len(spy.calls) == 1

        # 离线补全后立即能通过背景实体词搜中
        search_out = app.search("MyHelper 检索优化 延迟")
        assert "200ms 降至 15ms" in search_out

        # 未配置模型（NoopModelClient）时返回友好引导而非崩溃
        app.consolidator.llm = NoopModelClient()
        noop_out = _handle_repl_command(app, "/rag enrich")
        assert noop_out is not None and "未配置可用大模型" in noop_out

        # 验证 run_repl 遇到任意报错斜杠命令（如空关键词 /search !!!）不会崩溃退出，可继续处理后续命令
        inputs = iter(["/search !!!", "/rag status nonexistent_dir", "/quit"])
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
        run_repl(app)
        captured = capsys.readouterr().out
        assert "命令执行出错" in captured or "失败" in captured
    finally:
        app.close()
