"""检索前 Query 侧优化单元测试（TDD）：覆盖路由、多轮指代改写、多查询扩展/拆解、HyDE 质心融合、Multi-Query RRF 与 CLI /rag query。"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from personal_assistant.ai import (
    AbortController,
    AbortSignal,
    AssistantMessage,
    Context,
    NoopModelClient,
    StreamAbortedError,
    StreamDone,
    StreamOptions,
    TokenUsage,
    normalize_vector,
)
from personal_assistant.assistant.app import AssistantApp
from personal_assistant.assistant.cli import _handle_repl_command, run_repl
from personal_assistant.assistant.memory import MemoryStore
from personal_assistant.assistant.query_optimizer import (
    QueryOptimizationConfig,
    QueryOptimizer,
    expand_and_decompose_query,
    generate_hyde_document,
    rewrite_query,
    route_query,
)


# 录制每次 LLM 调用并按队列返回预设文本的轻量模拟模型（支持协同取消与思考关闭检查）
class _SpyQueryLLM:
    last_usage: TokenUsage | None = None

    # 初始化预设回复序列、可选故障注入与取消钩子
    def __init__(
        self,
        replies: list[str],
        *,
        fail_with: Exception | None = None,
        abort_after: int | None = None,
        controller: AbortController | None = None,
    ) -> None:
        self._replies = list(replies)
        self._fail_with = fail_with
        self._abort_after = abort_after
        self._controller = controller
        self.calls: list[tuple[Context, StreamOptions | None]] = []

    # 流式接口（满足 ModelInvoker 协议）
    def stream(self, context: Context, options: StreamOptions | None = None):
        msg = self.complete(context, options)
        yield StreamDone(reason="stop", message=msg)

    # 同步接口：检查取消、记录调用并返回队列下一条文本
    def complete(self, context: Context, options: StreamOptions | None = None) -> AssistantMessage:
        if options is not None and options.abort_signal is not None and options.abort_signal.aborted:
            raise StreamAbortedError(options.abort_signal.reason or "aborted")
        if self._fail_with is not None:
            raise self._fail_with
        self.calls.append((context, options))
        reply = self._replies.pop(0) if self._replies else ""
        if self._abort_after is not None and len(self.calls) == self._abort_after and self._controller is not None:
            self._controller.abort("取消 Query 优化")
        return AssistantMessage.from_text(reply, stop_reason="stop")


# 确定性概念向量模拟实现（测试 HyDE“答案-答案”比“问题-答案”语义更近，以及 BM25 零污染）
class _HydeConceptEmbedder:
    # 初始化模型名与调用记录
    def __init__(self, model_id: str = "mock-hyde-bge") -> None:
        self.model_id = model_id
        self.calls: list[list[str]] = []

    # 问题句落在轴 0，假设答案与真实文档落在轴 1，无关词落在轴 2（验证 Query+HyDE 质心把相似度从 0 抬升到 ~0.707）
    @staticmethod
    def _encode_one(text: str) -> tuple[float, ...]:
        lower = text.lower()
        axes = [0.0, 0.0, 0.0]
        if any(k in lower for k in ("怎么退钱", "退款怎么算", "question_only")):
            axes[0] += 1.0
        if any(k in lower for k in ("7天无理由", "原路退回", "年卡退款政策", "按未用天数折算")):
            axes[1] += 1.0
        if sum(axes) == 0.0:
            axes[2] = 1.0
        return normalize_vector(axes)

    # 批量编码文本为单位向量
    def embed(
        self,
        texts: Sequence[str],
        *,
        abort_signal: AbortSignal | None = None,
    ) -> list[tuple[float, ...]]:
        if abort_signal is not None and abort_signal.aborted:
            raise StreamAbortedError(abort_signal.reason or "aborted")
        batch = list(texts)
        self.calls.append(batch)
        return [self._encode_one(t) for t in batch]


# 测试 4 项原子能力：多轮改写（无历史 0 成本跳过）、路由判定（按可用工具约束）、多查询扩展/拆解与 HyDE 生成及配置校验
def test_atomic_query_optimization_primitives_and_config_validation() -> None:
    # 1. 配置参数严格校验
    for bad in (
        {"enabled": "yes"},
        {"routing": 1},
        {"max_sub_queries": 0},
        {"max_sub_queries": True},
        {"max_sub_queries": 10},
        {"max_sub_queries": 2.5},
    ):
        with pytest.raises(ValueError):
            QueryOptimizationConfig.from_dict(bad)

    # 2. 查询改写：无历史时零 LLM 调用直接返回原句；有历史时补全代词指代且显式关闭 reasoning
    spy_no_hist = _SpyQueryLLM(["不应被调用"])
    assert rewrite_query("它支持退款吗？", [], spy_no_hist) == "它支持退款吗？"
    assert len(spy_no_hist.calls) == 0

    history = [
        {"role": "user", "content": "帮我看看极客会员年卡有哪些权益？"},
        {"role": "assistant", "content": "极客会员年卡包含全部课程与离线下载权限。"},
    ]
    spy_rw = _SpyQueryLLM(['  "极客会员年卡的退款规则与条件"  '])
    rewritten = rewrite_query("它支持退款吗？", history, spy_rw)
    assert rewritten == "极客会员年卡的退款规则与条件"
    assert len(spy_rw.calls) == 1
    assert spy_rw.calls[0][1] is not None and spy_rw.calls[0][1].reasoning == "off"
    assert "极客会员年卡" in spy_rw.calls[0][0].messages[0].content

    # 3. 路由判定：区分 direct / docs / chat_history / web / multi，并过滤未启用工具对应路由
    spy_rt = _SpyQueryLLM(
        [
            '{"routes": ["direct"], "needs_retrieval": false, "reason": "基础算术问答无需检索"}',
            '```json\n{"routes": ["docs", "web"], "needs_retrieval": true, "reason": "需对比本地笔记与最新公开资料"}\n```',
            '{"routes": ["web"], "needs_retrieval": true, "reason": "查询今日新闻"}',
        ]
    )
    r_direct = route_query("1 + 1 等于几？", spy_rt)
    assert r_direct.routes == ("direct",) and r_direct.needs_retrieval is False

    r_multi = route_query("对比我笔记里的旧方案和网上最新架构", spy_rt, available_tools=["search_markdown", "web_search"])
    assert r_multi.routes == ("docs", "web") and r_multi.needs_retrieval is True

    # 当 web_search 未在白名单启用时，不允许路由到不可用的 web（安全退回 docs）
    r_filtered = route_query("今天新闻", spy_rt, available_tools=["search_markdown"])
    assert r_filtered.routes == ("docs",) and r_filtered.needs_retrieval is True

    # 4. 扩展与拆解：multi_query 保留原句并去重；decompose 拆分复合子问题；max_sub_queries=1 零调用
    spy_exp = _SpyQueryLLM(
        [
            '{"strategy": "decompose", "queries": ["Atlas 服务部署端口", "Nova 遥测采集器所在机房", "Nova 遥测采集器所在机房"]}',
            '{"strategy": "multi_query", "queries": ["本地回退扫描机制", "LinearRetriever fallback scan"]}',
        ]
    )
    dec = expand_and_decompose_query("对比 Atlas 端口和 Nova 机房", spy_exp, max_sub_queries=3)
    assert dec.strategy == "decompose"
    assert dec.sub_queries == ("Atlas 服务部署端口", "Nova 遥测采集器所在机房")

    mq = expand_and_decompose_query("无外挂插件怎么兜底查找", spy_exp, max_sub_queries=3)
    assert mq.strategy == "multi_query"
    assert mq.sub_queries[0] == "无外挂插件怎么兜底查找"
    assert len(mq.sub_queries) == 3

    spy_one = _SpyQueryLLM(["不应调用"])
    single = expand_and_decompose_query("单问题", spy_one, max_sub_queries=1)
    assert single.sub_queries == ("单问题",) and len(spy_one.calls) == 0

    # 5. HyDE 生成：折叠多余空白并截断至安全上限
    spy_hyde = _SpyQueryLLM(["  极客会员年卡支持 7 天无理由原路退回，\n超过 7 天按未用天数折算退款。  "])
    hyde_doc = generate_hyde_document("极客会员年卡怎么退钱", spy_hyde)
    assert "\n" not in hyde_doc and "7 天无理由原路退回" in hyde_doc


# 测试单次合并快路径 QueryOptimizer.plan()：1 次调用完成路由+改写+拆解，仅在 docs+embedder 时触发 HyDE，direct/故障/取消均安全收口
def test_query_optimizer_single_pass_plan_hyde_gating_and_abort() -> None:
    cfg = QueryOptimizationConfig(enabled=True, routing=True, rewrite=True, multi_query=True, hyde=True, max_sub_queries=3)
    opt = QueryOptimizer(cfg)

    # 1. direct 路由：仅 1 次 LLM 调用，绝不触发 HyDE
    spy_direct = _SpyQueryLLM(['{"routes": ["direct"], "needs_retrieval": false, "reason": "闲聊打招呼"}'])
    p_direct = opt.plan("你好呀", spy_direct, has_embedder=True)
    assert p_direct.route.needs_retrieval is False
    assert p_direct.hyde_document == ""
    assert len(spy_direct.calls) == 1

    # 2. docs 路由 + 无 embedder：仅 1 次合并调用（完成路由+改写+拆解），绝不浪费 Token 调 HyDE
    spy_no_emb = _SpyQueryLLM(
        [
            json.dumps(
                {
                    "routes": ["docs"],
                    "needs_retrieval": True,
                    "reason": "查询本地笔记多跳事实",
                    "rewritten_query": "对比 Atlas 端口与 Nova 机房",
                    "strategy": "decompose",
                    "queries": ["Atlas 部署端口", "Nova 部署机房"],
                },
                ensure_ascii=False,
            )
        ]
    )
    p_no_emb = opt.plan(
        "对比它和 Nova 的机房",
        spy_no_emb,
        history=[{"role": "user", "content": "Atlas 监听哪个端口？"}],
        has_embedder=False,
    )
    assert len(spy_no_emb.calls) == 1
    assert p_no_emb.rewritten_query == "对比 Atlas 端口与 Nova 机房"
    assert p_no_emb.expansion.sub_queries == ("Atlas 部署端口", "Nova 部署机房")
    assert p_no_emb.hyde_document == ""

    # 3. docs 路由 + 有 embedder：1 次合并调用 + 1 次 HyDE 调用
    spy_with_emb = _SpyQueryLLM(
        [
            json.dumps(
                {
                    "routes": ["docs"],
                    "needs_retrieval": True,
                    "reason": "查询会员退款政策",
                    "rewritten_query": "极客会员年卡退款怎么算",
                    "strategy": "single",
                    "queries": ["极客会员年卡退款怎么算"],
                },
                ensure_ascii=False,
            ),
            "极客会员年卡支持 7天无理由原路退回，超过期限按未用天数折算。",
        ]
    )
    p_emb = opt.plan("退款怎么算", spy_with_emb, has_embedder=True)
    assert len(spy_with_emb.calls) == 2
    assert "7天无理由" in p_emb.hyde_document

    # 4. 非取消异常优雅退回透传计划（且不向主模型注入噪声提示）；AbortSignal 必须向上穿透
    spy_err = _SpyQueryLLM([], fail_with=RuntimeError("upstream 500"))
    p_fb = opt.plan("Atlas 端口", spy_err, has_embedder=True)
    assert p_fb.rewritten_query == "Atlas 端口" and p_fb.route.routes == ("docs",)
    assert p_fb.applied is False and p_fb.format_agent_hint() == ""

    # 5. 纯引号用户输入不被清洗为空，且 format_agent_hint 绝不泄漏 HyDE 假设文本
    p_quote = opt.plan('""', _SpyQueryLLM(['{"routes": ["direct"], "needs_retrieval": false, "reason": "空符号"}']))
    assert p_quote.original_query == '""'
    assert "7天无理由" not in p_emb.format_agent_hint()

    ctrl = AbortController()
    ctrl.abort("用户中止")
    with pytest.raises(StreamAbortedError):
        opt.plan("Atlas 端口", _SpyQueryLLM([]), has_embedder=True, abort_signal=ctrl.signal)


# 测试检索引擎底层增强：Multi-Query RRF 拆解解决复合多跳检索 + HyDE 仅作用于向量通道（BM25 零幻觉词污染）
def test_rag_retriever_multi_query_rrf_and_hyde_vector_only_fusion(tmp_path: Path) -> None:
    from plugins.rag_retriever import RAGConfig, RAGRetriever

    data_dir = tmp_path / "data"
    mem_dir = data_dir / "memory"
    mem_dir.mkdir(parents=True)
    (mem_dir / "atlas.md").write_text("# 基础设施\n\n## Atlas\nAtlas 监听端口 9092，运行在东京机房。\n", encoding="utf-8")
    (mem_dir / "nova.md").write_text("# 遥测\n\n## Nova\nNova 部署在新加坡机房，上报端口 8443。\n", encoding="utf-8")
    (mem_dir / "refund.md").write_text(
        "# 会员服务\n\n## 售后条款\n极客年卡支持 7天无理由 原路退回，逾期按未用天数折算。\n",
        encoding="utf-8",
    )

    # 1. Multi-Query RRF：拆解后的两条子查询并行召回两篇不同文档，融合结果同时包含 Atlas 与 Nova
    ret_bm25 = RAGRetriever(RAGConfig())
    mem_bm25 = MemoryStore(data_dir, create_dirs=True, retriever=ret_bm25)
    try:
        multi_hits = ret_bm25.search_multi_hits(
            mem_bm25,
            ["Atlas 监听端口 9092", "Nova 新加坡机房 8443"],
            root="data/memory",
            limit=5,
        )
        hit_paths = {h.path for h in multi_hits}
        assert "data/memory/atlas.md" in hit_paths and "data/memory/nova.md" in hit_paths

        # 确认无 embedder 时即使传 hyde_text 也绝不把 hyde_text 里的幻觉词送进 BM25 污染匹配！
        no_pollute = ret_bm25.search_hits(
            mem_bm25,
            "完全不存在的词 ZZZ_NONE",
            root="data/memory",
            hyde_text="Atlas 监听端口 9092 东京机房",
        )
        assert no_pollute == []
        # 验证非法目录快速失败、limit=0 优雅空返回与取消信号穿透
        with pytest.raises(ValueError):
            ret_bm25.search_multi_hits(mem_bm25, ["Atlas", "Nova"], root="data/memory/not_exists")
        assert ret_bm25.search_multi_hits(mem_bm25, ["Atlas", "Nova"], root="data/memory", limit=0) == []
        aborted_ctrl = AbortController()
        aborted_ctrl.abort("取消检索")
        with pytest.raises(StreamAbortedError):
            ret_bm25.search_multi_hits(mem_bm25, ["Atlas", "Nova"], root="data/memory", abort_signal=aborted_ctrl.signal)
    finally:
        mem_bm25.close()

    # 2. HyDE 质心融合：原问题“怎么退钱”与文档“7天无理由 原路退回”正交（纯问题余弦=0，搜不到）；
    #    传入 HyDE 假设答案后，Query+HyDE 质心与真实文档余弦升至 ~0.707 >= 0.5，成功命中售后条款！
    emb = _HydeConceptEmbedder()
    ret_hyde = RAGRetriever(RAGConfig(min_vector_score=0.5), embedder=emb)
    mem_hyde = MemoryStore(data_dir, create_dirs=True, retriever=ret_hyde)
    try:
        assert mem_hyde.search_hits("怎么退钱") == []
        hyde_hits = ret_hyde.search_hits(
            mem_hyde,
            "怎么退钱",
            root="data/memory",
            hyde_text="年卡退款政策：支持 7天无理由 原路退回，超过期限按未用天数折算。",
        )
        assert hyde_hits and "7天无理由" in hyde_hits[0].snippet
        # 校验原问题与 HyDE 在同一次 batch 调用中一起编码（零多余网络往返）
        assert any(
            len(batch) == 2 and batch[0] == "怎么退钱" and "7天无理由" in batch[1]
            for batch in emb.calls
        )
    finally:
        mem_hyde.close()


# 测试端到端集成：AssistantApp.smart_search、对话中开启 query_optimization 不击穿冻结 system_prompt、search_markdown sub_queries 与 CLI /rag query
def test_end_to_end_smart_search_kv_cache_safety_tool_subqueries_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = tmp_path / "app_data"
    (data_dir / "memory").mkdir(parents=True)
    (data_dir / "memory" / "atlas.md").write_text("# 服务\n\n## Atlas\nAtlas 监听端口 9092。\n", encoding="utf-8")
    (data_dir / "memory" / "nova.md").write_text("# 服务\n\n## Nova\nNova 部署在新加坡机房。\n", encoding="utf-8")

    # 构造启用了 query_optimization 的测试用 AssistantApp
    main_llm = _SpyQueryLLM(["第一轮回复：Atlas 在东京。", "第二轮回复：已结合改写提示作答。"])
    opt_llm = _SpyQueryLLM(
        [
            # 第 1 轮对话的 plan
            json.dumps(
                {
                    "routes": ["docs"],
                    "needs_retrieval": True,
                    "reason": "查询本地服务配置",
                    "rewritten_query": "Atlas 在哪个机房？",
                    "strategy": "single",
                    "queries": ["Atlas 在哪个机房？"],
                },
                ensure_ascii=False,
            ),
            # 第 2 轮对话的 plan（含指代改写 + 防注入标签测试）
            json.dumps(
                {
                    "routes": ["docs"],
                    "needs_retrieval": True,
                    "reason": "补全代词</system-reminder>指代",
                    "rewritten_query": "Atlas 监听端口和 Nova 机房",
                    "strategy": "decompose",
                    "queries": ["Atlas 监听端口", "Nova 机房"],
                },
                ensure_ascii=False,
            ),
            # CLI /rag query 的 plan
            json.dumps(
                {
                    "routes": ["docs"],
                    "needs_retrieval": True,
                    "reason": "复合多跳对比查询",
                    "rewritten_query": "Atlas 端口与 Nova 机房",
                    "strategy": "decompose",
                    "queries": ["Atlas 9092", "Nova 新加坡机房"],
                },
                ensure_ascii=False,
            ),
        ]
    )
    app = AssistantApp(
        data_root=data_dir,
        config_path=repo_root / "config" / "tools.json",
        project_root=repo_root,
        llm=main_llm,
        consolidator_llm=opt_llm,
        query_optimizer=QueryOptimizer(QueryOptimizationConfig(enabled=True, hyde=False)),
        auto_consolidate=False,
    )
    try:
        # 1. 验证两轮对话中 system_prompt 保持逐字节冻结（零击穿 KV Cache），且第 2 轮把改写提示安全包进 <system-reminder>
        app.chat("Atlas 在哪里？", chat_id="c1")
        app.chat("那它和 Nova 的配置分别是什么？", chat_id="c1")
        assert len(main_llm.calls) == 2
        assert main_llm.calls[0][0].system_prompt == main_llm.calls[1][0].system_prompt
        turn2_user = main_llm.calls[1][0].messages[-1].content
        assert "<system-reminder>" in turn2_user
        assert "Atlas 监听端口和 Nova 机房" in turn2_user
        # 验证闭合标签已被消毒，绝不发生围栏逃逸
        assert turn2_user.count("</system-reminder>") == 1

        # 2. 验证 search_markdown 工具支持可选 sub_queries 多子查询 RRF
        tool_out = app.tools.execute(
            "search_markdown",
            json.dumps({"query": "Atlas 9092", "sub_queries": "Nova 新加坡机房"}, ensure_ascii=False),
        )
        assert "9092" in tool_out and "新加坡机房" in tool_out

        # 3. 验证 CLI /rag query 输出路由、改写、子查询与最终多跳命中结果
        cli_out = _handle_repl_command(app, "/rag query 对比 Atlas 端口和 Nova 机房", chat_id="c1")
        assert cli_out is not None
        assert "路由" in cli_out and "decompose" in cli_out
        assert "9092" in cli_out and "新加坡机房" in cli_out

        # 4. 缺参数与未配置模型均友好提示且不崩 REPL
        missing_q = _handle_repl_command(app, "/rag query")
        assert missing_q is not None and "用法" in missing_q

        app.consolidator.llm = NoopModelClient()
        noop_q = _handle_repl_command(app, "/rag query Atlas 端口")
        assert noop_q is not None and "9092" in noop_q

        inputs = iter(["/rag query", "/quit"])
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
        run_repl(app)
        assert "用法" in capsys.readouterr().out
    finally:
        app.close()


# 测试在线对话全链路 HyDE 隐藏注入（不进主 LLM Prompt）、规划阶段取消零脏写与 LinearRetriever 多子查询 RRF 兜底
def test_live_chat_hyde_hidden_injection_plan_abort_and_linear_subqueries(tmp_path: Path) -> None:
    from personal_assistant.ai import ToolCall

    repo_root = Path(__file__).resolve().parent.parent
    data_dir = tmp_path / "hyde_app_data"
    (data_dir / "memory").mkdir(parents=True)
    (data_dir / "memory" / "refund.md").write_text(
        "# 会员服务\n\n## 售后条款\n极客年卡支持 7天无理由 原路退回，逾期按未用天数折算。\n",
        encoding="utf-8",
    )

    # 主模型先调用 search_markdown("怎么退钱")，拿到 HyDE 质心检索到的售后条款后再作答
    class _ToolThenReplyLLM:
        last_usage: TokenUsage | None = None

        # 初始化记录容器
        def __init__(self) -> None:
            self.calls: list[Context] = []

        # 流式产出：第 1 步触发工具调用，第 2 步基于工具结果回复
        def stream(self, context: Context, options: StreamOptions | None = None):
            self.calls.append(context)
            if len(self.calls) == 1:
                msg = AssistantMessage(
                    blocks=(ToolCall(id="tc1", name="search_markdown", arguments='{"query": "怎么退钱"}'),),
                    stop_reason="toolUse",
                )
                yield StreamDone(reason="toolUse", message=msg)
                return
            msg = AssistantMessage.from_text("支持 7 天无理由原路退回。", stop_reason="stop")
            yield StreamDone(reason="stop", message=msg)

        # 同步入口（满足 ModelInvoker）
        def complete(self, context: Context, options: StreamOptions | None = None) -> AssistantMessage:
            for ev in self.stream(context, options):
                if isinstance(ev, StreamDone):
                    return ev.message
            raise RuntimeError("no done")

    main_llm = _ToolThenReplyLLM()
    opt_llm = _SpyQueryLLM(
        [
            json.dumps(
                {
                    "routes": ["docs"],
                    "needs_retrieval": True,
                    "reason": "查询售后退款政策",
                    "strategy": "single",
                    "queries": ["怎么退钱"],
                },
                ensure_ascii=False,
            ),
            "年卡退款政策：支持 7天无理由 原路退回，超过期限按未用天数折算。",
        ]
    )
    emb = _HydeConceptEmbedder()
    app = AssistantApp(
        data_root=data_dir,
        config_path=repo_root / "config" / "tools.json",
        project_root=repo_root,
        llm=main_llm,
        consolidator_llm=opt_llm,
        embedder=emb,
        query_optimizer=QueryOptimizer(QueryOptimizationConfig(enabled=True, hyde=True)),
        auto_consolidate=False,
    )
    try:
        res = app.chat("怎么退钱", chat_id="c_hyde")
        # 1. HyDE 绝不进入主 LLM 第 1 步 Prompt（防幻觉污染），但通过隐藏 turn_ctx 生效于 search_markdown 工具！
        first_prompt = main_llm.calls[0].messages[-1].content
        assert "7天无理由" not in first_prompt
        tool_calls = res["tool_calls"]
        assert isinstance(tool_calls, list) and len(tool_calls) == 1
        assert "7天无理由" in str(tool_calls[0]["observation"])

        # 2. 在规划阶段触发 AbortSignal：干净抛 StreamAbortedError 且 SQLite 零脏写
        before_count = app.state.count_messages()
        ctrl = AbortController()
        abort_opt_llm = _SpyQueryLLM(["{}", "hyde"], abort_after=1, controller=ctrl)
        app.consolidator.llm = abort_opt_llm
        with pytest.raises(StreamAbortedError):
            app.chat("任意问题", chat_id="c_abort", abort_signal=ctrl.signal)
        assert app.state.count_messages() == before_count
    finally:
        app.close()

    # 3. LinearRetriever（无 search_multi_hits）下 search_markdown(sub_queries=...) 仍正常执行通用 RRF 融合
    linear_data = tmp_path / "linear_data"
    (linear_data / "memory").mkdir(parents=True)
    (linear_data / "memory" / "a.md").write_text("# A\nAlpha 9092\n", encoding="utf-8")
    (linear_data / "memory" / "b.md").write_text("# B\nBeta 8443\n", encoding="utf-8")
    linear_mem = MemoryStore(linear_data, create_dirs=True)
    linear_app = AssistantApp(
        data_root=linear_data,
        memory=linear_mem,
        llm=_SpyQueryLLM(["ok"]),
        auto_consolidate=False,
    )
    try:
        out = linear_app.tools.execute(
            "search_markdown",
            json.dumps({"query": "Alpha 9092", "sub_queries": "Beta 8443"}, ensure_ascii=False),
        )
        assert "Alpha 9092" in out and "Beta 8443" in out
    finally:
        linear_app.close()
