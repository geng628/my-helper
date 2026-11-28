"""RAG Step 3（检索后增强）全链路 TDD 测试：/v1/rerank 重排、去重压缩、Lost-in-the-Middle 与 CRAG 兜底。"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError

import pytest

from personal_assistant.ai import (
    AbortController,
    AbortSignal,
    AssistantMessage,
    Context,
    LLMNotConfigured,
    ModelProfile,
    NoopModelClient,
    StreamAbortedError,
    StreamDone,
    StreamOptions,
)
from personal_assistant.assistant.app import AssistantApp
from personal_assistant.assistant.builtin_tools import MemoryToolsetProvider
from personal_assistant.assistant.cli import _handle_rag_command
from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.assistant.memory import MemoryStore, SearchHit
from personal_assistant.assistant.rag_metadata import MetadataFilter
from plugins.models_dsl import DSLCompileError, compile_models_dsl
from plugins.rag_retriever import RAGConfig, RAGRetriever


# 离线假大模型：记录调用上下文与 StreamOptions，按预设队列吐出固定文本
class _StubLLM:
    last_usage = None

    # 初始化预设回复序列与调用记录列表
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[Context, StreamOptions | None]] = []

    # 按流式契约吐出单条完成事件
    def stream(self, context: Context, options: StreamOptions | None = None):
        self.calls.append((context, options))
        if options is not None and options.abort_signal is not None and options.abort_signal.aborted:
            raise StreamAbortedError(options.abort_signal.reason or "cancelled")
        text = self._replies.pop(0) if self._replies else ""
        yield StreamDone(reason="stop", message=AssistantMessage.from_text(text))

    # 同步返回下一条预设消息
    def complete(self, context: Context, options: StreamOptions | None = None) -> AssistantMessage:
        self.calls.append((context, options))
        if options is not None and options.abort_signal is not None and options.abort_signal.aborted:
            raise StreamAbortedError(options.abort_signal.reason or "cancelled")
        text = self._replies.pop(0) if self._replies else ""
        return AssistantMessage.from_text(text)


# 离线假 Cross-Encoder 重排器：按给定规则对 (query, documents) 打分并记录收到过的全部原文
class _FakeReranker:
    # 绑定模型名与自定义打分回调（或报错异常）
    def __init__(
        self,
        model_id: str = "BAAI/bge-reranker-v2-m3",
        *,
        scorer: Any = None,
        error: Exception | None = None,
    ) -> None:
        self.model_id = model_id
        self._scorer = scorer
        self._error = error
        self.calls: list[tuple[str, list[str], int | None]] = []

    # 记录输入文档并按回调返回排序后的 RerankResult 列表
    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> list[Any]:
        from personal_assistant.ai import RerankResult

        if abort_signal is not None and abort_signal.aborted:
            raise StreamAbortedError(abort_signal.reason or "aborted")
        self.calls.append((query, list(documents), top_n))
        if self._error is not None:
            raise self._error
        scored: list[RerankResult] = []
        for idx, doc in enumerate(documents):
            score = float(self._scorer(query, doc, idx)) if callable(self._scorer) else 0.5
            scored.append(RerankResult(index=idx, score=score))
        scored.sort(key=lambda r: (-r.score, r.index))
        if top_n is not None:
            scored = scored[:top_n]
        return scored


# 构造带最小 tools.json 的临时项目骨架，便于隔离测试配置与 App 编排
def _make_project(tmp_path: Path, *, retrieval_extra: dict[str, object] | None = None, enabled_tools: list[str] | None = None) -> tuple[Path, Path, Path]:
    repo_root = Path(__file__).resolve().parent.parent
    cfg_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    (data_dir / "memory").mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    retrieval: dict[str, object] = {"plugin": "rag", "allowed_access": ["public"]}
    if retrieval_extra:
        retrieval.update(retrieval_extra)
    tools_cfg = {
        "enabled_tools": enabled_tools
        if enabled_tools is not None
        else ["list_files", "search_markdown", "read_markdown", "search_chat_history", "web_search"],
        "plugins": ["plugins"],
        "retrieval": retrieval,
        "files": {
            "read_roots": ["data/memory", "data/workspace"],
            "write_root": "data/workspace",
        },
    }
    tools_json = cfg_dir / "tools.json"
    tools_json.write_text(json.dumps(tools_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return tools_json, data_dir, repo_root


# ======================== 1. Layer 1 /v1/rerank 客户端 ========================


# 测试 OpenAIRerankClient 标准 /v1/rerank 与 /v2/rerank 地址解析、relevance_score/score 兼容、降序排序与 top_n
def test_openai_rerank_client_endpoints_and_scoring() -> None:
    from personal_assistant.ai import OpenAIRerankClient, RerankInvoker, RerankResult

    captured: list[tuple[str, dict[str, object], dict[str, str]]] = []

    # 模拟硅基流动/Cohere 标准响应（乱序返回 relevance_score）
    def fake_post(url: str, payload: dict[str, object], headers: dict[str, str], *, abort_signal: AbortSignal | None = None) -> dict[str, object]:
        captured.append((url, payload, headers))
        return {
            "results": [
                {"index": 2, "relevance_score": 0.91},
                {"index": 0, "relevance_score": 0.12},
                {"index": 1, "relevance_score": 0.77},
            ]
        }

    profile = ModelProfile.from_dict(
        "bge-rerank",
        {
            "api": "openai-rerank",
            "base_url": "https://api.siliconflow.cn/v1?api-version=1",
            "model": "BAAI/bge-reranker-v2-m3",
            "api_key": "sk-test-rerank-123",
        },
    )
    client = OpenAIRerankClient(profile, post_json=fake_post)
    assert isinstance(client, RerankInvoker)
    assert client.model_id == "BAAI/bge-reranker-v2-m3"

    res = client.rerank("会员退款规则", ["doc0", "doc1", "doc2"], top_n=2)
    assert res == [RerankResult(index=2, score=0.91), RerankResult(index=1, score=0.77)]
    assert captured[0][0] == "https://api.siliconflow.cn/v1/rerank?api-version=1"
    assert captured[0][1]["top_n"] == 2
    assert captured[0][1]["return_documents"] is False
    assert captured[0][2]["Authorization"] == "Bearer sk-test-rerank-123"

    # 测试已带 /rerank 后缀的 Cohere v2 地址不重复拼接，且兼容 score 字段
    captured.clear()

    # 模拟返回 score 字段且无顶层截断的 TEI/网关响应
    def fake_post_score(url: str, payload: dict[str, object], headers: dict[str, str], *, abort_signal: AbortSignal | None = None) -> dict[str, object]:
        captured.append((url, payload, headers))
        return {"results": [{"index": 0, "score": 0.4}, {"index": 1, "score": 0.85}]}

    cohere_profile = ModelProfile.from_dict(
        "cohere",
        {"base_url": "https://api.cohere.com/v2/rerank", "model": "rerank-v3.5", "api_key": "sk-cohere"},
    )
    cohere_client = OpenAIRerankClient(cohere_profile, post_json=fake_post_score)
    res2 = cohere_client.rerank("query", ["a", "b"])
    assert captured[0][0] == "https://api.cohere.com/v2/rerank"
    assert res2 == [RerankResult(index=1, score=0.85), RerankResult(index=0, score=0.4)]
    # 空文档列表零网络调用直接返回空列表
    assert cohere_client.rerank("query", []) == []
    assert len(captured) == 1


# 测试 OpenAIRerankClient 严格输入/响应校验、本地免密、远端缺密报错、AbortSignal 与令牌脱敏
def test_openai_rerank_client_validation_abort_and_security() -> None:
    from personal_assistant.ai import OpenAIRerankClient

    # 远端无 key 必须报错 LLMNotConfigured
    remote_no_key = ModelProfile.from_dict("r", {"base_url": "https://api.example.com/v1", "model": "bge-reranker"})
    with pytest.raises(LLMNotConfigured):
        OpenAIRerankClient(remote_no_key, post_json=lambda *a, **kw: {}).rerank("q", ["doc"])

    # 本机回环允许免 key
    local_profile = ModelProfile.from_dict("loc", {"base_url": "http://127.0.0.1:8080/v1", "model": "bge-reranker"})
    local_client = OpenAIRerankClient(
        local_profile,
        post_json=lambda u, p, h, *, abort_signal=None: {"results": [{"index": 0, "score": 0.8}]},
    )
    assert local_client.rerank("q", ["doc"])[0].score == 0.8

    # 非法参数拒绝：空 query、裸 str documents、bool top_n、负数 top_n
    with pytest.raises(ValueError):
        local_client.rerank("   ", ["doc"])
    with pytest.raises(TypeError):
        local_client.rerank("q", cast(Any, "not-a-list"))
    with pytest.raises(ValueError):
        local_client.rerank("q", ["doc"], top_n=0)
    with pytest.raises(ValueError):
        local_client.rerank("q", ["doc"], top_n=True)

    # 非法响应拒绝：越界 index、重复 index、bool index、NaN/Inf/bool score
    bad_payloads = [
        {"results": [{"index": 5, "relevance_score": 0.9}]},
        {"results": [{"index": 0, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.5}]},
        {"results": [{"index": True, "relevance_score": 0.9}]},
        {"results": [{"index": 0, "relevance_score": float("nan")}]},
        {"results": [{"index": 0, "relevance_score": float("inf")}]},
        {"results": [{"index": 0, "relevance_score": True}]},
        {"results": "not-a-list"},
    ]
    for bad in bad_payloads:
        bad_client = OpenAIRerankClient(
            local_profile,
            post_json=cast(Any, lambda u, p, h, *, abort_signal=None, b=bad: b),
        )
        with pytest.raises(ValueError):
            bad_client.rerank("q", ["d0", "d1"])

    # AbortSignal 触发时无论前置还是网络回调期抛 HTTPError 均必须收敛为 StreamAbortedError
    ctrl = AbortController()
    ctrl.abort("用户中止")
    with pytest.raises(StreamAbortedError):
        local_client.rerank("q", ["doc"], abort_signal=ctrl.signal)

    ctrl2 = AbortController()

    # 模拟传输途中触发取消并伴随 HTTPError
    def aborting_http(url: str, payload: dict[str, object], headers: dict[str, str], *, abort_signal: AbortSignal | None = None) -> dict[str, object]:
        ctrl2.abort("中断重排")
        raise HTTPError(url, 500, "err", None, io.BytesIO(b"secret sk-leak-9999"))

    with pytest.raises(StreamAbortedError):
        OpenAIRerankClient(local_profile, post_json=aborting_http).rerank("q", ["doc"], abort_signal=ctrl2.signal)

    # 非取消 HTTP 报错必须对 sk- 密钥脱敏
    def leaking_http(url: str, payload: dict[str, object], headers: dict[str, str], *, abort_signal: AbortSignal | None = None) -> dict[str, object]:
        raise HTTPError(url, 401, "Unauthorized", None, io.BytesIO(b"bad token sk-secret-xyz987"))

    with pytest.raises(RuntimeError) as exc_info:
        OpenAIRerankClient(local_profile, post_json=leaking_http).rerank("q", ["doc"])
    assert "sk-secret-xyz987" not in str(exc_info.value)
    assert "sk-***" in str(exc_info.value)


# 测试 from_env_reranker 同源密钥回退策略（严禁把 LLM_API_KEY 泄漏到不同主机）
def test_from_env_reranker_same_origin_key_safety() -> None:
    from personal_assistant.ai import from_env_reranker

    assert from_env_reranker({}) is None

    # 仅设 RERANK_MODEL 且与 LLM 同源时安全复用 LLM_API_KEY
    same = from_env_reranker(
        {
            "LLM_BASE_URL": "https://api.siliconflow.cn/v1",
            "LLM_API_KEY": "sk-shared-key",
            "RERANK_MODEL": "BAAI/bge-reranker-v2-m3",
        }
    )
    assert same is not None
    assert same.model_id == "BAAI/bge-reranker-v2-m3"
    assert same.profile.resolve_api_key({}) == "sk-shared-key"

    # 指定不同主机 RERANK_BASE_URL 时绝不回退泄漏 LLM_API_KEY
    cross = from_env_reranker(
        {
            "LLM_BASE_URL": "https://api.deepseek.com/v1",
            "LLM_API_KEY": "sk-deepseek-secret",
            "RERANK_BASE_URL": "http://127.0.0.1:8080/v1",
            "RERANK_MODEL": "bge-reranker-v2-m3",
        }
    )
    assert cross is not None
    assert cross.profile.resolve_api_key({}) == ""


# ======================== 2. models.dsl reranker 角色与 ToolsConfig ========================


# 测试 models.dsl 顶层 reranker 指令编译、重复报错、未定义报错与 ToolsConfig.build_reranker
def test_models_dsl_reranker_role_and_tools_config_build(tmp_path: Path) -> None:
    dsl = """
    use deepseek sk-chat-123
    reranker bge_rerank

    model bge_rerank {
      base_url https://api.siliconflow.cn/v1
      model    BAAI/bge-reranker-v2-m3
      api_key  sk-sf-456
    }
    """
    compiled = compile_models_dsl(dsl)
    assert compiled["roles"] == {"reranker": "bge_rerank"}

    with pytest.raises(DSLCompileError):
        compile_models_dsl(dsl + "\nreranker bge_rerank\n")
    with pytest.raises(DSLCompileError):
        compile_models_dsl("use deepseek sk-1\nreranker missing_model\n")

    tools_json, _, repo_root = _make_project(tmp_path)
    tcfg = ToolsConfig.load(tools_json, project_root=repo_root)
    try:
        # 无 models.dsl 时拒绝把对话预设名当作 reranker
        with pytest.raises(ValueError, match="对话厂商预设名"):
            tcfg.build_reranker(tmp_path / "config", model_name="deepseek", env={})

        # 写入 models.dsl 后自动按 roles['reranker'] 构造 RerankInvoker
        (tmp_path / "config" / "models.dsl").write_text(dsl, encoding="utf-8")
        reranker = tcfg.build_reranker(tmp_path / "config", env={})
        assert reranker is not None
        assert reranker.model_id == "BAAI/bge-reranker-v2-m3"
    finally:
        tcfg.plugin_manager.terminate_all()


# ======================== 3. RAGRetriever Cross-Encoder 精排 ========================


# 测试 RAGRetriever 粗排 Top-N -> Cross-Encoder 精排改写排序、min_rerank_score 过滤与前置权限硬隔离
def test_rag_retriever_rerank_reorders_filters_and_enforces_access_boundary(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    mem_dir = data_dir / "memory"
    mem_dir.mkdir(parents=True)

    # 写入多篇公开文档 + 1 篇机密文档，确保粗排能捞出多篇且机密文档绝对不能送给外部 reranker
    (mem_dir / "doc_noise.md").write_text(
        "# 退款FAQ\n\n退款 退款 退款 退款说明见客服公告。\n",
        encoding="utf-8",
    )
    (mem_dir / "doc_exact.md").write_text(
        "# 会员年卡政策\n\n年度会员在购买后 7 天内未使用可全额退款，逾期按剩余月份折算。\n",
        encoding="utf-8",
    )
    (mem_dir / "doc_secret.md").write_text(
        "---\naccess: confidential\n---\n# 内部退款赔付底线\n\n退款内部审批最高赔付 5000 元。\n",
        encoding="utf-8",
    )

    # 假 Cross-Encoder：对包含 "7 天内未使用可全额退款" 的文档打 0.96，噪声文档打 0.08
    reranker = _FakeReranker(
        scorer=lambda q, doc, idx: 0.96 if "7 天内未使用" in doc else 0.08,
    )
    retriever = RAGRetriever(
        RAGConfig(rerank_candidates=50, min_rerank_score=0.2),
        reranker=reranker,
    )
    store = MemoryStore(data_dir, retriever=retriever)
    try:
        hits = store.search_hits(
            "退款",
            limit=5,
            metadata_filter=MetadataFilter(allowed_access=("public",)),
        )
        # 精排应把 doc_exact.md 推至第 1 名，并滤掉低于 min_rerank_score=0.2 的噪声文档
        assert len(hits) == 1
        assert hits[0].path.endswith("doc_exact.md")
        assert hits[0].score == pytest.approx(0.96)
        # 严格校验：外部 reranker 绝未收到任何 confidential 文档文本！
        assert len(reranker.calls) == 1
        sent_docs = reranker.calls[0][1]
        assert all("5000" not in d and "doc_secret" not in d for d in sent_docs)

        # index_status 应暴露 reranker_model 与 rerank_candidates
        status = retriever.index_status(store)
        assert status["reranker_model"] == "BAAI/bge-reranker-v2-m3"
        assert status["rerank_candidates"] == 50
    finally:
        store.close()


# 测试 RAGRetriever 在 reranker 临时故障时优雅降级保留粗排结果，但在 AbortSignal 触发时立即传播取消
def test_rag_retriever_rerank_degrades_on_error_and_propagates_abort(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    mem_dir = data_dir / "memory"
    mem_dir.mkdir(parents=True)
    (mem_dir / "policy.md").write_text("# 政策\n\n退款流程需要提交订单号。\n", encoding="utf-8")

    # 瞬时网络异常应优雅降级返回粗排结果
    failing = _FakeReranker(error=RuntimeError("503 Service Unavailable"))
    retriever = RAGRetriever(RAGConfig(rerank_candidates=20), reranker=failing)
    store = MemoryStore(data_dir, retriever=retriever)
    try:
        hits = store.search_hits("退款", limit=5)
        assert len(hits) == 1
        assert hits[0].path.endswith("policy.md")

        # 主动取消必须抛出 StreamAbortedError 绝不吞掉
        ctrl = AbortController()
        ctrl.abort("cancel rerank")
        with pytest.raises(StreamAbortedError):
            retriever.search_hits(store, "退款", limit=5, abort_signal=ctrl.signal)
    finally:
        store.close()


# ======================== 4. 去重、上下文压缩、头尾重排与 CRAG ========================


# 测试跨文档近重复去重、句/行级上下文压缩与 Lost-in-the-Middle 头尾重排
def test_postprocess_dedup_compress_and_lost_in_middle() -> None:
    from personal_assistant.assistant.rag_postprocess import (
        PostRetrievalConfig,
        compress_hits,
        deduplicate_hits,
        reorder_lost_in_middle,
    )

    cfg = PostRetrievalConfig.from_dict(
        {
            "dedup_similarity": 0.8,
            "max_context_chars": 800,
            "max_chars_per_hit": 300,
            "compress_long_hits": True,
            "lost_in_middle_reorder": True,
            "crag_high_threshold": 0.7,
            "crag_low_threshold": 0.2,
        }
    )
    assert cfg.lost_in_middle_reorder is True

    # 校验非法配置被拒绝（low > high、布尔值、超界）
    with pytest.raises(ValueError):
        PostRetrievalConfig.from_dict({"crag_low_threshold": 0.8, "crag_high_threshold": 0.5})
    with pytest.raises(ValueError):
        PostRetrievalConfig.from_dict({"dedup_similarity": True})

    # 1. 跨文档近重复片段应被抑制，不同主题片段应保留
    h1 = SearchHit("data/memory/a.md", 1, 4, "退款", "  1: 年度会员购买后七天内支持全额退款，请提交订单编号办理。", 0.95)
    h2 = SearchHit("data/memory/b.md", 10, 13, "退款副本", "  10: 年度会员购买后七天内支持全额退款，请提交订单编号办理。", 0.90)
    h3 = SearchHit("data/memory/c.md", 1, 3, "发票", "  1: 企业增值税专用发票可在结算中心按月申请开具。", 0.70)
    deduped = deduplicate_hits([h1, h2, h3], similarity_threshold=0.82)
    assert [h.path for h in deduped] == ["data/memory/a.md", "data/memory/c.md"]

    # 2. 超长命中块压缩：保留背景摘要、表头与命中核心句及邻接行，裁剪无关冗长句子，不超 max_chars_per_hit
    filler_lines = [f"  {i}: 这是第 {i} 行与退款无关的常规产品介绍和免责声明凑字数段落。" for i in range(2, 20)]
    long_snippet = "\n".join(
        [
            "  [背景: 会员账单与售后规范]",
            "  1: ## 售后条款",
            *filler_lines[:8],
            "  10: 核心规则：年度会员在 7 天内未激活权益可申请全额原路退款。",
            *filler_lines[8:],
        ]
    )
    long_hit = SearchHit("data/memory/long.md", 1, 20, "售后条款", long_snippet, 0.92)
    compressed = compress_hits([long_hit], "会员 退款", max_context_chars=1200, max_chars_per_hit=320)
    assert len(compressed) == 1
    sn = compressed[0].snippet
    assert "[背景: 会员账单与售后规范]" in sn
    assert "7 天内未激活权益可申请全额原路退款" in sn
    assert len(sn) <= 360

    # 3. Lost-in-the-Middle 重排：[0, 1, 2, 3, 4, 5] -> [0, 2, 4, 5, 3, 1]（第 1 名放开头，第 2 名放末尾）
    ranked = [SearchHit(f"f{i}.md", 1, 1, "", f"  1: item {i}", 1.0 - i * 0.1) for i in range(6)]
    reordered = reorder_lost_in_middle(ranked)
    assert [h.path for h in reordered] == ["f0.md", "f2.md", "f4.md", "f5.md", "f3.md", "f1.md"]
    assert [h.path for h in reorder_lost_in_middle(ranked[:2])] == ["f0.md", "f1.md"]


# 测试 CRAG 三级相关性评估与白名单数据源切换/拒答兜底逻辑
def test_crag_sufficiency_grading_and_fallback_actions() -> None:
    from personal_assistant.assistant.rag_postprocess import (
        PostRetrievalConfig,
        corrective_rewrite_query,
        evaluate_retrieval_sufficiency,
    )

    cfg = PostRetrievalConfig(crag_high_threshold=0.65, crag_low_threshold=0.25)
    high_hit = [SearchHit("a.md", 1, 2, "退款", "  1: 退款政策", 0.88)]
    mid_hit = [SearchHit("a.md", 1, 2, "会员", "  1: 会员说明", 0.45)]
    low_hit = [SearchHit("a.md", 1, 2, "无关", "  1: 天气晴朗", 0.10)]

    # Cross-Encoder 模式下严格按 high/low 阈值分档
    a_high = evaluate_retrieval_sufficiency("退款", high_hit, has_reranker=True, config=cfg)
    assert a_high.verdict == "correct" and a_high.action == "answer"

    a_mid = evaluate_retrieval_sufficiency("退款", mid_hit, has_reranker=True, config=cfg, allow_rewrite=True)
    assert a_mid.verdict == "ambiguous" and a_mid.action == "rewrite_retry"

    a_low_switch = evaluate_retrieval_sufficiency(
        "退款",
        low_hit,
        has_reranker=True,
        config=cfg,
        available_tools=["search_markdown", "web_search", "search_chat_history"],
        allow_rewrite=False,
    )
    assert a_low_switch.verdict == "incorrect"
    assert a_low_switch.action == "switch_source"
    assert a_low_switch.suggested_sources == ("web_search", "search_chat_history")

    # 未启用外部检索工具时，低相关必须走 abstain 拒答而非推荐禁用工具
    a_low_abstain = evaluate_retrieval_sufficiency(
        "退款",
        low_hit,
        has_reranker=True,
        config=cfg,
        available_tools=["search_markdown", "read_markdown"],
        allow_rewrite=False,
    )
    assert a_low_abstain.verdict == "incorrect"
    assert a_low_abstain.action == "abstain"
    assert a_low_abstain.suggested_sources == ()

    # corrective_rewrite_query 必须传 reasoning="off" 且 NoopModelClient 返回空串
    assert corrective_rewrite_query("退款", low_hit, NoopModelClient()) == ""
    stub = _StubLLM(["年度会员 退款 取消订阅 退费政策"])
    rewritten = corrective_rewrite_query("退钱", low_hit, stub)
    assert rewritten == "年度会员 退款 取消订阅 退费政策"
    assert stub.calls[0][1] is not None and stub.calls[0][1].reasoning == "off"


# ======================== 5. 端到端 App / smart_search / builtin_tools / CLI / evals ========================


# 测试 AssistantApp.smart_search CRAG 一次纠错改写重查（含权限防越权）与失败后清空噪声兜底
def test_app_smart_search_crag_rewrite_retry_and_abstain(tmp_path: Path) -> None:
    from personal_assistant.assistant.rag_postprocess import PostRetrievalConfig

    tools_json, data_dir, repo_root = _make_project(tmp_path, enabled_tools=["search_markdown", "read_markdown"])
    (data_dir / "memory" / "billing.md").write_text(
        "# 订阅退费\n\n会员退费流程：在账单页点击申请原路退回，3 个工作日到账。\n",
        encoding="utf-8",
    )
    (data_dir / "memory" / "secret.md").write_text(
        "---\naccess: confidential\n---\n# 机密\n\n会员退费内部特批密码 SECRET-999。\n",
        encoding="utf-8",
    )

    # 场景 A：首查 "怎么退钱" 命中为空 -> 触发 CRAG 一次改写为 "会员退费流程" -> 重查成功并严守 public 权限！
    plan_json = json.dumps(
        {
            "needs_retrieval": True,
            "routes": ["docs"],
            "reason": "查询退费规则",
            "rewritten_query": "怎么退钱",
            "strategy": "single",
            "queries": ["怎么退钱"],
        },
        ensure_ascii=False,
    )
    stub_llm = _StubLLM([plan_json, "会员退费流程"])
    reranker = _FakeReranker(scorer=lambda q, doc, idx: 0.93 if "原路退回" in doc else 0.05)

    app = AssistantApp(
        data_root=data_dir,
        config_path=tools_json,
        project_root=repo_root,
        llm=stub_llm,
        reranker=reranker,
        post_retrieval_config=PostRetrievalConfig(crag_enabled=True, crag_high_threshold=0.65, crag_low_threshold=0.25),
        auto_consolidate=False,
    )
    try:
        assert app.status()["reranker"] == "BAAI/bge-reranker-v2-m3"
        res = app.smart_search("怎么退钱", force_enable=True)
        crag = res["crag"]
        assert isinstance(crag, dict)
        assert crag["verdict"] == "correct"
        assert crag["action"] == "answer"
        assert crag["attempts"] == 2
        assert crag["rewritten_query"] == "会员退费流程"
        hits = res["hits"]
        assert isinstance(hits, list) and len(hits) == 1
        assert hits[0]["path"].endswith("billing.md")
        assert "SECRET-999" not in str(res["rendered"])
        # 总 LLM 调用恰好 2 次（1 次 QueryOptimizer + 最多 1 次 CRAG 改写）
        assert len(stub_llm.calls) == 2

        # 场景 B：即使重查仍低分，必须清空噪声 hits 并触发 abstain 拒答（且不会循环超过 1 次重试）
        low_reranker = _FakeReranker(scorer=lambda q, doc, idx: 0.05)
        cast(Any, app.memory.retriever).set_reranker(low_reranker)
        app.reranker = low_reranker
        stub_llm2 = _StubLLM([plan_json, "会员退费流程", "绝不应触发第二次改写"])
        app.consolidator.llm = stub_llm2
        res_low = app.smart_search("会员退费", force_enable=True)
        assert res_low["hits"] == []
        assert res_low["crag"]["verdict"] == "incorrect"
        assert res_low["crag"]["action"] == "abstain"
        assert res_low["crag"]["attempts"] == 2
        assert "不足" in str(res_low["rendered"]) or "拒答" in str(res_low["rendered"])
        assert len(stub_llm2.calls) == 2

        # CLI /rag query 输出必须展示 CRAG 校验状态
        stub_llm3 = _StubLLM([plan_json, "会员退费流程"])
        app.consolidator.llm = stub_llm3
        cli_out = _handle_rag_command(app, "/rag query 会员退费", chat_id="cli")
        assert "【CRAG】" in cli_out
    finally:
        app.close()


# 测试 builtin_tools.search_markdown 经由工具调用自动执行跨文档去重与上下文压缩
def test_builtin_search_markdown_applies_post_retrieval_dedup_and_compression(tmp_path: Path) -> None:
    from personal_assistant.assistant.rag_postprocess import PostRetrievalConfig

    data_dir = tmp_path / "data"
    mem_dir = data_dir / "memory"
    mem_dir.mkdir(parents=True)
    dup_body = "# 退款规则\n\n年度会员购买后七天内未使用权益支持原路全额退款，请携带订单编号联系客服办理。\n"
    (mem_dir / "copy1.md").write_text(dup_body, encoding="utf-8")
    (mem_dir / "copy2.md").write_text(dup_body, encoding="utf-8")

    store = MemoryStore(data_dir, retriever=RAGRetriever())
    try:
        tools = {
            t.name: t
            for t in MemoryToolsetProvider(store).build_tools(
                context={"post_retrieval_config": PostRetrievalConfig(dedup_similarity=0.8)}
            )
        }
        out = tools["search_markdown"].handler(query="退款", root="data/memory")
        # 两份完全重复的文件去重后只应保留 1 份
        assert out.count("copy1.md") + out.count("copy2.md") == 1
    finally:
        store.close()


# 测试 evals CLI --reranker 参数校验（与 --retriever linear 冲突报错）
def test_evals_cli_reranker_flag_rejects_linear(tmp_path: Path) -> None:
    from evals.cli import main as evals_main

    with pytest.raises(ValueError, match="--reranker"):
        evals_main(
            [
                "--mode",
                "search",
                "--retriever",
                "linear",
                "--reranker",
                "bge-rerank",
                "--max-cases",
                "1",
            ]
        )


# 测试多子查询 search_multi_hits 精排、单行多句压缩、纯语义零字面重叠保底与 AssistantApp 离线隔离/热切换
def test_multi_query_rerank_sentence_compression_and_app_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from personal_assistant.assistant.rag_postprocess import compress_hits

    # 1. 单行超长多句段压缩保留关键句，且零字面重叠纯语义命中平滑保留句首
    long_line = (
        "  5: "
        + "这是前置免责声明和通用版本更新背景介绍。" * 6
        + "重点提示：发票开具需在每月 25 日前提交企业税号。"
        + "这是尾部联系地址和补充说明文字。" * 6
    )
    hit_long = SearchHit("data/memory/invoice.md", 5, 5, "发票", long_line, 0.9)
    comp = compress_hits([hit_long], "企业税号 发票", max_context_chars=600, max_chars_per_hit=180)
    assert "企业税号" in comp[0].snippet
    assert len(comp[0].snippet) <= 200

    comp_sem = compress_hits([hit_long], "完全不重合的同义词查询", max_context_chars=600, max_chars_per_hit=180)
    assert "免责声明" in comp_sem[0].snippet

    # 2. RAGRetriever.search_multi_hits 传入 rerank_query 精排
    tools_json, data_dir, repo_root = _make_project(tmp_path)
    (data_dir / "memory" / "m1.md").write_text("# 退费\n\n退费政策详情：支持 7 天退款。\n", encoding="utf-8")
    (data_dir / "memory" / "m2.md").write_text("# 续费\n\n续费政策详情：自动扣费前 3 天提醒。\n", encoding="utf-8")
    rr = _FakeReranker(scorer=lambda q, doc, idx: 0.92 if "自动扣费" in doc else 0.4)
    ret = RAGRetriever(RAGConfig(rerank_candidates=20), reranker=rr)
    store = MemoryStore(data_dir, retriever=ret)
    try:
        mhits = ret.search_multi_hits(
            store,
            ["退费", "续费"],
            limit=2,
            rerank_query="自动续费提醒",
        )
        assert len(mhits) == 2
        assert mhits[0].path.endswith("m2.md")
        assert rr.calls[-1][0] == "自动续费提醒"
    finally:
        store.close()

    # 3. AssistantApp 注入 llm 时必须与宿主机 RERANK_* 环境变量隔离；reload_llm 可按 models.dsl 热生效 reranker
    monkeypatch.setenv("RERANK_BASE_URL", "https://rerank.host.example/v1")
    monkeypatch.setenv("RERANK_API_KEY", "sk-host-rerank")
    monkeypatch.setenv("RERANK_MODEL", "host-reranker")
    app = AssistantApp(
        data_root=data_dir,
        config_path=tools_json,
        project_root=repo_root,
        llm=_StubLLM([]),
        auto_consolidate=False,
    )
    try:
        assert app.reranker is None
        assert app.status()["reranker"] is None

        (tmp_path / "config" / "models.dsl").write_text(
            """
            use deepseek sk-chat-1
            reranker rr_model
            model rr_model {
              base_url https://api.siliconflow.cn/v1
              model    BAAI/bge-reranker-v2-m3
              api_key  sk-rr-2
            }
            """,
            encoding="utf-8",
        )
        app.reload_llm()
        assert app.reranker is not None
        assert app.reranker.model_id == "BAAI/bge-reranker-v2-m3"
        assert getattr(app.memory.retriever, "reranker", None) is app.reranker
    finally:
        app.close()

