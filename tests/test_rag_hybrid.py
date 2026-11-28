"""RAG Step 2 向量持久化缓存、BM25+Dense RRF 混合检索、抗噪门控与 CLI /rag embed 单元测试（TDD）。"""
from __future__ import annotations

import math
import sqlite3
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pytest

from personal_assistant.ai import (
    AbortController,
    AbortSignal,
    AssistantMessage,
    Context,
    StreamAbortedError,
    StreamDone,
    StreamOptions,
    TokenUsage,
    normalize_vector,
)
from personal_assistant.assistant.app import AssistantApp
from personal_assistant.assistant.cli import _handle_repl_command, run_repl
from personal_assistant.assistant.memory import MemoryStore


# 确定性概念向量模拟实现（零外部依赖）：按语义概念词群映射到正交 6 维单位向量，并支持记录调用与取消/故障注入
class _ConceptEmbedder:
    # 初始化模型名、概念词表、可选分批上限、取消或故障钩子
    def __init__(
        self,
        model_id: str = "mock-bge-m3",
        *,
        abort_after_calls: int | None = None,
        controller: AbortController | None = None,
        fail_with: Exception | None = None,
    ) -> None:
        self.model_id = model_id
        self.calls: list[list[str]] = []
        self._abort_after = abort_after_calls
        self._controller = controller
        self._fail_with = fail_with

    # 将输入文本映射为确定性 7 维单位向量：前 5 维对应已知概念，轴 5 为“火星负例”，轴 6 为其余普通文本（确保无关负例与正文正交）
    @staticmethod
    def _encode_one(text: str) -> tuple[float, ...]:
        lower = text.lower()
        axes = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if any(k in lower for k in ("linearretriever", "fallback", "兜底查找", "回退扫描", "无外挂插件")):
            axes[0] += 1.0
        if any(k in lower for k in ("东京", "9092", "deployment", "tokyo", "机房")):
            axes[1] += 1.0
        if any(k in lower for k in ("sqlite", "wal", "派生数据库", "索引库")):
            axes[2] += 1.0
        if any(k in lower for k in ("星河科技", "营收", "财报", "毛利率")):
            axes[3] += 1.0
        if any(k in lower for k in ("expired_vec_secret", "过期口令")):
            axes[4] += 1.0
        if any(k in lower for k in ("火星殖民地", "火箭燃料")):
            axes[5] += 1.0
        if sum(axes) == 0.0:
            axes[6] = 1.0
        return normalize_vector(axes)

    # 批量向量化文本（支持协同取消与预设异常注入）
    def embed(
        self,
        texts: Sequence[str],
        *,
        abort_signal: AbortSignal | None = None,
    ) -> list[tuple[float, ...]]:
        if abort_signal is not None and abort_signal.aborted:
            raise StreamAbortedError(abort_signal.reason or "aborted")
        if self._fail_with is not None:
            raise self._fail_with
        batch = list(texts)
        self.calls.append(batch)
        result = [self._encode_one(t) for t in batch]
        if self._abort_after is not None and len(self.calls) == self._abort_after and self._controller is not None:
            self._controller.abort("中断批量向量化")
        return result


# 最小假对话模型（用于测试 enrich_contexts 与 embed_chunks 联动更新 embed_hash）
class _StaticSummaryLLM:
    last_usage: TokenUsage | None = None

    # 初始化固定摘要回复
    def __init__(self, reply: str) -> None:
        self._reply = reply

    # 流式调用返回固定文本
    def stream(self, context: Context, options: StreamOptions | None = None):
        msg = self.complete(context, options)
        yield StreamDone(reason="stop", message=msg)

    # 同步调用返回固定文本
    def complete(self, context: Context, options: StreamOptions | None = None) -> AssistantMessage:
        return AssistantMessage.from_text(self._reply, stop_reason="stop")


# 测试混合检索核心价值：救回零字面重叠同义改写、RRF 双路协同超越单路命中、抗噪门控守住 100% 负例干净度
def test_rag_hybrid_rescues_zero_overlap_paraphrase_rrf_synergy_and_negative_gate(tmp_path: Path) -> None:
    from plugins.rag_retriever import RAGConfig, RAGRetriever

    data_dir = tmp_path / "data"
    mem_dir = data_dir / "memory"
    mem_dir.mkdir(parents=True)
    (mem_dir / "fallback.md").write_text(
        "# Retrieval Architecture\n\n"
        "## Builtin Engine\n"
        "LinearRetriever handles built-in fallback scan when no plugin is enabled.\n",
        encoding="utf-8",
    )
    (mem_dir / "infra.md").write_text(
        "# 运维手册\n\n"
        "## 双路协同节点\n"
        "Atlas 集群在东京机房监听 9092，本地采用 SQLite WAL 持久化。\n\n"
        "## 纯词面节点\n"
        "Atlas 集群测试编号 ALPHA_EXACT_ONLY，不涉及常规部署描述。\n\n"
        "## 纯语义节点\n"
        "海外区域 Tokyo deployment 节点运行在标准容器环境中。\n",
        encoding="utf-8",
    )

    # 1. 纯 BM25（无 embedder）：零共享词中文同义改写无法命中英文文档
    bm25_only = RAGRetriever(RAGConfig())
    mem_bm25 = MemoryStore(data_dir, create_dirs=True, retriever=bm25_only)
    try:
        assert mem_bm25.search_hits("无外挂插件时系统如何兜底查找") == []
    finally:
        mem_bm25.close()

    # 2. 开启混合检索（BM25 + Dense RRF）：成功救回零共享词同义改写！
    embedder = _ConceptEmbedder()
    hybrid = RAGRetriever(RAGConfig(min_vector_score=0.5), embedder=embedder)
    mem_hybrid = MemoryStore(data_dir, create_dirs=True, retriever=hybrid)
    try:
        paraphrase_hits = mem_hybrid.search_hits("无外挂插件时系统如何兜底查找")
        assert paraphrase_hits
        assert "LinearRetriever" in paraphrase_hits[0].snippet
        assert paraphrase_hits[0].score > 0.0

        # 3. RRF 双路协同：同时命中 BM25 与 Dense 的「双路协同节点」排名第 1，高于纯词面与纯语义单路块
        rrf_hits = mem_hybrid.search_hits("Atlas 东京机房 9092", limit=5)
        assert len(rrf_hits) >= 2
        assert "双路协同节点" in rrf_hits[0].section

        # 4. 抗噪门控：与所有文档正交（相似度 0.0 < min_vector_score）的负例严格返回空列表
        assert mem_hybrid.search_hits("火星殖民地氧气配方与火箭燃料") == []
    finally:
        mem_hybrid.close()


# 测试 chunk_embeddings 持久化缓存、跨重启零重算、AbortSignal 断点续跑、重建保留、上下文摘要联动与 v2->v3 原地升级
def test_rag_embeddings_persistence_abort_resume_rebuild_survival_and_v2_upgrade(tmp_path: Path) -> None:
    from plugins.rag_retriever import RAGConfig, RAGRetriever

    data_dir = tmp_path / "data"
    mem_dir = data_dir / "memory"
    state_dir = data_dir / "state"
    mem_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)

    # 先预埋一个 Step 1.5 v2 派生库（含已付费 chunk_contexts），验证平滑原地升级至 v3 且不丢 chunk_contexts
    db_path = state_dir / "rag_index.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.executemany(
            "INSERT INTO index_meta (key, value) VALUES (?, ?)",
            [
                ("schema_version", "2"),
                ("chunker_version", "2"),
                ("parent_max_chars", "1000"),
                ("child_max_chars", "320"),
                ("child_overlap_chars", "64"),
            ],
        )
        conn.execute("CREATE TABLE indexed_files (root_key TEXT, path TEXT)")
        conn.execute("CREATE TABLE parent_chunks (id INTEGER PRIMARY KEY, text TEXT)")
        conn.execute("CREATE TABLE child_chunks (id INTEGER PRIMARY KEY, text TEXT)")
        conn.execute("CREATE VIRTUAL TABLE child_chunks_fts USING fts5(body_tokens)")
        conn.execute(
            "CREATE TABLE chunk_contexts (content_hash TEXT, model TEXT, summary TEXT, created_at TEXT, PRIMARY KEY (content_hash, model))"
        )
        conn.execute(
            "INSERT INTO chunk_contexts VALUES ('legacy_hash', 'm1', '保留的历史摘要', '2026-11-28T00:00:00+00:00')"
        )

    (mem_dir / "doc1.md").write_text(
        "# 报告\n\n"
        "## 章节一\n第一段普通说明文字，不含主体名称。\n\n"
        "## 章节二\n第二段东京机房与端口 9092 配置。\n\n"
        "## 章节三\n第三段 SQLite WAL 持久化索引说明。\n",
        encoding="utf-8",
    )

    # 1. 分批向量化（batch_size=1）在首批后触发 AbortSignal，验证首批已落盘、重启后自动续跑零重算
    ctrl = AbortController()
    emb_abort = _ConceptEmbedder(abort_after_calls=1, controller=ctrl)
    r1 = RAGRetriever(RAGConfig(parent_max_chars=500, child_max_chars=240, child_overlap_chars=40), embedder=emb_abort)
    mem1 = MemoryStore(data_dir, create_dirs=True, retriever=r1)
    try:
        with pytest.raises(StreamAbortedError):
            r1.embed_chunks(mem1, root="data/memory", batch_size=1, abort_signal=ctrl.signal)
        assert len(emb_abort.calls) == 1

        # 验证 v2->v3 原地升级保留了历史 chunk_contexts 且已落盘 1 条向量
        with sqlite3.connect(db_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM chunk_contexts WHERE content_hash = 'legacy_hash'").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0] == 1

        # 续跑 embed_chunks：第 1 块命中缓存，只为剩余 2 块调用 embed
        emb_resume = _ConceptEmbedder()
        r1.set_embedder(emb_resume)
        stats = r1.embed_chunks(mem1, root="data/memory", batch_size=10)
        assert stats["embedded"] == 2
        assert stats["cached_hits"] == 1
        assert stats["children_total"] == 3
        assert len(emb_resume.calls) == 1 and len(emb_resume.calls[0]) == 2
    finally:
        mem1.close()

    # 2. 重启新建实例搜索：所有子块命中缓存，只对 query 调 1 次 embed()
    emb_restart = _ConceptEmbedder()
    r2 = RAGRetriever(RAGConfig(parent_max_chars=500, child_max_chars=240, child_overlap_chars=40), embedder=emb_restart)
    mem2 = MemoryStore(data_dir, create_dirs=True, retriever=r2)
    try:
        hits = mem2.search_hits("tokyo deployment")
        assert hits and "9092" in hits[0].snippet
        assert len(emb_restart.calls) == 1 and emb_restart.calls[0] == ["tokyo deployment"]

        # 3. 执行离线上下文摘要补全：背景摘要写入后子块 embed_hash 自动刷新，触发增量重算携带 [背景: 星河科技] 的新向量
        r2.enrich_contexts(mem2, _StaticSummaryLLM("本段属于星河科技年度营收复盘报告。"), root="data/memory", model_name="m1")
        emb_after_ctx = _ConceptEmbedder()
        r2.set_embedder(emb_after_ctx)
        stats_ctx = r2.embed_chunks(mem2, root="data/memory")
        assert stats_ctx["embedded"] == 3
        assert any("| 背景: 本段属于星河科技年度营收复盘报告。]" in txt for batch in emb_after_ctx.calls for txt in batch)
        # 此时纯靠背景摘要中的“星河科技”语义向量也能搜中“章节一”
        ctx_hits = mem2.search_hits("毛利率")
        assert ctx_hits and "[背景:" in ctx_hits[0].snippet
    finally:
        mem2.close()

    # 4. 调整 child_overlap_chars 重建索引时，chunk_embeddings 缓存表必须保留不丢失
    r3 = RAGRetriever(RAGConfig(parent_max_chars=500, child_max_chars=240, child_overlap_chars=80), embedder=_ConceptEmbedder())
    mem3 = MemoryStore(data_dir, create_dirs=True, retriever=r3)
    try:
        assert mem3.search_hits("9092")
        with sqlite3.connect(db_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0] >= 3
    finally:
        mem3.close()


# 测试过期条目绝不外发给 Embedding API、只读零写盘、在线查询网络故障降级 BM25 与配置参数校验
def test_rag_hybrid_expiry_readonly_error_fallback_and_config_validation(tmp_path: Path) -> None:
    from plugins.rag_retriever import RAGConfig, RAGRetriever

    for bad_cfg in (
        {"min_vector_score": -0.1},
        {"min_vector_score": 1.1},
        {"min_vector_score": True},
        {"min_vector_score": float("nan")},
        {"rrf_k": 0},
        {"rrf_k": True},
        {"rrf_k": 1.5},
        {"bm25_weight": -1.0},
        {"vector_weight": -0.5},
        {"bm25_weight": 0.0, "vector_weight": 0.0},
    ):
        with pytest.raises(ValueError):
            RAGConfig.from_dict(bad_cfg)

    data_dir = tmp_path / "data"
    mem_dir = data_dir / "memory"
    mem_dir.mkdir(parents=True)
    (mem_dir / "sched.md").write_text(
        "# 日程\n\n"
        "## 常驻\n"
        "Atlas 部署在东京机房 9092。\n\n"
        "- [2026-10-01] 过期口令 EXPIRED_VEC_SECRET (until: 2026-10-15)\n",
        encoding="utf-8",
    )

    # 1. 只读模式下开启混合检索严守零写盘，且已过期条目绝不外发至 embedder 也绝不被向量通道召回
    ro_emb = _ConceptEmbedder()
    ro_ret = RAGRetriever(RAGConfig(min_vector_score=0.5), embedder=ro_emb)
    ro_mem = MemoryStore(data_dir, create_dirs=False, retriever=ro_ret)
    try:
        hits = ro_mem.search_hits("过期口令 EXPIRED_VEC_SECRET", today=date(2026, 11, 20))
        assert hits == []
        # 确认 EXPIRED_VEC_SECRET 从未作为文档子块发送给 embedder（仅 query 自身会编码）
        doc_texts_sent = [txt for batch in ro_emb.calls for txt in batch if txt != "过期口令 EXPIRED_VEC_SECRET"]
        assert all("EXPIRED_VEC_SECRET" not in txt for txt in doc_texts_sent)
        assert not (data_dir / "state").exists()
    finally:
        ro_mem.close()

    # 2. 在线查询时 Embedding 接口异常优雅降级到 BM25；显式 embed_chunks 则直接暴露错误
    broken_emb = _ConceptEmbedder(fail_with=RuntimeError("embedding server 503"))
    fb_ret = RAGRetriever(RAGConfig(), embedder=broken_emb)
    fb_mem = MemoryStore(data_dir, create_dirs=True, retriever=fb_ret)
    try:
        fallback_hits = fb_mem.search_hits("东京机房 9092", today=date(2026, 11, 20))
        assert fallback_hits and "9092" in fallback_hits[0].snippet
        with pytest.raises(RuntimeError, match="503"):
            fb_ret.embed_chunks(fb_mem, root="data/memory", today=date(2026, 11, 20))
    finally:
        fb_mem.close()


# 测试同一 model_id 切换维度（7 维 -> 3 维）自动重算缓存不归零、纯 Dense/纯 BM25 权重短路、损坏 BLOB 容错与 evals --embedder 校验
def test_rag_hybrid_dimension_switch_weight_shortcircuit_corrupt_blob_and_eval_cli(tmp_path: Path) -> None:
    from evals import main as eval_main
    from plugins.rag_retriever import RAGConfig, RAGRetriever

    # 支持切换输出维度的同 model_id 模拟向量器（先 7 维，后切 3 维）
    class _DimSwitchEmbedder:
        # 绑定模型标识（故意含首尾空白验证归一化）与目标维度
        def __init__(self, dims: int, *, model_id: str = "  shared-bge-m3  ") -> None:
            self.model_id = model_id
            self.dimensions = dims
            self.calls: list[list[str]] = []

        # 按当前维度编码概念向量
        def embed(self, texts: Sequence[str], *, abort_signal: AbortSignal | None = None) -> list[tuple[float, ...]]:
            batch = list(texts)
            self.calls.append(batch)
            out: list[tuple[float, ...]] = []
            for txt in batch:
                low = txt.lower()
                vec = [0.0] * self.dimensions
                if any(k in low for k in ("linearretriever", "兜底查找", "fallback", "フォールバック")):
                    vec[0] = 1.0
                elif any(k in low for k in ("东京", "9092")):
                    vec[1] = 1.0
                else:
                    vec[-1] = 1.0
                out.append(normalize_vector(vec))
            return out

    data_dir = tmp_path / "data"
    mem_dir = data_dir / "memory"
    mem_dir.mkdir(parents=True)
    (mem_dir / "a.md").write_text(
        "# 架构\n\n## 回退\nLinearRetriever handles fallback scan.\n\n"
        "- [2026-10-01] 过期项 EXPIRED_OLD (until: 2026-10-15)\n",
        encoding="utf-8",
    )

    # 1. 先按 7 维落盘缓存，随后同一 model_id 切换为 3 维：验证 search_hits 与 index_status 精确识别新维度并重算命中
    emb7 = _DimSwitchEmbedder(dims=7)
    ret = RAGRetriever(RAGConfig(min_vector_score=0.5), embedder=emb7)
    mem = MemoryStore(data_dir, create_dirs=True, retriever=ret)
    try:
        assert mem.search_hits("兜底查找", today=date(2026, 11, 20))
        st7 = ret.index_status(mem, today=date(2026, 11, 20))
        assert st7["embedder_model"] == "shared-bge-m3"
        assert st7["embedded_children"] == 1  # 已过期子块绝不计入活跃已嵌入数

        emb3 = _DimSwitchEmbedder(dims=3)
        ret.set_embedder(emb3)
        st3_before = ret.index_status(mem, today=date(2026, 11, 20))
        assert st3_before["embedded_children"] == 0

        hits3 = mem.search_hits("兜底查找", today=date(2026, 11, 20))
        assert hits3 and "LinearRetriever" in hits3[0].snippet
        st3_after = ret.index_status(mem, today=date(2026, 11, 20))
        assert st3_after["embedded_children"] == 1
        assert st3_after["cached_embeddings"] == 2  # 7 维与 3 维按 (content_hash, model, dims) 共存

        # 2. 注入非法类型 vector_blob（整数 10**9）验证 get_cached_vector 安全返回 None 零 OOM，且 search_hits 自动重算自愈
        db_path = data_dir / "state" / "rag_index.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("UPDATE chunk_embeddings SET vector_blob = ? WHERE dims = 3", (10**9,))
        ret_corrupt = RAGRetriever(RAGConfig(min_vector_score=0.5), embedder=_DimSwitchEmbedder(dims=3))
        mem_corrupt = MemoryStore(data_dir, create_dirs=True, retriever=ret_corrupt)
        try:
            _, idx_obj = ret_corrupt._get_index(mem_corrupt)
            assert idx_obj.get_cached_vector("bad_h", "shared-bge-m3", 3, 10**9) is None
            assert idx_obj.get_cached_vector("bad_h", "shared-bge-m3", 3, "not-bytes") is None
            # 损坏 BLOB 自动识别为无效缓存并重算自愈
            healed_hits = mem_corrupt.search_hits("兜底查找", today=date(2026, 11, 20))
            assert healed_hits and "LinearRetriever" in healed_hits[0].snippet
        finally:
            mem_corrupt.close()
    finally:
        mem.close()

    # 3. vector_weight=0.0 时绝不调用 embedder；bm25_weight=0.0 时支持纯稠密与纯日文假名查询
    spy_zero_vec = _DimSwitchEmbedder(dims=3)
    ret_bm25_only = RAGRetriever(RAGConfig(bm25_weight=1.0, vector_weight=0.0), embedder=spy_zero_vec)
    mem_bm25 = MemoryStore(data_dir, create_dirs=True, retriever=ret_bm25_only)
    try:
        assert mem_bm25.search_hits("LinearRetriever", today=date(2026, 11, 20))
        assert len(spy_zero_vec.calls) == 0
    finally:
        mem_bm25.close()

    spy_dense_only = _DimSwitchEmbedder(dims=3)
    ret_dense_only = RAGRetriever(RAGConfig(bm25_weight=0.0, vector_weight=1.0, min_vector_score=0.5), embedder=spy_dense_only)
    mem_dense = MemoryStore(data_dir, create_dirs=True, retriever=ret_dense_only)
    try:
        assert mem_dense.search_hits("兜底查找", today=date(2026, 11, 20))
        kana_hits = mem_dense.search_hits("フォールバック", today=date(2026, 11, 20))
        assert kana_hits and "LinearRetriever" in kana_hits[0].snippet
    finally:
        mem_dense.close()

    # 4. evals/cli.py --embedder 与 --retriever linear 混用必须显式报错，绝不静默忽略
    with pytest.raises(ValueError, match="linear"):
        eval_main(["--mode", "search", "--retriever", "linear", "--embedder", "mock-emb", "--max-cases", "1"])


# 测试 CLI /rag embed 与 /rag status（含未配 embedder、越界路径、多余参数、异常容错与 REPL 不崩）
def test_cli_rag_embed_and_status_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = tmp_path / "app_data"
    (data_dir / "memory").mkdir(parents=True)
    (data_dir / "memory" / "guide.md").write_text(
        "# 架构\n\n## 回退引擎\nLinearRetriever handles built-in fallback scan when no plugin is enabled.\n",
        encoding="utf-8",
    )

    emb = _ConceptEmbedder()
    app = AssistantApp(
        data_root=data_dir,
        config_path=repo_root / "config" / "tools.json",
        project_root=repo_root,
        embedder=emb,
        auto_consolidate=False,
    )
    try:
        st0 = _handle_repl_command(app, "/rag status")
        assert st0 is not None
        assert '"embedder_model": "mock-bge-m3"' in st0
        assert '"embedded_children": 0' in st0

        out1 = _handle_repl_command(app, "/rag embed data/memory")
        assert out1 is not None and "新生成 1" in out1 and "复用缓存 0" in out1

        st1 = _handle_repl_command(app, "/rag status")
        assert st1 is not None and '"embedded_children": 1' in st1

        out2 = _handle_repl_command(app, "/rag embed")
        assert out2 is not None and "新生成 0" in out2 and "复用缓存 1" in out2

        # 语义同义改写直接可搜中
        search_res = app.search("无外挂插件兜底查找")
        assert "LinearRetriever" in search_res

        # 越界路径与未配置 embedder 均友好提示且不崩 REPL
        bad_path = _handle_repl_command(app, "/rag embed ../outside")
        assert bad_path is not None and "失败" in bad_path

        app.embedder = None
        set_emb = getattr(app.memory.retriever, "set_embedder", None)
        assert callable(set_emb)
        set_emb(None)
        unconf = _handle_repl_command(app, "/rag embed")
        assert unconf is not None and "未配置" in unconf

        inputs = iter(["/rag embed ../outside", "/rag embed data/memory extra", "/quit"])
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
        run_repl(app)
        repl_out = capsys.readouterr().out
        assert "未配置" in repl_out and "用法" in repl_out
    finally:
        app.close()
