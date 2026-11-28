"""检索中增强单元测试（TDD）：覆盖 YAML Frontmatter 元数据解析、BM25/Dense 前置硬过滤、防越权提级与领域 Embedding 微调三元组导出。"""
from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pytest

from personal_assistant.ai import AbortSignal, AssistantMessage, Context, StreamDone, StreamOptions, TokenUsage, normalize_vector
from personal_assistant.assistant.app import AssistantApp
from personal_assistant.assistant.cli import _handle_repl_command
from personal_assistant.assistant.memory import MemoryStore
from personal_assistant.assistant.rag_metadata import (
    DocumentMetadata,
    MetadataFilter,
    parse_filter_string,
    parse_markdown_frontmatter,
)


# 确定性概念向量模拟实现（同时记录全部被编码的文本，用于校验越权文档绝不外发给 Embedding API）
class _ScopedEmbedder:
    # 初始化模型名与调用日志
    def __init__(self, model_id: str = "mock-meta-bge") -> None:
        self.model_id = model_id
        self.calls: list[list[str]] = []

    # 将退款/薪资/部署映射到高相似度向量空间，便于校验“若无前置硬过滤必然高分命中，启用前置过滤则零命中”
    @staticmethod
    def _encode_one(text: str) -> tuple[float, ...]:
        lower = text.lower()
        axes = [0.0, 0.0, 0.0, 0.0]
        if any(k in lower for k in ("退款", "退钱", "年卡", "billing", "refund")):
            axes[0] += 1.0
        if any(k in lower for k in ("薪资", "薪酬", "机密", "salary", "payroll")):
            axes[1] += 1.0
        if any(k in lower for k in ("部署", "端口", "运维", "deploy")):
            axes[2] += 1.0
        if sum(axes) == 0.0:
            axes[3] = 1.0
        return normalize_vector(axes)

    # 批量编码文本并记录明细
    def embed(
        self,
        texts: Sequence[str],
        *,
        abort_signal: AbortSignal | None = None,
    ) -> list[tuple[float, ...]]:
        batch = list(texts)
        self.calls.append(batch)
        return [self._encode_one(t) for t in batch]


# 极简离线占位模型（满足 AssistantApp 测试注入）
class _DummyLLM:
    last_usage: TokenUsage | None = None

    # 流式返回固定回复
    def stream(self, context: Context, options: StreamOptions | None = None):
        yield StreamDone(reason="stop", message=AssistantMessage.from_text("ok", stop_reason="stop"))

    # 同步返回固定回复
    def complete(self, context: Context, options: StreamOptions | None = None) -> AssistantMessage:
        return AssistantMessage.from_text("ok", stop_reason="stop")


# 测试 Frontmatter 解析（行号对齐、水平分割线防误判、字段归一化）与 MetadataFilter 权限防提级语义
def test_frontmatter_parsing_line_alignment_and_filter_access_ceiling() -> None:
    md = (
        "---\n"
        "date: 2026-05-20T09:30:00+08:00\n"
        "product: Billing\n"
        "department: Finance\n"
        "access: Internal\n"
        "tags:\n"
        "  - 退款\n"
        "  - 年卡\n"
        "  - 退款\n"
        "---\n"
        "# 售后规则\n"
        "极客年卡支持 7 天无理由退款。\n"
    )
    meta, masked = parse_markdown_frontmatter(md)
    assert meta.doc_date == "2026-05-20"
    assert meta.product == "billing"
    assert meta.department == "finance"
    assert meta.access == "internal"
    assert meta.tags == ("年卡", "退款")
    assert meta.body_start_line == 11
    # 屏蔽后的正文行号必须与原始磁盘文件严格 1:1 对齐，且绝不含 frontmatter 键词
    masked_lines = masked.splitlines()
    assert masked_lines[10] == "# 售后规则"
    assert masked_lines[11] == "极客年卡支持 7 天无理由退款。"
    assert "department" not in masked and "internal" not in masked

    # 顶部水平分割线 `---` 后接普通正文，绝不能被误判为 Frontmatter
    hr_md = "---\n这是一段普通 Markdown 正文，不是 Frontmatter。\n---\n"
    hr_meta, hr_masked = parse_markdown_frontmatter(hr_md)
    assert hr_meta == DocumentMetadata()
    assert hr_masked == hr_md

    # 权限收敛交集与防提级：允许 public 时传 access=internal 返回不可满足空集；允许 public+internal 时可收窄为 internal
    f_escalate = parse_filter_string("product=billing; access=internal", allowed_access=("public",))
    assert f_escalate.is_unsatisfiable is True
    assert f_escalate.matches(meta) is False

    f_narrow = parse_filter_string(
        "product=Billing; department=finance; date_from=2026-05-01; date_to=2026-05-31; tags=退款,年卡; access=internal",
        allowed_access=("public", "internal"),
    )
    assert f_narrow.is_unsatisfiable is False
    assert f_narrow.matches(meta) is True

    # 标签精确边界校验：tag=退 不能误中 tag=退款
    f_tag_miss = parse_filter_string("tags=退", allowed_access=("public", "internal"))
    assert f_tag_miss.matches(meta) is False

    # 非法日期区间或未知字段严格报错
    for bad in (
        "date_from=2026-06-01; date_to=2026-05-01",
        "date_from=not-a-date",
        "unknown_field=abc",
        "missing_equals",
    ):
        with pytest.raises(ValueError):
            parse_filter_string(bad)


# 测试前置元数据硬过滤在 BM25 与 Dense 相似度计算前生效：越权/超期/跨产品线文档零召回且绝不进入余弦打分或在线向量外发
def test_rag_and_linear_prefiltering_before_bm25_and_dense_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from plugins import rag_retriever as rag_mod
    from plugins.rag_retriever import RAGConfig, RAGRetriever

    data_dir = tmp_path / "data"
    mem_dir = data_dir / "memory"
    mem_dir.mkdir(parents=True)
    # 1. 公开计费文档（2026-06-15）
    (mem_dir / "billing_2026.md").write_text(
        "---\n"
        "date: 2026-06-15\n"
        "product: billing\n"
        "department: finance\n"
        "access: public\n"
        "tags: [退款, 年卡]\n"
        "---\n"
        "# 2026 年卡售后\n"
        "极客会员年卡支持 7 天无理由退款。\n",
        encoding="utf-8",
    )
    # 2. 旧版计费文档（2024-03-01）
    (mem_dir / "billing_2024.md").write_text(
        "---\n"
        "date: 2024-03-01\n"
        "product: billing\n"
        "department: finance\n"
        "access: public\n"
        "tags: [退款]\n"
        "---\n"
        "# 2024 旧版售后\n"
        "2024 旧规则仅支持 3 天内退款。\n",
        encoding="utf-8",
    )
    # 3. 机密薪酬文档（access: hr，默认 public 严禁召回且严禁在线外发向量）
    (mem_dir / "hr_salary.md").write_text(
        "---\n"
        "date: 2026-06-20\n"
        "product: hr_portal\n"
        "department: hr\n"
        "access: hr\n"
        "tags: [薪资, 机密]\n"
        "---\n"
        "# 薪酬方案\n"
        "高管薪资机密方案：年底双薪加期权。\n",
        encoding="utf-8",
    )
    # 4. 普通无 Frontmatter 旧笔记（默认视为 access=public，零破坏）
    (mem_dir / "plain.md").write_text("# 运维笔记\nAtlas 服务部署在端口 9092。\n", encoding="utf-8")

    emb = _ScopedEmbedder()
    ret = RAGRetriever(RAGConfig(min_vector_score=0.5), embedder=emb)
    mem = MemoryStore(data_dir, create_dirs=True, retriever=ret)
    try:
        # 1. 默认 public 权限检索 "高管薪资机密方案"：不仅零命中，而且 hr_salary.md 绝不能被在线外发给 embedder！
        assert mem.search_hits("高管薪资机密方案") == []
        sent_texts = [txt for batch in emb.calls for txt in batch]
        assert all("年底双薪加期权" not in txt for txt in sent_texts)

        # 2. 默认 embed_chunks（仅 public）绝不外发 hr_salary.md；管理员显式传 allowed_access=('*',) 预热全库后再测余弦计数！
        emb.calls.clear()
        ret.embed_chunks(mem, root="data/memory", embedder=emb)
        assert all("年底双薪加期权" not in txt for batch in emb.calls for txt in batch)
        ret.embed_chunks(mem, root="data/memory", embedder=emb, metadata_filter=MetadataFilter(allowed_access=("*",)))
        cosine_calls = 0
        orig_cosine = rag_mod.cosine_similarity

        # 统计稠密检索阶段实际参与余弦相似度计算的候选向量个数
        def _counting_cosine(a: Sequence[float], b: Sequence[float]) -> float:
            nonlocal cosine_calls
            cosine_calls += 1
            return orig_cosine(a, b)

        monkeypatch.setattr(rag_mod, "cosine_similarity", _counting_cosine)

        # 2a. public 权限查 "薪资"：hr_salary.md 被 SQLite WHERE 前置剔除，余弦打分零命中
        assert mem.search_hits("薪资") == []

        # 2b. 仅当系统侧显式授予 allowed_access=("public", "hr") 时才命中 hr_salary.md，且返回行号精确对齐真实文件第 9 行正文
        hr_hits = mem.search_hits("薪资", metadata_filter=MetadataFilter(allowed_access=("public", "hr")))
        assert hr_hits and hr_hits[0].path == "data/memory/hr_salary.md"
        assert hr_hits[0].section == "薪酬方案"
        assert hr_hits[0].start_line == 9 and hr_hits[0].end_line == 9

        # 2c. 时间范围 + 产品线 + 标签前置硬过滤：只筛出 2026 年 billing 文档，排除 2024 旧版，且余弦仅对被筛出的子块执行！
        cosine_calls = 0
        f_2026 = MetadataFilter(
            date_from="2026-01-01",
            date_to="2026-12-31",
            product="billing",
            department="finance",
            tags=("年卡",),
            allowed_access=("public",),
        )
        hits_2026 = mem.search_hits("退钱", metadata_filter=f_2026)
        assert [h.path for h in hits_2026] == ["data/memory/billing_2026.md"]
        assert cosine_calls == 1

        # 2d. 无 Frontmatter 的 plain.md 在默认 public 权限下正常命中，零破坏
        plain_hits = mem.search_hits("Atlas 9092")
        assert plain_hits and plain_hits[0].path == "data/memory/plain.md"

        # 2e. 不可满足权限集合零命中且零 embed 调用
        emb.calls.clear()
        assert mem.search_hits("退款", metadata_filter=MetadataFilter(allowed_access=())) == []
        assert emb.calls == []

        # 2f. index_status 报告 metadata_docs 计数
        status = ret.index_status(mem, root="data/memory")
        assert int(status["metadata_docs"]) == 3
    finally:
        mem.close()

    # 3. 同步验证 LinearRetriever（无外挂插件兜底）与 RAGRetriever 权限/元数据过滤口径完全一致
    lin_mem = MemoryStore(data_dir, create_dirs=True)
    try:
        assert lin_mem.search_hits("高管薪资机密方案") == []
        lin_hr = lin_mem.search_hits("高管薪资机密方案", metadata_filter=MetadataFilter(allowed_access=("public", "hr")))
        assert lin_hr and lin_hr[0].path == "data/memory/hr_salary.md"
        lin_2026 = lin_mem.search_hits(
            "退款",
            metadata_filter=MetadataFilter(date_from="2026-01-01", product="billing", allowed_access=("public",)),
        )
        assert [h.path for h in lin_2026] == ["data/memory/billing_2026.md"]
    finally:
        lin_mem.close()


# 测试 AssistantApp 系统级权限锁死、search_markdown(filters=...) 防提级、CLI /rag query --filter 与 schema v3->v4 保留双缓存
def test_app_tool_anti_escalation_cli_filter_and_schema_v4_preserves_caches(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = tmp_path / "app_data"
    (data_dir / "memory").mkdir(parents=True)
    (data_dir / "memory" / "pub.md").write_text(
        "---\ndate: 2026-06-10\nproduct: billing\naccess: public\n---\n# 计费\n公开退款说明：7天无理由。\n",
        encoding="utf-8",
    )
    (data_dir / "memory" / "sec.md").write_text(
        "---\ndate: 2026-06-10\nproduct: billing\naccess: confidential\n---\n# 机密\n高管内部定价底线：1折特批。\n",
        encoding="utf-8",
    )

    app_pub = AssistantApp(
        data_root=data_dir,
        config_path=repo_root / "config" / "tools.json",
        project_root=repo_root,
        llm=_DummyLLM(),
        auto_consolidate=False,
    )
    try:
        # 1. 模型调用 search_markdown 即使恶意传入 filters="access=confidential" 也绝不能越权提级！
        blocked = app_pub.tools.execute(
            "search_markdown",
            json.dumps({"query": "高管内部定价底线", "filters": "access=confidential"}, ensure_ascii=False),
            context={"allowed_access": app_pub.allowed_access},
        )
        assert "1折特批" not in blocked

        # 2. filters="" 与正常缩小过滤（product=billing;date_from=2026-06-01）均正常生效
        ok_empty = app_pub.tools.execute(
            "search_markdown",
            json.dumps({"query": "7天无理由", "filters": ""}, ensure_ascii=False),
        )
        assert "7天无理由" in ok_empty
        ok_narrow = app_pub.tools.execute(
            "search_markdown",
            json.dumps({"query": "7天无理由", "filters": "product=billing;date_from=2026-06-01"}, ensure_ascii=False),
        )
        assert "7天无理由" in ok_narrow

        # 3. CLI /rag query --filter 条件 问题
        cli_out = _handle_repl_command(app_pub, "/rag query --filter product=billing;date_from=2026-06-01 7天无理由")
        assert cli_out is not None and "7天无理由" in cli_out and "【过滤】" in cli_out

        cli_miss = _handle_repl_command(app_pub, "/rag query --filter product=other 7天无理由")
        assert cli_miss is not None and "公开退款说明" not in cli_miss and "未在 data/memory 下找到" in cli_miss
    finally:
        app_pub.close()

    # 4. 验证旧 schema v3 原地升级至 v4 时，已付费生成的 chunk_contexts 与 chunk_embeddings 缓存表完整保留
    import sqlite3
    db_file = data_dir / "state" / "rag_index.db"
    conn = sqlite3.connect(str(db_file))
    try:
        conn.execute("UPDATE index_meta SET value = '3' WHERE key = 'schema_version'")
        conn.execute(
            "INSERT OR REPLACE INTO chunk_contexts (content_hash, model, summary, created_at) VALUES ('h_ctx', 'm', '保留摘要', '2026-06-01')"
        )
        conn.execute(
            "INSERT OR REPLACE INTO chunk_embeddings (content_hash, model, dims, vector_blob, created_at) VALUES ('h_emb', 'm', 1, X'0000803F', '2026-06-01')"
        )
        conn.commit()
    finally:
        conn.close()

    # 5. 当系统侧显式授予 allowed_access=("public", "confidential") 时可查到机密文档，且升级后双缓存表完好
    app_priv = AssistantApp(
        data_root=data_dir,
        config_path=repo_root / "config" / "tools.json",
        project_root=repo_root,
        llm=_DummyLLM(),
        allowed_access=("public", "confidential"),
        auto_consolidate=False,
    )
    try:
        res = app_priv.smart_search("高管内部定价底线", force_enable=False)
        assert "1折特批" in str(res["rendered"])
        status = app_priv.memory.retriever.index_status(app_priv.memory, root="data/memory")
        assert status["schema_version"] == "4"
        assert int(status["cached_contexts"]) >= 1 and int(status["cached_embeddings"]) >= 1
    finally:
        app_priv.close()


# 测试垂直领域 Embedding 微调三元组导出（自动挖掘 BM25 难负样本、隔离过期与越权文档、独立脚本语法与 CLI 入口）
def test_embedding_finetune_triplet_export_hard_negatives_and_security(tmp_path: Path) -> None:
    from evals.embedding_finetune import export_embedding_triplets

    data_dir = tmp_path / "domain_data"
    mem_dir = data_dir / "memory"
    mem_dir.mkdir(parents=True)
    (mem_dir / "refund_card.md").write_text(
        "---\ndate: 2026-06-01\nproduct: billing\naccess: public\n---\n"
        "# 极客年卡退款政策\n"
        "极客会员年卡购买后 7 天内支持无理由原路退回，超过 7 天按未用天数折算。\n"
        "- 2025 临时双十一活动已结束 (until: 2025-11-12)\n",
        encoding="utf-8",
    )
    (mem_dir / "refund_book.md").write_text(
        "---\ndate: 2026-06-02\nproduct: store\naccess: public\n---\n"
        "# 纸质图书退款政策\n"
        "纸质图书拆封后不支持无理由退款，如遇印刷缺页可免费换货。\n",
        encoding="utf-8",
    )
    (mem_dir / "hr_secret.md").write_text(
        "---\ndate: 2026-06-03\nproduct: hr\naccess: confidential\n---\n"
        "# 高管退款审批机密\n"
        "内部高管专属机密薪酬与特批额度。\n",
        encoding="utf-8",
    )

    cases_file = tmp_path / "domain_cases.jsonl"
    cases_file.write_text(
        json.dumps(
            {
                "id": "c1",
                "query": "极客会员年卡支持无理由退款吗？",
                "expected_sources": [{"path": "data/memory/refund_card.md", "quote": "7 天内支持无理由原路退回"}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    # 1. 有标注问答模式：anchor=问题，positive=目标年卡退款块，BM25 难负样本精准挖出词面高度相似的纸质图书退款块，绝不混入 confidential/过期条目
    out_supervised = tmp_path / "triplets_supervised.jsonl"
    summary_sup = export_embedding_triplets(
        data_dir,
        root="data/memory",
        cases_path=cases_file,
        output_path=out_supervised,
        today=date(2026, 6, 15),
    )
    assert summary_sup["triplets_exported"] == 1
    assert summary_sup["hard_negative_hits"] == 1
    rows = [json.loads(line) for line in out_supervised.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["anchor"] == "极客会员年卡支持无理由退款吗？"
    assert "7 天内支持无理由原路退回" in rows[0]["positive"]
    assert "纸质图书" in rows[0]["negative"]
    # 校验 BM25 难负样本与正样本严格同分布（绝不带 `"  <行号>: "` 渲染前缀防对比学习捷径污染）
    import re
    assert not re.match(r"^\s*\d+:\s", rows[0]["negative"])
    assert not rows[0]["negative"].startswith("  ")
    assert "高管" not in rows[0]["negative"] and "双十一" not in rows[0]["positive"]

    # 2. 无标注自监督模式（直接扫文档库）：产出跨文档难负样本，且绝不泄露 confidential 或已过期行
    out_unsup = tmp_path / "triplets_unsup.jsonl"
    summary_unsup = export_embedding_triplets(
        data_dir,
        root="data/memory",
        output_path=out_unsup,
        today=date(2026, 6, 15),
    )
    assert int(summary_unsup["triplets_exported"]) >= 2
    unsup_text = out_unsup.read_text(encoding="utf-8")
    assert "内部高管专属机密" not in unsup_text and "双十一" not in unsup_text

    # 3. 禁止将导出文件写进 --data-dir 内部（防止污染知识库）
    with pytest.raises(ValueError):
        export_embedding_triplets(data_dir, output_path=data_dir / "leak.jsonl", today=date(2026, 6, 15))

    # 4. evals CLI `--mode export-triplets` 端到端跑通，且 scripts/finetune_embedding.py --help 零依赖可正常响应
    repo_root = Path(__file__).resolve().parent.parent
    out_cli = tmp_path / "triplets_cli.jsonl"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "evals",
            "--mode",
            "export-triplets",
            "--data-dir",
            str(data_dir),
            "--cases",
            str(cases_file),
            "--output",
            str(out_cli),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    assert '"triplets_exported": 1' in proc.stdout and out_cli.is_file()

    help_proc = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "finetune_embedding.py"), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert "--triplets" in help_proc.stdout and "--base-model" in help_proc.stdout


# 对抗审查回归测试（B1-B4、R2-B1 与 Nit 1-7）：Frontmatter Fail-Closed、预绑定权限、离线 enrich/embed 零泄露、CLI 引号过滤与钳制
def test_review_regressions_frontmatter_fail_closed_prebound_access_and_offline_isolation(tmp_path: Path) -> None:
    from plugins.rag_retriever import RAGConfig, RAGRetriever

    # 1. B1 + R2-B1: 列表权限、嵌套 YAML、块标量、多行列表、带点键名、引号空串遮蔽、方括号标题/行内注释/未闭合 `[` 与超长未闭合均严格 Fail-Closed
    for bad_md, expected_access in (
        ("---\naccess: [confidential]\n---\n正文\n", "confidential"),
        ("---\naccess:\n  - hr\n---\n正文\n", "hr"),
        (
            "---\naccess: confidential\nmeta:\n  owner: hr\ndescription: |\n  多行说明\ntags: [\n  a,\n  b\n]\ndoc.id: 123\n---\n正文\n",
            "confidential",
        ),
        ("---\naccess: \"\"\npermission: confidential\n---\n正文\n", "confidential"),
        ("---\naccess: \"\"\n---\n正文\n", "__restricted__"),
        ("---\naccess: [hr, finance]\n---\n正文\n", "__restricted__"),
        ("---\naccess: confidential\n" + ("tag_pad: x\n" * 270) + "---\n正文\n", "confidential"),
        ("---\ntitle: [Internal] Q4 Salary\naccess: confidential\n---\n正文\n", "confidential"),
        ("---\ntags: [a, b] # ok\naccess: confidential\n---\n正文\n", "confidential"),
        ("---\ntags: [\n  a,\n  b\n] # end\naccess: confidential\n---\n正文\n", "confidential"),
        ("---\ntitle: [Draft Q4\naccess: confidential\n---\n正文\n", "confidential"),
    ):
        m, _ = parse_markdown_frontmatter(bad_md)
        assert m.access == expected_access and m.access != "public"

    m_comment, _ = parse_markdown_frontmatter("---\ntags: [a, b] # ok\naccess: confidential\n---\n正文\n")
    assert m_comment.tags == ("a", "b") and m_comment.access == "confidential"

    # Nit 5: parse_filter_string 引号权限、空值/重复单值键校验
    assert parse_filter_string('access="public"', allowed_access=("public",)).allowed_access == ("public",)
    for bad_f in ('product=""', "tags=,,,", "product=a;product=b", "access=*"):
        with pytest.raises(ValueError):
            parse_filter_string(bad_f, allowed_access=("public",))

    # 2. B2 + B3 + Nit 1 + Nit 2: 预绑定 allowed_access 生效（不传 context）、app 级钳制、/rag enrich|embed 零泄露机密与 --filter= 语法
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = tmp_path / "sec_app"
    (data_dir / "memory").mkdir(parents=True)
    (data_dir / "memory" / "pub.md").write_text(
        "---\ndate: 2026-06-10\nproduct: billing\naccess: public\n---\n# 计费\n公开退款说明：7天无理由。\n",
        encoding="utf-8",
    )
    (data_dir / "memory" / "sec.md").write_text(
        "---\ndate: 2026-06-10\nproduct: billing\naccess: confidential\n---\n# 机密\n高管内部定价底线：1折特批。\n",
        encoding="utf-8",
    )

    # 录制全部被发送至外部 LLM 的 Prompt，验证 /rag enrich 绝不发送机密文档
    class _SpyEnrichLLM:
        last_usage: TokenUsage | None = None

        # 初始化提示词日志
        def __init__(self) -> None:
            self.prompts: list[str] = []

        # 流式返回简短摘要
        def stream(self, context: Context, options: StreamOptions | None = None):
            msg = self.complete(context, options)
            yield StreamDone(reason="stop", message=msg)

        # 同步记录系统与用户提示词并返回摘要
        def complete(self, context: Context, options: StreamOptions | None = None) -> AssistantMessage:
            self.prompts.append(context.system_prompt + "\n" + "\n".join(str(m.content) for m in context.messages))
            return AssistantMessage.from_text("公开计费文档背景摘要。", stop_reason="stop")

    spy_llm = _SpyEnrichLLM()
    spy_emb = _ScopedEmbedder()

    # 2a. allowed_access=()（全拒绝）：不传 context 直接 execute 必须返回无结果，且 app.search / smart_search 严格钳制
    app_deny = AssistantApp(
        data_root=data_dir,
        config_path=repo_root / "config" / "tools.json",
        project_root=repo_root,
        llm=_DummyLLM(),
        consolidator_llm=spy_llm,
        allowed_access=(),
        auto_consolidate=False,
    )
    try:
        out_deny = app_deny.tools.execute("search_markdown", '{"query": "7天无理由"}')
        assert "公开退款说明" not in out_deny
        assert "公开退款说明" not in app_deny.search("7天无理由", metadata_filter=MetadataFilter(allowed_access=("public",)))
        res_deny = app_deny.smart_search("7天无理由", metadata_filter=MetadataFilter(allowed_access=("public",)))
        assert res_deny["hits"] == [] and spy_llm.prompts == []
    finally:
        app_deny.close()

    # 2b. allowed_access=("public",)：/rag enrich 与 /rag embed 绝不把 confidential 文档外发；--filter= 与引号过滤生效
    app_pub = AssistantApp(
        data_root=data_dir,
        config_path=repo_root / "config" / "tools.json",
        project_root=repo_root,
        llm=_DummyLLM(),
        consolidator_llm=spy_llm,
        embedder=spy_emb,
        allowed_access=("public",),
        auto_consolidate=False,
    )
    try:
        _handle_repl_command(app_pub, "/rag enrich data/memory")
        assert spy_llm.prompts and all("1折特批" not in p for p in spy_llm.prompts)
        _handle_repl_command(app_pub, "/rag embed data/memory")
        assert spy_emb.calls and all("1折特批" not in txt for batch in spy_emb.calls for txt in batch)

        # Nit 2: app.search / smart_search 传入越权 metadata_filter 仍被钳制为 0 命中
        assert "1折特批" not in app_pub.search("高管内部定价底线", metadata_filter=MetadataFilter(allowed_access=("confidential",)))
        assert app_pub.smart_search("高管内部定价底线", metadata_filter=MetadataFilter(allowed_access=("confidential",)))["hits"] == []

        # Nit 1 + R2 Nit 2: --filter= 与引号含空格过滤；越权过滤明确提示权限拦截而非误报未配模型
        out_eq = _handle_repl_command(app_pub, "/rag query --filter=product=billing 7天无理由")
        assert out_eq is not None and "公开退款说明" in out_eq and "【过滤】" in out_eq
        out_q = _handle_repl_command(app_pub, '/rag query --filter "product=billing; date_from=2026-06-01" 7天无理由')
        assert out_q is not None and "公开退款说明" in out_q and "【过滤】" in out_q
        out_denied = _handle_repl_command(app_pub, "/rag query --filter access=confidential 机密")
        assert out_denied is not None and "权限边界" in out_denied and "未配置可用大模型" not in out_denied
    finally:
        app_pub.close()

    # 2c. allowed_access=("public", "confidential")：不传 context 直接 execute 应保留预绑定机密权限
    app_priv = AssistantApp(
        data_root=data_dir,
        config_path=repo_root / "config" / "tools.json",
        project_root=repo_root,
        llm=_DummyLLM(),
        allowed_access=("public", "confidential"),
        auto_consolidate=False,
    )
    try:
        out_priv = app_priv.tools.execute("search_markdown", '{"query": "高管内部定价底线"}')
        assert "1折特批" in out_priv
    finally:
        app_priv.close()

    # 3. Nit 7: scripts/finetune_embedding.py 拒绝非法超参与非 JSON 对象行
    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text('["not_a_dict"]\n', encoding="utf-8")
    from scripts.finetune_embedding import load_triplets, main as ft_main

    with pytest.raises(ValueError):
        load_triplets(bad_jsonl)
    with pytest.raises(ValueError):
        ft_main(["--triplets", str(bad_jsonl), "--output-dir", str(tmp_path / "out"), "--epochs", "0"])

    # 4. Nit 3: 大量其他产品线含高频词干扰块时，极小过滤子集（<10 块）的 BM25 检索绝不被全表 COUNT(*) 误杀
    skew_dir = tmp_path / "skew"
    (skew_dir / "memory").mkdir(parents=True)
    for i in range(12):
        (skew_dir / "memory" / f"other_{i}.md").write_text(
            f"---\nproduct: noise\naccess: public\n---\n# 干扰 {i}\n平台通用配置说明项 {i}。\n",
            encoding="utf-8",
        )
    (skew_dir / "memory" / "target.md").write_text(
        "---\nproduct: billing\naccess: public\n---\n# 目标\n平台通用配置说明：年卡计费。\n",
        encoding="utf-8",
    )
    skew_mem = MemoryStore(skew_dir, create_dirs=True, retriever=RAGRetriever(RAGConfig()))
    try:
        # 查询含未出现词以使 coverage < 0.6 且触发 bm25_val < 0.1，真正检验 filtered total_children < 10 放行
        hits = skew_mem.search_hits("平台 通用 缺失甲 缺失乙 缺失丙", metadata_filter=MetadataFilter(product="billing"))
        assert len(hits) == 1 and hits[0].path == "data/memory/target.md"
    finally:
        skew_mem.close()

