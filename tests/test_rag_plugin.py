"""持久化结构化 RAG 检索插件单元测试（TDD）：覆盖插件发现、结构块完整召回、派生库持久化/重建、过期隔离与只读边界。"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from evals import main as eval_main
from personal_assistant.agent.plugins import PluginManager
from personal_assistant.ai import NoopModelClient
from personal_assistant.assistant.app import AssistantApp
from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.assistant.memory import MemoryStore, SearchHit


# 测试插件加载器能从 plugins/ 自动发现 rag 检索插件，并校验配置参数与 tools.json 切换
def test_rag_plugin_discovery_and_tools_config_toggle(tmp_path: Path) -> None:
    from plugins.rag_retriever import RAGConfig

    with pytest.raises(ValueError):
        RAGConfig.from_dict({"min_term_coverage": True})
    with pytest.raises(ValueError):
        RAGConfig.from_dict({"min_term_coverage": 1.5})
    with pytest.raises(ValueError):
        RAGConfig.from_dict({"min_bm25_score": -0.1})
    with pytest.raises(ValueError):
        RAGConfig.from_dict({"max_hits_per_file": 0})
    with pytest.raises(ValueError):
        RAGConfig(max_hits_per_file=False)
    with pytest.raises(ValueError):
        RAGConfig.from_dict({"chunk_max_chars": 0})
    with pytest.raises(ValueError):
        RAGConfig.from_dict({"chunk_max_chars": True})

    repo_root = Path(__file__).resolve().parent.parent
    manager = PluginManager.from_entries(["plugins"], repo_root)
    try:
        assert "rag" in manager.retrievers
    finally:
        manager.terminate_all()

    cfg_path = tmp_path / "tools.json"
    cfg_path.write_text(
        json.dumps(
            {
                "enabled_tools": ["search_markdown"],
                "plugins": ["plugins"],
                "retrieval": {"plugin": "rag", "min_term_coverage": 0.25, "chunk_max_chars": 500},
                "files": {"read_roots": ["data/memory"], "write_root": "data/workspace"},
                "commands": [],
            }
        ),
        encoding="utf-8",
    )
    cfg = ToolsConfig.load(cfg_path, project_root=repo_root)
    try:
        active = cfg.build_retriever()
        assert active is not None
        assert getattr(active, "name", "") == "rag"
    finally:
        cfg.plugin_manager.terminate_all()

    # 切换为 linear 时回退到内置线性检索（build_retriever 返回 None）
    cfg_linear = tmp_path / "tools_linear.json"
    cfg_linear.write_text(
        json.dumps(
            {
                "enabled_tools": ["search_markdown"],
                "plugins": ["plugins"],
                "retrieval": {"plugin": "linear"},
                "files": {"read_roots": ["data/memory"], "write_root": "data/workspace"},
                "commands": [],
            }
        ),
        encoding="utf-8",
    )
    linear_cfg = ToolsConfig.load(cfg_linear, project_root=repo_root)
    try:
        assert linear_cfg.build_retriever() is None
    finally:
        linear_cfg.plugin_manager.terminate_all()


# 测试 RAG BM25 的结构块完整召回（代码块/表格与多级面包屑）、双语分词、干扰负例拦截与只读零写盘
def test_rag_retriever_structural_chunks_ranking_negatives_and_readonly(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    manager = PluginManager.from_entries(["plugins"], repo_root)
    retriever = manager.retrievers["rag"]({"min_term_coverage": 0.25, "max_hits_per_file": 2})

    data_dir = tmp_path / "data"
    mem_dir = data_dir / "memory"
    mem_dir.mkdir(parents=True)
    (mem_dir / "projects.md").write_text(
        "# 项目备忘\n\n"
        "## Atlas 数据同步服务\n"
        "Atlas 部署在东京机房，监听端口 9092，本地使用 SQLite WAL 模式。\n\n"
        "## Nova 遥测采集器\n"
        "Nova 部署在新加坡机房。\n"
        "Nova 上报接口监听端口为 8443。\n",
        encoding="utf-8",
    )
    (mem_dir / "dialog.md").write_text(
        "# LoCoMo Session 1\n"
        "- [D1:1] Caroline: Yesterday I researched adoption agencies for our family plan.\n"
        "- [D1:2] Melanie: What did you think about the weather this week?\n"
        "- [D1:3] Caroline: It was sunny and warm in the park.\n",
        encoding="utf-8",
    )
    (mem_dir / "code_notes.md").write_text(
        "# 算法笔记\n\n"
        "## 归并排序\n\n"
        "```python\n"
        "# 递归分治归并排序\n"
        "def merge_sort_demo(nums: list[int]) -> list[int]:\n"
        "    if len(nums) <= 1:\n"
        "        return nums\n"
        "    mid = len(nums) // 2\n"
        "    return nums[:mid] + nums[mid:]\n"
        "```\n",
        encoding="utf-8",
    )

    try:
        memory = MemoryStore(data_dir, create_dirs=False, retriever=retriever)

        # 1. 英文疑问句含大量停用词 + 词形变化（research vs researched, agency vs agencies）应精准命中 D1:1
        hits_en = memory.search_hits("What did Caroline research about adoption agency?", limit=3)
        assert hits_en and isinstance(hits_en[0], SearchHit)
        assert "[D1:1]" in hits_en[0].snippet
        assert hits_en[0].start_line >= 1 and hits_en[0].end_line >= hits_en[0].start_line

        # 2. 中文查询命中正确章节与端口，且 search_markdown 输出精确行号区间格式
        hits_zh = memory.search_hits("Nova 遥测采集器上报端口", limit=3)
        assert hits_zh and "8443" in hits_zh[0].snippet
        assert hits_zh[0].section == "项目备忘 > Nova 遥测采集器"
        formatted = memory.search_markdown("Nova 遥测采集器上报端口", limit=3)
        assert f"{hits_zh[0].path}:{hits_zh[0].start_line}-{hits_zh[0].end_line}" in formatted

        # 3. 命中代码块时返回完整未截断代码块与正确起止行号和标题面包屑
        hits_code = memory.search_hits("merge_sort_demo 递归分治", limit=3)
        assert hits_code
        assert hits_code[0].section == "算法笔记 > 归并排序"
        assert "```python" in hits_code[0].snippet
        assert "return nums[:mid] + nums[mid:]" in hits_code[0].snippet
        assert "```" in hits_code[0].snippet.splitlines()[-1]
        assert (hits_code[0].start_line, hits_code[0].end_line) == (5, 12)

        # 4. 高相似干扰负例（仅命中一个实体词 Atlas，其余高信息词全无）应返回空列表
        assert memory.search_hits("Atlas Redis 集群密码和哨兵主从切换地址", limit=5) == []

        # 5. 只读模式（create_dirs=False）绝不往 data_dir 写入任何 SQLite 索引文件
        created_files = sorted(p.relative_to(data_dir).as_posix() for p in data_dir.rglob("*"))
        assert created_files == ["memory", "memory/code_notes.md", "memory/dialog.md", "memory/projects.md"]
    finally:
        manager.terminate_all()


# 测试派生索引库持久化到 data/state/rag_index.db、跨实例重启复用、删库无损重建、损坏自愈与符号链接拒绝
def test_rag_retriever_persistent_derived_db_lifecycle_and_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from plugins.rag_retriever import RAGConfig, RAGRetriever

    data_dir = tmp_path / "data"
    mem_dir = data_dir / "memory"
    mem_dir.mkdir(parents=True)
    note = mem_dir / "guide.md"
    note.write_text("# 指南\n\n## 持久化\n派生索引独立存放在 state/rag_index.db。\n", encoding="utf-8")

    # 1. 读写模式下首次搜索在 data/state/rag_index.db 持久化派生索引（不改动源文件，不生成 assistant.db）
    r1 = RAGRetriever(RAGConfig(chunk_max_chars=400))
    mem1 = MemoryStore(data_dir, create_dirs=True, retriever=r1)
    try:
        hits1 = mem1.search_hits("派生索引 state")
        assert hits1 and "rag_index.db" in hits1[0].snippet
        db_path = data_dir / "state" / "rag_index.db"
        assert db_path.is_file()
        assert not (data_dir / "state" / "assistant.db").exists()
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT content_hash, start_line, end_line FROM chunks LIMIT 1").fetchone()
            assert row is not None and len(row[0]) == 64 and row[1] <= row[2]
    finally:
        mem1.close()

    # 2. 新建 RAGRetriever 实例重启后，对未修改文件零重读即可直接复用磁盘索引命中
    r2 = RAGRetriever(RAGConfig(chunk_max_chars=400))
    mem2 = MemoryStore(data_dir, create_dirs=True, retriever=r2)
    try:
        orig_read_text = Path.read_text

        # 若未修改文件仍触发磁盘重读则立即失败，验证增量指纹持久化跨重启生效
        def _forbid_read_guide(self: Path, *args: object, **kwargs: object) -> str:
            if self.resolve() == note.resolve():
                raise AssertionError("未变更文件不应在重启后被重新读取分块")
            return orig_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "read_text", _forbid_read_guide)
        hits2 = mem2.search_hits("派生索引 state")
        assert hits2 and "rag_index.db" in hits2[0].snippet
    finally:
        monkeypatch.undo()
        mem2.close()

    # 3. 删掉派生库 rag_index.db 后，下次查询自动无损重建
    for suffix in ("", "-wal", "-shm"):
        candidate = data_dir / "state" / f"rag_index.db{suffix}"
        if candidate.exists():
            candidate.unlink()
    r3 = RAGRetriever(RAGConfig(chunk_max_chars=400))
    mem3 = MemoryStore(data_dir, create_dirs=True, retriever=r3)
    try:
        hits3 = mem3.search_hits("派生索引 state")
        assert hits3 and (data_dir / "state" / "rag_index.db").is_file()
    finally:
        mem3.close()

    # 4. 派生库文件损坏时自动清理损坏派生库并从权威 Markdown 重建
    (data_dir / "state" / "rag_index.db").write_bytes(b"corrupted-not-a-sqlite-header" * 16)
    r4 = RAGRetriever(RAGConfig(chunk_max_chars=400))
    mem4 = MemoryStore(data_dir, create_dirs=True, retriever=r4)
    try:
        hits4 = mem4.search_hits("派生索引 state")
        assert hits4 and "rag_index.db" in hits4[0].snippet
    finally:
        mem4.close()

    # 5. 分块配置（chunk_max_chars）变更时自动失效并重建派生索引
    r5 = RAGRetriever(RAGConfig(chunk_max_chars=800))
    mem5 = MemoryStore(data_dir, create_dirs=True, retriever=r5)
    try:
        hits5 = mem5.search_hits("派生索引 state")
        assert hits5
        with sqlite3.connect(data_dir / "state" / "rag_index.db") as conn:
            meta = dict(conn.execute("SELECT key, value FROM index_meta").fetchall())
            assert meta.get("chunk_max_chars") == "800"
    finally:
        mem5.close()

    # 6. 符号链接替换 rag_index.db 时必须显式拒绝，防止借派生库路径写越界文件
    outside = tmp_path / "outside.db"
    db_file = data_dir / "state" / "rag_index.db"
    db_file.unlink()
    os.symlink(outside, db_file)
    r6 = RAGRetriever(RAGConfig(chunk_max_chars=800))
    mem6 = MemoryStore(data_dir, create_dirs=True, retriever=r6)
    try:
        with pytest.raises(ValueError, match="符号链接"):
            mem6.search_hits("派生索引 state")
    finally:
        mem6.close()


# 测试 SQL 级过期过滤（today 边界）、归档屏蔽、文件更新增量同步与高频主题 IDF 托底
def test_rag_retriever_temporal_expiry_archive_incremental_and_idf_floor(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    manager = PluginManager.from_entries(["plugins"], repo_root)
    retriever = manager.retrievers["rag"]({})

    data_dir = tmp_path / "data"
    mem_dir = data_dir / "memory"
    (mem_dir / "archive").mkdir(parents=True)
    sched = mem_dir / "schedule.md"
    sched.write_text(
        "# 日程\n"
        "- [2026-11-15] 参加 Atlas 2.0 冬季压测演练。 (until: 2026-11-20)\n"
        "- [2026-10-01] 参加旧版迁移评审会。 (until: 2026-10-15)\n",
        encoding="utf-8",
    )
    (mem_dir / "archive" / "2026-10.md").write_text(
        "# 归档\n- [2026-10-01] 已归档的神秘火星计划。\n",
        encoding="utf-8",
    )

    try:
        memory = MemoryStore(data_dir, create_dirs=False, retriever=retriever)

        # 截止当日（until == today）应命中
        on_day = memory.search_hits("冬季压测演练", today=date(2026, 11, 20))
        assert on_day and "冬季压测演练" in on_day[0].snippet

        # 截止次日（until < today）应被 SQL 直接过滤，无需重建索引
        assert memory.search_hits("冬季压测演练", today=date(2026, 11, 21)) == []

        # 归档目录默认不参与检索
        assert memory.search_hits("神秘火星计划", today=date(2026, 11, 20)) == []

        # 动态改写磁盘文件后，增量同步应立刻反映最新内容并移除旧内容
        sched.write_text("# 日程\n- [2026-11-21] 启动星链遥测新里程碑。\n", encoding="utf-8")
        updated_hits = memory.search_hits("星链遥测新里程碑", today=date(2026, 11, 21))
        assert updated_hits and "星链遥测新里程碑" in updated_hits[0].snippet
        assert memory.search_hits("旧版迁移评审会", today=date(2026, 10, 10)) == []

        # 高频主题词出现在 >=10 独立分块中（SQLite FTS5 BM25 IDF 钳位到 ~1e-6）时仍须命中且正分排序
        atlas_file = memory.memory_dir / "projects.md"
        atlas_file.write_text(
            "# 项目\n\n"
            + "\n\n".join(f"## 子项 {i}\nAtlas 子任务 {i} 使用 Go 服务。" for i in range(12)),
            encoding="utf-8",
        )
        atlas_hits = memory.search_hits("Atlas", today=date(2026, 11, 22))
        assert atlas_hits and atlas_hits[0].score > 0.0 and "Atlas" in atlas_hits[0].snippet
    finally:
        manager.terminate_all()


# 测试 AssistantApp 与 eval CLI 支持按配置或 --retriever 参数无缝切换 rag 与 linear
def test_app_and_eval_cli_integrate_rag_plugin(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    app = AssistantApp(
        data_root=tmp_path / "app_data",
        config_path=repo_root / "config" / "tools.json",
        project_root=repo_root,
        llm=NoopModelClient(),
    )
    try:
        status = app.status()
        assert status["retriever"] == "rag"
    finally:
        app.close()

    out_linear = tmp_path / "linear.json"
    out_rag = tmp_path / "rag.json"
    assert eval_main(["--mode", "search", "--retriever", "linear", "--output", str(out_linear)]) == 0
    assert eval_main(["--mode", "search", "--retriever", "rag", "--output", str(out_rag)]) == 0
    rep_linear = json.loads(out_linear.read_text(encoding="utf-8"))
    rep_rag = json.loads(out_rag.read_text(encoding="utf-8"))
    assert rep_linear["retriever"] == "linear"
    assert rep_rag["retriever"] == "rag"


# 测试 Small-to-Big 上下文扩展守住边界（同章节无缝补齐、绝不跨同名标题、绝不拉入过期块、同文件零重复行）且 terminate() 关闭连接
def test_rag_small_to_big_boundaries_dedup_and_plugin_terminate(tmp_path: Path) -> None:
    from plugins.rag_retriever import RAGConfig, RAGRetriever

    data_dir = tmp_path / "data"
    mem_dir = data_dir / "memory"
    mem_dir.mkdir(parents=True)

    # 1. 同章节多块连续问答 + 中间夹独立过期行 + 后接同名标题新章节
    (mem_dir / "study.md").write_text(
        "# 学习记录\n\n"
        "## 章节\n"
        "第一段：关于量子退相干机制的前置推导与背景假设说明，这里刻意写满较长文本确保独立成块。\n\n"
        "第二段：承接上文给出核心公式 ZETA_ANCHOR_42 与实验结论总结。\n\n"
        "- [2026-10-01] 过期干扰项 EXPIRED_LEAK_999。 (until: 2026-10-15)\n\n"
        "第三段：同一章节下过期项之后的另一条独立结论 OMEGA_POST_88。\n\n"
        "## 章节\n"
        "第四段：同名新章节正文 DUPLICATE_HEADING_BLOCK，严禁被上一章节扩展越界带出。\n",
        encoding="utf-8",
    )

    retriever = RAGRetriever(RAGConfig(chunk_max_chars=50, context_max_chars=400))
    memory = MemoryStore(data_dir, create_dirs=True, retriever=retriever)
    try:
        # 命中第二段应向左拉入同章节第一段，但绝不能向右拉入过期条目
        hits = memory.search_hits("ZETA_ANCHOR_42", today=date(2026, 11, 20), limit=5)
        assert hits
        assert "第一段" in hits[0].snippet
        assert "ZETA_ANCHOR_42" in hits[0].snippet
        assert "EXPIRED_LEAK_999" not in hits[0].snippet

        # 命中第三段绝不能跨过过期条目拉入第二段，也绝不能跨过同名 ## 章节拉入第四段
        post_hits = memory.search_hits("OMEGA_POST_88", today=date(2026, 11, 20), limit=5)
        assert post_hits
        assert "OMEGA_POST_88" in post_hits[0].snippet
        assert "EXPIRED_LEAK_999" not in post_hits[0].snippet
        assert "DUPLICATE_HEADING_BLOCK" not in post_hits[0].snippet

        # 同文件双命中（chunk 0 与 chunk 2）不得把中间共享邻居 chunk 1 重复拼进两条命中
        (mem_dir / "dedup.md").write_text(
            "# 去重验证\n\n"
            "第一块包含 ALPHA_DUP_TOKEN 并且包含足够长的描述文本以强制单独切分。\n\n"
            "第二块是中间共享桥接段落 SHARED_BRIDGE_LINE_777，本身不含目标锚点。\n\n"
            "第三块包含 ALPHA_DUP_TOKEN 与额外补充说明文本以确保单独切分为第三块。\n",
            encoding="utf-8",
        )
        dup_hits = memory.search_hits("ALPHA_DUP_TOKEN", today=date(2026, 11, 20), limit=5)
        same_file = [h for h in dup_hits if h.path.endswith("dedup.md")]
        assert len(same_file) == 2
        # 两条命中的行号区间必须完全不相交，中间桥接行最多出现一次
        a, b = sorted(same_file, key=lambda h: h.start_line)
        assert a.end_line < b.start_line
        assert sum("SHARED_BRIDGE_LINE_777" in h.snippet for h in same_file) <= 1
    finally:
        memory.close()

    # 2. 通过 PluginManager 创建的 RAGRetriever 在 terminate_all() 后底层 SQLite 连接已彻底关闭
    repo_root = Path(__file__).resolve().parent.parent
    manager = PluginManager.from_entries(["plugins"], repo_root)
    managed_ret = manager.retrievers["rag"]({})
    managed_mem = MemoryStore(data_dir, create_dirs=True, retriever=managed_ret)
    assert managed_mem.search_hits("ZETA_ANCHOR_42", today=date(2026, 11, 20))
    assert len(managed_ret._indices) == 1
    manager.terminate_all()
    assert len(managed_ret._indices) == 0


# 回归 B1 / B2 与 Nits 2-5：index_path 边界、祖先与 WAL 符号链接拦截、只读零写盘、锁竞争绝不删库、运行期损坏自愈、换 Inode 与删光清空、并发安全
def test_rag_security_boundaries_concurrency_and_recovery_regressions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import concurrent.futures
    from plugins.rag_retriever import RAGConfig, RAGRetriever

    # 1. RAGConfig 拒绝绝对路径、.. 越界路径与浮点数截断
    with pytest.raises(ValueError):
        RAGConfig(index_path="/tmp/evil.db")
    with pytest.raises(ValueError):
        RAGConfig(index_path="../outside.db")
    with pytest.raises(ValueError):
        RAGConfig.from_dict({"chunk_max_chars": 1.9})
    with pytest.raises(ValueError):
        RAGConfig.from_dict({"max_hits_per_file": float("inf")})

    # 2. read_only=True 即使配置了 index_path 也严守零写盘，绝不创建派生文件
    ro_data = tmp_path / "ro_data"
    (ro_data / "memory").mkdir(parents=True)
    (ro_data / "memory" / "a.md").write_text("# 只读\n只读模式测试内容 ALPHA_RO。\n", encoding="utf-8")
    ro_mem = MemoryStore(ro_data, create_dirs=False, retriever=RAGRetriever(RAGConfig(index_path="state/custom.db")))
    try:
        assert ro_mem.search_hits("ALPHA_RO")
        assert not (ro_data / "state").exists()
    finally:
        ro_mem.close()

    # 3. 祖先目录符号链接在 mkdir 之前即被拦截，绝不在 data_root 外创建任何目录
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    anc_data = tmp_path / "anc_data"
    (anc_data / "memory").mkdir(parents=True)
    (anc_data / "memory" / "a.md").write_text("# 祖先\n测试祖先符号链接拦截。\n", encoding="utf-8")
    os.symlink(outside_dir, anc_data / "state")
    anc_mem = MemoryStore(
        anc_data,
        create_dirs=True,
        retriever=RAGRetriever(RAGConfig(index_path="state/nested/rag_index.db")),
    )
    try:
        with pytest.raises(ValueError, match="符号链接"):
            anc_mem.search_hits("祖先符号链接")
        assert not (outside_dir / "nested").exists()
    finally:
        anc_mem.close()

    # 4. WAL 附属文件符号链接在打开 SQLite 前即被拦截，保护外部目标文件不被覆写
    victim = tmp_path / "victim.txt"
    victim.write_text("KEEP_INTACT", encoding="utf-8")
    wal_data = tmp_path / "wal_data"
    (wal_data / "memory").mkdir(parents=True)
    (wal_data / "state").mkdir(parents=True)
    (wal_data / "memory" / "a.md").write_text("# WAL\n测试 WAL 符号链接拦截。\n", encoding="utf-8")
    os.symlink(victim, wal_data / "state" / "rag_index.db-wal")
    wal_mem = MemoryStore(wal_data, create_dirs=True, retriever=RAGRetriever())
    try:
        with pytest.raises(ValueError, match="符号链接"):
            wal_mem.search_hits("WAL 符号链接")
        assert victim.read_text(encoding="utf-8") == "KEEP_INTACT"
    finally:
        wal_mem.close()

    # 5. 非损坏型数据库异常（如 SQLITE_BUSY / database is locked）绝不误删活派生库
    busy_data = tmp_path / "busy_data"
    (busy_data / "memory").mkdir(parents=True)
    (busy_data / "memory" / "a.md").write_text("# 锁\n测试数据库锁竞争保护 BUSY_GUARD。\n", encoding="utf-8")
    busy_mem = MemoryStore(busy_data, create_dirs=True, retriever=RAGRetriever())
    try:
        assert busy_mem.search_hits("BUSY_GUARD")
    finally:
        busy_mem.close()
    busy_db = busy_data / "state" / "rag_index.db"
    before_bytes = busy_db.read_bytes()
    # 持有外部排他锁触发真实 OperationalError(database is locked)：分别覆盖 Schema 升级写锁与增量同步写锁
    locker = sqlite3.connect(str(busy_db), isolation_level=None)
    try:
        locker.execute("BEGIN EXCLUSIVE")
        from plugins.rag_retriever import _DerivedRAGIndex

        orig_conn_disk = _DerivedRAGIndex._connect_disk

        # 构造极短 busy_timeout 的磁盘连接模拟多进程高并发锁超时
        def _short_disk_connect(db_p: Path, *, busy_timeout_ms: int = 5000) -> sqlite3.Connection:
            return orig_conn_disk(db_p, busy_timeout_ms=20)

        monkeypatch.setattr(_DerivedRAGIndex, "_connect_disk", staticmethod(_short_disk_connect))
        # a) 变更 chunk_max_chars 迫使 _ensure_schema 走写事务并命中锁超时，绝不删除活派生库
        busy_mem2 = MemoryStore(busy_data, create_dirs=True, retriever=RAGRetriever(RAGConfig(chunk_max_chars=300)))
        try:
            with pytest.raises(sqlite3.OperationalError):
                busy_mem2.search_hits("BUSY_GUARD")
            assert busy_db.is_file() and busy_db.read_bytes() == before_bytes
        finally:
            busy_mem2.close()

        # b) 修改文档迫使 sync() 走 BEGIN IMMEDIATE 写事务并命中锁超时，同样绝不删除活派生库
        (busy_data / "memory" / "a.md").write_text("# 锁\n变更后的内容 BUSY_GUARD_V2。\n", encoding="utf-8")
        busy_mem3 = MemoryStore(busy_data, create_dirs=True, retriever=RAGRetriever())
        try:
            with pytest.raises(sqlite3.OperationalError):
                busy_mem3.search_hits("BUSY_GUARD_V2")
            assert busy_db.is_file() and busy_db.read_bytes() == before_bytes
        finally:
            busy_mem3.close()
            monkeypatch.undo()
    finally:
        locker.execute("ROLLBACK")
        locker.close()

    # 6. 外部重建换 Inode 检测、运行期损坏自愈、删光目录清空旧行与规范相对 root_key
    rec_data = tmp_path / "rec_data"
    (rec_data / "memory").mkdir(parents=True)
    doc = rec_data / "memory" / "note.md"
    doc.write_text("# 恢复\n初始关键内容 RECOVER_ALPHA。\n", encoding="utf-8")
    rec_ret = RAGRetriever()
    rec_mem = MemoryStore(rec_data, create_dirs=True, retriever=rec_ret)
    try:
        assert rec_mem.search_hits("RECOVER_ALPHA")
        rec_db = rec_data / "state" / "rag_index.db"
        with sqlite3.connect(rec_db) as conn:
            root_keys = {r[0] for r in conn.execute("SELECT DISTINCT root_key FROM chunks").fetchall()}
            assert root_keys == {"data/memory"}

        # 外部原子替换 rag_index.db 为全新 Inode（模拟另一进程重建），当前实例应感知换 Inode 并正常查询
        replacement = rec_data / "state" / "replacement.db"
        tmp_ret = RAGRetriever(RAGConfig(index_path="state/replacement.db"))
        tmp_mem = MemoryStore(rec_data, create_dirs=True, retriever=tmp_ret)
        try:
            assert tmp_mem.search_hits("RECOVER_ALPHA")
        finally:
            tmp_mem.close()
        os.replace(replacement, rec_db)
        assert rec_mem.search_hits("RECOVER_ALPHA")

        # 模拟运行期中查询阶段触发物理损坏（SQLITE_CORRUPT），应自动关闭重建并重试成功
        calls = {"n": 0}
        orig_sync_query = rec_ret._sync_and_query

        # 首次调用抛出 SQLITE_CORRUPT，验证 search_hits 清理损坏派生库并重试一次成功
        def _flaky_corrupt(*args: object, **kwargs: object) -> list[SearchHit]:
            calls["n"] += 1
            if calls["n"] == 1:
                err = sqlite3.DatabaseError("database disk image is malformed")
                err.sqlite_errorcode = sqlite3.SQLITE_CORRUPT  # type: ignore[attr-defined]
                raise err
            return orig_sync_query(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(rec_ret, "_sync_and_query", _flaky_corrupt)
        hits_after_corrupt = rec_mem.search_hits("RECOVER_ALPHA")
        assert calls["n"] == 2 and hits_after_corrupt
        monkeypatch.undo()

        # 删除目录内最后一个文件后，派生库旧行必须被同步清空
        doc.unlink()
        assert rec_mem.search_hits("RECOVER_ALPHA") == []
        with sqlite3.connect(rec_db) as conn:
            assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM indexed_files").fetchone()[0] == 0
    finally:
        rec_mem.close()

    # 7. 多线程并发 build_retriever 与并发 search_hits / terminate_all 零竞态
    conc_data = tmp_path / "conc_data"
    (conc_data / "memory").mkdir(parents=True)
    (conc_data / "memory" / "c.md").write_text("# 并发\n多线程安全访问 CONCURRENT_SAFE_88。\n", encoding="utf-8")
    repo_root = Path(__file__).resolve().parent.parent
    manager = PluginManager.from_entries(["plugins"], repo_root)
    try:
        # 并发调用插件工厂与搜索验证锁保护正常
        def _worker(_idx: int) -> int:
            ret = manager.retrievers["rag"]({})
            mem = MemoryStore(conc_data, create_dirs=True, retriever=ret)
            hits = mem.search_hits("CONCURRENT_SAFE_88")
            return len(hits)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(_worker, range(8)))
        assert all(r >= 1 for r in results)
    finally:
        manager.terminate_all()


