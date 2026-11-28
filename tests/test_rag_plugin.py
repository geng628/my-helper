"""持久化父子 RAG 检索插件单元测试（TDD）：覆盖父子两层表、parent_id 聚合去重、派生库持久化/自愈、安全边界与并发安全。"""
from __future__ import annotations

import concurrent.futures
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


# 测试插件加载器自动发现 rag 插件，并严格校验父子分块/Overlap/路径配置参数与 linear 切换
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
        RAGConfig.from_dict({"parent_max_chars": 0})
    with pytest.raises(ValueError):
        RAGConfig.from_dict({"child_max_chars": 0})
    with pytest.raises(ValueError):
        RAGConfig.from_dict({"child_overlap_chars": -1})
    with pytest.raises(ValueError):
        RAGConfig.from_dict({"child_max_chars": 100, "child_overlap_chars": 100})
    with pytest.raises(ValueError):
        RAGConfig.from_dict({"child_overlap_chars": True})
    with pytest.raises(ValueError):
        RAGConfig.from_dict({"parent_max_chars": 1.9})
    with pytest.raises(ValueError):
        RAGConfig.from_dict({"max_hits_per_file": float("inf")})

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
                "retrieval": {
                    "plugin": "rag",
                    "min_term_coverage": 0.25,
                    "parent_max_chars": 900,
                    "child_max_chars": 300,
                    "child_overlap_chars": 60,
                },
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


# 测试父子索引结构块召回、双语分词、干扰负例拦截与只读模式零写盘
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

        hits_en = memory.search_hits("What did Caroline research about adoption agency?", limit=3)
        assert hits_en and isinstance(hits_en[0], SearchHit)
        assert "[D1:1]" in hits_en[0].snippet
        assert hits_en[0].start_line >= 1 and hits_en[0].end_line >= hits_en[0].start_line

        hits_zh = memory.search_hits("Nova 遥测采集器上报端口", limit=3)
        assert hits_zh and "8443" in hits_zh[0].snippet
        assert hits_zh[0].section == "项目备忘 > Nova 遥测采集器"
        formatted = memory.search_markdown("Nova 遥测采集器上报端口", limit=3)
        assert f"{hits_zh[0].path}:{hits_zh[0].start_line}-{hits_zh[0].end_line}" in formatted

        hits_code = memory.search_hits("merge_sort_demo 递归分治", limit=3)
        assert hits_code
        assert hits_code[0].section == "算法笔记 > 归并排序"
        assert "```python" in hits_code[0].snippet
        assert "return nums[:mid] + nums[mid:]" in hits_code[0].snippet
        assert "```" in hits_code[0].snippet.splitlines()[-1]
        assert (hits_code[0].start_line, hits_code[0].end_line) == (5, 12)

        assert memory.search_hits("Atlas Redis 集群密码和哨兵主从切换地址", limit=5) == []

        created_files = sorted(p.relative_to(data_dir).as_posix() for p in data_dir.rglob("*"))
        assert created_files == ["memory", "memory/code_notes.md", "memory/dialog.md", "memory/projects.md"]
    finally:
        manager.terminate_all()


# 测试真正两层父子索引：小块高精命中、多子块命中按 parent_id 去重聚合返回唯一完整父段落、且绝不跨节或拉入过期条目
def test_rag_true_parent_child_aggregation_and_expiry_boundaries(tmp_path: Path) -> None:
    from plugins.rag_retriever import RAGConfig, RAGRetriever

    data_dir = tmp_path / "data"
    mem_dir = data_dir / "memory"
    mem_dir.mkdir(parents=True)
    (mem_dir / "study.md").write_text(
        "# 原理笔记\n\n"
        "## 分布式共识\n"
        "第一段铺垫了网络分区与多数派法定人数背景，并提到核心协议 RAFT_CORE_77。\n"
        "第二段详细展开领导人选举任期号递增机制，确保旧主节点无法继续提交日志。\n"
        "第三段说明日志复制与状态机提交顺序，同时再次核对核心协议 RAFT_CORE_77 的安全性证明。\n\n"
        "- [2026-10-01] 过期干扰项 EXPIRED_LEAK_999。 (until: 2026-10-15)\n\n"
        "## 分布式共识\n"
        "同名新章节正文 DUPLICATE_HEADING_BLOCK，绝不能与上一章节父块混合。\n",
        encoding="utf-8",
    )

    retriever = RAGRetriever(
        RAGConfig(parent_max_chars=500, child_max_chars=90, child_overlap_chars=25)
    )
    memory = MemoryStore(data_dir, create_dirs=True, retriever=retriever)
    try:
        hits = memory.search_hits("RAFT_CORE_77", today=date(2026, 11, 20), limit=5)
        # 尽管父块内第 1 子块与末子块同时命中 RAFT_CORE_77，按 parent_id 聚合后只返回 1 条完整父块
        assert len(hits) == 1
        assert "第一段铺垫" in hits[0].snippet
        assert "第二段详细展开" in hits[0].snippet
        assert "第三段说明" in hits[0].snippet
        assert "EXPIRED_LEAK_999" not in hits[0].snippet
        assert "DUPLICATE_HEADING_BLOCK" not in hits[0].snippet

        # 验证派生 SQLite 中确实存在 parent_chunks 与多条 child_chunks(parent_id)
        db_path = data_dir / "state" / "rag_index.db"
        with sqlite3.connect(db_path) as conn:
            p_cnt = conn.execute("SELECT COUNT(*) FROM parent_chunks").fetchone()[0]
            c_cnt = conn.execute("SELECT COUNT(*) FROM child_chunks").fetchone()[0]
            assert p_cnt == 3 and c_cnt > p_cnt
    finally:
        memory.close()


# 测试派生索引库持久化、跨重启零重读、删库重建、损坏自愈、配置升级重建与符号链接拒绝
def test_rag_retriever_persistent_derived_db_lifecycle_and_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from plugins.rag_retriever import RAGConfig, RAGRetriever

    data_dir = tmp_path / "data"
    mem_dir = data_dir / "memory"
    mem_dir.mkdir(parents=True)
    note = mem_dir / "guide.md"
    note.write_text("# 指南\n\n## 持久化\n派生索引独立存放在 state/rag_index.db。\n", encoding="utf-8")

    # 1. 首次搜索写入 data/state/rag_index.db（parent_chunks + child_chunks）
    r1 = RAGRetriever(RAGConfig(parent_max_chars=800, child_max_chars=300, child_overlap_chars=60))
    mem1 = MemoryStore(data_dir, create_dirs=True, retriever=r1)
    try:
        hits1 = mem1.search_hits("派生索引 state")
        assert hits1 and "rag_index.db" in hits1[0].snippet
        db_path = data_dir / "state" / "rag_index.db"
        assert db_path.is_file()
        assert not (data_dir / "state" / "assistant.db").exists()
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT content_hash, start_line, end_line FROM child_chunks LIMIT 1").fetchone()
            assert row is not None and len(row[0]) == 64 and row[1] <= row[2]
    finally:
        mem1.close()

    # 2. 重启新建实例后，未修改文件零重读直接命中
    r2 = RAGRetriever(RAGConfig(parent_max_chars=800, child_max_chars=300, child_overlap_chars=60))
    mem2 = MemoryStore(data_dir, create_dirs=True, retriever=r2)
    try:
        orig_read_text = Path.read_text

        # 若未变更文件被重读则立即失败，验证跨重启增量指纹生效
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

    # 3. 删除派生库后无损重建
    for suffix in ("", "-wal", "-shm"):
        candidate = data_dir / "state" / f"rag_index.db{suffix}"
        if candidate.exists():
            candidate.unlink()
    r3 = RAGRetriever(RAGConfig(parent_max_chars=800, child_max_chars=300, child_overlap_chars=60))
    mem3 = MemoryStore(data_dir, create_dirs=True, retriever=r3)
    try:
        hits3 = mem3.search_hits("派生索引 state")
        assert hits3 and (data_dir / "state" / "rag_index.db").is_file()
    finally:
        mem3.close()

    # 4. 物理损坏文件自动重建
    (data_dir / "state" / "rag_index.db").write_bytes(b"corrupted-not-a-sqlite-header" * 16)
    r4 = RAGRetriever(RAGConfig(parent_max_chars=800, child_max_chars=300, child_overlap_chars=60))
    mem4 = MemoryStore(data_dir, create_dirs=True, retriever=r4)
    try:
        hits4 = mem4.search_hits("派生索引 state")
        assert hits4 and "rag_index.db" in hits4[0].snippet
    finally:
        mem4.close()

    # 5. 分块参数变更时自动重建表并更新 index_meta
    r5 = RAGRetriever(RAGConfig(parent_max_chars=1200, child_max_chars=360, child_overlap_chars=80))
    mem5 = MemoryStore(data_dir, create_dirs=True, retriever=r5)
    try:
        assert mem5.search_hits("派生索引 state")
        with sqlite3.connect(data_dir / "state" / "rag_index.db") as conn:
            meta = dict(conn.execute("SELECT key, value FROM index_meta").fetchall())
            assert meta.get("parent_max_chars") == "1200"
            assert meta.get("child_overlap_chars") == "80"
    finally:
        mem5.close()

    # 6. 符号链接替换 rag_index.db 时显式拒绝
    outside = tmp_path / "outside.db"
    db_file = data_dir / "state" / "rag_index.db"
    db_file.unlink()
    os.symlink(outside, db_file)
    r6 = RAGRetriever(RAGConfig(parent_max_chars=1200, child_max_chars=360, child_overlap_chars=80))
    mem6 = MemoryStore(data_dir, create_dirs=True, retriever=r6)
    try:
        with pytest.raises(ValueError, match="符号链接"):
            mem6.search_hits("派生索引 state")
    finally:
        mem6.close()


# 测试 SQL 级过期过滤、归档屏蔽、增量更新与高频主题 IDF 托底
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
        on_day = memory.search_hits("冬季压测演练", today=date(2026, 11, 20))
        assert on_day and "冬季压测演练" in on_day[0].snippet
        assert memory.search_hits("冬季压测演练", today=date(2026, 11, 21)) == []
        assert memory.search_hits("神秘火星计划", today=date(2026, 11, 20)) == []

        sched.write_text("# 日程\n- [2026-11-21] 启动星链遥测新里程碑。\n", encoding="utf-8")
        updated_hits = memory.search_hits("星链遥测新里程碑", today=date(2026, 11, 21))
        assert updated_hits and "星链遥测新里程碑" in updated_hits[0].snippet
        assert memory.search_hits("旧版迁移评审会", today=date(2026, 10, 10)) == []

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


# 测试安全与并发回归：index_path 边界、祖先/WAL 符号链接、只读零写盘、锁竞争绝不删库、换 Inode、运行期损坏自愈与多线程并发
def test_rag_security_boundaries_concurrency_and_recovery_regressions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from plugins.rag_retriever import RAGConfig, RAGRetriever, _DerivedRAGIndex

    with pytest.raises(ValueError):
        RAGConfig(index_path="/tmp/evil.db")
    with pytest.raises(ValueError):
        RAGConfig(index_path="../outside.db")
    with pytest.raises(ValueError):
        RAGConfig(index_path=".")
    with pytest.raises(ValueError):
        RAGConfig(index_path="./")

    # 1. read_only=True 即使配置 index_path 也严守零写盘
    ro_data = tmp_path / "ro_data"
    (ro_data / "memory").mkdir(parents=True)
    (ro_data / "memory" / "a.md").write_text("# 只读\n只读模式测试内容 ALPHA_RO。\n", encoding="utf-8")
    ro_mem = MemoryStore(ro_data, create_dirs=False, retriever=RAGRetriever(RAGConfig(index_path="state/custom.db")))
    try:
        assert ro_mem.search_hits("ALPHA_RO")
        assert not (ro_data / "state").exists()
    finally:
        ro_mem.close()

    # 2. 祖先符号链接在 mkdir 前被拦截
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    anc_data = tmp_path / "anc_data"
    (anc_data / "memory").mkdir(parents=True)
    (anc_data / "memory" / "a.md").write_text("# 祖先\n测试祖先符号链接拦截。\n", encoding="utf-8")
    os.symlink(outside_dir, anc_data / "state")
    anc_mem = MemoryStore(
        anc_data, create_dirs=True, retriever=RAGRetriever(RAGConfig(index_path="state/nested/rag_index.db"))
    )
    try:
        with pytest.raises(ValueError, match="符号链接"):
            anc_mem.search_hits("祖先符号链接")
        assert not (outside_dir / "nested").exists()
    finally:
        anc_mem.close()

    # 3. WAL 附属文件符号链接在打开 SQLite 前被拦截
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

    # 4. 写锁超时绝不误删活派生库
    busy_data = tmp_path / "busy_data"
    (busy_data / "memory").mkdir(parents=True)
    (busy_data / "memory" / "a.md").write_text("# 锁\n测试锁竞争保护 BUSY_GUARD。\n", encoding="utf-8")
    busy_mem = MemoryStore(busy_data, create_dirs=True, retriever=RAGRetriever())
    try:
        assert busy_mem.search_hits("BUSY_GUARD")
    finally:
        busy_mem.close()
    busy_db = busy_data / "state" / "rag_index.db"
    before_bytes = busy_db.read_bytes()
    locker = sqlite3.connect(str(busy_db), isolation_level=None)
    try:
        locker.execute("BEGIN EXCLUSIVE")
        orig_conn_disk = _DerivedRAGIndex._connect_disk

        # 缩短 busy_timeout 复现锁超时
        def _short_disk_connect(db_p: Path, *, busy_timeout_ms: int = 5000) -> sqlite3.Connection:
            return orig_conn_disk(db_p, busy_timeout_ms=20)

        monkeypatch.setattr(_DerivedRAGIndex, "_connect_disk", staticmethod(_short_disk_connect))
        (busy_data / "memory" / "a.md").write_text("# 锁\n变更内容 BUSY_GUARD_V2。\n", encoding="utf-8")
        busy_mem2 = MemoryStore(busy_data, create_dirs=True, retriever=RAGRetriever())
        try:
            with pytest.raises(sqlite3.OperationalError):
                busy_mem2.search_hits("BUSY_GUARD_V2")
            assert busy_db.is_file() and busy_db.read_bytes() == before_bytes
        finally:
            busy_mem2.close()
            monkeypatch.undo()
    finally:
        locker.execute("ROLLBACK")
        locker.close()

    # 5. 换 Inode、运行期损坏自愈与删光清空
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
            assert {r[0] for r in conn.execute("SELECT DISTINCT root_key FROM parent_chunks").fetchall()} == {
                "data/memory"
            }

        replacement = rec_data / "state" / "replacement.db"
        tmp_mem = MemoryStore(rec_data, create_dirs=True, retriever=RAGRetriever(RAGConfig(index_path="state/replacement.db")))
        try:
            assert tmp_mem.search_hits("RECOVER_ALPHA")
        finally:
            tmp_mem.close()
        os.replace(replacement, rec_db)
        assert rec_mem.search_hits("RECOVER_ALPHA")

        calls = {"n": 0}
        orig_sync_query = rec_ret._sync_and_query

        # 首次模拟运行期 SQLITE_CORRUPT，验证自动清理重建并重试成功
        def _flaky_corrupt(*args: object, **kwargs: object) -> list[SearchHit]:
            calls["n"] += 1
            if calls["n"] == 1:
                err = sqlite3.DatabaseError("database disk image is malformed")
                err.sqlite_errorcode = sqlite3.SQLITE_CORRUPT  # type: ignore[attr-defined]
                raise err
            return orig_sync_query(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(rec_ret, "_sync_and_query", _flaky_corrupt)
        assert rec_mem.search_hits("RECOVER_ALPHA") and calls["n"] == 2
        monkeypatch.undo()

        doc.unlink()
        assert rec_mem.search_hits("RECOVER_ALPHA") == []
        with sqlite3.connect(rec_db) as conn:
            assert conn.execute("SELECT COUNT(*) FROM parent_chunks").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM child_chunks").fetchone()[0] == 0
    finally:
        rec_mem.close()

    # 6. 多线程并发打开/搜索与 terminate_all 零竞态
    conc_data = tmp_path / "conc_data"
    (conc_data / "memory").mkdir(parents=True)
    (conc_data / "memory" / "c.md").write_text("# 并发\n多线程安全访问 CONCURRENT_SAFE_88。\n", encoding="utf-8")
    repo_root = Path(__file__).resolve().parent.parent
    manager = PluginManager.from_entries(["plugins"], repo_root)
    try:
        # 并发调用插件工厂与搜索验证 WAL 冷启动与弱引用集合锁保护正常
        def _worker(_idx: int) -> int:
            ret = manager.retrievers["rag"]({})
            mem = MemoryStore(conc_data, create_dirs=True, retriever=ret)
            return len(mem.search_hits("CONCURRENT_SAFE_88"))

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(_worker, range(8)))
        assert all(r >= 1 for r in results)

        # 验证 terminate_all 真正关闭所有底层 SQLite 连接
        tracked_ret = manager.retrievers["rag"]({})
        tracked_mem = MemoryStore(conc_data, create_dirs=True, retriever=tracked_ret)
        assert tracked_mem.search_hits("CONCURRENT_SAFE_88")
        opened_conns = [idx.conn for idx in tracked_ret._indices.values()]
        assert opened_conns
    finally:
        manager.terminate_all()
    for conn in opened_conns:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


# 测试旧版 v1 派生库原地平滑升级至 v2 与超长表格续片按表头列名端到端检索命中（展示 (表头) 前缀与精确行号）
def test_rag_v1_inplace_upgrade_and_oversized_table_continuation_retrieval(tmp_path: Path) -> None:
    from plugins.rag_retriever import RAGConfig, RAGRetriever

    data_dir = tmp_path / "data"
    mem_dir = data_dir / "memory"
    state_dir = data_dir / "state"
    mem_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)

    # 1. 构造旧版 PR #102 v1 schema（仅含 chunks / chunks_fts，schema_version="1"）
    db_path = state_dir / "rag_index.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.executemany(
            "INSERT INTO index_meta (key, value) VALUES (?, ?)",
            [("schema_version", "1"), ("chunker_version", "1"), ("chunk_max_chars", "500")],
        )
        conn.execute("CREATE TABLE indexed_files (root_key TEXT, path TEXT)")
        conn.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, text TEXT)")
        conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(body_tokens)")

    # 2. 写入超出 2 * parent_max_chars 的大表格，验证原地升级后按表头列名 + 尾部行键精确命中续片并带 (表头)
    rows = [f"| item_{i:02d} | 普通配置参数详细说明记录 {i:02d} |" for i in range(1, 14)]
    rows.append("| item_tail_99 | 量子加速引擎延迟指标 1.2ms |")
    (mem_dir / "metrics.md").write_text(
        "# 压测报告\n\n| 压测项编码 | 核心指标明细 |\n| --- | --- |\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )

    retriever = RAGRetriever(RAGConfig(parent_max_chars=160, child_max_chars=160, child_overlap_chars=32))
    memory = MemoryStore(data_dir, create_dirs=True, retriever=retriever)
    try:
        hits = memory.search_hits("核心指标明细 item_tail_99 量子加速引擎")
        assert hits
        assert "(表头) | 压测项编码 | 核心指标明细 |" in hits[0].snippet
        assert "item_tail_99" in hits[0].snippet
        # 续片 start_line 为数据行真实行号，而非第 3 行表头行号
        assert hits[0].start_line > 5

        with sqlite3.connect(db_path) as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()}
            assert "chunks" not in tables and "chunks_fts" not in tables
            assert {"parent_chunks", "child_chunks", "child_chunks_fts", "chunk_contexts"}.issubset(tables)
    finally:
        memory.close()


# 测试 AssistantApp 与 eval CLI 集成 rag 与 linear
def test_app_and_eval_cli_integrate_rag_plugin(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    app = AssistantApp(
        data_root=tmp_path / "app_data",
        config_path=repo_root / "config" / "tools.json",
        project_root=repo_root,
        llm=NoopModelClient(),
    )
    try:
        assert app.status()["retriever"] == "rag"
    finally:
        app.close()

    out_linear = tmp_path / "linear.json"
    out_rag = tmp_path / "rag.json"
    assert eval_main(["--mode", "search", "--retriever", "linear", "--output", str(out_linear)]) == 0
    assert eval_main(["--mode", "search", "--retriever", "rag", "--output", str(out_rag)]) == 0
    assert json.loads(out_linear.read_text(encoding="utf-8"))["retriever"] == "linear"
    assert json.loads(out_rag.read_text(encoding="utf-8"))["retriever"] == "rag"
