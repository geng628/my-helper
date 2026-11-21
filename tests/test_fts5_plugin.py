"""SQLite FTS5 + BM25 检索插件单元测试（TDD）：验证插件加载、中英分词、负例过滤、过期与增量同步。"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from personal_assistant.assistant.app import AssistantApp
from personal_assistant.assistant.commands import ToolsConfig
from evals import main as eval_main
from personal_assistant.ai import NoopLLMClient
from personal_assistant.assistant.memory import MemoryStore
from personal_assistant.agent.plugins import load_plugin_retrievers


# 测试插件加载器能从 plugins/ 自动发现 fts5_bm25 检索插件，并受 tools.json 的 retrieval 段开关控制
def test_fts5_plugin_discovery_and_tools_config_toggle(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    retrievers = load_plugin_retrievers(["plugins"], repo_root)
    assert "fts5_bm25" in retrievers

    cfg_path = tmp_path / "tools.json"
    cfg_path.write_text(
        json.dumps(
            {
                "enabled_tools": ["search_markdown"],
                "plugins": ["plugins"],
                "retrieval": {"plugin": "fts5_bm25", "min_term_coverage": 0.25},
                "files": {"read_roots": ["data/memory"], "write_root": "data/workspace"},
                "commands": [],
            }
        ),
        encoding="utf-8",
    )
    cfg = ToolsConfig.load(cfg_path, project_root=repo_root)
    active = cfg.build_retriever()
    assert active is not None
    assert getattr(active, "name", "") == "fts5_bm25"

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
    assert ToolsConfig.load(cfg_linear, project_root=repo_root).build_retriever() is None


# 测试 FTS5 BM25 插件的英文停用词/词干召回、中文双字索引、高相似干扰负例拦截与多文件多样性
def test_fts5_retriever_bm25_ranking_hard_negatives_and_diversity(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    factory = load_plugin_retrievers(["plugins"], repo_root)["fts5_bm25"]
    retriever = factory({"min_term_coverage": 0.25, "max_hits_per_file": 2})

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
    (mem_dir / "rules.md").write_text(
        "# 规则\n"
        "外部命令必须使用固定 argv 数组并设置 shell=False，防止命令注入。\n",
        encoding="utf-8",
    )

    memory = MemoryStore(data_dir, create_dirs=False, retriever=retriever)

    # 1. 英文疑问句含大量停用词 + 词形变化（research vs researched, agency vs agencies）应精准命中 D1:1
    hits_en = memory.search_hits("What did Caroline research about adoption agency?", limit=3)
    assert hits_en
    assert "[D1:1]" in hits_en[0].snippet

    # 2. 中文具体查询命中正确正文行（而非仅停在章节标题）
    hits_zh = memory.search_hits("Nova 遥测采集器上报端口", limit=3)
    assert hits_zh
    assert "8443" in hits_zh[0].snippet

    # 3. 高相似干扰负例（仅命中一个实体词 Atlas，其余 6 个高信息词全无）应返回空列表
    decoy_hits = memory.search_hits("Atlas Redis 集群密码和哨兵主从切换地址", limit=5)
    assert decoy_hits == []

    # 4. 只读模式（create_dirs=False）下绝不往 data_dir 写入任何 SQLite 索引文件
    created_files = sorted(p.relative_to(data_dir).as_posix() for p in data_dir.rglob("*"))
    assert created_files == ["memory", "memory/dialog.md", "memory/projects.md", "memory/rules.md"]


# 测试 FTS5 检索插件的 SQL 级过期过滤（today 边界）、归档屏蔽以及文件修改后的增量同步
def test_fts5_retriever_temporal_expiry_archive_and_incremental_sync(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    factory = load_plugin_retrievers(["plugins"], repo_root)["fts5_bm25"]
    retriever = factory({})

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

    memory = MemoryStore(data_dir, create_dirs=False, retriever=retriever)

    # 截止当日（until == today）应命中
    on_day = memory.search_hits("冬季压测演练", today=date(2026, 11, 20))
    assert on_day and "冬季压测演练" in on_day[0].snippet

    # 截止次日（until < today）应被 SQL 直接过滤，无需重建索引
    next_day = memory.search_hits("冬季压测演练", today=date(2026, 11, 21))
    assert next_day == []

    # 归档目录默认不参与检索
    assert memory.search_hits("神秘火星计划", today=date(2026, 11, 20)) == []

    # 动态改写磁盘文件后，增量同步应立刻反映最新内容并移除旧内容
    sched.write_text(
        "# 日程\n"
        "- [2026-11-21] 启动星链遥测新里程碑。\n",
        encoding="utf-8",
    )
    # 触发明显 mtime/size 变化后立刻可搜新词且旧词消失
    updated_hits = memory.search_hits("星链遥测新里程碑", today=date(2026, 11, 21))
    assert updated_hits and "星链遥测新里程碑" in updated_hits[0].snippet
    assert memory.search_hits("旧版迁移评审会", today=date(2026, 10, 10)) == []


# 测试 AssistantApp 与 eval CLI 支持按配置或 --retriever 参数无缝切换 fts5_bm25 与 linear
def test_app_and_eval_cli_integrate_fts5_plugin(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    app = AssistantApp(
        data_root=tmp_path / "app_data",
        config_path=repo_root / "config" / "tools.json",
        project_root=repo_root,
        llm=NoopLLMClient(),
    )
    try:
        status = app.status()
        assert status["retriever"] == "fts5_bm25"
    finally:
        app.close()

    out_linear = tmp_path / "linear.json"
    out_fts5 = tmp_path / "fts5.json"
    assert eval_main(["--mode", "search", "--retriever", "linear", "--output", str(out_linear)]) == 0
    assert eval_main(["--mode", "search", "--retriever", "fts5_bm25", "--output", str(out_fts5)]) == 0
    rep_linear = json.loads(out_linear.read_text(encoding="utf-8"))
    rep_fts5 = json.loads(out_fts5.read_text(encoding="utf-8"))
    assert rep_linear["retriever"] == "linear"
    assert rep_fts5["retriever"] == "fts5_bm25"
