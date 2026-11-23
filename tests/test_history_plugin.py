"""只读增量 FTS5 历史对话检索插件（plugins/tools/history_tools.py）单元测试（TDD）。"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from personal_assistant.agent.plugins import Plugin, PluginManager
from personal_assistant.agent.tools import ToolDocument, ToolMetadata
from personal_assistant.ai import LLMClient
from personal_assistant.assistant.app import AssistantApp
from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.assistant.common import REPO_ROOT
from personal_assistant.assistant.db import StateStore
from personal_assistant.assistant.prompts import SYSTEM_PROMPT


# 测试用脚本化模拟模型
class ScriptedHistoryLLM(LLMClient):
    # 保存预设回复序列并记录每次收到的消息上下文
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = list(responses)
        self.requests: list[list[dict[str, object]]] = []

    # 记录请求并弹出下一条预设回复
    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        self.requests.append([dict(m) for m in messages])
        return self._responses.pop(0)


# 测试插件发现、ToolsConfig 白名单门控、单参数契约及对 chat_history.sqlite3 的只读零侵入边界
def test_history_plugin_discovery_allowlist_and_readonly_schema(tmp_path: Path) -> None:
    manager = PluginManager.from_entries(["plugins"], REPO_ROOT)
    assert len(manager.plugins) == 3
    assert all(isinstance(p, Plugin) for p in manager.plugins)
    assert {p.plugin_id for p in manager.plugins} == {"tools", "fts5_bm25", "models_dsl"}
    assert "search_chat_history" in {t.name for t in manager.default_tools}

    cfg = ToolsConfig.load(REPO_ROOT / "config" / "tools.json", project_root=REPO_ROOT)
    db_path = tmp_path / "data" / "state" / "chat_history.sqlite3"
    state = StateStore(db_path)
    try:
        state.append_message("s1", "user", "请推荐一本手工具木工入门书籍。")
        state.append_message("s1", "assistant", "推荐 Christopher Schwarz 的《The Anarchist's Tool Chest》。")
    finally:
        state.close()

    registry = cfg.build_registry(
        data_root=tmp_path / "data",
        tool_context={"history_db_path": db_path},
    )
    assert "search_chat_history" in registry.names
    assert registry.metadata("search_chat_history") == ToolMetadata(risk="read", always_on=False)
    assert registry.document("search_chat_history") == ToolDocument(
        source_type="plugin",
        source_name="history_tools",
    )
    tool_obj = next(t for t in registry if t.name == "search_chat_history")
    assert set(tool_obj.parameters) == {"query"}

    # 调用工具只读检索历史原文，且绝不改动 chat_history.sqlite3 磁盘表结构
    with sqlite3.connect(db_path) as before_conn:
        tables_before = {
            row[0]
            for row in before_conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'trigger')"
            ).fetchall()
        }
    obs = registry.execute("search_chat_history", '{"query": "Christopher Schwarz"}')
    assert "Christopher Schwarz" in obs
    with sqlite3.connect(db_path) as check_conn:
        tables_after = {
            row[0]
            for row in check_conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'trigger')"
            ).fetchall()
        }
    assert tables_after == tables_before
    assert {"messages", "session_compactions"}.issubset(tables_after)
    assert not any("fts" in name.lower() for name in tables_after)

    # 从 enabled_tools 移除后必须被白名单严格拦截
    disabled_cfg_path = tmp_path / "tools_disabled.json"
    disabled_cfg_path.write_text(
        json.dumps({"enabled_tools": ["list_files"], "plugins": ["plugins"], "commands": []}),
        encoding="utf-8",
    )
    disabled_reg = ToolsConfig.load(disabled_cfg_path, project_root=REPO_ROOT).build_registry(
        data_root=tmp_path / "data",
        tool_context={"history_db_path": db_path},
    )
    assert "不在白名单中" in disabled_reg.execute("search_chat_history", '{"query": "Schwarz"}')


# 测试中英双语 FTS5 召回、跨轮 prev_tokens、相邻上下文窗口、会话过滤、去重与 before_msg_id 快照隔离
def test_history_fts5_bilingual_recall_context_window_and_snapshot_isolation(tmp_path: Path) -> None:
    from plugins.tools.history_tools import execute_search_chat_history

    db_path = tmp_path / "data" / "state" / "chat_history.sqlite3"
    state = StateStore(db_path)
    try:
        state.append_message("s1", "user", "Can you recommend three practical woodworking books for hand tools?")
        state.append_message(
            "s1",
            "assistant",
            "1. The Anarchist's Tool Chest by Christopher Schwarz.\n"
            "2. The Essential Woodworker by Robert Wearing.\n"
            "3. Understanding Wood by Bruce Hoadley.",
        )
        state.append_message(
            "s1",
            "user",
            "How many participants were in that sleep deprivation experiment you mentioned?",
        )
        state.append_message(
            "s1",
            "assistant",
            "The sleep experiment recruited 38 healthy adult participants across four weeks.",
        )
        state.append_message("s1", "user", "东京机房 Atlas 服务监听端口是多少？")
        atlas_reply_id = state.append_message(
            "s1",
            "assistant",
            "东京机房 Atlas 数据同步服务监听端口是 9092，采用 SQLite WAL 模式。",
        )
        state.append_message("s2", "user", "Tokio 工作线程数配置怎么调整？")
        state.append_message("s2", "assistant", "可通过 Builder::worker_threads 设置 Tokio 工作线程数。")

        # 1. 英文词形还原 + 命中用户提问时通过 context_window=1 带出紧邻助手推荐列表
        res_en = execute_search_chat_history(db_path, "woodworking hand tools")
        assert "Christopher Schwarz" in res_en
        assert "Robert Wearing" in res_en

        # 2. 跨轮 prev_tokens 联合查询（用户问句词 hand tools + 助手回答词 Christopher Schwarz）
        res_cross = execute_search_chat_history(db_path, "hand tools Christopher Schwarz")
        assert "Christopher Schwarz" in res_cross

        # 3. 命中助手数值答案（sleep study 38 participants）
        res_num = execute_search_chat_history(db_path, "sleep experiment participants 38")
        assert "38 healthy adult participants" in res_num

        # 4. 中文查询命中与相邻上下文窗口展示
        res_zh = execute_search_chat_history(db_path, "东京机房 Atlas 端口 9092")
        assert "9092" in res_zh
        assert "东京机房 Atlas 服务监听端口是多少？" in res_zh

        # 5. 同会话相邻命中去重：搜索 Atlas 只产出 1 个命中块而非重复块
        assert res_zh.count("[命中 ") == 1

        # 6. chat_id 过滤隔离不同会话
        res_s2 = execute_search_chat_history(db_path, "Tokio", chat_id="s2")
        assert "worker_threads" in res_s2
        assert execute_search_chat_history(db_path, "Atlas", chat_id="s2") == "未找到匹配的历史对话记录。"

        # 7. before_msg_id 严格隔离：>= before_msg_id 的消息既不被检索命中，也不通过 context_window 泄漏
        cur_id = state.append_message(
            "s1",
            "user",
            "请确认东京机房 Atlas 端口是不是 9092 还是 Christopher Schwarz",
        )
        isolated_cur = execute_search_chat_history(db_path, "Christopher Schwarz", before_msg_id=cur_id)
        assert "请确认东京机房 Atlas" not in isolated_cur
        assert "Christopher Schwarz" in isolated_cur

        isolated_reply = execute_search_chat_history(
            db_path, "东京机房 Atlas", before_msg_id=atlas_reply_id
        )
        assert "9092" not in isolated_reply
        assert "东京机房 Atlas 服务监听端口是多少？" in isolated_reply
    finally:
        state.close()


# 测试缺失库、增量追加同步、重建库自动重置索引及硬负例空返回防幻觉
def test_history_fts5_incremental_sync_db_reset_missing_and_abstention(tmp_path: Path) -> None:
    from plugins.tools.history_tools import execute_search_chat_history

    missing_db = tmp_path / "missing" / "chat_history.sqlite3"
    assert execute_search_chat_history(missing_db, "anything") == "未找到匹配的历史对话记录。"
    assert not missing_db.exists()

    db_path = tmp_path / "data" / "state" / "chat_history.sqlite3"
    state = StateStore(db_path)
    try:
        state.append_message("s1", "user", "Atlas 服务部署在东京机房。")
        state.append_message("s1", "assistant", "收到，已记下 Atlas 机房位置。")
        for idx in range(8):
            state.append_message("s1", "user", f"日常记录第 {idx} 条测试消息。")
            state.append_message("s1", "assistant", f"收到第 {idx} 条测试消息。")

        # 1. 初次检索命中后，再追加新消息应被增量同步检索到
        assert "东京机房" in execute_search_chat_history(db_path, "Atlas 东京机房")
        state.append_message("s1", "assistant", "Nova 遥测采集器监听端口为 8443。")
        assert "8443" in execute_search_chat_history(db_path, "Nova 遥测采集器 8443")

        # 2. 高相似干扰负例（仅命中单个高频词 Atlas，其余关键词全无）必须干净返回空提示
        assert (
            execute_search_chat_history(db_path, "Atlas Redis 集群密码和哨兵主从切换地址")
            == "未找到匹配的历史对话记录。"
        )
    finally:
        state.close()

    # 3. 删库重建为更少条目后，内存索引应自动清零并仅反映新库内容
    db_path.unlink()
    fresh_state = StateStore(db_path)
    try:
        fresh_state.append_message("s1", "assistant", "重建后的全新独立会话只讨论 Vulkan 渲染管线。")
        assert execute_search_chat_history(db_path, "Nova 遥测采集器") == "未找到匹配的历史对话记录。"
        assert "Vulkan" in execute_search_chat_history(db_path, "Vulkan 渲染管线")
    finally:
        fresh_state.close()


# 测试 AssistantApp 端到端 ReAct 调用 search_chat_history 找回已压缩历史推荐并隔离当前提问
def test_assistant_app_react_uses_search_chat_history_with_turn_isolation(tmp_path: Path) -> None:
    assert len(SYSTEM_PROMPT) <= 300
    assert "search_chat_history" in SYSTEM_PROMPT
    assert "不要编造未观测到的结果" in SYSTEM_PROMPT
    assert "建议/推荐/规划" in SYSTEM_PROMPT

    llm = ScriptedHistoryLLM(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_hist",
                        "type": "function",
                        "function": {
                            "name": "search_chat_history",
                            "arguments": '{"query": "木工书 Christopher Schwarz"}',
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "我之前为你推荐过《The Anarchist's Tool Chest》（作者 Christopher Schwarz）。",
            },
        ]
    )
    app = AssistantApp(
        data_root=tmp_path / "data",
        config_path=REPO_ROOT / "config" / "tools.json",
        project_root=REPO_ROOT,
        llm=llm,
    )
    try:
        app.state.append_message("s-app", "user", "推荐几本实用木工书。")
        until_id = app.state.append_message(
            "s-app",
            "assistant",
            "木工入门推荐《The Anarchist's Tool Chest》（作者 Christopher Schwarz），适合打磨手工工具基础。",
        )
        app.state.save_compaction("s-app", "早期曾讨论过木工入门书籍推荐。", until_id)

        current_question = "你之前推荐过哪本木工书来着？作者 Christopher Schwarz 那本"
        result = app.chat(current_question, chat_id="s-app")

        assert "Christopher Schwarz" in str(result["reply"])
        calls = result["tool_calls"]
        assert isinstance(calls, list) and len(calls) == 1
        assert calls[0]["tool"] == "search_chat_history"
        obs = str(calls[0]["observation"])
        assert "The Anarchist's Tool Chest" in obs
        assert current_question not in obs
    finally:
        app.close()
