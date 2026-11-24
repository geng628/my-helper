"""只读增量 FTS5 历史对话检索插件（plugins/tools/history_tools.py）单元测试（TDD）。"""
from __future__ import annotations

import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

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
    assert len(manager.plugins) == 4
    assert all(isinstance(p, Plugin) for p in manager.plugins)
    assert {p.plugin_id for p in manager.plugins} == {"tools", "fts5_bm25", "models_dsl", "telemetry"}
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
    assert "原对话语言" in tool_obj.parameters["query"]

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
        hist_tool = next(t for t in app.build_tools() if t.name == "search_chat_history")
        assert "data/memory/" in hist_tool.description
    finally:
        app.close()


# 测试有界 LRU 淘汰下的租约钉住机制与多线程并发检索（淘汰后不崩且不静默漏查，释放后延迟关闭）
def test_history_lru_eviction_lease_pinning_and_concurrency(tmp_path: Path) -> None:
    from plugins.tools.history_tools import (
        MAX_CACHED_DBS,
        _checkout_history_index,
        _release_history_index,
        execute_search_chat_history,
    )

    total_dbs = MAX_CACHED_DBS + 4
    db_paths: list[Path] = []
    for i in range(total_dbs):
        db_p = tmp_path / f"db_{i:02d}" / "chat_history.sqlite3"
        st = StateStore(db_p)
        try:
            st.append_message("s1", "assistant", f"唯一项目代号 NebulaToken{i:02d} 部署完毕。")
        finally:
            st.close()
        db_paths.append(db_p)

    # 1. 确定性租约钉住测试：借出 db_0 后触发多次淘汰，pin 期间仍可正常检索，释放后才关闭连接
    pinned = _checkout_history_index(db_paths[0])
    try:
        for i in range(1, total_dbs):
            out = execute_search_chat_history(db_paths[i], f"NebulaToken{i:02d}")
            assert f"NebulaToken{i:02d}" in out
        assert pinned._retired is True
        hits = pinned.search(db_paths[0], "NebulaToken00")
        assert hits and "NebulaToken00" in hits[0].context_messages[0][2]
    finally:
        _release_history_index(pinned)

    with pytest.raises(sqlite3.ProgrammingError):
        pinned.conn.execute("SELECT 1")

    # 2. 多线程并发跨 > MAX_CACHED_DBS 库检索无竞态异常且 100% 准确命中
    def _worker(idx: int) -> str:
        target_idx = idx % total_dbs
        return execute_search_chat_history(db_paths[target_idx], f"NebulaToken{target_idx:02d}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_worker, range(total_dbs * 3)))
    for idx, res in enumerate(results):
        assert f"NebulaToken{idx % total_dbs:02d}" in res


# 测试 AssistantApp 多轮 chat() 复用同一缓存插件模块并保持增量游标跨轮递增（不重复全表重扫）
def test_assistant_app_multi_turn_reuses_incremental_history_index(tmp_path: Path) -> None:
    llm = ScriptedHistoryLLM(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "t1_call",
                        "type": "function",
                        "function": {
                            "name": "search_chat_history",
                            "arguments": '{"query": "HeliosFirstMilestone"}',
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "第1轮已查到 HeliosFirstMilestone。"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "t2_call",
                        "type": "function",
                        "function": {
                            "name": "search_chat_history",
                            "arguments": '{"query": "HeliosSecondMilestone"}',
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "第2轮已查到 HeliosSecondMilestone。"},
        ]
    )
    app = AssistantApp(
        data_root=tmp_path / "data",
        config_path=REPO_ROOT / "config" / "tools.json",
        project_root=REPO_ROOT,
        llm=llm,
    )
    try:
        app.state.append_message("s-multi", "assistant", "早期记录 HeliosFirstMilestone 完成。")
        r1 = app.chat("查一下 HeliosFirstMilestone", chat_id="s-multi")
        assert "HeliosFirstMilestone" in str(r1["tool_calls"][0]["observation"])

        hist_mods = [m for name, m in sys.modules.items() if name.startswith("my_helper_plugin_history_tools_")]
        assert len(hist_mods) == 1
        mod_turn1 = hist_mods[0]
        cache_dict_id_turn1 = id(mod_turn1._HISTORY_INDICES)
        idx_turn1 = mod_turn1._HISTORY_INDICES[app.state.db_path.resolve()]
        last_id_turn1 = idx_turn1.last_indexed_id
        assert last_id_turn1 > 0

        # 轮间追加新记录后执行第2轮对话，模块与索引实例均应复用且游标向前增量推进
        new_msg_id = app.state.append_message("s-multi", "assistant", "新增记录 HeliosSecondMilestone 上线。")
        r2 = app.chat("查一下 HeliosSecondMilestone", chat_id="s-multi")
        obs_turn2 = str(r2["tool_calls"][0]["observation"])
        assert "新增记录 HeliosSecondMilestone 上线。" in obs_turn2
        assert "查一下 HeliosSecondMilestone" not in obs_turn2

        mod_turn2 = next(m for name, m in sys.modules.items() if name.startswith("my_helper_plugin_history_tools_"))
        assert mod_turn2 is mod_turn1
        assert id(mod_turn2._HISTORY_INDICES) == cache_dict_id_turn1
        idx_turn2 = mod_turn2._HISTORY_INDICES[app.state.db_path.resolve()]
        assert idx_turn2 is idx_turn1
        assert idx_turn2.last_indexed_id >= new_msg_id
        assert idx_turn2.last_indexed_id > last_id_turn1
    finally:
        app.close()


# 测试长消息裁剪（配对助手长推荐保留至 1800 字符 vs 普通邻居 600 字符）、总字节上限与 retrieval_config 透传
def test_history_truncation_and_retrieval_config_forwarding(tmp_path: Path) -> None:
    from plugins.tools.history_tools import (
        MAX_MATCHED_MSG_CHARS,
        MAX_NEIGHBOR_MSG_CHARS,
        MAX_OUTPUT_BYTES,
        execute_search_chat_history,
    )

    db_path = tmp_path / "data" / "state" / "chat_history.sqlite3"
    state = StateStore(db_path)
    try:
        state.append_message("s-trunc", "assistant", "PRE_NEIGHBOR " + ("A" * 900) + " PRE_TAIL_SHOULD_CLIP")
        state.append_message("s-trunc", "user", "请推荐学习书单 UniqueTriggerQuery")
        state.append_message(
            "s-trunc",
            "assistant",
            "PAIRED_REPLY " + ("B" * 1500) + " PAIRED_TAIL_KEPT " + ("C" * 500) + " OVER_1800_CLIPPED",
        )
    finally:
        state.close()

    formatted = execute_search_chat_history(db_path, "UniqueTriggerQuery")
    assert "PAIRED_TAIL_KEPT" in formatted
    assert "OVER_1800_CLIPPED" not in formatted
    assert "PRE_NEIGHBOR" in formatted
    assert "PRE_TAIL_SHOULD_CLIP" not in formatted
    assert MAX_MATCHED_MSG_CHARS == 1800 and MAX_NEIGHBOR_MSG_CHARS == 600

    # 验证多条长命中叠加时总输出字节严格受 MAX_OUTPUT_BYTES 限制
    big_db = tmp_path / "big" / "chat_history.sqlite3"
    big_state = StateStore(big_db)
    try:
        for i in range(6):
            big_state.append_message(
                f"c{i}",
                "assistant",
                f"BatchMarkerHit {i} " + ("内容块" * 800),
            )
    finally:
        big_state.close()
    big_out = execute_search_chat_history(big_db, "BatchMarkerHit", limit=6)
    notice = "\n…（输出已截断）"
    assert big_out.endswith(notice)
    assert len(big_out.encode("utf-8")) <= MAX_OUTPUT_BYTES + len(notice.encode("utf-8"))

    # 验证 config/tools.json 的 retrieval 段透传到 search_chat_history 的 retrieval_config 门控
    strict_cfg_path = tmp_path / "strict_tools.json"
    strict_cfg_path.write_text(
        json.dumps(
            {
                "enabled_tools": ["search_chat_history"],
                "plugins": ["plugins"],
                "retrieval": {"plugin": "fts5_bm25", "min_term_coverage": 0.99},
                "commands": [],
            }
        ),
        encoding="utf-8",
    )
    strict_reg = ToolsConfig.load(strict_cfg_path, project_root=REPO_ROOT).build_registry(
        data_root=tmp_path / "data",
        tool_context={"history_db_path": db_path},
    )
    # 2 个查询词只命中 1 个词（覆盖率 0.5 < 0.99），严格透传配置下应拦截并返回空
    assert (
        strict_reg.execute("search_chat_history", '{"query": "UniqueTriggerQuery NonExistentWord"}')
        == "未找到匹配的历史对话记录。"
    )


# 测试单主题会话下 FTS5 BM25 IDF <= 0 时，高覆盖命中不被误杀且无关干扰词仍严格拦截
def test_history_single_topic_session_idf_zero_still_recalls(tmp_path: Path) -> None:
    from plugins.tools.history_tools import execute_search_chat_history

    db_path = tmp_path / "data" / "state" / "chat_history.sqlite3"
    state = StateStore(db_path)
    try:
        # 构造 12 条均包含 front-end back-end development 的单主题对话，使该词组 IDF <= 0
        for idx in range(6):
            state.append_message(
                "single-topic",
                "user",
                f"Question {idx}: explain front-end and back-end development responsibilities.",
            )
            extra = (
                " I recommended Ruby, Python, and PHP for back-end web services."
                if idx == 3
                else ""
            )
            state.append_message(
                "single-topic",
                "assistant",
                f"Answer {idx}: front-end and back-end development handle UI and server logic.{extra}",
            )
    finally:
        state.close()

    recalled = execute_search_chat_history(db_path, "front-end back-end development")
    assert recalled != "未找到匹配的历史对话记录。"
    assert "front-end and back-end development" in recalled

    # 高相似低覆盖负例仍必须严格返回空
    assert (
        execute_search_chat_history(db_path, "development Kubernetes etcd certificate rotation secret")
        == "未找到匹配的历史对话记录。"
    )


# 测试单主题 IDF<=0 时 2/3 查询召回、排序分数不坍缩为 0.0 与负例门控
def test_history_single_topic_two_thirds_recall_score_nonzero_and_negative_gates(
    tmp_path: Path,
) -> None:
    from plugins.tools.history_tools import _HistoryFTSIndex, execute_search_chat_history

    db_path = tmp_path / "data" / "state" / "chat_history.sqlite3"
    state = StateStore(db_path)
    try:
        # 12 条消息均含 back-end（使 back / end 在 FTS5 内 IDF<=0 钳位为 1e-6）
        for idx in range(1, 7):
            if idx == 5:
                state.append_message(
                    "be-topic",
                    "user",
                    "Do you recommend online resources for learning back-end development?",
                )
                state.append_message(
                    "be-topic",
                    "assistant",
                    "I recommend NodeSchool, Udacity, and Coursera for back-end learning.",
                )
                continue
            state.append_message(
                "be-topic",
                "user",
                f"Turn {idx}: tell me about back-end architecture and services.",
            )
            extra = (
                " Tip 2: learn a back-end programming language such as Ruby, Python, or PHP."
                if idx == 4
                else ""
            )
            state.append_message(
                "be-topic",
                "assistant",
                f"Answer {idx}: back-end services manage data and APIs.{extra}",
            )
    finally:
        state.close()

    # msg #8（Ruby, Python, PHP）仅命中 back + end（2/3 = 0.67 覆盖率且 IDF<=0），与 msg #10 均须召回
    both_recalled = execute_search_chat_history(db_path, "recommend back-end", limit=5)
    assert "NodeSchool" in both_recalled
    assert "Ruby, Python, or PHP" in both_recalled

    # 验证直接调用 _HistoryFTSIndex.search() 在 IDF<=0 时分数 > 0.0 且高覆盖命中严格排在低覆盖前
    index = _HistoryFTSIndex()
    try:
        # services(10/12) 与 back/end(12/12) 全部在 FTS5 内触发 IDF<=0，验证 3 词全覆盖(msg #12)严格高于 2/3 覆盖(msg #10)
        ranked = index.search(db_path, "services back-end", limit=12, context_window=0)
        assert ranked
        assert all(hit.score > 0.0 for hit in ranked)
        scores_by_id = {hit.msg_id: hit.score for hit in ranked}
        assert scores_by_id[12] > scores_by_id[10]
    finally:
        index.close()

    # 验证单主题 IDF<=0 场景下低覆盖负例（1/2、2/4、1/6）仍严格返回空
    assert execute_search_chat_history(db_path, "services nonexistent") == "未找到匹配的历史对话记录。"
    assert (
        execute_search_chat_history(db_path, "back-end nonexistent alpha beta")
        == "未找到匹配的历史对话记录。"
    )
    assert (
        execute_search_chat_history(db_path, "services Kubernetes etcd vault rotation secret")
        == "未找到匹配的历史对话记录。"
    )


# 测试由 prev_tokens 跨轮命中触发的前一条助手消息保留 1800 字符完整证据，不被 600 字符腰斩
def test_history_prev_tokens_evidence_not_clipped_at_600(tmp_path: Path) -> None:
    from plugins.tools.history_tools import _HistoryFTSIndex, execute_search_chat_history

    db_path = tmp_path / "data" / "state" / "chat_history.sqlite3"
    state = StateStore(db_path)
    try:
        filler = "Recommendation item details and craft notes for India textiles. " * 12
        assert len(filler) > 650
        state.append_message(
            "s-prev",
            "assistant",
            f"1. Fabriclore.\n{filler}\n4. NostalgiaStoreMarker is an online store based in India.",
        )
        state.append_message(
            "s-prev",
            "user",
            "Thanks! Do you have online communities forum discussion suggestions?",
        )
        state.append_message(
            "s-prev",
            "assistant",
            "Sure, you can explore craft groups.",
        )
    finally:
        state.close()

    # user #2 本体命中 4 个词且 prev_tokens 命中 NostalgiaStoreMarker（cov=1.0）；assistant #1 本体仅 1/5=0.20 < 0.22
    query = "NostalgiaStoreMarker online communities forum discussion"
    index = _HistoryFTSIndex()
    try:
        hits = index.search(db_path, query, limit=5)
        assert len(hits) == 1
        assert hits[0].msg_id == 2
        assert hits[0].prev_matched is True
    finally:
        index.close()

    rendered = execute_search_chat_history(db_path, query, limit=5)
    assert "NostalgiaStoreMarker is an online store based in India." in rendered
