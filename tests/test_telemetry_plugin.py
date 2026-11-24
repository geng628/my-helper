"""TDD 单测：零微内核改动的运行期观测插件 plugins/telemetry.py。"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from personal_assistant.agent import ObserverProtocol, PluginManager, TaskEvent, ToolStep, TurnEnd
from personal_assistant.ai import LLMClient, StreamDone, TextDelta, TokenUsage
from personal_assistant.assistant.app import AssistantApp
from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.assistant.common import REPO_ROOT
from personal_assistant.assistant.consolidator import ConsolidationPolicy
from plugins.telemetry import TelemetryConfig, TelemetryObserver
from tests.test_streaming import ScriptedStreamLLM


# 测试观测插件自动发现、实例工厂隔离以及 telemetry.enabled 开关控制
def test_telemetry_plugin_discovery_and_config_toggle(tmp_path: Path) -> None:
    manager = PluginManager.from_entries(["plugins"], REPO_ROOT)
    assert "telemetry" in manager.observers

    obs_a = manager.observers["telemetry"]({})
    obs_b = manager.observers["telemetry"]({})
    assert obs_a is not obs_b
    assert isinstance(obs_a, ObserverProtocol)
    assert obs_a.name == "telemetry"

    base_files = {"read_roots": ["data/memory"], "write_root": "data/workspace"}
    cfg_on = tmp_path / "tools_on.json"
    cfg_on.write_text(
        json.dumps(
            {
                "enabled_tools": ["get_current_time"],
                "plugins": ["plugins"],
                "telemetry": {"enabled": True, "window_size": 5},
                "files": base_files,
                "commands": [],
            }
        ),
        encoding="utf-8",
    )
    loaded_on = ToolsConfig.load(cfg_on, project_root=REPO_ROOT)
    observers_on = loaded_on.build_observers()
    assert len(observers_on) == 1
    assert isinstance(observers_on[0], ObserverProtocol)
    assert observers_on[0].name == "telemetry"
    assert observers_on[0].config.window_size == 5

    cfg_off = tmp_path / "tools_off.json"
    cfg_off.write_text(
        json.dumps(
            {
                "enabled_tools": ["get_current_time"],
                "plugins": ["plugins"],
                "telemetry": {"enabled": False},
                "files": base_files,
                "commands": [],
            }
        ),
        encoding="utf-8",
    )
    app_off = AssistantApp(
        data_root=tmp_path / "data_off",
        config_path=cfg_off,
        project_root=REPO_ROOT,
        llm=ScriptedStreamLLM([{"role": "assistant", "content": "已关闭遥测"}]),
    )
    try:
        assert app_off.observers == ()
        reply = app_off.chat("测试关闭开关", chat_id="off")
        assert "telemetry" not in reply
    finally:
        app_off.close()


# 测试分阶段耗时拆解（ttft/llm/tool/overhead）、会话与前后台隔离及有界窗口游标
def test_telemetry_stage_latency_session_isolation_and_bounded_window() -> None:
    obs = TelemetryObserver(TelemetryConfig(window_size=3, max_sessions=4))

    # 1. 多步 ReAct 事件：验证 ttft 取首轮、llm_ms/tool_ms 累加及 overhead 差值
    obs.on_event(
        "main",
        StreamDone(
            message={"role": "assistant", "content": None},
            usage=TokenUsage(prompt_tokens=100, cached_tokens=60, completion_tokens=20),
            ttft_ms=120.0,
            latency_ms=300.0,
        ),
    )
    obs.on_event(
        "main",
        ToolStep(tool="list_files", arguments="{}", observation="ok", duration_ms=45.5),
    )
    obs.on_event("main", TextDelta(text="好"))
    obs.on_event(
        "main",
        StreamDone(
            message={"role": "assistant", "content": "好"},
            usage=TokenUsage(prompt_tokens=150, cached_tokens=120, completion_tokens=10),
            ttft_ms=80.0,
            latency_ms=200.0,
        ),
    )
    obs.on_event("main", TurnEnd(reply="好", total_ms=560.0))

    snap_main = obs.snapshot(session_key="main")
    last_turn = snap_main["last_turn"]
    assert last_turn["ttft_ms"] == 120.0
    assert last_turn["llm_ms"] == 500.0
    assert last_turn["tool_ms"] == 45.5
    assert last_turn["overhead_ms"] == 14.5
    assert last_turn["llm_calls"] == 2
    assert last_turn["tool_calls"] == 1
    assert last_turn["tools"] == ["list_files"]
    assert last_turn["prompt_tokens"] == 250
    assert last_turn["cached_tokens"] == 180
    assert last_turn["completion_tokens"] == 30
    assert last_turn["cache_hit_rate"] == 0.72

    # 2. 会话隔离与前后台隔离：s1 进行中插入 s2 完整回合及 s1 后台 TaskEvent，不得污染 s1 前台
    obs.on_event(
        "s1",
        StreamDone(
            message={"role": "assistant", "content": None},
            usage=TokenUsage(prompt_tokens=40, cached_tokens=10, completion_tokens=5),
            ttft_ms=20.0,
            latency_ms=50.0,
        ),
    )
    obs.on_event("s1", ToolStep(tool="get_current_time", arguments="{}", observation="now", duration_ms=10.0))
    obs.on_event(
        "s2",
        StreamDone(
            message={"role": "assistant", "content": "s2回复"},
            usage=TokenUsage(prompt_tokens=80, cached_tokens=40, completion_tokens=8),
            ttft_ms=15.0,
            latency_ms=30.0,
        ),
    )
    obs.on_event("s2", TurnEnd(reply="s2回复", total_ms=32.0))
    obs.on_event(
        "s1",
        TaskEvent(
            kind="consolidation",
            duration_ms=88.0,
            usage=TokenUsage(prompt_tokens=90, cached_tokens=30, completion_tokens=15),
            details={"compacted_messages": 4},
        ),
    )
    obs.on_event(
        "s1",
        StreamDone(
            message={"role": "assistant", "content": "s1完成"},
            usage=TokenUsage(prompt_tokens=60, cached_tokens=20, completion_tokens=6),
            ttft_ms=10.0,
            latency_ms=40.0,
        ),
    )
    obs.on_event("s1", TurnEnd(reply="s1完成", total_ms=105.0))

    snap_s1 = obs.snapshot(session_key="s1")
    snap_s2 = obs.snapshot(session_key="s2")
    assert snap_s1["session"]["turns"] == 1
    assert snap_s1["session"]["tool_calls"] == 1
    assert snap_s1["session"]["llm_ms"] == 90.0
    assert snap_s1["session"]["prompt_tokens"] == 100
    assert snap_s2["session"]["turns"] == 1
    assert snap_s2["session"]["tool_calls"] == 0
    assert snap_s2["session"]["prompt_tokens"] == 80
    assert snap_s1["background"]["runs"] == 1
    assert snap_s1["background"]["prompt_tokens"] == 90

    # 3. 验证有界窗口（window_size=3）、有界 MRU（max_sessions=4）、recent_tasks 与非负配置防错
    # 先对已存在的 main 触发一次 TurnEnd 刷新 MRU，再新增 3 个会话淘汰较久未活跃的 s1/s2
    obs.on_event("main", StreamDone(message={"role": "assistant", "content": "refresh"}, ttft_ms=-3.0, latency_ms=-5.0))
    obs.on_event("main", ToolStep(tool="noop", arguments="{}", observation="ok", duration_ms=-2.0))
    obs.on_event("main", TurnEnd(reply="refresh", total_ms=10.0))
    assert obs.snapshot(session_key="main")["last_turn"]["ttft_ms"] == 0.0
    assert obs.snapshot(session_key="main")["last_turn"]["llm_ms"] == 0.0
    assert obs.snapshot(session_key="main")["last_turn"]["tool_ms"] == 0.0

    for idx in range(3):
        sk = f"extra-{idx}"
        obs.on_event(sk, StreamDone(message={"role": "assistant", "content": "ok"}, latency_ms=10.0))
        obs.on_event(sk, TurnEnd(reply="ok", total_ms=12.0))

    final_snap = obs.snapshot()
    assert final_snap["cursor"] == 7
    assert final_snap["buffered_turns"] == 3
    assert final_snap["totals"]["turns"] == 7
    recent = obs.recent_turns(since_seq=5)
    assert [item["seq"] for item in recent] == [6, 7]
    assert len(obs._sessions) == 4 and "main" in obs._sessions
    assert len(obs.recent_tasks(session_key="s1")) == 1

    # 未收到 TurnEnd 的进行中会话超过 max_sessions 时也必须有界淘汰
    for idx in range(6):
        obs.on_event(f"pending-{idx}", ToolStep(tool="ping", arguments="{}", observation="ok", duration_ms=1.0))
    assert len(obs._in_flight) == 4

    # 非法/布尔配置安全回退默认值，直接构造 0 抛出 ValueError
    bad_cfg = TelemetryConfig.from_dict({"window_size": True, "max_sessions": 0})
    assert bad_cfg == TelemetryConfig(window_size=200, max_sessions=256)
    import pytest
    with pytest.raises(ValueError):
        TelemetryConfig(window_size=0)


# 测试 AssistantApp 真实配置自动装配 telemetry 插件并采集工具回合与后台沉淀指标
def test_assistant_app_end_to_end_telemetry_and_consolidation(tmp_path: Path) -> None:
    # 区分主对话工具调用、工具后总结及后台记忆沉淀响应
    def dynamic_reply(self: LLMClient, messages: list[dict[str, object]], _tools: object) -> dict[str, object]:
        if "session_summary" in str(messages[0].get("content") or ""):
            self.last_usage = TokenUsage(prompt_tokens=90, cached_tokens=30, completion_tokens=20)
            return {
                "role": "assistant",
                "content": json.dumps({"session_summary": "已查询当前时间", "memories": []}, ensure_ascii=False),
            }
        if messages[-1].get("role") == "tool":
            self.last_usage = TokenUsage(prompt_tokens=110, cached_tokens=50, completion_tokens=12)
            return {"role": "assistant", "content": "现在时间已查到。"}
        self.last_usage = TokenUsage(prompt_tokens=70, cached_tokens=20, completion_tokens=8)
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "get_current_time", "arguments": "{}"},
                }
            ],
        }

    policy = ConsolidationPolicy(
        max_active_messages=2,
        max_active_chars=10000,
        keep_recent_messages=0,
        min_batch_messages=2,
    )
    app = AssistantApp(
        data_root=tmp_path / "data",
        config_path=REPO_ROOT / "config" / "tools.json",
        project_root=REPO_ROOT,
        llm=ScriptedStreamLLM(dynamic_reply),
        consolidation_policy=policy,
    )
    try:
        res = app.chat("现在几点？", chat_id="web")
        last_turn = res["telemetry"]["telemetry"]["last_turn"]
        assert last_turn["tool_calls"] == 1
        assert last_turn["tools"] == ["get_current_time"]
        assert last_turn["llm_calls"] == 2
        assert last_turn["prompt_tokens"] == 180

        app.wait_for_consolidation(timeout=5.0)
        st = app.status()
        assert st["observers"] == ["telemetry"]
        bg = st["telemetry"]["telemetry"]["background"]
        assert bg["runs"] == 1
        assert bg["failures"] == 0
        assert bg["last_duration_ms"] >= 0.0
        assert bg["prompt_tokens"] > 0
        assert bg["last_task"]["details"]["compacted_messages"] == 2
    finally:
        app.close()


# 测试多线程并发上报主回合与后台成功/失败事件时的严格线程安全性
def test_telemetry_thread_safety_under_concurrent_load() -> None:
    obs = TelemetryObserver(TelemetryConfig(window_size=12))
    threads_n = 8
    turns_per_thread = 10
    barrier = threading.Barrier(threads_n)

    # 单个工作线程模拟并发会话回合与后台任务上报
    def worker(tid: int) -> None:
        sk = f"t-{tid}"
        barrier.wait()
        for step in range(turns_per_thread):
            obs.on_event(
                sk,
                StreamDone(
                    message={"role": "assistant", "content": "ok"},
                    usage=TokenUsage(prompt_tokens=10, cached_tokens=4, completion_tokens=2),
                    ttft_ms=5.0,
                    latency_ms=10.0,
                ),
            )
            obs.on_event(sk, TurnEnd(reply="ok", total_ms=12.0))
            err = "RuntimeError: boom" if step == 0 else None
            obs.on_event(
                sk,
                TaskEvent(
                    kind="consolidation",
                    duration_ms=7.0,
                    usage=None if err else TokenUsage(prompt_tokens=20, cached_tokens=5, completion_tokens=3),
                    error=err,
                ),
            )

    workers = [threading.Thread(target=worker, args=(idx,)) for idx in range(threads_n)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()

    snap = obs.snapshot()
    total_turns = threads_n * turns_per_thread
    assert snap["totals"]["turns"] == total_turns
    assert snap["totals"]["prompt_tokens"] == total_turns * 10
    assert snap["buffered_turns"] == 12
    assert snap["background"]["runs"] == total_turns
    assert snap["background"]["failures"] == threads_n
    assert snap["background"]["prompt_tokens"] == (total_turns - threads_n) * 20
