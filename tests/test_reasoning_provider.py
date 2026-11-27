"""Slot-4 推理模型翻译器插件（plugins/reasoning_provider.py）单元测试与多轮 ReAct 集成测试。"""
from __future__ import annotations

import io
import json
from collections.abc import Iterator
from pathlib import Path
from urllib.error import HTTPError

import pytest

from personal_assistant.agent import PluginManager, ToolStart, ToolStep, TurnEnd
from personal_assistant.ai import (
    ApiProvider,
    ApiProviderRegistry,
    AssistantMessage,
    Context,
    ModelClient,
    ModelInvoker,
    ModelProfile,
    StreamDone,
    StreamError,
    StreamStart,
    TextBlock,
    TextDelta,
    TextEnd,
    TextStart,
    ThinkingBlock,
    ThinkingDelta,
    ThinkingEnd,
    ThinkingStart,
    TokenUsage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    from_model_profile,
)
from personal_assistant.assistant.app import AssistantApp, build_app
from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.assistant.common import REPO_ROOT


# 测试 Slot-4 插件自动发现、ApiProvider 注册与 models_dsl 一键切换 DeepSeek Reasoner（零环境变量）
def test_reasoning_provider_plugin_registers_slot4_and_builds_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LLM_MODE", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API", raising=False)

    manager = PluginManager.from_entries(["plugins"], REPO_ROOT)
    assert "openai-reasoning" in manager.providers
    assert isinstance(manager.providers["openai-reasoning"], ApiProvider)

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "tools.json").write_text(
        json.dumps(
            {
                "enabled_tools": ["get_current_time"],
                "plugins": ["plugins"],
                "files": {"read_roots": ["data/memory"], "write_root": "data/workspace"},
                "commands": [],
            }
        ),
        encoding="utf-8",
    )

    tcfg = ToolsConfig.load(cfg_dir / "tools.json", project_root=REPO_ROOT)
    assert "openai-reasoning" in tcfg.plugin_manager.providers
    assert any(p["id"] == "deepseek-reasoner" for p in tcfg.model_presets())

    app = build_app(data_root=tmp_path / "data", config_root=cfg_dir, project_root=REPO_ROOT)
    try:
        saved = app.configure_model(preset="reasoner", api_key="sk-r1-direct-secret")
        assert saved["default"] == "deepseek-reasoner"
        assert "sk-r1-direct-secret" not in json.dumps(saved, ensure_ascii=False)
        assert app.status()["llm_configured"] is True
        assert isinstance(app.llm, ModelInvoker)
        assert isinstance(app.llm, ModelClient)
        assert app.llm.api == "openai-reasoning"
        assert app.llm.model == "deepseek-reasoner"
    finally:
        app.close()


# 测试 OpenAIReasoningProvider 的请求报文规则（禁发 temperature、跨轮剥离/轮内透传 reasoning_content）与 12 事件流解析
def test_openai_reasoning_provider_wire_protocol_and_sse_parsing() -> None:
    from plugins.reasoning_provider import API_NAME, OpenAIReasoningProvider

    captured_requests: list[dict[str, object]] = []
    stream_attempts = 0

    # 模拟首包因 stream_options 返回 HTTP 400、去参数重试后流式吐出 reasoning_content + content + usage
    def fake_post_stream(url: str, payload: dict[str, object], headers: dict[str, str]) -> Iterator[str]:
        nonlocal stream_attempts
        stream_attempts += 1
        assert url == "https://api.deepseek.com/v1/chat/completions"
        assert headers["Authorization"] == "Bearer sk-wire-test"
        captured_requests.append(payload)
        if stream_attempts == 1:
            assert "stream_options" in payload
            raise HTTPError(url, 400, "Bad Request", {}, io.BytesIO(b'{"error":"stream_options unsupported"}'))
        assert "stream_options" not in payload
        lines = [
            'data: {"choices":[{"delta":{"role":"assistant","reasoning_content":"先梳理步骤"}}]}\n',
            'data: {"choices":[{"delta":{"reasoning_content":"，再作答"}}]}\n',
            'data: {"choices":[{"delta":{"content":"最终"}}]}\n',
            'data: {"choices":[{"delta":{"content":"结论"}}]}\n',
            'data: {"choices":[],"usage":{"prompt_tokens":120,"prompt_cache_hit_tokens":64,"completion_tokens":30}}\n',
            "data: [DONE]\n",
        ]
        return iter(lines)

    provider = OpenAIReasoningProvider(post_stream=fake_post_stream)
    assert provider.api == API_NAME
    reg = ApiProviderRegistry()
    reg.register(provider)

    profile = ModelProfile(
        name="r1",
        api=API_NAME,
        base_url="https://api.deepseek.com/v1/",
        model="deepseek-reasoner",
        api_key="sk-wire-test",
    )
    client = ModelClient(profile=profile, registry=reg)

    old_assistant = AssistantMessage(
        blocks=(
            ThinkingBlock("上一轮私密思维链不应再发给模型"),
            TextBlock("上一轮已回答"),
        ),
        stop_reason="stop",
    )
    intra_tool_msg = AssistantMessage(
        blocks=(
            ThinkingBlock("轮内工具调用思维链必须保留"),
            ToolCall(id="call_1", name="pong", arguments="{}"),
        ),
        stop_reason="toolUse",
    )
    ctx = Context(
        system_prompt="sys",
        messages=(
            UserMessage("第一轮问题"),
            old_assistant,
            UserMessage("第二轮问题"),
            intra_tool_msg,
            ToolResultMessage(tool_call_id="call_1", tool_name="pong", content="ok1"),
        ),
    )

    events = list(client.stream(ctx))
    assert stream_attempts == 2
    sent_payload = captured_requests[-1]
    assert "temperature" not in sent_payload
    assert "reasoning_effort" not in sent_payload
    wire_msgs = sent_payload["messages"]
    assert isinstance(wire_msgs, list)

    # 1. 上一轮助手消息的思维链在发往接口时被剥离，不污染历史
    assert "reasoning_content" not in wire_msgs[2]
    assert old_assistant.thinking == "上一轮私密思维链不应再发给模型"

    # 2. 当前轮次（最后一个 UserMessage 之后）的轮内工具调用思维链保持透传
    assert wire_msgs[4]["reasoning_content"] == "轮内工具调用思维链必须保留"

    # 3. 实时产出完整规范事件序列（ThinkingStart/Delta/End + TextStart/Delta/End + StreamDone）
    assert [type(e) for e in events] == [
        StreamStart,
        ThinkingStart,
        ThinkingDelta,
        ThinkingDelta,
        ThinkingEnd,
        TextStart,
        TextDelta,
        TextDelta,
        TextEnd,
        StreamDone,
    ]
    assert [e.text for e in events if isinstance(e, ThinkingDelta)] == ["先梳理步骤", "，再作答"]
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["最终", "结论"]
    done = next(e for e in events if isinstance(e, StreamDone))
    assert done.reason == "stop"
    assert done.message.text == "最终结论"
    assert done.message.thinking == "先梳理步骤，再作答"
    assert done.usage == TokenUsage(prompt_tokens=120, cached_tokens=64, completion_tokens=30)
    assert done.ttft_ms is not None and done.ttft_ms >= 0.0
    assert client.last_usage == done.usage


# 测试 AssistantApp 双轮端到端推理 + 工具调用：轮内透传 reasoning_content、跨轮剥离、观测指标与历史干净持久化
def test_assistant_app_two_turn_reasoning_tool_loop_and_telemetry(tmp_path: Path) -> None:
    from plugins.reasoning_provider import API_NAME, OpenAIReasoningProvider

    wire_history: list[list[dict[str, object]]] = []
    turn_streams = [
        # Turn 1 Step 1：思考后发起 get_current_time 工具调用
        [
            'data: {"choices":[{"delta":{"role":"assistant","reasoning_content":"先查当前时间"}}]}\n',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_time_1","type":"function","function":{"name":"get_current_time","arguments":"{}"}}]}}]}\n',
            'data: {"choices":[],"usage":{"prompt_tokens":200,"prompt_cache_hit_tokens":0,"completion_tokens":20}}\n',
            "data: [DONE]\n",
        ],
        # Turn 1 Step 2：收到工具结果与轮内 reasoning_content 后给出最终回复
        [
            'data: {"choices":[{"delta":{"role":"assistant","reasoning_content":"核对完毕"}}]}\n',
            'data: {"choices":[{"delta":{"content":"现在时间已确认。"}}]}\n',
            'data: {"choices":[],"usage":{"prompt_tokens":260,"prompt_cache_hit_tokens":192,"completion_tokens":25}}\n',
            "data: [DONE]\n",
        ],
        # Turn 2 Step 1：下一轮追问
        [
            'data: {"choices":[{"delta":{"role":"assistant","reasoning_content":"回顾上轮记录"}}]}\n',
            'data: {"choices":[{"delta":{"content":"上轮调用了 get_current_time。"}}]}\n',
            'data: {"choices":[],"usage":{"prompt_tokens":310,"prompt_cache_hit_tokens":256,"completion_tokens":18}}\n',
            "data: [DONE]\n",
        ],
    ]

    # 按序记录发往上游的消息切片并返回预设 SSE 帧
    def scripted_stream(url: str, payload: dict[str, object], headers: dict[str, str]) -> Iterator[str]:
        del url, headers
        assert "temperature" not in payload
        wire_history.append([dict(m) for m in payload["messages"]])  # type: ignore[arg-type]
        return iter(turn_streams.pop(0))

    reg = ApiProviderRegistry()
    reg.register(OpenAIReasoningProvider(post_stream=scripted_stream))
    llm = ModelClient(
        profile=ModelProfile(
            name="r1",
            api=API_NAME,
            base_url="https://api.deepseek.com/v1",
            model="deepseek-reasoner",
            api_key="sk-app-test",
        ),
        registry=reg,
    )
    app = AssistantApp(
        data_root=tmp_path / "data",
        config_path=REPO_ROOT / "config" / "tools.json",
        project_root=REPO_ROOT,
        llm=llm,
    )
    try:
        turn1_events: list[object] = []
        frames1 = list(app.stream_chat("现在几点了？", chat_id="r1-sess", on_event=turn1_events.append))
        assert [f["text"] for f in frames1 if f["type"] == "text_delta"] == ["现在时间已确认。"]
        assert [e.text for e in turn1_events if isinstance(e, ThinkingDelta)] == ["先查当前时间", "核对完毕"]
        assert any(isinstance(e, ToolStart) and e.tool == "get_current_time" for e in turn1_events)
        assert any(isinstance(e, ToolStep) and e.tool == "get_current_time" for e in turn1_events)
        assert any(isinstance(e, TurnEnd) and e.reply == "现在时间已确认。" for e in turn1_events)
        done1 = next(f for f in frames1 if f["type"] == "done")
        assert done1["cache"]["cached_tokens"] == 192  # type: ignore[index]
        last_telemetry = done1["telemetry"]["telemetry"]["last_turn"]  # type: ignore[index]
        assert last_telemetry["ttft_ms"] is not None
        assert last_telemetry["llm_ms"] > 0 and last_telemetry["tool_ms"] >= 0

        # Turn 1 Step 2（第二次 HTTP 请求）的最后一条 user 之后的 assistant 工具调用消息必须携带轮内 reasoning_content
        step2_msgs = wire_history[1]
        assistant_tc_msgs = [m for m in step2_msgs if m.get("role") == "assistant" and m.get("tool_calls")]
        assert len(assistant_tc_msgs) == 1
        assert assistant_tc_msgs[0]["reasoning_content"] == "先查当前时间"

        # Turn 2（第三次 HTTP 请求）：跨轮历史绝不携带任何 reasoning_content，且 StateStore 只存纯文本回复
        turn2 = app.chat("刚才用了什么工具？", chat_id="r1-sess")
        assert turn2["reply"] == "上轮调用了 get_current_time。"
        turn2_msgs = wire_history[2]
        assert all("reasoning_content" not in m for m in turn2_msgs)
        persisted = app.state.active_messages("r1-sess")
        assert [m["content"] for m in persisted] == [
            "现在几点了？",
            "现在时间已确认。",
            "刚才用了什么工具？",
            "上轮调用了 get_current_time。",
        ]
    finally:
        app.close()


# 测试推理插件边界条件：空白归一化、无思考工具调用轮内补 reasoning_content=""、分片 tool_calls 与流中错误收敛为 StreamError
def test_reasoning_provider_edge_cases_and_stream_errors() -> None:
    from plugins.reasoning_provider import API_NAME, OpenAIReasoningProvider

    # 1. ModelProfile.from_dict + from_model_profile 归一化字段后直通 ModelClient + ApiProviderRegistry
    profile = ModelProfile.from_dict(
        "r1",
        {
            "api": " openai-reasoning ",
            "base_url": " https://api.deepseek.com/v1 ",
            "model": " deepseek-reasoner ",
            "api_key": " sk-k ",
        },
    )
    reg = ApiProviderRegistry()
    reg.register(OpenAIReasoningProvider())
    built = from_model_profile(profile, registry=reg)
    assert isinstance(built, ModelClient)
    assert built.profile.api == API_NAME
    assert built.profile.api_key == "sk-k"
    assert built.profile.base_url == "https://api.deepseek.com/v1"
    assert built.profile.model == "deepseek-reasoner"

    # 2. 轮内无思维链但带 tool_calls 时，序列化仍补齐 reasoning_content="" 以满足 DeepSeek Reasoner 协议要求
    wire_captured: list[dict[str, object]] = []
    provider = OpenAIReasoningProvider(
        post_stream=lambda _u, p, _h: (
            wire_captured.append(p)
            or iter(['data: {"choices":[{"delta":{"content":"done"}}]}\n', "data: [DONE]\n"])
        )
    )
    list(
        provider.stream(
            profile,
            Context(
                messages=(
                    UserMessage("go"),
                    AssistantMessage(blocks=(ToolCall(id="c1", name="t", arguments="{}"),), stop_reason="toolUse"),
                    ToolResultMessage(tool_call_id="c1", tool_name="t", content="ok"),
                )
            ),
        )
    )
    sent_msgs = wire_captured[0]["messages"]
    assert isinstance(sent_msgs, list)
    assert sent_msgs[1]["reasoning_content"] == ""

    # 3. 多分片 tool_calls 拼接
    frag_provider = OpenAIReasoningProvider(
        post_stream=lambda _u, _p, _h: iter(
            [
                'data: {"choices":[{"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"c_frag","type":"function","function":{"name":"read_","arguments":"{\\"pa"}}]}}]}\n',
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"file","arguments":"th\\":\\"a\\"}"}}]}}]}\n',
                "data: [DONE]\n",
            ]
        )
    )
    frag_events = list(frag_provider.stream(profile, Context(messages=(UserMessage("x"),))))
    assert not any(isinstance(e, ThinkingDelta) for e in frag_events)
    frag_done = next(e for e in frag_events if isinstance(e, StreamDone))
    assert frag_done.reason == "toolUse"
    assert frag_done.message.tool_calls == (ToolCall(id="c_frag", name="read_file", arguments='{"path":"a"}'),)

    # 4. 流中 error 帧收敛为 StreamError，ModelClient.complete() 抛出 RuntimeError
    err_reg = ApiProviderRegistry()
    err_reg.register(
        OpenAIReasoningProvider(
            post_stream=lambda _u, _p, _h: iter(['data: {"error":{"message":"bad upstream"}}\n'])
        )
    )
    err_client = ModelClient(profile=profile, registry=err_reg)
    err_events = list(err_client.stream(Context(messages=(UserMessage("x"),))))
    assert isinstance(err_events[-1], StreamError)
    assert "bad upstream" in err_events[-1].error_message
    with pytest.raises(RuntimeError, match="bad upstream"):
        err_client.complete(Context(messages=(UserMessage("x"),)))
