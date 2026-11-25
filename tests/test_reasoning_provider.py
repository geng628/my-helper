"""Slot-4 推理模型协议插件（plugins/reasoning_provider.py）单元测试与多轮 ReAct 集成测试（TDD）。"""
from __future__ import annotations

import io
import json
import threading
from pathlib import Path
from urllib.error import HTTPError

import pytest

from personal_assistant.agent import PluginManager, ToolStep, TurnEnd
from personal_assistant.ai import (
    LLMClient,
    OpenAIChatClient,
    OpenAICompatibleLLM,
    StreamDone,
    TextDelta,
    TokenUsage,
    available_providers,
)
from personal_assistant.assistant.app import AssistantApp, build_app
from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.assistant.common import REPO_ROOT


# 测试 Slot-4 插件自动发现、Provider 注册与 models_dsl 一键切换 DeepSeek Reasoner（零环境变量）
def test_reasoning_provider_plugin_registers_slot4_and_builds_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LLM_MODE", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    manager = PluginManager.from_entries(["plugins"], REPO_ROOT)
    assert "openai_reasoning" in manager.providers

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
    assert "openai_reasoning" in available_providers()
    assert any(p["id"] == "deepseek_reasoner" for p in tcfg.model_presets())

    app = build_app(data_root=tmp_path / "data", config_root=cfg_dir, project_root=REPO_ROOT)
    try:
        saved = app.configure_model(preset="reasoner", api_key="sk-r1-direct-secret")
        assert saved["default"] == "deepseek_reasoner"
        assert "sk-r1-direct-secret" not in json.dumps(saved, ensure_ascii=False)
        assert app.status()["llm_configured"] is True
        assert isinstance(app.llm, LLMClient)
        assert isinstance(app.llm, OpenAIChatClient)
        assert not isinstance(app.llm, OpenAICompatibleLLM)
        assert type(app.llm).__name__ == "OpenAIReasoningLLM"
        assert app.llm.model == "deepseek-reasoner"  # type: ignore[attr-defined]
        assert not hasattr(app.llm, "temperature")
    finally:
        app.close()


# 测试 OpenAIReasoningLLM 的请求报文规则（禁发 temperature、跨轮剥离/轮内透传 reasoning_content）与 SSE 解析
def test_openai_reasoning_llm_wire_protocol_and_sse_parsing() -> None:
    from plugins.reasoning_provider import PROVIDER_NAME, OpenAIReasoningLLM, OpenAIReasoningProvider

    provider = OpenAIReasoningProvider()
    assert provider.name == PROVIDER_NAME
    built = provider.build({"LLM_API_KEY": "sk-env-r1", "LLM_BASE_URL": "https://api.deepseek.com/v1"})
    assert isinstance(built, OpenAIReasoningLLM)
    assert built.model == "deepseek-reasoner"

    captured_requests: list[dict[str, object]] = []
    stream_attempts = 0

    # 模拟首包因 stream_options 返回 HTTP 400、去参数重试后流式吐出 reasoning_content + content + usage
    def fake_post_stream(url: str, payload: dict[str, object], headers: dict[str, str]):
        nonlocal stream_attempts
        stream_attempts += 1
        assert url == "https://api.deepseek.com/v1/chat/completions"
        assert headers["Authorization"] == "Bearer sk-wire-test"
        captured_requests.append(payload)
        if stream_attempts == 1:
            assert "stream_options" in payload
            raise HTTPError(url, 400, "stream_options unsupported", None, io.BytesIO(b""))  # type: ignore[arg-type]
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

    llm = OpenAIReasoningLLM(
        api_key="sk-wire-test",
        base_url="https://api.deepseek.com/v1/",
        model="deepseek-reasoner",
        post_stream=fake_post_stream,
    )

    old_assistant = {
        "role": "assistant",
        "content": "上一轮已回答",
        "reasoning_content": "上一轮私密思维链不应再发给模型",
    }
    intra_tool_msg_with_rc = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "轮内工具调用思维链必须保留",
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "pong", "arguments": "{}"}}
        ],
    }
    input_messages: list[dict[str, object]] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "第一轮问题"},
        old_assistant,
        {"role": "user", "content": "第二轮问题"},
        intra_tool_msg_with_rc,
        {"role": "tool", "tool_call_id": "call_1", "content": "ok1"},
    ]

    events = list(llm.stream(input_messages))
    assert stream_attempts == 2
    sent_payload = captured_requests[-1]
    assert "temperature" not in sent_payload
    wire_msgs = sent_payload["messages"]
    assert isinstance(wire_msgs, list)

    # 1. 上一轮助手消息的 reasoning_content 在发往接口时被剥离，且不原地篡改调用方字典
    assert "reasoning_content" not in wire_msgs[2]
    assert old_assistant["reasoning_content"] == "上一轮私密思维链不应再发给模型"

    # 2. 当前轮次（最后一个 user 之后）的轮内工具调用思维链保持透传
    assert wire_msgs[4]["reasoning_content"] == "轮内工具调用思维链必须保留"

    # 3. TextDelta 仅吐出 delta.content（绝不泄漏 reasoning_content），StreamDone 携带完整 reasoning_content 与 usage
    text_deltas = [e.text for e in events if isinstance(e, TextDelta)]
    assert text_deltas == ["最终", "结论"]
    done = next(e for e in events if isinstance(e, StreamDone))
    assert done.message == {
        "role": "assistant",
        "content": "最终结论",
        "reasoning_content": "先梳理步骤，再作答",
    }
    assert done.usage == TokenUsage(prompt_tokens=120, cached_tokens=64, completion_tokens=30)
    assert done.ttft_ms is not None and done.ttft_ms >= 0.0
    assert llm.last_usage == done.usage

    # 4. 同步 respond() 同样禁发 temperature 并保证 tool_calls 返回携带字符串形式 reasoning_content
    sync_payloads: list[dict[str, object]] = []

    # 模拟非流式接口返回完整助手消息与 usage
    def fake_post_json(url: str, payload: dict[str, object], headers: dict[str, str]) -> dict[str, object]:
        sync_payloads.append(payload)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": None,
                        "tool_calls": [
                            {"id": "c_sync", "type": "function", "function": {"name": "t", "arguments": "{}"}}
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 50, "prompt_cache_hit_tokens": 0, "completion_tokens": 12},
        }

    sync_llm = OpenAIReasoningLLM(
        api_key="sk-sync",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-reasoner",
        post_json=fake_post_json,
    )
    sync_msg = sync_llm.respond([{"role": "user", "content": "同步调用"}])
    assert "temperature" not in sync_payloads[0]
    assert sync_msg["reasoning_content"] == ""
    assert sync_llm.last_usage == TokenUsage(prompt_tokens=50, cached_tokens=0, completion_tokens=12)


# 测试 AssistantApp 双轮端到端推理 + 工具调用：轮内透传 reasoning_content、跨轮剥离、观测指标与历史干净持久化
def test_assistant_app_two_turn_reasoning_tool_loop_and_telemetry(tmp_path: Path) -> None:
    from plugins.reasoning_provider import OpenAIReasoningLLM

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
    def scripted_stream(url: str, payload: dict[str, object], headers: dict[str, str]):
        assert "temperature" not in payload
        wire_history.append([dict(m) for m in payload["messages"]])  # type: ignore[arg-type]
        return iter(turn_streams.pop(0))

    llm = OpenAIReasoningLLM(
        api_key="sk-app-test",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-reasoner",
        post_stream=scripted_stream,
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
        assert any(isinstance(e, ToolStep) and e.tool == "get_current_time" for e in turn1_events)
        assert any(isinstance(e, TurnEnd) and e.reply == "现在时间已确认。" for e in turn1_events)
        done1 = next(f for f in frames1 if f["type"] == "done")
        assert done1["cache"]["cached_tokens"] == 192
        last_telemetry = done1["telemetry"]["telemetry"]["last_turn"]
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


# 测试推理客户端边界条件：空白环境变量兜底、流中报错、分片 tool_calls 拼接、HTTPError 不盲试与多线程 last_usage 隔离
def test_reasoning_provider_edge_cases_errors_and_thread_isolation() -> None:
    from plugins.reasoning_provider import OpenAIReasoningLLM, OpenAIReasoningProvider

    # 1. env 中传入空字符串时必须回退默认 base_url 与 model
    built = OpenAIReasoningProvider().build({"LLM_API_KEY": " sk-k ", "LLM_BASE_URL": "  ", "LLM_MODEL": ""})
    assert isinstance(built, OpenAIReasoningLLM)
    assert built.api_key == "sk-k"
    assert built.base_url == "https://api.deepseek.com/v1"
    assert built.model == "deepseek-reasoner"

    # 2. respond() 无工具调用且 reasoning_content 为 None 或 "" 时均不带冗余字段（与 stream 保持一致）
    none_reply_llm = OpenAIReasoningLLM(
        api_key="sk",
        base_url="https://api.deepseek.com/v1",
        post_json=lambda _u, _p, _h: {
            "choices": [{"message": {"role": "assistant", "content": "纯文本", "reasoning_content": None}}],
        },
    )
    assert "reasoning_content" not in none_reply_llm.respond([{"role": "user", "content": "hi"}])

    empty_reply_llm = OpenAIReasoningLLM(
        api_key="sk",
        base_url="https://api.deepseek.com/v1",
        post_json=lambda _u, _p, _h: {
            "choices": [{"message": {"role": "assistant", "content": "纯文本", "reasoning_content": ""}}],
        },
    )
    assert "reasoning_content" not in empty_reply_llm.respond([{"role": "user", "content": "hi"}])

    # 3. 多分片 tool_calls 拼接（不含 reasoning_content 时也必须补 reasoning_content=""）
    frag_llm = OpenAIReasoningLLM(
        api_key="sk",
        base_url="https://api.deepseek.com/v1",
        post_stream=lambda _u, _p, _h: iter(
            [
                'data: {"choices":[{"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"c_frag","type":"function","function":{"name":"read_","arguments":"{\\"pa"}}]}}]}\n',
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"file","arguments":"th\\":\\"a\\"}"}}]}}]}\n',
                "data: [DONE]\n",
            ]
        ),
    )
    frag_done = next(e for e in frag_llm.stream([{"role": "user", "content": "x"}]) if isinstance(e, StreamDone))
    assert frag_done.message["reasoning_content"] == ""
    assert frag_done.message["tool_calls"] == [
        {"id": "c_frag", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"a"}'}}
    ]

    # 4. 流中 error 帧直接抛出 RuntimeError；非 400 或已产出首包后的 400 不重试
    err_llm = OpenAIReasoningLLM(
        api_key="sk",
        base_url="https://api.deepseek.com/v1",
        post_stream=lambda _u, _p, _h: iter(['data: {"error":{"message":"bad upstream"}}\n']),
    )
    with pytest.raises(RuntimeError, match="bad upstream"):
        list(err_llm.stream([{"role": "user", "content": "x"}]))

    # 非 400 错误不重试
    attempts_500 = 0

    # 模拟 500 错误，验证不会触发 stream_options 降级重试
    def fail_500(url: str, _p: dict[str, object], _h: dict[str, str]):
        nonlocal attempts_500
        attempts_500 += 1
        raise HTTPError(url, 500, "server error", None, io.BytesIO(b""))  # type: ignore[arg-type]

    with pytest.raises(HTTPError):
        list(OpenAIReasoningLLM("sk", "https://x/v1", post_stream=fail_500).stream([{"role": "user", "content": "x"}]))
    assert attempts_500 == 1

    # 已吐出首块后再报 400 不重试
    attempts_mid400 = 0

    # 先吐出一行后抛出 400，验证 yielded_any=True 时不重发整段请求
    def mid_400(url: str, _p: dict[str, object], _h: dict[str, str]):
        nonlocal attempts_mid400
        attempts_mid400 += 1
        yield 'data: {"choices":[{"delta":{"reasoning_content":"ok"}}]}\n'
        raise HTTPError(url, 400, "mid error", None, io.BytesIO(b""))  # type: ignore[arg-type]

    with pytest.raises(HTTPError):
        list(OpenAIReasoningLLM("sk", "https://x/v1", post_stream=mid_400).stream([{"role": "user", "content": "x"}]))
    assert attempts_mid400 == 1

    # 5. 多线程共享同一 OpenAIReasoningLLM 实例时 last_usage 按线程严格隔离
    barrier = threading.Barrier(2)

    # 根据消息内容返回不同 token 用量以检验 threading.local 隔离性
    def threaded_post_json(_u: str, payload: dict[str, object], _h: dict[str, str]) -> dict[str, object]:
        tag = payload["messages"][0]["content"]  # type: ignore[index]
        tok = 111 if tag == "t1" else 222
        return {
            "choices": [{"message": {"role": "assistant", "content": str(tag)}}],
            "usage": {"prompt_tokens": tok, "prompt_cache_hit_tokens": 0, "completion_tokens": 1},
        }

    shared_llm = OpenAIReasoningLLM("sk", "https://x/v1", post_json=threaded_post_json)
    results: dict[str, int] = {}

    # 在各自线程完成调用与栅栏同步后读取本线程 last_usage
    def worker(tag: str) -> None:
        shared_llm.respond([{"role": "user", "content": tag}])
        barrier.wait(timeout=3.0)
        assert shared_llm.last_usage is not None
        results[tag] = shared_llm.last_usage.prompt_tokens

    t1 = threading.Thread(target=worker, args=("t1",))
    t2 = threading.Thread(target=worker, args=("t2",))
    t1.start()
    t2.start()
    t1.join(timeout=3.0)
    t2.join(timeout=3.0)
    assert results == {"t1": 111, "t2": 222}
