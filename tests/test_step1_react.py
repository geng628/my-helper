"""Unit tests for Step 1: minimal ReAct loop, ToolRegistry, and LLM client."""
from __future__ import annotations

import json

import pytest

from personal_assistant.assistant.cli import build_demo_tools
from personal_assistant.ai import (
    LLMClient,
    LLMNotConfigured,
    NoopLLMClient,
    OpenAICompatibleLLM,
    ToolCall,
    extract_tool_calls,
    from_env,
)
from personal_assistant.agent.react import ToolStart, ToolStep, TurnEnd, iter_react, run_react
from personal_assistant.agent.tools import Tool, ToolRegistry


# 测试用确定性脚本模拟模型
class ScriptedLLM(LLMClient):
    """Deterministic fake LLM that snapshots every request and replays scripted replies."""

    # 初始化预设响应序列与请求记录列表
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = list(responses)
        self.requests: list[tuple[list[dict[str, object]], list[dict[str, object]] | None]] = []

    # 记录请求快照并弹出下一条预设响应
    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        self.requests.append(([dict(msg) for msg in messages], tools))
        if not self._responses:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        return self._responses.pop(0)


# 构造标准 OpenAI 格式工具调用字典
def _tool_call(call_id: str, name: str, arguments: str) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


# 测试无工具调用时直接返回文本并拒绝已废弃的旧参数 on_step
def test_direct_answer_without_tool_calls() -> None:
    llm = ScriptedLLM([{"role": "assistant", "content": "你好，有什么可以帮你？"}])
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "你好"}]

    with pytest.raises(TypeError):
        run_react(llm, messages, build_demo_tools(), on_step=lambda *_: None)  # type: ignore[call-arg]

    reply = run_react(llm, messages, build_demo_tools())

    assert reply == "你好，有什么可以帮你？"
    assert len(llm.requests) == 1
    assert llm.requests[0][1] is not None
    assert messages[-1] == {"role": "assistant", "content": "你好，有什么可以帮你？"}


# 测试单步工具调用 ReAct 循环与 ToolStep 耗时采集
def test_single_step_react_loop() -> None:
    steps: list[ToolStep] = []
    tools = ToolRegistry(
        Tool(
            name="lookup_weather",
            description="查询城市天气",
            parameters={"city": "城市名称"},
            handler=lambda city: f"{city}：晴，22°C",
        )
    )
    llm = ScriptedLLM(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("call_1", "lookup_weather", '{"city": "上海"}')],
            },
            {"role": "assistant", "content": "上海今天晴，气温 22°C。"},
        ]
    )
    messages: list[dict[str, object]] = [{"role": "user", "content": "上海今天天气怎么样？"}]

    reply = run_react(
        llm,
        messages,
        tools,
        on_event=lambda ev: steps.append(ev) if isinstance(ev, ToolStep) else None,
    )

    assert reply == "上海今天晴，气温 22°C。"
    assert [(s.tool, s.arguments, s.observation) for s in steps] == [
        ("lookup_weather", '{"city": "上海"}', "上海：晴，22°C")
    ]
    assert steps[0].duration_ms >= 0.0
    second_msgs = llm.requests[1][0]
    assert second_msgs[-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "上海：晴，22°C",
    }


# 测试多步链式工具调用 ReAct 循环
def test_multi_step_chained_react_loop() -> None:
    steps: list[ToolStep] = []
    tools = ToolRegistry(
        Tool(
            name="find_note",
            description="根据主题查找笔记路径",
            parameters={"topic": "主题"},
            handler=lambda topic: f"notes/{topic}.md",
        ),
        Tool(
            name="read_file",
            description="读取笔记内容",
            parameters={"path": "文件路径"},
            handler=lambda path: f"[{path}] 持有自旋锁时禁止 sleep。",
        ),
    )
    llm = ScriptedLLM(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("c1", "find_note", '{"topic": "spinlock"}')],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("c2", "read_file", '{"path": "notes/spinlock.md"}')],
            },
            {"role": "assistant", "content": "根据 notes/spinlock.md，持有自旋锁时禁止 sleep。"},
        ]
    )
    messages: list[dict[str, object]] = [{"role": "user", "content": "自旋锁能 sleep 吗？"}]

    reply = run_react(
        llm,
        messages,
        tools,
        on_event=lambda ev: steps.append(ev) if isinstance(ev, ToolStep) else None,
    )

    assert "禁止 sleep" in reply
    assert [s.tool for s in steps] == ["find_note", "read_file"]
    assert len(llm.requests) == 3


# 测试工具错误作为观察结果回传并自愈
def test_tool_error_as_observation_self_healing() -> None:
    steps: list[ToolStep] = []
    tools = ToolRegistry(
        Tool(
            name="add_numbers",
            description="两数相加",
            parameters={"a": "第一个整数", "b": "第二个整数"},
            handler=lambda a, b: str(int(a) + int(b)),
        )
    )
    llm = ScriptedLLM(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _tool_call("bad_tool", "nonexistent", "{}"),
                    _tool_call("bad_json", "add_numbers", "{not-json"),
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("good", "add_numbers", '{"a": "19", "b": "23"}')],
            },
            {"role": "assistant", "content": "19 + 23 = 42。"},
        ]
    )
    messages: list[dict[str, object]] = [{"role": "user", "content": "算一下 19 加 23"}]

    reply = run_react(
        llm,
        messages,
        tools,
        on_event=lambda ev: steps.append(ev) if isinstance(ev, ToolStep) else None,
    )

    assert reply == "19 + 23 = 42。"
    assert "未知工具" in steps[0].observation
    assert "JSON 解析失败" in steps[1].observation
    assert steps[2].observation == "42"


# 测试纯 ReAct 循环移除轮次上限与 tools=None 兜底，仅在无 tool_calls 时自然终止且全程保持 tools 前缀恒定
def test_unbounded_react_loop_stops_only_when_no_tool_calls(tmp_path) -> None:
    import personal_assistant.agent as agent_pkg
    import personal_assistant.agent.react as react_mod
    from evals.agent_eval import evaluate_agent
    from personal_assistant.agent.react import iter_react
    from personal_assistant.assistant.app import AssistantApp

    assert not hasattr(react_mod, "MAX_TOOL_ROUNDS")
    assert not hasattr(agent_pkg, "MAX_TOOL_ROUNDS")

    steps: list[ToolStep] = []
    tools = ToolRegistry(
        Tool(
            name="ping",
            description="探测工具",
            parameters={},
            handler=lambda: "pong",
        )
    )
    expected_defs = tools.definitions()
    llm = ScriptedLLM(
        [
            {"role": "assistant", "content": None, "tool_calls": [_tool_call(f"c{i}", "ping", "{}")]}
            for i in range(1, 7)
        ]
        + [{"role": "assistant", "content": "已完成 6 轮连续探测。"}]
    )
    messages: list[dict[str, object]] = [{"role": "user", "content": "连续探测直到完成"}]

    with pytest.raises(TypeError):
        iter_react(llm, messages, tools, max_rounds=4)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        run_react(llm, messages, tools, max_rounds=4)  # type: ignore[call-arg]

    app = AssistantApp(data_root=tmp_path / "app_data", llm=llm)
    try:
        with pytest.raises(TypeError):
            app.chat("你好", max_rounds=4)  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            app.stream_chat("你好", max_rounds=4)  # type: ignore[call-arg]
    finally:
        app.close()

    with pytest.raises(TypeError):
        evaluate_agent(llm, tmp_path / "eval_data", [], max_rounds=4)  # type: ignore[call-arg]

    reply = run_react(
        llm,
        messages,
        tools,
        on_event=lambda ev: steps.append(ev) if isinstance(ev, ToolStep) else None,
    )

    assert reply == "已完成 6 轮连续探测。"
    assert len(steps) == 6
    assert len(llm.requests) == 7
    assert all(req_tools == expected_defs for _, req_tools in llm.requests)
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 6
    assert all(m["content"] == "pong" for m in tool_msgs)


# 测试从环境变量构造客户端及 OpenAI 请求负载格式
def test_llm_client_from_env_and_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODE", "off")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    client = from_env()
    assert isinstance(client, NoopLLMClient)
    with pytest.raises(LLMNotConfigured):
        client.respond([{"role": "user", "content": "hi"}])

    captured: dict[str, object] = {}

    # 模拟捕获请求参数的同步 JSON 发送桩
    def fake_post(url: str, payload: dict[str, object], headers: dict[str, str]) -> dict[str, object]:
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    real_client = OpenAICompatibleLLM(
        api_key="sk-test",
        base_url="https://example.com/v1/",
        model="demo-model",
        post_json=fake_post,
    )
    msg = real_client.respond(
        [{"role": "user", "content": "hi"}],
        tools=build_demo_tools().definitions(),
    )
    assert msg == {"role": "assistant", "content": "ok"}
    assert captured["url"] == "https://example.com/v1/chat/completions"
    assert captured["headers"] == {"Authorization": "Bearer sk-test"}
    sent_payload = captured["payload"]
    assert isinstance(sent_payload, dict)
    assert sent_payload["model"] == "demo-model"
    assert sent_payload["tool_choice"] == "auto"
    assert json.loads(json.dumps(sent_payload["tools"]))[0]["function"]["name"] == "get_current_time"


# 测试 default_post_json 对瞬时 TLS/网络抖动自动重试，但对 HTTP 400 客户端错误绝不重试
def test_default_post_json_retries_transient_error_but_not_http_400(monkeypatch) -> None:
    import io
    import ssl
    from urllib.error import HTTPError, URLError

    from personal_assistant.ai import client as llm_mod

    # 避免单元测试等待真实退避时间
    monkeypatch.setattr(llm_mod.time, "sleep", lambda _: None)

    attempts = {"count": 0}

    # 模拟前 2 次遇到 TLS EOF，第 3 次成功返回 JSON
    def flaky_urlopen(request, timeout=60):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise URLError(ssl.SSLEOFError("EOF occurred in violation of protocol"))

        class _Resp:
            # 进入上下文返回自身
            def __enter__(self):
                return self

            # 退出上下文无操作
            def __exit__(self, *args):
                return False

            # 返回成功 JSON 字节串
            def read(self) -> bytes:
                return b'{"ok": true}'

        return _Resp()

    monkeypatch.setattr(llm_mod, "urlopen", flaky_urlopen)
    assert llm_mod.default_post_json("https://example.com/v1", {"a": 1}, {}) == {"ok": True}
    assert attempts["count"] == 3

    bad_attempts = {"count": 0}

    # 模拟 400 Bad Request：必须立刻抛出，不重试
    def bad_request_urlopen(request, timeout=60):
        bad_attempts["count"] += 1
        raise HTTPError("https://example.com/v1", 400, "Bad Request", {}, io.BytesIO(b"bad"))

    monkeypatch.setattr(llm_mod, "urlopen", bad_request_urlopen)
    with pytest.raises(HTTPError):
        llm_mod.default_post_json("https://example.com/v1", {"a": 1}, {})
    assert bad_attempts["count"] == 1


# 测试 ToolCall 双向契约与 extract_tool_calls 对字典和实例的兼容提取
def test_tool_call_contract_and_extract_tool_calls() -> None:
    tc = ToolCall(id="call_a", name="ping", arguments='{"host": "127.0.0.1"}')
    wire = tc.to_openai_dict()
    assert wire == {
        "id": "call_a",
        "type": "function",
        "function": {"name": "ping", "arguments": '{"host": "127.0.0.1"}'},
    }
    assert ToolCall.from_raw(wire) == tc
    assert ToolCall.from_raw(tc) is tc
    assert ToolCall.from_raw({"id": "c_empty", "type": "function", "function": {"name": "noop", "arguments": ""}}) == ToolCall(
        id="c_empty",
        name="noop",
        arguments="{}",
    )
    assert ToolCall.from_raw({"id": "c_none", "type": "function", "function": {"name": "noop", "arguments": None}}) == ToolCall(
        id="c_none",
        name="noop",
        arguments="{}",
    )
    assert ToolCall.from_raw({"id": "c_flat", "name": "echo", "arguments": {"text": "hi"}}) == ToolCall(
        id="c_flat",
        name="echo",
        arguments='{"text": "hi"}',
    )
    with pytest.raises(TypeError):
        ToolCall.from_raw("invalid-tool-call")

    extracted = extract_tool_calls({"role": "assistant", "tool_calls": [tc, wire]})
    assert extracted == [tc, tc]
    assert extract_tool_calls({"role": "assistant", "content": "hi"}) == []
    assert extract_tool_calls({"role": "assistant", "tool_calls": None}) == []

    # 验证 iter_react 可直接消费 ToolCall 实例，并先产出 ToolStart 再执行工具后产出 ToolStep
    exec_order: list[str] = []
    tools = ToolRegistry(
        Tool(
            name="ping",
            description="探测",
            parameters={"host": "主机"},
            handler=lambda host: (exec_order.append(f"exec:{host}") or f"pong:{host}"),
        )
    )
    llm = ScriptedLLM(
        [
            {"role": "assistant", "content": None, "tool_calls": [tc]},
            {"role": "assistant", "content": "完成探测"},
        ]
    )
    events: list[object] = []
    for ev in iter_react(llm, [{"role": "user", "content": "ping"}], tools):
        if isinstance(ev, ToolStart):
            exec_order.append(f"start:{ev.call_id}")
        elif isinstance(ev, ToolStep):
            exec_order.append(f"step:{ev.call_id}")
        events.append(ev)

    assert exec_order == ["start:call_a", "exec:127.0.0.1", "step:call_a"]
    assert ToolStart(tool="ping", arguments='{"host": "127.0.0.1"}', call_id="call_a") in events
    assert any(isinstance(ev, ToolStep) and ev.call_id == "call_a" for ev in events)


# 测试净化退出消息：移除硬编码中文兜底并保留 assistant_msg 上的 reasoning_content 等扩展字段
def test_exit_message_purified_and_preserves_extension_fields() -> None:
    empty_llm = ScriptedLLM(
        [
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "空回复前的内部推理",
                "custom_meta": {"source": "model"},
            }
        ]
    )
    msgs1: list[dict[str, object]] = [{"role": "user", "content": "hi"}]
    reply1 = run_react(empty_llm, msgs1, ToolRegistry())
    assert reply1 == ""
    assert "模型没有返回回复" not in reply1
    assert msgs1[-1] == {
        "role": "assistant",
        "content": "",
        "reasoning_content": "空回复前的内部推理",
        "custom_meta": {"source": "model"},
    }

    text_llm = ScriptedLLM(
        [
            {
                "role": "assistant",
                "content": "  最终答案  ",
                "reasoning_content": "推演细节",
            }
        ]
    )
    msgs2: list[dict[str, object]] = [{"role": "user", "content": "q"}]
    reply2 = run_react(text_llm, msgs2, ToolRegistry())
    assert reply2 == "最终答案"
    assert msgs2[-1] == {
        "role": "assistant",
        "content": "最终答案",
        "reasoning_content": "推演细节",
    }


# 测试双阶段上下文管道（transform_context -> convert_to_llm）不污染原列表且每轮过滤应用自定义消息
def test_transform_context_and_convert_to_llm_pipeline() -> None:
    seen_stages: list[str] = []
    tools = ToolRegistry(
        Tool(name="echo", description="回显", parameters={"v": "值"}, handler=lambda v: f"ok:{v}")
    )
    llm = ScriptedLLM(
        [
            {"role": "assistant", "content": None, "tool_calls": [_tool_call("c1", "echo", '{"v": "1"}')]},
            {"role": "assistant", "content": "管道执行完毕"},
        ]
    )
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "sys"},
        {"role": "artifact", "content": "仅前端渲染的卡片对象"},
        {"role": "user", "content": "请调用 echo"},
    ]
    snapshot_before = list(messages)

    # 第一阶段：克隆后追加临时提示（并尝试污染列表尾部验证不影响主历史）
    def transform_context(ctx: list[dict[str, object]]) -> list[dict[str, object]]:
        seen_stages.append("transform")
        ctx.append({"role": "transient_hint", "content": "临时提醒"})
        return ctx

    # 第二阶段：过滤掉应用层自定义角色（artifact / transient_hint），仅保留标准 LLM 消息
    def convert_to_llm(ctx: list[dict[str, object]]) -> list[dict[str, object]]:
        seen_stages.append("convert")
        return [m for m in ctx if m.get("role") in {"system", "user", "assistant", "tool"}]

    reply = run_react(
        llm,
        messages,
        tools,
        transform_context=transform_context,
        convert_to_llm=convert_to_llm,
    )

    assert reply == "管道执行完毕"
    assert seen_stages == ["transform", "convert", "transform", "convert"]
    # 原始自定义消息仍完整保留在主 messages 中，且未被注入 transient_hint
    assert messages[:3] == snapshot_before
    assert not any(m.get("role") == "transient_hint" for m in messages)
    # 发往模型的请求已彻底剥离 artifact 与 transient_hint
    for req_msgs, _ in llm.requests:
        assert all(m.get("role") in {"system", "user", "assistant", "tool"} for m in req_msgs)
    assert [m["role"] for m in llm.requests[0][0]] == ["system", "user"]
    assert [m["role"] for m in llm.requests[1][0]] == ["system", "user", "assistant", "tool"]

    # 剥离全部钩子（默认 None）且仅传 convert_to_llm 时同样不污染原 messages
    filtered_msgs: list[dict[str, object]] = [
        {"role": "artifact", "content": "ui"},
        {"role": "user", "content": "hi"},
    ]
    clean_llm = ScriptedLLM([{"role": "assistant", "content": "ok"}])

    # 仅提供 convert_to_llm，并在回调中 pop 测试是否隔离原列表
    def mutating_convert(ctx: list[dict[str, object]]) -> list[dict[str, object]]:
        return [m for m in ctx if m.get("role") != "artifact"] + [{"role": "user", "content": "extra"}]

    assert run_react(clean_llm, filtered_msgs, ToolRegistry(), convert_to_llm=mutating_convert) == "ok"
    assert [m["role"] for m in filtered_msgs] == ["artifact", "user", "assistant"]
    assert [m["role"] for m in clean_llm.requests[0][0]] == ["user", "user"]
