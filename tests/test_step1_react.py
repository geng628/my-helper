"""Unit tests for Step 1: minimal ReAct loop, ToolRegistry, and LLM client."""
from __future__ import annotations

import json

import pytest

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


# 构造纯内存演示工具注册表，确保 Layer 1/2 单元测试与 Layer 3 cli.py 零反向耦合
def _demo_tools() -> ToolRegistry:
    return ToolRegistry(
        Tool(
            name="get_current_time",
            description="获取当前时间",
            parameters={},
            handler=lambda: "2026-11-26 12:00:00 (Asia/Shanghai)",
        )
    )


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
        run_react(llm, messages, _demo_tools(), on_step=lambda *_: None)  # type: ignore[call-arg]

    reply = run_react(llm, messages, _demo_tools())

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
        tools=_demo_tools().definitions(),
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

        # 模拟成功 JSON 响应的上下文管理器
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


# 测试 ToolCall 单轨标准 OpenAI wire 契约与 extract_tool_calls 严格校验
def test_tool_call_contract_and_extract_tool_calls() -> None:
    tc = ToolCall(id="call_a", name="ping", arguments='{"host": "127.0.0.1"}')
    wire = tc.to_openai_dict()
    assert wire == {
        "id": "call_a",
        "type": "function",
        "function": {"name": "ping", "arguments": '{"host": "127.0.0.1"}'},
    }
    assert ToolCall.from_raw(wire) == tc
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
    for bad_raw in (
        "invalid-tool-call",
        tc,
        {"id": "c_flat", "name": "echo", "arguments": '{"text": "hi"}'},
        {"id": "c_map", "type": "function", "function": {"name": "echo", "arguments": {"text": "hi"}}},
    ):
        with pytest.raises(TypeError):
            ToolCall.from_raw(bad_raw)

    extracted = extract_tool_calls({"role": "assistant", "tool_calls": [wire]})
    assert extracted == [tc]
    assert extract_tool_calls({"role": "assistant", "content": "hi"}) == []
    assert extract_tool_calls({"role": "assistant", "tool_calls": None}) == []

    # 验证 iter_react 按标准 wire 字典先产出 ToolStart 再执行工具后产出 ToolStep
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
            {"role": "assistant", "content": None, "tool_calls": [wire]},
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

    # 验证 arguments=None / 空串的 wire tool_calls 写回 messages 与发往下一轮 LLM 时均已归一化为 "{}"
    noop_tools = ToolRegistry(Tool(name="ping", description="探测", parameters={}, handler=lambda: "pong"))
    norm_llm = ScriptedLLM(
        [
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "先探测",
                "tool_calls": [
                    {"id": "wire_1", "type": "function", "function": {"name": "ping", "arguments": '{"host": "127.0.0.1"}'}},
                    {"id": "none_2", "type": "function", "function": {"name": "ping", "arguments": None}},
                ],
            },
            {"role": "assistant", "content": "归一化完成", "tool_calls": []},
        ]
    )
    norm_msgs: list[dict[str, object]] = [{"role": "user", "content": "探测"}]
    assert run_react(norm_llm, norm_msgs, noop_tools) == "归一化完成"
    expected_wire_calls = [
        {"id": "wire_1", "type": "function", "function": {"name": "ping", "arguments": '{"host": "127.0.0.1"}'}},
        {"id": "none_2", "type": "function", "function": {"name": "ping", "arguments": "{}"}},
    ]
    assert norm_msgs[1] == {
        "role": "assistant",
        "content": None,
        "reasoning_content": "先探测",
        "tool_calls": expected_wire_calls,
    }
    assert norm_llm.requests[1][0][1]["tool_calls"] == expected_wire_calls
    assert "tool_calls" not in norm_msgs[-1]


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


# 测试单阶段 convert_to_llm 边界转换不污染原消息列表，且已彻底移除冗余 transform_context 参数
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
    snapshot_before = [dict(m) for m in messages]

    with pytest.raises(TypeError):
        run_react(llm, messages, tools, transform_context=lambda x: x)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        iter_react(llm, messages, tools, transform_context=lambda x: x)  # type: ignore[call-arg]

    # 单一边界转换：即使在回调内原地修改字典或 pop 键，也绝不污染外部主历史列表
    def convert_to_llm(ctx: list[dict[str, object]]) -> list[dict[str, object]]:
        seen_stages.append("convert")
        ctx[0]["injected"] = "dirty"
        for msg in ctx:
            if msg.get("role") == "artifact":
                msg.pop("content", None)
        return [m for m in ctx if m.get("role") in {"system", "user", "assistant", "tool"}]

    reply = run_react(
        llm,
        messages,
        tools,
        convert_to_llm=convert_to_llm,
    )

    assert reply == "管道执行完毕"
    assert seen_stages == ["convert", "convert"]
    assert messages[:3] == snapshot_before
    assert "injected" not in messages[0]
    assert messages[1] == {"role": "artifact", "content": "仅前端渲染的卡片对象"}
    for req_msgs, _ in llm.requests:
        assert all(m.get("role") in {"system", "user", "assistant", "tool"} for m in req_msgs)
    assert [m["role"] for m in llm.requests[0][0]] == ["system", "user"]
    assert [m["role"] for m in llm.requests[1][0]] == ["system", "user", "assistant", "tool"]


# 测试 default_post_json 读取响应体中途发生 IncompleteRead/SSLError/ConnectionError 等瞬时异常时退避重试，且对损坏 JSON 不重试；同步 respond 校验 error 帧与空 choices
def test_sync_post_json_body_read_retry_and_openai_response_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    import http.client
    import ssl
    from urllib.error import URLError
    from personal_assistant.ai import OpenAIChatClient, TokenUsage
    from personal_assistant.ai import client as llm_mod

    monkeypatch.setattr(llm_mod.time, "sleep", lambda _: None)

    # 构造可定制 read() 与退出回调的模拟 HTTP 响应上下文对象
    def _make_resp(read_fn, on_exit=None):
        # 模拟标准库 HTTPResponse 上下文协议
        class _FakeResp:
            # 进入上下文返回自身
            def __enter__(self):
                return self

            # 退出上下文并触发可选清理回调
            def __exit__(self, *args):
                if on_exit is not None:
                    on_exit()
                return False

            # 执行注入的响应体读取行为
            def read(self) -> bytes:
                return read_fn()

        return _FakeResp()

    read_attempts = {"count": 0, "closed": 0}

    # 模拟连接建立成功但 response.read() 前两次分别抛 IncompleteRead 与 SSLError，第三次完整返回 JSON
    def flaky_body_urlopen(request, timeout=60):
        read_attempts["count"] += 1
        attempt = read_attempts["count"]

        # 按调用次数返回瞬时读取异常或最终正常载荷
        def _read() -> bytes:
            if attempt == 1:
                raise http.client.IncompleteRead(b'{"choices":', 12)
            if attempt == 2:
                raise ssl.SSLError("TLS read timed out")
            return b'{"ok": true}'

        return _make_resp(_read, on_exit=lambda: read_attempts.__setitem__("closed", read_attempts["closed"] + 1))

    monkeypatch.setattr(llm_mod, "urlopen", flaky_body_urlopen)
    assert llm_mod.default_post_json("https://example.com/v1", {"a": 1}, {}) == {"ok": True}
    assert read_attempts["count"] == 3
    assert read_attempts["closed"] == 3

    # 验证 read() 抛 ConnectionResetError / TimeoutError / URLError 时同样触发重试并受 max_retries 上限约束
    err_seq = [ConnectionResetError("reset"), TimeoutError("timed out"), URLError("eof")]
    seq_attempts = {"count": 0}

    # 依次抛出不同类型的读取阶段瞬时异常直至重试耗尽
    def exhausted_body_urlopen(request, timeout=60):
        idx = seq_attempts["count"]
        seq_attempts["count"] += 1

        # 抛出当前索引对应的瞬时异常
        def _read_fail() -> bytes:
            raise err_seq[idx]

        return _make_resp(_read_fail)

    monkeypatch.setattr(llm_mod, "urlopen", exhausted_body_urlopen)
    with pytest.raises(URLError):
        llm_mod.default_post_json("https://example.com/v1", {"a": 1}, {})
    assert seq_attempts["count"] == 3

    # 模拟响应体读取成功但为语法损坏的 JSON：不得盲目重试
    bad_json_attempts = {"count": 0}

    # 返回语法错误 JSON 字节流
    def bad_json_urlopen(request, timeout=60):
        bad_json_attempts["count"] += 1
        return _make_resp(lambda: b"{broken-json")

    monkeypatch.setattr(llm_mod, "urlopen", bad_json_urlopen)
    with pytest.raises(json.JSONDecodeError):
        llm_mod.default_post_json("https://example.com/v1", {"a": 1}, {})
    assert bad_json_attempts["count"] == 1

    # 测试 OpenAIChatClient.respond() 在 HTTP 200 error 帧、空 choices、非 dict 响应时抛出清晰 RuntimeError 且清理 last_usage
    scripted_payloads: list[object] = [
        {"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
        {"error": {"message": "rate_limit_exceeded", "code": 429}},
        {"choices": [], "usage": {"prompt_tokens": 99, "completion_tokens": 0}},
        {"choices": [{"message": "not-a-dict"}]},
        ["unexpected-list"],
    ]
    client = OpenAIChatClient(
        api_key="sk-test",
        base_url="https://example.com/v1",
        model="demo-model",
        post_json=lambda _u, _p, _h: scripted_payloads.pop(0),  # type: ignore[return-value]
    )
    assert client.respond([{"role": "user", "content": "hi"}])["content"] == "ok"
    assert client.last_usage == TokenUsage(prompt_tokens=10, cached_tokens=0, completion_tokens=2)

    with pytest.raises(RuntimeError, match="模型接口返回错误.*rate_limit_exceeded"):
        client.respond([{"role": "user", "content": "hi"}])
    assert client.last_usage is None

    with pytest.raises(RuntimeError, match="choices"):
        client.respond([{"role": "user", "content": "hi"}])
    assert client.last_usage is None

    with pytest.raises(RuntimeError, match="message"):
        client.respond([{"role": "user", "content": "hi"}])

    with pytest.raises(RuntimeError, match="JSON 对象|格式"):
        client.respond([{"role": "user", "content": "hi"}])


# 测试复用同一 ToolRegistry 实例跨多轮调用 run_react/iter_react 透传当轮 tool_context 且隔离自定义子类
def test_react_forwards_runtime_tool_context_across_turns() -> None:
    from personal_assistant.agent.tools import build_tool

    # 记录模型参数与执行期注入上下文的函数工具
    def probe_ctx(
        query: str,
        limit: str = "5",
        *,
        turn_id: int | None = None,
        label: str = "base",
    ) -> str:
        return f"q={query}|limit={limit}|turn={turn_id}|label={label}"

    # 显式关闭 accept_context=False 的 **values 工具：绝不能被混入执行期 tool_context
    def isolated_fn(**values: str) -> str:
        assert set(values) == {"msg"}, f"unexpected keys: {values!r}"
        return f"pure:{values['msg']}"

    probe_tool = build_tool(
        probe_ctx,
        description="探针工具",
        parameters={"query": "关键词", "limit": "条数"},
        context={"label": "pinned"},
    )
    sub_tool = build_tool(
        isolated_fn,
        name="pure_sub",
        description="上下文隔离工具",
        parameters={"msg": "消息"},
        accept_context=False,
    )
    registry = ToolRegistry(probe_tool, sub_tool)

    # 第 1 轮：仅传必选参数 query 并注入 turn_id=1（同时包含未声明键与同名参数干扰键）
    llm1 = ScriptedLLM(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _tool_call("c1", "probe_ctx", '{"query": "alpha"}'),
                    _tool_call("c2", "pure_sub", '{"msg": "hi"}'),
                ],
            },
            {"role": "assistant", "content": "轮次1完成"},
        ]
    )
    msgs1: list[dict[str, object]] = [{"role": "user", "content": "r1"}]
    steps1: list[ToolStep] = []
    reply1 = run_react(
        llm1,
        msgs1,
        registry,
        on_event=lambda ev: steps1.append(ev) if isinstance(ev, ToolStep) else None,
        tool_context={"turn_id": 1, "unused_ctx": "skip", "query": "HIJACK"},
    )
    assert reply1 == "轮次1完成"
    assert steps1[0].observation == "q=alpha|limit=5|turn=1|label=pinned"
    assert steps1[1].observation == "pure:hi"

    # 第 2 轮复用同一 registry：切换为 turn_id=2 与新 label，验证不残留上一轮状态
    llm2 = ScriptedLLM(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("c3", "probe_ctx", '{"query": "beta", "limit": 10}')],
            },
            {"role": "assistant", "content": "轮次2完成"},
        ]
    )
    msgs2: list[dict[str, object]] = [{"role": "user", "content": "r2"}]
    steps2: list[ToolStep] = []
    reply2 = run_react(
        llm2,
        msgs2,
        registry,
        on_event=lambda ev: steps2.append(ev) if isinstance(ev, ToolStep) else None,
        tool_context={"turn_id": 2, "label": "override"},
    )
    assert reply2 == "轮次2完成"
    assert steps2[0].observation == "q=beta|limit=10|turn=2|label=override"

    # 第 3 轮直接消费 iter_react 并省略 tool_context，确认回到初始绑定值且不残留 turn_id=2
    llm3 = ScriptedLLM(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("c4", "probe_ctx", '{"query": "gamma"}')],
            },
            {"role": "assistant", "content": "轮次3完成"},
        ]
    )
    msgs3: list[dict[str, object]] = [{"role": "user", "content": "r3"}]
    events3 = list(iter_react(llm3, msgs3, registry))
    step3 = next(ev for ev in events3 if isinstance(ev, ToolStep))
    assert step3.observation == "q=gamma|limit=5|turn=None|label=pinned"

