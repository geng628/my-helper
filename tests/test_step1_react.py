"""Unit tests for Step 1: minimal ReAct loop, ToolRegistry, and neutral ModelInvoker."""
from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from personal_assistant.agent.react import ToolStart, ToolStep, TurnEnd, iter_react, run_react
from personal_assistant.agent.tools import Tool, ToolRegistry
from personal_assistant.ai import (
    ApiProviderRegistry,
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    LLMNotConfigured,
    ModelClient,
    ModelProfile,
    NoopModelClient,
    OpenAICompletionsProvider,
    StreamDone,
    StreamError,
    StreamOptions,
    StreamStart,
    TextBlock,
    TextDelta,
    TextEnd,
    TextStart,
    ThinkingBlock,
    TokenUsage,
    ToolCall,
    ToolCallEnd,
    ToolCallStart,
    ToolResultMessage,
    UserMessage,
    from_env,
)


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


# 测试用确定性脚本模拟模型（实现 ModelInvoker 协议）
class ScriptedLLM:
    """Deterministic fake ModelInvoker that snapshots every Context and replays scripted replies."""

    # 初始化预设响应序列与请求记录列表
    def __init__(self, responses: list[AssistantMessage]) -> None:
        self._responses = list(responses)
        self.requests: list[Context] = []
        self.last_usage: TokenUsage | None = None

    # 记录请求 Context 快照并按统一事件流回放下一条预设 AssistantMessage
    def stream(
        self,
        context: Context,
        options: StreamOptions | None = None,
    ) -> Iterator[AssistantMessageEvent]:
        del options
        self.requests.append(
            Context(
                system_prompt=context.system_prompt,
                messages=tuple(context.messages),
                tools=tuple(context.tools),
            )
        )
        if not self._responses:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        msg = self._responses.pop(0)
        self.last_usage = msg.usage
        yield StreamStart(partial=AssistantMessage())
        for idx, block in enumerate(msg.blocks):
            if isinstance(block, TextBlock):
                yield TextStart(content_index=idx, partial=AssistantMessage())
                yield TextDelta(content_index=idx, text=block.text, partial=msg)
                yield TextEnd(content_index=idx, text=block.text, partial=msg)
            elif isinstance(block, ToolCall):
                yield ToolCallStart(content_index=idx, call_id=block.id, name=block.name, partial=msg)
                yield ToolCallEnd(content_index=idx, tool_call=block, partial=msg)
        reason = "toolUse" if msg.tool_calls else "stop"
        yield StreamDone(reason=reason, message=msg, usage=msg.usage)

    # 同步返回下一条预设响应
    def complete(
        self,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessage:
        for ev in self.stream(context, options):
            if isinstance(ev, StreamDone):
                return ev.message
        raise AssertionError("ScriptedLLM did not emit StreamDone")


# 构造仅含单个或多个工具调用的中立 AssistantMessage
def _tool_msg(*calls: ToolCall, thinking: str = "") -> AssistantMessage:
    blocks: list[ThinkingBlock | ToolCall] = []
    if thinking:
        blocks.append(ThinkingBlock(text=thinking))
    blocks.extend(calls)
    return AssistantMessage(blocks=tuple(blocks), stop_reason="toolUse")


# 测试无工具调用时直接返回文本、TurnEnd 携带完整轨迹并拒绝已废弃旧参数 on_step
def test_direct_answer_without_tool_calls() -> None:
    llm = ScriptedLLM([AssistantMessage.from_text("你好，有什么可以帮你？", stop_reason="stop")])
    ctx = Context(system_prompt="sys", messages=(UserMessage("你好"),))

    with pytest.raises(TypeError):
        run_react(llm, ctx, _demo_tools(), on_step=lambda *_: None)  # type: ignore[call-arg]

    ends: list[TurnEnd] = []
    reply = run_react(
        llm,
        ctx,
        _demo_tools(),
        on_event=lambda ev: ends.append(ev) if isinstance(ev, TurnEnd) else None,
    )

    assert reply == "你好，有什么可以帮你？"
    assert len(llm.requests) == 1
    assert len(llm.requests[0].tools) == 1
    assert ends and ends[0].messages[-1] == AssistantMessage.from_text("你好，有什么可以帮你？", stop_reason="stop")


# 测试单步工具调用 ReAct 循环与 ToolStep 耗时采集及中立 ToolResultMessage 回传
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
            _tool_msg(ToolCall(id="call_1", name="lookup_weather", arguments='{"city": "上海"}')),
            AssistantMessage.from_text("上海今天晴，气温 22°C。", stop_reason="stop"),
        ]
    )
    ctx = Context(messages=(UserMessage("上海今天天气怎么样？"),))

    reply = run_react(
        llm,
        ctx,
        tools,
        on_event=lambda ev: steps.append(ev) if isinstance(ev, ToolStep) else None,
    )

    assert reply == "上海今天晴，气温 22°C。"
    assert [(s.tool, s.arguments, s.observation) for s in steps] == [
        ("lookup_weather", '{"city": "上海"}', "上海：晴，22°C")
    ]
    assert steps[0].duration_ms >= 0.0
    second_msgs = llm.requests[1].messages
    assert second_msgs[-1] == ToolResultMessage(
        tool_call_id="call_1",
        tool_name="lookup_weather",
        content="上海：晴，22°C",
        is_error=False,
    )


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
            _tool_msg(ToolCall(id="c1", name="find_note", arguments='{"topic": "spinlock"}')),
            _tool_msg(ToolCall(id="c2", name="read_file", arguments='{"path": "notes/spinlock.md"}')),
            AssistantMessage.from_text("根据 notes/spinlock.md，持有自旋锁时禁止 sleep。", stop_reason="stop"),
        ]
    )
    ctx = Context(messages=(UserMessage("自旋锁能 sleep 吗？"),))

    reply = run_react(
        llm,
        ctx,
        tools,
        on_event=lambda ev: steps.append(ev) if isinstance(ev, ToolStep) else None,
    )

    assert "禁止 sleep" in reply
    assert [s.tool for s in steps] == ["find_note", "read_file"]
    assert len(llm.requests) == 3


# 测试工具错误作为观察结果回传（is_error=True）并自愈
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
            _tool_msg(
                ToolCall(id="bad_tool", name="nonexistent", arguments="{}"),
                ToolCall(id="bad_json", name="add_numbers", arguments="{not-json"),
            ),
            _tool_msg(ToolCall(id="good", name="add_numbers", arguments='{"a": "19", "b": "23"}')),
            AssistantMessage.from_text("19 + 23 = 42。", stop_reason="stop"),
        ]
    )
    ctx = Context(messages=(UserMessage("算一下 19 加 23"),))

    reply = run_react(
        llm,
        ctx,
        tools,
        on_event=lambda ev: steps.append(ev) if isinstance(ev, ToolStep) else None,
    )

    assert reply == "19 + 23 = 42。"
    assert "未知工具" in steps[0].observation
    assert "JSON 解析失败" in steps[1].observation
    assert steps[2].observation == "42"
    err_results = [m for m in llm.requests[1].messages if isinstance(m, ToolResultMessage)]
    assert [r.is_error for r in err_results] == [True, True]


# 测试纯 ReAct 循环移除轮次上限，仅在无 toolUse 时自然终止且全程保持 tools 前缀恒定
def test_unbounded_react_loop_stops_only_when_no_tool_calls(tmp_path) -> None:
    import personal_assistant.agent as agent_pkg
    import personal_assistant.agent.react as react_mod
    from evals.agent_eval import evaluate_agent
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
    expected_schemas = tools.schemas()
    llm = ScriptedLLM(
        [_tool_msg(ToolCall(id=f"c{i}", name="ping", arguments="{}")) for i in range(1, 7)]
        + [AssistantMessage.from_text("已完成 6 轮连续探测。", stop_reason="stop")]
    )
    ctx = Context(messages=(UserMessage("连续探测直到完成"),))

    with pytest.raises(TypeError):
        iter_react(llm, ctx, tools, max_rounds=4)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        run_react(llm, ctx, tools, max_rounds=4)  # type: ignore[call-arg]

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

    ends: list[TurnEnd] = []
    reply = run_react(
        llm,
        ctx,
        tools,
        on_event=lambda ev: (
            steps.append(ev)
            if isinstance(ev, ToolStep)
            else (ends.append(ev) if isinstance(ev, TurnEnd) else None)
        ),
    )

    assert reply == "已完成 6 轮连续探测。"
    assert len(steps) == 6
    assert len(llm.requests) == 7
    assert all(req.tools == expected_schemas for req in llm.requests)
    tool_msgs = [m for m in ends[0].messages if isinstance(m, ToolResultMessage)]
    assert len(tool_msgs) == 6
    assert all(m.content == "pong" for m in tool_msgs)


# 测试从环境变量构造句柄、OpenAI 翻译器载荷格式与 StreamError 在 iter_react/run_react 中的优雅收束
def test_model_invoker_from_env_payload_and_stream_error_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODE", "off")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    client = from_env()
    assert isinstance(client, NoopModelClient)
    with pytest.raises(LLMNotConfigured):
        client.complete(Context(messages=(UserMessage("hi"),)))

    captured: dict[str, object] = {}

    # 模拟捕获请求参数的流式发送桩
    def fake_post_stream(url: str, payload: dict[str, object], headers: dict[str, str], *, abort_signal=None) -> Iterator[str]:
        del abort_signal
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        return iter(
            [
                'data: {"choices":[{"delta":{"role":"assistant","content":"ok"}}]}\n',
                "data: [DONE]\n",
            ]
        )

    reg = ApiProviderRegistry()
    reg.register(OpenAICompletionsProvider(post_stream=fake_post_stream))
    real_client = ModelClient(
        profile=ModelProfile(
            name="demo",
            api="openai-completions",
            base_url="https://example.com/v1/",
            model="demo-model",
            api_key="sk-test",
        ),
        registry=reg,
    )
    msg = real_client.complete(Context(messages=(UserMessage("hi"),), tools=_demo_tools().schemas()))
    assert msg.text == "ok"
    assert captured["url"] == "https://example.com/v1/chat/completions"
    assert captured["headers"] == {"Authorization": "Bearer sk-test"}
    sent_payload = captured["payload"]
    assert isinstance(sent_payload, dict)
    assert sent_payload["model"] == "demo-model"
    assert sent_payload["tool_choice"] == "auto"
    assert json.loads(json.dumps(sent_payload["tools"]))[0]["function"]["name"] == "get_current_time"

    # 验证 iter_react 遇 StreamError 透传事件后立即终止且不发 TurnEnd；run_react 遇 StreamError 抛 RuntimeError
    class _ErrorLLM:
        last_usage: TokenUsage | None = None

        # 直接产出 StreamStart 与 StreamError
        def stream(self, context: Context, options: StreamOptions | None = None) -> Iterator[AssistantMessageEvent]:
            del context, options
            err_msg = AssistantMessage(stop_reason="error", error_message="rate limit 429")
            yield StreamStart(partial=AssistantMessage())
            yield StreamError(reason="error", message=err_msg, error_message="rate limit 429")

        # 同步接口抛出对应错误
        def complete(self, context: Context, options: StreamOptions | None = None) -> AssistantMessage:
            raise RuntimeError("rate limit 429")

    err_events = list(iter_react(_ErrorLLM(), Context(messages=(UserMessage("hi"),)), ToolRegistry()))
    assert [type(ev) for ev in err_events] == [StreamStart, StreamError]
    with pytest.raises(RuntimeError, match="rate limit 429"):
        run_react(_ErrorLLM(), Context(messages=(UserMessage("hi"),)), ToolRegistry())


# 测试 default_post_json 对瞬时 TLS/网络抖动自动重试，但对 HTTP 400 客户端错误绝不重试
def test_default_post_json_retries_transient_error_but_not_http_400(monkeypatch) -> None:
    import io
    import ssl
    from urllib.error import HTTPError, URLError

    from personal_assistant.ai.providers import http as http_mod

    monkeypatch.setattr(http_mod.time, "sleep", lambda _: None)

    attempts = {"count": 0}

    # 模拟前 2 次遇到 TLS EOF，第 3 次成功返回 JSON
    def flaky_urlopen(request, timeout=60, abort_signal=None):
        del request, timeout, abort_signal
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

    monkeypatch.setattr(http_mod, "open_http_request", flaky_urlopen)
    assert http_mod.default_post_json("https://example.com/v1", {"a": 1}, {}) == {"ok": True}
    assert attempts["count"] == 3

    bad_attempts = {"count": 0}

    # 模拟 400 Bad Request：必须立刻抛出，不重试
    def bad_request_urlopen(request, timeout=60, abort_signal=None):
        del request, timeout, abort_signal
        bad_attempts["count"] += 1
        raise HTTPError("https://example.com/v1", 400, "Bad Request", {}, io.BytesIO(b"bad"))

    monkeypatch.setattr(http_mod, "open_http_request", bad_request_urlopen)
    with pytest.raises(HTTPError):
        http_mod.default_post_json("https://example.com/v1", {"a": 1}, {})
    assert bad_attempts["count"] == 1


# 测试 ToolStart -> 执行 -> ToolStep 顺序与纯中立 AssistantMessage 空回复/思维链保留
def test_tool_execution_order_and_neutral_message_preservation() -> None:
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
            _tool_msg(ToolCall(id="call_a", name="ping", arguments='{"host": "127.0.0.1"}'), thinking="先探测"),
            AssistantMessage(
                blocks=(ThinkingBlock(text="推演细节"), TextBlock(text="  完成探测  ")),
                stop_reason="stop",
            ),
        ]
    )
    events: list[object] = []
    for ev in iter_react(llm, Context(messages=(UserMessage("ping"),)), tools):
        if isinstance(ev, ToolStart):
            exec_order.append(f"start:{ev.call_id}")
        elif isinstance(ev, ToolStep):
            exec_order.append(f"step:{ev.call_id}")
        events.append(ev)

    assert exec_order == ["start:call_a", "exec:127.0.0.1", "step:call_a"]
    assert ToolStart(tool="ping", arguments='{"host": "127.0.0.1"}', call_id="call_a") in events
    end = next(ev for ev in events if isinstance(ev, TurnEnd))
    assert end.reply == "完成探测"
    assert isinstance(end.messages[-1], AssistantMessage)
    assert end.messages[-1].thinking == "推演细节"

    # 空文本退出不伪造中文兜底
    empty_llm = ScriptedLLM([AssistantMessage(blocks=(ThinkingBlock(text="空回复前内部推理"),), stop_reason="stop")])
    assert run_react(empty_llm, Context(messages=(UserMessage("hi"),)), ToolRegistry()) == ""


# 测试单阶段 convert_to_llm 边界转换面向不可变 Context 工作，且已彻底移除冗余 transform_context 参数
def test_convert_to_llm_context_transformer_immutability() -> None:
    seen_stages: list[str] = []
    tools = ToolRegistry(
        Tool(name="echo", description="回显", parameters={"v": "值"}, handler=lambda v: f"ok:{v}")
    )
    llm = ScriptedLLM(
        [
            _tool_msg(ToolCall(id="c1", name="echo", arguments='{"v": "1"}')),
            AssistantMessage.from_text("管道执行完毕", stop_reason="stop"),
        ]
    )
    initial_ctx = Context(
        system_prompt="sys",
        messages=(UserMessage("请调用 echo"),),
    )

    with pytest.raises(TypeError):
        run_react(llm, initial_ctx, tools, transform_context=lambda x: x)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        iter_react(llm, initial_ctx, tools, transform_context=lambda x: x)  # type: ignore[call-arg]

    # 上下文边界转换：改写 system_prompt 副本但绝不污染原始 initial_ctx
    def convert_to_llm(ctx: Context) -> Context:
        seen_stages.append("convert")
        return Context(
            system_prompt=f"{ctx.system_prompt} [transformed]",
            messages=ctx.messages,
            tools=ctx.tools,
        )

    reply = run_react(llm, initial_ctx, tools, convert_to_llm=convert_to_llm)

    assert reply == "管道执行完毕"
    assert seen_stages == ["convert", "convert"]
    assert initial_ctx.system_prompt == "sys"
    assert len(initial_ctx.messages) == 1
    assert all(req.system_prompt == "sys [transformed]" for req in llm.requests)
    assert len(llm.requests[0].messages) == 1
    assert len(llm.requests[1].messages) == 3


# 测试 default_post_json 读取响应体中途发生瞬时异常时退避重试且对损坏 JSON 不重试
def test_sync_post_json_body_read_retry_and_broken_json(monkeypatch: pytest.MonkeyPatch) -> None:
    import http.client
    import ssl
    from urllib.error import URLError
    from personal_assistant.ai.providers import http as http_mod

    monkeypatch.setattr(http_mod.time, "sleep", lambda _: None)

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
    def flaky_body_urlopen(request, timeout=60, abort_signal=None):
        del request, timeout, abort_signal
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

    monkeypatch.setattr(http_mod, "open_http_request", flaky_body_urlopen)
    assert http_mod.default_post_json("https://example.com/v1", {"a": 1}, {}) == {"ok": True}
    assert read_attempts["count"] == 3
    assert read_attempts["closed"] == 3

    err_seq = [ConnectionResetError("reset"), TimeoutError("timed out"), URLError("eof")]
    seq_attempts = {"count": 0}

    # 依次抛出不同类型的读取阶段瞬时异常直至重试耗尽
    def exhausted_body_urlopen(request, timeout=60, abort_signal=None):
        del request, timeout, abort_signal
        idx = seq_attempts["count"]
        seq_attempts["count"] += 1

        # 抛出当前索引对应的瞬时异常
        def _read_fail() -> bytes:
            raise err_seq[idx]

        return _make_resp(_read_fail)

    monkeypatch.setattr(http_mod, "open_http_request", exhausted_body_urlopen)
    with pytest.raises(URLError):
        http_mod.default_post_json("https://example.com/v1", {"a": 1}, {})
    assert seq_attempts["count"] == 3

    bad_json_attempts = {"count": 0}

    # 返回语法错误 JSON 字节流：不得盲目重试
    def bad_json_urlopen(request, timeout=60, abort_signal=None):
        del request, timeout, abort_signal
        bad_json_attempts["count"] += 1
        return _make_resp(lambda: b"{broken-json")

    monkeypatch.setattr(http_mod, "open_http_request", bad_json_urlopen)
    with pytest.raises(json.JSONDecodeError):
        http_mod.default_post_json("https://example.com/v1", {"a": 1}, {})
    assert bad_json_attempts["count"] == 1


# 测试复用同一 ToolRegistry 实例跨多轮调用 run_react/iter_react 透传当轮 tool_context 且隔离 accept_context=False 工具
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
            _tool_msg(
                ToolCall(id="c1", name="probe_ctx", arguments='{"query": "alpha"}'),
                ToolCall(id="c2", name="pure_sub", arguments='{"msg": "hi"}'),
            ),
            AssistantMessage.from_text("轮次1完成", stop_reason="stop"),
        ]
    )
    steps1: list[ToolStep] = []
    reply1 = run_react(
        llm1,
        Context(messages=(UserMessage("r1"),)),
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
            _tool_msg(ToolCall(id="c3", name="probe_ctx", arguments='{"query": "beta", "limit": 10}')),
            AssistantMessage.from_text("轮次2完成", stop_reason="stop"),
        ]
    )
    steps2: list[ToolStep] = []
    reply2 = run_react(
        llm2,
        Context(messages=(UserMessage("r2"),)),
        registry,
        on_event=lambda ev: steps2.append(ev) if isinstance(ev, ToolStep) else None,
        tool_context={"turn_id": 2, "label": "override"},
    )
    assert reply2 == "轮次2完成"
    assert steps2[0].observation == "q=beta|limit=10|turn=2|label=override"

    # 第 3 轮直接消费 iter_react 并省略 tool_context，确认回到初始绑定值且不残留 turn_id=2
    llm3 = ScriptedLLM(
        [
            _tool_msg(ToolCall(id="c4", name="probe_ctx", arguments='{"query": "gamma"}')),
            AssistantMessage.from_text("轮次3完成", stop_reason="stop"),
        ]
    )
    events3 = list(iter_react(llm3, Context(messages=(UserMessage("r3"),)), registry))
    step3 = next(ev for ev in events3 if isinstance(ev, ToolStep))
    assert step3.observation == "q=gamma|limit=5|turn=None|label=pinned"
