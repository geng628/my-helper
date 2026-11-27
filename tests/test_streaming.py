"""TDD 单测：微内核 SSE 流式管道、AgentEvent 透传、ObserverProtocol 插槽隔离与 Web SSE 端点。"""
from __future__ import annotations

import io
import json
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from personal_assistant.agent import (
    ObserverProtocol,
    TaskEvent,
    Tool,
    ToolRegistry,
    ToolStart,
    ToolStep,
    TurnEnd,
    iter_react,
)
from personal_assistant.ai import (
    CacheMonitor,
    LLMClient,
    LLMStreamEvent,
    NoopLLMClient,
    OpenAICompatibleLLM,
    StreamDone,
    TextDelta,
    ThinkingDelta,
    TokenUsage,
)
from personal_assistant.assistant.app import AssistantApp
from personal_assistant.assistant.consolidator import ConsolidationPolicy


# 通用脚本化测试模型：支持逐轮流式事件列表、静态响应字典与动态回调，避免在各用例重复造轮子
class ScriptedStreamLLM(LLMClient):
    # 保存逐轮剧本或动态生成回调
    def __init__(self, turns: list[list[LLMStreamEvent] | dict[str, object]] | Callable[..., dict[str, object]]) -> None:
        self._turns = turns if callable(turns) else list(turns)

    # 弹出或动态生成同步响应消息
    def respond(self, messages: list[dict[str, object]], tools: list[dict[str, object]] | None = None) -> dict[str, object]:
        if callable(self._turns):
            return self._turns(self, messages, tools)
        item = self._turns.pop(0)
        assert isinstance(item, dict)
        return item

    # 回放逐轮流式事件序列；非列表剧本回退走基类 respond() 包装
    def stream(self, messages: list[dict[str, object]], tools: list[dict[str, object]] | None = None) -> Iterator[LLMStreamEvent]:
        if not callable(self._turns) and self._turns and isinstance(self._turns[0], list):
            yield from self._turns.pop(0)
            return
        yield from super().stream(messages, tools)


# 测试 OpenAICompatibleLLM.stream 解析多帧 SSE 文本、计算 ttft/latency 并由 TrackingLLMClient 单次记账
def test_openai_sse_text_streaming_and_tracking_client() -> None:
    sse_lines = [
        ": keep-alive\n",
        "data:\n",
        'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n',
        'data: {"choices":[{"index":0,"delta":{"content":"你好"}}]}\n',
        'data: {"choices":[{"index":0,"delta":{"content":"，世界！"}}]}\n',
        'data: {"choices":[],"usage":{"prompt_tokens":120,"prompt_cache_hit_tokens":64,"completion_tokens":6}}\n',
        "data: [DONE]\n",
    ]
    captured_payloads: list[dict[str, object]] = []

    # 返回预设 SSE 行序列并记录请求载荷的测试桩
    def fake_post_stream(url: str, payload: dict[str, object], headers: dict[str, str]) -> Iterator[str]:
        assert url == "https://api.example.com/v1/chat/completions"
        assert headers["Authorization"] == "Bearer sk-stream"
        captured_payloads.append(dict(payload))
        yield from sse_lines

    raw = OpenAICompatibleLLM(api_key="sk-stream", base_url="https://api.example.com/v1/", model="deepseek-chat", post_stream=fake_post_stream)
    monitor = CacheMonitor()
    tracked = monitor.wrap(raw, session_key="s-stream")
    events = list(tracked.stream([{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]))

    assert [ev.text for ev in events if isinstance(ev, TextDelta)] == ["你好", "，世界！"]
    done = next(ev for ev in events if isinstance(ev, StreamDone))
    assert done.message == {"role": "assistant", "content": "你好，世界！"}
    assert done.usage == TokenUsage(prompt_tokens=120, cached_tokens=64, completion_tokens=6)
    assert done.ttft_ms is not None and 0.0 <= done.ttft_ms <= done.latency_ms
    assert tracked.last_usage == raw.last_usage == done.usage
    assert captured_payloads[0]["stream_options"] == {"include_usage": True}
    assert monitor.summary()["cached_tokens"] == 64


# 测试 SSE 分片增量 tool_calls 拼装、stream_options 400 自动降级，以及流内 error JSON 抛错
def test_openai_sse_tool_calls_400_fallback_and_midstream_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from personal_assistant.ai import client as llm_mod

    call_payloads: list[dict[str, object]] = []

    # 模拟首请求含 stream_options 抛 400、去掉后流式返回多索引分片 tool_calls
    def fake_urlopen(request: Request, timeout: int = 60):
        body = json.loads((request.data or b"{}").decode("utf-8"))
        call_payloads.append(body)
        if "stream_options" in body:
            raise HTTPError(
                request.full_url,
                400,
                "Bad Request",
                {},
                io.BytesIO(b'{"error":"stream_options unsupported"}'),
            )
        raw_sse = (
            'data: {"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":['
            '{"index":0,"id":"c_1","type":"function","function":{"name":"add_","arguments":"{\\"a\\":"}},'
            '{"index":1,"id":"c_2","type":"function","function":{"name":"ping","arguments":""}}]}}]}\n'
            'data: {"choices":[{"index":0,"delta":{"tool_calls":['
            '{"index":0,"function":{"name":"num","arguments":" \\"1\\"}"}},'
            '{"index":1,"function":{"arguments":"{}"}}]}}]}\n'
            "data: [DONE]\n"
        ).encode("utf-8")

        # 包装内存字节流为上下文管理器，模拟 HTTP 响应行迭代
        class _StreamResp(io.BytesIO):
            # 返回自身进入上下文
            def __enter__(self):
                return self

            # 关闭流退出上下文
            def __exit__(self, *args):
                self.close()
                return False

        return _StreamResp(raw_sse)

    monkeypatch.setattr(llm_mod, "urlopen", fake_urlopen)
    client = OpenAICompatibleLLM(api_key="k", base_url="https://api.example.com/v1", model="m")
    events = list(client.stream([{"role": "user", "content": "calc"}], tools=[{"type": "function", "function": {"name": "add_num", "parameters": {}}}]))

    assert len(call_payloads) == 2 and "stream_options" in call_payloads[0] and "stream_options" not in call_payloads[1]
    done = next(ev for ev in events if isinstance(ev, StreamDone))
    assert done.message == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c_1", "type": "function", "function": {"name": "add_num", "arguments": '{"a": "1"}'}},
            {"id": "c_2", "type": "function", "function": {"name": "ping", "arguments": "{}"}},
        ],
    }

    err_client = OpenAICompatibleLLM(
        api_key="k",
        base_url="https://api.example.com/v1",
        model="m",
        post_stream=lambda *_: iter(['data: {"choices":[{"index":0,"delta":{"content":"一半"}}]}\n', 'data: {"error":{"message":"rate_limit"}}\n']),
    )
    gen = err_client.stream([{"role": "user", "content": "hi"}])
    assert next(gen) == TextDelta(text="一半")
    with pytest.raises(RuntimeError, match="rate_limit"):
        list(gen)


# 测试 iter_react 直接透传 ThinkingDelta / TextDelta / StreamDone 并按序产出 ToolStart -> ToolStep -> TurnEnd
def test_iter_react_emits_slim_typed_events() -> None:
    observed_before_exec: list[object] = []
    streamed_events: list[object] = []

    # 在工具执行瞬间快照已产出事件，验证 ToolStart 必先于工具执行、ToolStep 必在之后
    def double_handler(n: str) -> str:
        observed_before_exec.extend(streamed_events)
        return str(int(n) * 2)

    llm = ScriptedStreamLLM(
        [
            [
                ThinkingDelta(text="先调用翻倍工具"),
                StreamDone(
                    message={
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "先调用翻倍工具",
                        "tool_calls": [
                            {
                                "id": "t1",
                                "type": "function",
                                "function": {"name": "double", "arguments": '{"n": "21"}'},
                            }
                        ],
                    },
                    usage=TokenUsage(prompt_tokens=50, cached_tokens=20, completion_tokens=10),
                    ttft_ms=12.0,
                    latency_ms=30.0,
                ),
            ],
            [
                ThinkingDelta(text="整理答案"),
                TextDelta(text="结果"),
                TextDelta(text="是 42"),
                StreamDone(
                    message={
                        "role": "assistant",
                        "content": "结果是 42",
                        "reasoning_content": "整理答案",
                    },
                    usage=TokenUsage(prompt_tokens=80, cached_tokens=50, completion_tokens=6),
                    ttft_ms=8.5,
                    latency_ms=25.0,
                ),
            ],
        ]
    )
    registry = ToolRegistry(Tool(name="double", description="翻倍", parameters={"n": "整数"}, handler=double_handler))
    msgs: list[dict[str, object]] = [{"role": "user", "content": "21 翻倍"}]
    for ev in iter_react(llm, msgs, registry):
        streamed_events.append(ev)

    assert isinstance(observed_before_exec[-1], ToolStart)
    assert not any(isinstance(ev, ToolStep) for ev in observed_before_exec)
    assert [type(ev) for ev in streamed_events] == [
        ThinkingDelta,
        StreamDone,
        ToolStart,
        ToolStep,
        ThinkingDelta,
        TextDelta,
        TextDelta,
        StreamDone,
        TurnEnd,
    ]
    assert streamed_events[0] == ThinkingDelta(text="先调用翻倍工具")
    assert streamed_events[2] == ToolStart(tool="double", arguments='{"n": "21"}', call_id="t1")
    step = streamed_events[3]
    assert isinstance(step, ToolStep)
    assert step.tool == "double" and step.observation == "42" and step.call_id == "t1" and step.duration_ms >= 0.0
    assert [ev.text for ev in streamed_events if isinstance(ev, TextDelta)] == ["结果", "是 42"]
    end = streamed_events[-1]
    assert isinstance(end, TurnEnd) and end.reply == "结果是 42" and end.total_ms >= 0.0

    # 验证同步基类 LLMClient.stream() 保持纯净（不跨层嗅探 reasoning_content），思考增量由流式实现直接产出
    sync_llm = ScriptedStreamLLM(
        [
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "跨层字段不应由基类嗅探",
                "tool_calls": [{"id": "c0", "type": "function", "function": {"name": "double", "arguments": '{"n":"1"}'}}],
            },
            {
                "role": "assistant",
                "content": "同步回答",
                "reasoning_content": "跨层字段不应由基类嗅探",
            },
            [
                ThinkingDelta(text="流式思考"),
                TextDelta(text="流式回答"),
                StreamDone(message={"role": "assistant", "content": "流式回答", "reasoning_content": "流式思考"}),
            ],
        ]
    )
    sync_ev1 = list(sync_llm.stream([{"role": "user", "content": "a"}]))
    assert [type(ev) for ev in sync_ev1] == [StreamDone]
    sync_ev2 = list(sync_llm.stream([{"role": "user", "content": "b"}]))
    assert [type(ev) for ev in sync_ev2] == [TextDelta, StreamDone]
    assert sync_ev2[0] == TextDelta(text="同步回答")
    stream_ev3 = list(sync_llm.stream([{"role": "user", "content": "c"}]))
    assert [type(ev) for ev in stream_ev3] == [ThinkingDelta, TextDelta, StreamDone]
    assert stream_ev3[0] == ThinkingDelta(text="流式思考")
    assert stream_ev3[1] == TextDelta(text="流式回答")


# 测试 TelemetryObserver 与 AssistantApp 忽略 ThinkingDelta / ToolStart 等新事件不崩且统计精确
def test_telemetry_and_app_handle_thinking_delta_and_tool_start(tmp_path: Path) -> None:
    from plugins.telemetry import TelemetryObserver

    obs = TelemetryObserver()
    llm = ScriptedStreamLLM(
        [
            [
                ThinkingDelta(text="先看当前时间"),
                StreamDone(
                    message={
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "先看当前时间",
                        "tool_calls": [
                            {"id": "tc_1", "type": "function", "function": {"name": "get_current_time", "arguments": "{}"}}
                        ],
                    },
                    usage=TokenUsage(prompt_tokens=60, cached_tokens=20, completion_tokens=10),
                    ttft_ms=11.0,
                    latency_ms=22.0,
                ),
            ],
            [
                ThinkingDelta(text="整理答复"),
                TextDelta(text="你好！"),
                StreamDone(
                    message={"role": "assistant", "content": "你好！", "reasoning_content": "整理答复"},
                    usage=TokenUsage(prompt_tokens=90, cached_tokens=60, completion_tokens=5),
                    ttft_ms=7.0,
                    latency_ms=18.0,
                ),
            ],
        ]
    )
    seen_events: list[object] = []
    app = AssistantApp(data_root=tmp_path / "data", llm=llm, observers=[obs])
    try:
        frames = list(app.stream_chat("现在几点", chat_id="s-think", on_event=seen_events.append))
    finally:
        app.close()

    assert any(isinstance(ev, ThinkingDelta) for ev in seen_events)
    assert any(isinstance(ev, ToolStart) and ev.tool == "get_current_time" and ev.call_id == "tc_1" for ev in seen_events)
    assert [f["type"] for f in frames] == [
        "thinking_delta",
        "tool_start",
        "tool_step",
        "thinking_delta",
        "text_delta",
        "done",
    ]
    assert frames[0] == {"type": "thinking_delta", "text": "先看当前时间"}
    assert frames[1] == {
        "type": "tool_start",
        "tool": "get_current_time",
        "arguments": "{}",
        "call_id": "tc_1",
    }
    assert frames[2]["tool"] == "get_current_time" and frames[2]["call_id"] == "tc_1"
    assert frames[3] == {"type": "thinking_delta", "text": "整理答复"}
    assert frames[4] == {"type": "text_delta", "text": "你好！"}
    done_frame = frames[-1]
    assert done_frame["reply"] == "你好！"
    assert done_frame["thinking"] == "先看当前时间整理答复"
    assert done_frame["tool_calls"][0]["call_id"] == "tc_1"
    last_turn = done_frame["telemetry"]["telemetry"]["last_turn"]  # type: ignore[index]
    assert last_turn["llm_calls"] == 2
    assert last_turn["tool_calls"] == 1
    assert last_turn["tools"] == ["get_current_time"]
    assert last_turn["ttft_ms"] == 11.0
    assert last_turn["llm_ms"] == 40.0
    assert last_turn["prompt_tokens"] == 150
    assert last_turn["cached_tokens"] == 80


# 测试 AssistantApp.stream_chat、后台沉淀 TaskEvent 广播及坏观测插件的故障隔离
def test_assistant_app_stream_chat_and_observer_fault_isolation(tmp_path: Path) -> None:
    # 记录按 session_key 分发的全部事件
    class RecordingObserver:
        name = "recorder"

        # 初始化事件缓冲列表
        def __init__(self) -> None:
            self.events: list[tuple[str, object]] = []

        # 记录会话标识与事件对象
        def on_event(self, session_key: str, event: object) -> None:
            self.events.append((session_key, event))

        # 返回只读观测快照
        def snapshot(self, *, session_key: str | None = None) -> dict[str, object]:
            return {"count": len(self.events), "session_key": session_key}

    # 始终抛异常的损坏观测器（验证故障隔离）
    class BrokenObserver:
        name = "broken"

        # 回调抛错
        def on_event(self, session_key: str, event: object) -> None:
            raise RuntimeError("observer crashed")

        # 读取快照抛错
        def snapshot(self, *, session_key: str | None = None) -> dict[str, object]:
            raise RuntimeError("snapshot crashed")

    # 区分主对话与后台沉淀调用返回预设回复
    def dynamic_reply(self: LLMClient, messages: list[dict[str, object]], _tools: object) -> dict[str, object]:
        if "session_summary" in str(messages[0].get("content") or ""):
            self.last_usage = TokenUsage(prompt_tokens=90, cached_tokens=30, completion_tokens=20)
            return {"role": "assistant", "content": json.dumps({"session_summary": "已聊问候", "memories": []}, ensure_ascii=False)}
        self.last_usage = TokenUsage(prompt_tokens=40, cached_tokens=0, completion_tokens=5)
        return {"role": "assistant", "content": "收到啦！"}

    rec, broken = RecordingObserver(), BrokenObserver()
    assert isinstance(rec, ObserverProtocol) and isinstance(broken, ObserverProtocol)
    policy = ConsolidationPolicy(max_active_messages=2, max_active_chars=10000, keep_recent_messages=0, min_batch_messages=2)
    app = AssistantApp(data_root=tmp_path / "data", llm=ScriptedStreamLLM(dynamic_reply), consolidation_policy=policy, observers=[broken, rec])
    try:
        frames = list(app.stream_chat("你好", chat_id="c1"))
        assert [(f["type"], f.get("text") or f.get("reply")) for f in frames] == [("text_delta", "收到啦！"), ("done", "收到啦！")]
        app.wait_for_consolidation(timeout=5.0)
        bg = [(sk, ev) for sk, ev in rec.events if isinstance(ev, TaskEvent)]
        assert len(bg) == 1 and bg[0][0] == "c1" and bg[0][1].kind == "consolidation"
        assert bg[0][1].usage == TokenUsage(prompt_tokens=90, cached_tokens=30, completion_tokens=20)
        st = app.status()
        assert st["observers"] == ["broken", "recorder"] and st["telemetry"]["recorder"]["count"] == len(rec.events)
    finally:
        app.close()


# 测试 POST /api/chat/stream 的 400 空消息、503 未配置模型与 200 实时 SSE 端到端链路
def test_web_chat_sse_stream_endpoint_end_to_end(tmp_path: Path) -> None:
    from scripts.web_chat import create_web_server

    # 分别验证已配置流式对话（200/400）与未配置模型（503）
    llm = ScriptedStreamLLM(
        [
            [StreamDone(message={"role": "assistant", "content": None, "tool_calls": [{"id": "w1", "type": "function", "function": {"name": "list_files", "arguments": '{"root": "data/memory"}'}}]}, latency_ms=5.0)],
            [TextDelta(text="目录"), TextDelta(text="正常"), StreamDone(message={"role": "assistant", "content": "目录正常"}, ttft_ms=2.0, latency_ms=6.0)],
        ]
    )
    for idx, (test_llm, body, expected_status) in enumerate(
        [(llm, b'{"message": "   "}', 400), (NoopLLMClient(), b'{"message": "hi"}', 503)]
    ):
        err_app = AssistantApp(data_root=tmp_path / f"err_{idx}", llm=test_llm)
        srv = create_web_server(err_app, host="127.0.0.1", port=0)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            with pytest.raises(HTTPError) as exc:
                urlopen(Request(f"http://127.0.0.1:{srv.server_port}/api/chat/stream", data=body, headers={"Content-Type": "application/json"}, method="POST"), timeout=5)
            assert exc.value.code == expected_status
        finally:
            srv.shutdown()
            srv.server_close()
            err_app.close()

    app = AssistantApp(data_root=tmp_path / "web_data", llm=llm)
    srv = create_web_server(app, host="127.0.0.1", port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        req = Request(f"http://127.0.0.1:{srv.server_port}/api/chat/stream", data=json.dumps({"message": "检查目录", "chat_id": "web1"}).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=10) as resp:
            assert resp.status == 200 and "text/event-stream" in (resp.headers.get("Content-Type") or "")
            evs = [json.loads(ln.decode("utf-8").strip()[5:].strip()) for ln in resp if ln.decode("utf-8").strip().startswith("data:")]
        assert [ev["type"] for ev in evs] == ["tool_start", "tool_step", "text_delta", "text_delta", "done"]
        assert evs[0] == {
            "type": "tool_start",
            "tool": "list_files",
            "arguments": '{"root": "data/memory"}',
            "call_id": "w1",
        }
        assert evs[1]["tool"] == "list_files" and evs[1]["call_id"] == "w1" and float(evs[1]["duration_ms"]) >= 0.0
        assert [ev["text"] for ev in evs[2:4]] == ["目录", "正常"]
        assert evs[4]["reply"] == "目录正常" and evs[4]["thinking"] == ""
    finally:
        srv.shutdown()
        srv.server_close()
        app.close()


# 测试 OpenAIChatClient 通用协议底座、声明式 _thinking_delta_fields 扩展与底层流/线程语义
def test_openai_chat_client_base_and_delta_field_extension() -> None:
    from dataclasses import dataclass
    from personal_assistant import agent as agent_pkg
    from personal_assistant import ai as ai_pkg
    from personal_assistant.ai import OpenAIChatClient

    assert issubclass(OpenAICompatibleLLM, OpenAIChatClient)
    assert OpenAICompatibleLLM.stream is OpenAIChatClient.stream
    assert not hasattr(ai_pkg, "open_openai_stream")
    assert not hasattr(ai_pkg, "consume_openai_sse")
    assert not hasattr(OpenAIChatClient, "_stream_delta_fields")
    assert not hasattr(agent_pkg, "ThinkingDelta")
    assert not hasattr(agent_pkg, "TextDelta")
    assert not hasattr(agent_pkg, "StreamDone")

    # 声明式扩展子类：收集自定义思考 delta 字段实时吐出 ThinkingDelta，并统一规范化同步与流式响应
    @dataclass
    class _ThinkingDeltaClient(OpenAIChatClient):
        # 声明需从 SSE delta 中实时吐出 ThinkingDelta 并聚合的字段
        def _thinking_delta_fields(self) -> tuple[str, ...]:
            return ("extra",)

        # 同步与流式共用响应规范化钩子
        def _normalize_response_message(self, message: dict[str, object]) -> dict[str, object]:
            if isinstance(message.get("extra"), str):
                message["extra"] = f"[{message['extra']}]"
            return message

    sse_lines = [
        'data: {"choices":[{"delta":{"role":"assistant","extra":"首段"}}]}\n',
        'data: {"choices":[{"delta":{"extra":"续段","content":"正文"}}]}\n',
        "data: [DONE]\n",
    ]
    ext_client = _ThinkingDeltaClient(
        api_key="sk-ext",
        base_url="https://api.example.com/v1",
        model="m-ext",
        post_stream=lambda _u, _p, _h: iter(sse_lines),
        post_json=lambda _u, _p, _h: {"choices": [{"message": {"role": "assistant", "content": "同步", "extra": "元"}}]},
    )
    events = list(ext_client.stream([{"role": "user", "content": "hi"}]))
    assert [e.text for e in events if isinstance(e, ThinkingDelta)] == ["首段", "续段"]
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["正文"]
    done = next(e for e in events if isinstance(e, StreamDone))
    assert done.ttft_ms is not None and done.ttft_ms >= 0.0
    assert done.message == {"role": "assistant", "content": "正文", "extra": "[首段续段]"}
    assert ext_client.respond([{"role": "user", "content": "hi"}]) == {
        "role": "assistant",
        "content": "同步",
        "extra": "[元]",
    }

    # SSE 流式返回无参 tool_calls 且未推送 arguments 时，必须补齐合法 JSON "{}" 而非空串
    empty_args_client = OpenAIChatClient(
        api_key="sk-args",
        base_url="https://api.example.com/v1",
        model="m-args",
        post_stream=lambda _u, _p, _h: iter(
            [
                'data: {"choices":[{"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"c_empty","type":"function","function":{"name":"ping","arguments":""}}]}}]}\n',
                "data: [DONE]\n",
            ]
        ),
    )
    empty_args_done = next(e for e in empty_args_client.stream([{"role": "user", "content": "ping"}]) if isinstance(e, StreamDone))
    assert empty_args_done.message["tool_calls"] == [
        {"id": "c_empty", "type": "function", "function": {"name": "ping", "arguments": "{}"}}
    ]

    captured_payloads: list[dict[str, object]] = []

    # 记录同步请求载荷并验证默认底座不附带 temperature
    def fake_post_json(_u: str, payload: dict[str, object], _h: dict[str, str]) -> dict[str, object]:
        captured_payloads.append(payload)
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    base_client = OpenAIChatClient(api_key="sk-base", base_url="https://api.example.com/v1", model="m-base", post_json=fake_post_json)
    assert not hasattr(base_client, "temperature")
    assert base_client.respond([{"role": "user", "content": "hi"}])["content"] == "ok"
    assert "temperature" not in captured_payloads[0]

    # 底座流打开器对非 400 错误或已吐出首包后的 400 绝不盲目重试
    attempts_500 = 0

    # 模拟 500 错误验证不触发 stream_options 降级重试
    def fail_500(url: str, _p: dict[str, object], _h: dict[str, str]) -> Iterator[str]:
        nonlocal attempts_500
        attempts_500 += 1
        raise HTTPError(url, 500, "server error", {}, io.BytesIO(b""))

    with pytest.raises(HTTPError):
        list(OpenAIChatClient("sk", "https://x/v1", "m", post_stream=fail_500).stream([{"role": "user", "content": "x"}]))
    assert attempts_500 == 1

    attempts_mid400 = 0

    # 模拟吐出首行后再抛 400，验证不重发整段请求
    def mid_400(url: str, _p: dict[str, object], _h: dict[str, str]) -> Iterator[str]:
        nonlocal attempts_mid400
        attempts_mid400 += 1
        yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n'
        raise HTTPError(url, 400, "mid error", {}, io.BytesIO(b""))

    with pytest.raises(HTTPError):
        list(OpenAIChatClient("sk", "https://x/v1", "m", post_stream=mid_400).stream([{"role": "user", "content": "x"}]))
    assert attempts_mid400 == 1

    # 多线程共享同一 OpenAIChatClient 实例时 last_usage 按线程严格隔离
    barrier = threading.Barrier(2)
    results: dict[str, int] = {}

    # 按请求内容返回不同 token 用量以检验 threading.local 隔离
    def threaded_post_json(_u: str, payload: dict[str, object], _h: dict[str, str]) -> dict[str, object]:
        tag = str(payload["messages"][0]["content"])  # type: ignore[index]
        tok = 111 if tag == "t1" else 222
        return {
            "choices": [{"message": {"role": "assistant", "content": tag}}],
            "usage": {"prompt_tokens": tok, "prompt_cache_hit_tokens": 0, "completion_tokens": 1},
        }

    shared_base = OpenAIChatClient("sk", "https://x/v1", "m", post_json=threaded_post_json)

    # 在各自线程调用并栅栏同步后读取本线程 last_usage
    def worker(tag: str) -> None:
        shared_base.respond([{"role": "user", "content": tag}])
        barrier.wait(timeout=3.0)
        assert shared_base.last_usage is not None
        results[tag] = shared_base.last_usage.prompt_tokens

    t1 = threading.Thread(target=worker, args=("t1",))
    t2 = threading.Thread(target=worker, args=("t2",))
    t1.start()
    t2.start()
    t1.join(timeout=3.0)
    t2.join(timeout=3.0)
    assert results == {"t1": 111, "t2": 222}


# 测试终端 _stream_printer 三态切换（idle / thinking / text）、ToolStart 实时提示与无流式文本兜底打印
def test_cli_stream_printer_three_state_thinking_tool_start_and_fallback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from personal_assistant.assistant.cli import _stream_printer

    # 1. 完整推导 -> 工具开始 -> 工具完成 -> 再推导 -> 正文流式 -> finish
    on_event, finish = _stream_printer()
    on_event(ThinkingDelta(text=""))
    on_event(ThinkingDelta(text="先分析"))
    on_event(ThinkingDelta(text="条件"))
    on_event(ToolStart(tool="calc", arguments='{"a":1}', call_id="c1"))
    on_event(ToolStep(tool="calc", arguments='{"a":1}', observation="2", duration_ms=3.5, call_id="c1"))
    on_event(ThinkingDelta(text="核对结果"))
    on_event(TextDelta(text=""))
    on_event(TextDelta(text="答案"))
    on_event(TextDelta(text="是 2"))
    finish("答案是 2")
    out1 = capsys.readouterr().out
    assert out1 == (
        "  [推导] 先分析条件\n"
        '  [调用] calc({"a":1}) ...\n'
        '  [Act] calc({"a":1}) (3.5ms)\n'
        "  [Observe] 2\n"
        "  [推导] 核对结果\n"
        "助手> 答案是 2\n"
    )

    # 2. 仅输出思维链后直接进入 finish（未流式产出 TextDelta）时需先换行再输出兜底答复
    on_event2, finish2 = _stream_printer()
    on_event2(ThinkingDelta(text="仅推导"))
    finish2("兜底答复")
    assert capsys.readouterr().out == "  [推导] 仅推导\n助手> 兜底答复\n"

    # 3. 无任何流式正文事件时 finish 直接打印完整回复
    _, finish3 = _stream_printer()
    finish3("同步答复")
    assert capsys.readouterr().out == "助手> 同步答复\n"

    # 4. 正文与思维链交替时（text -> thinking -> text）自动补齐换行与前缀
    on_event4, finish4 = _stream_printer()
    on_event4(TextDelta(text="前"))
    on_event4(ThinkingDelta(text="想"))
    on_event4(TextDelta(text="后"))
    finish4("前后")
    assert capsys.readouterr().out == "助手> 前\n  [推导] 想\n助手> 后\n"

    # 5. 过渡正文后调工具再直接走 finish(reply) 时（第二步无 TextDelta），绝不能吞掉最终答复
    on_event5, finish5 = _stream_printer()
    on_event5(TextDelta(text="我先查一下"))
    on_event5(ToolStart(tool="calc", arguments="{}", call_id="c5"))
    on_event5(ToolStep(tool="calc", arguments="{}", observation="2", duration_ms=1.0, call_id="c5"))
    finish5("最终答复")
    assert capsys.readouterr().out == (
        "助手> 我先查一下\n"
        "  [调用] calc({}) ...\n"
        "  [Act] calc({}) (1.0ms)\n"
        "  [Observe] 2\n"
        "助手> 最终答复\n"
    )


# 测试 AssistantApp.tools 常驻注册表通过执行期 context 注入逻辑日期 today，按签名自动绑定插件且不影响未声明插件
def test_assistant_app_tools_injects_logical_today_into_plugins(tmp_path: Path) -> None:
    from datetime import date

    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    (plugins_dir / "clock_probe.py").write_text(
        """
from datetime import date
from personal_assistant.agent.plugins import Plugin
from personal_assistant.agent.tools import tool


# 声明 today 逻辑时钟参数的测试工具
@tool(description="返回主题与注入逻辑日期", parameters={"topic": "主题"})
def peek_logical_today(topic: str, *, today: date | None = None) -> str:
    return f"{topic}@{today.isoformat() if today is not None else 'none'}"


# 未声明 today 参数的常规工具（绝不应因注入而报错）
@tool(description="原样回显文本", parameters={"text": "文本"})
def plain_echo(text: str) -> str:
    return text


# 显式注册两个时钟探测工具的测试插件
class ClockProbePlugin(Plugin):
    plugin_id = "clock_probe"

    # 向主板插槽注册探测工具
    def initialize(self) -> None:
        self.context.register_tool(peek_logical_today)
        self.context.register_tool(plain_echo)
""",
        encoding="utf-8",
    )
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    tools_json = cfg_dir / "tools.json"
    tools_json.write_text(
        json.dumps(
            {
                "enabled_tools": ["peek_logical_today", "plain_echo"],
                "plugins": ["plugins"],
                "files": {"read_roots": ["data/memory"], "write_root": "data/workspace"},
                "commands": [],
            }
        ),
        encoding="utf-8",
    )

    current_day = date(2026, 5, 1)
    app = AssistantApp(
        data_root=tmp_path / "data",
        config_path=tools_json,
        project_root=tmp_path,
        llm=NoopLLMClient(),
        today_fn=lambda: current_day,
    )
    try:
        reg = app.tools
        assert not hasattr(app, "build_tools")
        assert reg.execute("peek_logical_today", '{"topic": "math"}', context={"today": app._today()}) == "math@2026-05-01"
        assert reg.execute("plain_echo", '{"text": "ok"}', context={"today": app._today()}) == "ok"

        current_day = date(2026, 5, 2)
        assert reg.execute("peek_logical_today", '{"topic": "math"}', context={"today": app._today()}) == "math@2026-05-02"
    finally:
        app.close()

    # 验证真实插件 get_current_time 在注入 today_fn 时自动对齐逻辑日期且保持对模型零参数
    repo_root = Path(__file__).resolve().parent.parent
    real_app = AssistantApp(
        data_root=tmp_path / "real_data",
        config_path=repo_root / "config" / "tools.json",
        project_root=repo_root,
        llm=NoopLLMClient(),
        today_fn=lambda: date(2026, 11, 16),
    )
    try:
        real_reg = real_app.tools
        time_res = real_reg.execute("get_current_time", "{}", context={"today": real_app._today()})
        assert time_res.startswith("2026-11-16 ") and "(Asia/Shanghai)" in time_res
        time_schema = next(d for d in real_reg.definitions() if d["function"]["name"] == "get_current_time")
        assert time_schema["function"]["parameters"]["properties"] == {}
    finally:
        real_app.close()

    # 兼容直接传入 str ISO 日期及非法字符串回退宿主当天
    from plugins.tools.custom_tools import get_current_time

    assert get_current_time(today="2026-11-16").startswith("2026-11-16 ")
    assert "(Asia/Shanghai)" in get_current_time(today="not-a-date")


# 测试 SSE 单帧多元素无 index 位置兜底、缺省 id 补齐与 StreamDone.finish_reason 强类型归位
def test_sse_tool_calls_without_index_and_finish_reason_preservation() -> None:
    from personal_assistant.ai import OpenAIChatClient

    # 1. 单帧包含多个无 index 且无 id 元素时，按位置分配独立槽位并补齐 call_0 / call_1，finish_reason 归位至 StreamDone
    sse_multi_in_one_frame = [
        'data: {"choices":[{"delta":{"role":"assistant","tool_calls":['
        '{"function":{"name":"read_markdown","arguments":"{\\"path\\":\\"a.md\\"}"}},'
        '{"function":{"name":"list_files","arguments":"{}"}}'
        ']}}]}\n',
        'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n',
        "data: [DONE]\n",
    ]
    client = OpenAIChatClient(
        api_key="sk-multi",
        base_url="https://api.example.com/v1",
        model="m-multi",
        post_stream=lambda _u, _p, _h: iter(sse_multi_in_one_frame),
        post_json=lambda _u, _p, _h: {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "同步完成"},
                    "finish_reason": "stop",
                }
            ]
        },
    )
    done_multi = next(ev for ev in client.stream([{"role": "user", "content": "go"}]) if isinstance(ev, StreamDone))
    assert done_multi.finish_reason == "tool_calls"
    assert "finish_reason" not in done_multi.message
    assert done_multi.message["tool_calls"] == [
        {"id": "call_0", "type": "function", "function": {"name": "read_markdown", "arguments": '{"path":"a.md"}'}},
        {"id": "call_1", "type": "function", "function": {"name": "list_files", "arguments": "{}"}},
    ]

    # 2. 同步 respond() 返回纯净消息字典，不再把 finish_reason 混入消息字典
    sync_msg = client.respond([{"role": "user", "content": "hi"}])
    assert sync_msg == {"role": "assistant", "content": "同步完成"}
    assert "finish_reason" not in sync_msg


# 测试 HTTPError 错误体截断透出与令牌脱敏、stream_options 400 精准降级以及 429 Retry-After 退避
def test_http_error_body_redaction_stream_options_400_precision_and_429_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io
    from personal_assistant.ai import client as llm_mod
    from personal_assistant.ai import OpenAIChatClient

    # 1. context_length_exceeded 返回 400 时绝不去掉 stream_options 重试，且透出截断并脱敏的错误体
    stream_calls: list[dict[str, object]] = []
    secret_body = (
        b'{"error":{"code":"context_length_exceeded","message":"maximum context length is 8192 tokens, '
        b'auth=Bearer tok_abc123 key=sk-secret987654"}}'
    )

    # 模拟上游返回 context_length_exceeded 400 错误
    def fail_ctx_urlopen(request: Request, timeout: int = 60):
        stream_calls.append(json.loads((request.data or b"{}").decode("utf-8")))
        raise HTTPError(request.full_url, 400, "Bad Request", {}, io.BytesIO(secret_body))

    monkeypatch.setattr(llm_mod, "urlopen", fail_ctx_urlopen)
    client = OpenAIChatClient(api_key="sk-live", base_url="https://api.example.com/v1", model="m-err")
    with pytest.raises(HTTPError) as exc_info:
        list(client.stream([{"role": "user", "content": "x"}]))
    assert len(stream_calls) == 1
    err_text = str(exc_info.value)
    assert "context_length_exceeded" in err_text
    assert "tok_abc123" not in err_text
    assert "sk-secret987654" not in err_text
    assert "Bearer ***" in err_text
    assert "sk-***" in err_text

    # 2. 非超长且无参数不兼容特征的 400 同样不重发；而 unknown stream_options 400 应降级重试一次
    unrelated_calls = 0

    # 模拟无关参数的非空 400 错误
    def fail_unrelated(request: Request, timeout: int = 60):
        nonlocal unrelated_calls
        unrelated_calls += 1
        raise HTTPError(
            request.full_url,
            400,
            "Bad Request",
            {},
            io.BytesIO(b'{"error":{"message":"invalid image_url payload"}}'),
        )

    monkeypatch.setattr(llm_mod, "urlopen", fail_unrelated)
    with pytest.raises(HTTPError, match="invalid image_url payload"):
        list(client.stream([{"role": "user", "content": "x"}]))
    assert unrelated_calls == 1

    # 3. 含 unknown parameter: stream_options 的 400 精准降级重试一次；而上下文/token 超长即便带 stream_options 关键词也不重试
    incompat_payloads: list[dict[str, object]] = []

    # 模拟 stream_options 不兼容返回 400 后去掉参数重试成功
    def compat_fallback_stream(_u: str, payload: dict[str, object], _h: dict[str, str]) -> Iterator[str]:
        incompat_payloads.append(dict(payload))
        if "stream_options" in payload:
            raise HTTPError(
                _u,
                400,
                "Bad Request",
                {},
                io.BytesIO(b'{"error":{"message":"Unknown parameter: stream_options"}}'),
            )
        return iter(['data: {"choices":[{"delta":{"content":"fallback ok"}}]}\n', "data: [DONE]\n"])

    fb_client = OpenAIChatClient(
        api_key="sk-fb",
        base_url="https://api.example.com/v1",
        model="m-fb",
        post_stream=compat_fallback_stream,
    )
    fb_events = list(fb_client.stream([{"role": "user", "content": "x"}]))
    assert len(incompat_payloads) == 2
    assert "stream_options" in incompat_payloads[0] and "stream_options" not in incompat_payloads[1]
    assert next(ev for ev in fb_events if isinstance(ev, StreamDone)).message["content"] == "fallback ok"

    # 4. 验证 1024 字节截断上限
    oversized_body = b"A" * 2048
    sync_calls = 0

    # 模拟同步调用收到超大错误体
    def fail_oversized(request: Request, timeout: int = 60):
        nonlocal sync_calls
        sync_calls += 1
        raise HTTPError(request.full_url, 400, "Bad Request", {}, io.BytesIO(oversized_body))

    monkeypatch.setattr(llm_mod, "urlopen", fail_oversized)
    with pytest.raises(HTTPError) as oversized_exc:
        client.respond([{"role": "user", "content": "x"}])
    assert sync_calls == 1
    assert len(str(oversized_exc.value).encode("utf-8")) < 1200

    # 5. 验证 429 Retry-After 夹紧在 [0.1, 10.0] 秒内优先退避
    sleeps: list[float] = []
    monkeypatch.setattr(llm_mod.time, "sleep", sleeps.append)
    attempt_429 = 0

    # 模拟连续两次 429 分别带 3.5 与超大 60 Retry-After，第三次返回成功 JSON
    def retry_429_urlopen(request: Request, timeout: int = 60):
        nonlocal attempt_429
        attempt_429 += 1
        if attempt_429 == 1:
            raise HTTPError(request.full_url, 429, "Too Many Requests", {"Retry-After": "3.5"}, io.BytesIO(b"rate"))
        if attempt_429 == 2:
            raise HTTPError(request.full_url, 429, "Too Many Requests", {"Retry-After": "60"}, io.BytesIO(b"rate"))

        # 模拟成功 HTTP 响应上下文管理器
        class _OkResp(io.BytesIO):
            # 返回自身
            def __enter__(self):
                return self

            # 关闭流
            def __exit__(self, *args):
                self.close()
                return False

        return _OkResp(json.dumps({"choices": [{"message": {"role": "assistant", "content": "recovered"}}]}).encode("utf-8"))

    monkeypatch.setattr(llm_mod, "urlopen", retry_429_urlopen)
    reply = client.respond([{"role": "user", "content": "retry"}])
    assert reply["content"] == "recovered"
    assert sleeps == [3.5, 10.0]

    # 6. 验证 HTTPError 底层流确定性关闭，且 stream_options 报错含 "usage tokens" 仍能触发 400 自动降级
    closed_flags: list[bool] = []

    # 追踪 close() 调用次数的假响应流
    class _TrackedBytesIO(io.BytesIO):
        # 记录关闭状态
        def close(self) -> None:
            closed_flags.append(True)
            super().close()

    tracked_err = HTTPError(
        "https://api.example.com/v1/chat/completions",
        400,
        "Bad Request",
        {},
        _TrackedBytesIO(b'{"error":"stream_options.include_usage is not supported for usage tokens"}'),
    )
    _, diag = llm_mod._format_http_error(tracked_err)
    assert closed_flags == [True]
    assert llm_mod._is_stream_options_incompat(diag) is True
    assert llm_mod._is_stream_options_incompat("maximum context length exceeded (too many tokens)") is False

    # 7. 验证 cli.main(["chat"]) 退出时必调用 app.close()，且 run_repl 遇运行时异常友好提示不崩退
    from personal_assistant.assistant import cli as cli_mod

    closed_cli: list[bool] = []
    printed_cli: list[str] = []

    # 模拟聊天报错并记录 close() 调用的假应用
    class _FailingReplApp:
        # 返回模拟工具注册表
        @property
        def tools(self):
            # 最小工具名容器
            class _Names:
                names = ["read_markdown"]

            return _Names()

        llm = object()

        # 模拟网络请求失败
        def chat(self, text: str, *, chat_id: str = "cli", on_event=None):
            raise RuntimeError("network timeout simulated")

        # 记录资源释放
        def close(self) -> None:
            closed_cli.append(True)

    inputs = iter(["你好", "exit"])
    monkeypatch.setattr(cli_mod, "build_app", lambda **kw: _FailingReplApp())
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed_cli.append(" ".join(str(x) for x in a)))
    cli_mod.main(["chat"])
    assert closed_cli == [True]
    assert any("本轮请求出错" in line for line in printed_cli)
