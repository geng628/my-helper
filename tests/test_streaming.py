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

from personal_assistant.agent import ObserverProtocol, TaskEvent, Tool, ToolRegistry, TurnEnd, ToolStep, iter_react
from personal_assistant.ai import (
    CacheMonitor,
    LLMClient,
    LLMStreamEvent,
    NoopLLMClient,
    OpenAICompatibleLLM,
    StreamDone,
    TextDelta,
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
            raise HTTPError(request.full_url, 400, "Unsupported", {}, io.BytesIO(b"bad"))
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


# 测试 iter_react 直接透传 TextDelta / StreamDone 并产出 ToolStep(duration_ms) 与 TurnEnd
def test_iter_react_emits_slim_typed_events() -> None:
    llm = ScriptedStreamLLM(
        [
            [
                StreamDone(
                    message={"role": "assistant", "content": None, "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "double", "arguments": '{"n": "21"}'}}]},
                    usage=TokenUsage(prompt_tokens=50, cached_tokens=20, completion_tokens=10),
                    ttft_ms=12.0,
                    latency_ms=30.0,
                )
            ],
            [
                TextDelta(text="结果"),
                TextDelta(text="是 42"),
                StreamDone(message={"role": "assistant", "content": "结果是 42"}, usage=TokenUsage(prompt_tokens=80, cached_tokens=50, completion_tokens=6), ttft_ms=8.5, latency_ms=25.0),
            ],
        ]
    )
    registry = ToolRegistry(Tool(name="double", description="翻倍", parameters={"n": "整数"}, handler=lambda n: str(int(n) * 2)))
    msgs: list[dict[str, object]] = [{"role": "user", "content": "21 翻倍"}]
    events = list(iter_react(llm, msgs, registry))

    assert isinstance(events[0], StreamDone)
    assert isinstance(events[1], ToolStep) and events[1].tool == "double" and events[1].observation == "42" and events[1].duration_ms >= 0.0
    assert [ev.text for ev in events if isinstance(ev, TextDelta)] == ["结果", "是 42"]
    assert isinstance(events[-1], TurnEnd) and events[-1].reply == "结果是 42" and events[-1].total_ms >= 0.0


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
        assert [ev["type"] for ev in evs] == ["tool_step", "text_delta", "text_delta", "done"]
        assert evs[0]["tool"] == "list_files" and float(evs[0]["duration_ms"]) >= 0.0
        assert [ev["text"] for ev in evs[1:3]] == ["目录", "正常"] and evs[3]["reply"] == "目录正常"
    finally:
        srv.shutdown()
        srv.server_close()
        app.close()


# 测试 OpenAIChatClient 通用协议底座、声明式 _stream_delta_fields 扩展与底层流/线程语义
def test_openai_chat_client_base_and_delta_field_extension() -> None:
    from dataclasses import dataclass
    from personal_assistant import ai as ai_pkg
    from personal_assistant.ai import OpenAIChatClient

    assert issubclass(OpenAICompatibleLLM, OpenAIChatClient)
    assert OpenAICompatibleLLM.stream is OpenAIChatClient.stream
    assert not hasattr(ai_pkg, "open_openai_stream")
    assert not hasattr(ai_pkg, "consume_openai_sse")

    # 声明式扩展子类：收集自定义字符串 delta 字段并统一规范化同步与流式响应
    @dataclass
    class _ExtraDeltaClient(OpenAIChatClient):
        # 声明需从 SSE delta 中聚合的扩展文本字段
        def _stream_delta_fields(self) -> tuple[str, ...]:
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
    ext_client = _ExtraDeltaClient(
        api_key="sk-ext",
        base_url="https://api.example.com/v1",
        model="m-ext",
        post_stream=lambda _u, _p, _h: iter(sse_lines),
        post_json=lambda _u, _p, _h: {"choices": [{"message": {"role": "assistant", "content": "同步", "extra": "元"}}]},
    )
    events = list(ext_client.stream([{"role": "user", "content": "hi"}]))
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["正文"]
    done = next(e for e in events if isinstance(e, StreamDone))
    assert done.ttft_ms is not None and done.ttft_ms >= 0.0
    assert done.message == {"role": "assistant", "content": "正文", "extra": "[首段续段]"}
    assert ext_client.respond([{"role": "user", "content": "hi"}]) == {
        "role": "assistant",
        "content": "同步",
        "extra": "[元]",
    }

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
