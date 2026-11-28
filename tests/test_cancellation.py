"""全链路协同取消（AbortController / AbortSignal）端到端 TDD 验收测试。"""
from __future__ import annotations

import json
import socket
import sys
import threading
import time
from collections.abc import Iterator
from email.message import Message as EmailMessage
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from personal_assistant.agent import ToolRegistry, iter_react, run_react, tool
from personal_assistant.ai import (
    AbortController,
    AbortSignal,
    AnthropicMessagesProvider,
    ApiProviderRegistry,
    AssistantMessage,
    CacheMonitor,
    Context,
    GoogleGenAIProvider,
    ModelClient,
    ModelProfile,
    OpenAICompletionsProvider,
    OpenAIReasoningProvider,
    StreamAbortedError,
    StreamDone,
    StreamError,
    StreamOptions,
    StreamStart,
    TextDelta,
    TextEnd,
    TextStart,
    ToolCall,
    UserMessage,
    complete,
)
from personal_assistant.ai.providers import http as http_mod
from personal_assistant.assistant.app import AssistantApp, _GuardedTurnLLM
from personal_assistant.assistant.cli import _sigint_to_abort, run_repl
from plugins.telemetry import TelemetryObserver
from plugins.tools import shell_tools, web_tools
from scripts.web_chat import create_web_server


# 启动一次性本地 TCP 测试服务端（用于确定性模拟 TTFB 挂起或 SSE 中途挂起，不依赖外网）
def _start_stalling_tcp_server(
    *,
    send_before_stall: bytes = b"",
    on_connected: threading.Event | None = None,
    release: threading.Event | None = None,
) -> tuple[int, threading.Thread, threading.Event]:
    srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv_sock.bind(("127.0.0.1", 0))
    port = int(srv_sock.getsockname()[1])
    srv_sock.listen(4)
    srv_sock.settimeout(5.0)
    stop_event = release or threading.Event()

    # 接受单条连接、消费请求头、按需发送半截字节后阻塞等待释放信号
    def _serve() -> None:
        try:
            conn, _ = srv_sock.accept()
            with conn:
                conn.settimeout(2.0)
                try:
                    conn.recv(4096)
                except OSError:
                    pass
                if send_before_stall:
                    try:
                        conn.sendall(send_before_stall)
                    except OSError:
                        pass
                if on_connected is not None:
                    on_connected.set()
                stop_event.wait(timeout=5.0)
        except OSError:
            pass
        finally:
            srv_sock.close()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    return port, thread, stop_event


# 测试 1：AbortController / AbortSignal 只读隔离、幂等触发、回调异常隔离、退订与可中断等待
def test_abort_controller_and_signal_semantics_and_unsubscribe() -> None:
    ctrl = AbortController()
    sig = ctrl.signal

    assert isinstance(sig, AbortSignal)
    assert not hasattr(sig, "abort")
    assert sig.aborted is False
    assert sig.reason == ""

    calls: list[str] = []

    # 正常回调：可在锁外安全读取 aborted 并再次注册订阅而不死锁
    def _cb_ok() -> None:
        assert sig.aborted is True
        assert sig.reason == "首次取消"
        calls.append("ok")
        sig.subscribe(lambda: calls.append("nested"))

    # 抛异常回调：不应阻断后续监听者执行
    def _cb_boom() -> None:
        calls.append("boom")
        raise RuntimeError("listener failure")

    # 将被退订的回调：取消后绝不应触发
    def _cb_removed() -> None:
        calls.append("removed")

    unsub_ok = sig.subscribe(_cb_ok)
    unsub_boom = sig.subscribe(_cb_boom)
    unsub_removed = sig.subscribe(_cb_removed)
    unsub_removed()
    unsub_removed()

    # 未取消时 wait(0.01) 超时返回 False
    assert sig.wait(0.01) is False

    # 跨线程触发取消应立刻唤醒 wait()
    wait_result: list[bool] = []
    waiting_started = threading.Event()

    # 在子线程等待取消信号
    def _waiter() -> None:
        waiting_started.set()
        wait_result.append(sig.wait(5.0))

    t = threading.Thread(target=_waiter, daemon=True)
    t.start()
    assert waiting_started.wait(timeout=1.0)

    t0 = time.monotonic()
    assert ctrl.abort("首次取消") is True
    assert ctrl.abort("重复取消") is False
    t.join(timeout=1.0)
    assert (time.monotonic() - t0) < 0.5
    assert wait_result == [True]

    assert sig.aborted is True
    assert sig.reason == "首次取消"
    assert "ok" in calls
    assert "boom" in calls
    assert "nested" in calls
    assert "removed" not in calls

    # 已取消后再订阅应立刻同步触发一次并返回无副作用退订函数
    late_called: list[bool] = []
    unsub_late = sig.subscribe(lambda: late_called.append(True))
    assert late_called == [True]
    unsub_late()
    unsub_ok()
    unsub_boom()


# 测试 2：Layer 1 HTTP 底座毫秒级打断预取消、429 退避睡眠、真实阻塞 TTFB Socket 与 SSE 中途挂起
def test_http_transport_aborts_retry_backoff_and_blocking_socket_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 2a) 预取消直接抛 StreamAbortedError，零网络调用
    pre_ctrl = AbortController()
    pre_ctrl.abort("发请求前取消")
    called = False

    # 断言不会被调用的桩函数
    def _should_not_open(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("预取消不应发起网络请求")

    monkeypatch.setattr(http_mod, "open_http_request", _should_not_open)
    with pytest.raises(StreamAbortedError, match="发请求前取消"):
        list(http_mod.default_post_stream("http://127.0.0.1:9/v1", {}, {}, abort_signal=pre_ctrl.signal))
    assert called is False

    # 2b) HTTP 429 Retry-After: 10 退避等待期间取消应在 <0.5s 立即中断
    retry_ctrl = AbortController()
    entered_429 = threading.Event()

    # 首次返回带 10s Retry-After 的 429 响应
    def _open_429(request: Request, *, timeout: float = 60, abort_signal: AbortSignal | None = None) -> object:
        del request, timeout, abort_signal
        entered_429.set()
        hdrs = EmailMessage()
        hdrs["Retry-After"] = "10"
        raise HTTPError("http://example.invalid/v1", 429, "Too Many Requests", hdrs, None)

    monkeypatch.setattr(http_mod, "open_http_request", _open_429)

    # 收到 429 后迅速触发取消
    def _abort_soon() -> None:
        assert entered_429.wait(timeout=2.0)
        time.sleep(0.03)
        retry_ctrl.abort("退避中取消")

    ab = threading.Thread(target=_abort_soon, daemon=True)
    ab.start()
    t0 = time.monotonic()
    with pytest.raises(StreamAbortedError, match="退避中取消"):
        http_mod.default_post_json("http://example.invalid/v1", {"a": 1}, {}, abort_signal=retry_ctrl.signal)
    elapsed = time.monotonic() - t0
    ab.join(timeout=1.0)
    assert elapsed < 0.8

    # 恢复真实 open_http_request 以验证真实 TCP Socket 毫秒级打断
    monkeypatch.undo()

    # 2c) 真实 TCP 服务端接收请求后永不发响应头（TTFB 挂起），取消后应立刻唤醒
    ttfb_connected = threading.Event()
    ttfb_port, ttfb_thread, ttfb_release = _start_stalling_tcp_server(on_connected=ttfb_connected)
    ttfb_ctrl = AbortController()

    # 确认连上服务端后立即触发取消
    def _abort_ttfb() -> None:
        assert ttfb_connected.wait(timeout=2.0)
        time.sleep(0.03)
        ttfb_ctrl.abort("TTFB 挂起取消")

    t_ttfb = threading.Thread(target=_abort_ttfb, daemon=True)
    t_ttfb.start()
    t0 = time.monotonic()
    try:
        with pytest.raises(StreamAbortedError, match="TTFB 挂起取消"):
            list(
                http_mod.default_post_stream(
                    f"http://127.0.0.1:{ttfb_port}/v1/chat/completions",
                    {"messages": []},
                    {},
                    max_retries=0,
                    abort_signal=ttfb_ctrl.signal,
                )
            )
    finally:
        ttfb_release.set()
        t_ttfb.join(timeout=1.0)
        ttfb_thread.join(timeout=2.0)
    assert (time.monotonic() - t0) < 1.0

    # 2d) 真实 TCP 服务端发送首帧 SSE 后中途挂起，取消后立即退出且已读出首帧
    sse_connected = threading.Event()
    sse_head = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/event-stream\r\n"
        b"Connection: close\r\n\r\n"
        b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
    )
    sse_port, sse_thread, sse_release = _start_stalling_tcp_server(
        send_before_stall=sse_head,
        on_connected=sse_connected,
    )
    sse_ctrl = AbortController()
    received: list[str] = []
    t0 = time.monotonic()
    try:
        with pytest.raises(StreamAbortedError, match="SSE 中途取消"):
            for line in http_mod.default_post_stream(
                f"http://127.0.0.1:{sse_port}/v1/chat/completions",
                {"messages": []},
                {},
                max_retries=0,
                abort_signal=sse_ctrl.signal,
            ):
                received.append(line)
                if "hello" in line:
                    sse_ctrl.abort("SSE 中途取消")
    finally:
        sse_release.set()
        sse_thread.join(timeout=2.0)
    assert any("hello" in line for line in received)
    assert (time.monotonic() - t0) < 1.0


# 测试 3：四大内建协议翻译器均响应 pre-abort 与 mid-stream/EOF-race abort，且不污染 CacheMonitor
def test_all_four_providers_emit_stream_error_aborted_cleanly() -> None:
    ctx = Context(system_prompt="sys", messages=(UserMessage("hi"),))
    specs = [
        (
            "openai-completions",
            OpenAICompletionsProvider,
            [
                'data: {"choices":[{"delta":{"content":"part1"}}]}\n\n',
                'data: {"choices":[{"delta":{"content":"part2"}}]}\n\n',
            ],
        ),
        (
            "openai-reasoning",
            OpenAIReasoningProvider,
            [
                'data: {"choices":[{"delta":{"reasoning_content":"think1"}}]}\n\n',
                'data: {"choices":[{"delta":{"content":"ans"}}]}\n\n',
            ],
        ),
        (
            "anthropic-messages",
            AnthropicMessagesProvider,
            [
                'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
                'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"part1"}}\n\n',
                'event: message_stop\ndata: {"type":"message_stop"}\n\n',
            ],
        ),
        (
            "google-generative-ai",
            GoogleGenAIProvider,
            [
                'data: {"candidates":[{"content":{"parts":[{"text":"part1"}]}}]}\n\n',
                'data: {"candidates":[{"content":{"parts":[{"text":"part2"}]}}]}\n\n',
            ],
        ),
    ]

    for api_name, provider_cls, lines in specs:
        model = ModelProfile(name="m", api=api_name, base_url="http://x.invalid/v1", model="m", api_key="sk-test")

        # 3a) 预取消：绝不发起 post_stream，产出 StreamStart -> StreamError(reason="aborted")
        calls = 0

        # 不应被触发的流假桩
        def _never_stream(
            url: str,
            payload: dict[str, object],
            headers: dict[str, str],
            *,
            abort_signal: AbortSignal | None = None,
        ) -> Iterator[str]:
            nonlocal calls
            del url, payload, headers, abort_signal
            calls += 1
            return iter(())

        pre_ctrl = AbortController()
        pre_ctrl.abort(f"pre-{api_name}")
        evs = list(provider_cls(post_stream=_never_stream).stream(model, ctx, StreamOptions(abort_signal=pre_ctrl.signal)))
        assert calls == 0
        assert isinstance(evs[0], StreamStart)
        assert isinstance(evs[-1], StreamError)
        assert evs[-1].reason == "aborted"
        assert evs[-1].message.stop_reason == "aborted"
        assert f"pre-{api_name}" in evs[-1].error_message
        assert not any(isinstance(e, StreamDone) for e in evs)

        # 3b) 流中途取消（甚至假桩直接 EOF 不抛异常）也必须收敛为 StreamError(reason="aborted")，绝不误发 StreamDone
        mid_ctrl = AbortController()

        # 吐出第一帧后即触发取消并直接结束流（模拟 socket 关闭导致 EOF 的竞态）
        def _eof_after_first(
            url: str,
            payload: dict[str, object],
            headers: dict[str, str],
            *,
            abort_signal: AbortSignal | None = None,
            _lines: list[str] = lines,
            _ctrl: AbortController = mid_ctrl,
            _api: str = api_name,
        ) -> Iterator[str]:
            del url, payload, headers
            assert abort_signal is _ctrl.signal
            yield _lines[0]
            if len(_lines) > 1 and _api == "anthropic-messages":
                yield _lines[1]
            _ctrl.abort(f"mid-{_api}")

        reg = ApiProviderRegistry()
        reg.register(provider_cls(post_stream=_eof_after_first))
        monitor = CacheMonitor()
        client = monitor.wrap(ModelClient(profile=model, registry=reg), session_key="s-abort")

        mid_evs = list(client.stream(ctx, StreamOptions(abort_signal=mid_ctrl.signal)))
        assert isinstance(mid_evs[-1], StreamError)
        assert mid_evs[-1].reason == "aborted"
        assert mid_evs[-1].message.stop_reason == "aborted"
        assert not any(isinstance(e, StreamDone) for e in mid_evs)
        assert monitor.summary()["llm_calls"] == 0

        # 3c) complete() 遇取消统一抛出 StreamAbortedError（且是 RuntimeError 子类）
        with pytest.raises(StreamAbortedError) as exc_info:
            complete(model, ctx, StreamOptions(abort_signal=pre_ctrl.signal), registry=reg)
        assert isinstance(exc_info.value, RuntimeError)


# 测试 4：Layer 2 iter_react / run_react 协同取消（预取消、并行批唤醒、未启动排队工具跳过与原上下文零变异）
def test_react_loop_cooperative_cancellation_parallel_batch_and_queued_futures() -> None:
    started_parallel: list[str] = []
    started_lock = threading.Lock()
    two_running = threading.Barrier(2, timeout=2.0)
    write_called = False

    # 可中断并发只读工具：阻塞等待取消信号并在收到后迅速返回
    @tool(description="可中断只读工具", parameters={"tag": "标记"}, parallel=True)
    def slow_read(tag: str, *, abort_signal: AbortSignal | None = None) -> str:
        with started_lock:
            started_parallel.append(tag)
        if tag in {"r1", "r2"}:
            two_running.wait()
            assert abort_signal is not None
            abort_signal.wait(5.0)
            return f"cancelled:{tag}" if abort_signal.aborted else f"done:{tag}"
        return f"should-not-run:{tag}"

    # 后续串行写路障：取消后绝不应被执行
    @tool(description="写路障", parameters={"x": "值"}, risk="write", parallel=False)
    def write_barrier(x: str) -> str:
        nonlocal write_called
        write_called = True
        return f"wrote:{x}"

    # 固定返回 [r1, r2, r3, r4, write_barrier] 的单轮模型桩
    class _ToolBatchLLM:
        last_usage = None

        # 初始化调用计数
        def __init__(self) -> None:
            self.calls = 0

        # 产出 4 个并发只读工具加 1 个写路障
        def stream(self, context: Context, options: StreamOptions | None = None) -> Iterator[object]:
            del context, options
            self.calls += 1
            msg = AssistantMessage(
                blocks=(
                    ToolCall(id="c1", name="slow_read", arguments='{"tag":"r1"}'),
                    ToolCall(id="c2", name="slow_read", arguments='{"tag":"r2"}'),
                    ToolCall(id="c3", name="slow_read", arguments='{"tag":"r3"}'),
                    ToolCall(id="c4", name="slow_read", arguments='{"tag":"r4"}'),
                    ToolCall(id="c5", name="write_barrier", arguments='{"x":"1"}'),
                ),
                stop_reason="toolUse",
            )
            yield StreamDone(reason="toolUse", message=msg)

        # 同步占位
        def complete(self, context: Context, options: StreamOptions | None = None) -> AssistantMessage:
            raise NotImplementedError

    registry = ToolRegistry(slow_read.__agent_tool__, write_barrier.__agent_tool__)

    # 4a) 预取消不调 LLM、不发 TurnEnd
    pre_ctrl = AbortController()
    pre_ctrl.abort("引擎预取消")
    pre_llm = _ToolBatchLLM()
    pre_evs = list(
        iter_react(
            pre_llm,
            Context(messages=(UserMessage("go"),)),
            registry,
            options=StreamOptions(abort_signal=pre_ctrl.signal),
        )
    )
    assert pre_llm.calls == 0
    assert len(pre_evs) == 1 and isinstance(pre_evs[0], StreamError) and pre_evs[0].reason == "aborted"

    # 4b) 并发批上限 2：r1/r2 同时进入后取消，r3/r4 与 write_barrier 绝不启动，且 caller_ctx 不被原地污染
    ctrl = AbortController()
    caller_ctx: dict[str, object] = {"custom": "keep"}

    # 等待 r1/r2 都启动后触发取消
    def _trigger_after_two() -> None:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with started_lock:
                if len(started_parallel) >= 2:
                    break
            time.sleep(0.01)
        ctrl.abort("批处理中止")

    t_abort = threading.Thread(target=_trigger_after_two, daemon=True)
    t_abort.start()
    llm = _ToolBatchLLM()
    t0 = time.monotonic()
    evs = list(
        iter_react(
            llm,
            Context(messages=(UserMessage("go"),)),
            registry,
            tool_context=caller_ctx,
            options=StreamOptions(abort_signal=ctrl.signal),
            max_parallel_tools=2,
        )
    )
    elapsed = time.monotonic() - t0
    t_abort.join(timeout=1.0)

    assert elapsed < 1.5
    assert llm.calls == 1
    assert set(started_parallel) == {"r1", "r2"}
    assert write_called is False
    assert caller_ctx == {"custom": "keep"}
    assert isinstance(evs[-1], StreamError) and evs[-1].reason == "aborted"

    # 4c) run_react 遇取消抛出 StreamAbortedError
    with pytest.raises(StreamAbortedError, match="引擎预取消"):
        run_react(
            _ToolBatchLLM(),
            Context(messages=(UserMessage("go"),)),
            registry,
            options=StreamOptions(abort_signal=pre_ctrl.signal),
        )


# 测试 5：慢 I/O 工具（shell_tools 子进程组秒杀 + web_tools 阻塞抓页中断与 DDG 不回退 Bing）且零泄漏至 ToolSchema
def test_shell_and_web_tools_honor_abort_signal_and_clean_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 5a) ToolSchema 不泄漏 abort_signal，accepted_context_keys 包含 abort_signal
    for fn in (shell_tools.run_command, web_tools.web_search, web_tools.fetch_webpage):
        spec = fn.__agent_tool__
        assert "abort_signal" not in spec.parameters
        assert "abort_signal" not in spec.required_names
        assert spec.accepted_context_keys is not None and "abort_signal" in spec.accepted_context_keys

    # 5b) shell_tools：正在 sleep 30s 的子进程组在取消后 <1s 内被 SIGKILL 强杀
    sh_ctrl = AbortController()
    timer = threading.Timer(0.1, sh_ctrl.abort, args=("停止命令",))
    timer.start()
    t0 = time.monotonic()
    try:
        rc, _, _, run_err = shell_tools._run_subprocess_limited(
            (sys.executable, "-c", "import time; time.sleep(30)"),
            cwd=tmp_path,
            timeout=15.0,
            env={"PATH": "/usr/bin:/bin"},
            max_capture_bytes=65536,
            abort_signal=sh_ctrl.signal,
        )
    finally:
        timer.cancel()
    assert (time.monotonic() - t0) < 1.5
    assert rc == -1
    assert run_err is not None and "取消" in run_err

    # 5c) shell_tools 预取消直接返回取消错误，不启动子进程
    pre_ctrl = AbortController()
    pre_ctrl.abort("预取消")
    policy = shell_tools.ShellPolicy(project_root=tmp_path, cwd=tmp_path)
    out = shell_tools.execute_safe_command("pwd", policy, abort_signal=pre_ctrl.signal)
    assert out.startswith("错误：") and "取消" in out

    # 5d) web_tools：阻塞抓页套接字被取消信号 <1s 打断
    web_connected = threading.Event()
    web_port, web_thread, web_release = _start_stalling_tcp_server(on_connected=web_connected)
    web_ctrl = AbortController()

    # 将测试 URL 指向本地挂起服务端以验证底层 socket 中断机制
    def _fake_resolve(url: str, **_kwargs: object) -> tuple[str, str, int, str]:
        return url, "example.test", web_port, "127.0.0.1"

    monkeypatch.setattr(web_tools, "resolve_and_validate_public_url", _fake_resolve)

    # 连上本地挂起服务端后立刻取消
    def _abort_web() -> None:
        assert web_connected.wait(timeout=2.0)
        time.sleep(0.03)
        web_ctrl.abort("网页取消")

    t_web = threading.Thread(target=_abort_web, daemon=True)
    t_web.start()
    t0 = time.monotonic()
    try:
        web_out = web_tools.execute_fetch_webpage(
            f"http://example.test:{web_port}/slow",
            web_tools.WebPolicy(timeout=15.0),
            abort_signal=web_ctrl.signal,
        )
    finally:
        web_release.set()
        t_web.join(timeout=1.0)
        web_thread.join(timeout=2.0)
    assert (time.monotonic() - t0) < 1.0
    assert web_out.startswith("错误：") and "取消" in web_out

    # 5e) web_search 在 DDG 请求期间被取消时，绝不能吞掉取消去回退查 Bing RSS
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    search_ctrl = AbortController()
    search_urls: list[str] = []

    # 模拟 DDG 请求期间触发取消
    def _cancel_on_ddg(url: str, headers: dict[str, str], timeout: float, max_bytes: int) -> tuple[str, str, bytes]:
        del headers, timeout, max_bytes
        search_urls.append(url)
        search_ctrl.abort("搜索中止")
        raise StreamAbortedError("搜索中止")

    search_out = web_tools.execute_web_search(
        "python",
        web_tools.WebPolicy(timeout=10.0),
        http_get=_cancel_on_ddg,
        abort_signal=search_ctrl.signal,
    )
    assert len(search_urls) == 1 and "duckduckgo" in search_urls[0]
    assert search_out.startswith("错误：") and "取消" in search_out


# 测试 6：Layer 3 AssistantApp.abort_turn 回滚 facts、零写 SQLite、清遥测、即释会话锁、护栏优先取消与 CLI 容错
def test_app_abort_turn_rolls_back_facts_telemetry_and_releases_lock_plus_cli(tmp_path: Path) -> None:
    round2_entered = threading.Event()
    round2_unblocked = threading.Event()

    # 第 1 轮调用 remember_fact，第 2 轮阻塞等待 abort_signal；后续新回合直接正常回答
    class _BlockingAppLLM:
        last_usage = None

        # 初始化回合计数
        def __init__(self) -> None:
            self.calls = 0

        # 模拟先写暂存事实再挂起等待取消的多步流
        def stream(self, context: Context, options: StreamOptions | None = None) -> Iterator[object]:
            del context
            self.calls += 1
            sig = options.abort_signal if options is not None else None
            if self.calls == 1:
                msg = AssistantMessage(
                    blocks=(ToolCall(id="tc1", name="remember_fact", arguments='{"fact":"这是一条应被回滚的暂存事实"}'),),
                    stop_reason="toolUse",
                )
                yield StreamStart(partial=AssistantMessage())
                yield StreamDone(reason="toolUse", message=msg)
                return
            if self.calls == 2:
                yield StreamStart(partial=AssistantMessage())
                yield TextStart(content_index=0, partial=AssistantMessage())
                yield TextDelta(content_index=0, text="半截回复", partial=AssistantMessage.from_text("半截回复"))
                round2_entered.set()
                assert sig is not None
                sig.wait(timeout=5.0)
                round2_unblocked.set()
                reason = sig.reason or "已取消"
                yield StreamError(
                    reason="aborted",
                    message=AssistantMessage(stop_reason="aborted", error_message=reason),
                    error_message=reason,
                )
                return
            reply = AssistantMessage.from_text("恢复后正常回复", stop_reason="stop")
            yield StreamStart(partial=AssistantMessage())
            yield TextStart(content_index=0, partial=AssistantMessage())
            yield TextDelta(content_index=0, text="恢复后正常回复", partial=reply)
            yield TextEnd(content_index=0, text="恢复后正常回复", partial=reply)
            yield StreamDone(reason="stop", message=reply)

        # 同步占位
        def complete(self, context: Context, options: StreamOptions | None = None) -> AssistantMessage:
            raise NotImplementedError

    observer = TelemetryObserver()
    app = AssistantApp(
        tmp_path / "data",
        llm=_BlockingAppLLM(),
        auto_consolidate=False,
        observers=(observer,),
    )
    try:
        # 空闲会话调用 abort_turn 应返回 False
        assert app.abort_turn("s1") is False

        frames: list[dict[str, object]] = []

        # 在子线程消费 stream_chat
        def _run_stream() -> None:
            for frame in app.stream_chat("记住这个并回答", chat_id="s1"):
                frames.append(frame)

        t_stream = threading.Thread(target=_run_stream, daemon=True)
        t_stream.start()
        assert round2_entered.wait(timeout=2.0)

        t0 = time.monotonic()
        assert app.abort_turn("s1", reason="测试停止") is True
        t_stream.join(timeout=2.0)
        assert not t_stream.is_alive()
        assert (time.monotonic() - t0) < 1.0
        assert round2_unblocked.is_set()

        # 校验终帧为 aborted、零写入 SQLite、facts.md 暂存行已回滚、遥测无残留
        assert frames[-1] == {"type": "aborted", "reason": "测试停止", "chat_id": "s1"}
        assert not any(f.get("type") == "done" for f in frames)
        assert app.state.count_messages("s1") == 0
        facts_file = tmp_path / "data" / "memory" / "facts.md"
        if facts_file.exists():
            assert "应被回滚的暂存事实" not in facts_file.read_text(encoding="utf-8")
        assert "s1" not in observer._in_flight

        # 会话锁已立即释放，紧接着的新回合可正常完成落库
        follow = app.chat("继续聊", chat_id="s1")
        assert follow["reply"] == "恢复后正常回复"
        assert app.state.count_messages("s1") == 2

        # 同步 app.chat 传入已取消信号应直接抛出 StreamAbortedError
        ext_ctrl = AbortController()
        ext_ctrl.abort("外部信号取消")
        with pytest.raises(StreamAbortedError, match="外部信号取消"):
            app.chat("不应执行", chat_id="s1", abort_signal=ext_ctrl.signal)

        # _GuardedTurnLLM 在预算耗尽且已取消时必须优先产出 StreamError(reason="aborted")，而非预算兜底文本
        guarded = _GuardedTurnLLM(
            _BlockingAppLLM(),
            max_steps=1,
            max_tokens=100,
            max_seconds=10.0,
            max_repeated_tool_calls=1,
            clock=time.monotonic,
        )
        guarded.steps = 1
        g_evs = list(guarded.stream(Context(messages=(UserMessage("x"),)), StreamOptions(abort_signal=ext_ctrl.signal)))
        assert len(g_evs) == 1 and isinstance(g_evs[0], StreamError) and g_evs[0].reason == "aborted"
        assert guarded.breach is None

        # CLI _sigint_to_abort 上下文正常恢复处理器，且 run_repl 捕获 StreamAbortedError 后保持 REPL 可继续运行
        with _sigint_to_abort(app, "s1"):
            pass

        # 始终抛出 StreamAbortedError 的最小替身应用
        class _AbortingCliApp:
            tools = app.tools

            # 模拟被 Ctrl+C 中止的对话回合
            def chat(self, text: str, chat_id: str = "cli", **kwargs: object) -> dict[str, object]:
                del text, chat_id, kwargs
                raise StreamAbortedError("用户按下 Ctrl+C 中断")

            # 空中止实现
            def abort_turn(self, chat_id: str, *, reason: str = "") -> bool:
                del chat_id, reason
                return False

            # 空关闭实现
            def close(self) -> None:
                return None

        inputs = iter(["hello", "quit"])
        printed: list[str] = []
        mp = pytest.MonkeyPatch()
        try:
            mp.setattr("builtins.input", lambda _prompt="": next(inputs))
            mp.setattr("builtins.print", lambda *args, **kwargs: printed.append(" ".join(str(a) for a in args)))
            fake_cli_app: Any = _AbortingCliApp()
            run_repl(fake_cli_app)
        finally:
            mp.undo()
        assert any("已中断本轮回复" in line for line in printed)
    finally:
        app.close()


# 测试 7：Web 服务 POST /api/chat/abort 打断在途 SSE 流与 /api/chat，严格执行同源/鉴权守卫，前端具备停止按钮
def test_web_chat_abort_endpoint_and_security_boundaries(tmp_path: Path) -> None:
    stream_entered = threading.Event()
    blocking_entered = threading.Event()

    # 阻塞直到 abort_signal 触发的最小模型桩
    class _WebBlockingLLM:
        last_usage = None

        # 根据用户输入阻塞等待取消信号
        def stream(self, context: Context, options: StreamOptions | None = None) -> Iterator[object]:
            sig = options.abort_signal if options is not None else None
            last_msg = context.messages[-1].content if context.messages else ""
            yield StreamStart(partial=AssistantMessage())
            yield TextStart(content_index=0, partial=AssistantMessage())
            yield TextDelta(content_index=0, text="思考中", partial=AssistantMessage.from_text("思考中"))
            if "sync-block" in str(last_msg):
                blocking_entered.set()
            else:
                stream_entered.set()
            assert sig is not None
            sig.wait(timeout=5.0)
            reason = sig.reason or "用户已停止生成"
            yield StreamError(
                reason="aborted",
                message=AssistantMessage(stop_reason="aborted", error_message=reason),
                error_message=reason,
            )

        # 同步占位
        def complete(self, context: Context, options: StreamOptions | None = None) -> AssistantMessage:
            raise NotImplementedError

    app = AssistantApp(tmp_path / "data", llm=_WebBlockingLLM(), auto_consolidate=False)
    server = create_web_server(app, host="127.0.0.1", port=0, auth_token="secret-token")
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    auth_headers = {"Content-Type": "application/json", "Authorization": "Bearer secret-token"}

    try:
        # 7a) 安全边界：缺 Bearer 返 401，跨源 Origin 返 403
        with pytest.raises(HTTPError) as unauth:
            urlopen(
                Request(f"{base}/api/chat/abort", data=b'{"chat_id":"web"}', headers={"Content-Type": "application/json"}, method="POST"),
                timeout=3,
            )
        assert unauth.value.code == 401

        with pytest.raises(HTTPError) as cross_origin:
            urlopen(
                Request(
                    f"{base}/api/chat/abort",
                    data=b'{"chat_id":"web"}',
                    headers={**auth_headers, "Origin": "http://evil.example"},
                    method="POST",
                ),
                timeout=3,
            )
        assert cross_origin.value.code == 403

        # 7b) 在途 /api/chat/stream 被 /api/chat/abort 毫秒级打断并收到 aborted SSE 帧
        sse_frames: list[dict[str, object]] = []

        # 读取 SSE 响应帧直至流结束
        def _consume_sse() -> None:
            req = Request(
                f"{base}/api/chat/stream",
                data=json.dumps({"message": "长任务", "chat_id": "w1"}).encode("utf-8"),
                headers=auth_headers,
                method="POST",
            )
            with urlopen(req, timeout=5) as resp:
                for raw in resp:
                    line = raw.decode("utf-8").strip()
                    if line.startswith("data:"):
                        sse_frames.append(json.loads(line[5:].strip()))

        t_sse = threading.Thread(target=_consume_sse, daemon=True)
        t_sse.start()
        assert stream_entered.wait(timeout=2.0)

        t0 = time.monotonic()
        abort_req = Request(
            f"{base}/api/chat/abort",
            data=json.dumps({"chat_id": "w1"}).encode("utf-8"),
            headers=auth_headers,
            method="POST",
        )
        with urlopen(abort_req, timeout=3) as abort_resp:
            abort_body = json.loads(abort_resp.read().decode("utf-8"))
        t_sse.join(timeout=2.0)
        assert (time.monotonic() - t0) < 1.0
        assert abort_body == {"ok": True, "chat_id": "w1", "aborted": True}
        assert any(f.get("type") == "text_delta" for f in sse_frames)
        assert sse_frames[-1]["type"] == "aborted"
        assert app.state.count_messages("w1") == 0

        # 7c) 同步 /api/chat 被 /api/chat/abort 中止后返回 HTTP 409 与 aborted=True
        sync_status: list[tuple[int, dict[str, object]]] = []

        # 调用同步 /api/chat 并记录返回的 HTTP 状态码与体
        def _call_sync_chat() -> None:
            req = Request(
                f"{base}/api/chat",
                data=json.dumps({"message": "sync-block", "chat_id": "w2"}).encode("utf-8"),
                headers=auth_headers,
                method="POST",
            )
            try:
                with urlopen(req, timeout=5) as resp:
                    sync_status.append((resp.status, json.loads(resp.read().decode("utf-8"))))
            except HTTPError as err:
                sync_status.append((err.code, json.loads(err.read().decode("utf-8"))))

        t_sync = threading.Thread(target=_call_sync_chat, daemon=True)
        t_sync.start()
        assert blocking_entered.wait(timeout=2.0)
        with urlopen(
            Request(
                f"{base}/api/chat/abort",
                data=json.dumps({"chat_id": "w2"}).encode("utf-8"),
                headers=auth_headers,
                method="POST",
            ),
            timeout=3,
        ) as resp2:
            assert json.loads(resp2.read().decode("utf-8"))["aborted"] is True
        t_sync.join(timeout=2.0)
        assert len(sync_status) == 1
        assert sync_status[0][0] == 409
        assert sync_status[0][1]["aborted"] is True

        # 7d) 前端 HTML 包含停止按钮与 /api/chat/abort 接入
        html = (Path(__file__).resolve().parent.parent / "scripts" / "web_ui.html").read_text(encoding="utf-8")
        assert 'id="btnStop"' in html
        assert "/api/chat/abort" in html
    finally:
        server.shutdown()
        server.server_close()
        srv_thread.join(timeout=2.0)
        app.close()
