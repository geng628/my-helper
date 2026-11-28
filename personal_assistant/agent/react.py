"""Layer 2 通用 ReAct 核心循环（对标 pi-agent-core）：中立 Context 驱动、只读并发、写路障与协同取消。"""
from __future__ import annotations

from collections import ChainMap
from collections.abc import Callable, Generator, Iterator, Mapping
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType

from personal_assistant.agent.tools import ToolRegistry
from personal_assistant.ai.protocol import (
    AbortSignal,
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    Message,
    ModelInvoker,
    StreamAbortedError,
    StreamDone,
    StreamError,
    StreamOptions,
    ToolCall,
    ToolResultMessage,
)

ContextTransformer = Callable[[Context], Context]
DEFAULT_MAX_PARALLEL_TOOLS = 4


# 单步工具开始执行事件：在工具运行前向外部观察者与界面通知工具名、参数与调用标识
@dataclass(frozen=True)
class ToolStart:
    tool: str
    arguments: str
    call_id: str = ""


# 单步工具执行完成事件：携带工具名、参数、观察结果、执行耗时与调用标识
@dataclass(frozen=True)
class ToolStep:
    tool: str
    arguments: str
    observation: str
    duration_ms: float = 0.0
    call_id: str = ""


# 回合结束事件：标识一次 ReAct 推理全部完成并返回最终回复、总耗时与完整中立消息轨迹
@dataclass(frozen=True)
class TurnEnd:
    reply: str
    total_ms: float = 0.0
    messages: tuple[Message, ...] = ()


AgentEvent = AssistantMessageEvent | ToolStart | ToolStep | TurnEnd


# 校验单批最大并发工具数为正整数（拒绝布尔值与非正整数）
def _validate_max_parallel_tools(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("max_parallel_tools 必须是正整数。")
    return value


# 构造规范的协同取消终结事件 StreamError(reason="aborted")
def _make_aborted_event(abort_signal: AbortSignal | None, started_at: float) -> StreamError:
    reason = (abort_signal.reason if abort_signal is not None and abort_signal.reason else "") or "请求已取消"
    return StreamError(
        reason="aborted",
        message=AssistantMessage(stop_reason="aborted", error_message=reason),
        error_message=reason,
        latency_ms=round((perf_counter() - started_at) * 1000, 3),
    )


# 将单一真相源 abort_signal 以零拷贝 ChainMap 视图注入工具执行上下文，绝不原地修改调用方字典
def _with_abort_context(
    tool_context: Mapping[str, object] | None,
    abort_signal: AbortSignal | None,
) -> Mapping[str, object] | None:
    if abort_signal is None:
        return tool_context
    if tool_context is None:
        return MappingProxyType({"abort_signal": abort_signal})
    return ChainMap({"abort_signal": abort_signal}, tool_context)


# 将同轮工具调用按并发安全性切分为有序批次：连续 parallel=True 合并为并发批，其余作为单条串行路障
def _split_tool_batches(
    tool_calls: tuple[ToolCall, ...],
    tools: ToolRegistry,
) -> list[tuple[bool, tuple[ToolCall, ...]]]:
    is_parallel_fn = getattr(tools, "is_parallel", None)
    batches: list[tuple[bool, tuple[ToolCall, ...]]] = []
    parallel_buf: list[ToolCall] = []
    for call in tool_calls:
        if callable(is_parallel_fn) and bool(is_parallel_fn(call.name)):
            parallel_buf.append(call)
        else:
            if parallel_buf:
                batches.append((True, tuple(parallel_buf)))
                parallel_buf.clear()
            batches.append((False, (call,)))
    if parallel_buf:
        batches.append((True, tuple(parallel_buf)))
    return batches


# 执行单条工具调用并测量其独立执行耗时（毫秒）
def _execute_single_tool(
    tools: ToolRegistry,
    call: ToolCall,
    context: Mapping[str, object] | None,
) -> tuple[str, float]:
    t_tool = perf_counter()
    observation = tools.execute(call.name, call.arguments, context=context)
    duration_ms = round((perf_counter() - t_tool) * 1000, 3)
    return observation, duration_ms


# 执行单个工具批次：支持协同取消与排队任务跳过，并发批经有界线程池执行并按发射原序返回结果（取消时返回 None）
def _execute_tool_batch(
    tools: ToolRegistry,
    batch: tuple[ToolCall, ...],
    *,
    is_parallel: bool,
    tool_context: Mapping[str, object] | None,
    max_parallel_tools: int,
    abort_signal: AbortSignal | None = None,
) -> list[tuple[str, float]] | None:
    if abort_signal is not None and abort_signal.aborted:
        return None
    if not is_parallel or len(batch) <= 1 or max_parallel_tools <= 1:
        serial_results: list[tuple[str, float]] = []
        for call in batch:
            if abort_signal is not None and abort_signal.aborted:
                return None
            serial_results.append(_execute_single_tool(tools, call, tool_context))
        if abort_signal is not None and abort_signal.aborted:
            return None
        return serial_results

    # 并发工作线程仅接收只读快照，隔离主线程可变上下文防止竞态与误改写
    readonly_ctx: Mapping[str, object] | None = (
        MappingProxyType(dict(tool_context)) if tool_context is not None else None
    )
    workers = min(len(batch), max_parallel_tools)
    pool = ThreadPoolExecutor(max_workers=workers)
    futures: list[Future[tuple[str, float]]] = []
    unsub: Callable[[], None] | None = None

    # 工作线程入口：在真正调用工具前再次核查取消信号，避免排队任务在取消瞬间抢跑
    def _run_item(call: ToolCall) -> tuple[str, float]:
        if abort_signal is not None and abort_signal.aborted:
            return "错误：工具调用已取消。", 0.0
        return _execute_single_tool(tools, call, readonly_ctx)

    # 取消所有尚未启动的排队 Future
    def _cancel_queued() -> None:
        for fut in futures:
            fut.cancel()

    aborted = False
    try:
        futures = [pool.submit(_run_item, call) for call in batch]
        if abort_signal is not None:
            unsub = abort_signal.subscribe(_cancel_queued)
        results: list[tuple[str, float]] = []
        for fut in futures:
            try:
                results.append(fut.result())
            except CancelledError:
                aborted = True
                break
        if abort_signal is not None and abort_signal.aborted:
            aborted = True
    except BaseException:
        if unsub is not None:
            unsub()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    finally:
        if unsub is not None:
            unsub()
    pool.shutdown(wait=True, cancel_futures=aborted)
    return None if aborted else results


# 透传单轮模型流式事件，确定性关闭底层流并响应协同取消，返回 (StreamDone, StreamError) 终止态
def _stream_llm_round(
    llm: ModelInvoker,
    context: Context,
    options: StreamOptions | None = None,
) -> Generator[AgentEvent, None, tuple[StreamDone | None, StreamError | None]]:
    t_round = perf_counter()
    abort_signal = options.abort_signal if options is not None else None
    if abort_signal is not None and abort_signal.aborted:
        aborted_err = _make_aborted_event(abort_signal, t_round)
        yield aborted_err
        return None, aborted_err

    done: StreamDone | None = None
    err: StreamError | None = None
    event_iter = llm.stream(context, options) if options is not None else llm.stream(context)
    try:
        for event in event_iter:
            if isinstance(event, StreamError):
                yield event
                err = event
                break
            if abort_signal is not None and abort_signal.aborted:
                err = _make_aborted_event(abort_signal, t_round)
                yield err
                break
            yield event
            if isinstance(event, StreamDone):
                done = event
                break
    finally:
        close_fn = getattr(event_iter, "close", None)
        if callable(close_fn):
            close_fn()
    if err is None and abort_signal is not None and abort_signal.aborted:
        err = _make_aborted_event(abort_signal, t_round)
        yield err
        return None, err
    return done, err


# 内部流式 ReAct 循环生成器：在完成入参校验后执行多轮 Reason -> Act & Observe，支持并发批与全节点协同取消
def _iter_react_loop(
    llm: ModelInvoker,
    context: Context,
    tools: ToolRegistry,
    *,
    convert_to_llm: ContextTransformer | None,
    tool_context: Mapping[str, object] | None,
    options: StreamOptions | None,
    max_parallel_tools: int,
) -> Iterator[AgentEvent]:
    t_turn = perf_counter()
    abort_signal = options.abort_signal if options is not None else None
    effective_tool_ctx = _with_abort_context(tool_context, abort_signal)
    history: list[Message] = list(context.messages)
    active_tools = tuple(context.tools) if context.tools else tools.schemas()

    while True:
        if abort_signal is not None and abort_signal.aborted:
            yield _make_aborted_event(abort_signal, t_turn)
            return

        # 1. Reason：组装当轮不可变中立 Context，按需经 convert_to_llm 变换后流式调用模型
        round_ctx = Context(
            system_prompt=context.system_prompt,
            messages=tuple(history),
            tools=active_tools,
        )
        llm_ctx = convert_to_llm(round_ctx) if convert_to_llm is not None else round_ctx
        done, err = yield from _stream_llm_round(llm, llm_ctx, options)
        if err is not None:
            return
        if abort_signal is not None and abort_signal.aborted:
            yield _make_aborted_event(abort_signal, t_turn)
            return
        if done is None:
            empty_msg = AssistantMessage()
            history.append(empty_msg)
            yield TurnEnd(
                reply="",
                total_ms=round((perf_counter() - t_turn) * 1000, 3),
                messages=tuple(history),
            )
            return

        assistant_msg = done.message
        tool_calls = assistant_msg.tool_calls
        if done.reason != "toolUse" or not tool_calls:
            history.append(assistant_msg)
            yield TurnEnd(
                reply=assistant_msg.text.strip(),
                total_ms=round((perf_counter() - t_turn) * 1000, 3),
                messages=tuple(history),
            )
            return

        # 2. Act & Observe：按只读并发批与写路障分批执行，主线程按原序发出 ToolStart -> 批量执行 -> 按原序提交 ToolStep 与轨迹
        history.append(assistant_msg)
        for is_parallel, batch in _split_tool_batches(tool_calls, tools):
            if abort_signal is not None and abort_signal.aborted:
                yield _make_aborted_event(abort_signal, t_turn)
                return
            for call in batch:
                yield ToolStart(tool=call.name, arguments=call.arguments, call_id=call.id)
            results = _execute_tool_batch(
                tools,
                batch,
                is_parallel=is_parallel,
                tool_context=effective_tool_ctx,
                max_parallel_tools=max_parallel_tools,
                abort_signal=abort_signal,
            )
            if results is None or (abort_signal is not None and abort_signal.aborted):
                yield _make_aborted_event(abort_signal, t_turn)
                return
            for call, (observation, duration_ms) in zip(batch, results, strict=True):
                yield ToolStep(
                    tool=call.name,
                    arguments=call.arguments,
                    observation=observation,
                    duration_ms=duration_ms,
                    call_id=call.id,
                )
                history.append(
                    ToolResultMessage(
                        tool_call_id=call.id,
                        tool_name=call.name,
                        content=observation,
                        is_error=observation.startswith("错误："),
                    )
                )


# 执行流式通用 ReAct 主循环：入参即时校验，纯中立 Context 驱动，支持并发只读批、写路障与协同取消
def iter_react(
    llm: ModelInvoker,
    context: Context,
    tools: ToolRegistry,
    *,
    convert_to_llm: ContextTransformer | None = None,
    tool_context: Mapping[str, object] | None = None,
    options: StreamOptions | None = None,
    max_parallel_tools: int = DEFAULT_MAX_PARALLEL_TOOLS,
) -> Iterator[AgentEvent]:
    parallel_limit = _validate_max_parallel_tools(max_parallel_tools)
    return _iter_react_loop(
        llm,
        context,
        tools,
        convert_to_llm=convert_to_llm,
        tool_context=tool_context,
        options=options,
        max_parallel_tools=parallel_limit,
    )


# 执行通用 ReAct 主循环：支持订阅强类型 AgentEvent、上下文转换、StreamOptions 与并发上限；遇取消抛 StreamAbortedError
def run_react(
    llm: ModelInvoker,
    context: Context,
    tools: ToolRegistry,
    *,
    on_event: Callable[[AgentEvent], None] | None = None,
    convert_to_llm: ContextTransformer | None = None,
    tool_context: Mapping[str, object] | None = None,
    options: StreamOptions | None = None,
    max_parallel_tools: int = DEFAULT_MAX_PARALLEL_TOOLS,
) -> str:
    reply = ""
    event_iter = iter_react(
        llm,
        context,
        tools,
        convert_to_llm=convert_to_llm,
        tool_context=tool_context,
        options=options,
        max_parallel_tools=max_parallel_tools,
    )
    try:
        for event in event_iter:
            if on_event is not None:
                on_event(event)
            if isinstance(event, StreamError):
                if event.reason == "aborted":
                    raise StreamAbortedError(event.error_message or "请求已取消")
                raise RuntimeError(event.error_message)
            if isinstance(event, TurnEnd):
                reply = event.reply
    finally:
        close_fn = getattr(event_iter, "close", None)
        if callable(close_fn):
            close_fn()
    return reply
