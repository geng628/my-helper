"""Layer 2 通用 ReAct 核心循环（对标 pi-agent-core）：面向中立 Context 与 12 种统一事件驱动，零厂商方言。"""
from __future__ import annotations

from collections.abc import Callable, Generator, Iterator, Mapping
from dataclasses import dataclass
from time import perf_counter

from personal_assistant.agent.tools import ToolRegistry
from personal_assistant.ai.protocol import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    Message,
    ModelInvoker,
    StreamDone,
    StreamError,
    ToolResultMessage,
)

ContextTransformer = Callable[[Context], Context]


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


# 透传单轮模型流式事件，确定性关闭底层流并返回 (StreamDone, StreamError) 终止态
def _stream_llm_round(
    llm: ModelInvoker,
    context: Context,
) -> Generator[AgentEvent, None, tuple[StreamDone | None, StreamError | None]]:
    done: StreamDone | None = None
    err: StreamError | None = None
    event_iter = llm.stream(context)
    try:
        for event in event_iter:
            yield event
            if isinstance(event, StreamDone):
                done = event
                break
            if isinstance(event, StreamError):
                err = event
                break
    finally:
        close_fn = getattr(event_iter, "close", None)
        if callable(close_fn):
            close_fn()
    return done, err


# 执行流式通用 ReAct 主循环：纯中立 Context 驱动，遇 StreamError 透传后立即退出，遇非 toolUse 自然收尾
def iter_react(
    llm: ModelInvoker,
    context: Context,
    tools: ToolRegistry,
    *,
    convert_to_llm: ContextTransformer | None = None,
    tool_context: Mapping[str, object] | None = None,
) -> Iterator[AgentEvent]:
    t_turn = perf_counter()
    history: list[Message] = list(context.messages)
    active_tools = tuple(context.tools) if context.tools else tools.schemas()

    while True:
        # 1. Reason：组装当轮不可变中立 Context，按需经 convert_to_llm 变换后流式调用模型
        round_ctx = Context(
            system_prompt=context.system_prompt,
            messages=tuple(history),
            tools=active_tools,
        )
        llm_ctx = convert_to_llm(round_ctx) if convert_to_llm is not None else round_ctx
        done, err = yield from _stream_llm_round(llm, llm_ctx)
        if err is not None:
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

        # 2. Act & Observe：先发出 ToolStart，再执行强类型 ToolCall 并产出 ToolStep，结果写回中立轨迹
        history.append(assistant_msg)
        for call in tool_calls:
            yield ToolStart(tool=call.name, arguments=call.arguments, call_id=call.id)
            t_tool = perf_counter()
            observation = tools.execute(call.name, call.arguments, context=tool_context)
            duration_ms = round((perf_counter() - t_tool) * 1000, 3)
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


# 执行通用 ReAct 主循环：可选订阅强类型 AgentEvent 与上下文转换；遇 StreamError 抛 RuntimeError，正常返回最终回答
def run_react(
    llm: ModelInvoker,
    context: Context,
    tools: ToolRegistry,
    *,
    on_event: Callable[[AgentEvent], None] | None = None,
    convert_to_llm: ContextTransformer | None = None,
    tool_context: Mapping[str, object] | None = None,
) -> str:
    reply = ""
    event_iter = iter_react(
        llm,
        context,
        tools,
        convert_to_llm=convert_to_llm,
        tool_context=tool_context,
    )
    try:
        for event in event_iter:
            if on_event is not None:
                on_event(event)
            if isinstance(event, StreamError):
                raise RuntimeError(event.error_message)
            if isinstance(event, TurnEnd):
                reply = event.reply
    finally:
        close_fn = getattr(event_iter, "close", None)
        if callable(close_fn):
            close_fn()
    return reply
