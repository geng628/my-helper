"""Layer 2 通用 ReAct 核心循环（对标 pi-agent-core）：复用 ai 层流式事件并产出 ToolStart / ToolStep / TurnEnd，不含业务与统计逻辑。"""
from __future__ import annotations

from collections.abc import Callable, Generator, Iterator, Mapping
from dataclasses import dataclass, field
from time import perf_counter

from personal_assistant.agent.tools import ToolRegistry
from personal_assistant.ai.client import LLMClient
from personal_assistant.ai.types import (
    StreamDone,
    TextDelta,
    ThinkingDelta,
    TokenUsage,
    ToolCall,
    extract_tool_calls,
)

ContextTransformer = Callable[[list[dict[str, object]]], list[dict[str, object]]]


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


# 回合结束事件：标识一次 ReAct 推理全部完成并返回最终回复与总耗时
@dataclass(frozen=True)
class TurnEnd:
    reply: str
    total_ms: float = 0.0


# 通用后台任务完成事件：供业务层广播异步任务耗时与 Token 开销，不侵入微内核具体业务
@dataclass(frozen=True)
class TaskEvent:
    kind: str
    duration_ms: float = 0.0
    usage: TokenUsage | None = None
    details: dict[str, object] = field(default_factory=dict)
    error: str | None = None


AgentEvent = TextDelta | ThinkingDelta | StreamDone | ToolStart | ToolStep | TurnEnd
RuntimeEvent = AgentEvent | TaskEvent


# 准备单轮发往大模型的上下文消息切片，确保两阶段管道变换绝不原地污染主消息列表及字典键
def _prepare_llm_messages(
    messages: list[dict[str, object]],
    transform_context: ContextTransformer | None,
    convert_to_llm: ContextTransformer | None,
) -> list[dict[str, object]]:
    if transform_context is None and convert_to_llm is None:
        return messages
    ctx_msgs: list[dict[str, object]] = [dict(msg) for msg in messages]
    if transform_context is not None:
        transformed = transform_context(ctx_msgs)
        ctx_msgs = transformed if isinstance(transformed, list) else list(transformed)
    if convert_to_llm is not None:
        converted = convert_to_llm(ctx_msgs)
        ctx_msgs = converted if isinstance(converted, list) else list(converted)
    return ctx_msgs


# 规范化写入历史的消息字典：保留扩展字段、补齐 role，并统一将 tool_calls 归一化或剥离
def _normalize_stored_assistant(
    assistant_msg: Mapping[str, object],
    *,
    reply: str | None = None,
    tool_calls: list[ToolCall] | None = None,
) -> dict[str, object]:
    stored = dict(assistant_msg)
    stored["role"] = str(stored.get("role") or "assistant")
    if reply is not None:
        stored["content"] = reply
        stored.pop("tool_calls", None)
    elif tool_calls:
        stored["tool_calls"] = [call.to_openai_dict() for call in tool_calls]
    return stored


# 透传单轮 LLM 流式事件并返回本轮完整 assistant 消息
def _stream_llm_round(
    llm: LLMClient,
    messages: list[dict[str, object]],
    tool_defs: list[dict[str, object]] | None,
) -> Generator[AgentEvent, None, dict[str, object]]:
    done: StreamDone | None = None
    for event in llm.stream(messages, tool_defs):
        yield event
        if isinstance(event, StreamDone):
            done = event
    return done.message if done is not None else {"role": "assistant", "content": ""}


# 执行流式通用 ReAct 主循环：单层 while True 内核，支持两阶段上下文转换并仅在无 ToolCall 时自然终止
def iter_react(
    llm: LLMClient,
    messages: list[dict[str, object]],
    tools: ToolRegistry,
    *,
    transform_context: ContextTransformer | None = None,
    convert_to_llm: ContextTransformer | None = None,
    tool_context: Mapping[str, object] | None = None,
) -> Iterator[AgentEvent]:
    t_turn = perf_counter()
    tool_defs = tools.definitions() if tools.names else None

    while True:
        # 1. Reason：按需经 transform_context -> convert_to_llm 转换上下文副本后流式调用模型
        llm_msgs = _prepare_llm_messages(messages, transform_context, convert_to_llm)
        assistant_msg = yield from _stream_llm_round(llm, llm_msgs, tool_defs)
        tool_calls = extract_tool_calls(assistant_msg)
        if not tool_calls:
            reply = str(assistant_msg.get("content") or "").strip()
            messages.append(_normalize_stored_assistant(assistant_msg, reply=reply))
            yield TurnEnd(reply=reply, total_ms=round((perf_counter() - t_turn) * 1000, 3))
            return

        # 2. Act & Observe：先发出 ToolStart，再执行强类型 ToolCall 并产出 ToolStep，结果写回上下文
        messages.append(_normalize_stored_assistant(assistant_msg, tool_calls=tool_calls))
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
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": observation,
                }
            )


# 执行通用 ReAct 主循环：可选订阅强类型 AgentEvent 与两阶段上下文管道，并在模型停止调用工具时返回最终回答
def run_react(
    llm: LLMClient,
    messages: list[dict[str, object]],
    tools: ToolRegistry,
    *,
    on_event: Callable[[AgentEvent], None] | None = None,
    transform_context: ContextTransformer | None = None,
    convert_to_llm: ContextTransformer | None = None,
    tool_context: Mapping[str, object] | None = None,
) -> str:
    reply = ""
    for event in iter_react(
        llm,
        messages,
        tools,
        transform_context=transform_context,
        convert_to_llm=convert_to_llm,
        tool_context=tool_context,
    ):
        if on_event is not None:
            on_event(event)
        if isinstance(event, TurnEnd):
            reply = event.reply
    return reply
