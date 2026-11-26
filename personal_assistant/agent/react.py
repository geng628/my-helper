"""Layer 2 通用 ReAct 核心循环（对标 pi-agent-core）：复用 ai 层流式事件并产出 ToolStep / TurnEnd，不含业务与统计逻辑。"""
from __future__ import annotations

from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, field
from time import perf_counter

from personal_assistant.agent.tools import ToolRegistry
from personal_assistant.ai.client import LLMClient
from personal_assistant.ai.types import StreamDone, TextDelta, TokenUsage


# 单步工具执行完成事件：携带工具名、参数、观察结果与执行耗时
@dataclass(frozen=True)
class ToolStep:
    tool: str
    arguments: str
    observation: str
    duration_ms: float = 0.0


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


AgentEvent = TextDelta | StreamDone | ToolStep | TurnEnd
RuntimeEvent = AgentEvent | TaskEvent


# 从模型返回的消息字典中提取纯文本回复，为空时使用默认提示文案
def _extract_reply(message: dict[str, object]) -> str:
    content = str(message.get("content") or "").strip()
    return content or "模型没有返回回复，请重试。"


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


# 执行流式通用 ReAct 主循环：透传模型流式增量与 StreamDone，产出 ToolStep / TurnEnd 并仅在无 tool_calls 时自然终止
def iter_react(
    llm: LLMClient,
    messages: list[dict[str, object]],
    tools: ToolRegistry,
) -> Iterator[AgentEvent]:
    t_turn = perf_counter()
    tool_defs = tools.definitions() if tools.names else None

    while True:
        # 1. Reason：流式调用模型，实时吐字或请求工具调用
        assistant_msg = yield from _stream_llm_round(llm, messages, tool_defs)
        tool_calls = assistant_msg.get("tool_calls") or []
        if not tool_calls:
            reply = _extract_reply(assistant_msg)
            messages.append({"role": "assistant", "content": reply})
            yield TurnEnd(reply=reply, total_ms=round((perf_counter() - t_turn) * 1000, 3))
            return

        # 2. Act & Observe：依次执行模型请求的工具并计时，结果写回上下文
        messages.append(assistant_msg)
        for call in tool_calls:
            fn = call.get("function") or {}
            name = str(fn.get("name") or "")
            args = str(fn.get("arguments") or "{}")
            t_tool = perf_counter()
            observation = tools.execute(name, args)
            duration_ms = round((perf_counter() - t_tool) * 1000, 3)
            yield ToolStep(tool=name, arguments=args, observation=observation, duration_ms=duration_ms)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(call.get("id") or ""),
                    "content": observation,
                }
            )


# 执行通用 ReAct 主循环：可选订阅强类型 AgentEvent，并在模型停止调用工具时返回最终自然语言回答
def run_react(
    llm: LLMClient,
    messages: list[dict[str, object]],
    tools: ToolRegistry,
    *,
    on_event: Callable[[AgentEvent], None] | None = None,
) -> str:
    reply = ""
    for event in iter_react(llm, messages, tools):
        if on_event is not None:
            on_event(event)
        if isinstance(event, TurnEnd):
            reply = event.reply
    return reply
