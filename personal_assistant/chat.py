"""Minimal ReAct loop (Reason -> Act -> Observe) with native tool calls."""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from personal_assistant.llm import LLMClient

MAX_TOOL_ROUNDS = 4

SYSTEM_PROMPT = (
    "你是用户的个人助手，用简洁自然的中文交流。\n"
    "能力严格以本轮提供的工具为准；需要查询事实或借助工具时主动调用工具，"
    "拿到足够证据后直接给出清晰回答，不要编造未观测到的结果。"
)


@dataclass(frozen=True)
class ToolResult:
    text: str
    error_type: str = ""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, str]
    handler: Callable[..., ToolResult | str]


class ToolRegistry:
    """Publish function-calling schemas and convert execution errors into observations."""

    def __init__(self, *tools: Tool) -> None:
        self._tools = {tool.name: tool for tool in tools}
        if len(self._tools) != len(tools):
            raise ValueError("工具名称不能重复。")

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def definitions(self) -> list[dict[str, object]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            name: {"type": "string", "description": desc}
                            for name, desc in tool.parameters.items()
                        },
                        "required": list(tool.parameters),
                        "additionalProperties": False,
                    },
                },
            }
            for tool in self._tools.values()
        ]

    def execute(self, name: str, raw_arguments: str) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            available = "、".join(self._tools) or "（无）"
            return ToolResult(f"未知工具：{name}。可用工具：{available}", error_type="UnknownTool")
        try:
            values = json.loads(raw_arguments or "{}")
        except (json.JSONDecodeError, TypeError) as error:
            return ToolResult(f"工具参数 JSON 无效：{error}", error_type="InvalidArguments")
        if not isinstance(values, dict):
            return ToolResult(f"工具参数无效：{name} 需要 JSON 对象。", error_type="InvalidArguments")
        if set(values) != set(tool.parameters) or any(
            not isinstance(v, str) or not v.strip() for v in values.values()
        ):
            if not tool.parameters:
                return ToolResult(f"工具参数无效：{name} 不接受参数，请传空对象 {{}}。", error_type="InvalidArguments")
            expected = ", ".join(tool.parameters)
            return ToolResult(f"工具参数无效：{name} 需要非空字符串参数 {expected}。", error_type="InvalidArguments")
        try:
            output = tool.handler(**values)
        except Exception as error:
            return ToolResult(f"工具执行失败：{error}", error_type=type(error).__name__)
        return output if isinstance(output, ToolResult) else ToolResult(str(output))


@dataclass(frozen=True)
class StepTrace:
    round: int
    tool: str
    arguments: str
    observation: str
    is_error: bool


@dataclass(frozen=True)
class TurnResult:
    reply: str
    messages: list[dict[str, object]]
    traces: list[StepTrace]


def run_react_loop(
    llm: LLMClient,
    messages: list[dict[str, object]],
    tools: ToolRegistry,
    *,
    max_rounds: int = MAX_TOOL_ROUNDS,
    on_step: Callable[[StepTrace], None] | None = None,
) -> TurnResult:
    """Run a bounded Reason -> Act -> Observe loop until the model returns a final answer."""
    history = list(messages)
    traces: list[StepTrace] = []

    for turn in range(max_rounds + 1):
        # Reason: ask the model for either a tool call or a final answer.
        tool_defs = tools.definitions() if (turn < max_rounds and tools.names) else None
        assistant_msg = llm.respond(history, tool_defs)
        calls = assistant_msg.get("tool_calls") or []

        if not calls:
            reply = str(assistant_msg.get("content") or "").strip() or "模型没有返回回复，请重试。"
            history.append({"role": "assistant", "content": reply})
            return TurnResult(reply=reply, messages=history, traces=traces)

        if turn == max_rounds:
            reply = "本轮工具调用已达到上限，请缩小问题范围后再试。"
            history.append({"role": "assistant", "content": reply})
            return TurnResult(reply=reply, messages=history, traces=traces)

        # Act & Observe: run requested tools and append each observation back to context.
        history.append(assistant_msg)
        for call in calls:
            fn = call.get("function") or {}
            tool_name = str(fn.get("name") or "")
            raw_args = str(fn.get("arguments") or "{}")
            result = tools.execute(tool_name, raw_args)

            trace = StepTrace(
                round=turn + 1,
                tool=tool_name,
                arguments=raw_args,
                observation=result.text,
                is_error=bool(result.error_type),
            )
            traces.append(trace)
            if on_step is not None:
                on_step(trace)

            history.append(
                {
                    "role": "tool",
                    "tool_call_id": str(call.get("id") or ""),
                    "content": result.text,
                }
            )

        if turn == max_rounds - 1 and history and history[-1].get("role") == "tool":
            history[-1]["content"] = (
                str(history[-1]["content"])
                + "\n\n【系统提示：已达到工具调用轮次上限，下一轮不再提供工具。请直接基于以上观察结果给出完整回答。】"
            )

    reply = "本轮工具调用已达到上限，请缩小问题范围后再试。"
    history.append({"role": "assistant", "content": reply})
    return TurnResult(reply=reply, messages=history, traces=traces)
