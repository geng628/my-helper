"""Minimal ReAct loop (Reason -> Act -> Observe)."""
from __future__ import annotations

from collections.abc import Callable

from personal_assistant.llm import LLMClient
from personal_assistant.tools import ToolRegistry

MAX_TOOL_ROUNDS = 4

SYSTEM_PROMPT = (
    "你是用户的个人助手，用简洁自然的中文交流。\n"
    "能力严格以本轮提供的工具为准；需要查证事实或借助工具时先调用工具，"
    "拿到足够证据后直接回答，不要编造未观测到的结果。"
)


def _extract_reply(message: dict[str, object], fallback: str = "模型没有返回回复，请重试。") -> str:
    content = str(message.get("content") or "").strip()
    return content or fallback


def run_react(
    llm: LLMClient,
    messages: list[dict[str, object]],
    tools: ToolRegistry,
    *,
    max_rounds: int = MAX_TOOL_ROUNDS,
    on_step: Callable[[str, str, str], None] | None = None,
) -> str:
    """Run a bounded Reason -> Act -> Observe loop, mutating `messages` in place."""
    tool_defs = tools.definitions() if tools.names else None

    for _ in range(max_rounds):
        # 1. Reason: let the model decide whether to call tools or answer directly.
        assistant_msg = llm.respond(messages, tool_defs)
        tool_calls = assistant_msg.get("tool_calls") or []
        if not tool_calls:
            reply = _extract_reply(assistant_msg)
            messages.append({"role": "assistant", "content": reply})
            return reply

        # 2. Act & Observe: run requested tools and append observations to context.
        messages.append(assistant_msg)
        for call in tool_calls:
            fn = call.get("function") or {}
            name = str(fn.get("name") or "")
            args = str(fn.get("arguments") or "{}")
            observation = tools.execute(name, args)
            if on_step is not None:
                on_step(name, args, observation)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(call.get("id") or ""),
                    "content": observation,
                }
            )

    # 3. Max rounds reached: drop tools and force the model to summarize and stop.
    if messages and messages[-1].get("role") == "tool":
        messages[-1]["content"] = (
            str(messages[-1]["content"])
            + "\n\n【系统提示：已达到工具调用轮次上限，请直接基于以上观察结果给出完整回答。】"
        )
    final_msg = llm.respond(messages, tools=None)
    reply = _extract_reply(final_msg, fallback="本轮工具调用已达到上限，请缩小问题范围后再试。")
    messages.append({"role": "assistant", "content": reply})
    return reply
