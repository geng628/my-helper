"""最小 ReAct 循环（Reason 推理 -> Act 调用工具 -> Observe 观察结果）。"""
from __future__ import annotations

from collections.abc import Callable

from personal_assistant.llm import LLMClient
from personal_assistant.tools import ToolRegistry

MAX_TOOL_ROUNDS = 4

SYSTEM_PROMPT = (
    "你是用户的个人助手，用简洁自然的中文交流。\n"
    "能力严格以本轮提供的工具为准；需要查证事实或借助工具时先调用工具，"
    "拿到足够证据后直接回答，不要编造未观测到的结果。\n"
    "系统提示中的 MEMORY.md 仅作记忆目录索引，详细事实与主题文件存放在 data/memory/，"
    "需要回忆往事或核对细节时请按需调用 search_markdown 或 read_markdown 查阅。"
)


# 从模型返回的消息字典中提取纯文本回复，为空时使用兜底文案
def _extract_reply(message: dict[str, object], fallback: str = "模型没有返回回复，请重试。") -> str:
    content = str(message.get("content") or "").strip()
    return content or fallback


# 执行有轮次上限的 ReAct 主循环：原地追加对话与工具消息，并返回最终自然语言回答
def run_react(
    llm: LLMClient,
    messages: list[dict[str, object]],
    tools: ToolRegistry,
    *,
    max_rounds: int = MAX_TOOL_ROUNDS,
    on_step: Callable[[str, str, str], None] | None = None,
) -> str:
    tool_defs = tools.definitions() if tools.names else None

    for _ in range(max_rounds):
        # 1. Reason：让模型决定直接回答，还是发起 tool_calls
        assistant_msg = llm.respond(messages, tool_defs)
        tool_calls = assistant_msg.get("tool_calls") or []
        if not tool_calls:
            reply = _extract_reply(assistant_msg)
            messages.append({"role": "assistant", "content": reply})
            return reply

        # 2. Act & Observe：依次执行模型请求的工具，并将结果作为 tool 消息写回上下文
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

    # 3. 达到工具调用轮次上限：不再提供工具，强制模型基于已有观察总结作答
    if messages and messages[-1].get("role") == "tool":
        messages[-1]["content"] = (
            str(messages[-1]["content"])
            + "\n\n【系统提示：已达到工具调用轮次上限，请直接基于以上观察结果给出完整回答。】"
        )
    final_msg = llm.respond(messages, tools=None)
    reply = _extract_reply(final_msg, fallback="本轮工具调用已达到上限，请缩小问题范围后再试。")
    messages.append({"role": "assistant", "content": reply})
    return reply
