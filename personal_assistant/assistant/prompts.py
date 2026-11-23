"""Layer 3 个人助手业务系统提示词定义与组装（从通用 ReAct 引擎中剥离）。"""
from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from personal_assistant.assistant.common import format_date_with_weekday

if TYPE_CHECKING:
    from personal_assistant.assistant.memory import MemoryStore

SYSTEM_PROMPT = (
    "你是用户的个人助手，用简洁自然的中文交流。\n"
    "能力以本轮提供的工具为准；查证事实时先调工具，拿到证据后直接回答，不要编造未观测到的结果。\n"
    "MEMORY.md 仅作目录索引，详细事实在 data/memory/；"
    "回忆往事、核对细节或用户寻求建议/推荐/规划且索引有相关主题时，先调 search_markdown 或 read_markdown；"
    "记忆未记细节（如过往推荐或原话）时，再调 search_chat_history 查历史对话。\n"
    "用户更新偏好或事实时直接确认即可，无需写工作区或让用户改 data/memory/，后台会自动沉淀。"
)


# 组装个人助手系统提示词：静态指令与规则在前、目录索引与日期在后，最大化服务端 KV Cache 命中率
def format_system_prompt(
    memory: "MemoryStore",
    *,
    tool_names: list[str] | None = None,
    today: date | None = None,
) -> str:
    parts = [SYSTEM_PROMPT]
    if tool_names is not None:
        tool_list = "、".join(tool_names) or "（无）"
        parts.append(f"【可用白名单工具】\n{tool_list}")
    rules = memory.read_rules_summary()
    if rules:
        parts.append(f"【行为规则 (data/memory/rules/)】\n{rules}")
    index = memory.read_memory_index()
    if index:
        parts.append(f"【记忆索引 (data/memory/MEMORY.md)】\n{index}")
    ref_day = today or datetime.now(ZoneInfo("Asia/Shanghai")).date()
    parts.append(f"今天是 {format_date_with_weekday(ref_day)}。")
    return "\n\n".join(parts)
