"""用户可自由编辑的通用工具实现（plugins/tools/custom_tools.py）：仅声明 @tool 函数，统一由 ToolsPlugin 装配。"""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from personal_assistant.agent.tools import tool


# 插件工具：返回当前北京时间字符串（若宿主注入逻辑日期 today 则对齐该日期并保留当前时分秒）
@tool(
    description="获取当前北京时间（YYYY-MM-DD HH:MM:SS）。",
    parallel=True,
)
def get_current_time(*, today: date | str | None = None) -> str:
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if isinstance(today, datetime):
        ref_day = today.date()
    elif isinstance(today, date):
        ref_day = today
    elif isinstance(today, str):
        try:
            ref_day = date.fromisoformat(today.strip())
        except ValueError:
            ref_day = now.date()
    else:
        ref_day = now.date()
    return f"{ref_day.isoformat()} {now.strftime('%H:%M:%S')} (Asia/Shanghai)"


# 插件工具：计算两个整数的和并返回字符串结果（默认未加入白名单，按需在 tools.json 开启）
@tool(
    description="计算两个整数之和。",
    parameters={"a": "第一个整数", "b": "第二个整数"},
    parallel=True,
)
def add_numbers(a: str, b: str) -> str:
    return str(int(a) + int(b))
