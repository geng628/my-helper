"""可自由编辑的工具插件文件（无需修改 personal_assistant/ 任何核心代码）。

如何新增自定义工具：
1. 在本文件（或在 plugins/ 下新建任意 .py 文件）写一个普通函数并加上 @tool(...)；
2. 把工具名加入 config/tools.json 的 enabled_tools 白名单即可立即生效；
3. 想临时禁用某个工具，只需把它从 enabled_tools 里删掉即可。
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from personal_assistant.tools import tool


@tool(
    description="获取当前北京时间（YYYY-MM-DD HH:MM:SS）。",
)
def get_current_time() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S (Asia/Shanghai)")


@tool(
    description="计算两个整数之和。",
    parameters={"a": "第一个整数", "b": "第二个整数"},
)
def add_numbers(a: str, b: str) -> str:
    return str(int(a) + int(b))
