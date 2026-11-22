"""自定义工具插件（继承统一主板 Plugin 基类）：通过 self.context.register_tool 注册内层 Tool 子协议实例。"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from personal_assistant.agent.plugins import Plugin
from personal_assistant.agent.tools import build_tool, tool


# 插件工具：返回当前北京时间字符串
@tool(
    description="获取当前北京时间（YYYY-MM-DD HH:MM:SS）。",
)
def get_current_time() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S (Asia/Shanghai)")


# 插件工具：计算两个整数的和并返回字符串结果（默认未加入白名单，按需在 tools.json 开启）
@tool(
    description="计算两个整数之和。",
    parameters={"a": "第一个整数", "b": "第二个整数"},
)
def add_numbers(a: str, b: str) -> str:
    return str(int(a) + int(b))


# 自定义工具主板插件（外层继承 Plugin ABC，内层向 tool_registry 插槽注册 Tool 子协议）
class CustomToolsPlugin(Plugin):
    plugin_id = "custom_tools"

    # 统一初始化生命周期：将本插件内的工具通过 build_tool 注册进 self.context.tool_registry
    def initialize(self) -> None:
        for fn in (get_current_time, add_numbers):
            spec = getattr(fn, "__agent_tool__")
            self.context.register_tool(
                build_tool(
                    fn,
                    name=spec.name,
                    description=spec.description,
                    parameters=spec.parameters,
                    risk="read",
                    source_type="plugin",
                    source_name=self.plugin_id,
                )
            )
