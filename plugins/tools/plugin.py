"""工具扩展主板插件（plugins/tools/plugin.py）：将 plugins/tools/ 目录下的全部 @tool 模块统一接入内层 Tool 子协议。"""
from __future__ import annotations

from pathlib import Path

from personal_assistant.agent.plugins import Plugin, _import_plugin_module
from personal_assistant.agent.tools import is_tool


# 工具扩展主板插件（与 FTS5RetrieverPlugin、ModelsDSLPlugin 同级，唯一负责内层 Tool 协议插槽）
class ToolsPlugin(Plugin):
    plugin_id = "tools"

    # 统一初始化生命周期：自动扫描本目录下的全部工具模块（custom_tools / shell_tools / web_tools 等）并注册进 ToolRegistry
    def initialize(self) -> None:
        tools_dir = Path(__file__).resolve().parent
        for py_file in sorted(tools_dir.glob("*.py")):
            if py_file.name == "plugin.py" or py_file.name.startswith("_"):
                continue
            module = _import_plugin_module(py_file)
            for value in vars(module).values():
                if is_tool(value):
                    self.context.register_tool(value, source_name=py_file.stem)
