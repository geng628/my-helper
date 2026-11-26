"""工具扩展主板插件（plugins/tools/plugin.py）：将 plugins/tools/ 目录下的全部 @tool 模块统一接入内层 Tool 子协议。"""
from __future__ import annotations

from pathlib import Path

from personal_assistant.agent.plugins import Plugin


# 工具扩展主板插件（与 FTS5RetrieverPlugin、ModelsDSLPlugin 同级，唯一负责内层 Tool 协议插槽）
class ToolsPlugin(Plugin):
    plugin_id = "tools"

    # 统一初始化生命周期：自动扫描本目录下的全部工具模块并注册进主板上下文
    def initialize(self) -> None:
        tools_dir = Path(__file__).resolve().parent
        for py_file in sorted(tools_dir.glob("*.py")):
            if py_file.name == "plugin.py" or py_file.name.startswith("_"):
                continue
            self.context.register_tools_from_module(py_file)
