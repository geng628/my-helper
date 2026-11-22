"""Layer 2 公共导出接口（对标 pi-agent-core）：仅依赖 ai 层，可脱离个人助手业务独立运行通用 ReAct Agent。"""
from personal_assistant.agent.plugins import (
    ConfigCompiler,
    PluginManager,
    RetrieverFactory,
    load_plugin_compilers,
    load_plugin_providers,
    load_plugin_retrievers,
    load_plugin_tools,
    load_plugins,
)
from personal_assistant.agent.react import MAX_TOOL_ROUNDS, run_react
from personal_assistant.agent.tools import (
    Tool,
    ToolDocument,
    ToolMetadata,
    ToolRegistry,
    ToolsetProvider,
    build_dynamic_plugin_tool,
    tool,
)

__all__ = [
    "ConfigCompiler",
    "MAX_TOOL_ROUNDS",
    "PluginManager",
    "RetrieverFactory",
    "Tool",
    "ToolDocument",
    "ToolMetadata",
    "ToolRegistry",
    "ToolsetProvider",
    "build_dynamic_plugin_tool",
    "load_plugin_compilers",
    "load_plugin_providers",
    "load_plugin_retrievers",
    "load_plugin_tools",
    "load_plugins",
    "run_react",
    "tool",
]
