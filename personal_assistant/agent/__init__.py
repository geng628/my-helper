"""Layer 2 公共导出接口（对标 pi-agent-core）：仅依赖 ai 层，提供统一插件主板（Plugin + PluginContext）与内层极简子协议。"""
from personal_assistant.agent.plugins import (
    ConfigCompilerProtocol,
    LLMProviderProtocol,
    Plugin,
    PluginContext,
    PluginManager,
    RetrieverFactory,
    RetrieverProtocol,
)
from personal_assistant.agent.react import MAX_TOOL_ROUNDS, run_react
from personal_assistant.agent.tools import (
    Tool,
    ToolDocument,
    ToolMetadata,
    ToolRegistry,
    ToolsetProvider,
    build_tool,
    tool,
)

__all__ = [
    "ConfigCompilerProtocol",
    "LLMProviderProtocol",
    "MAX_TOOL_ROUNDS",
    "Plugin",
    "PluginContext",
    "PluginManager",
    "RetrieverFactory",
    "RetrieverProtocol",
    "Tool",
    "ToolDocument",
    "ToolMetadata",
    "ToolRegistry",
    "ToolsetProvider",
    "build_tool",
    "run_react",
    "tool",
]
