"""Layer 3 公共导出接口（按需延迟导入，避免导入轻量工具子模块时拖入全套应用编排）。"""
from __future__ import annotations

import importlib

_EXPORT_MAP: dict[str, tuple[str, str]] = {
    "AssistantApp": ("personal_assistant.assistant.app", "AssistantApp"),
    "build_app": ("personal_assistant.assistant.app", "build_app"),
    "MemoryToolsetProvider": ("personal_assistant.assistant.builtin_tools", "MemoryToolsetProvider"),
    "CommandSpec": ("personal_assistant.assistant.commands", "CommandSpec"),
    "ParamSpec": ("personal_assistant.assistant.commands", "ParamSpec"),
    "ToolsConfig": ("personal_assistant.assistant.commands", "ToolsConfig"),
    "REPO_ROOT": ("personal_assistant.assistant.common", "REPO_ROOT"),
    "load_dotenv": ("personal_assistant.assistant.common", "load_dotenv"),
    "truncate_utf8": ("personal_assistant.assistant.common", "truncate_utf8"),
    "BackgroundConsolidator": ("personal_assistant.assistant.consolidator", "BackgroundConsolidator"),
    "ConsolidationPolicy": ("personal_assistant.assistant.consolidator", "ConsolidationPolicy"),
    "ConsolidationResult": ("personal_assistant.assistant.consolidator", "ConsolidationResult"),
    "MemoryConsolidator": ("personal_assistant.assistant.consolidator", "MemoryConsolidator"),
    "StateStore": ("personal_assistant.assistant.db", "StateStore"),
    "FilesPolicy": ("personal_assistant.assistant.files_policy", "FilesPolicy"),
    "LinearRetriever": ("personal_assistant.assistant.memory", "LinearRetriever"),
    "MemoryStore": ("personal_assistant.assistant.memory", "MemoryStore"),
    "SearchHit": ("personal_assistant.assistant.memory", "SearchHit"),
    "is_line_expired": ("personal_assistant.assistant.memory_lifecycle", "is_line_expired"),
    "SYSTEM_PROMPT": ("personal_assistant.assistant.prompts", "SYSTEM_PROMPT"),
    "format_system_prompt": ("personal_assistant.assistant.prompts", "format_system_prompt"),
    "format_user_envelope": ("personal_assistant.assistant.prompts", "format_user_envelope"),
    "tokenize_text": ("personal_assistant.assistant.text_search", "tokenize_text"),
}

__all__ = sorted(_EXPORT_MAP)


# 按需延迟导入并缓存 Layer 3 公开符号，保持导入底座模块时零耦合开销
def __getattr__(name: str) -> object:
    target = _EXPORT_MAP.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value


# 返回当前模块已加载属性与延迟公开符号全集
def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORT_MAP})
