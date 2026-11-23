"""Layer 3 公共导出接口（对标 pi-coding-agent）：整合个人记忆、SQLite 历史、后台沉淀与白名单工具编排。"""
from personal_assistant.assistant.app import AssistantApp, build_app
from personal_assistant.assistant.commands import CommandSpec, ParamSpec, ToolsConfig
from personal_assistant.assistant.common import REPO_ROOT, load_dotenv, truncate_utf8
from personal_assistant.assistant.consolidator import (
    BackgroundConsolidator,
    ConsolidationPolicy,
    ConsolidationResult,
    MemoryConsolidator,
)
from personal_assistant.assistant.db import StateStore
from personal_assistant.assistant.memory import FilesPolicy, MemoryStore, MemoryToolsetProvider, SearchHit
from personal_assistant.assistant.memory_lifecycle import is_line_expired
from personal_assistant.assistant.prompts import SYSTEM_PROMPT, format_system_prompt

__all__ = [
    "AssistantApp",
    "BackgroundConsolidator",
    "CommandSpec",
    "ConsolidationPolicy",
    "ConsolidationResult",
    "FilesPolicy",
    "MemoryConsolidator",
    "MemoryStore",
    "MemoryToolsetProvider",
    "ParamSpec",
    "REPO_ROOT",
    "SYSTEM_PROMPT",
    "SearchHit",
    "StateStore",
    "ToolsConfig",
    "build_app",
    "format_system_prompt",
    "is_line_expired",
    "load_dotenv",
    "truncate_utf8",
]
