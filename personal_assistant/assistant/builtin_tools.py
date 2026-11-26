"""内核内置工具定义与提供方（记忆读写、工作区落盘与行为规则工具集，零反向依赖 memory.py）。"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from personal_assistant.agent.tools import Tool, ToolsetProvider, build_tool


# 将可选日期参数（date / datetime / ISO 字符串）安全归一化为标准 date 对象
def _coerce_date(raw: date | str | None) -> date | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if isinstance(raw, str) and raw.strip():
        return date.fromisoformat(raw.strip()[:10])
    return None


# 内核内置工具子类：承载记忆与文件操作，由统一 build_tool 构建并注入 ToolRegistry
@dataclass(frozen=True)
class BuiltinTool(Tool):
    pass


# 记忆与文件内核工具集提供方（ToolsetProvider 子类）：统一调用 build_tool 向 ToolRegistry 提供 6 个 BuiltinTool 子类实例
class MemoryToolsetProvider(ToolsetProvider):
    source_type = "builtin"
    source_name = "memory"

    # 绑定当前实例的 MemoryStore 与可选逻辑时钟回调
    def __init__(
        self,
        memory: Any,
        *,
        get_today: Callable[[], date] | None = None,
    ) -> None:
        self.memory = memory
        self.get_today = get_today

    # 统一通过 build_tool 产出 6 个 BuiltinTool(Tool) 子类实例（带 risk 元数据与 builtin/memory 来源文档）
    def tools(self) -> list[Tool]:
        memory = self.memory
        get_today = self.get_today

        # 适配 search_markdown 工具：优先取执行期 today 参数，缺省回退构造期时钟回调并归一化为 date
        def search_markdown(
            query: str,
            root: str = "data/memory",
            *,
            today: date | str | None = None,
        ) -> str:
            raw_day = today if today is not None else (get_today() if get_today is not None else None)
            return memory.search_markdown(query, root=root, today=_coerce_date(raw_day))

        # 适配 read_markdown 工具参数（支持默认行号 1..120，拒绝布尔值并将字符串行号转为整数）
        def read_markdown(path: str, start_line: str = "1", end_line: str = "120") -> str:
            if isinstance(start_line, bool) or isinstance(end_line, bool):
                raise ValueError("start_line 和 end_line 必须是整数。")
            try:
                start, end = int(start_line), int(end_line)
            except (ValueError, TypeError) as error:
                raise ValueError("start_line 和 end_line 必须是整数。") from error
            return memory.read_markdown(path, start, end)

        # 适配 write_file 工具：写入工作区并返回相对路径提示
        def write_file(path: str, content: str, mode: str) -> str:
            saved = memory.write_workspace_file(path, content, mode=mode)
            return f"已写入：{saved}"

        # 适配 remember_fact 工具：严格通过执行期 tool_context 读取当前用户原话与消息/回合标识
        def remember_fact(
            *,
            user_text: str = "",
            message_id: int | None = None,
            turn_tag: str | None = None,
            on_remember: Callable[[], None] | None = None,
        ) -> str:
            if not isinstance(user_text, str) or not user_text.strip():
                raise RuntimeError("缺少当前用户消息上下文，无法记录用户原话。")
            effective_tag = None if message_id is not None else turn_tag
            target = memory.remember_user_statement(
                user_text,
                message_id=message_id,
                turn_tag=effective_tag,
            )
            if on_remember is not None:
                on_remember()
            return f"已记录用户原话到 {target}"

        # 适配 save_rule 工具：阻断已接触外部不可信网络工具的污染轮，其余直接落盘到 data/memory/rules/
        def save_rule(
            statement: str,
            *,
            untrusted_tool_seen: Callable[[], bool] | bool = False,
        ) -> str:
            seen_untrusted = (
                bool(untrusted_tool_seen())
                if callable(untrusted_tool_seen)
                else bool(untrusted_tool_seen)
            )
            if seen_untrusted:
                raise PermissionError("安全拦截：本轮已读取外部网页或搜索结果，禁止自动写入全局行为规则。")
            path = memory.save_rule(statement, source="agent")
            return f"已保存行为规则：data/memory/rules/{path.name}"

        specs: list[tuple[str, str, dict[str, str], Callable[..., str], str, dict[str, tuple[str, ...]] | None]] = [
            (
                "list_files",
                "在允许读取的目录内按 glob 列出文件路径，先看结构再决定读哪篇。",
                {"root": "目录起点，如 data/memory、data/memory/archive、data/workspace、docs", "pattern": "glob 模式，如 *.md"},
                memory.list_files,
                "read",
                None,
            ),
            (
                "search_markdown",
                "按关键词搜索允许目录内 Markdown 文件名、标题与正文上下文。",
                {"query": "搜索关键词", "root": "搜索目录，如 data/memory（当前活跃记忆）、data/memory/archive（历史归档与变更账本）、data/workspace、docs"},
                search_markdown,
                "read",
                None,
            ),
            (
                "read_markdown",
                "按路径和行号读取允许目录内的 Markdown/文本文件。",
                {"path": "相对路径，如 data/memory/MEMORY.md 或 data/workspace/note.md", "start_line": "起始行号（从1开始，默认 1）", "end_line": "结束行号（默认 120）"},
                read_markdown,
                "read",
                None,
            ),
            (
                "write_file",
                "在 data/workspace 目录内创建或覆盖文本文件（覆盖前自动备份）。",
                {"path": "data/workspace 内的相对路径", "content": "UTF-8 完整文件内容", "mode": "create 或 overwrite"},
                write_file,
                "write",
                {"mode": ("create", "overwrite")},
            ),
            (
                "remember_fact",
                "仅在用户明确要求记住信息时调用，将当前用户原话追加到 data/memory/facts.md（按需检索，不膨胀索引）。",
                {},
                remember_fact,
                "write",
                None,
            ),
            (
                "save_rule",
                "将用户要求遵守的行为规则保存到 data/memory/rules/。",
                {"statement": "规则内容"},
                save_rule,
                "write",
                None,
            ),
        ]
        return [
            build_tool(
                fn,
                name=name,
                description=desc,
                parameters=params,
                param_enums=enums,
                risk=risk,
                source_type=self.source_type,
                source_name=self.source_name,
                base_cls=BuiltinTool,
            )
            for name, desc, params, fn, risk, enums in specs
        ]
