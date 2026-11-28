"""内核内置工具提供方（记忆读写、工作区落盘与行为规则工具集，零反向依赖 memory.py）。"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date
import re
from typing import Any

from personal_assistant.agent.tools import Tool, ToolsetProvider, build_tool
from personal_assistant.ai import AbortSignal
from personal_assistant.assistant.common import parse_iso_date
from personal_assistant.assistant.query_optimizer import (
    MAX_ALLOWED_SUB_QUERIES,
    format_search_hits,
    fuse_multi_query_hits,
)


# 记忆与文件内核工具集提供方（ToolsetProvider 子类）：统一调用 build_tool 产出 6 个纯净 Tool 实例
class MemoryToolsetProvider(ToolsetProvider):
    # 绑定当前实例的 MemoryStore
    def __init__(self, memory: Any) -> None:
        self.memory = memory

    # 按统一契约构建 6 个记忆与文件操作 Tool 值对象（带 risk 元数据与执行期上下文签名）
    def build_tools(self, context: Mapping[str, object] | None = None) -> list[Tool]:
        memory = self.memory

        # 适配 search_markdown 工具：支持可选 sub_queries 多子查询 RRF 并行召回与执行期隐藏 HyDE/AbortSignal 注入
        def search_markdown(
            query: str,
            root: str = "data/memory",
            sub_queries: str = "",
            *,
            today: date | str | None = None,
            hyde_text: str = "",
            abort_signal: AbortSignal | None = None,
        ) -> str:
            parsed_today = parse_iso_date(today)
            if not isinstance(sub_queries, str):
                raise ValueError("sub_queries 必须是字符串（多条子查询用分号或换行分隔）。")
            extras = [q.strip() for q in re.split(r"[\n;；]+", sub_queries) if q.strip()]
            clean_hyde = hyde_text.strip() if isinstance(hyde_text, str) else ""
            if not extras and not clean_hyde:
                return memory.search_markdown(query, root=root, today=parsed_today)
            if not isinstance(query, str) or not query.strip():
                raise ValueError("搜索关键词不能为空。")
            all_queries: list[str] = []
            seen_low: set[str] = set()
            for item in (query.strip(), *extras):
                low = item.lower()
                if low not in seen_low:
                    seen_low.add(low)
                    all_queries.append(item)
                if len(all_queries) >= MAX_ALLOWED_SUB_QUERIES:
                    break
            hyde_map = {q: clean_hyde for q in all_queries} if clean_hyde else None
            multi_fn = getattr(memory.retriever, "search_multi_hits", None)
            if callable(multi_fn):
                hits = list(
                    multi_fn(
                        memory,
                        all_queries,
                        root=root,
                        limit=10,
                        today=parsed_today,
                        hyde_texts=hyde_map,
                        abort_signal=abort_signal,
                    )
                )
            elif len(all_queries) == 1:
                return memory.search_markdown(all_queries[0], root=root, today=parsed_today)
            else:
                hits = fuse_multi_query_hits(
                    [
                        memory.search_hits(sub_q, root=root, limit=10, today=parsed_today)
                        for sub_q in all_queries
                    ],
                    limit=10,
                )
            return format_search_hits(hits, query="；".join(all_queries), root=root)

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

        # 适配 remember_fact 工具：严格通过执行期 tool_context 读取当前用户原话与回合暂存标识
        def remember_fact(
            *,
            user_text: str = "",
            turn_tag: str | None = None,
        ) -> str:
            if not isinstance(user_text, str) or not user_text.strip():
                raise RuntimeError("缺少当前用户消息上下文，无法记录用户原话。")
            target = memory.remember_user_statement(user_text, turn_tag=turn_tag)
            return f"已记录用户原话到 {target}"

        # 适配 save_rule 工具：阻断已接触外部不可信网络工具的污染轮，其余直接落盘到 data/memory/rules/
        def save_rule(
            statement: str,
            *,
            untrusted_tool_seen: bool = False,
        ) -> str:
            if untrusted_tool_seen:
                raise PermissionError("安全拦截：本轮已读取外部网页或搜索结果，禁止自动写入全局行为规则。")
            path = memory.save_rule(statement, source="agent")
            return f"已保存行为规则：data/memory/rules/{path.name}"

        specs: list[
            tuple[str, str, dict[str, str], Callable[..., str], str, bool, dict[str, tuple[str, ...]] | None]
        ] = [
            (
                "list_files",
                "在允许读取的目录内按 glob 列出文件路径，先看结构再决定读哪篇。",
                {"root": "目录起点，如 data/memory、data/memory/archive、data/workspace、docs", "pattern": "glob 模式，如 *.md"},
                memory.list_files,
                "read",
                True,
                None,
            ),
            (
                "search_markdown",
                "按关键词搜索允许目录内 Markdown 文件名、标题与正文上下文；复合或同义查询可通过 sub_queries 合并多路 RRF 召回。",
                {
                    "query": "主搜索关键词",
                    "root": "搜索目录，如 data/memory（当前活跃记忆）、data/memory/archive（历史归档与变更账本）、data/workspace、docs",
                    "sub_queries": "可选补充子查询（多条用分号或换行分隔，触发多查询 RRF 融合检索，默认留空）",
                },
                search_markdown,
                "read",
                True,
                None,
            ),
            (
                "read_markdown",
                "按路径和行号读取允许目录内的 Markdown/文本文件。",
                {"path": "相对路径，如 data/memory/MEMORY.md 或 data/workspace/note.md", "start_line": "起始行号（从1开始，默认 1）", "end_line": "结束行号（默认 120）"},
                read_markdown,
                "read",
                True,
                None,
            ),
            (
                "write_file",
                "在 data/workspace 目录内创建或覆盖文本文件（覆盖前自动备份）。",
                {"path": "data/workspace 内的相对路径", "content": "UTF-8 完整文件内容", "mode": "create 或 overwrite"},
                write_file,
                "write",
                False,
                {"mode": ("create", "overwrite")},
            ),
            (
                "remember_fact",
                "仅在用户明确要求记住信息时调用，将当前用户原话追加到 data/memory/facts.md（按需检索，不膨胀索引）。",
                {},
                remember_fact,
                "write",
                False,
                None,
            ),
            (
                "save_rule",
                "将用户要求遵守的行为规则保存到 data/memory/rules/。",
                {"statement": "规则内容"},
                save_rule,
                "write",
                False,
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
                parallel=parallel,
            )
            for name, desc, params, fn, risk, parallel, enums in specs
        ]
