"""Minimal tool definition and function-calling registry."""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, str]
    handler: Callable[..., str]


class ToolRegistry:
    """Publish OpenAI tool schemas and return tool errors as plain observations."""

    def __init__(self, *tools: Tool) -> None:
        self._tools = {tool.name: tool for tool in tools}
        if len(self._tools) != len(tools):
            raise ValueError("工具名称不能重复。")

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def definitions(self) -> list[dict[str, object]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            name: {"type": "string", "description": desc}
                            for name, desc in tool.parameters.items()
                        },
                        "required": list(tool.parameters),
                        "additionalProperties": False,
                    },
                },
            }
            for tool in self._tools.values()
        ]

    def execute(self, name: str, raw_arguments: str) -> str:
        tool = self._tools.get(name)
        if tool is None:
            available = "、".join(self._tools) or "（无）"
            return f"错误：未知工具 {name}。可用工具：{available}"

        try:
            values = json.loads(raw_arguments or "{}")
        except (json.JSONDecodeError, TypeError) as error:
            return f"错误：参数 JSON 解析失败（{error}）。"

        if not isinstance(values, dict):
            return f"错误：{name} 的参数必须是 JSON 对象。"

        if set(values) != set(tool.parameters) or any(
            not isinstance(v, str) or not v.strip() for v in values.values()
        ):
            if not tool.parameters:
                return f"错误：{name} 不接受参数，请传空对象 {{}}。"
            expected = ", ".join(tool.parameters)
            return f"错误：{name} 需要非空字符串参数：{expected}。"

        try:
            return str(tool.handler(**values))
        except Exception as error:
            return f"错误：工具执行异常（{type(error).__name__}: {error}）。"
