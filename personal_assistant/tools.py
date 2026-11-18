"""Minimal tool definition and allowlisted function-calling registry."""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, str]
    handler: Callable[..., str]
    param_enums: dict[str, tuple[str, ...]] | None = None


class ToolRegistry:
    """Publish allowlisted tool schemas and run validated tool calls directly."""

    def __init__(
        self,
        *tools: Tool,
        allowed_names: Iterable[str] | None = None,
    ) -> None:
        raw_map: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in raw_map:
                raise ValueError(f"工具名称不能重复：{tool.name}")
            raw_map[tool.name] = tool

        if allowed_names is not None:
            allowset = set(allowed_names)
            self._tools = {name: tool for name, tool in raw_map.items() if name in allowset}
        else:
            self._tools = raw_map

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def definitions(self) -> list[dict[str, object]]:
        schemas: list[dict[str, object]] = []
        for tool in self._tools.values():
            properties: dict[str, dict[str, object]] = {}
            for param, desc in tool.parameters.items():
                prop: dict[str, object] = {"type": "string", "description": desc}
                if tool.param_enums and param in tool.param_enums:
                    prop["enum"] = list(tool.param_enums[param])
                properties[param] = prop
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": list(tool.parameters),
                            "additionalProperties": False,
                        },
                    },
                }
            )
        return schemas

    def execute(self, name: str, raw_arguments: str) -> str:
        tool = self._tools.get(name)
        if tool is None:
            available = "、".join(self._tools) or "（无）"
            return f"错误：未知工具或不在白名单中：{name}。可用工具：{available}"

        try:
            parsed = json.loads(raw_arguments or "{}")
        except (json.JSONDecodeError, TypeError) as error:
            return f"错误：参数 JSON 解析失败（{error}）。"

        values, error = self._validate_values(tool, parsed)
        if error:
            return f"错误：{error}"

        try:
            return str(tool.handler(**values))
        except Exception as error:
            return f"错误：工具执行异常（{type(error).__name__}: {error}）。"

    @staticmethod
    def _validate_values(tool: Tool, parsed: object) -> tuple[dict[str, str], str]:
        if not isinstance(parsed, dict):
            return {}, f"{tool.name} 的参数必须是 JSON 对象。"
        if set(parsed) != set(tool.parameters):
            if not tool.parameters:
                return {}, f"{tool.name} 不接受参数，请传空对象 {{}}。"
            return {}, f"{tool.name} 需要参数：{', '.join(tool.parameters)}。"
        clean: dict[str, str] = {}
        for key in tool.parameters:
            val = parsed[key]
            if not isinstance(val, str) or not val.strip():
                return {}, f"{tool.name} 的参数 {key} 必须是非空字符串。"
            if tool.param_enums and key in tool.param_enums and val not in tool.param_enums[key]:
                return {}, f"{tool.name} 的参数 {key} 只能取：{', '.join(tool.param_enums[key])}。"
            clean[key] = val
        return clean, ""
