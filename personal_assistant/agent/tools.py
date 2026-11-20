"""Layer 2 通用工具抽象与白名单注册表（对标 pi-agent-core）：递进继承 ai.ToolSchema 叠加执行能力。"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from personal_assistant.ai.types import ToolSchema


# 中层可执行工具：继承底层 ToolSchema，仅叠加 handler 执行回调，不修改底层 Schema 契约
@dataclass(frozen=True)
class Tool(ToolSchema):
    handler: Callable[..., str]


# 装饰器：把普通 Python 函数标记为 Agent 插件工具，同时不影响函数原本调用方式
def tool(
    *,
    description: str,
    parameters: dict[str, str] | None = None,
    name: str | None = None,
    param_enums: dict[str, tuple[str, ...]] | None = None,
) -> Callable[[Callable[..., str]], Callable[..., str]]:
    # 为目标函数挂载 __agent_tool__ 元信息供插件加载器识别
    def decorator(fn: Callable[..., str]) -> Callable[..., str]:
        spec = Tool(
            name=name or fn.__name__,
            description=description,
            parameters=dict(parameters or {}),
            handler=fn,
            param_enums=param_enums,
        )
        setattr(fn, "__agent_tool__", spec)
        return fn

    return decorator


# 工具白名单注册表：负责过滤允许的工具、导出 OpenAI Schema 以及校验并执行工具调用
class ToolRegistry:
    # 初始化注册表，校验工具名不重复；若传入 allowed_names 则只保留白名单内工具
    def __init__(
        self,
        *tools: Tool,
        allowed_names: Iterable[str] | None = None,
    ) -> None:
        raw_map: dict[str, Tool] = {}
        for item in tools:
            if item.name in raw_map:
                raise ValueError(f"工具名称不能重复：{item.name}")
            raw_map[item.name] = item

        if allowed_names is not None:
            allowset = set(allowed_names)
            self._tools = {name: item for name, item in raw_map.items() if name in allowset}
        else:
            self._tools = raw_map

    # 返回当前已启用（在白名单内）的所有工具名称列表
    @property
    def names(self) -> list[str]:
        return list(self._tools)

    # 支持直接遍历或解包注册表中的所有 Tool 对象
    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    # 将已注册工具委派给底层 ToolSchema.to_openai_schema 导出标准工具列表
    def definitions(self) -> list[dict[str, object]]:
        return [item.to_openai_schema() for item in self._tools.values()]

    # 按名称执行白名单内的工具；任何参数或执行错误都转为文本字符串回传给模型自愈
    def execute(self, name: str, raw_arguments: str) -> str:
        item = self._tools.get(name)
        if item is None:
            available = "、".join(self._tools) or "（无）"
            return f"错误：未知工具或不在白名单中：{name}。可用工具：{available}"

        try:
            parsed = json.loads(raw_arguments or "{}")
        except (json.JSONDecodeError, TypeError) as error:
            return f"错误：参数 JSON 解析失败（{error}）。"

        values, error = self._validate_values(item, parsed)
        if error:
            return f"错误：{error}"

        try:
            return str(item.handler(**values))
        except Exception as error:
            return f"错误：工具执行异常（{type(error).__name__}: {error}）。"

    # 校验模型传入的 JSON 参数键名、非空字符串类型及可选 enum 取值范围
    @staticmethod
    def _validate_values(item: Tool, parsed: object) -> tuple[dict[str, str], str]:
        if not isinstance(parsed, dict):
            return {}, f"{item.name} 的参数必须是 JSON 对象。"
        if set(parsed) != set(item.parameters):
            if not item.parameters:
                return {}, f"{item.name} 不接受参数，请传空对象 {{}}。"
            return {}, f"{item.name} 需要参数：{', '.join(item.parameters)}。"
        clean: dict[str, str] = {}
        for key in item.parameters:
            val = parsed[key]
            # 兼容模型直接传入数字参数（如行号 start_line: 1），同时严格拒绝布尔值与空串
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                val = str(val)
            if not isinstance(val, str) or not val.strip():
                return {}, f"{item.name} 的参数 {key} 必须是非空字符串。"
            if item.param_enums and key in item.param_enums and val not in item.param_enums[key]:
                return {}, f"{item.name} 的参数 {key} 只能取：{', '.join(item.param_enums[key])}。"
            clean[key] = val
        return clean, ""
