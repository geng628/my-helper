"""Layer 2 通用工具抽象与统一注册表：多源工具统一适配为 Tool 子类并汇入 ToolRegistry（_tools + _metadata + _documents）。"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from personal_assistant.ai.types import ToolSchema


# 工具治理元数据：记录风险级别（read / write / exec / network）与是否常驻激活（always_on）
@dataclass(frozen=True)
class ToolMetadata:
    risk: str = "read"
    always_on: bool = False


# 工具来源归档文档：记录来源类别（builtin / plugin / command / mcp / peer）与具体来源模块名
@dataclass(frozen=True)
class ToolDocument:
    source_type: str = "builtin"
    source_name: str = "core"


# 中层可执行工具基类：继承底层 ToolSchema，支持子类重写 run() 或委派给 handler 回调
@dataclass(frozen=True)
class Tool(ToolSchema):
    handler: Callable[..., str] | None = None

    # 执行工具核心逻辑；子类可重写此方法，默认调用 handler 回调
    def run(self, **values: str) -> str:
        if self.handler is None:
            raise NotImplementedError(f"工具 {self.name} 未实现 run() 或 handler。")
        return str(self.handler(**values))


# 内核内置工具集提供方基类：批量产出 Tool 子类实例并统一注册进 ToolRegistry
class ToolsetProvider:
    source_type: str = "builtin"
    source_name: str = "core"

    # 返回当前工具集提供的全部 Tool 子类实例（由具体业务子类实现）
    def tools(self) -> list[Tool]:
        raise NotImplementedError

    # 将当前工具集的全部 Tool 实例连同元数据与来源文档统一写入目标 ToolRegistry
    def register_into(self, registry: "ToolRegistry") -> None:
        for item in self.tools():
            meta = getattr(item, "__tool_metadata__", None) or ToolMetadata()
            doc = getattr(item, "__tool_document__", None) or ToolDocument(
                source_type=self.source_type,
                source_name=self.source_name,
            )
            registry.register(item, metadata=meta, document=doc)


# 用动态 type() 为插件函数生成专属 Tool 子类实例，并绑定 ToolMetadata 与 ToolDocument
def build_dynamic_plugin_tool(
    fn: Callable[..., str],
    *,
    description: str,
    parameters: dict[str, str] | None = None,
    name: str | None = None,
    param_enums: dict[str, tuple[str, ...]] | None = None,
    risk: str = "read",
    always_on: bool = False,
    source_name: str = "plugin",
) -> Tool:
    tool_name = name or fn.__name__
    safe_suffix = re.sub(r"[^A-Za-z0-9_]", "_", tool_name).title().replace("_", "") or "Dynamic"

    # 动态 Tool 子类的 run 实现：直接转发已校验的关键字参数给被装饰的插件函数
    def _run(self: Tool, **values: str) -> str:
        return str(fn(**values))

    dynamic_cls = type(
        f"{safe_suffix}PluginTool",
        (Tool,),
        {"run": _run, "__doc__": f"动态生成的插件工具子类：{tool_name}"},
    )
    instance: Tool = dynamic_cls(
        name=tool_name,
        description=description,
        parameters=dict(parameters or {}),
        handler=fn,
        param_enums=param_enums,
    )
    object.__setattr__(instance, "__tool_metadata__", ToolMetadata(risk=risk, always_on=always_on))
    object.__setattr__(
        instance,
        "__tool_document__",
        ToolDocument(source_type="plugin", source_name=source_name),
    )
    return instance


# 装饰器：将普通 Python 函数包装为动态 type() Tool 子类，同时保持原函数可直接调用
def tool(
    *,
    description: str,
    parameters: dict[str, str] | None = None,
    name: str | None = None,
    param_enums: dict[str, tuple[str, ...]] | None = None,
    risk: str = "read",
    always_on: bool = False,
) -> Callable[[Callable[..., str]], Callable[..., str]]:
    # 为被装饰函数构造动态 Tool 子类实例并挂载到 __agent_tool__
    def decorator(fn: Callable[..., str]) -> Callable[..., str]:
        mod_name = (getattr(fn, "__module__", "") or "plugin").rsplit(".", 1)[-1]
        spec = build_dynamic_plugin_tool(
            fn,
            description=description,
            parameters=parameters,
            name=name,
            param_enums=param_enums,
            risk=risk,
            always_on=always_on,
            source_name=mod_name,
        )
        setattr(fn, "__agent_tool__", spec)
        return fn

    return decorator


# 统一工具注册表：汇聚内置 ToolsetProvider、插件 @tool 与外部 Wrapper，统一维护 _tools / _metadata / _documents 并按需激活
class ToolRegistry:
    # 初始化统一工具表、元数据表与来源文档表；若指定 allowed_names 则按白名单（含 always_on）按需激活
    def __init__(
        self,
        *tools: Tool,
        allowed_names: Iterable[str] | None = None,
    ) -> None:
        self._all_tools: dict[str, Tool] = {}
        self._tools: dict[str, Tool] = {}
        self._metadata: dict[str, ToolMetadata] = {}
        self._documents: dict[str, ToolDocument] = {}
        self._allowed_names: set[str] | None = (
            set(allowed_names) if allowed_names is not None else None
        )
        for item in tools:
            self.register(item)

    # 将单个 Tool 子类实例注册入表，同步写入 _metadata 与 _documents 并按需激活到 _tools
    def register(
        self,
        item: Tool,
        *,
        metadata: ToolMetadata | None = None,
        document: ToolDocument | None = None,
    ) -> None:
        if item.name in self._all_tools:
            raise ValueError(f"工具名称不能重复：{item.name}")
        resolved_meta = (
            metadata
            or getattr(item, "__tool_metadata__", None)
            or ToolMetadata()
        )
        resolved_doc = (
            document
            or getattr(item, "__tool_document__", None)
            or ToolDocument()
        )
        self._all_tools[item.name] = item
        self._metadata[item.name] = resolved_meta
        self._documents[item.name] = resolved_doc
        if (
            self._allowed_names is None
            or item.name in self._allowed_names
            or resolved_meta.always_on
        ):
            self._tools[item.name] = item

    # 将一个 ToolsetProvider 的全部工具批量注册进当前注册表
    def register_provider(self, provider: ToolsetProvider) -> None:
        provider.register_into(self)

    # 按白名单列表动态更新当前激活的工具集 _tools（always_on 工具始终保持激活）
    def activate(self, allowed_names: Iterable[str] | None) -> None:
        self._allowed_names = set(allowed_names) if allowed_names is not None else None
        if self._allowed_names is None:
            self._tools = dict(self._all_tools)
            return
        self._tools = {
            name: item
            for name, item in self._all_tools.items()
            if name in self._allowed_names or self._metadata[name].always_on
        }

    # 查询指定工具的治理元数据（risk / always_on）
    def metadata(self, name: str) -> ToolMetadata | None:
        return self._metadata.get(name)

    # 查询指定工具的来源归档信息（source_type / source_name）
    def document(self, name: str) -> ToolDocument | None:
        return self._documents.get(name)

    # 返回当前已激活（在白名单内或 always_on）的所有工具名称列表
    @property
    def names(self) -> list[str]:
        return list(self._tools)

    # 支持直接遍历或解包当前已激活的所有 Tool 对象
    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    # 将当前已激活工具导出为标准 OpenAI function-calling Schema 列表
    def definitions(self) -> list[dict[str, object]]:
        return [item.to_openai_schema() for item in self._tools.values()]

    # 按名称执行已激活工具的 item.run(**values)；参数或执行错误统一转为文本回传模型自愈
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
            return str(item.run(**values))
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
