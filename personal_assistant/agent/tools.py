"""Layer 2 通用工具抽象、统一构建器（build_tool）、上下文重绑器与统一注册表（ToolRegistry）。"""
from __future__ import annotations

import inspect
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace

from personal_assistant.ai.protocol import ToolSchema


# 工具治理元数据：记录风险级别（read / write / exec / network）、是否常驻激活（always_on）与是否允许同轮并发只读执行（parallel）
@dataclass(frozen=True)
class ToolMetadata:
    risk: str = "read"
    always_on: bool = False
    parallel: bool = False


# 按工具接受键白名单与非空约束过滤上下文参数，绝不覆盖模型声明参数
def _filter_context_kwargs(
    parameters: Mapping[str, str],
    accepted_keys: frozenset[str] | None,
    context: Mapping[str, object] | None,
) -> dict[str, object]:
    if not context:
        return {}
    return {
        key: val
        for key, val in context.items()
        if key not in parameters
        and val is not None
        and (accepted_keys is None or key in accepted_keys)
    }


# 内层子协议 1：纯值对象可执行工具（继承底层 ToolSchema，唯一执行入口收敛于 run_with_context）
@dataclass(frozen=True)
class Tool(ToolSchema):
    handler: Callable[..., str] | None = None
    metadata: ToolMetadata = field(default_factory=ToolMetadata, kw_only=True, repr=False, compare=False)
    bound_kwargs: Mapping[str, object] = field(default_factory=dict, kw_only=True, repr=False, compare=False)
    accepted_context_keys: frozenset[str] | None = field(
        default=frozenset(), kw_only=True, repr=False, compare=False
    )

    # 携带执行期上下文运行工具：合并预绑定上下文、执行期动态上下文与模型入参后调用 handler
    def run_with_context(
        self,
        values: Mapping[str, str],
        context: Mapping[str, object] | None = None,
    ) -> str:
        if self.handler is None:
            raise NotImplementedError(f"工具 {self.name} 未绑定可执行 handler。")
        call_kwargs = dict(self.bound_kwargs)
        call_kwargs.update(_filter_context_kwargs(self.parameters, self.accepted_context_keys, context))
        call_kwargs.update(values)
        return str(self.handler(**call_kwargs))

    # 无额外运行时上下文时直接委派给唯一执行入口 run_with_context
    def run(self, **values: str) -> str:
        return self.run_with_context(values, None)


# 统一工具构建器：仅在构造期解析一次函数签名，推断必填参数与可接受上下文键并构造纯净 Tool 实例
def build_tool(
    fn: Callable[..., str],
    *,
    description: str,
    parameters: dict[str, str] | None = None,
    name: str | None = None,
    param_enums: dict[str, tuple[str, ...]] | None = None,
    required: Iterable[str] | None = None,
    risk: str = "read",
    always_on: bool = False,
    parallel: bool = False,
    accept_context: bool = True,
    context: Mapping[str, object] | None = None,
) -> Tool:
    tool_name = name or fn.__name__
    schema_params = dict(parameters or {})

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        sig = None

    if required is not None:
        required_set = set(required)
        resolved_required = tuple(k for k in schema_params if k in required_set)
    elif sig is None:
        resolved_required = tuple(schema_params)
    else:
        optional_kinds = (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        resolved_required = tuple(
            k
            for k in schema_params
            if (param := sig.parameters.get(k)) is None
            or param.kind not in optional_kinds
            or param.default is inspect.Parameter.empty
        )

    if not accept_context or sig is None:
        accepted_keys: frozenset[str] | None = frozenset()
    elif any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        accepted_keys = None
    else:
        ctx_kinds = (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        accepted_keys = frozenset(
            param_name
            for param_name, param in sig.parameters.items()
            if param_name not in schema_params and param.kind in ctx_kinds
        )

    bound_context = _filter_context_kwargs(schema_params, accepted_keys, context)
    return Tool(
        name=tool_name,
        description=description,
        parameters=schema_params,
        handler=fn,
        param_enums=param_enums,
        required=resolved_required,
        metadata=ToolMetadata(risk=risk, always_on=always_on, parallel=bool(parallel)),
        bound_kwargs=bound_context,
        accepted_context_keys=accepted_keys,
    )


# 纯函数重绑工具上下文与治理元数据（基于 dataclasses.replace，零反射、零属性猴子补丁）
def bind_tool_context(
    item: Tool,
    context: Mapping[str, object] | None = None,
    *,
    metadata: ToolMetadata | None = None,
) -> Tool:
    merged = dict(item.bound_kwargs)
    merged.update(_filter_context_kwargs(item.parameters, item.accepted_context_keys, context))
    return replace(
        item,
        metadata=metadata if metadata is not None else item.metadata,
        bound_kwargs=merged,
    )


# 统一工具集提供方抽象基类（内置 / 插件 / 命令等均实现单一 build_tools 契约）
class ToolsetProvider(ABC):
    # 按可选运行时上下文产出当前提供方的全部纯净 Tool 实例
    @abstractmethod
    def build_tools(self, context: Mapping[str, object] | None = None) -> list[Tool]: ...

    # 将当前提供方的全部 Tool 实例统一注册进目标 ToolRegistry
    def register_into(
        self,
        registry: "ToolRegistry",
        *,
        context: Mapping[str, object] | None = None,
    ) -> None:
        for item in self.build_tools(context=context):
            registry.register(item)


# 装饰器：将普通 Python 函数统一通过 build_tool 包装为纯净 Tool 实例，同时保持原函数可直接调用
def tool(
    *,
    description: str,
    parameters: dict[str, str] | None = None,
    name: str | None = None,
    param_enums: dict[str, tuple[str, ...]] | None = None,
    required: Iterable[str] | None = None,
    risk: str = "read",
    always_on: bool = False,
    parallel: bool = False,
) -> Callable[[Callable[..., str]], Callable[..., str]]:
    # 为被装饰函数构造 Tool 实例并挂载到 __agent_tool__
    def decorator(fn: Callable[..., str]) -> Callable[..., str]:
        spec = build_tool(
            fn,
            description=description,
            parameters=parameters,
            name=name,
            param_enums=param_enums,
            required=required,
            risk=risk,
            always_on=always_on,
            parallel=parallel,
        )
        setattr(fn, "__agent_tool__", spec)
        return fn

    return decorator


# 判断对象是否为 Tool 实例或经过 @tool 装饰的函数
def is_tool(item: object) -> bool:
    return isinstance(item, Tool) or isinstance(getattr(item, "__agent_tool__", None), Tool)


# 从 Tool 实例或 @tool 装饰的函数中提取底层 Tool 对象
def extract_tool(item: object) -> Tool:
    if isinstance(item, Tool):
        return item
    marked = getattr(item, "__agent_tool__", None)
    if isinstance(marked, Tool):
        return marked
    raise TypeError(f"对象不是合法的 Tool 或 @tool 函数：{item!r}")


# 统一工具注册表：汇聚所有 ToolsetProvider 与工具对象，注册期一次性完成白名单（含 always_on）激活
class ToolRegistry:
    # 初始化统一工具表并按可选 allowed_names 白名单（含 always_on）激活工具
    def __init__(
        self,
        *tools: Tool | Callable[..., str],
        allowed_names: Iterable[str] | None = None,
    ) -> None:
        self._all_tools: dict[str, Tool] = {}
        self._tools: dict[str, Tool] = {}
        self._allowed_names: set[str] | None = (
            set(allowed_names) if allowed_names is not None else None
        )
        for item in tools:
            self.register(item)

    # 将单个 Tool 实例或 @tool 函数注册入表，按白名单与 always_on 一次性决定是否激活
    def register(
        self,
        item: Tool | Callable[..., str],
        *,
        metadata: ToolMetadata | None = None,
    ) -> None:
        resolved = extract_tool(item)
        if metadata is not None:
            resolved = bind_tool_context(resolved, None, metadata=metadata)
        if resolved.name in self._all_tools:
            raise ValueError(f"工具名称不能重复：{resolved.name}")
        self._all_tools[resolved.name] = resolved
        if (
            self._allowed_names is None
            or resolved.name in self._allowed_names
            or resolved.metadata.always_on
        ):
            self._tools[resolved.name] = resolved

    # 将任意 ToolsetProvider（内置 / 插件 / 命令）按统一上下文批量注册进当前注册表
    def register_provider(
        self,
        provider: ToolsetProvider,
        *,
        context: Mapping[str, object] | None = None,
    ) -> None:
        provider.register_into(self, context=context)

    # 直接从单一真源 _all_tools 查询指定工具的治理元数据（risk / always_on / parallel）
    def metadata(self, name: str) -> ToolMetadata | None:
        item = self._all_tools.get(name)
        return item.metadata if item is not None else None

    # 判断指定已激活工具是否声明允许同轮并发只读执行（未激活或未知工具默认按串行安全处理）
    def is_parallel(self, name: str) -> bool:
        item = self._tools.get(name)
        return bool(item is not None and item.metadata.parallel)

    # 返回当前已激活（在白名单内或 always_on）的所有工具名称列表
    @property
    def names(self) -> list[str]:
        return list(self._tools)

    # 支持直接遍历或解包当前已激活的所有 Tool 对象
    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    # 导出当前已激活工具的中立 ToolSchema 元组（不含任何厂商协议格式与本地执行逻辑）
    def schemas(self) -> tuple[ToolSchema, ...]:
        return tuple(
            ToolSchema(
                name=item.name,
                description=item.description,
                parameters=dict(item.parameters),
                param_enums=dict(item.param_enums) if item.param_enums is not None else None,
                required=item.required,
            )
            for item in self._tools.values()
        )

    # 按名称并携带可选执行期上下文运行已激活工具；参数或执行错误统一转为文本回传模型自愈
    def execute(
        self,
        name: str,
        raw_arguments: str,
        *,
        context: Mapping[str, object] | None = None,
    ) -> str:
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
            return str(item.run_with_context(values, context))
        except Exception as error:
            return f"错误：工具执行异常（{type(error).__name__}: {error}）。"

    # 校验模型传入的 JSON 参数键名、非空字符串类型、必填约束及可选 enum 取值范围
    @staticmethod
    def _validate_values(item: Tool, parsed: object) -> tuple[dict[str, str], str]:
        if not isinstance(parsed, dict):
            return {}, f"{item.name} 的参数必须是 JSON 对象。"
        if not item.parameters:
            if parsed:
                return {}, f"{item.name} 不接受参数，请传空对象 {{}}。"
            return {}, ""
        required_keys = set(item.required_names)
        if any(k not in item.parameters for k in parsed) or any(k not in parsed for k in required_keys):
            return {}, f"{item.name} 需要参数：{', '.join(item.parameters)}。"
        clean: dict[str, str] = {}
        for key in item.parameters:
            if key not in parsed or (parsed[key] is None and key not in required_keys):
                continue
            val = parsed[key]
            # 兼容模型直接传入数字参数（如 start_line: 1），同时严格拒绝布尔值与空串
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                val = str(val)
            if not isinstance(val, str) or not val.strip():
                return {}, f"{item.name} 的参数 {key} 必须是非空字符串。"
            if item.param_enums and key in item.param_enums and val not in item.param_enums[key]:
                return {}, f"{item.name} 的参数 {key} 只能取：{', '.join(item.param_enums[key])}。"
            clean[key] = val
        return clean, ""
