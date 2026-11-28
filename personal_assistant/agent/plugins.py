"""Layer 2 统一插件主板（Plugin ABC + PluginContext）与内层强类型极简子协议。"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar, Protocol, runtime_checkable

from personal_assistant.agent.react import AgentEvent
from personal_assistant.agent.tools import (
    Tool,
    ToolMetadata,
    ToolsetProvider,
    bind_tool_context,
    build_tool,
    extract_tool,
    is_tool,
)
from personal_assistant.ai.protocol import ApiProvider, TokenUsage


# 通用后台任务完成事件：供上层广播异步任务耗时与 Token 开销，不侵入微内核业务实现
@dataclass(frozen=True)
class TaskEvent:
    kind: str
    duration_ms: float = 0.0
    usage: TokenUsage | None = None
    details: dict[str, object] = field(default_factory=dict)
    error: str | None = None


RuntimeEvent = AgentEvent | TaskEvent
RetrieverFactory = Callable[[dict[str, object]], "RetrieverProtocol"]
ObserverFactory = Callable[[dict[str, object]], "ObserverProtocol"]


# 内层子协议 2：检索引擎极简协议（只认 search_hits 契约，支持可选前置过滤对象，不关心实现是否来自插件）
@runtime_checkable
class RetrieverProtocol(Protocol):
    name: str

    # 按关键词、可选逻辑日期与可选前置元数据过滤对象检索匹配的文档片段列表
    def search_hits(
        self,
        memory: Any,
        query: str,
        root: str = "",
        limit: int = 10,
        today: Any = None,
        *,
        metadata_filter: Any = None,
    ) -> list[Any]: ...


# 内层子协议 3：配置编译器极简协议
@runtime_checkable
class ConfigCompilerProtocol(Protocol):
    name: str

    # 将文本编译为标准配置字典
    def compile(self, text: str, source: str = "models.dsl") -> dict[str, object]: ...

    # 返回内置厂商预设列表
    def presets(self) -> list[dict[str, object]]: ...

    # 按通用模型参数自动渲染生成配置文本（具体插件实现可按需追加带默认值的扩展关键字参数）
    def render(
        self,
        *,
        preset: str,
        api_key: str = "",
        model: str = "",
        base_url: str = "",
        name: str = "",
        api: str = "",
        temperature: float = 0.2,
    ) -> str: ...


# 内层子协议 5：运行期可观测器极简协议（微内核仅按 session_key 广播事件与导出快照）
@runtime_checkable
class ObserverProtocol(Protocol):
    name: str

    # 接收微内核广播的运行期事件
    def on_event(self, session_key: str, event: RuntimeEvent) -> None: ...

    # 按可选会话键导出只读观测快照
    def snapshot(self, *, session_key: str | None = None) -> dict[str, object]: ...


# 校验并向主板插槽注册子协议实例或工厂函数，非法对象统一抛出 TypeError
def _register_slot_factory(
    slot: dict[str, Callable[[dict[str, object]], Any]],
    name: str,
    item: Any,
    protocol_cls: type[object],
    label: str,
) -> None:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError(f"{label}名称不能为空。")
    if clean_name in slot:
        raise ValueError(f"{label}名称重复：{clean_name}")
    if isinstance(item, protocol_cls) and not callable(item):
        instance = item

        # 返回已实例化的子协议对象
        def _factory(_cfg: dict[str, object]) -> Any:
            return instance

        slot[clean_name] = _factory
    elif callable(item):
        slot[clean_name] = item
    else:
        raise TypeError(f"{label}必须实现 {protocol_cls.__name__} 或可调用工厂：{clean_name}")


# 校验并向主板插槽注册单实例强类型协议对象（供配置编译器与协议翻译器插槽共用）
def _register_named_protocol(
    slot: dict[str, Any],
    item: Any,
    protocol_cls: type[object],
    label: str,
    *,
    key_attr: str = "name",
    case_insensitive: bool = False,
) -> None:
    if not isinstance(item, protocol_cls):
        raise TypeError(f"{label}必须实现 {protocol_cls.__name__}：{item!r}")
    raw_name = getattr(item, key_attr, "")
    clean_name = str(raw_name or "").strip()
    if case_insensitive:
        clean_name = clean_name.lower()
    if not clean_name:
        raise ValueError(f"{label}名称不能为空。")
    if clean_name in slot:
        raise ValueError(f"{label}名称重复：{clean_name}")
    slot[clean_name] = item


# 插件主板通用上下文（PluginContext）：零业务字段泄漏，仅承载通用根路径、配置、绑定字典与 5 个标准插槽
@dataclass
class PluginContext:
    project_root: Path
    config: Mapping[str, object] = field(default_factory=dict)
    bindings: Mapping[str, object] = field(default_factory=dict)
    tool_specs: list[Tool] = field(default_factory=list)
    retrievers: dict[str, RetrieverFactory] = field(default_factory=dict)
    compilers: dict[str, ConfigCompilerProtocol] = field(default_factory=dict)
    providers: dict[str, ApiProvider] = field(default_factory=dict)
    observers: dict[str, ObserverFactory] = field(default_factory=dict)

    # 插槽 1：向主板登记单个工具（裸 Tool 实例首次解析函数签名，绑定初始上下文并拦截重名）
    def register_tool(
        self,
        item: Tool | Callable[..., str],
        *,
        metadata: ToolMetadata | None = None,
    ) -> None:
        raw_tool = extract_tool(item)
        resolved_meta = metadata if metadata is not None else raw_tool.metadata
        if (
            isinstance(item, Tool)
            and raw_tool.handler is not None
            and raw_tool.required is None
            and raw_tool.accepted_context_keys == frozenset()
        ):
            raw_tool = build_tool(
                raw_tool.handler,
                name=raw_tool.name,
                description=raw_tool.description,
                parameters=raw_tool.parameters,
                param_enums=raw_tool.param_enums,
                risk=resolved_meta.risk,
                always_on=resolved_meta.always_on,
                parallel=resolved_meta.parallel,
                context=raw_tool.bound_kwargs,
            )
        init_bindings = {"project_root": self.project_root, **self.bindings}
        base_tool = bind_tool_context(raw_tool, init_bindings, metadata=resolved_meta)
        if any(existing.name == base_tool.name for existing in self.tool_specs):
            raise ValueError(f"工具名称不能重复：{base_tool.name}")
        self.tool_specs.append(base_tool)

    # 校验目标 .py 模块位于 project_root 内，扫描全部 @tool 装饰函数并按对象身份去重注册进当前主板插槽
    def register_tools_from_module(self, file_path: Path) -> None:
        root = self.project_root.resolve()
        candidate = file_path if file_path.is_absolute() else (root / file_path)
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"插件工具模块路径超出项目范围：{file_path}")
        if resolved.suffix != ".py" or not resolved.is_file():
            raise ValueError(f"插件工具模块不存在或非 .py 文件：{file_path}")
        module = _import_plugin_module(resolved)
        seen_ids: set[int] = set()
        for value in vars(module).values():
            if is_tool(value):
                spec = extract_tool(value)
                if id(spec) not in seen_ids:
                    seen_ids.add(id(spec))
                    self.register_tool(spec)

    # 插槽 2：向主板注册检索引擎子协议实现（支持传入 RetrieverProtocol 实例或工厂函数）
    def register_retriever(self, name: str, retriever_or_factory: RetrieverProtocol | RetrieverFactory) -> None:
        _register_slot_factory(self.retrievers, name, retriever_or_factory, RetrieverProtocol, "检索插件")

    # 插槽 3：向主板注册配置编译器子协议实现（仅接受 ConfigCompilerProtocol 实例）
    def register_compiler(self, compiler: ConfigCompilerProtocol) -> None:
        _register_named_protocol(self.compilers, compiler, ConfigCompilerProtocol, "配置编译器插件")

    # 插槽 4：向主板注册 Layer 1 ApiProvider 协议翻译器实例（按 provider.api 小写归一化登记）
    def register_provider(self, provider: ApiProvider) -> None:
        _register_named_protocol(
            self.providers,
            provider,
            ApiProvider,
            "协议翻译器插件",
            key_attr="api",
            case_insensitive=True,
        )

    # 插槽 5：向主板注册运行观测器子协议实现（支持传入 ObserverProtocol 实例或工厂函数）
    def register_observer(self, name: str, observer_or_factory: ObserverProtocol | ObserverFactory) -> None:
        _register_slot_factory(self.observers, name, observer_or_factory, ObserverProtocol, "观测插件")


# 外层统一主板基类：所有插件均显式继承此类，统一 initialize / terminate 生命周期
class Plugin(ABC):
    plugin_id: ClassVar[str] = ""

    # 绑定统一系统资源上下文 PluginContext
    def __init__(self, context: PluginContext) -> None:
        self.context = context

    # 统一初始化生命周期：从 self.context 获取配置/绑定资源，并将内层子协议实现插入对应主板插槽
    @abstractmethod
    def initialize(self) -> None: ...

    # 统一停用清理生命周期（默认空操作，有状态插件可按需重写）
    def terminate(self) -> None:
        return None


# 收集单个已解析路径（单文件或目录）下的候选插件 .py 文件列表
def _collect_target_py_files(target: Path) -> list[Path]:
    if not target.is_dir():
        return [target]
    if (target / "plugin.py").is_file():
        return [target / "plugin.py"]
    candidates: list[Path] = []
    for child in sorted(target.iterdir()):
        if child.name.startswith(("_", ".")):
            continue
        if child.is_dir():
            if (child / "plugin.py").is_file():
                candidates.append(child / "plugin.py")
            else:
                candidates.extend(p for p in sorted(child.glob("*.py")) if not p.name.startswith("_"))
        elif child.is_file() and child.suffix == ".py":
            candidates.append(child)
    return candidates


# 解析配置中的插件文件或目录路径（同时支持单文件插件与目录 plugin.py 文件夹插件）
def _resolve_plugin_targets(entries: Iterable[str], project_root: Path) -> list[Path]:
    root = project_root.resolve()
    files: list[Path] = []
    seen_files: set[Path] = set()
    for entry in entries:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError("plugins 配置项必须是非空相对路径。")
        rel = Path(entry.strip())
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"插件路径必须是项目内相对路径：{entry}")
        target = (root / rel).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"插件路径超出项目范围：{entry}")
        if not target.exists():
            continue
        for file_path in _collect_target_py_files(target):
            resolved = file_path.resolve()
            if resolved.is_relative_to(root) and resolved.suffix == ".py" and resolved.is_file() and resolved not in seen_files:
                seen_files.add(resolved)
                files.append(resolved)
    return files


_IMPORT_LOCK = threading.RLock()
_MODULE_FINGERPRINTS: dict[str, tuple[int, int, int]] = {}


# 动态导入单个插件 .py 文件并注册到 sys.modules（按文件指纹缓存未改动模块，支持热重载）
def _import_plugin_module(file_path: Path) -> ModuleType:
    resolved = file_path.resolve()
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:10]
    module_name = f"my_helper_plugin_{resolved.stem}_{digest}"
    with _IMPORT_LOCK:
        try:
            st = resolved.stat()
            fingerprint: tuple[int, int, int] | None = (st.st_mtime_ns, st.st_size, st.st_ino)
        except OSError:
            fingerprint = None
        cached = sys.modules.get(module_name)
        if cached is not None and fingerprint is not None and _MODULE_FINGERPRINTS.get(module_name) == fingerprint:
            return cached
        spec = importlib.util.spec_from_file_location(module_name, resolved)
        if spec is None or spec.loader is None:
            raise ValueError(f"无法加载插件文件：{file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            _MODULE_FINGERPRINTS.pop(module_name, None)
            raise
        if fingerprint is not None:
            _MODULE_FINGERPRINTS[module_name] = fingerprint
        return module


# 扫描模块命名空间内定义的具体 Plugin 子类并依次执行 initialize() 生命周期（无 Plugin 子类的模块直接忽略）
def _initialize_modules(modules: Iterable[ModuleType], context: PluginContext) -> list[Plugin]:
    active_plugins: list[Plugin] = []
    for module in modules:
        for obj in vars(module).values():
            if (
                isinstance(obj, type)
                and issubclass(obj, Plugin)
                and obj is not Plugin
                and not inspect.isabstract(obj)
                and obj.__module__ == module.__name__
            ):
                instance = obj(context)
                instance.initialize()
                active_plugins.append(instance)
    return active_plugins


# 插件管理器（继承 ToolsetProvider）：加载全部显式 Plugin(ABC) 子类并支持轻量工具上下文重绑
@dataclass(frozen=True)
class PluginManager(ToolsetProvider):
    retrievers: dict[str, RetrieverFactory] = field(default_factory=dict)
    providers: dict[str, ApiProvider] = field(default_factory=dict)
    compilers: dict[str, ConfigCompilerProtocol] = field(default_factory=dict)
    observers: dict[str, ObserverFactory] = field(default_factory=dict)
    plugins: tuple[Plugin, ...] = ()
    tool_specs: tuple[Tool, ...] = ()
    project_root: Path | None = None
    bindings: Mapping[str, object] = field(default_factory=dict)

    # 扫描并导入配置的全部插件模块，创建纯净 PluginContext 并统一执行 Plugin.initialize()
    @classmethod
    def from_entries(
        cls,
        entries: Iterable[str],
        project_root: Path,
        *,
        config: Mapping[str, object] | None = None,
        bindings: Mapping[str, object] | None = None,
    ) -> "PluginManager":
        root = project_root.resolve()
        loaded_modules = [_import_plugin_module(fp) for fp in _resolve_plugin_targets(entries, root)]
        base_cfg = dict(config) if config else {}
        base_bindings: dict[str, object] = {"project_root": root, **(dict(bindings) if bindings else {})}
        ctx = PluginContext(project_root=root, config=base_cfg, bindings=base_bindings)
        active_plugins = _initialize_modules(loaded_modules, ctx)
        return cls(
            retrievers=dict(ctx.retrievers),
            providers=dict(ctx.providers),
            compilers=dict(ctx.compilers),
            observers=dict(ctx.observers),
            plugins=tuple(active_plugins),
            tool_specs=tuple(ctx.tool_specs),
            project_root=root,
            bindings=base_bindings,
        )

    # 按可选运行时上下文基于缓存的 Tool 规格调用 bind_tool_context 轻量重绑，不再重跑 initialize()
    def build_tools(self, context: Mapping[str, object] | None = None) -> list[Tool]:
        if not self.tool_specs:
            return []
        if not context:
            return list(self.tool_specs)
        merged = {**self.bindings, **context}
        return [bind_tool_context(spec, merged) for spec in self.tool_specs]

    # 统一触发全部已加载 Plugin 实例的 terminate() 停止清理钩子
    def terminate_all(self) -> None:
        for plugin in self.plugins:
            plugin.terminate()
