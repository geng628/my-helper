"""Layer 2 统一插件主板（Plugin ABC + PluginContext）与内层强类型极简子协议（RetrieverProtocol / ConfigCompilerProtocol / LLMProviderProtocol）。"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar, Protocol, runtime_checkable

from personal_assistant.agent.react import RuntimeEvent
from personal_assistant.agent.tools import (
    Tool,
    ToolDocument,
    ToolMetadata,
    ToolRegistry,
    ToolsetProvider,
    build_tool,
    extract_tool,
)
from personal_assistant.ai.client import LLMClient, ProviderFactory

RetrieverFactory = Callable[[dict[str, object]], object]
ObserverFactory = Callable[[dict[str, object]], "ObserverProtocol"]


# 内层子协议 2：检索引擎极简协议（MemoryStore 只认此协议的 search_hits 契约，不关心实现是否来自插件）
@runtime_checkable
class RetrieverProtocol(Protocol):
    name: str

    # 按关键词与可选逻辑日期检索匹配的文档片段列表
    def search_hits(
        self,
        memory: Any,
        query: str,
        root: str = "data/memory",
        limit: int = 10,
        today: Any = None,
    ) -> list[Any]: ...


# 内层子协议 3：配置编译器极简协议（ToolsConfig 只认此协议的 compile / presets / render 契约）
@runtime_checkable
class ConfigCompilerProtocol(Protocol):
    name: str

    # 将 DSL 文本编译为标准配置字典
    def compile(self, text: str, source: str = "models.dsl") -> dict[str, object]: ...

    # 返回内置模型厂商预设列表
    def presets(self) -> list[dict[str, object]]: ...

    # 按表单参数自动渲染生成 DSL 文本
    def render(self, **kwargs: object) -> str: ...


# 内层子协议 4：大模型客户端驱动极简协议（ai/client.py 只认此协议的 build 契约）
@runtime_checkable
class LLMProviderProtocol(Protocol):
    name: str

    # 根据环境变量或配置映射构建 LLMClient 实例
    def build(self, env: Mapping[str, str]) -> LLMClient: ...


# 校验并向主板插槽注册子协议实例或工厂函数，复用检索与观测插件的统一装配逻辑
def _register_slot_factory(
    slot: dict[str, Callable[[dict[str, object]], Any]],
    name: str,
    item: Any,
    protocol_cls: type[object],
    label: str,
    *,
    invalid_exc: type[Exception] = ValueError,
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
        raise invalid_exc(f"{label}必须实现 {protocol_cls.__name__} 或可调用工厂：{clean_name}")


# 内层子协议 5：运行期可观测器极简协议（微内核仅按 session_key 广播事件与导出快照，不感知具体统计策略）
@runtime_checkable
class ObserverProtocol(Protocol):
    name: str

    # 接收微内核广播的运行期事件（回合起止、流式文本增量、模型调用、工具执行与后台任务）
    def on_event(self, session_key: str, event: RuntimeEvent) -> None: ...

    # 按可选会话键导出只读观测快照，供 /status 与界面展示
    def snapshot(self, *, session_key: str | None = None) -> dict[str, object]: ...


# 缓存插件首次 initialize() 收集到的工具注册规格，区分函数型工具与自定义 Tool 子类
@dataclass(frozen=True)
class _PluginToolSpec:
    raw_tool: Tool
    is_function_tool: bool
    source_name: str | None
    metadata: ToolMetadata | None
    document: ToolDocument | None


# 判断传入对象是否为函数型工具（@tool / build_tool 产物或带 handler 的基类 Tool），区分于自定义 Tool 子类
def _is_function_tool(item: Tool | Callable[..., str]) -> bool:
    return (
        not isinstance(item, Tool)
        or hasattr(item, "_raw_fn")
        or (type(item) is Tool and item.handler is not None)
    )


# 克隆自定义 Tool 子类实例并挂载元数据与来源文档，完整保留子类重写 run() 且不混入额外上下文参数
def _clone_custom_tool(
    item: Tool,
    *,
    source_name: str | None = None,
    metadata: ToolMetadata | None = None,
    document: ToolDocument | None = None,
) -> tuple[Tool, ToolMetadata, ToolDocument]:
    resolved_meta = metadata or getattr(item, "__tool_metadata__", None) or ToolMetadata()
    existing_doc = getattr(item, "__tool_document__", None)
    if document is not None:
        resolved_doc = document
    elif existing_doc is not None and source_name is None:
        resolved_doc = existing_doc
    else:
        resolved_doc = ToolDocument(
            source_type=getattr(existing_doc, "source_type", "plugin") if existing_doc else "plugin",
            source_name=source_name or getattr(existing_doc, "source_name", "plugin") or "plugin",
        )
    cloned = copy.copy(item)
    object.__setattr__(cloned, "__tool_metadata__", resolved_meta)
    object.__setattr__(cloned, "__tool_document__", resolved_doc)
    return cloned, resolved_meta, resolved_doc


# 插件主板上下文（PluginContext）：统一向所有 Plugin 子类提供系统资源与 5 个标准子协议插槽
@dataclass
class PluginContext:
    project_root: Path
    data_root: Path | None = None
    config_path: Path | None = None
    offload_dir: Path | None = None
    raw_config: Mapping[str, object] = field(default_factory=dict)
    extra_bindings: Mapping[str, object] = field(default_factory=dict)
    tool_registry: ToolRegistry = field(default_factory=ToolRegistry)
    tool_specs: list[_PluginToolSpec] = field(default_factory=list)
    active_plugin_id: str = ""
    retrievers: dict[str, RetrieverFactory] = field(default_factory=dict)
    compilers: dict[str, ConfigCompilerProtocol] = field(default_factory=dict)
    providers: dict[str, ProviderFactory] = field(default_factory=dict)
    observers: dict[str, ObserverFactory] = field(default_factory=dict)

    # 提取供 build_tool 自动按签名绑定到 @tool 函数上的系统资源上下文
    def tool_bindings(self) -> dict[str, object]:
        raw_shell = self.raw_config.get("shell")
        raw_web = self.raw_config.get("web")
        raw_retrieval = self.raw_config.get("retrieval")
        bindings: dict[str, object] = dict(self.extra_bindings)
        bindings.update(
            {
                "project_root": self.project_root,
                "data_root": self.data_root,
                "config_path": self.config_path,
                "offload_dir": self.offload_dir,
                "raw_config": dict(self.raw_config),
                "shell_config": dict(raw_shell) if isinstance(raw_shell, Mapping) else None,
                "web_config": dict(raw_web) if isinstance(raw_web, Mapping) else None,
                "retrieval_config": dict(raw_retrieval) if isinstance(raw_retrieval, Mapping) else None,
            }
        )
        return bindings

    # 插槽 1：向主板 ToolRegistry 注册工具（函数型工具按签名重绑上下文；自定义 Tool 子类保留原样与元数据）
    def register_tool(
        self,
        item: Tool | Callable[..., str],
        *,
        source_name: str | None = None,
        metadata: ToolMetadata | None = None,
        document: ToolDocument | None = None,
    ) -> None:
        if _is_function_tool(item):
            raw_tool = extract_tool(item)
            existing_doc = getattr(raw_tool, "__tool_document__", None)
            existing_src = getattr(existing_doc, "source_name", None)
            src = (
                source_name
                or self.active_plugin_id
                or (existing_src if existing_src and existing_src != "core" else None)
                or "plugin"
            )
            normalized = _normalize_plugin_tool(
                raw_tool,
                source_name=src,
                context=self.tool_bindings(),
                metadata=metadata,
                document=document,
            )
            self.tool_registry.register(normalized, metadata=metadata, document=document)
            self.tool_specs.append(
                _PluginToolSpec(
                    raw_tool=raw_tool,
                    is_function_tool=True,
                    source_name=src,
                    metadata=metadata,
                    document=document,
                )
            )
            return

        assert isinstance(item, Tool)
        src_hint = source_name or self.active_plugin_id or None
        cloned, resolved_meta, resolved_doc = _clone_custom_tool(
            item,
            source_name=src_hint,
            metadata=metadata,
            document=document,
        )
        self.tool_registry.register(cloned, metadata=resolved_meta, document=resolved_doc)
        self.tool_specs.append(
            _PluginToolSpec(
                raw_tool=item,
                is_function_tool=False,
                source_name=src_hint,
                metadata=metadata,
                document=document,
            )
        )

    # 插槽 2：向主板注册检索引擎子协议实现（支持传入 RetrieverProtocol 实例或工厂函数）
    def register_retriever(
        self,
        name: str,
        retriever_or_factory: RetrieverProtocol | RetrieverFactory,
    ) -> None:
        _register_slot_factory(
            self.retrievers, name, retriever_or_factory, RetrieverProtocol, "检索插件"
        )

    # 插槽 3：向主板注册配置编译器子协议实现（仅接受 ConfigCompilerProtocol 实例）
    def register_compiler(self, compiler: ConfigCompilerProtocol) -> None:
        if not isinstance(compiler, ConfigCompilerProtocol):
            raise TypeError(f"配置编译器插件必须实现 ConfigCompilerProtocol：{compiler!r}")
        clean_name = compiler.name.strip()
        if not clean_name:
            raise ValueError("配置编译器插件名称不能为空。")
        if clean_name in self.compilers:
            raise ValueError(f"配置编译器插件名称重复：{clean_name}")
        self.compilers[clean_name] = compiler

    # 插槽 4：向主板注册大模型客户端驱动子协议实现（仅接受 LLMProviderProtocol 实例）
    def register_provider(self, provider: LLMProviderProtocol) -> None:
        if not isinstance(provider, LLMProviderProtocol):
            raise TypeError(f"模型 Provider 插件必须实现 LLMProviderProtocol：{provider!r}")
        clean_name = provider.name.strip()
        if not clean_name:
            raise ValueError("模型 Provider 插件名称不能为空。")
        if clean_name in self.providers:
            raise ValueError(f"模型 Provider 插件名称重复：{clean_name}")
        self.providers[clean_name] = provider.build

    # 插槽 5：向主板注册运行观测器子协议实现（支持传入 ObserverProtocol 实例或工厂函数）
    def register_observer(
        self,
        name: str,
        observer_or_factory: ObserverProtocol | ObserverFactory,
    ) -> None:
        _register_slot_factory(
            self.observers,
            name,
            observer_or_factory,
            ObserverProtocol,
            "观测插件",
            invalid_exc=TypeError,
        )


# 外层统一主板基类：所有 plugins/*.py 插件均继承此类，通过 __init_subclass__ 自动发现、共享 self.context 与统一生命周期
class Plugin(ABC):
    plugin_id: ClassVar[str] = ""

    # 子类定义时自动登记到所属模块的 __plugin_classes__ 表，零魔法全局变量即可被 PluginManager 自动发现
    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "plugin_id", "").strip():
            return
        mod = sys.modules.get(cls.__module__)
        if mod is not None:
            registered: list[type[Plugin]] = list(getattr(mod, "__plugin_classes__", ()))
            if cls not in registered:
                registered.append(cls)
                setattr(mod, "__plugin_classes__", registered)

    # 绑定统一系统资源上下文 PluginContext
    def __init__(self, context: PluginContext) -> None:
        self.context = context

    # 统一初始化生命周期：从 self.context 获取配置/路径资源，并将内层子协议实现插入对应主板插槽
    @abstractmethod
    def initialize(self) -> None: ...

    # 统一停用清理生命周期（默认空操作，有状态插件可按需重写）
    def terminate(self) -> None:
        return None


# 解析配置中的插件文件或目录路径（同时支持 plugins/*.py 单文件插件与 plugins/*/plugin.py 文件夹插件）
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

        candidates: list[Path] = []
        if target.is_dir():
            if (target / "plugin.py").is_file():
                candidates.append(target / "plugin.py")
            else:
                for child in sorted(target.iterdir()):
                    if child.name.startswith(("_", ".")):
                        continue
                    if child.is_dir():
                        if (child / "plugin.py").is_file():
                            candidates.append(child / "plugin.py")
                        else:
                            candidates.extend(
                                p for p in sorted(child.glob("*.py")) if not p.name.startswith("_")
                            )
                    elif child.is_file() and child.suffix == ".py":
                        candidates.append(child)
        else:
            candidates.append(target)

        for file_path in candidates:
            resolved = file_path.resolve()
            if not resolved.is_relative_to(root) or resolved.suffix != ".py" or not resolved.is_file():
                continue
            if resolved not in seen_files:
                seen_files.add(resolved)
                files.append(resolved)
    return files


_IMPORT_LOCK = threading.RLock()
_MODULE_FINGERPRINTS: dict[str, tuple[int, int, int]] = {}


# 动态导入单个插件 .py 文件并注册到 sys.modules（按文件指纹缓存未改动模块，支持热重载并保留模块级状态）
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
        if (
            cached is not None
            and fingerprint is not None
            and _MODULE_FINGERPRINTS.get(module_name) == fingerprint
        ):
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


# 统一通过 build_tool 将 Tool 实例标准化为带 source_name 与预绑定上下文的动态 Tool 子类
def _normalize_plugin_tool(
    item: Tool,
    source_name: str,
    *,
    context: Mapping[str, object] | None = None,
    metadata: ToolMetadata | None = None,
    document: ToolDocument | None = None,
) -> Tool:
    meta = metadata or getattr(item, "__tool_metadata__", None) or ToolMetadata()
    raw_fn = getattr(item, "_raw_fn", None) or item.handler or item.run
    return build_tool(
        raw_fn,
        description=item.description,
        parameters=item.parameters,
        name=item.name,
        param_enums=item.param_enums,
        risk=meta.risk,
        always_on=meta.always_on,
        source_type=document.source_type if document is not None else "plugin",
        source_name=document.source_name if document is not None else source_name,
        context=context,
    )


# 为仅写了 @tool 函数而未显式声明 Plugin 子类的简单脚本模块自动生成统一 Plugin 外壳
class _AutoScriptPlugin(Plugin):
    # 绑定待包装的脚本模块与文件名
    def __init__(self, context: PluginContext, module: ModuleType, file_path: Path) -> None:
        super().__init__(context)
        self.module = module
        self.file_path = file_path

    # 扫描脚本中的 @tool 装饰函数并统一注册进 self.context 插槽
    def initialize(self) -> None:
        source_name = self.file_path.stem
        seen_in_file: set[str] = set()
        for value in vars(self.module).values():
            marked = getattr(value, "__agent_tool__", None)
            if isinstance(marked, Tool) and marked.name not in seen_in_file:
                seen_in_file.add(marked.name)
                self.context.register_tool(marked, source_name=source_name)


# 在给定 PluginContext 上实例化并执行全部插件模块的 Plugin.initialize() 生命周期
def _initialize_modules(
    modules: Iterable[tuple[Path, ModuleType]],
    context: PluginContext,
) -> list[Plugin]:
    active_plugins: list[Plugin] = []
    for file_path, module in modules:
        plugin_classes: list[type[Plugin]] = list(getattr(module, "__plugin_classes__", ()))
        if plugin_classes:
            for plugin_cls in plugin_classes:
                instance = plugin_cls(context)
                prev_id = context.active_plugin_id
                context.active_plugin_id = getattr(plugin_cls, "plugin_id", "").strip()
                try:
                    instance.initialize()
                finally:
                    context.active_plugin_id = prev_id
                active_plugins.append(instance)
        else:
            auto_plugin = _AutoScriptPlugin(context, module, file_path)
            auto_plugin.initialize()
            active_plugins.append(auto_plugin)
    return active_plugins


# 插件管理器（继承 ToolsetProvider）：负责加载全部 Plugin(ABC) 子类并按 PluginContext 驱动 initialize/terminate 生命周期
@dataclass(frozen=True)
class PluginManager(ToolsetProvider):
    modules: tuple[tuple[Path, ModuleType], ...]
    default_tools: tuple[Tool, ...]
    retrievers: dict[str, RetrieverFactory]
    providers: dict[str, ProviderFactory]
    compilers: dict[str, ConfigCompilerProtocol]
    observers: dict[str, ObserverFactory] = field(default_factory=dict)
    plugins: tuple[Plugin, ...] = ()
    tool_specs: tuple[_PluginToolSpec, ...] = ()
    project_root: Path | None = None
    base_context: Mapping[str, object] = field(default_factory=dict)
    source_type: str = "plugin"
    source_name: str = "plugins"

    # 扫描并导入配置的全部插件模块，创建 PluginContext 并统一执行 Plugin.initialize()
    @classmethod
    def from_entries(
        cls,
        entries: Iterable[str],
        project_root: Path,
        *,
        context: Mapping[str, object] | None = None,
    ) -> "PluginManager":
        root = project_root.resolve()
        loaded_modules: list[tuple[Path, ModuleType]] = []
        for file_path in _resolve_plugin_targets(entries, root):
            module = _import_plugin_module(file_path)
            loaded_modules.append((file_path, module))

        base_ctx = dict(context) if context else {}
        ctx = cls._build_plugin_context(root, base_ctx)
        active_plugins = _initialize_modules(loaded_modules, ctx)
        return cls(
            modules=tuple(loaded_modules),
            default_tools=tuple(ctx.tool_registry),
            retrievers=dict(ctx.retrievers),
            providers=dict(ctx.providers),
            compilers=dict(ctx.compilers),
            observers=dict(ctx.observers),
            plugins=tuple(active_plugins),
            tool_specs=tuple(ctx.tool_specs),
            project_root=root,
            base_context=base_ctx,
        )

    # 将外部传入的上下文字典组装为强类型 PluginContext
    @staticmethod
    def _build_plugin_context(
        project_root: Path,
        context: Mapping[str, object] | None = None,
    ) -> PluginContext:
        if not context:
            return PluginContext(project_root=project_root)
        raw_data = context.get("data_root")
        data_root = Path(str(raw_data)).resolve() if isinstance(raw_data, (str, Path)) else None
        raw_cfg_path = context.get("config_path")
        config_path = Path(str(raw_cfg_path)).resolve() if isinstance(raw_cfg_path, (str, Path)) else None
        raw_offload = context.get("offload_dir")
        offload_dir = (
            Path(str(raw_offload)).resolve()
            if isinstance(raw_offload, (str, Path))
            else ((data_root / "workspace" / "tool_outputs") if data_root is not None else None)
        )
        raw_cfg = context.get("raw_config")
        cfg_map = raw_cfg if isinstance(raw_cfg, Mapping) else {}
        raw_extra = context.get("extra_bindings")
        extra_map = dict(raw_extra) if isinstance(raw_extra, Mapping) else {}
        return PluginContext(
            project_root=project_root,
            data_root=data_root,
            config_path=config_path,
            offload_dir=offload_dir,
            raw_config=cfg_map,
            extra_bindings=extra_map,
        )

    # 返回默认无额外上下文时的插件工具列表（实现 ToolsetProvider.tools）
    def tools(self) -> list[Tool]:
        return list(self.default_tools)

    # 按给定运行时上下文基于缓存的原始工具定义重新绑定实例资源，不再重跑全部 Plugin.initialize()
    def build_tools(self, context: Mapping[str, object] | None = None) -> list[Tool]:
        if context is None:
            return list(self.default_tools)
        if not self.tool_specs:
            return list(self.default_tools)
        merged_ctx: dict[str, object] = dict(self.base_context)
        merged_ctx.update(context)
        base_extra = self.base_context.get("extra_bindings")
        turn_extra = context.get("extra_bindings")
        if isinstance(base_extra, Mapping) and isinstance(turn_extra, Mapping):
            merged_ctx["extra_bindings"] = {**base_extra, **turn_extra}
        raw_proj = merged_ctx.get("project_root")
        proj = (
            Path(str(raw_proj)).resolve()
            if isinstance(raw_proj, (str, Path))
            else (self.project_root or Path.cwd())
        )
        bindings = self._build_plugin_context(proj, merged_ctx).tool_bindings()
        tools: list[Tool] = []
        for spec in self.tool_specs:
            if spec.is_function_tool:
                tools.append(
                    _normalize_plugin_tool(
                        spec.raw_tool,
                        source_name=spec.source_name or "plugin",
                        context=bindings,
                        metadata=spec.metadata,
                        document=spec.document,
                    )
                )
            else:
                cloned, _, _ = _clone_custom_tool(
                    spec.raw_tool,
                    source_name=spec.source_name,
                    metadata=spec.metadata,
                    document=spec.document,
                )
                tools.append(cloned)
        return tools

    # 统一触发全部已加载 Plugin 实例的 terminate() 停止清理钩子
    def terminate_all(self) -> None:
        for plugin in self.plugins:
            plugin.terminate()
