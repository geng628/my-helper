"""Layer 2 通用插件管理器（PluginManager）：统一扫描插件文件，将 @tool 包装为动态 Tool 子类并收集 RETRIEVERS / LLM_PROVIDERS / CONFIG_COMPILERS。"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from personal_assistant.agent.tools import (
    Tool,
    ToolDocument,
    ToolMetadata,
    ToolRegistry,
    build_dynamic_plugin_tool,
)
from personal_assistant.ai.client import ProviderFactory

RetrieverFactory = Callable[[dict[str, object]], object]
ConfigCompiler = Callable[..., dict[str, object]]
PluginToolFactory = Callable[[Mapping[str, object]], list[Tool]]


# 解析配置中的插件文件或目录路径，拦截项目外越权路径并过滤掉下划线开头的辅助文件
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

        candidates = (
            [p for p in sorted(target.glob("*.py")) if not p.name.startswith("_")]
            if target.is_dir()
            else [target]
        )
        for file_path in candidates:
            resolved = file_path.resolve()
            if not resolved.is_relative_to(root) or resolved.suffix != ".py" or not resolved.is_file():
                continue
            if resolved not in seen_files:
                seen_files.add(resolved)
                files.append(resolved)
    return files


# 动态导入单个插件 .py 文件并注册到 sys.modules，确保 @dataclass 等反射特性正常工作
def _import_plugin_module(file_path: Path) -> ModuleType:
    digest = hashlib.sha1(str(file_path).encode("utf-8")).hexdigest()[:10]
    module_name = f"my_helper_plugin_{file_path.stem}_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载插件文件：{file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


# 确保插件产出的 Tool 实例统一为动态 type() Tool 子类并附带标准 ToolMetadata 与 ToolDocument
def _normalize_plugin_tool(item: Tool, source_name: str) -> Tool:
    meta = getattr(item, "__tool_metadata__", None) or ToolMetadata()
    doc = ToolDocument(source_type="plugin", source_name=source_name)
    if type(item) is Tool:
        wrapped = build_dynamic_plugin_tool(
            item.run,
            description=item.description,
            parameters=item.parameters,
            name=item.name,
            param_enums=item.param_enums,
            risk=meta.risk,
            always_on=meta.always_on,
            source_name=source_name,
        )
        return wrapped
    object.__setattr__(item, "__tool_metadata__", meta)
    object.__setattr__(item, "__tool_document__", doc)
    return item


# 从已导入插件模块中收集 TOOLS 列表、@tool 装饰函数（以及可选上下文绑定工厂 build_plugin_tools）
def _extract_module_tools(
    module: ModuleType,
    file_path: Path,
    context: Mapping[str, object] | None = None,
) -> list[Tool]:
    source_name = file_path.stem
    found: list[Tool] = []
    seen_in_file: set[str] = set()

    context_builder = getattr(module, "build_plugin_tools", None)
    if context is not None and callable(context_builder):
        built = context_builder(context)
        if not isinstance(built, (list, tuple)):
            raise ValueError(f"插件 {file_path.name} 的 build_plugin_tools 必须返回列表或元组。")
        for item in built:
            if not isinstance(item, Tool):
                raise ValueError(f"插件 {file_path.name} 的 build_plugin_tools 包含非 Tool 对象。")
            if item.name not in seen_in_file:
                seen_in_file.add(item.name)
                found.append(_normalize_plugin_tool(item, source_name))

    exported = getattr(module, "TOOLS", None)
    if exported is not None:
        if not isinstance(exported, (list, tuple)):
            raise ValueError(f"插件 {file_path.name} 的 TOOLS 必须是列表或元组。")
        for item in exported:
            if not isinstance(item, Tool):
                raise ValueError(f"插件 {file_path.name} 的 TOOLS 中包含非 Tool 对象。")
            if item.name not in seen_in_file:
                seen_in_file.add(item.name)
                found.append(_normalize_plugin_tool(item, source_name))

    for value in vars(module).values():
        marked = getattr(value, "__agent_tool__", None)
        if isinstance(marked, Tool) and marked.name not in seen_in_file:
            seen_in_file.add(marked.name)
            found.append(_normalize_plugin_tool(marked, source_name))

    return found


# 从已导入插件模块中收集字典型工厂映射（如 RETRIEVERS 或 LLM_PROVIDERS）及单对象简写声明
def _extract_named_factories(
    module: ModuleType,
    file_path: Path,
    *,
    dict_attr: str,
    name_attr: str,
    factory_attr: str,
) -> dict[str, Callable]:
    found: dict[str, Callable] = {}
    exported = getattr(module, dict_attr, None)
    if exported is not None:
        if not isinstance(exported, dict):
            raise ValueError(f"插件 {file_path.name} 的 {dict_attr} 必须是字典。")
        for name, factory in exported.items():
            if not isinstance(name, str) or not name.strip() or not callable(factory):
                raise ValueError(f"插件 {file_path.name} 的 {dict_attr} 项格式非法：{name!r}")
            found[name.strip()] = factory

    single_name = getattr(module, name_attr, None)
    single_factory = getattr(module, factory_attr, None)
    if isinstance(single_name, str) and single_name.strip() and callable(single_factory):
        found.setdefault(single_name.strip(), single_factory)

    return found


# 插件管理器：统一管理已加载插件模块，按运行时上下文生成动态 Tool 子类并注入同一个 ToolRegistry
@dataclass(frozen=True)
class PluginManager:
    modules: tuple[tuple[Path, ModuleType], ...]
    tools: tuple[Tool, ...]
    retrievers: dict[str, RetrieverFactory]
    providers: dict[str, ProviderFactory]
    compilers: dict[str, ConfigCompiler]

    # 扫描并导入配置的全部插件文件，一次性构建 PluginManager 实例
    @classmethod
    def from_entries(
        cls,
        entries: Iterable[str],
        project_root: Path,
        *,
        context: Mapping[str, object] | None = None,
    ) -> "PluginManager":
        loaded_modules: list[tuple[Path, ModuleType]] = []
        loaded_tools: list[Tool] = []
        seen_tools: set[str] = set()
        loaded_retrievers: dict[str, RetrieverFactory] = {}
        loaded_providers: dict[str, ProviderFactory] = {}
        loaded_compilers: dict[str, ConfigCompiler] = {}

        for file_path in _resolve_plugin_targets(entries, project_root):
            module = _import_plugin_module(file_path)
            loaded_modules.append((file_path, module))
            for item in _extract_module_tools(module, file_path, context=context):
                if item.name in seen_tools:
                    raise ValueError(f"插件工具名称重复：{item.name}（文件：{file_path.name}）")
                seen_tools.add(item.name)
                loaded_tools.append(item)

            for name, factory in _extract_named_factories(
                module,
                file_path,
                dict_attr="RETRIEVERS",
                name_attr="RETRIEVER_NAME",
                factory_attr="create_retriever",
            ).items():
                if name in loaded_retrievers:
                    raise ValueError(f"检索插件名称重复：{name}（文件：{file_path.name}）")
                loaded_retrievers[name] = factory

            for name, factory in _extract_named_factories(
                module,
                file_path,
                dict_attr="LLM_PROVIDERS",
                name_attr="PROVIDER_NAME",
                factory_attr="create_provider",
            ).items():
                if name in loaded_providers:
                    raise ValueError(f"模型 Provider 插件名称重复：{name}（文件：{file_path.name}）")
                loaded_providers[name] = factory

            for name, compiler in _extract_named_factories(
                module,
                file_path,
                dict_attr="CONFIG_COMPILERS",
                name_attr="COMPILER_NAME",
                factory_attr="compile_config",
            ).items():
                if name in loaded_compilers:
                    raise ValueError(f"配置编译器插件名称重复：{name}（文件：{file_path.name}）")
                loaded_compilers[name] = compiler

        return cls(
            modules=tuple(loaded_modules),
            tools=tuple(loaded_tools),
            retrievers=loaded_retrievers,
            providers=loaded_providers,
            compilers=loaded_compilers,
        )

    # 按给定运行时上下文（如实例专属 data_root / offload_dir / tools.json 配置）构建全部插件 Tool 子类列表
    def build_tools(self, context: Mapping[str, object] | None = None) -> list[Tool]:
        if context is None:
            return list(self.tools)
        built: list[Tool] = []
        seen: set[str] = set()
        for file_path, module in self.modules:
            for item in _extract_module_tools(module, file_path, context=context):
                if item.name not in seen:
                    seen.add(item.name)
                    built.append(item)
        return built

    # 将全部插件工具连同 _metadata 与 _documents（source_type="plugin"）统一注册进目标 ToolRegistry
    def register_tools_into(
        self,
        registry: ToolRegistry,
        *,
        context: Mapping[str, object] | None = None,
    ) -> None:
        for item in self.build_tools(context=context):
            meta = getattr(item, "__tool_metadata__", None) or ToolMetadata()
            doc = getattr(item, "__tool_document__", None) or ToolDocument(
                source_type="plugin",
                source_name="plugin",
            )
            registry.register(item, metadata=meta, document=doc)


# 一次扫描同时加载插件工具、插件检索后端、插件 LLM Provider 与配置编译器插件
def load_plugins(
    entries: Iterable[str],
    project_root: Path,
) -> tuple[
    list[Tool],
    dict[str, RetrieverFactory],
    dict[str, ProviderFactory],
    dict[str, ConfigCompiler],
]:
    manager = PluginManager.from_entries(entries, project_root)
    return list(manager.tools), manager.retrievers, manager.providers, manager.compilers


# 批量加载配置声明的全部插件工具，并禁止跨文件重名
def load_plugin_tools(entries: Iterable[str], project_root: Path) -> list[Tool]:
    tools, _, _, _ = load_plugins(entries, project_root)
    return tools


# 批量加载配置声明的全部插件检索后端工厂（如 fts5_bm25），并禁止跨文件重名
def load_plugin_retrievers(
    entries: Iterable[str],
    project_root: Path,
) -> dict[str, RetrieverFactory]:
    _, retrievers, _, _ = load_plugins(entries, project_root)
    return retrievers


# 批量加载配置声明的全部插件大模型 Provider 工厂（如自定义非 OpenAI 厂商适配器）
def load_plugin_providers(
    entries: Iterable[str],
    project_root: Path,
) -> dict[str, ProviderFactory]:
    _, _, providers, _ = load_plugins(entries, project_root)
    return providers


# 批量加载配置声明的全部外部配置编译器插件（如 models_dsl -> models.json）
def load_plugin_compilers(
    entries: Iterable[str],
    project_root: Path,
) -> dict[str, ConfigCompiler]:
    _, _, _, compilers = load_plugins(entries, project_root)
    return compilers
