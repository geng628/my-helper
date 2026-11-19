"""Dynamic plugin loader for user-editable Python tool files."""
from __future__ import annotations

import hashlib
import importlib.util
from collections.abc import Iterable
from pathlib import Path

from personal_assistant.tools import Tool


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


def _load_module_tools(file_path: Path) -> list[Tool]:
    digest = hashlib.sha1(str(file_path).encode("utf-8")).hexdigest()[:10]
    module_name = f"my_helper_plugin_{file_path.stem}_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载插件文件：{file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    found: list[Tool] = []
    seen_in_file: set[str] = set()

    exported = getattr(module, "TOOLS", None)
    if exported is not None:
        if not isinstance(exported, (list, tuple)):
            raise ValueError(f"插件 {file_path.name} 的 TOOLS 必须是列表或元组。")
        for item in exported:
            if not isinstance(item, Tool):
                raise ValueError(f"插件 {file_path.name} 的 TOOLS 中包含非 Tool 对象。")
            if item.name not in seen_in_file:
                seen_in_file.add(item.name)
                found.append(item)

    for value in vars(module).values():
        marked = getattr(value, "__agent_tool__", None)
        if isinstance(marked, Tool) and marked.name not in seen_in_file:
            seen_in_file.add(marked.name)
            found.append(marked)

    return found


def load_plugin_tools(entries: Iterable[str], project_root: Path) -> list[Tool]:
    """Load `@tool`-decorated functions and `TOOLS` lists from project plugin files."""
    loaded: list[Tool] = []
    seen_names: set[str] = set()
    for file_path in _resolve_plugin_targets(entries, project_root):
        for item in _load_module_tools(file_path):
            if item.name in seen_names:
                raise ValueError(f"插件工具名称重复：{item.name}（文件：{file_path.name}）")
            seen_names.add(item.name)
            loaded.append(item)
    return loaded
