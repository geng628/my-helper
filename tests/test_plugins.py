"""Unit tests for the user-editable Python tool plugin system."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.ai import LLMClient
from personal_assistant.agent.plugins import load_plugin_tools
from personal_assistant.agent.react import run_react


class ScriptedLLM(LLMClient):
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = list(responses)

    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return self._responses.pop(0)


def test_load_plugins_supports_decorator_and_tools_list_and_allowlist(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    (plugins_dir / "_helper.py").write_text("raise RuntimeError('should be ignored')\n", encoding="utf-8")
    (plugins_dir / "my_tools.py").write_text(
        """
from personal_assistant.agent.tools import Tool, tool

@tool(description="反转字符串", parameters={"text": "输入文本"})
def reverse_text(text: str) -> str:
    return text[::-1]

@tool(description="未加入白名单的工具")
def disabled_plugin() -> str:
    return "should not run"

TOOLS = [
    Tool(
        name="shout",
        description="转成大写",
        parameters={"text": "文本"},
        handler=lambda text: text.upper(),
    )
]
""",
        encoding="utf-8",
    )

    cfg_path = tmp_path / "config" / "tools.json"
    cfg_path.parent.mkdir()
    cfg_path.write_text(
        json.dumps(
            {
                "enabled_tools": ["reverse_text", "shout"],
                "plugins": ["plugins"],
                "commands": [],
            }
        ),
        encoding="utf-8",
    )

    cfg = ToolsConfig.load(cfg_path, project_root=tmp_path)
    registry = cfg.build_registry()

    assert registry.names == ["shout", "reverse_text"]
    assert registry.execute("reverse_text", '{"text": "abc"}') == "cba"
    assert registry.execute("shout", '{"text": "hello"}') == "HELLO"
    assert "不在白名单中" in registry.execute("disabled_plugin", "{}")


def test_plugin_security_rejects_outside_paths_and_duplicates(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="项目内相对路径"):
        load_plugin_tools(["../outside.py"], tmp_path)
    with pytest.raises(ValueError, match="项目内相对路径"):
        load_plugin_tools(["/etc/passwd"], tmp_path)

    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    code = """
from personal_assistant.agent.tools import tool

@tool(description="重复名称")
def dup_tool() -> str:
    return "x"
"""
    (plugins_dir / "a.py").write_text(code, encoding="utf-8")
    (plugins_dir / "b.py").write_text(code, encoding="utf-8")
    with pytest.raises(ValueError, match="重复"):
        load_plugin_tools(["plugins"], tmp_path)


def test_plugin_tool_runs_inside_react_loop(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    (plugins_dir / "math_plugin.py").write_text(
        """
from personal_assistant.agent.tools import tool

@tool(description="计算两数乘积", parameters={"x": "乘数1", "y": "乘数2"})
def multiply(x: str, y: str) -> str:
    return str(int(x) * int(y))
""",
        encoding="utf-8",
    )
    cfg_path = tmp_path / "tools.json"
    cfg_path.write_text(
        json.dumps({"enabled_tools": ["multiply"], "plugins": ["plugins/math_plugin.py"]}),
        encoding="utf-8",
    )
    cfg = ToolsConfig.load(cfg_path, project_root=tmp_path)
    registry = cfg.build_registry()

    llm = ScriptedLLM(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_mul",
                        "type": "function",
                        "function": {"name": "multiply", "arguments": '{"x": "6", "y": "7"}'},
                    }
                ],
            },
            {"role": "assistant", "content": "6 乘以 7 等于 42。"},
        ]
    )
    messages: list[dict[str, object]] = [{"role": "user", "content": "6 乘 7 等于多少？"}]
    reply = run_react(llm, messages, registry)
    assert reply == "6 乘以 7 等于 42。"
    assert messages[-2]["content"] == "42"


# 测试外层统一主板基类 Plugin(ABC)（__init_subclass__ 自动发现、self.context、initialize/terminate）与内层 4 个强类型极简子协议
def test_plugin_motherboard_abc_and_four_inner_protocols() -> None:
    from personal_assistant.agent.plugins import (
        ConfigCompilerProtocol,
        Plugin,
        PluginManager,
        RetrieverProtocol,
    )
    from personal_assistant.agent.tools import Tool
    from personal_assistant.assistant.common import REPO_ROOT
    from plugins.models_dsl import ModelsDSLCompiler

    manager = PluginManager.from_entries(["plugins"], REPO_ROOT)

    # 1. 外层：plugins/ 下全部 5 个插件模块均继承同一个 Plugin(ABC) 主板基类并自动发现初始化
    assert all(isinstance(p, Plugin) for p in manager.plugins)
    plugin_ids = {p.plugin_id for p in manager.plugins}
    assert plugin_ids == {
        "custom_tools",
        "shell_tools",
        "web_tools",
        "fts5_bm25",
        "models_dsl",
    }
    assert all(p.context.project_root == REPO_ROOT for p in manager.plugins)

    # 2. 内层子协议 1：Tool 子类实例全部汇入 manager.default_tools
    assert all(isinstance(t, Tool) for t in manager.default_tools)
    assert {"get_current_time", "run_command", "web_search", "fetch_webpage"}.issubset(
        {t.name for t in manager.default_tools}
    )

    # 3. 内层子协议 2：RetrieverProtocol（MemoryStore 只认 search_hits）
    fts_instance = manager.retrievers["fts5_bm25"]({})
    assert isinstance(fts_instance, RetrieverProtocol)
    assert fts_instance.name == "fts5_bm25"

    # 4. 内层子协议 3：ConfigCompilerProtocol（ToolsConfig 只认 compile / presets / render）
    dsl_compiler = ModelsDSLCompiler()
    assert isinstance(dsl_compiler, ConfigCompilerProtocol)
    compiled = manager.compilers["models_dsl"]("use ollama", "models.dsl")
    assert compiled["default"] == "ollama"

    # 5. 统一生命周期清理钩子可正常调用
    manager.terminate_all()
