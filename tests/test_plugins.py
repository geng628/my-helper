"""Unit tests for the user-editable Python tool plugin system."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from personal_assistant.commands import ToolsConfig
from personal_assistant.llm import LLMClient
from personal_assistant.plugins import load_plugin_tools
from personal_assistant.react import run_react


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
from personal_assistant.tools import Tool, tool

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
from personal_assistant.tools import tool

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
from personal_assistant.tools import tool

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
