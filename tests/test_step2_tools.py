"""Unit tests for Step 2: pure allowlisted ToolRegistry and fixed-argv commands."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.ai import LLMClient
from personal_assistant.agent.react import run_react
from personal_assistant.agent.tools import Tool, ToolRegistry


class ScriptedLLM(LLMClient):
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = list(responses)

    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return self._responses.pop(0)


def test_registry_allowlist_filters_schemas_and_blocks_calls() -> None:
    read_tool = Tool("read_note", "读取笔记", {"path": "路径"}, lambda path: f"content:{path}")
    secret_tool = Tool("delete_all", "危险工具", {}, lambda: "deleted!")

    registry = ToolRegistry(read_tool, secret_tool, allowed_names=["read_note"])

    assert registry.names == ["read_note"]
    assert [d["function"]["name"] for d in registry.definitions()] == ["read_note"]
    assert registry.execute("read_note", '{"path": "a.md"}') == "content:a.md"
    blocked = registry.execute("delete_all", "{}")
    assert "不在白名单中" in blocked


def test_load_real_project_tools_config() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    cfg = ToolsConfig.load(repo_root / "config" / "tools.json", project_root=repo_root)

    assert "search_markdown" in cfg.enabled_tools
    assert [cmd.tool_name for cmd in cfg.commands] == ["cmd_git_status", "cmd_disk_usage"]

    builtin = [
        Tool("search_markdown", "搜笔记", {"query": "词"}, lambda query: f"hit:{query}"),
        Tool("unlisted_tool", "未在白名单的工具", {}, lambda: "nope"),
    ]
    registry = cfg.build_registry(*builtin)
    assert registry.names == [
        "search_markdown",
        "get_current_time",
        "search_chat_history",
        "run_command",
        "web_search",
        "fetch_webpage",
        "cmd_git_status",
        "cmd_disk_usage",
    ]
    assert "不在白名单中" in registry.execute("unlisted_tool", "{}")
    # Plugin tools not in enabled_tools stay blocked by the allowlist
    assert "不在白名单中" in registry.execute("add_numbers", '{"a": "1", "b": "2"}')


def test_command_spec_validation_errors(tmp_path: Path) -> None:
    bad_cases = [
        # Placeholder in argv[0]
        {"name": "bad_exe", "argv": ["{bin}", "status"]},
        # Placeholder without matching params
        {"name": "bad_param", "argv": ["echo", "{msg}"], "params": {}},
        # cwd escaping project root
        {"name": "bad_cwd", "argv": ["echo", "hi"], "cwd": "../outside"},
        # invalid timeout
        {"name": "bad_timeout", "argv": ["echo", "hi"], "timeout": 0},
    ]
    for case in bad_cases:
        cfg_file = tmp_path / "tools.json"
        cfg_file.write_text(json.dumps({"commands": [case]}), encoding="utf-8")
        with pytest.raises(ValueError):
            ToolsConfig.load(cfg_file, project_root=tmp_path)


def test_command_runner_sanitizes_env_validates_params_and_truncates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "super-secret-key-should-not-leak")
    monkeypatch.setenv("MY_HELPER_DATA", str(tmp_path / "data"))

    cfg_file = tmp_path / "tools.json"
    cfg_file.write_text(
        json.dumps(
            {
                "enabled_tools": [],
                "commands": [
                    {
                        "name": "show_mode",
                        "description": "安全输出模式参数并检查环境变量",
                        "enabled": True,
                        "argv": [
                            sys.executable,
                            "-c",
                            "import os, sys; print(sys.argv[1], os.getenv('LLM_API_KEY', 'MISSING'))",
                            "{mode}",
                        ],
                        "params": {
                            "mode": {
                                "description": "运行模式",
                                "enum": ["fast", "safe"],
                            }
                        },
                        "cwd": ".",
                        "timeout": 5,
                        "max_output_bytes": 40,
                    },
                    {
                        "name": "long_out",
                        "description": "输出截断测试",
                        "enabled": True,
                        "argv": [sys.executable, "-c", "print('A' * 100)"],
                        "max_output_bytes": 16,
                    },
                    {
                        "name": "disabled_cmd",
                        "description": "已禁用命令",
                        "enabled": False,
                        "argv": [sys.executable, "-c", "print('should not run')"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    cfg = ToolsConfig.load(cfg_file, project_root=tmp_path)
    registry = cfg.build_registry()

    # Disabled command is excluded
    assert registry.names == ["cmd_show_mode", "cmd_long_out"]
    schema = registry.definitions()[0]["function"]["parameters"]["properties"]["mode"]
    assert schema["enum"] == ["fast", "safe"]

    truncated = registry.execute("cmd_long_out", "{}")
    assert "输出已截断" in truncated
    assert "data/workspace/tool_outputs/" in truncated
    offloaded_files = list((tmp_path / "data" / "workspace" / "tool_outputs").glob("*.txt"))
    assert len(offloaded_files) == 1
    assert "A" * 100 in offloaded_files[0].read_text(encoding="utf-8")

    # Injection / out-of-enum value rejected cleanly
    bad = registry.execute("cmd_show_mode", '{"mode": "fast; rm -rf /"}')
    assert "错误" in bad
    assert "fast, safe" in bad

    # Valid call executes without leaking LLM_API_KEY
    ok = registry.execute("cmd_show_mode", '{"mode": "safe"}')
    assert "退出码：0" in ok
    assert "safe MISSING" in ok
    assert "super-secret-key" not in ok


def test_command_runner_handles_timeout_and_react_integration(tmp_path: Path) -> None:
    cfg_file = tmp_path / "tools.json"
    cfg_file.write_text(
        json.dumps(
            {
                "enabled_tools": [],
                "commands": [
                    {
                        "name": "greet",
                        "description": "输出问候语",
                        "enabled": True,
                        "argv": [sys.executable, "-c", "import sys; print('hello ' + sys.argv[1])", "{target}"],
                        "params": {
                            "target": {
                                "description": "目标名称（字母数字）",
                                "pattern": "^[a-zA-Z0-9_-]{1,20}$",
                            }
                        },
                        "timeout": 5,
                    },
                    {
                        "name": "slow",
                        "description": "超时命令",
                        "enabled": True,
                        "argv": [sys.executable, "-c", "import time; time.sleep(2)"],
                        "timeout": 0.1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = ToolsConfig.load(cfg_file, project_root=tmp_path)
    registry = cfg.build_registry()

    timeout_msg = registry.execute("cmd_slow", "{}")
    assert "超时" in timeout_msg

    # Pattern check blocks spaces/shell characters
    bad_target = registry.execute("cmd_greet", '{"target": "world && id"}')
    assert "参数格式不符合要求" in bad_target

    llm = ScriptedLLM(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_cmd",
                        "type": "function",
                        "function": {"name": "cmd_greet", "arguments": '{"target": "GitHub"}'},
                    }
                ],
            },
            {"role": "assistant", "content": "命令返回了 hello GitHub。"},
        ]
    )
    messages: list[dict[str, object]] = [{"role": "user", "content": "跟 GitHub 打个招呼"}]
    reply = run_react(llm, messages, registry)
    assert reply == "命令返回了 hello GitHub。"
    assert "hello GitHub" in str(messages[-2]["content"])


# 测试工具参数校验兼容模型直接传入数字（如 start_line: 1），同时依然拒绝布尔值与空串
def test_tool_registry_accepts_numeric_arguments_and_rejects_booleans() -> None:
    registry = ToolRegistry(
        Tool(
            "slice_lines",
            "读取指定行号范围",
            {"path": "路径", "start_line": "起始行", "end_line": "结束行"},
            lambda path, start_line, end_line: f"{path}:{start_line}-{end_line}",
        )
    )
    ok = registry.execute("slice_lines", '{"path": "a.md", "start_line": 1, "end_line": 40}')
    assert ok == "a.md:1-40"

    bad_bool = registry.execute("slice_lines", '{"path": "a.md", "start_line": true, "end_line": 40}')
    assert "错误" in bad_bool
    bad_empty = registry.execute("slice_lines", '{"path": "a.md", "start_line": " ", "end_line": 40}')
    assert "错误" in bad_empty


# 测试固定白名单命令超长输出接入 Head+Tail TOAST 卸载（同时合并 stdout 与 stderr）
def test_command_spec_offloads_long_stdout_and_stderr_with_head_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MY_HELPER_DATA", str(tmp_path / "data"))
    script = (
        "import sys; "
        "print('HEAD_BANNER_START'); "
        "[print(f'body line {i:02d}: detail_payload_{i}') for i in range(1, 45)]; "
        "print('TAIL_SUMMARY_OK'); "
        "print('FINAL_STDERR_LINE', file=sys.stderr)"
    )
    cfg_file = tmp_path / "tools.json"
    cfg_file.write_text(
        json.dumps(
            {
                "enabled_tools": [],
                "commands": [
                    {
                        "name": "verbose_job",
                        "description": "多行输出与错误汇总测试",
                        "enabled": True,
                        "argv": [sys.executable, "-c", script],
                        "max_output_bytes": 380,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    registry = ToolsConfig.load(cfg_file, project_root=tmp_path).build_registry()
    out = registry.execute("cmd_verbose_job", "{}")
    assert "退出码：0" in out
    assert "HEAD_BANNER_START" in out
    assert "FINAL_STDERR_LINE" in out
    assert "输出已截断" in out
    assert "body line 20" not in out
    offloaded = list((tmp_path / "data" / "workspace" / "tool_outputs").glob("*.txt"))
    assert len(offloaded) == 1
    full_saved = offloaded[0].read_text(encoding="utf-8")
    assert "body line 20" in full_saved
    assert "FINAL_STDERR_LINE" in full_saved


# 测试三类异构工具来源（BuiltinTool / 动态 type() PluginTool / CommandToolWrapper）统一汇聚进同一个 ToolRegistry（_tools + _metadata + _documents）并支持按需激活
def test_unified_tool_registry_converges_builtin_plugin_and_command_subclasses(
    tmp_path: Path,
) -> None:
    from personal_assistant.agent.tools import ToolDocument, ToolMetadata, tool
    from personal_assistant.assistant.app import AssistantApp
    from personal_assistant.assistant.commands import CommandToolWrapper
    from personal_assistant.assistant.common import REPO_ROOT
    from personal_assistant.assistant.memory import BuiltinTool

    app = AssistantApp(
        data_root=tmp_path / "data",
        config_path=REPO_ROOT / "config" / "tools.json",
        project_root=REPO_ROOT,
    )
    try:
        registry = app.build_tools()

        # 1. 内核内置工具 (MemoryToolsetProvider) ──▶ BuiltinTool(Tool) 子类
        read_tool = registry._tools["read_markdown"]
        assert isinstance(read_tool, BuiltinTool)
        assert isinstance(read_tool, Tool)
        assert registry.metadata("read_markdown") == ToolMetadata(risk="read", always_on=False)
        assert registry.document("read_markdown") == ToolDocument(
            source_type="builtin",
            source_name="memory",
        )
        assert registry.metadata("write_file") == ToolMetadata(risk="write", always_on=False)

        # 2. 插件 @tool (PluginManager) ──▶ 动态 type() Tool 子类
        run_cmd_tool = registry._tools["run_command"]
        assert isinstance(run_cmd_tool, Tool)
        assert type(run_cmd_tool) is not Tool
        assert issubclass(type(run_cmd_tool), Tool)
        assert registry.metadata("run_command") == ToolMetadata(risk="exec", always_on=False)
        assert registry.document("run_command") == ToolDocument(
            source_type="plugin",
            source_name="shell_tools",
        )

        # 3. 固定命令包装器 ──▶ CommandToolWrapper(Tool) 子类
        git_tool = registry._tools["cmd_git_status"]
        assert isinstance(git_tool, CommandToolWrapper)
        assert isinstance(git_tool, Tool)
        assert registry.metadata("cmd_git_status") == ToolMetadata(risk="exec", always_on=False)
        assert registry.document("cmd_git_status") == ToolDocument(
            source_type="command",
            source_name="tools.json",
        )

        # 4. 按需加载与 always_on 常驻激活
        @tool(description="常驻心跳", always_on=True)
        def always_ping() -> str:
            return "pong"

        registry.register(always_ping.__agent_tool__)
        registry.activate(["read_markdown"])
        assert set(registry.names) == {"read_markdown", "always_ping"}
    finally:
        app.close()
