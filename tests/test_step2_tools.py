"""Unit tests for Step 2: pure allowlisted ToolRegistry and fixed-argv commands."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.ai import AssistantMessage, Context, StreamDone, ToolCall, UserMessage
from personal_assistant.agent.react import TurnEnd, run_react
from personal_assistant.agent.tools import Tool, ToolRegistry


# 测试用脚本化模拟模型（实现 ModelInvoker）
class ScriptedLLM:
    last_usage = None

    # 初始化预设回复序列
    def __init__(self, responses: list[AssistantMessage]) -> None:
        self._responses = list(responses)

    # 流式弹出下一条预设中立回复
    def stream(self, context: Context, options=None):
        del context, options
        msg = self._responses.pop(0)
        yield StreamDone(reason="toolUse" if msg.tool_calls else "stop", message=msg)

    # 同步返回下一条预设中立回复
    def complete(self, context: Context, options=None) -> AssistantMessage:
        del context, options
        return self._responses.pop(0)


# 测试白名单过滤导出 Schema 并拦截未启用工具调用
def test_registry_allowlist_filters_schemas_and_blocks_calls() -> None:
    read_tool = Tool("read_note", "读取笔记", {"path": "路径"}, lambda path: f"content:{path}")
    secret_tool = Tool("delete_all", "危险工具", {}, lambda: "deleted!")

    registry = ToolRegistry(read_tool, secret_tool, allowed_names=["read_note"])

    assert registry.names == ["read_note"]
    assert [s.name for s in registry.schemas()] == ["read_note"]
    assert registry.execute("read_note", '{"path": "a.md"}') == "content:a.md"
    blocked = registry.execute("delete_all", "{}")
    assert "不在白名单中" in blocked


# 测试加载项目真实 config/tools.json 与统一工具注册表
def test_load_real_project_tools_config() -> None:
    from collections.abc import Mapping
    from personal_assistant.agent.tools import ToolsetProvider

    repo_root = Path(__file__).resolve().parent.parent
    cfg = ToolsConfig.load(repo_root / "config" / "tools.json", project_root=repo_root)

    assert "search_markdown" in cfg.enabled_tools
    assert [cmd.tool_name for cmd in cfg.commands] == ["cmd_git_status", "cmd_disk_usage"]

    # 模拟内置工具提供方（同时含白名单内外工具）
    class _BuiltinStubProvider(ToolsetProvider):
        # 产出内置桩工具列表
        def build_tools(self, context: Mapping[str, object] | None = None) -> list[Tool]:
            return [
                Tool("search_markdown", "搜笔记", {"query": "词"}, lambda query: f"hit:{query}"),
                Tool("unlisted_tool", "未在白名单的工具", {}, lambda: "nope"),
            ]

    with pytest.raises(TypeError):
        cfg.build_registry(Tool("x", "x", {}, lambda: "x"))  # type: ignore[misc]
    with pytest.raises(TypeError):
        cfg.build_registry(extra_tools=[])  # type: ignore[call-arg]

    registry = cfg.build_registry(providers=[_BuiltinStubProvider()])
    assert registry.names == [
        "search_markdown",
        "get_current_time",
        "search_chat_history",
        "run_command",
        "save_study_card",
        "list_due_reviews",
        "web_search",
        "fetch_webpage",
        "cmd_git_status",
        "cmd_disk_usage",
    ]
    assert "不在白名单中" in registry.execute("unlisted_tool", "{}")
    # Plugin tools not in enabled_tools stay blocked by the allowlist
    assert "不在白名单中" in registry.execute("add_numbers", '{"a": "1", "b": "2"}')


# 测试固定命令配置非法占位符、越界路径与超时校验
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


# 测试固定命令执行器剥离敏感环境变量、参数枚举校验与超长截断
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
    first_schema = registry.schemas()[0]
    assert first_schema.param_enums == {"mode": ("fast", "safe")}

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


# 测试固定命令超时处理与 ReAct 循环集成执行
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
            AssistantMessage(
                blocks=(ToolCall(id="call_cmd", name="cmd_greet", arguments='{"target": "GitHub"}'),),
                stop_reason="toolUse",
            ),
            AssistantMessage.from_text("命令返回了 hello GitHub。", stop_reason="stop"),
        ]
    )
    ends: list[TurnEnd] = []
    reply = run_react(
        llm,
        Context(messages=(UserMessage("跟 GitHub 打个招呼"),)),
        registry,
        on_event=lambda ev: ends.append(ev) if isinstance(ev, TurnEnd) else None,
    )
    assert reply == "命令返回了 hello GitHub。"
    assert "hello GitHub" in ends[0].messages[-2].content  # type: ignore[union-attr]


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


# 测试三类工具来源（内置记忆工具 / 插件 @tool / 固定命令）全部收敛为单一纯值对象 Tool，并消除空壳子类与死归档
def test_unified_tool_registry_converges_builtin_plugin_and_command_subclasses(
    tmp_path: Path,
) -> None:
    import personal_assistant.agent.tools as tools_mod
    import personal_assistant.assistant.builtin_tools as builtin_mod
    import personal_assistant.assistant.commands as commands_mod
    from personal_assistant.agent.tools import ToolMetadata, extract_tool, is_tool, tool
    from personal_assistant.assistant.app import AssistantApp
    from personal_assistant.assistant.common import REPO_ROOT

    assert not hasattr(tools_mod, "ToolDocument")
    assert not hasattr(ToolRegistry, "document")
    assert not hasattr(ToolRegistry, "activate")
    assert not hasattr(builtin_mod, "BuiltinTool")
    assert not hasattr(commands_mod, "CommandToolWrapper")

    app = AssistantApp(
        data_root=tmp_path / "data",
        config_path=REPO_ROOT / "config" / "tools.json",
        project_root=REPO_ROOT,
    )
    try:
        registry = app.tools

        # 1. 内核内置工具、插件工具与固定命令全部收敛为纯净 Tool 值对象（type(t) is Tool）
        for tool_name, expected_risk in (
            ("read_markdown", "read"),
            ("write_file", "write"),
            ("run_command", "exec"),
            ("cmd_git_status", "exec"),
        ):
            item = registry._tools[tool_name]
            assert type(item) is Tool
            assert registry.metadata(tool_name) == ToolMetadata(risk=expected_risk, always_on=False)

        # 2. @tool 装饰函数可直接传给 ToolRegistry.__init__ / register，always_on=True 在注册时立即激活
        @tool(description="常驻心跳", always_on=True)
        def always_ping() -> str:
            return "pong"

        assert is_tool(always_ping) is True
        assert type(extract_tool(always_ping)) is Tool
        registry.register(always_ping)
        assert "always_ping" in registry.names
        assert registry.execute("always_ping", "{}") == "pong"
    finally:
        app.close()


# 测试 ToolSchema.required_names 精确导出、零 OpenAI 方法泄漏与 4 位置参数构造兼容
def test_tool_schema_required_export_and_positional_compat() -> None:
    from personal_assistant.ai.protocol import ToolSchema

    # 1. required=None 时保持导出全部参数元组
    schema_all = ToolSchema(
        name="search",
        description="搜索",
        parameters={"query": "词", "root": "目录"},
    )
    assert schema_all.required_names == ("query", "root")
    assert not hasattr(schema_all, "to_openai_schema")

    # 2. 显式 required 按 parameters 声明顺序裁剪导出
    schema_sub = ToolSchema(
        name="search",
        description="搜索",
        parameters={"query": "词", "root": "目录", "mode": "模式"},
        required=("mode", "query"),
    )
    assert schema_sub.required_names == ("query", "mode")

    # 3. 全部参数可选时 required=() 导出空元组
    schema_opt = ToolSchema(
        name="list_items",
        description="列举",
        parameters={"root": "目录"},
        required=(),
    )
    assert schema_opt.required_names == ()

    # 4. 4 位置参数构造 Tool("name", "desc", {"p": "d"}, fn) 保持完全兼容
    pos_tool = Tool("echo", "回显", {"p": "参数"}, lambda p: f"ok:{p}")
    assert pos_tool.run(p="hi") == "ok:hi"
    assert pos_tool.required_names == ("p",)


# 测试 build_tool 自动识别带默认值可选参数、CommandSpec 全必填与零动态造类/猴子补丁
def test_build_tool_infers_required_and_accepted_context_keys(tmp_path: Path) -> None:
    from personal_assistant.agent.tools import ToolMetadata, build_tool
    from personal_assistant.assistant.commands import CommandSpec, ParamSpec

    # 同时含必选参数、POSITIONAL_OR_KEYWORD 默认值、KEYWORD_ONLY 默认值与上下文参数
    def search_fn(
        query: str,
        root: str = "default_root",
        *,
        limit: str = "10",
        turn_id: int | None = None,
    ) -> str:
        return f"{query}|{root}|{limit}|{turn_id}"

    built = build_tool(
        search_fn,
        description="搜索",
        parameters={"query": "查询", "root": "目录", "limit": "条数"},
        risk="read",
        always_on=True,
    )
    assert type(built) is Tool
    assert not hasattr(built, "_raw_fn")
    assert not hasattr(built, "__tool_metadata__")
    assert not hasattr(built, "document")
    assert built.metadata == ToolMetadata(risk="read", always_on=True)
    assert built.required == ("query",)
    assert built.accepted_context_keys == frozenset({"turn_id"})
    assert built.required_names == ("query",)

    # 显式传入 required 覆盖自动推断（按 parameters 声明顺序保留）
    explicit = build_tool(
        search_fn,
        description="搜索",
        parameters={"query": "查询", "root": "目录", "limit": "条数"},
        required=("limit", "query"),
    )
    assert explicit.required == ("query", "limit")
    assert explicit.required_names == ("query", "limit")

    # 含 **kwargs 的函数接受任意上下文键（accepted_context_keys=None）
    def flex_fn(query: str, **extra: object) -> str:
        return f"{query}:{sorted(extra)}"

    flex_tool = build_tool(flex_fn, description="灵活", parameters={"query": "查询"})
    assert flex_tool.required == ("query",)
    assert flex_tool.accepted_context_keys is None

    # CommandSpec.run(self, **values) 通过 accept_context=False 隔离上下文且 params 全为必填
    spec = CommandSpec(
        name="demo_cmd",
        description="示例命令",
        argv=("echo", "{mode}"),
        params={"mode": ParamSpec(description="模式", enum=("fast", "safe"))},
        cwd=tmp_path,
    )
    cmd_tool = spec.to_tool()
    assert type(cmd_tool) is Tool
    assert cmd_tool.required == ("mode",)
    assert cmd_tool.accepted_context_keys == frozenset()
    assert cmd_tool.required_names == ("mode",)


# 测试 ToolRegistry.execute 对可选参数省略/null 生效默认值，并严厉拦截非法参数
def test_tool_registry_execute_optional_params_and_validation() -> None:
    from personal_assistant.agent.tools import build_tool

    # 含必选参数、默认值参数与枚举默认值参数
    def query_items(
        query: str,
        root: str = "default_dir",
        *,
        mode: str = "fast",
        limit: str = "5",
    ) -> str:
        return f"q={query}|root={root}|mode={mode}|limit={limit}"

    registry = ToolRegistry(
        build_tool(
            query_items,
            description="检索",
            parameters={"query": "词", "root": "目录", "mode": "模式", "limit": "条数"},
            param_enums={"mode": ("fast", "deep")},
        ),
        Tool("ping", "心跳", {}, lambda: "pong"),
    )

    # 省略可选参数与传 null 均自动使用 Python 默认值
    assert registry.execute("query_items", '{"query": "rust"}') == "q=rust|root=default_dir|mode=fast|limit=5"
    assert (
        registry.execute("query_items", '{"query": "rust", "root": null, "mode": null}')
        == "q=rust|root=default_dir|mode=fast|limit=5"
    )
    # 传入可选参数（含数字自动转字符串）可正常覆盖默认值
    assert (
        registry.execute("query_items", '{"query": "rust", "root": "docs", "mode": "deep", "limit": 20}')
        == "q=rust|root=docs|mode=deep|limit=20"
    )

    # 未知参数、缺必填参数、必填传 null、可选传空白/布尔/非法枚举、非 JSON 对象及无参传值仍准确报错
    assert "需要参数" in registry.execute("query_items", '{"query": "rust", "unknown": "x"}')
    assert "需要参数" in registry.execute("query_items", '{"root": "docs"}')
    assert "必须是非空字符串" in registry.execute("query_items", '{"query": null}')
    assert "必须是非空字符串" in registry.execute("query_items", '{"query": "rust", "root": "  "}')
    assert "必须是非空字符串" in registry.execute("query_items", '{"query": "rust", "limit": true}')
    assert "只能取：fast, deep" in registry.execute("query_items", '{"query": "rust", "mode": "bad"}')
    assert "必须是 JSON 对象" in registry.execute("query_items", '["rust"]')
    assert "不接受参数" in registry.execute("ping", '{"x": "1"}')


# 测试执行期 context 动态注入、bind_tool_context 纯函数重绑与 accept_context=False 上下文隔离保护
def test_tool_registry_execute_runtime_context_and_subclass_isolation() -> None:
    from personal_assistant.agent.tools import ToolMetadata, bind_tool_context, build_tool

    # 声明静态/动态上下文键的函数工具
    def ctx_fn(query: str, *, env: str = "unset", turn_id: int | None = None) -> str:
        return f"q={query}|env={env}|turn={turn_id}"

    # 接受任意 **kwargs 上下文的灵活工具
    def kwargs_fn(query: str, **ctx: object) -> str:
        return f"q={query}|keys={sorted(ctx)}|turn={ctx.get('turn_id')}"

    # 关闭 accept_context=False 的 **values 工具：绝不能把 bound_kwargs 或执行期 context 混入 **values
    def strict_values_fn(**values: str) -> str:
        assert set(values) == {"text"}, f"unexpected kwargs: {values!r}"
        return f"sub:{values['text']}"

    t_ctx = build_tool(
        ctx_fn,
        description="上下文探针",
        parameters={"query": "词"},
        context={"env": "prod", "turn_id": 0, "ignored": "x", "query": "BLOCKED"},
    )
    assert t_ctx.bound_kwargs == {"env": "prod", "turn_id": 0}

    t_kw = build_tool(kwargs_fn, description="kwargs探针", parameters={"query": "词"})
    t_sub = build_tool(
        strict_values_fn,
        name="strict_sub",
        description="隔离工具",
        parameters={"text": "文本"},
        accept_context=False,
    )

    reg = ToolRegistry(t_ctx, t_kw, t_sub)
    # 1. 执行期 context 覆盖 turn_id、保留 env、忽略未声明键，且绝不覆盖模型声明参数 query
    res1 = reg.execute(
        "ctx_fn",
        '{"query": "hello"}',
        context={"turn_id": 42, "ignored": "bad", "query": "HIJACK", "env": None},
    )
    assert res1 == "q=hello|env=prod|turn=42"

    # 2. 不传 context 时回落初始 bound_kwargs（前次执行未污染工具状态）
    assert reg.execute("ctx_fn", '{"query": "again"}') == "q=again|env=prod|turn=0"

    # 3. **kwargs 工具接收除 parameters 外所有非 None 上下文键，绝不覆盖 query
    assert (
        reg.execute("kwargs_fn", '{"query": "ok"}', context={"turn_id": 7, "query": "HIJACK", "skip": None})
        == "q=ok|keys=['turn_id']|turn=7"
    )

    # 4. accept_context=False 工具绝不混入 context
    assert reg.execute("strict_sub", '{"text": "safe"}', context={"turn_id": 9, "extra": "bad"}) == "sub:safe"

    # 5. bind_tool_context 纯函数返回新实例且不篡改原实例
    rebound = bind_tool_context(
        t_ctx,
        {"turn_id": 88, "ignored": "nope"},
        metadata=ToolMetadata(risk="write", always_on=True),
    )
    assert rebound is not t_ctx
    assert t_ctx.bound_kwargs["turn_id"] == 0
    assert rebound.bound_kwargs == {"env": "prod", "turn_id": 88}
    assert rebound.metadata == ToolMetadata(risk="write", always_on=True)

    # 6. bind_tool_context 对 accept_context=False 工具同样不混入 bound_kwargs
    rebound_sub = bind_tool_context(t_sub, {"turn_id": 5}, metadata=ToolMetadata(risk="exec", always_on=True))
    assert rebound_sub is not t_sub
    assert type(rebound_sub) is Tool
    assert rebound_sub.bound_kwargs == {}
    assert rebound_sub.run_with_context({"text": "clean"}, {"turn_id": 99}) == "sub:clean"
