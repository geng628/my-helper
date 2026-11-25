"""Unit tests for the user-editable Python tool plugin system."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.ai import LLMClient
from personal_assistant.agent.plugins import (
    ConfigCompilerProtocol,
    LLMProviderProtocol,
    Plugin,
    PluginContext,
    PluginManager,
)
from personal_assistant.agent.react import run_react


# 测试用脚本化模拟模型
class ScriptedLLM(LLMClient):
    # 初始化预设回复序列
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = list(responses)

    # 弹出并返回下一条预设回复
    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return self._responses.pop(0)


# 测试 @tool 装饰器插件、显式 Plugin 子类与白名单拦截机制
def test_load_plugins_supports_decorator_and_plugin_class_and_allowlist(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    (plugins_dir / "_helper.py").write_text("raise RuntimeError('should be ignored')\n", encoding="utf-8")
    (plugins_dir / "my_tools.py").write_text(
        """
from personal_assistant.agent.tools import tool

@tool(description="反转字符串", parameters={"text": "输入文本"})
def reverse_text(text: str) -> str:
    return text[::-1]

@tool(description="未加入白名单的工具")
def disabled_plugin() -> str:
    return "should not run"
""",
        encoding="utf-8",
    )
    (plugins_dir / "a_shout_plugin.py").write_text(
        """
from personal_assistant.agent.plugins import Plugin
from personal_assistant.agent.tools import Tool

# 测试用显式工具插件子类
class ShoutPlugin(Plugin):
    plugin_id = "shout_plugin"

    # 通过主板插槽注册大写转换工具
    def initialize(self) -> None:
        self.context.register_tool(
            Tool(
                name="shout",
                description="转成大写",
                parameters={"text": "文本"},
                handler=lambda text: text.upper(),
            ),
            source_name="a_shout_plugin",
        )
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


# 测试 PluginManager.from_entries 拒绝越界路径并拦截跨文件重名工具
def test_plugin_security_rejects_outside_paths_and_duplicates(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="项目内相对路径"):
        PluginManager.from_entries(["../outside.py"], tmp_path)
    with pytest.raises(ValueError, match="项目内相对路径"):
        PluginManager.from_entries(["/etc/passwd"], tmp_path)

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
        PluginManager.from_entries(["plugins"], tmp_path)


# 测试插件工具在完整 ReAct 循环中的执行与观察回传
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


# 测试外层统一主板基类 Plugin(ABC)（5 个同级插件）与内层全部 5 个子协议插槽
def test_plugin_motherboard_abc_and_four_inner_protocols() -> None:
    from personal_assistant.agent.plugins import (
        ConfigCompilerProtocol,
        LLMProviderProtocol,
        ObserverProtocol,
        Plugin,
        PluginManager,
        RetrieverProtocol,
    )
    from personal_assistant.agent.tools import Tool
    from personal_assistant.ai import LLMClient
    from personal_assistant.assistant.common import REPO_ROOT
    from plugins.models_dsl import ModelsDSLCompiler
    from plugins.reasoning_provider import OpenAIReasoningProvider
    from plugins.tools import ToolsPlugin

    manager = PluginManager.from_entries(["plugins"], REPO_ROOT)

    # 1. 外层：主板加载 5 个同级 Plugin(ABC) 实例（tools 文件夹插件 + 4 个单文件插件）
    assert len(manager.plugins) == 5
    assert all(isinstance(p, Plugin) for p in manager.plugins)
    plugin_ids = {p.plugin_id for p in manager.plugins}
    assert plugin_ids == {
        "tools",
        "fts5_bm25",
        "models_dsl",
        "openai_reasoning",
        "telemetry",
    }
    assert {type(p).__name__ for p in manager.plugins} == {
        "ToolsPlugin",
        "FTS5RetrieverPlugin",
        "ModelsDSLPlugin",
        "ReasoningProviderPlugin",
        "TelemetryPlugin",
    }
    assert all(p.context.project_root == REPO_ROOT for p in manager.plugins)

    # 2. 内层子协议 1：plugins/tools/ 下的全部 @tool 模块由唯一的 ToolsPlugin 统一装配为 Tool 子类实例
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
    assert not callable(dsl_compiler)
    assert isinstance(manager.compilers["models_dsl"], ConfigCompilerProtocol)
    compiled = manager.compilers["models_dsl"].compile("use ollama", "models.dsl")
    assert compiled["default"] == "ollama"

    # 5. 内层子协议 4：LLMProviderProtocol（from_model_profile / from_env 只认已注册 Provider 工厂）
    reasoning_provider = OpenAIReasoningProvider()
    assert isinstance(reasoning_provider, LLMProviderProtocol)
    assert not callable(reasoning_provider)
    assert "openai_reasoning" in manager.providers
    reasoning_client = manager.providers["openai_reasoning"]({"LLM_API_KEY": "sk-slot4-test"})
    assert isinstance(reasoning_client, LLMClient)
    assert type(reasoning_client).__name__ == "OpenAIReasoningLLM"

    # 6. 内层子协议 5：ObserverProtocol（AssistantApp 只认 on_event / snapshot）
    obs = manager.observers["telemetry"]({})
    assert isinstance(obs, ObserverProtocol)
    assert obs.name == "telemetry"

    # 7. 统一生命周期清理钩子可正常调用
    manager.terminate_all()


# 测试所有已移除的历史兼容胶水与旧版全局变量写法均彻底失效
def test_legacy_compat_shims_removed_and_rejected(tmp_path: Path) -> None:
    import personal_assistant.agent as agent_pkg
    import personal_assistant.agent.plugins as plugins_mod
    import personal_assistant.agent.tools as tools_mod
    import personal_assistant.assistant as assistant_pkg
    import personal_assistant.assistant.memory as memory_mod
    from personal_assistant.assistant.memory import MemoryStore

    repo_root = Path(__file__).resolve().parent.parent

    # 1. 不再把 plugins/<x>.py 静默兜底解析为 plugins/tools/<x>.py
    assert PluginManager.from_entries(["plugins/custom_tools.py"], repo_root).tools() == []

    # 2. 旧式魔法全局变量与 TOOLS = [...] 列表不再被静默扫描加载
    legacy_plugin = tmp_path / "legacy_plugin.py"
    legacy_plugin.write_text(
        """
from personal_assistant.agent.tools import Tool

TOOLS = [Tool(name="legacy_tool", description="旧列表工具", parameters={}, handler=lambda: "ok")]
RETRIEVERS = {"legacy_ret": lambda cfg: None}
RETRIEVER_NAME = "legacy_single_ret"
create_retriever = lambda cfg: None
LLM_PROVIDERS = {"legacy_prov": lambda env: None}
PROVIDER_NAME = "legacy_single_prov"
create_provider = lambda env: None
CONFIG_COMPILERS = {"legacy_comp": lambda text, source="": {}}
COMPILER_NAME = "legacy_single_comp"
compile_config = lambda text, source="": {}
""",
        encoding="utf-8",
    )
    legacy_mgr = PluginManager.from_entries([legacy_plugin.name], tmp_path)
    assert legacy_mgr.tools() == []
    assert legacy_mgr.retrievers == {}
    assert legacy_mgr.providers == {}
    assert legacy_mgr.compilers == {}

    # 3. 已清理的历史别名、薄壳与死接口不再存在于公开包与底层模块
    for sym in (
        "load_plugins",
        "load_plugin_tools",
        "load_plugin_retrievers",
        "load_plugin_providers",
        "load_plugin_compilers",
        "ConfigCompiler",
    ):
        assert not hasattr(agent_pkg, sym), f"agent_pkg 仍残留 {sym}"
        assert not hasattr(plugins_mod, sym), f"plugins_mod 仍残留 {sym}"

    assert not hasattr(agent_pkg, "build_dynamic_plugin_tool")
    assert not hasattr(tools_mod, "build_dynamic_plugin_tool")

    assert not hasattr(assistant_pkg, "memory_tools")
    assert not hasattr(memory_mod, "memory_tools")
    assert not hasattr(MemoryStore, "save_memory_entries")

    for attr in ("command_tools", "plugin_tools", "plugin_retrievers", "plugin_compilers"):
        assert not hasattr(ToolsConfig, attr), f"ToolsConfig 仍残留 {attr}"

    # 4. PluginContext 仅接受强类型 ConfigCompilerProtocol / LLMProviderProtocol，拒绝裸函数
    ctx = PluginContext(project_root=tmp_path)
    with pytest.raises((TypeError, ValueError)):
        ctx.register_compiler(lambda text, source="": {})  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        ctx.register_provider(lambda env: None)  # type: ignore[arg-type]


# 测试第 5 插槽 ObserverProtocol 的实例/工厂注册、重名校验与 ToolsConfig.build_observers 装配
def test_observer_protocol_slot_registration_and_app_isolation(tmp_path: Path) -> None:
    from personal_assistant.agent.plugins import ObserverProtocol

    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    (plugins_dir / "mem_observer.py").write_text(
        """
from personal_assistant.agent.plugins import Plugin

# 测试用内存观测器实现（符合 ObserverProtocol）
class InMemoryObserver:
    name = "mem_obs"

    # 初始化事件收集列表与配置副本
    def __init__(self, cfg: dict[str, object]) -> None:
        self.cfg = dict(cfg)
        self.events: list[object] = []

    # 收集微内核广播的事件
    def on_event(self, session_key: str, event: object) -> None:
        self.events.append((session_key, event))

    # 返回当前已收集事件数量与配置快照
    def snapshot(self, *, session_key: str | None = None) -> dict[str, object]:
        return {"count": len(self.events), "session_key": session_key, "tag": self.cfg.get("tag", "default")}


# 测试用观测插件（通过工厂函数注册，确保每个应用实例独享状态）
class MemObserverPlugin(Plugin):
    plugin_id = "mem_obs"

    # 向主板插槽 5 注册观测器工厂
    def initialize(self) -> None:
        self.context.register_observer("mem_obs", lambda cfg: InMemoryObserver(cfg))
""",
        encoding="utf-8",
    )

    cfg_path = tmp_path / "tools.json"
    cfg_path.write_text(
        json.dumps(
            {
                "enabled_tools": [],
                "plugins": ["plugins"],
                "telemetry": {"tag": "isolated"},
            }
        ),
        encoding="utf-8",
    )
    cfg = ToolsConfig.load(cfg_path, project_root=tmp_path)
    obs_list_a = cfg.build_observers()
    obs_list_b = cfg.build_observers()
    assert len(obs_list_a) == 1 and len(obs_list_b) == 1
    assert isinstance(obs_list_a[0], ObserverProtocol)
    assert obs_list_a[0] is not obs_list_b[0]
    obs_list_a[0].on_event("c1", "ping")  # type: ignore[arg-type]
    assert obs_list_a[0].snapshot(session_key="c1") == {
        "count": 1,
        "session_key": "c1",
        "tag": "isolated",
    }
    assert obs_list_b[0].snapshot()["count"] == 0

    ctx = PluginContext(project_root=tmp_path)
    with pytest.raises(ValueError, match="不能为空"):
        ctx.register_observer("  ", obs_list_a[0])
    ctx.register_observer("obs1", obs_list_a[0])
    with pytest.raises(ValueError, match="重复"):
        ctx.register_observer("obs1", obs_list_b[0])
    with pytest.raises((TypeError, ValueError)):
        ctx.register_observer("bad", 123)  # type: ignore[arg-type]
