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
def test_plugin_motherboard_abc_and_all_five_inner_protocols() -> None:
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


# 测试自定义 Plugin 与非工具插件在多次 build_tools/app.chat 中仅 initialize() 一次，且工具每轮拿到最新动态上下文
def test_plugin_initialize_runs_once_while_tools_rebind_dynamic_context(tmp_path: Path) -> None:
    from personal_assistant.assistant.app import AssistantApp
    from personal_assistant.assistant.common import REPO_ROOT

    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    (plugins_dir / "counter_plugins.py").write_text(
        """
from pathlib import Path
from personal_assistant.agent.plugins import Plugin
from personal_assistant.agent.tools import Tool, ToolDocument, ToolMetadata, tool

TOOL_INIT_CALLS = 0
OBSERVER_INIT_CALLS = 0


# 记录运行期动态注入参数的插件工具
@tool(description="返回当前注入的 before_msg_id 与 db 路径", parameters={"tag": "标签"})
def peek_turn_ctx(
    tag: str,
    *,
    before_msg_id: int | None = None,
    history_db_path: Path | None = None,
) -> str:
    return f"{tag}:{before_msg_id}:{Path(history_db_path).name if history_db_path else 'none'}"


# 测试用工具插件：统计 initialize() 调用次数并注册 @tool 与显式 Tool 两种工具
class CounterToolPlugin(Plugin):
    plugin_id = "counter_tools"

    # 首次初始化时累加计数并向主板注册工具
    def initialize(self) -> None:
        global TOOL_INIT_CALLS
        TOOL_INIT_CALLS += 1
        self.context.register_tool(peek_turn_ctx, source_name="counter_plugins")
        self.context.register_tool(
            Tool(
                name="peek_explicit_tool",
                description="显式 Tool 上下文探针",
                parameters={},
                handler=lambda before_msg_id=None: f"explicit:{before_msg_id}",
            ),
            source_name="counter_plugins",
            metadata=ToolMetadata(risk="read", always_on=True),
            document=ToolDocument(source_type="plugin", source_name="custom_doc"),
        )


# 测试用非工具观测插件：验证每轮 build_tools 绝不再重新实例化并执行 initialize()
class CounterObserver:
    name = "counter_obs"

    # 忽略事件广播
    def on_event(self, session_key: str, event: object) -> None:
        return None

    # 返回观测插件自身初始化计数
    def snapshot(self, *, session_key: str | None = None) -> dict[str, object]:
        return {"observer_init_calls": OBSERVER_INIT_CALLS}


# 测试用非工具插件子类
class CounterObserverPlugin(Plugin):
    plugin_id = "counter_obs"

    # 注册观测器工厂并统计初始化次数
    def initialize(self) -> None:
        global OBSERVER_INIT_CALLS
        OBSERVER_INIT_CALLS += 1
        self.context.register_observer("counter_obs", lambda _cfg: CounterObserver())
""",
        encoding="utf-8",
    )

    cfg_path = tmp_path / "config" / "tools.json"
    cfg_path.parent.mkdir()
    cfg_path.write_text(
        json.dumps(
            {
                "enabled_tools": ["peek_turn_ctx"],
                "plugins": ["plugins"],
                "commands": [],
            }
        ),
        encoding="utf-8",
    )

    cfg = ToolsConfig.load(cfg_path, project_root=tmp_path)
    mod = cfg.plugin_manager.modules[0][1]
    assert mod.TOOL_INIT_CALLS == 1
    assert mod.OBSERVER_INIT_CALLS == 1

    # 1. 多次 build_tools(ctx) 绝不再触发 initialize()，且每轮动态上下文与元数据完整生效
    reg1 = cfg.build_registry(data_root=tmp_path / "data", tool_context={"before_msg_id": 10})
    reg2 = cfg.build_registry(data_root=tmp_path / "data", tool_context={"before_msg_id": 25})
    assert mod.TOOL_INIT_CALLS == 1
    assert mod.OBSERVER_INIT_CALLS == 1
    assert reg1.execute("peek_turn_ctx", '{"tag": "t1"}') == "t1:10:none"
    assert reg2.execute("peek_turn_ctx", '{"tag": "t2"}') == "t2:25:none"
    assert reg1.execute("peek_explicit_tool", "{}") == "explicit:10"
    assert reg2.execute("peek_explicit_tool", "{}") == "explicit:25"
    assert reg2.document("peek_explicit_tool").source_name == "custom_doc"  # type: ignore[union-attr]
    assert reg2.metadata("peek_explicit_tool").always_on is True  # type: ignore[union-attr]

    # 2. 跨多轮 AssistantApp.chat() 与 status()：initialize() 仍严格保持 1 次，before_msg_id 逐轮递增
    llm = ScriptedLLM(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "peek_turn_ctx", "arguments": '{"tag": "r1"}'}}
                ],
            },
            {"role": "assistant", "content": "第1轮完成"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c2", "type": "function", "function": {"name": "peek_turn_ctx", "arguments": '{"tag": "r2"}'}}
                ],
            },
            {"role": "assistant", "content": "第2轮完成"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c3", "type": "function", "function": {"name": "peek_explicit_tool", "arguments": "{}"}}
                ],
            },
            {"role": "assistant", "content": "第3轮完成"},
        ]
    )
    app = AssistantApp(data_root=tmp_path / "app_data", config_path=cfg_path, project_root=tmp_path, llm=llm)
    try:
        app_mod = app.tools_config.plugin_manager.modules[0][1]  # type: ignore[union-attr]
        # 新建 AssistantApp 时再次执行 ToolsConfig.load 初始化 1 次（累计 2 次），后续每轮对话不得再递增
        assert app_mod.TOOL_INIT_CALLS == 2
        assert app_mod.OBSERVER_INIT_CALLS == 2

        t1 = app.chat("第一句", chat_id="s1")
        t2 = app.chat("第二句", chat_id="s1")
        t3 = app.chat("第三句", chat_id="s1")
        _ = app.status()

        assert app_mod.TOOL_INIT_CALLS == 2
        assert app_mod.OBSERVER_INIT_CALLS == 2
        assert t1["tool_calls"][0]["observation"] == "r1:1:chat_history.sqlite3"
        assert t2["tool_calls"][0]["observation"] == "r2:3:chat_history.sqlite3"
        assert t3["tool_calls"][0]["observation"] == "explicit:5"
    finally:
        app.close()

    # 3. 验证项目自带 5 个真实插件在真实 config/tools.json 下多轮调用同样只初始化一次
    real_cfg = ToolsConfig.load(REPO_ROOT / "config" / "tools.json", project_root=REPO_ROOT)
    for msg_id in (1, 3, 5):
        reg = real_cfg.build_registry(data_root=tmp_path / "real", tool_context={"before_msg_id": msg_id})
        assert "search_chat_history" in reg.names


# 测试自定义 Tool 子类保留子类类型/重写 run/always_on 且不污染 **values，以及省略 source_name 的函数工具重绑与 base_context 合并
def test_custom_tool_subclass_preserved_and_base_context_merged(tmp_path: Path) -> None:
    from personal_assistant.agent.tools import Tool, ToolMetadata, ToolRegistry

    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    (plugins_dir / "subclass_and_fn_plugin.py").write_text(
        """
from collections.abc import Mapping
from pathlib import Path
from personal_assistant.agent.plugins import Plugin
from personal_assistant.agent.tools import Tool, ToolMetadata, build_tool


# 真正的自定义 Tool 子类（重写 run 且含 **values），绝不能被 build_tool 洗壳或注入上下文参数
class StrictSubclassTool(Tool):
    # 直接校验只收到模型声明的 text 参数，绝无 project_root/data_root 混入
    def run(self, **values: str) -> str:
        assert set(values) == {"text"}, f"unexpected kwargs: {values!r}"
        return f"subclass:{self.name}:{values['text']}"


# 测试用函数型工具：省略 source_name 注册时也必须能动态绑定 data_root / config_path / web_config
def probe_omitted_source(
    tag: str,
    *,
    data_root: Path | None = None,
    config_path: Path | None = None,
    web_config: Mapping[str, object] | None = None,
    before_msg_id: int | None = None,
) -> str:
    data_name = Path(data_root).name if data_root is not None else "none"
    cfg_name = Path(config_path).name if config_path is not None else "none"
    timeout_val = web_config.get("timeout") if web_config is not None else "none"
    return f"{tag}:{data_name}:{cfg_name}:{timeout_val}:{before_msg_id}"


FN_BUILT_TOOL = build_tool(
    probe_omitted_source,
    description="省略 source_name 的 build_tool 产物",
    parameters={"tag": "标签"},
)


# 同时注册自定义 Tool 子类（带/不带 source_name）与省略 source_name 的函数型工具
class MixedCustomPlugin(Plugin):
    plugin_id = "mixed_custom"

    # 注册三类工具
    def initialize(self) -> None:
        self.context.register_tool(
            StrictSubclassTool(
                name="sub_with_src",
                description="带 source_name 的自定义子类",
                parameters={"text": "文本"},
            ),
            source_name="subclass_src",
            metadata=ToolMetadata(risk="read", always_on=True),
        )
        self.context.register_tool(
            StrictSubclassTool(
                name="sub_no_src",
                description="省略 source_name 的自定义子类",
                parameters={"text": "文本"},
            ),
            metadata=ToolMetadata(risk="exec", always_on=True),
        )
        self.context.register_tool(FN_BUILT_TOOL)
""",
        encoding="utf-8",
    )

    cfg_path = tmp_path / "tools.json"
    cfg_path.write_text("{}", encoding="utf-8")
    manager = PluginManager.from_entries(
        ["plugins/subclass_and_fn_plugin.py"],
        tmp_path,
        context={
            "project_root": tmp_path,
            "config_path": cfg_path,
            "raw_config": {"web": {"timeout": 9}},
        },
    )
    mod = manager.modules[0][1]

    # 局部上下文仅传 data_root 与 extra_bindings，必须保留 base_context 中的 config_path 与 web_config
    reg = ToolRegistry(allowed_names=["probe_omitted_source"])
    manager.register_into(
        reg,
        context={
            "data_root": tmp_path / "data_turn_a",
            "extra_bindings": {"before_msg_id": 31},
        },
    )

    # 自定义子类保持原子类类型、always_on=True 常驻激活，且 **values 不被上下文参数污染
    assert set(reg.names) == {"probe_omitted_source", "sub_with_src", "sub_no_src"}
    sub_a = reg._tools["sub_with_src"]
    sub_b = reg._tools["sub_no_src"]
    assert type(sub_a) is mod.StrictSubclassTool
    assert type(sub_b) is mod.StrictSubclassTool
    assert reg.metadata("sub_with_src").always_on is True  # type: ignore[union-attr]
    assert reg.metadata("sub_no_src").risk == "exec"  # type: ignore[union-attr]
    assert reg.document("sub_with_src").source_name == "subclass_src"  # type: ignore[union-attr]
    assert reg.document("sub_no_src").source_name == "mixed_custom"  # type: ignore[union-attr]
    assert reg.execute("sub_with_src", '{"text": "ok"}') == "subclass:sub_with_src:ok"
    assert reg.execute("sub_no_src", '{"text": "ok"}') == "subclass:sub_no_src:ok"

    # 省略 source_name 的函数型工具同样在 build_tools(ctx) 中拿到动态覆盖 + 初始 base_context
    assert reg.execute("probe_omitted_source", '{"tag": "t1"}') == "t1:data_turn_a:tools.json:9:31"

    reg2 = ToolRegistry(allowed_names=["probe_omitted_source"])
    manager.register_into(
        reg2,
        context={
            "data_root": tmp_path / "data_turn_b",
            "extra_bindings": {"before_msg_id": 45},
        },
    )
    assert reg2.execute("probe_omitted_source", '{"tag": "t2"}') == "t2:data_turn_b:tools.json:9:45"
    assert reg2._tools["sub_with_src"] is not sub_a


# 测试 PluginContext.tool_bindings 通用顶层配置映射与裸 Tool 实例零猴子补丁签名解析
def test_plugin_context_generic_section_config_mapping_and_bare_tool_rebind(tmp_path: Path) -> None:
    from personal_assistant.agent.tools import Tool, ToolRegistry

    # 验证任意顶层 Mapping 配置自动按 <section>_config 注入，且不覆盖显式 extra_bindings
    ctx = PluginContext(
        project_root=tmp_path,
        raw_config={
            "custom_search": {"top_k": 8},
            "web": {"timeout": 12},
            "enabled_tools": ["demo"],
        },
        extra_bindings={"custom_search_config": {"top_k": 3}},
    )
    bindings = ctx.tool_bindings()
    assert bindings["web_config"] == {"timeout": 12}
    assert bindings["custom_search_config"] == {"top_k": 3}
    assert "enabled_tools_config" not in bindings

    # 验证插件注册裸 Tool(handler=...) 时自动解析 accepted_context_keys 与 optional required，且后续 build_tools 轻量化重绑
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    (plugins_dir / "bare_tool_plugin.py").write_text(
        """
from collections.abc import Mapping
from personal_assistant.agent.plugins import Plugin
from personal_assistant.agent.tools import Tool


# 测试裸 Tool 实例通过主板按通用配置键注入
class BareToolPlugin(Plugin):
    plugin_id = "bare_tool"

    # 注册带可选参数与配置键的裸 Tool 实例
    def initialize(self) -> None:
        # 接收模型参数与自动映射的 custom_mod_config 上下文
        def _run(query: str, mode: str = "fast", *, custom_mod_config: Mapping[str, object] | None = None) -> str:
            limit = custom_mod_config.get("limit") if custom_mod_config else "none"
            return f"{query}:{mode}:{limit}"

        self.context.register_tool(
            Tool(
                name="bare_probe",
                description="裸 Tool 探针",
                parameters={"query": "查询", "mode": "模式"},
                handler=_run,
            )
        )
""",
        encoding="utf-8",
    )
    manager = PluginManager.from_entries(
        ["plugins/bare_tool_plugin.py"],
        tmp_path,
        context={"raw_config": {"custom_mod": {"limit": 7}}},
    )
    reg = ToolRegistry(allowed_names=["bare_probe"])
    manager.register_into(reg)
    assert reg.execute("bare_probe", '{"query": "hello"}') == "hello:fast:7"

    reg_override = ToolRegistry(allowed_names=["bare_probe"])
    manager.register_into(reg_override, context={"raw_config": {"custom_mod": {"limit": 15}}})
    assert reg_override.execute("bare_probe", '{"query": "hi", "mode": "deep"}') == "hi:deep:15"
