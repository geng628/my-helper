"""三层漏斗架构守卫测试（TDD）：AST 依赖方向校验、Pi ch04 三层解耦守卫与子进程断层独立运行测试。"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_ROOT = REPO_ROOT / "personal_assistant"
EVALS_ROOT = REPO_ROOT / "evals"


# 判断模块名是否属于本仓库受控包（personal_assistant 或顶层 evals）
def _is_repo_internal(name: str) -> bool:
    return (
        name == "personal_assistant"
        or name.startswith("personal_assistant.")
        or name == "evals"
        or name.startswith("evals.")
    )


# 提取指定 Python 文件内部对 personal_assistant.* 与 evals.* 的全部模块依赖
def _internal_imports(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    deps: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                pkg_dir = py_file.parent
                for _ in range(node.level - 1):
                    pkg_dir = pkg_dir.parent
                rel_parts = pkg_dir.relative_to(REPO_ROOT).parts
                base = ".".join(rel_parts + ((node.module,) if node.module else ()))
            if _is_repo_internal(base):
                deps.add(base)
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    candidate = f"{base}.{alias.name}"
                    candidate_path = REPO_ROOT.joinpath(*candidate.split("."))
                    if candidate_path.with_suffix(".py").is_file() or candidate_path.is_dir():
                        deps.add(candidate)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _is_repo_internal(alias.name):
                    deps.add(alias.name)
    return deps


# 遍历顶层 evals/ 包内的受管 Python 源码（排除 gitignore 的 evals/public/ 与 __pycache__）
def _iter_evals_python_files() -> list[Path]:
    files: list[Path] = []
    for py_file in sorted(EVALS_ROOT.rglob("*.py")):
        rel_parts = py_file.relative_to(EVALS_ROOT).parts
        if "public" in rel_parts or "__pycache__" in rel_parts:
            continue
        files.append(py_file)
    return files


# 将包内文件路径转换为标准点分模块名（如 personal_assistant.ai.registry 或 evals.cli）
def _module_name(py_file: Path) -> str:
    rel = py_file.relative_to(REPO_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


# 方法 1：AST 依赖漏斗守卫——严禁任何跨层反向依赖、ai 内部三层越界、产品包依赖 evals/ 与循环依赖
def test_dependency_funnel_has_no_upward_or_cyclic_imports() -> None:
    assert (PKG_ROOT / "ai").is_dir(), "缺少 Layer 1 包 personal_assistant/ai/"
    assert (PKG_ROOT / "agent").is_dir(), "缺少 Layer 2 包 personal_assistant/agent/"
    assert (PKG_ROOT / "assistant").is_dir(), "缺少 Layer 3 包 personal_assistant/assistant/"
    assert not (PKG_ROOT / "eval").exists(), "评测引擎应迁移至顶层 evals/，personal_assistant/ 不得保留 eval/"
    assert not (PKG_ROOT / "ai" / "types.py").exists(), "旧版 ai/types.py 应已彻底删除"
    assert not (PKG_ROOT / "ai" / "client.py").exists(), "旧版 ai/client.py 应已彻底删除"
    subpackages = {p.name for p in PKG_ROOT.iterdir() if p.is_dir() and p.name != "__pycache__"}
    assert subpackages == {"ai", "agent", "assistant"}, (
        f"personal_assistant/ 仅允许保留纯净三层产品包，发现多余目录：{sorted(subpackages - {'ai', 'agent', 'assistant'})}"
    )
    assert (EVALS_ROOT / "__init__.py").is_file(), "缺少顶层评测包入口 evals/__init__.py"
    assert (EVALS_ROOT / "__main__.py").is_file(), "缺少顶层评测包命令行入口 evals/__main__.py"

    graph: dict[str, set[str]] = {}
    for py_file in sorted(PKG_ROOT.rglob("*.py")):
        if "__pycache__" in py_file.parts:
            continue
        mod = _module_name(py_file)
        deps = _internal_imports(py_file)
        graph[mod] = deps

        # 产品包任何层级都绝对不能反向依赖顶层 evals/ 或残留 personal_assistant.eval
        for dep in deps:
            assert dep != "evals" and not dep.startswith("evals."), (
                f"产品包反向依赖顶层 evals/：{mod} -> {dep}"
            )
            assert dep != "personal_assistant.eval" and not dep.startswith("personal_assistant.eval."), (
                f"产品包残留旧版 personal_assistant.eval 引用：{mod} -> {dep}"
            )

        # Layer 1 (ai) 绝对不能依赖 agent / assistant / evals，且内部严守 Pi ch04 三层解耦边界
        if mod == "personal_assistant.ai" or mod.startswith("personal_assistant.ai."):
            for dep in deps:
                assert dep == "personal_assistant.ai" or dep.startswith("personal_assistant.ai."), (
                    f"Layer 1 越权向上依赖：{mod} -> {dep}"
                )
            if mod == "personal_assistant.ai.protocol":
                assert not deps, f"ai/protocol.py 必须零内部依赖，实际：{deps}"
            elif mod == "personal_assistant.ai.registry":
                assert deps <= {"personal_assistant.ai.protocol"}, (
                    f"ai/registry.py 仅允许依赖 ai.protocol，严禁导入具体翻译器实现，实际：{deps}"
                )
            elif mod == "personal_assistant.ai.cache":
                assert deps <= {"personal_assistant.ai.protocol"}, (
                    f"ai/cache.py 仅允许依赖 ai.protocol，实际：{deps}"
                )
            elif mod == "personal_assistant.ai.embeddings":
                assert deps <= {
                    "personal_assistant.ai.protocol",
                    "personal_assistant.ai.providers.http",
                }, f"ai/embeddings.py 仅允许依赖 ai.protocol 与 providers.http，实际：{deps}"
            elif mod == "personal_assistant.ai.rerank":
                assert deps <= {
                    "personal_assistant.ai.protocol",
                    "personal_assistant.ai.providers.http",
                }, f"ai/rerank.py 仅允许依赖 ai.protocol 与 providers.http，实际：{deps}"
            elif mod == "personal_assistant.ai.providers.http":
                assert deps <= {"personal_assistant.ai.protocol"}, (
                    f"ai/providers/http.py 仅允许依赖 ai.protocol，实际：{deps}"
                )
            elif mod == "personal_assistant.ai.providers.openai_reasoning":
                assert deps <= {
                    "personal_assistant.ai.protocol",
                    "personal_assistant.ai.providers.http",
                    "personal_assistant.ai.providers.openai_completions",
                }, (
                    f"推理方言翻译器 {mod} 仅允许依赖 protocol / providers.http / "
                    f"providers.openai_completions，实际：{deps}"
                )
            elif mod.startswith("personal_assistant.ai.providers."):
                assert deps <= {
                    "personal_assistant.ai.protocol",
                    "personal_assistant.ai.providers.http",
                }, f"具体翻译器 {mod} 仅允许依赖 protocol 与 providers.http，实际：{deps}"

        # Layer 2 (agent) 只能依赖 ai 或 agent，绝不能依赖 assistant / evals
        elif mod == "personal_assistant.agent" or mod.startswith("personal_assistant.agent."):
            for dep in deps:
                allowed = (
                    dep == "personal_assistant.ai"
                    or dep.startswith("personal_assistant.ai.")
                    or dep == "personal_assistant.agent"
                    or dep.startswith("personal_assistant.agent.")
                )
                assert allowed, f"Layer 2 越权依赖业务层：{mod} -> {dep}"

        # Layer 3 约束：底座模块零内部依赖；memory 不横向依赖 commands；非入口模块绝不反向依赖 cli
        elif mod in {
            "personal_assistant.assistant.common",
            "personal_assistant.assistant.db",
            "personal_assistant.assistant.files_policy",
            "personal_assistant.assistant.text_search",
        }:
            assert not deps, f"底座模块 {mod} 不应有任何内部依赖，实际：{deps}"
        elif mod == "personal_assistant.assistant.memory":
            for dep in deps:
                assert "commands" not in dep and "cli" not in dep and ".app" not in dep, (
                    f"memory.py 横向/反向依赖违规：{mod} -> {dep}"
                )
        elif mod == "personal_assistant.assistant.context":
            for dep in deps:
                assert not any(
                    bad in dep for bad in (".app", ".db", ".consolidator", ".commands", ".cli")
                ), f"context.py 横向/反向依赖违规：{mod} -> {dep}"
        elif mod.startswith("personal_assistant.assistant.") and mod != "personal_assistant.assistant.cli":
            for dep in deps:
                assert dep != "personal_assistant.assistant.cli", (
                    f"业务层反向依赖入口 cli.py：{mod} -> {dep}"
                )

    # 顶层 evals/ 绝不允许反向依赖 assistant/cli.py 或残留旧版 personal_assistant.eval
    eval_py_files = _iter_evals_python_files()
    assert eval_py_files, "顶层 evals/ 缺少评测引擎 Python 模块"
    for py_file in eval_py_files:
        mod = _module_name(py_file)
        deps = _internal_imports(py_file)
        graph[mod] = deps
        for dep in deps:
            assert dep != "personal_assistant.assistant.cli" and not dep.startswith("personal_assistant.assistant.cli."), (
                f"evals/ 反向依赖入口 assistant/cli.py：{mod} -> {dep}"
            )
            assert dep != "personal_assistant.eval" and not dep.startswith("personal_assistant.eval."), (
                f"evals/ 残留旧版 personal_assistant.eval 引用：{mod} -> {dep}"
            )

    # 全包零循环依赖检查（三色 DFS，覆盖 personal_assistant 与 evals）
    state: dict[str, int] = {}

    # 深度优先检测模块依赖图是否存在有向环
    def dfs(node: str, stack: list[str]) -> None:
        state[node] = 1
        for nxt in graph.get(node, ()):
            if nxt not in graph:
                continue
            if state.get(nxt) == 1:
                raise AssertionError(f"检测到循环依赖：{' -> '.join(stack + [nxt])}")
            if state.get(nxt, 0) == 0:
                dfs(nxt, stack + [nxt])
        state[node] = 2

    for node in sorted(graph):
        if state.get(node, 0) == 0:
            dfs(node, [node])


# 方法 2：类型递进扩展与零 OpenAI 方言渗透守卫——ToolSchema 纯中立，agent/assistant 零厂商序列化符号
def test_progressive_type_extension_tool_schema_to_tool() -> None:
    from personal_assistant.agent.tools import Tool, ToolRegistry
    from personal_assistant.ai.protocol import ToolCall, ToolSchema
    from personal_assistant.assistant.commands import CommandSpec

    schema = ToolSchema(
        name="ping",
        description="测试工具签名",
        parameters={"host": "目标主机"},
        param_enums={"host": ("localhost",)},
    )
    assert not hasattr(schema, "handler")
    assert not hasattr(schema, "to_openai_schema")
    assert schema.required_names == ("host",)

    tc = ToolCall(id="c1", name="ping", arguments='{"host": "localhost"}')
    assert not hasattr(tc, "to_openai_dict")
    assert not hasattr(tc, "from_raw")

    assert issubclass(Tool, ToolSchema)
    exec_tool = Tool(
        name="ping",
        description="测试工具签名",
        parameters={"host": "目标主机"},
        handler=lambda host: f"pong:{host}",
    )
    assert isinstance(exec_tool, ToolSchema)
    assert not hasattr(ToolRegistry, "definitions")

    spec = CommandSpec(
        name="hello",
        description="输出问候",
        argv=("echo", "hi"),
        params={},
        cwd=REPO_ROOT,
    )
    cmd_tool = spec.to_tool()
    assert isinstance(cmd_tool, Tool)
    assert isinstance(cmd_tool, ToolSchema)

    # 硬守卫：agent/ 与 assistant/ 彻底零残留旧 OpenAI 方言方法名
    forbidden_openai_syms = ("to_openai_schema", "to_openai_dict", "extract_tool_calls")
    for sub in ("agent", "assistant"):
        for py_file in sorted((PKG_ROOT / sub).rglob("*.py")):
            if "__pycache__" in py_file.parts:
                continue
            src = py_file.read_text(encoding="utf-8")
            for sym in forbidden_openai_syms:
                assert sym not in src, f"{py_file.relative_to(REPO_ROOT)} 严禁残留旧 OpenAI 符号 {sym!r}"


# 方法 3：子进程“可独立使用”测试——逐层剥离上层后，Layer 1 (ai) 与 Layer 1+2 (ai+agent) 均可独立运行，且顶层 evals 可独立调用
def test_layers_run_independently_in_clean_subprocesses() -> None:
    # 1. 单独使用 Layer 1 (ai)：只导 ai，零加载 agent/assistant/evals
    code_ai = """
import json, sys
from personal_assistant.ai import (
    ApiProviderRegistry,
    CacheMonitor,
    Context,
    ModelClient,
    ModelProfile,
    OpenAICompletionsProvider,
    ToolSchema,
    UserMessage,
    available_api_providers,
)

assert set(available_api_providers()) >= {
    "anthropic-messages",
    "google-generative-ai",
    "openai-completions",
    "openai-reasoning",
}
schema = ToolSchema(name="echo", description="回显", parameters={"text": "内容"})
def fake_stream(url, payload, headers, *, abort_signal=None):
    last_msg = payload["messages"][-1]["content"]
    yield 'data: ' + json.dumps({"choices": [{"delta": {"role": "assistant", "content": "ok:" + last_msg}}]}) + '\\n'
    yield 'data: ' + json.dumps({"choices": [], "usage": {"prompt_tokens": 100, "prompt_cache_hit_tokens": 60, "completion_tokens": 5}}) + '\\n'
    yield 'data: [DONE]\\n'

reg = ApiProviderRegistry()
reg.register(OpenAICompletionsProvider(post_stream=fake_stream))
profile = ModelProfile(name="m", api="openai-completions", base_url="http://x/v1", model="m", api_key="k")
monitor = CacheMonitor()
llm = monitor.wrap(ModelClient(profile=profile, registry=reg), session_key="s1")
r1 = llm.complete(Context(messages=(UserMessage("hi"),), tools=(schema,)))
r2 = llm.complete(Context(messages=(UserMessage("hi"), r1, UserMessage("next")), tools=(schema,)))
leaked = [m for m in sys.modules if m in {"evals", "plugins"} or m.startswith(("personal_assistant.agent", "personal_assistant.assistant", "personal_assistant.eval", "evals.", "plugins."))]
print(json.dumps({"reply": r2.text, "cached": monitor.summary()["cached_tokens"], "leaked": leaked}))
"""
    res_ai = subprocess.run(
        [sys.executable, "-c", code_ai],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    out_ai = json.loads(res_ai.stdout.strip())
    assert out_ai["reply"] == "ok:next"
    assert out_ai["cached"] == 120
    assert out_ai["leaked"] == []

    # 2. 单独使用 Layer 1 + 2 (ai + agent)：跑完全无关业务的计算器 Agent，零加载 assistant/evals
    code_agent = """
import json, sys
from personal_assistant.ai import AssistantMessage, Context, StreamDone, ToolCall, UserMessage
from personal_assistant.agent import ToolRegistry, run_react, tool

@tool(description="两数相乘", parameters={"a": "左操作数", "b": "右操作数"})
def multiply(a: str, b: str) -> str:
    return str(int(a) * int(b))

class FakeLLM:
    last_usage = None
    def __init__(self):
        self.turns = [
            AssistantMessage(blocks=(ToolCall(id="c1", name="multiply", arguments='{"a": "6", "b": "7"}'),), stop_reason="toolUse"),
            AssistantMessage.from_text("6 * 7 = 42", stop_reason="stop"),
        ]
    def stream(self, context, options=None):
        msg = self.turns.pop(0)
        yield StreamDone(reason="toolUse" if msg.tool_calls else "stop", message=msg)
    def complete(self, context, options=None):
        return self.turns.pop(0)

registry = ToolRegistry(multiply.__agent_tool__)
ctx = Context(system_prompt="You are a calculator.", messages=(UserMessage("6*7?"),))
reply = run_react(FakeLLM(), ctx, registry)
leaked = [m for m in sys.modules if m == "evals" or m.startswith(("personal_assistant.assistant", "personal_assistant.eval", "evals."))]
print(json.dumps({"reply": reply, "leaked": leaked}))
"""
    res_agent = subprocess.run(
        [sys.executable, "-c", code_agent],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    out_agent = json.loads(res_agent.stdout.strip())
    assert out_agent["reply"] == "6 * 7 = 42"
    assert out_agent["leaked"] == []

    # 3. 顶层 evals 包可独立以 `python -m evals` 运行，且绝不加载 personal_assistant.assistant.cli
    code_evals = """
import json, sys
import evals

leaked = [m for m in sys.modules if m == "personal_assistant.assistant.cli" or m.startswith("personal_assistant.eval")]
print(json.dumps({"has_main": callable(evals.main), "leaked": leaked}))
"""
    res_evals = subprocess.run(
        [sys.executable, "-c", code_evals],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    out_evals = json.loads(res_evals.stdout.strip())
    assert out_evals["has_main"] is True
    assert out_evals["leaked"] == []

    res_cli = subprocess.run(
        [sys.executable, "-m", "evals", "--mode", "search", "--retriever", "rag", "--max-cases", "1"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    assert "Eval mode=search  retriever=rag" in res_cli.stdout


# 测试 Slot-4 ApiProvider 翻译器插件加载、局部隔离注册表与全局注册表分发
def test_api_provider_registry_and_plugin_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from personal_assistant.agent.plugins import PluginManager
    from personal_assistant.ai import (
        ApiProviderRegistry,
        Context,
        ModelInvoker,
        UserMessage,
        from_env,
        register_api_provider,
    )

    plugin_file = tmp_path / "custom_provider.py"
    plugin_file.write_text(
        "from personal_assistant.agent.plugins import Plugin\n"
        "from personal_assistant.ai import AssistantMessage, StreamDone, StreamStart, TextDelta, TextEnd, TextStart\n"
        "# 测试用假协议翻译器实现（ApiProvider）\n"
        "class DummyApiProvider:\n"
        "    api = 'dummy-test-api'\n"
        "    # 流式产出带模型名的固定回复\n"
        "    def stream(self, model, context, options=None):\n"
        "        text = f'dummy:{model.model}'\n"
        "        msg = AssistantMessage.from_text(text, stop_reason='stop')\n"
        "        yield StreamStart(partial=AssistantMessage())\n"
        "        yield TextStart(content_index=0, partial=AssistantMessage())\n"
        "        yield TextDelta(content_index=0, text=text, partial=msg)\n"
        "        yield TextEnd(content_index=0, text=text, partial=msg)\n"
        "        yield StreamDone(reason='stop', message=msg)\n"
        "# 测试用主板插件子类\n"
        "class DummyProviderPlugin(Plugin):\n"
        "    plugin_id = 'dummy-test-api'\n"
        "    # 注册测试翻译器\n"
        "    def initialize(self):\n"
        "        self.context.register_provider(DummyApiProvider())\n",
        encoding="utf-8",
    )

    providers = PluginManager.from_entries([plugin_file.name], tmp_path).providers
    assert "dummy-test-api" in providers

    monkeypatch.setenv("LLM_MODE", "on")
    monkeypatch.setenv("LLM_API", "dummy-test-api")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "claude-test")

    # 1. 不污染全局默认注册表时可直接传入局部隔离 registry 构建
    local_reg = ApiProviderRegistry(providers)
    local_client = from_env(registry=local_reg)
    assert isinstance(local_client, ModelInvoker)
    assert local_client.complete(Context(messages=(UserMessage("hi"),))).text == "dummy:claude-test"

    # 2. 显式调用 register_api_provider 后全局亦可分发
    register_api_provider(providers["dummy-test-api"])
    client = from_env()
    assert isinstance(client, ModelInvoker)
    assert client.complete(Context(messages=(UserMessage("hi"),))).text == "dummy:claude-test"


# 验证 AST 依赖提取器能准确捕获产品包反向导入 evals/ 以及 evals/ 反向导入 assistant/cli.py 的各类写法
def test_architecture_import_extractor_catches_evals_and_cli_backedges(tmp_path: Path) -> None:
    bad_prod = tmp_path / "bad_prod.py"
    bad_prod.write_text(
        "import evals\n"
        "from evals import cli\n"
        "from evals.common import load_cases\n",
        encoding="utf-8",
    )
    prod_deps = _internal_imports(bad_prod)
    assert "evals" in prod_deps
    assert "evals.cli" in prod_deps
    assert "evals.common" in prod_deps

    bad_eval = tmp_path / "bad_eval.py"
    bad_eval.write_text(
        "from personal_assistant.assistant import cli\n"
        "import personal_assistant.assistant.cli as app_cli\n",
        encoding="utf-8",
    )
    eval_deps = _internal_imports(bad_eval)
    assert "personal_assistant.assistant.cli" in eval_deps


# 方法 4：微内核纯净守卫——Layer 1/2 不含业务层路径/配置字面量，且彻底消除 object.__setattr__ 与动态 type() 造类
def test_lower_layers_contain_zero_business_literals_or_monkeypatching() -> None:
    forbidden_literals = {
        "data/memory",
        "data/workspace",
        "MEMORY.md",
        "facts.md",
        "shell_config",
        "web_config",
        "retrieval_config",
        "tool_outputs",
        "data_root",
        "consolidator",
    }
    lower_files = sorted(
        p
        for p in list((PKG_ROOT / "ai").rglob("*.py")) + list((PKG_ROOT / "agent").rglob("*.py"))
        if "__pycache__" not in p.parts
    )
    assert lower_files
    for py_file in lower_files:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for bad in forbidden_literals:
                    assert bad not in node.value, (
                        f"底层 {py_file.relative_to(REPO_ROOT)}:{node.lineno} 泄漏业务字面量 {bad!r}：{node.value!r}"
                    )
            if isinstance(node, ast.Attribute) and node.attr == "__setattr__":
                raise AssertionError(
                    f"底层 {py_file.relative_to(REPO_ROOT)}:{node.lineno} 禁止访问 __setattr__"
                )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "type"
                and (len(node.args) != 1 or bool(node.keywords))
            ):
                raise AssertionError(
                    f"底层 {py_file.relative_to(REPO_ROOT)}:{node.lineno} 禁止动态 type() 造类"
                )


# 方法 5：全仓四棵受管目录树零 type: ignore，无冗余包壳，PluginContext 零业务字段泄漏，且 ai 不导出内部 helper
def test_zero_type_ignore_in_package_and_purified_plugin_context() -> None:
    from dataclasses import fields
    from personal_assistant import ai as ai_pkg
    from personal_assistant.agent.plugins import PluginContext

    assert not (REPO_ROOT / "plugins" / "tools" / "__init__.py").exists(), (
        "plugins/tools/__init__.py 冗余包壳已删除，不应重新引入"
    )
    assert not hasattr(ai_pkg, "common_prefix_len"), "ai/__init__.py 不应再导出内部 common_prefix_len"

    guarded_trees: dict[str, list[Path]] = {
        "personal_assistant": sorted(p for p in PKG_ROOT.rglob("*.py") if "__pycache__" not in p.parts),
        "plugins": sorted(p for p in (REPO_ROOT / "plugins").rglob("*.py") if "__pycache__" not in p.parts),
        "scripts": sorted(p for p in (REPO_ROOT / "scripts").rglob("*.py") if "__pycache__" not in p.parts),
        "evals": _iter_evals_python_files(),
    }
    for tree_name, py_files in guarded_trees.items():
        assert py_files, f"受守卫目录 {tree_name}/ 未找到任何 Python 文件"
        for py_file in py_files:
            for lineno, line in enumerate(py_file.read_text(encoding="utf-8").splitlines(), start=1):
                assert "type: ignore" not in line, (
                    f"{py_file.relative_to(REPO_ROOT)}:{lineno} 禁止出现 # type: ignore：{line.strip()!r}"
                )

    plugins_py = PKG_ROOT / "agent" / "plugins.py"
    plugins_src = plugins_py.read_text(encoding="utf-8")
    for forbidden_field in (
        "data_root",
        "offload_dir",
        "config_path",
        "raw_config",
        "extra_bindings",
        "active_plugin_id",
        "tool_registry",
        "consolidator",
    ):
        assert forbidden_field not in plugins_src, (
            f"agent/plugins.py 严禁泄漏业务字段或角色名 {forbidden_field!r}"
        )

    ctx_fields = {f.name for f in fields(PluginContext)}
    assert ctx_fields == {
        "project_root",
        "config",
        "bindings",
        "tool_specs",
        "retrievers",
        "compilers",
        "providers",
        "observers",
    }

    # 验证 assistant/__init__.py 按需延迟导出：导入 text_search 时绝不急切拖入 app / context / db / consolidator
    lazy_code = (
        "import json, sys\n"
        "import personal_assistant.assistant.text_search\n"
        "eager = [m for m in sys.modules if m in {"
        "'personal_assistant.assistant.app', "
        "'personal_assistant.assistant.context', "
        "'personal_assistant.assistant.db', "
        "'personal_assistant.assistant.consolidator'}]\n"
        "print(json.dumps(eager))\n"
    )
    res_lazy = subprocess.run(
        [sys.executable, "-c", lazy_code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    assert json.loads(res_lazy.stdout.strip()) == []
