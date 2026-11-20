"""三层漏斗架构守卫测试（TDD）：AST 依赖方向校验、类型递进扩展校验与子进程断层独立运行测试。"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_ROOT = REPO_ROOT / "personal_assistant"


# 提取指定 Python 文件内部对 personal_assistant.* 的全部模块依赖
def _internal_imports(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    deps: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "personal_assistant" or node.module.startswith("personal_assistant."):
                deps.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "personal_assistant" or alias.name.startswith("personal_assistant."):
                    deps.add(alias.name)
    return deps


# 将包内文件路径转换为标准点分模块名（如 personal_assistant.ai.client）
def _module_name(py_file: Path) -> str:
    rel = py_file.relative_to(REPO_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


# 方法 1：AST 依赖漏斗守卫——严禁任何跨层反向依赖、横向串台与循环依赖
def test_dependency_funnel_has_no_upward_or_cyclic_imports() -> None:
    assert (PKG_ROOT / "ai").is_dir(), "缺少 Layer 1 包 personal_assistant/ai/"
    assert (PKG_ROOT / "agent").is_dir(), "缺少 Layer 2 包 personal_assistant/agent/"
    assert (PKG_ROOT / "assistant").is_dir(), "缺少 Layer 3 包 personal_assistant/assistant/"

    graph: dict[str, set[str]] = {}
    for py_file in sorted(PKG_ROOT.rglob("*.py")):
        mod = _module_name(py_file)
        deps = _internal_imports(py_file)
        graph[mod] = deps

        # Layer 1 (ai) 绝对不能依赖 agent / assistant / eval
        if mod == "personal_assistant.ai" or mod.startswith("personal_assistant.ai."):
            for dep in deps:
                assert dep == "personal_assistant.ai" or dep.startswith("personal_assistant.ai."), (
                    f"Layer 1 越权向上依赖：{mod} -> {dep}"
                )

        # Layer 2 (agent) 只能依赖 ai 或 agent，绝不能依赖 assistant / eval
        elif mod == "personal_assistant.agent" or mod.startswith("personal_assistant.agent."):
            for dep in deps:
                allowed = (
                    dep == "personal_assistant.ai"
                    or dep.startswith("personal_assistant.ai.")
                    or dep == "personal_assistant.agent"
                    or dep.startswith("personal_assistant.agent.")
                )
                assert allowed, f"Layer 2 越权依赖业务层：{mod} -> {dep}"

        # Layer 3 约束：底座模块零内部依赖；memory 不横向依赖 commands；非入口模块绝不反向依赖 cli/eval
        elif mod == "personal_assistant.assistant.common" or mod == "personal_assistant.assistant.db":
            assert not deps, f"底座模块 {mod} 不应有任何内部依赖，实际：{deps}"
        elif mod == "personal_assistant.assistant.memory":
            for dep in deps:
                assert "commands" not in dep and "cli" not in dep and ".app" not in dep, (
                    f"memory.py 横向/反向依赖违规：{mod} -> {dep}"
                )
        elif mod.startswith("personal_assistant.assistant.") and mod != "personal_assistant.assistant.cli":
            for dep in deps:
                assert dep != "personal_assistant.assistant.cli" and not dep.startswith("personal_assistant.eval"), (
                    f"业务层反向依赖入口/评测层：{mod} -> {dep}"
                )

        # Layer 4 (eval) 绝不允许反向 import 终端入口 cli.py
        elif mod == "personal_assistant.eval" or mod.startswith("personal_assistant.eval."):
            for dep in deps:
                assert dep != "personal_assistant.assistant.cli", (
                    f"eval 模块反向依赖入口 cli.py：{mod} -> {dep}"
                )

    # 全包零循环依赖检查（三色 DFS）
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


# 方法 2：类型递进扩展测试——底层 ToolSchema 不含执行器，中层 Tool 继承扩展，顶层 CommandSpec 叠加业务字段
def test_progressive_type_extension_tool_schema_to_tool() -> None:
    from personal_assistant.agent.tools import Tool
    from personal_assistant.ai.types import ToolSchema
    from personal_assistant.assistant.commands import CommandSpec

    schema = ToolSchema(
        name="ping",
        description="测试工具签名",
        parameters={"host": "目标主机"},
        param_enums={"host": ("localhost",)},
    )
    assert not hasattr(schema, "handler")
    exported = schema.to_openai_schema()
    assert exported["function"]["name"] == "ping"
    assert exported["function"]["parameters"]["properties"]["host"]["enum"] == ["localhost"]

    assert issubclass(Tool, ToolSchema)
    exec_tool = Tool(
        name="ping",
        description="测试工具签名",
        parameters={"host": "目标主机"},
        handler=lambda host: f"pong:{host}",
    )
    assert isinstance(exec_tool, ToolSchema)
    assert exec_tool.to_openai_schema()["function"]["name"] == "ping"

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


# 方法 3：子进程“可独立使用”测试——逐层剥离上层后，Layer 1 (ai) 与 Layer 1+2 (ai+agent) 均可独立运行
def test_layers_run_independently_in_clean_subprocesses() -> None:
    # 1. 单独使用 Layer 1 (ai)：只导 ai，零加载 agent/assistant/eval
    code_ai = """
import json, sys
from personal_assistant.ai import CacheMonitor, OpenAICompatibleLLM, ToolSchema

schema = ToolSchema(name="echo", description="回显", parameters={"text": "内容"})
def fake_post(url, payload, headers):
    return {
        "choices": [{"message": {"role": "assistant", "content": "ok:" + payload["messages"][-1]["content"]}}],
        "usage": {"prompt_tokens": 100, "prompt_cache_hit_tokens": 60, "completion_tokens": 5},
    }

monitor = CacheMonitor()
llm = monitor.wrap(OpenAICompatibleLLM(api_key="k", base_url="http://x/v1", model="m", post_json=fake_post), "s1")
r1 = llm.respond([{"role": "user", "content": "hi"}], tools=[schema.to_openai_schema()])
r2 = llm.respond([{"role": "user", "content": "hi"}, {"role": "assistant", "content": r1["content"]}, {"role": "user", "content": "next"}], tools=[schema.to_openai_schema()])
leaked = [m for m in sys.modules if m.startswith(("personal_assistant.agent", "personal_assistant.assistant", "personal_assistant.eval"))]
print(json.dumps({"reply": r2["content"], "cached": monitor.summary()["cached_tokens"], "leaked": leaked}))
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

    # 2. 单独使用 Layer 1 + 2 (ai + agent)：跑完全无关业务的计算器 Agent，零加载 assistant/eval
    code_agent = """
import json, sys
from personal_assistant.ai import LLMClient
from personal_assistant.agent import ToolRegistry, run_react, tool

@tool(description="两数相乘", parameters={"a": "左操作数", "b": "右操作数"})
def multiply(a: str, b: str) -> str:
    return str(int(a) * int(b))

class FakeLLM(LLMClient):
    def __init__(self):
        self.turns = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "multiply", "arguments": '{"a": "6", "b": "7"}'}}]},
            {"role": "assistant", "content": "6 * 7 = 42"},
        ]
    def respond(self, messages, tools=None):
        return self.turns.pop(0)

registry = ToolRegistry(multiply.__agent_tool__)
msgs = [{"role": "system", "content": "You are a calculator."}, {"role": "user", "content": "6*7?"}]
reply = run_react(FakeLLM(), msgs, registry)
leaked = [m for m in sys.modules if m.startswith(("personal_assistant.assistant", "personal_assistant.eval"))]
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


# 测试 Provider 插件化注册与按 LLM_PROVIDER 环境变量分发（换厂商不改 react.py）
def test_llm_provider_registry_and_plugin_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from personal_assistant.agent.plugins import load_plugin_providers
    from personal_assistant.ai.client import LLMClient, from_env, register_provider

    plugin_file = tmp_path / "custom_provider.py"
    plugin_file.write_text(
        "from personal_assistant.ai.client import LLMClient\n"
        "class DummyProviderLLM(LLMClient):\n"
        "    def __init__(self, model):\n"
        "        self.model = model\n"
        "    def respond(self, messages, tools=None):\n"
        "        return {'role': 'assistant', 'content': f'dummy:{self.model}'}\n"
        "def build_dummy(env):\n"
        "    return DummyProviderLLM(env.get('LLM_MODEL', 'demo'))\n"
        "LLM_PROVIDERS = {'dummy_test_provider': build_dummy}\n",
        encoding="utf-8",
    )

    providers = load_plugin_providers([plugin_file.name], tmp_path)
    assert "dummy_test_provider" in providers
    register_provider("dummy_test_provider", providers["dummy_test_provider"])

    monkeypatch.setenv("LLM_MODE", "on")
    monkeypatch.setenv("LLM_PROVIDER", "dummy_test_provider")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "claude-test")
    client = from_env()
    assert isinstance(client, LLMClient)
    assert client.respond([{"role": "user", "content": "hi"}])["content"] == "dummy:claude-test"
