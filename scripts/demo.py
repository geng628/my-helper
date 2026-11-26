"""便捷测试与演示脚本：支持离线零 Token 演示与真实模型单句实测。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from personal_assistant.agent.plugins import PluginManager
from personal_assistant.agent.react import AgentEvent, ToolStep, run_react
from personal_assistant.agent.tools import ToolRegistry
from personal_assistant.ai.client import LLMClient, LLMNotConfigured
from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.assistant.common import load_dotenv
from personal_assistant.assistant.prompts import SYSTEM_PROMPT


# 从插件目录加载单句提问演示所需的时间工具注册表
def build_demo_tools() -> ToolRegistry:
    plugin_tools = PluginManager.from_entries(["plugins/tools"], ROOT).build_tools()
    return ToolRegistry(*plugin_tools, allowed_names=["get_current_time"])


# 离线脚本化模拟模型：按预设顺序返回回复，用于零成本演示 ReAct 循环
class DemoScriptedLLM(LLMClient):
    # 保存预设回复队列
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = list(responses)

    # 弹出并返回下一条预设回复
    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        if not self._responses:
            raise RuntimeError("DemoScriptedLLM ran out of responses")
        return self._responses.pop(0)


# 打印每一步工具调用事件（Act 与耗时）与工具返回观察（Observe）
def _print_event(event: AgentEvent) -> None:
    if isinstance(event, ToolStep):
        print(f"  [Act]     {event.tool}({event.arguments}) ({event.duration_ms:.1f}ms)")
        print(f"  [Observe] {event.observation}")


# 运行两个离线演示场景：多轮链式工具调用 + 工具参数出错自愈
def run_offline_demo(show_messages: bool = False) -> None:
    tools = ToolRegistry(
        *PluginManager.from_entries(["plugins/tools"], ROOT).build_tools(),
        allowed_names=["get_current_time", "add_numbers"],
    )

    print("=== 场景 1：多轮 ReAct（先查时间，再做计算）===")
    llm1 = DemoScriptedLLM(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_current_time", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "add_numbers", "arguments": '{"a": "19", "b": "23"}'},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "现在是北京时间（见上方工具观察时间），另外 19 + 23 = 42。",
            },
        ]
    )
    msgs1: list[dict[str, object]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "现在几点了？顺便帮我算一下 19 加 23。"},
    ]
    print(f"你> {msgs1[-1]['content']}")
    reply1 = run_react(llm1, msgs1, tools, on_event=_print_event)
    print(f"助手> {reply1}")
    if show_messages:
        print("\n--- 完整 messages 轨迹 ---")
        print(json.dumps(msgs1, ensure_ascii=False, indent=2))

    print("\n=== 场景 2：工具参数出错后自愈（Error as Observation）===")
    llm2 = DemoScriptedLLM(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "bad_1",
                        "type": "function",
                        "function": {"name": "add_numbers", "arguments": '{"x": "10"}'},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "fix_1",
                        "type": "function",
                        "function": {"name": "add_numbers", "arguments": '{"a": "10", "b": "32"}'},
                    }
                ],
            },
            {"role": "assistant", "content": "已修正参数并计算完成：10 + 32 = 42。"},
        ]
    )
    msgs2: list[dict[str, object]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "帮我算 10 加 32。"},
    ]
    print(f"你> {msgs2[-1]['content']}")
    reply2 = run_react(llm2, msgs2, tools, on_event=_print_event)
    print(f"助手> {reply2}")
    if show_messages:
        print("\n--- 完整 messages 轨迹 ---")
        print(json.dumps(msgs2, ensure_ascii=False, indent=2))


# 统一通过 ToolsConfig 加载模型（支持 models.dsl / models.json / Provider 插件与 .env）并向真实模型发起单句提问
def run_live_ask(
    question: str,
    show_messages: bool = False,
    *,
    model_name: str | None = None,
) -> int:
    load_dotenv()
    cfg_path = ROOT / "config" / "tools.json"
    cfg = ToolsConfig.load(cfg_path, project_root=ROOT) if cfg_path.is_file() else ToolsConfig.empty(ROOT)
    try:
        client = cfg.build_llm(cfg_path.parent if cfg_path.is_file() else None, model_name=model_name)
        tools = (
            ToolRegistry(*cfg.plugin_manager.build_tools(), allowed_names=["get_current_time"])
            if cfg_path.is_file()
            else build_demo_tools()
        )
        messages: list[dict[str, object]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        print(f"你> {question}")
        try:
            reply = run_react(client, messages, tools, on_event=_print_event)
        except LLMNotConfigured as error:
            print(f"助手> {error}")
            print("提示：不配置 API Key 也可直接运行 `uv run python scripts/demo.py` 查看离线演示。")
            return 1
        print(f"助手> {reply}")
        if show_messages:
            print("\n--- 完整 messages 轨迹 ---")
            print(json.dumps(messages, ensure_ascii=False, indent=2))
        return 0
    finally:
        cfg.plugin_manager.terminate_all()


# 演示脚本命令行入口：不传 --ask 跑离线演示，传 --ask 跑真实模型提问
def main() -> int:
    parser = argparse.ArgumentParser(description="测试与演示最小 ReAct 循环")
    parser.add_argument("--ask", help="使用 config/models.dsl 或 .env 中配置的真实模型测试单句提问")
    parser.add_argument("--model", default=None, help="指定 models.dsl 档案名、预设名或模型 ID")
    parser.add_argument(
        "--show-messages",
        action="store_true",
        help="打印完整 OpenAI messages 对话上下文（含 tool_calls 与 tool 观察消息）",
    )
    args = parser.parse_args()
    if args.ask:
        return run_live_ask(args.ask, show_messages=args.show_messages, model_name=args.model)
    run_offline_demo(show_messages=args.show_messages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
