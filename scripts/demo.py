"""Convenient script to run and inspect the minimal ReAct loop (offline demo or live LLM)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from personal_assistant.cli import build_demo_tools, load_dotenv
from personal_assistant.llm import LLMClient, LLMNotConfigured, from_env
from personal_assistant.react import SYSTEM_PROMPT, run_react
from personal_assistant.tools import Tool, ToolRegistry


class DemoScriptedLLM(LLMClient):
    """Offline mock LLM so you can watch the full ReAct loop without an API key."""

    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = list(responses)

    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        if not self._responses:
            raise RuntimeError("DemoScriptedLLM ran out of responses")
        return self._responses.pop(0)


def _print_step(name: str, args: str, observation: str) -> None:
    print(f"  [Act]     {name}({args})")
    print(f"  [Observe] {observation}")


def run_offline_demo(show_messages: bool = False) -> None:
    tools = ToolRegistry(
        *build_demo_tools(),
        Tool(
            name="add_numbers",
            description="计算两个整数之和。",
            parameters={"a": "第一个整数", "b": "第二个整数"},
            handler=lambda a, b: str(int(a) + int(b)),
        ),
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
    reply1 = run_react(llm1, msgs1, tools, on_step=_print_step)
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
    reply2 = run_react(llm2, msgs2, tools, on_step=_print_step)
    print(f"助手> {reply2}")
    if show_messages:
        print("\n--- 完整 messages 轨迹 ---")
        print(json.dumps(msgs2, ensure_ascii=False, indent=2))


def run_live_ask(question: str, show_messages: bool = False) -> int:
    load_dotenv(ROOT / ".env")
    client = from_env()
    tools = build_demo_tools()
    messages: list[dict[str, object]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    print(f"你> {question}")
    try:
        reply = run_react(client, messages, tools, on_step=_print_step)
    except LLMNotConfigured as error:
        print(f"助手> {error}")
        print("提示：不配置 API Key 也可直接运行 `uv run python scripts/demo.py` 查看离线演示。")
        return 1
    print(f"助手> {reply}")
    if show_messages:
        print("\n--- 完整 messages 轨迹 ---")
        print(json.dumps(messages, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="测试与演示最小 ReAct 循环")
    parser.add_argument("--ask", help="使用 .env 中配置的真实模型测试单句提问")
    parser.add_argument(
        "--show-messages",
        action="store_true",
        help="打印完整 OpenAI messages 对话上下文（含 tool_calls 与 tool 观察消息）",
    )
    args = parser.parse_args()
    if args.ask:
        return run_live_ask(args.ask, show_messages=args.show_messages)
    run_offline_demo(show_messages=args.show_messages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
