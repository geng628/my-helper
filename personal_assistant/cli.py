"""Minimal terminal REPL entrypoint for testing the ReAct loop."""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from personal_assistant.llm import LLMClient, LLMNotConfigured, from_env
from personal_assistant.react import SYSTEM_PROMPT, run_react
from personal_assistant.tools import Tool, ToolRegistry


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def build_demo_tools() -> ToolRegistry:
    def get_current_time() -> str:
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        return now.strftime("%Y-%m-%d %H:%M:%S (Asia/Shanghai)")

    return ToolRegistry(
        Tool(
            name="get_current_time",
            description="获取当前北京时间（YYYY-MM-DD HH:MM:SS）。",
            parameters={},
            handler=get_current_time,
        )
    )


def _print_step(name: str, args: str, observation: str) -> None:
    print(f"  [Act] {name}({args})")
    print(f"  [Observe] {observation}")


def run_repl(llm: LLMClient | None = None, tools: ToolRegistry | None = None) -> None:
    client = llm or from_env()
    registry = tools or build_demo_tools()
    messages: list[dict[str, object]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("My Helper v2 ReAct REPL（输入 exit/quit 退出）")

    while True:
        try:
            user_text = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit", "/quit"}:
            break

        messages.append({"role": "user", "content": user_text})
        try:
            reply = run_react(client, messages, registry, on_step=_print_step)
        except LLMNotConfigured as error:
            messages.pop()
            print(f"助手> {error}")
            continue
        print(f"助手> {reply}")


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="my-helper")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("chat", help="在终端运行最小 ReAct 对话循环")
    args = parser.parse_args(argv)
    if args.command in (None, "chat"):
        run_repl()
