"""Minimal terminal REPL entrypoint for the ReAct agent."""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from personal_assistant.commands import ToolsConfig
from personal_assistant.llm import LLMClient, LLMNotConfigured, from_env
from personal_assistant.memory import FilesPolicy, MemoryStore, memory_tools
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


def build_repl_tools(
    config_path: Path | None = None,
    data_root: Path | None = None,
    get_user_message=None,
) -> tuple[ToolRegistry, MemoryStore]:
    cfg_path = config_path or Path(os.getenv("MY_HELPER_CONFIG", "config")) / "tools.json"
    data_dir = (data_root or Path(os.getenv("MY_HELPER_DATA", "data"))).resolve()
    project_root = cfg_path.resolve().parent.parent if cfg_path.is_file() else data_dir.parent
    demo = list(build_demo_tools())

    if cfg_path.is_file():
        cfg = ToolsConfig.load(cfg_path, project_root=project_root)
        policy = FilesPolicy.from_dict(cfg.files, project_root=project_root, data_root=data_dir)
        memory = MemoryStore(data_dir, policy=policy)
        registry = cfg.build_registry(
            *memory_tools(memory, get_user_message=get_user_message),
            extra_tools=demo,
        )
        return registry, memory

    memory = MemoryStore(data_dir)
    return ToolRegistry(*memory_tools(memory, get_user_message=get_user_message), *demo), memory


def format_system_prompt(memory: MemoryStore) -> str:
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
    parts = [SYSTEM_PROMPT, f"今天是 {today}（北京时间）。"]
    rules = memory.read_rules_summary()
    if rules:
        parts.append(f"【行为规则 (data/memory/rules/)】\n{rules}")
    core = memory.read_core_memory()
    if core:
        parts.append(f"【长期记忆 (data/memory/MEMORY.md)】\n{core}")
    return "\n\n".join(parts)


def _print_step(name: str, args: str, observation: str) -> None:
    print(f"  [Act] {name}({args})")
    print(f"  [Observe] {observation}")


def run_repl(llm: LLMClient | None = None, tools: ToolRegistry | None = None) -> None:
    client = llm or from_env()
    current_user: list[tuple[str, int | None]] = [("", None)]
    msg_id = 0

    if tools is None:
        registry, memory = build_repl_tools(get_user_message=lambda: current_user[0])
    else:
        registry, memory = tools, None

    messages: list[dict[str, object]] = [
        {"role": "system", "content": format_system_prompt(memory) if memory else SYSTEM_PROMPT}
    ]
    print(f"My Helper v2 ReAct REPL（可用工具：{', '.join(registry.names)}；输入 exit/quit 退出）")

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

        msg_id += 1
        current_user[0] = (user_text, msg_id)
        if memory is not None:
            messages[0]["content"] = format_system_prompt(memory)
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
