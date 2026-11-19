"""最小终端交互（REPL）入口，基于 AssistantApp 提供持久化对话与快捷斜杠命令。"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from personal_assistant.app import AssistantApp, build_app
from personal_assistant.commands import ToolsConfig
from personal_assistant.llm import LLMClient, LLMNotConfigured, from_env
from personal_assistant.memory import FilesPolicy, MemoryStore, memory_tools
from personal_assistant.plugins import load_plugin_tools
from personal_assistant.react import SYSTEM_PROMPT, run_react
from personal_assistant.tools import ToolRegistry

REPO_ROOT = Path(__file__).resolve().parent.parent


# 读取本地 .env 文件并将未设置的 KEY=VALUE 注入当前进程环境变量
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


# 从可编辑插件文件加载演示用时间工具（不硬编码工具实现）
def build_demo_tools() -> ToolRegistry:
    plugin_tools = load_plugin_tools(["plugins/custom_tools.py"], REPO_ROOT)
    return ToolRegistry(*plugin_tools, allowed_names=["get_current_time"])


# 根据 config/tools.json 装配白名单工具注册表与 Markdown 记忆库
def build_repl_tools(
    config_path: Path | None = None,
    data_root: Path | None = None,
    get_user_message=None,
) -> tuple[ToolRegistry, MemoryStore]:
    cfg_path = config_path or Path(os.getenv("MY_HELPER_CONFIG", "config")) / "tools.json"
    data_dir = (data_root or Path(os.getenv("MY_HELPER_DATA", "data"))).resolve()
    project_root = cfg_path.resolve().parent.parent if cfg_path.is_file() else data_dir.parent

    if cfg_path.is_file():
        cfg = ToolsConfig.load(cfg_path, project_root=project_root)
        policy = FilesPolicy.from_dict(cfg.files, project_root=project_root, data_root=data_dir)
        memory = MemoryStore(data_dir, policy=policy)
        registry = cfg.build_registry(*memory_tools(memory, get_user_message=get_user_message))
        return registry, memory

    memory = MemoryStore(data_dir)
    return ToolRegistry(*memory_tools(memory, get_user_message=get_user_message)), memory


# 组装系统提示词：静态指令与规则在前、目录索引与日期在后（供离线评测与独立调用复用）
def format_system_prompt(memory: MemoryStore) -> str:
    parts = [SYSTEM_PROMPT]
    rules = memory.read_rules_summary()
    if rules:
        parts.append(f"【行为规则 (data/memory/rules/)】\n{rules}")
    index = memory.read_memory_index()
    if index:
        parts.append(f"【记忆索引 (data/memory/MEMORY.md)】\n{index}")
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
    parts.append(f"今天是 {today}（北京时间）。")
    return "\n\n".join(parts)


# 终端回调函数：实时打印每一步工具调用（Act）与观察结果（Observe）
def _print_step(name: str, args: str, observation: str) -> None:
    print(f"  [Act] {name}({args})")
    print(f"  [Observe] {observation}")


# 处理终端本地快捷命令（/status、/memory、/search、/refresh），不消耗模型 Token
def _handle_repl_command(app: AssistantApp, text: str, chat_id: str = "cli") -> str | None:
    if text == "/status":
        return json.dumps(app.status(), ensure_ascii=False, indent=2)
    if text == "/memory":
        return app.memory_overview()
    if text == "/refresh":
        app.reset_session_prompt(chat_id)
        return "已清空当前会话冻结的系统提示词，下一轮将重新加载最新规则与 MEMORY.md 索引。"
    if text.startswith("/search "):
        query = text.removeprefix("/search ").strip()
        return app.search(query) if query else "用法：/search <关键词>"
    return None


# 启动交互式终端对话循环：默认基于 AssistantApp 持久化历史与加载白名单工具
def run_repl(
    app: AssistantApp | None = None,
    *,
    llm: LLMClient | None = None,
    tools: ToolRegistry | None = None,
    chat_id: str = "cli",
) -> None:
    if tools is not None:
        client = llm or from_env()
        messages: list[dict[str, object]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        print(f"My Helper v2 ReAct REPL（可用工具：{', '.join(tools.names)}；输入 exit/quit 退出）")
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
                reply = run_react(client, messages, tools, on_step=_print_step)
            except LLMNotConfigured as error:
                messages.pop()
                print(f"助手> {error}")
                continue
            print(f"助手> {reply}")
        return

    owned = app is None
    active_app = app or build_app(llm=llm)
    try:
        tool_names = ", ".join(active_app.build_tools().names)
        print(f"My Helper REPL（可用工具：{tool_names}；命令：/status /memory /search /refresh；exit 退出）")
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
            cmd_reply = _handle_repl_command(active_app, user_text, chat_id=chat_id)
            if cmd_reply is not None:
                print(f"助手>\n{cmd_reply}")
                continue
            try:
                result = active_app.chat(user_text, chat_id=chat_id, on_step=_print_step)
            except LLMNotConfigured as error:
                print(f"助手> {error}")
                continue
            print(f"助手> {result['reply']}")
    finally:
        if owned:
            active_app.close()


# 命令行总入口：加载 .env 并解析子命令（默认进入 chat 交互模式）
def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="my-helper")
    parser.add_argument("--data", default=None, help="数据目录路径（默认读取 MY_HELPER_DATA 或 data）")
    parser.add_argument("--config", default=None, help="配置目录路径（默认读取 MY_HELPER_CONFIG 或 config）")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("chat", help="在终端运行最小 ReAct 对话循环")
    args = parser.parse_args(argv)
    if args.command in (None, "chat"):
        app = build_app(
            data_root=Path(args.data) if args.data else None,
            config_root=Path(args.config) if args.config else None,
        )
        run_repl(app)
