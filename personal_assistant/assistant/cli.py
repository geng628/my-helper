"""Layer 3 终端交互（REPL）入口，基于 AssistantApp 提供持久化对话与快捷斜杠命令。"""
from __future__ import annotations

import argparse
import getpass
import json
import os
from collections.abc import Callable
from pathlib import Path

from personal_assistant.agent.plugins import load_plugin_tools
from personal_assistant.agent.react import run_react
from personal_assistant.agent.tools import ToolRegistry
from personal_assistant.ai.cache import format_cache_summary
from personal_assistant.ai.client import LLMClient, LLMNotConfigured, NoopLLMClient, from_env
from personal_assistant.ai.types import ModelsConfig
from personal_assistant.assistant.app import AssistantApp, build_app
from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.assistant.common import REPO_ROOT, load_dotenv
from personal_assistant.assistant.memory import FilesPolicy, MemoryStore, memory_tools
from personal_assistant.assistant.prompts import SYSTEM_PROMPT


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
        memory = MemoryStore(data_dir, policy=policy, retriever=cfg.build_retriever())
        registry = cfg.build_registry(
            *memory_tools(memory, get_user_message=get_user_message),
            data_root=data_dir,
        )
        return registry, memory

    memory = MemoryStore(data_dir)
    return ToolRegistry(*memory_tools(memory, get_user_message=get_user_message)), memory


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
            cache_info = result.get("cache")
            if isinstance(cache_info, dict) and cache_info.get("llm_calls"):
                print(f"  [Cache] {format_cache_summary(cache_info)}")
            if result.get("consolidation_scheduled"):
                print("  [Memory] 上下文已达阈值，后台正在自动沉淀长期记忆并缩减上下文…")
    finally:
        if owned:
            active_app.close()


# 零代码终端交互配置向导：只需选厂商编号并粘贴 API Key，自动生成 0600 权限的 models.dsl 与 models.json
def run_setup_wizard(
    config_dir: Path,
    *,
    project_root: Path = REPO_ROOT,
    input_fn: Callable[[str], str] = input,
    secret_fn: Callable[[str], str] = getpass.getpass,
    print_fn: Callable[[str], None] = print,
) -> ModelsConfig | None:
    cfg_dir = config_dir.resolve()
    tools_json = cfg_dir / "tools.json"
    tcfg = ToolsConfig.load(tools_json, project_root=project_root)
    presets = tcfg.model_presets()

    print_fn("=== My Helper 一键模型配置向导（零代码，10 秒配好）===")
    for idx, item in enumerate(presets, start=1):
        print_fn(f"  {idx}. {item['label']}（默认模型：{item['default_model']}）")
    custom_idx = len(presets) + 1
    print_fn(f"  {custom_idx}. 自定义 OpenAI 兼容接口（手动填写网址与模型名）")

    raw_choice = input_fn(f"请选择编号 [1-{custom_idx}]（直接回车默认选 1）：").strip() or "1"
    try:
        choice = int(raw_choice)
    except ValueError as error:
        raise ValueError(f"请输入 1 到 {custom_idx} 之间的数字。") from error
    if not (1 <= choice <= custom_idx):
        raise ValueError(f"请输入 1 到 {custom_idx} 之间的数字。")

    if choice == custom_idx:
        name = input_fn("给这个模型起个简短名字（如 my_model）：").strip() or "custom"
        base_url = input_fn("接口地址（例如 https://api.deepseek.com/v1）：").strip()
        model = input_fn("模型名称（例如 deepseek-chat）：").strip()
        api_key = secret_fn("API Key（输入不回显，自动本地加密权限保存）：").strip()
        saved = tcfg.save_model_setup(
            cfg_dir, preset="custom", name=name, base_url=base_url, model=model, api_key=api_key
        )
    else:
        picked = presets[choice - 1]
        api_key = ""
        if picked["needs_key"]:
            api_key = secret_fn(f"请粘贴 {picked['label']} 的 API Key（输入不回显）：").strip()
            if not api_key:
                raise ValueError("API Key 不能为空。")
        model = input_fn(f"模型名（直接回车使用默认 {picked['default_model']}）：").strip()
        saved = tcfg.save_model_setup(
            cfg_dir,
            preset=str(picked["id"]),
            api_key=api_key,
            model=model,
        )

    print_fn(f"\n配置已自动生成：{cfg_dir / 'models.dsl'} 与 {cfg_dir / 'models.json'}（权限 0600，已由 .gitignore 与沙箱自动保护）")
    print_fn(json.dumps(saved.redacted_dict(), ensure_ascii=False, indent=2))
    print_fn("现在可以直接运行 `uv run python -m personal_assistant chat` 开始对话！")
    return saved


# 命令行总入口：加载 .env 并解析子命令（支持 setup 一键向导、chat 对话与 compile-config 编译）
def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="my-helper")
    parser.add_argument("--data", default=None, help="数据目录路径（默认读取 MY_HELPER_DATA 或 data）")
    parser.add_argument("--config", default=None, help="配置目录路径（默认读取 MY_HELPER_CONFIG 或 config）")
    parser.add_argument("--model", default=None, help="指定 models.dsl / models.json 中定义的模型档案名称")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("chat", help="在终端运行最小 ReAct 对话循环")
    subparsers.add_parser("setup", help="10 秒零代码交互式向导：选厂商 + 贴密钥，自动生成模型配置")
    subparsers.add_parser("compile-config", help="将 config/models.dsl 编译为 0600 权限的标准 config/models.json")
    args = parser.parse_args(argv)
    cfg_dir = (Path(args.config) if args.config else Path(os.getenv("MY_HELPER_CONFIG", "config"))).resolve()
    if args.command == "setup":
        run_setup_wizard(cfg_dir, project_root=cfg_dir.parent if (cfg_dir / "tools.json").is_file() else REPO_ROOT)
        return
    if args.command == "compile-config":
        tools_json = cfg_dir / "tools.json"
        tcfg = ToolsConfig.load(tools_json, project_root=cfg_dir.parent)
        compiled = tcfg.load_models_config(cfg_dir, write_json=True)
        if compiled is None:
            print(f"未在 {cfg_dir} 下找到 models.dsl 或 models.json（运行 `uv run python -m personal_assistant setup` 可 10 秒自动生成）。")
            return
        print(f"已编译并写入：{cfg_dir / 'models.json'}（权限 0600，已由 .gitignore 保护）")
        print(json.dumps(compiled.redacted_dict(), ensure_ascii=False, indent=2))
        return
    if args.command in (None, "chat"):
        app = build_app(
            data_root=Path(args.data) if args.data else None,
            config_root=cfg_dir,
            model_name=args.model,
        )
        if isinstance(app.llm, NoopLLMClient):
            print("提示：当前尚未配置大模型。运行 `uv run python -m personal_assistant setup` 可 10 秒零代码一键配置。")
        run_repl(app)
