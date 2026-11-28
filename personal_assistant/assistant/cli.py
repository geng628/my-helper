"""Layer 3 终端交互（REPL）入口，基于 AssistantApp 提供持久化对话与快捷斜杠命令。"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import signal
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from personal_assistant.agent.react import AgentEvent, ToolStart, ToolStep
from personal_assistant.ai import (
    AbortController,
    AbortSignal,
    LLMNotConfigured,
    ModelInvoker,
    ModelsConfig,
    NoopModelClient,
    StreamAbortedError,
    TextDelta,
    ThinkingDelta,
    format_cache_summary,
)
from personal_assistant.assistant.app import AssistantApp, build_app
from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.assistant.common import REPO_ROOT, load_dotenv


# 打印单步工具执行日志与耗时
def _print_tool_step(step: ToolStep) -> None:
    print(f"  [Act] {step.tool}({step.arguments}) ({step.duration_ms:.1f}ms)")
    print(f"  [Observe] {step.observation}")


# 创建终端流式打印器：返回实时事件回调与回合收尾函数
def _stream_printer() -> tuple[Callable[[AgentEvent], None], Callable[[str], None]]:
    mode = "idle"
    streamed_text = False

    # 实时打印单个 Agent 事件：按 idle / thinking / text 三态切换行前缀与换行
    def on_event(ev: AgentEvent) -> None:
        nonlocal mode, streamed_text
        if isinstance(ev, ThinkingDelta) and ev.text:
            if mode != "thinking":
                if mode == "text":
                    print()
                print("  [推导] ", end="", flush=True)
                mode = "thinking"
            print(ev.text, end="", flush=True)
        elif isinstance(ev, ToolStart):
            if mode != "idle":
                print()
                mode = "idle"
            streamed_text = False
            print(f"  [调用] {ev.tool}({ev.arguments}) ...", flush=True)
        elif isinstance(ev, ToolStep):
            if mode != "idle":
                print()
                mode = "idle"
            streamed_text = False
            _print_tool_step(ev)
        elif isinstance(ev, TextDelta) and ev.text:
            if mode != "text":
                if mode == "thinking":
                    print()
                print("助手> ", end="", flush=True)
                mode = "text"
                streamed_text = True
            print(ev.text, end="", flush=True)

    # 回合收尾：已流式输出过正文仅在未归零时补换行，否则补换行并整段打印最终回复
    def finish(reply: str) -> None:
        nonlocal mode
        if streamed_text:
            if mode != "idle":
                print()
        else:
            if mode != "idle":
                print()
            print(f"助手> {reply}")
        mode = "idle"

    return on_event, finish


# 处理 /rag 快捷子命令（status 查看父子索引状态；enrich 复用轻量模型离线补全父块上下文摘要并支持 Ctrl+C 断点续跑）
def _handle_rag_command(app: AssistantApp, text: str, chat_id: str) -> str:
    parts = text.split()
    if len(parts) > 3:
        return "用法：/rag status [目录] 或 /rag enrich [目录]"
    sub = parts[1].lower() if len(parts) >= 2 else "status"
    root = parts[2] if len(parts) >= 3 else "data/memory"
    retriever = app.memory.retriever

    if sub == "status":
        status_fn = getattr(retriever, "index_status", None)
        if not callable(status_fn):
            return f"当前检索引擎（{getattr(retriever, 'name', 'unknown')}）未启用持久化 RAG 索引。"
        try:
            return json.dumps(status_fn(app.memory, root=root), ensure_ascii=False, indent=2)
        except Exception as error:  # noqa: BLE001
            return f"查询 /rag status 失败：{error}"

    if sub == "enrich":
        enrich_fn = getattr(retriever, "enrich_contexts", None)
        if not callable(enrich_fn):
            return f"当前检索引擎（{getattr(retriever, 'name', 'unknown')}）不支持离线上下文摘要补全。"
        enrich_llm = app.consolidator.llm
        if isinstance(enrich_llm, NoopModelClient):
            return "未配置可用大模型，无法执行离线上下文摘要补全。请先运行 setup 配置模型。"
        raw_model = getattr(enrich_llm, "model", None) or getattr(app, "_consolidator_model", None) or "default"
        model_name = str(raw_model).strip() or "default"
        try:
            with _sigint_to_abort(app, chat_id) as abort_sig:
                stats = enrich_fn(
                    app.memory,
                    enrich_llm,
                    root=root,
                    model_name=model_name,
                    abort_signal=abort_sig,
                )
        except (KeyboardInterrupt, StreamAbortedError):
            return "已中止离线上下文补全；已完成片段的摘要已保存至缓存，下次执行将自动续跑。"
        except LLMNotConfigured as error:
            return f"离线上下文补全未就绪：{error}"
        except Exception as error:  # noqa: BLE001
            return f"执行 /rag enrich 失败：{error}"
        return (
            f"离线上下文补全完成（目录：{root}，模型：{model_name}）："
            f"扫描 {stats['files']} 篇文档、{stats['parents_total']} 个父块；"
            f"新生成 {stats['enriched']} 条，复用缓存 {stats['cached_hits']} 条，跳过 {stats['skipped']} 条。"
        )

    return "用法：/rag status [目录] 或 /rag enrich [目录]"


# 处理终端本地快捷命令（/status、/memory、/search、/rag、/refresh），不消耗模型 Token（/rag enrich 除外）
def _handle_repl_command(app: AssistantApp, text: str, chat_id: str = "cli") -> str | None:
    head = text.split(None, 1)[0] if text.strip() else ""
    if text == "/status":
        return json.dumps(app.status(), ensure_ascii=False, indent=2)
    if text == "/memory":
        return app.memory_overview()
    if text == "/refresh":
        app.reset_session_prompt(chat_id)
        return "已清空当前会话冻结的系统提示词，下一轮将重新加载最新规则与 MEMORY.md 索引。"
    if head == "/rag":
        return _handle_rag_command(app, text, chat_id)
    if head == "/search":
        rest = text.split(None, 1)[1].strip() if len(text.split(None, 1)) > 1 else ""
        return app.search(rest) if rest else "用法：/search <关键词>"
    return None


# 在主线程执行单轮对话期间通过异步信号安全管道将首次 SIGINT 转交后台线程触发协同取消，避免主线程锁自死锁与流重入
@contextmanager
def _sigint_to_abort(app: Any, chat_id: str) -> Iterator[AbortSignal | None]:
    if threading.current_thread() is not threading.main_thread():
        yield None
        return

    controller = AbortController()
    read_fd, write_fd = os.pipe()
    prev_handler = signal.getsignal(signal.SIGINT)
    handler_installed = False
    triggered = False
    helper: threading.Thread | None = None
    read_closed_by_helper = False

    # 监听管道字节：收到取消标记后在独立工作线程触发取消信号与 abort_turn，彻底规避主线程重入锁与 BufferedReader 冲突
    def _dispatch_abort() -> None:
        nonlocal read_closed_by_helper
        try:
            while True:
                try:
                    chunk = os.read(read_fd, 16)
                except InterruptedError:
                    continue
                except OSError:
                    break
                if not chunk or b"Q" in chunk:
                    break
                if b"A" in chunk:
                    controller.abort("用户按下 Ctrl+C 中断")
                    abort_fn = getattr(app, "abort_turn", None)
                    if callable(abort_fn):
                        try:
                            abort_fn(chat_id, reason="用户按下 Ctrl+C 中断")
                        except Exception:  # noqa: BLE001
                            pass
                    break
        finally:
            read_closed_by_helper = True
            try:
                os.close(read_fd)
            except OSError:
                pass

    # 首次 Ctrl+C 仅执行异步信号安全非阻塞管道写入唤醒后台线程；二次按下立即回退既有处理器（KeyboardInterrupt）
    def _handle_sigint(signum: int, frame: Any) -> None:
        nonlocal triggered
        if not triggered:
            triggered = True
            try:
                os.write(write_fd, b"A")
                return
            except OSError:
                pass
        if callable(prev_handler):
            prev_handler(signum, frame)
            return
        raise KeyboardInterrupt

    try:
        os.set_blocking(write_fd, False)
        helper = threading.Thread(target=_dispatch_abort, name=f"sigint-abort-{chat_id}", daemon=True)
        helper.start()
        signal.signal(signal.SIGINT, _handle_sigint)
        handler_installed = True
        yield controller.signal
    finally:
        if handler_installed:
            signal.signal(signal.SIGINT, prev_handler)
        try:
            os.write(write_fd, b"Q")
        except OSError:
            pass
        try:
            os.close(write_fd)
        except OSError:
            pass
        if helper is not None:
            helper.join(timeout=2.0)
        if not read_closed_by_helper:
            try:
                os.close(read_fd)
            except OSError:
                pass


# 启动交互式终端对话循环：基于 AssistantApp 持久化历史与实时流式输出
def run_repl(
    app: AssistantApp | None = None,
    *,
    llm: ModelInvoker | None = None,
    chat_id: str = "cli",
) -> None:
    owned = app is None
    active_app = app or build_app(llm=llm)
    try:
        tool_names = ", ".join(active_app.tools.names)
        print(f"My Helper REPL（可用工具：{tool_names}；命令：/status /memory /search /rag /refresh；exit 退出）")
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
            try:
                cmd_reply = _handle_repl_command(active_app, user_text, chat_id=chat_id)
            except Exception as error:  # noqa: BLE001
                print(f"助手> 命令执行出错：{error}")
                continue
            if cmd_reply is not None:
                print(f"助手>\n{cmd_reply}")
                continue
            on_event, finish = _stream_printer()
            try:
                with _sigint_to_abort(active_app, chat_id) as abort_sig:
                    result = active_app.chat(
                        user_text,
                        chat_id=chat_id,
                        on_event=on_event,
                        abort_signal=abort_sig,
                    )
            except LLMNotConfigured as error:
                print(f"助手> {error}")
                continue
            except (KeyboardInterrupt, StreamAbortedError):
                print("\n助手> 已中断本轮回复。")
                continue
            except Exception as error:  # noqa: BLE001
                print(f"\n助手> 本轮请求出错：{error}")
                continue
            finish(str(result.get("reply", "")))
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
    try:
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
    finally:
        tcfg.plugin_manager.terminate_all()


# 命令行总入口：加载 .env 并解析子命令（支持 setup 一键向导、chat 对话与 compile-config 编译）
def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="my-helper")
    parser.add_argument("--data", default=None, help="数据目录路径（默认读取 MY_HELPER_DATA 或 data）")
    parser.add_argument("--config", default=None, help="配置目录路径（默认读取 MY_HELPER_CONFIG 或 config）")
    parser.add_argument("--model", default=None, help="指定 models.dsl / models.json 中定义的模型档案名称")
    parser.add_argument("--consolidator-model", default=None, help="指定后台记忆沉淀专用模型档案名称")
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
        try:
            compiled = tcfg.load_models_config(cfg_dir, write_json=True)
            if compiled is None:
                print(f"未在 {cfg_dir} 下找到 models.dsl 或 models.json（运行 `uv run python -m personal_assistant setup` 可 10 秒自动生成）。")
                return
            print(f"已编译并写入：{cfg_dir / 'models.json'}（权限 0600，已由 .gitignore 保护）")
            print(json.dumps(compiled.redacted_dict(), ensure_ascii=False, indent=2))
            return
        finally:
            tcfg.plugin_manager.terminate_all()
    if args.command in (None, "chat"):
        app = build_app(
            data_root=Path(args.data) if args.data else None,
            config_root=cfg_dir,
            model_name=args.model,
            consolidator_model=args.consolidator_model,
        )
        try:
            if isinstance(app.llm, NoopModelClient):
                print("提示：当前尚未配置大模型。运行 `uv run python -m personal_assistant setup` 可 10 秒零代码一键配置。")
            run_repl(app)
        finally:
            app.close()
