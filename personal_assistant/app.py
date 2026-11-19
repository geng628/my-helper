"""核心应用编排（AssistantApp）：串联记忆、白名单工具、SQLite 历史与 ReAct 循环。"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from personal_assistant.commands import ToolsConfig
from personal_assistant.db import StateStore
from personal_assistant.llm import LLMClient, NoopLLMClient, from_env
from personal_assistant.memory import FilesPolicy, MemoryStore, _atomic_write, memory_tools
from personal_assistant.react import MAX_TOOL_ROUNDS, SYSTEM_PROMPT, run_react
from personal_assistant.tools import ToolRegistry

MAX_HISTORY_MESSAGES = 12
MAX_HISTORY_ITEM_CHARS = 600
MAX_NOTE_BYTES = 2 * 1024 * 1024


# 统一编排层：供终端 CLI 与本地 HTTP 接口共同调用
class AssistantApp:
    # 装配数据目录、工具白名单配置、Markdown 记忆库、SQLite 历史与大模型客户端
    def __init__(
        self,
        data_root: Path | str,
        *,
        config_path: Path | None = None,
        project_root: Path | None = None,
        llm: LLMClient | None = None,
        memory: MemoryStore | None = None,
        state: StateStore | None = None,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.data_root.mkdir(parents=True, exist_ok=True)

        self.config_path = config_path.resolve() if config_path is not None else None
        self.project_root = (
            project_root.resolve()
            if project_root is not None
            else (self.config_path.parent.parent if self.config_path else self.data_root.parent)
        )
        self.tools_config = (
            ToolsConfig.load(self.config_path, project_root=self.project_root)
            if (self.config_path is not None and self.config_path.is_file())
            else None
        )

        if memory is not None:
            self.memory = memory
        elif self.tools_config is not None:
            policy = FilesPolicy.from_dict(
                self.tools_config.files,
                project_root=self.project_root,
                data_root=self.data_root,
            )
            self.memory = MemoryStore(self.data_root, policy=policy)
        else:
            self.memory = MemoryStore(self.data_root)

        self.state = state or StateStore(self.data_root / "state" / "chat_history.sqlite3")
        self.llm = llm or from_env()

    # 关闭底层状态存储资源
    def close(self) -> None:
        self.state.close()

    # 为单轮对话构建受白名单管控的工具注册表，并将当前用户消息绑定到 remember_fact
    def build_tools(
        self,
        user_text: str = "",
        message_id: int | None = None,
    ) -> ToolRegistry:
        getter = (lambda: (user_text, message_id)) if user_text else None
        builtins = memory_tools(self.memory, get_user_message=getter)
        if self.tools_config is not None:
            return self.tools_config.build_registry(*builtins)
        return ToolRegistry(*builtins)

    # 拼装本轮系统提示词：注入日期、当前可用白名单工具名、行为规则与 MEMORY.md 核心记忆
    def build_system_prompt(self, registry: ToolRegistry) -> str:
        today = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
        tool_list = "、".join(registry.names) or "（无）"
        parts = [
            SYSTEM_PROMPT,
            f"今天是 {today}（北京时间）。",
            f"【可用白名单工具】\n{tool_list}",
        ]
        rules = self.memory.read_rules_summary()
        if rules:
            parts.append(f"【行为规则 (data/memory/rules/)】\n{rules}")
        core = self.memory.read_core_memory()
        if core:
            parts.append(f"【长期记忆 (data/memory/MEMORY.md)】\n{core}")
        return "\n\n".join(parts)

    # 构造发给模型的 user 消息：若有历史记录则以参考上下文 JSON 包装，防止旧拒答误导本轮
    @staticmethod
    def _format_user_input(text: str, history: list[dict[str, object]]) -> str:
        if not history:
            return text
        compact = []
        for msg in history:
            content = str(msg.get("content", ""))
            if len(content) > MAX_HISTORY_ITEM_CHARS:
                content = content[:MAX_HISTORY_ITEM_CHARS] + "…"
            compact.append({"role": msg["role"], "content": content})
        return json.dumps(
            {"history_reference": compact, "current_user_message": text},
            ensure_ascii=False,
        )

    # 处理一轮完整对话：持久化提问、执行 ReAct 循环、持久化回答并返回结果与工具轨迹
    def chat(
        self,
        text: str,
        chat_id: str = "default",
        *,
        max_rounds: int = MAX_TOOL_ROUNDS,
        on_step: Callable[[str, str, str], None] | None = None,
    ) -> dict[str, object]:
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("消息内容不能为空。")

        history = self.state.recent_messages(chat_id, limit=MAX_HISTORY_MESSAGES)
        user_msg_id = self.state.append_message(chat_id, "user", clean_text)
        registry = self.build_tools(user_text=clean_text, message_id=user_msg_id)
        messages: list[dict[str, object]] = [
            {"role": "system", "content": self.build_system_prompt(registry)},
            {"role": "user", "content": self._format_user_input(clean_text, history)},
        ]

        traces: list[dict[str, str]] = []

        # 记录本轮工具轨迹并在需要时转发给终端等外部回调
        def record_step(name: str, args: str, obs: str) -> None:
            traces.append({"tool": name, "arguments": args, "observation": obs})
            if on_step is not None:
                on_step(name, args, obs)

        reply = run_react(
            self.llm,
            messages,
            registry,
            max_rounds=max_rounds,
            on_step=record_step,
        )
        assistant_msg_id = self.state.append_message(chat_id, "assistant", reply)
        return {
            "reply": reply,
            "chat_id": chat_id,
            "message_id": assistant_msg_id,
            "tool_calls": traces,
        }

    # 不经过模型，直接按关键词搜索本地 Markdown 笔记（离线可用）
    def search(self, query: str, root: str = "data/knowledge", limit: int = 10) -> str:
        return self.memory.search_markdown(query, root=root, limit=limit)

    # 汇总当前 MEMORY.md 与 rules/ 规则文本，供 /memory 命令查看
    def memory_overview(self) -> str:
        core = self.memory.read_core_memory() or "（暂无长期记忆）"
        rules = self.memory.read_rules_summary() or "（暂无行为规则）"
        return f"【长期记忆 (MEMORY.md)】\n{core}\n\n【行为规则 (rules/)】\n{rules}"

    # 安全收存上传的 .md/.txt 笔记到 data/knowledge/inbox/，按内容哈希去重且不覆盖旧文件
    def ingest_note(self, filename: str, content: str) -> dict[str, object]:
        if not isinstance(filename, str) or not isinstance(content, str):
            raise ValueError("文件名和内容必须是文本。")
        name = filename.strip()
        if (
            not name
            or len(name) > 160
            or "/" in name
            or "\\" in name
            or name.startswith(".")
            or any(ord(ch) < 32 or ord(ch) == 127 for ch in name)
        ):
            raise ValueError("文件名无效：只接受普通文件名。")
        suffix = Path(name).suffix.lower()
        if suffix not in {".md", ".txt"}:
            raise ValueError("只支持 .md 或 .txt 笔记文件。")
        if suffix == ".txt":
            name = f"{Path(name).stem}.md"
        if not content.strip():
            raise ValueError("笔记内容不能为空。")

        payload = content.encode("utf-8")
        if len(payload) > MAX_NOTE_BYTES:
            raise ValueError(f"笔记超过 {MAX_NOTE_BYTES} 字节上限。")

        inbox = self.memory.inbox_dir
        inbox.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(payload).hexdigest()
        target = inbox / name
        duplicate = False

        if target.exists():
            if target.is_symlink() or not target.is_file():
                raise ValueError("同名目标不是普通文件。")
            if hashlib.sha256(target.read_bytes()).hexdigest() == digest:
                duplicate = True
            else:
                stem = Path(name).stem
                counter = 0
                while True:
                    tail = digest[:8] if counter == 0 else f"{digest[:8]}-{counter + 1}"
                    candidate = inbox / f"{stem}-{tail}.md"
                    if not candidate.exists():
                        target = candidate
                        break
                    if (
                        candidate.is_file()
                        and not candidate.is_symlink()
                        and hashlib.sha256(candidate.read_bytes()).hexdigest() == digest
                    ):
                        target = candidate
                        duplicate = True
                        break
                    counter += 1
                    if counter > 20:
                        raise ValueError("同名文件过多，请改名后重试。")

        if not duplicate:
            _atomic_write(target, content)
        return {
            "path": self.memory.display_path(target.resolve()),
            "duplicate": duplicate,
        }

    # 返回系统运行状态摘要（LLM 配置状态、笔记数、消息数、白名单工具及已加载插件）
    def status(self) -> dict[str, object]:
        knowledge_count = sum(
            1
            for p in self.memory.knowledge_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".md", ".txt"} and not p.name.startswith(".")
        )
        registry = self.build_tools()
        return {
            "status": "ok",
            "llm_configured": not isinstance(self.llm, NoopLLMClient),
            "knowledge_files": knowledge_count,
            "messages": self.state.count_messages(),
            "tools": registry.names,
            "plugins": list(self.tools_config.plugins) if self.tools_config else [],
        }


# 按环境变量或显式路径构建默认 AssistantApp 实例
def build_app(
    data_root: Path | None = None,
    config_root: Path | None = None,
    llm: LLMClient | None = None,
) -> AssistantApp:
    data_dir = (data_root or Path(os.getenv("MY_HELPER_DATA", "data"))).resolve()
    cfg_dir = (config_root or Path(os.getenv("MY_HELPER_CONFIG", "config"))).resolve()
    tools_json = cfg_dir / "tools.json"
    return AssistantApp(
        data_root=data_dir,
        config_path=tools_json if tools_json.is_file() else None,
        project_root=cfg_dir.parent if tools_json.is_file() else data_dir.parent,
        llm=llm,
    )
