"""核心应用编排（AssistantApp）：串联记忆、白名单工具、SQLite 历史与 ReAct 循环。"""
from __future__ import annotations

import os
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from personal_assistant.cache_metrics import CacheMonitor
from personal_assistant.commands import ToolsConfig
from personal_assistant.consolidator import (
    BackgroundConsolidator,
    ConsolidationPolicy,
    MemoryConsolidator,
)
from personal_assistant.db import StateStore
from personal_assistant.llm import LLMClient, NoopLLMClient, from_env
from personal_assistant.memory import FilesPolicy, MemoryStore, memory_tools
from personal_assistant.react import MAX_TOOL_ROUNDS, SYSTEM_PROMPT, run_react
from personal_assistant.tools import ToolRegistry

MAX_HISTORY_MESSAGES = 20
MAX_HISTORY_ITEM_CHARS = 2000


# 统一编排层：供终端 CLI 与本地 HTTP 接口共同调用
class AssistantApp:
    # 装配数据目录、工具白名单配置、Markdown 记忆库、SQLite 历史、大模型与后台沉淀器
    def __init__(
        self,
        data_root: Path | str,
        *,
        config_path: Path | None = None,
        project_root: Path | None = None,
        llm: LLMClient | None = None,
        memory: MemoryStore | None = None,
        state: StateStore | None = None,
        consolidation_policy: ConsolidationPolicy | None = None,
        consolidator_llm: LLMClient | None = None,
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
        self._prompt_lock = threading.Lock()
        self._session_prompts: dict[str, str] = {}
        self.cache_monitor = CacheMonitor()
        self.consolidator = MemoryConsolidator(
            self.state,
            self.memory,
            consolidator_llm or self.llm,
            policy=consolidation_policy,
        )
        self.background_consolidator = BackgroundConsolidator(self.consolidator)

    # 等待所有在途后台沉淀任务完成（供退出收尾或测试调用）
    def wait_for_consolidation(self, timeout: float = 15.0) -> None:
        self.background_consolidator.wait(timeout=timeout)

    # 先等待后台沉淀收尾，再关闭底层状态存储资源
    def close(self) -> None:
        self.background_consolidator.wait(timeout=15.0)
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

    # 拼装系统提示词：静态指令与规则在前、目录索引与日期在后，最大化前缀 KV Cache 复用
    def build_system_prompt(self, registry: ToolRegistry) -> str:
        tool_list = "、".join(registry.names) or "（无）"
        parts = [
            SYSTEM_PROMPT,
            f"【可用白名单工具】\n{tool_list}",
        ]
        rules = self.memory.read_rules_summary()
        if rules:
            parts.append(f"【行为规则 (data/memory/rules/)】\n{rules}")
        index = self.memory.read_memory_index()
        if index:
            parts.append(f"【记忆索引 (data/memory/MEMORY.md)】\n{index}")
        today = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
        parts.append(f"今天是 {today}（北京时间）。")
        return "\n\n".join(parts)

    # 获取指定会话冻结的系统提示词：整场会话复用首次快照，防止中途改写记忆打穿 KV Cache
    def get_session_system_prompt(self, chat_id: str, registry: ToolRegistry) -> str:
        with self._prompt_lock:
            prompt = self._session_prompts.get(chat_id)
            if prompt is None:
                prompt = self.build_system_prompt(registry)
                self._session_prompts[chat_id] = prompt
            return prompt

    # 主动重置指定会话（或全部会话）冻结的系统提示词，用于后台沉淀或手动改文件后刷新
    def reset_session_prompt(self, chat_id: str | None = None) -> None:
        with self._prompt_lock:
            if chat_id is None:
                self._session_prompts.clear()
            else:
                self._session_prompts.pop(chat_id, None)

    # 将持久化历史转换为标准追加式消息列表，保持前缀字节级稳定以命中跨轮 KV Cache
    @staticmethod
    def _build_history_messages(history: list[dict[str, object]]) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for msg in history:
            role = str(msg.get("role") or "")
            if role not in {"user", "assistant"}:
                continue
            content = str(msg.get("content") or "")
            if len(content) > MAX_HISTORY_ITEM_CHARS:
                content = content[:MAX_HISTORY_ITEM_CHARS] + "…"
            result.append({"role": role, "content": content})
        return result

    # 处理一轮完整对话：加载摘要与活跃历史、执行 ReAct 循环、持久化回答并在达阈值时后台自动沉淀
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

        compaction = self.state.get_compaction(chat_id)
        history = self.state.active_messages(chat_id, limit=MAX_HISTORY_MESSAGES)
        user_msg_id = self.state.append_message(chat_id, "user", clean_text)
        registry = self.build_tools(user_text=clean_text, message_id=user_msg_id)
        system_prompt = self.get_session_system_prompt(chat_id, registry)
        messages: list[dict[str, object]] = [{"role": "system", "content": system_prompt}]
        summary_text = str(compaction.get("summary") or "").strip()
        if summary_text:
            # 使用标准 user/assistant 承接对注入前情摘要，保证仅含单条 system 消息且兼容全部模型接口
            messages.append(
                {
                    "role": "user",
                    "content": f"【本会话已压缩前情摘要（供承接上下文）】\n{summary_text}",
                }
            )
            messages.append({"role": "assistant", "content": "已了解前情摘要，请继续。"})
        messages.extend(self._build_history_messages(history))
        messages.append({"role": "user", "content": clean_text})

        traces: list[dict[str, str]] = []

        # 记录本轮工具轨迹并在需要时转发给终端等外部回调
        def record_step(name: str, args: str, obs: str) -> None:
            traces.append({"tool": name, "arguments": args, "observation": obs})
            if on_step is not None:
                on_step(name, args, obs)

        start_idx = self.cache_monitor.snapshot_index()
        tracked_llm = self.cache_monitor.wrap(self.llm, session_key=chat_id)
        reply = run_react(
            tracked_llm,
            messages,
            registry,
            max_rounds=max_rounds,
            on_step=record_step,
        )
        assistant_msg_id = self.state.append_message(chat_id, "assistant", reply)
        scheduled = self.background_consolidator.maybe_schedule(
            chat_id,
            on_complete=lambda res: self.reset_session_prompt(res.chat_id),
        )
        return {
            "reply": reply,
            "chat_id": chat_id,
            "message_id": assistant_msg_id,
            "tool_calls": traces,
            "cache": self.cache_monitor.summary(since=start_idx),
            "consolidation_scheduled": scheduled,
        }

    # 不经过模型，直接按关键词搜索本地 Markdown 文件（离线可用）
    def search(self, query: str, root: str = "data/memory", limit: int = 10) -> str:
        return self.memory.search_markdown(query, root=root, limit=limit)

    # 汇总当前 MEMORY.md 索引、rules/ 规则与最近 facts.md 备忘，供 /memory 命令查看
    def memory_overview(self) -> str:
        index = self.memory.read_memory_index() or "（暂无记忆索引）"
        rules = self.memory.read_rules_summary() or "（暂无行为规则）"
        facts = self.memory.read_recent_facts() or "（暂无事实备忘）"
        return (
            f"【记忆索引 (MEMORY.md)】\n{index}\n\n"
            f"【行为规则 (rules/)】\n{rules}\n\n"
            f"【事实备忘 (facts.md，按需检索不注入 Prompt)】\n{facts}"
        )

    # 返回系统运行状态摘要（LLM 配置状态、记忆文件数、消息数、白名单工具及已加载插件）
    def status(self) -> dict[str, object]:
        memory_count = (
            sum(
                1
                for p in self.memory.memory_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in {".md", ".txt"} and not p.name.startswith(".")
            )
            if self.memory.memory_dir.is_dir()
            else 0
        )
        registry = self.build_tools()
        return {
            "status": "ok",
            "llm_configured": not isinstance(self.llm, NoopLLMClient),
            "memory_files": memory_count,
            "messages": self.state.count_messages(),
            "tools": registry.names,
            "plugins": list(self.tools_config.plugins) if self.tools_config else [],
            "cache": self.cache_monitor.summary(),
            "consolidation": self.background_consolidator.status(),
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
