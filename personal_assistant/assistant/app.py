"""核心应用编排（AssistantApp）：串联记忆、白名单工具、SQLite 历史与 ReAct 循环。"""
from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterable, Iterator
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from personal_assistant.agent.plugins import ObserverProtocol
from personal_assistant.agent.react import (
    MAX_TOOL_ROUNDS,
    AgentEvent,
    RuntimeEvent,
    ToolStep,
    TurnEnd,
    iter_react,
)
from personal_assistant.agent.tools import ToolRegistry
from personal_assistant.ai.cache import CacheMonitor
from personal_assistant.ai.client import LLMClient, NoopLLMClient, from_env
from personal_assistant.ai.types import TextDelta
from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.assistant.consolidator import (
    BackgroundConsolidator,
    ConsolidationPolicy,
    MemoryConsolidator,
)
from personal_assistant.assistant.db import StateStore
from personal_assistant.assistant.memory import (
    FilesPolicy,
    MemoryStore,
    MemoryToolsetProvider,
)
from personal_assistant.assistant.prompts import format_system_prompt

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
        today_fn: Callable[[], date] | None = None,
        model_name: str | None = None,
        observers: Iterable[ObserverProtocol] | None = None,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.data_root.mkdir(parents=True, exist_ok=True)
        self._today_fn = today_fn

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
            retriever = self.tools_config.build_retriever()
            self.memory = MemoryStore(self.data_root, policy=policy, retriever=retriever)
        else:
            self.memory = MemoryStore(self.data_root)

        self.state = state or StateStore(self.data_root / "state" / "chat_history.sqlite3")
        if llm is not None:
            self.llm = llm
        elif self.tools_config is not None and self.config_path is not None:
            self.llm = self.tools_config.build_llm(self.config_path.parent, model_name=model_name)
        else:
            self.llm = from_env()
        self._prompt_lock = threading.Lock()
        self._session_prompts: dict[str, str] = {}
        self._last_archive_day = ""
        self.cache_monitor = CacheMonitor()
        if observers is not None:
            self.observers: tuple[ObserverProtocol, ...] = tuple(observers)
        elif self.tools_config is not None:
            self.observers = tuple(self.tools_config.build_observers())
        else:
            self.observers = ()
        self._custom_consolidator_llm = consolidator_llm is not None
        self.consolidator = MemoryConsolidator(
            self.state,
            self.memory,
            consolidator_llm or self.llm,
            policy=consolidation_policy,
        )
        self.background_consolidator = BackgroundConsolidator(
            self.consolidator,
            on_event=self._emit_event,
        )

    # 安全向所有已装配观测插件广播运行期事件，确保任何插件异常不拖垮微内核
    def _emit_event(self, session_key: str, event: RuntimeEvent) -> None:
        for observer in self.observers:
            try:
                observer.on_event(session_key, event)
            except Exception:  # noqa: BLE001
                continue

    # 安全读取各观测插件的只读快照，故障插件记录 error 字段而不阻断 /status
    def _collect_telemetry(self, *, session_key: str | None = None) -> dict[str, object]:
        snapshots: dict[str, object] = {}
        for observer in self.observers:
            try:
                snapshots[observer.name] = observer.snapshot(session_key=session_key)
            except Exception as error:  # noqa: BLE001
                snapshots[observer.name] = {"error": f"{type(error).__name__}: {error}"}
        return snapshots

    # 返回内置模型厂商预设目录，供零代码向导与网页设置表单展示
    def model_presets(self) -> list[dict[str, object]]:
        if self.tools_config is None:
            return []
        return self.tools_config.model_presets()

    # 返回当前模型预设与已配置模型的脱敏视图（绝不包含明文 api_key）
    def models_overview(self) -> dict[str, object]:
        configured: dict[str, object] | None = None
        if self.tools_config is not None and self.config_path is not None:
            cfg = self.tools_config.load_models_config(self.config_path.parent)
            if cfg is not None:
                configured = cfg.redacted_dict()
        return {
            "presets": self.model_presets(),
            "configured": configured,
            "llm_configured": not isinstance(self.llm, NoopLLMClient),
        }

    # 按最新 models.dsl / models.json 热切换当前运行中的 LLMClient（无需重启服务）
    def reload_llm(self, model_name: str | None = None) -> None:
        if self.tools_config is not None and self.config_path is not None:
            new_llm = self.tools_config.build_llm(self.config_path.parent, model_name=model_name)
        else:
            new_llm = from_env()
        self.llm = new_llm
        if not self._custom_consolidator_llm:
            self.consolidator.llm = new_llm

    # 零代码自动生成 models.dsl 与 models.json（0600 权限）并立即热生效，返回脱敏摘要
    def configure_model(
        self,
        *,
        preset: str,
        api_key: str = "",
        model: str = "",
        base_url: str = "",
        name: str = "",
        temperature: float = 0.2,
    ) -> dict[str, object]:
        if self.tools_config is None or self.config_path is None:
            raise ValueError("当前实例未加载 config/tools.json，无法写入模型配置。")
        models_cfg = self.tools_config.save_model_setup(
            self.config_path.parent,
            preset=preset,
            api_key=api_key,
            model=model,
            base_url=base_url,
            name=name,
            temperature=temperature,
        )
        self.reload_llm(model_name=models_cfg.default)
        return models_cfg.redacted_dict()

    # 等待所有在途后台沉淀任务完成（供退出收尾或测试调用）
    def wait_for_consolidation(self, timeout: float = 15.0) -> None:
        self.background_consolidator.wait(timeout=timeout)

    # 先等待后台沉淀收尾，再关闭底层状态存储资源
    def close(self) -> None:
        self.background_consolidator.wait(timeout=15.0)
        self.state.close()

    # 返回当前逻辑日期（默认北京时间，评测时可通过 today_fn 注入历史日期）
    def _today(self) -> date:
        if self._today_fn is not None:
            return self._today_fn()
        return datetime.now(ZoneInfo("Asia/Shanghai")).date()

    # 为单轮对话构建受白名单管控的统一工具注册表（汇聚 MemoryToolsetProvider、PluginManager 与 CommandToolWrapper）
    def build_tools(
        self,
        user_text: str = "",
        message_id: int | None = None,
    ) -> ToolRegistry:
        getter = (lambda: (user_text, message_id)) if user_text else None
        provider = MemoryToolsetProvider(self.memory, get_user_message=getter, get_today=self._today)
        if self.tools_config is not None:
            tool_ctx: dict[str, object] = {"history_db_path": self.state.db_path}
            if message_id is not None:
                tool_ctx["before_msg_id"] = message_id
            return self.tools_config.build_registry(
                providers=[provider],
                data_root=self.data_root,
                tool_context=tool_ctx,
            )
        registry = ToolRegistry()
        registry.register_provider(provider)
        return registry

    # 拼装系统提示词：静态指令与规则在前、目录索引与日期在后，最大化前缀 KV Cache 复用
    def build_system_prompt(self, registry: ToolRegistry) -> str:
        return format_system_prompt(self.memory, tool_names=registry.names, today=self._today())

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

    # 每个自然日首次对话时自动扫描归档已过期的临时记忆（纯本地零 Token 耗时 < 1ms）
    def _maybe_archive_expired(self) -> None:
        today = self._today()
        today_str = today.isoformat()
        with self._prompt_lock:
            if self._last_archive_day == today_str:
                return
            self._last_archive_day = today_str
        self.memory.archive_expired_memories(today=today)

    # 处理一轮流式对话（单一主流程）：先同步校验输入与模型就绪状态，再返回事件帧生成器
    def stream_chat(
        self,
        text: str,
        chat_id: str = "default",
        *,
        max_rounds: int = MAX_TOOL_ROUNDS,
        on_event: Callable[[AgentEvent], None] | None = None,
    ) -> Iterator[dict[str, object]]:
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("消息内容不能为空。")
        if isinstance(self.llm, NoopLLMClient):
            self.llm.respond([{"role": "user", "content": clean_text}])
        return self._run_stream_turn(
            clean_text,
            chat_id,
            max_rounds=max_rounds,
            on_event=on_event,
        )

    # 内部生成器：装配冻结上下文与会话历史、执行流式 ReAct 并将事件转发给全部观测插件，结束后持久化并触发沉淀
    def _run_stream_turn(
        self,
        clean_text: str,
        chat_id: str,
        *,
        max_rounds: int,
        on_event: Callable[[AgentEvent], None] | None,
    ) -> Iterator[dict[str, object]]:
        self._maybe_archive_expired()
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

        traces: list[dict[str, object]] = []
        reply = ""
        start_idx = self.cache_monitor.snapshot_index()
        tracked_llm = self.cache_monitor.wrap(self.llm, session_key=chat_id)
        for event in iter_react(
            tracked_llm,
            messages,
            registry,
            max_rounds=max_rounds,
        ):
            self._emit_event(chat_id, event)
            if on_event is not None:
                on_event(event)
            if isinstance(event, TextDelta):
                yield {"type": "text_delta", "text": event.text}
            elif isinstance(event, ToolStep):
                step_dict: dict[str, object] = {
                    "tool": event.tool,
                    "arguments": event.arguments,
                    "observation": event.observation,
                    "duration_ms": event.duration_ms,
                }
                traces.append(step_dict)
                yield {"type": "tool_step", **step_dict}
            elif isinstance(event, TurnEnd):
                reply = event.reply

        assistant_msg_id = self.state.append_message(chat_id, "assistant", reply)
        scheduled = self.background_consolidator.maybe_schedule(
            chat_id,
            on_complete=lambda res: self.reset_session_prompt(res.chat_id),
        )
        done_payload: dict[str, object] = {
            "type": "done",
            "reply": reply,
            "chat_id": chat_id,
            "message_id": assistant_msg_id,
            "tool_calls": traces,
            "cache": self.cache_monitor.summary(since=start_idx),
            "consolidation_scheduled": scheduled,
        }
        telemetry = self._collect_telemetry(session_key=chat_id)
        if telemetry:
            done_payload["telemetry"] = telemetry
        yield done_payload

    # 处理一轮完整同步对话：直接消费 stream_chat 单一实现并提取最终 done 结果字典
    def chat(
        self,
        text: str,
        chat_id: str = "default",
        *,
        max_rounds: int = MAX_TOOL_ROUNDS,
        on_event: Callable[[AgentEvent], None] | None = None,
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for frame in self.stream_chat(
            text,
            chat_id=chat_id,
            max_rounds=max_rounds,
            on_event=on_event,
        ):
            if frame.get("type") == "done":
                result = {k: v for k, v in frame.items() if k != "type"}
        return result

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

    # 返回系统运行状态摘要（LLM 配置状态、记忆文件数、消息数、工具、插件与观测快照）
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
            "retriever": str(getattr(self.memory.retriever, "name", "linear")),
            "observers": [obs.name for obs in self.observers],
            "telemetry": self._collect_telemetry(),
            "cache": self.cache_monitor.summary(),
            "consolidation": self.background_consolidator.status(),
        }


# 按环境变量或显式路径构建默认 AssistantApp 实例（自动优先加载 models.dsl / models.json）
def build_app(
    data_root: Path | None = None,
    config_root: Path | None = None,
    llm: LLMClient | None = None,
    *,
    project_root: Path | None = None,
    model_name: str | None = None,
) -> AssistantApp:
    data_dir = (data_root or Path(os.getenv("MY_HELPER_DATA", "data"))).resolve()
    cfg_dir = (config_root or Path(os.getenv("MY_HELPER_CONFIG", "config"))).resolve()
    tools_json = cfg_dir / "tools.json"
    proj = (
        project_root.resolve()
        if project_root is not None
        else (cfg_dir.parent if tools_json.is_file() else data_dir.parent)
    )
    return AssistantApp(
        data_root=data_dir,
        config_path=tools_json if tools_json.is_file() else None,
        project_root=proj,
        llm=llm,
        model_name=model_name,
    )
