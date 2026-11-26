"""核心应用编排（AssistantApp）：串联记忆、白名单工具、SQLite 历史与 ReAct 循环。"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from contextlib import closing, contextmanager
from datetime import date
from pathlib import Path

from personal_assistant.agent.plugins import ObserverProtocol, RuntimeEvent
from personal_assistant.agent.react import (
    AgentEvent,
    ToolStart,
    ToolStep,
    TurnEnd,
    iter_react,
)
from personal_assistant.agent.tools import ToolRegistry
from personal_assistant.ai.cache import CacheMonitor
from personal_assistant.ai.client import LLMClient, LLMNotConfigured, NoopLLMClient
from personal_assistant.ai.types import (
    LLMStreamEvent,
    ModelsConfig,
    StreamDone,
    TextDelta,
    ThinkingDelta,
    TokenUsage,
    extract_tool_calls,
)
from personal_assistant.assistant.builtin_tools import MemoryToolsetProvider
from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.assistant.common import require_positive_int, today_shanghai
from personal_assistant.assistant.consolidator import (
    BackgroundConsolidator,
    ConsolidationPolicy,
    MemoryConsolidator,
)
from personal_assistant.assistant.db import StateStore
from personal_assistant.assistant.files_policy import FilesPolicy
from personal_assistant.assistant.memory import MemoryStore
from personal_assistant.assistant.prompts import format_system_prompt

MAX_HISTORY_MESSAGES = 20
MAX_ADAPTIVE_HISTORY_MESSAGES = 60
MAX_ADAPTIVE_HISTORY_CHARS = 24_000
MAX_HISTORY_ITEM_CHARS = 2000
MAX_SESSION_PROMPTS = 256
MAX_SESSION_LOCKS = 256


# 单轮推理或工具调用超出运行期安全预算时构造的结构化异常
class TurnBudgetExceeded(RuntimeError):
    # 记录熔断类别与人类可读原因说明
    def __init__(self, kind: str, reason: str) -> None:
        self.kind = kind
        self.reason = reason
        super().__init__(reason)


# 校验运行期浮点秒数预算配置为非负有限数（严格拒绝布尔值、NaN 与 Inf）
def _validate_non_negative_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} 必须是非负有限数字。")
    return float(value)


# 将工具调用原始 JSON 参数规范化为紧凑有序串，便于检测等价参数的死循环重复调用
def _canonical_tool_args(raw_arguments: str) -> str:
    text = (raw_arguments or "{}").strip() or "{}"
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return " ".join(text.split())
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# 检查助手消息的 tool_calls 参数是否为非法/截断 JSON（用于 finish_reason == "length" 熔断判定）
def _has_truncated_tool_call_json(message: Mapping[str, object]) -> bool:
    for call in extract_tool_calls(message):
        try:
            parsed = json.loads(call.arguments or "{}")
        except (json.JSONDecodeError, TypeError):
            return True
        if not isinstance(parsed, dict):
            return True
    return False


# 当未压缩活跃消息超出常规滑窗上限时，在字符与条数总预算内自适应保留更早消息，防止沉淀滞后静默丢上下文
def _select_adaptive_history(
    active_all: list[dict[str, object]],
    *,
    base_messages: int = MAX_HISTORY_MESSAGES,
    max_messages: int = MAX_ADAPTIVE_HISTORY_MESSAGES,
    max_chars: int = MAX_ADAPTIVE_HISTORY_CHARS,
) -> list[dict[str, object]]:
    if len(active_all) <= base_messages:
        return active_all
    selected_rev: list[dict[str, object]] = []
    used_chars = 0
    for idx, msg in enumerate(reversed(active_all)):
        item_len = min(len(str(msg.get("content") or "")), MAX_HISTORY_ITEM_CHARS)
        if idx >= base_messages and (len(selected_rev) >= max_messages or used_chars + item_len > max_chars):
            break
        selected_rev.append(msg)
        used_chars += item_len
    while len(selected_rev) > 1 and selected_rev[-1].get("role") == "assistant":
        selected_rev.pop()
    selected_rev.reverse()
    return selected_rev


# 外层回合预算护栏包装器：在零改动 Layer 2 微内核的前提下执行步数、Token、墙钟与死循环熔断并优雅收束
class _GuardedTurnLLM(LLMClient):
    # 绑定底层 LLMClient、预算上限与单调时钟
    def __init__(
        self,
        inner: LLMClient,
        *,
        max_steps: int,
        max_tokens: int,
        max_seconds: float,
        max_repeated_tool_calls: int,
        clock: Callable[[], float],
    ) -> None:
        self._inner = inner
        self._max_steps = max_steps
        self._max_tokens = max_tokens
        self._max_seconds = max_seconds
        self._max_repeated = max_repeated_tool_calls
        self._clock = clock
        self._started_at = clock()
        self.steps = 0
        self.total_tokens = 0
        self.truncated_tool_call = False
        self.last_tool_sig: tuple[str, str] | None = None
        self.repeat_streak = 0
        self.breach: TurnBudgetExceeded | None = None
        self.last_usage: TokenUsage | None = None

    # 记录每步已执行工具的规范化签名，累计连续相同调用次数
    def observe_tool_call(self, tool_name: str, raw_arguments: str) -> None:
        sig = (tool_name, _canonical_tool_args(raw_arguments))
        if sig == self.last_tool_sig:
            self.repeat_streak += 1
        else:
            self.last_tool_sig = sig
            self.repeat_streak = 1

    # 进入新一轮模型推理前核查四大预算护栏，命中任一阈值即返回 TurnBudgetExceeded
    def _check_budget(self) -> TurnBudgetExceeded | None:
        if self.steps >= self._max_steps:
            return TurnBudgetExceeded("max_steps", f"推理步数达到上限 {self._max_steps}")
        elapsed = self._clock() - self._started_at
        if elapsed > self._max_seconds:
            return TurnBudgetExceeded("max_seconds", f"单轮耗时超过上限 {self._max_seconds:g}s")
        if self.total_tokens > self._max_tokens:
            return TurnBudgetExceeded("max_tokens", f"单轮累计 Token 超过上限 {self._max_tokens}")
        if self.truncated_tool_call:
            return TurnBudgetExceeded("truncated_tool_calls", "模型输出因长度上限截断导致工具参数 JSON 不完整")
        if self.repeat_streak >= self._max_repeated and self.last_tool_sig is not None:
            return TurnBudgetExceeded(
                "repeated_tool_calls",
                f"工具 {self.last_tool_sig[0]} 连续重复相同参数调用 {self.repeat_streak} 次",
            )
        return None

    # 构造不带 tool_calls 的优雅降级收束消息，使 iter_react 自然退出并释放会话锁
    @staticmethod
    def _fallback_message(breach: TurnBudgetExceeded) -> dict[str, object]:
        return {
            "role": "assistant",
            "content": (
                f"抱歉，本轮工具调用达到安全预算上限（原因：{breach.reason}），"
                "已自动停止继续重试。请缩小问题范围或重试。"
            ),
        }

    # 累加单次响应 TokenUsage（未触发熔断时同步 last_usage）
    def _record_usage(self, usage: object) -> None:
        if isinstance(usage, TokenUsage):
            self.total_tokens += usage.prompt_tokens + usage.completion_tokens
            self.last_usage = usage
        else:
            self.last_usage = None

    # 同步调用前执行预算核查，触发熔断时不再请求远端并直接返回收束答复
    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        breach = self._check_budget()
        if breach is not None:
            self.breach = breach
            self.last_usage = None
            return self._fallback_message(breach)
        self.steps += 1
        self.last_usage = None
        msg = self._inner.respond(messages, tools)
        self._record_usage(self._inner.last_usage)
        return msg

    # 流式调用前执行预算核查，触发熔断时吐出降级文本与无 tool_calls 的 StreamDone 使循环自然退出
    def stream(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> Iterator[LLMStreamEvent]:
        breach = self._check_budget()
        if breach is not None:
            self.breach = breach
            self.last_usage = None
            fallback_msg = self._fallback_message(breach)
            fallback_text = str(fallback_msg["content"])
            yield TextDelta(text=fallback_text)
            yield StreamDone(message=fallback_msg, usage=None, ttft_ms=None, latency_ms=0.0)
            return
        self.steps += 1
        self.last_usage = None
        for event in self._inner.stream(messages, tools):
            if isinstance(event, StreamDone):
                usage = event.usage if isinstance(event.usage, TokenUsage) else self._inner.last_usage
                self._record_usage(usage)
                if event.finish_reason == "length" and _has_truncated_tool_call_json(event.message):
                    self.truncated_tool_call = True
            yield event


# 向消息列表追加对话消息，遇到连续同角色（不含 tool_calls/reasoning_content）时安全合并以防推理模型报错
def _append_alternating_message(
    messages: list[dict[str, object]],
    message: dict[str, object],
) -> None:
    role = str(message.get("role") or "")
    if (
        messages
        and role in {"user", "assistant"}
        and messages[-1].get("role") == role
        and "tool_calls" not in messages[-1]
        and "tool_calls" not in message
        and "reasoning_content" not in messages[-1]
        and "reasoning_content" not in message
    ):
        prev_content = str(messages[-1].get("content") or "")
        curr_content = str(message.get("content") or "")
        merged = (
            f"{prev_content}\n\n{curr_content}"
            if (prev_content and curr_content)
            else (prev_content or curr_content)
        )
        messages[-1] = {**messages[-1], "content": merged}
        return
    messages.append(message)


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
        auto_consolidate: bool = True,
        today_fn: Callable[[], date] | None = None,
        model_name: str | None = None,
        consolidator_model: str | None = None,
        observers: Iterable[ObserverProtocol] | None = None,
        max_turn_steps: int = 24,
        max_turn_tokens: int = 250_000,
        max_turn_seconds: float = 180.0,
        max_repeated_tool_calls: int = 3,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.max_turn_steps = require_positive_int(max_turn_steps, "max_turn_steps")
        self.max_turn_tokens = require_positive_int(max_turn_tokens, "max_turn_tokens")
        self.max_turn_seconds = _validate_non_negative_float(max_turn_seconds, "max_turn_seconds")
        self.max_repeated_tool_calls = require_positive_int(
            max_repeated_tool_calls, "max_repeated_tool_calls"
        )
        self._clock: Callable[[], float] = clock or time.monotonic
        self.auto_consolidate = bool(auto_consolidate)
        self.data_root = Path(data_root).resolve()
        self.data_root.mkdir(parents=True, exist_ok=True)
        self._today_fn = today_fn

        self.config_path = config_path.resolve() if config_path is not None else None
        self.project_root = (
            project_root.resolve()
            if project_root is not None
            else (self.config_path.parent.parent if self.config_path else self.data_root.parent)
        )
        self.tools_config: ToolsConfig = (
            ToolsConfig.load(self.config_path, project_root=self.project_root)
            if (self.config_path is not None and self.config_path.is_file())
            else ToolsConfig.empty(self.project_root)
        )

        if memory is not None:
            self.memory = memory
        else:
            policy = FilesPolicy.from_dict(
                self.tools_config.files,
                project_root=self.project_root,
                data_root=self.data_root,
            )
            retriever = self.tools_config.build_retriever()
            self.memory = MemoryStore(self.data_root, policy=policy, retriever=retriever)

        self.state = state or StateStore(self.data_root / "state" / "chat_history.sqlite3")
        cfg_dir = self.config_path.parent if self.config_path is not None else None
        clean_cons_model = (consolidator_model or "").strip() or None
        need_models_cfg = cfg_dir is not None and (
            llm is None or (consolidator_llm is None and clean_cons_model is not None)
        )
        models_cfg = self.tools_config.load_models_config(cfg_dir) if need_models_cfg else None

        self.llm = (
            llm
            if llm is not None
            else self.tools_config.build_llm(cfg_dir, model_name=model_name, models_cfg=models_cfg)
        )
        self._llm_lock = threading.RLock()
        self._prompt_lock = threading.Lock()
        self._session_prompts: dict[str, tuple[str, str]] = {}
        self._last_archive_day = ""
        self._chat_locks_guard = threading.Lock()
        self._chat_locks: dict[str, tuple[threading.Lock, int]] = {}
        self.cache_monitor = CacheMonitor()
        self.observers: tuple[ObserverProtocol, ...] = (
            tuple(observers) if observers is not None else tuple(self.tools_config.build_observers())
        )
        self._consolidator_model = clean_cons_model
        self._custom_consolidator_llm = consolidator_llm is not None
        if consolidator_llm is not None:
            resolved_consolidator_llm = consolidator_llm
        elif llm is not None and clean_cons_model is None:
            resolved_consolidator_llm = self.llm
        else:
            resolved_consolidator_llm = self.tools_config.build_consolidator_llm(
                cfg_dir,
                fallback_llm=self.llm,
                model_name=clean_cons_model,
                models_cfg=models_cfg,
            )
        self.consolidator = MemoryConsolidator(
            self.state,
            self.memory,
            resolved_consolidator_llm,
            policy=consolidation_policy,
            today_fn=self._today,
        )
        self.background_consolidator = BackgroundConsolidator(
            self.consolidator,
            on_event=self._emit_event,
        )
        self._tool_registry = self._build_base_registry()

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
        return self.tools_config.model_presets()

    # 返回当前模型预设与已配置模型的脱敏视图（绝不包含明文 api_key）
    def models_overview(self) -> dict[str, object]:
        cfg = self.tools_config.load_models_config()
        return {
            "presets": self.model_presets(),
            "configured": cfg.redacted_dict() if cfg is not None else None,
            "llm_configured": not isinstance(self.llm, NoopLLMClient),
        }

    # 按最新 models.dsl / models.json 原子热切换当前运行中的主模型与后台沉淀 LLMClient（单次编译，失败零污染）
    def reload_llm(
        self,
        model_name: str | None = None,
        *,
        consolidator_model: str | None = None,
        models_cfg: ModelsConfig | None = None,
    ) -> None:
        with self._llm_lock:
            next_cons_model = (
                self._consolidator_model
                if consolidator_model is None
                else (consolidator_model.strip() or None)
            )
            cfg_dir = self.config_path.parent if self.config_path is not None else None
            resolved_cfg = (
                models_cfg
                if models_cfg is not None
                else self.tools_config.load_models_config(cfg_dir)
            )
            new_llm = self.tools_config.build_llm(
                cfg_dir,
                model_name=model_name,
                models_cfg=resolved_cfg,
            )
            new_cons_llm = (
                self.consolidator.llm
                if self._custom_consolidator_llm
                else self.tools_config.build_consolidator_llm(
                    cfg_dir,
                    fallback_llm=new_llm,
                    model_name=next_cons_model,
                    models_cfg=resolved_cfg,
                )
            )
            self._consolidator_model = next_cons_model
            self.llm = new_llm
            if not self._custom_consolidator_llm:
                self.consolidator.llm = new_cons_llm

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
        if self.config_path is None:
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
        self.reload_llm(
            model_name=models_cfg.default,
            consolidator_model="",
            models_cfg=models_cfg,
        )
        return models_cfg.redacted_dict()

    # 等待所有在途后台沉淀任务完成（供退出收尾或测试调用）
    def wait_for_consolidation(self, timeout: float = 15.0) -> None:
        self.background_consolidator.wait(timeout=timeout)

    # 先等待后台沉淀收尾，再关闭已加载插件生命周期，最后关闭底层状态存储资源
    def close(self) -> None:
        try:
            self.background_consolidator.wait(timeout=15.0)
            self.tools_config.plugin_manager.terminate_all()
        finally:
            self.state.close()

    # 返回当前逻辑日期（默认北京时间，评测时可通过 today_fn 注入历史日期）
    def _today(self) -> date:
        if self._today_fn is not None:
            return self._today_fn()
        return today_shanghai()

    # 在应用启动时一次性构建受白名单管控的常驻工具注册表（跨轮复用，动态上下文走执行期 tool_context）
    def _build_base_registry(self) -> ToolRegistry:
        return self.tools_config.build_registry(
            providers=[MemoryToolsetProvider(self.memory)],
            data_root=self.data_root,
            tool_context={"history_db_path": self.state.db_path},
        )

    # 返回应用常驻白名单工具注册表
    @property
    def tools(self) -> ToolRegistry:
        return self._tool_registry

    # 拼装系统提示词：静态指令与规则在前、目录索引与日期在后，最大化前缀 KV Cache 复用
    def build_system_prompt(self, registry: ToolRegistry) -> str:
        return format_system_prompt(self.memory, tool_names=registry.names, today=self._today())

    # 获取指定会话冻结的系统提示词：按自然日与会话键复用首次快照（有界 LRU），跨日自动失效重算
    def get_session_system_prompt(self, chat_id: str, registry: ToolRegistry) -> str:
        today_str = self._today().isoformat()
        with self._prompt_lock:
            cached = self._session_prompts.pop(chat_id, None)
            if cached is not None and cached[0] == today_str:
                prompt = cached[1]
            else:
                prompt = self.build_system_prompt(registry)
            if len(self._session_prompts) >= MAX_SESSION_PROMPTS:
                self._session_prompts.pop(next(iter(self._session_prompts)))
            self._session_prompts[chat_id] = (today_str, prompt)
            return prompt

    # 主动重置指定会话（或全部会话）冻结的系统提示词，用于后台沉淀或手动改文件后刷新
    def reset_session_prompt(self, chat_id: str | None = None) -> None:
        with self._prompt_lock:
            if chat_id is None:
                self._session_prompts.clear()
            else:
                self._session_prompts.pop(chat_id, None)

    # 为指定 chat_id 获取有界会话级互斥锁，确保同会话并发请求严格串行并在空闲或中断时安全清理引用计数
    @contextmanager
    def _chat_turn_lock(self, chat_id: str) -> Iterator[None]:
        with self._chat_locks_guard:
            entry = self._chat_locks.pop(chat_id, None)
            lock, refs = entry if entry is not None else (threading.Lock(), 0)
            self._chat_locks[chat_id] = (lock, refs + 1)
            if len(self._chat_locks) > MAX_SESSION_LOCKS:
                for idle_id in [k for k, (_, r) in self._chat_locks.items() if r == 0]:
                    if len(self._chat_locks) <= MAX_SESSION_LOCKS:
                        break
                    self._chat_locks.pop(idle_id, None)
        acquired = False
        try:
            acquired = lock.acquire()
            yield
        finally:
            if acquired:
                lock.release()
            with self._chat_locks_guard:
                current = self._chat_locks.get(chat_id)
                if current is not None and current[0] is lock:
                    new_refs = max(0, current[1] - 1)
                    if new_refs == 0 and len(self._chat_locks) > MAX_SESSION_LOCKS:
                        self._chat_locks.pop(chat_id, None)
                    else:
                        self._chat_locks[chat_id] = (lock, new_refs)

    # 将持久化历史转换为标准追加式消息列表，安全合并异常连续同角色消息，保持前缀字节级稳定以命中 KV Cache
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
            _append_alternating_message(result, {"role": role, "content": content})
        return result

    # 组装单条稳定 system 消息（按需尾部拼入已压缩前情摘要）与严格交替的历史及当前提问
    def _assemble_context_messages(
        self,
        system_prompt: str,
        summary_text: str,
        history: list[dict[str, object]],
        clean_text: str,
    ) -> list[dict[str, object]]:
        effective_system = (
            f"{system_prompt}\n\n【本会话已压缩前情摘要】\n{summary_text}"
            if summary_text
            else system_prompt
        )
        messages: list[dict[str, object]] = [{"role": "system", "content": effective_system}]
        messages.extend(self._build_history_messages(history))
        _append_alternating_message(messages, {"role": "user", "content": clean_text})
        return messages

    # 每个自然日首次对话时先执行过期归档，归档成功后再更新日期游标并清理过时会话冻结提示词
    def _maybe_archive_expired(self) -> None:
        today = self._today()
        today_str = today.isoformat()
        with self._prompt_lock:
            if self._last_archive_day == today_str:
                return
        self.memory.archive_expired_memories(today=today)
        with self._prompt_lock:
            if self._last_archive_day != today_str:
                self._last_archive_day = today_str
                stale = [
                    cid
                    for cid, (day_key, _) in self._session_prompts.items()
                    if day_key != today_str
                ]
                for cid in stale:
                    self._session_prompts.pop(cid, None)

    # 处理一轮流式对话（单一主流程）：先同步校验输入与模型就绪状态，再返回事件帧生成器
    def stream_chat(
        self,
        text: str,
        chat_id: str = "default",
        *,
        on_event: Callable[[AgentEvent], None] | None = None,
    ) -> Generator[dict[str, object], None, None]:
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("消息内容不能为空。")
        with self._llm_lock:
            active_llm = self.llm
        if isinstance(active_llm, NoopLLMClient):
            raise LLMNotConfigured(
                "未配置真实大模型（LLM_MODE!=on）。请运行 `uv run python -m personal_assistant setup` 或配置 API Key。"
            )
        return self._run_stream_turn(
            clean_text,
            chat_id,
            on_event=on_event,
        )

    # 内部生成器：同会话加锁串行执行，成功后再原子成对落库 user+assistant 与绑定事实标记，异常/中断零脏写
    def _run_stream_turn(
        self,
        clean_text: str,
        chat_id: str,
        *,
        on_event: Callable[[AgentEvent], None] | None,
    ) -> Generator[dict[str, object], None, None]:
        with self._chat_turn_lock(chat_id):
            self._maybe_archive_expired()
            compaction, active_all = self.state.get_context_snapshot(chat_id)
            history = _select_adaptive_history(active_all)
            before_msg_id = self.state.max_message_id() + 1
            turn_tag = uuid.uuid4().hex[:12]
            fact_used = False

            turn_persisted = False
            try:
                registry = self._tool_registry
                turn_ctx: dict[str, object] = {
                    "user_text": clean_text,
                    "before_msg_id": before_msg_id,
                    "turn_tag": turn_tag,
                    "untrusted_tool_seen": False,
                    "today": self._today(),
                    "history_db_path": self.state.db_path,
                }
                system_prompt = self.get_session_system_prompt(chat_id, registry)
                summary_text = str(compaction.get("summary") or "").strip()
                messages = self._assemble_context_messages(
                    system_prompt, summary_text, history, clean_text
                )

                traces: list[dict[str, object]] = []
                thinking_parts: list[str] = []
                reply = ""
                start_idx = self.cache_monitor.snapshot_index()
                with self._llm_lock:
                    active_llm = self.llm
                tracked_llm = self.cache_monitor.wrap(active_llm, session_key=chat_id)
                guarded_llm = _GuardedTurnLLM(
                    tracked_llm,
                    max_steps=self.max_turn_steps,
                    max_tokens=self.max_turn_tokens,
                    max_seconds=self.max_turn_seconds,
                    max_repeated_tool_calls=self.max_repeated_tool_calls,
                    clock=self._clock,
                )
                for event in iter_react(
                    guarded_llm,
                    messages,
                    registry,
                    convert_to_llm=None,
                    tool_context=turn_ctx,
                ):
                    if isinstance(event, ToolStart):
                        meta = registry.metadata(event.tool)
                        if meta is not None and meta.risk == "network":
                            turn_ctx["untrusted_tool_seen"] = True
                        if event.tool == "remember_fact":
                            fact_used = True
                        guarded_llm.observe_tool_call(event.tool, event.arguments)
                    self._emit_event(chat_id, event)
                    if on_event is not None:
                        on_event(event)
                    if isinstance(event, ThinkingDelta) and event.text:
                        thinking_parts.append(event.text)
                        yield {"type": "thinking_delta", "text": event.text}
                    elif isinstance(event, ToolStart):
                        yield {
                            "type": "tool_start",
                            "tool": event.tool,
                            "arguments": event.arguments,
                            "call_id": event.call_id,
                        }
                    elif isinstance(event, TextDelta):
                        yield {"type": "text_delta", "text": event.text}
                    elif isinstance(event, ToolStep):
                        step_dict: dict[str, object] = {
                            "tool": event.tool,
                            "arguments": event.arguments,
                            "observation": event.observation,
                            "call_id": event.call_id,
                            "duration_ms": event.duration_ms,
                        }
                        traces.append(step_dict)
                        yield {"type": "tool_step", **step_dict}
                    elif isinstance(event, TurnEnd):
                        reply = event.reply

                user_msg_id, assistant_msg_id = self.state.append_turn(chat_id, clean_text, reply)
                turn_persisted = True
                if fact_used:
                    self.memory.finalize_turn_fact(turn_tag, message_id=user_msg_id)
            finally:
                if not turn_persisted and fact_used:
                    self.memory.finalize_turn_fact(turn_tag, message_id=None)

            scheduled = (
                self.background_consolidator.maybe_schedule(
                    chat_id,
                    on_complete=lambda res: self.reset_session_prompt(res.chat_id),
                )
                if self.auto_consolidate
                else False
            )
            done_payload: dict[str, object] = {
                "type": "done",
                "reply": reply,
                "thinking": "".join(thinking_parts),
                "chat_id": chat_id,
                "message_id": assistant_msg_id,
                "tool_calls": traces,
                "cache": self.cache_monitor.summary(since=start_idx, session_key=chat_id),
                "consolidation_scheduled": scheduled,
            }
            if guarded_llm.breach is not None:
                done_payload["budget_exceeded"] = guarded_llm.breach.reason
            telemetry = self._collect_telemetry(session_key=chat_id)
            if telemetry:
                done_payload["telemetry"] = telemetry
        yield done_payload

    # 处理一轮完整同步对话：直接消费 stream_chat 单一实现并确保异常时确定性关闭生成器
    def chat(
        self,
        text: str,
        chat_id: str = "default",
        *,
        on_event: Callable[[AgentEvent], None] | None = None,
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        with closing(
            self.stream_chat(
                text,
                chat_id=chat_id,
                on_event=on_event,
            )
        ) as frames:
            for frame in frames:
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
        return {
            "status": "ok",
            "llm_configured": not isinstance(self.llm, NoopLLMClient),
            "memory_files": self.memory.count_files(),
            "messages": self.state.count_messages(),
            "tools": self._tool_registry.names,
            "plugins": list(self.tools_config.plugins),
            "retriever": self.memory.retriever.name,
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
    consolidator_model: str | None = None,
    auto_consolidate: bool = True,
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
        consolidator_model=consolidator_model,
        auto_consolidate=auto_consolidate,
    )
