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
from datetime import date, datetime
from pathlib import Path

from personal_assistant.agent.plugins import ObserverProtocol, RuntimeEvent, TaskEvent
from personal_assistant.agent.react import (
    AgentEvent,
    ToolStart,
    ToolStep,
    TurnEnd,
    iter_react,
)
from personal_assistant.agent.tools import ToolRegistry
from personal_assistant.ai import (
    AssistantMessage,
    AssistantMessageEvent,
    CacheMonitor,
    Context,
    LLMNotConfigured,
    Message,
    ModelInvoker,
    ModelsConfig,
    NoopModelClient,
    StreamDone,
    StreamError,
    StreamOptions,
    StreamStart,
    TextBlock,
    TextDelta,
    TextEnd,
    TextStart,
    ThinkingBlock,
    ThinkingDelta,
    TokenUsage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from personal_assistant.assistant.builtin_tools import MemoryToolsetProvider
from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.assistant.common import (
    SHANGHAI_TZ,
    now_shanghai,
    parse_iso_date,
    require_positive_int,
    today_shanghai,
)
from personal_assistant.assistant.consolidator import (
    BackgroundConsolidator,
    ConsolidationPolicy,
    MemoryConsolidator,
)
from personal_assistant.assistant.db import StateStore
from personal_assistant.assistant.files_policy import FilesPolicy
from personal_assistant.assistant.memory import MemoryStore
from personal_assistant.assistant.prompts import format_system_prompt, format_user_envelope

MAX_HISTORY_MESSAGES = 20
MAX_ADAPTIVE_HISTORY_MESSAGES = 60
MAX_ADAPTIVE_HISTORY_CHARS = 24_000
MAX_HISTORY_ITEM_CHARS = 2000
MAX_SESSION_PROMPTS = 256
MAX_SESSION_LOCKS = 256
DEFAULT_PROMPT_CACHE_TTL_SECONDS = 300.0
DEFAULT_FOLD_TRIGGER_CHARS = 48_000
DEFAULT_FOLD_KEEP_TAIL_MESSAGES = 4
DEFAULT_FOLD_MIN_RESULT_CHARS = 800
DEFAULT_FOLD_PREVIEW_CHARS = 240


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


# 判断消息字典的 id 字段是否为有效整数（排除布尔值）
def _valid_msg_id(raw_id: object) -> int | None:
    if isinstance(raw_id, int) and not isinstance(raw_id, bool):
        return raw_id
    return None


# 粘性边界按块裁剪活跃历史：未超上限保持队头纹丝不动，仅在超限时整块切至目标水位并推进 floor_id 以守护 KV Cache
def _select_chunked_history(
    active_all: list[dict[str, object]],
    *,
    floor_id: int = 0,
    base_messages: int = MAX_HISTORY_MESSAGES,
    max_messages: int = MAX_ADAPTIVE_HISTORY_MESSAGES,
    max_chars: int = MAX_ADAPTIVE_HISTORY_CHARS,
) -> tuple[list[dict[str, object]], int]:
    if floor_id > 0:
        eligible = [
            msg
            for msg in active_all
            if (mid := _valid_msg_id(msg.get("id"))) is None or mid > floor_id
        ]
    else:
        eligible = list(active_all)

    total_chars = sum(min(len(str(msg.get("content") or "")), MAX_HISTORY_ITEM_CHARS) for msg in eligible)
    if len(eligible) <= max_messages and total_chars <= max_chars:
        while len(eligible) > 1 and eligible[0].get("role") == "assistant":
            eligible.pop(0)
        return eligible, floor_id

    target_msgs = min(max(2, base_messages), max_messages)
    target_chars = min(max_chars, max(MAX_HISTORY_ITEM_CHARS * 2, max_chars // 2))
    selected_rev: list[dict[str, object]] = []
    used_chars = 0
    for msg in reversed(eligible):
        item_len = min(len(str(msg.get("content") or "")), MAX_HISTORY_ITEM_CHARS)
        if selected_rev and (len(selected_rev) >= target_msgs or used_chars + item_len > target_chars):
            break
        selected_rev.append(msg)
        used_chars += item_len
    while len(selected_rev) > 1 and selected_rev[-1].get("role") == "assistant":
        selected_rev.pop()
    selected_rev.reverse()

    new_floor = floor_id
    if selected_rev:
        first_id = _valid_msg_id(selected_rev[0].get("id"))
        if first_id is not None:
            new_floor = max(floor_id, first_id - 1)
    return selected_rev, new_floor


# 从 SQLite 消息的 created_at 字段解析北京时间自然日（兼容带时区 datetime / ISO 与纯日期字符串）
def _message_shanghai_date(raw_created_at: object) -> date | None:
    if isinstance(raw_created_at, datetime):
        return raw_created_at.astimezone(SHANGHAI_TZ).date() if raw_created_at.tzinfo is not None else raw_created_at.date()
    if isinstance(raw_created_at, str):
        text = raw_created_at.strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return dt.astimezone(SHANGHAI_TZ).date() if dt.tzinfo is not None else dt.date()
        except ValueError:
            return parse_iso_date(text)
    return parse_iso_date(raw_created_at)


# 计算单条中立消息的大致字符长度，供轮内高水位投影阈值判断
def _message_char_len(msg: Message) -> int:
    if isinstance(msg, UserMessage):
        return len(msg.content)
    if isinstance(msg, ToolResultMessage):
        return len(msg.content)
    total = 0
    for block in msg.blocks:
        if isinstance(block, TextBlock):
            total += len(block.text)
        elif isinstance(block, ThinkingBlock):
            total += len(block.text) + len(block.signature)
        elif isinstance(block, ToolCall):
            total += len(block.name) + len(block.arguments)
    return total


# 轮内粘性高水位非破坏性上下文投影器（接入 react.iter_react 的 convert_to_llm 钩子）：带动态滞回水位防抖，绝不放大原输出
class _TurnContextProjector:
    # 初始化触发高水位字符阈值、动态下次触发水位、尾部原文保护条数与单条大输出折叠阈值
    def __init__(
        self,
        *,
        trigger_chars: int = DEFAULT_FOLD_TRIGGER_CHARS,
        keep_tail_messages: int = DEFAULT_FOLD_KEEP_TAIL_MESSAGES,
        min_result_chars: int = DEFAULT_FOLD_MIN_RESULT_CHARS,
        preview_chars: int = DEFAULT_FOLD_PREVIEW_CHARS,
    ) -> None:
        self._trigger_chars = trigger_chars
        self._next_trigger_chars = trigger_chars
        self._keep_tail_messages = keep_tail_messages
        self._min_result_chars = min_result_chars
        self._preview_chars = preview_chars
        self._folded_until = 0

    # 将单条超长 ToolResultMessage 确定性地投影为紧凑卡片（仅提取有界 TOAST 回读指针，并严格保证折叠后更短）
    def _fold_tool_result(self, msg: ToolResultMessage) -> ToolResultMessage:
        if len(msg.content) <= self._min_result_chars:
            return msg
        pointer_line = ""
        for line in msg.content.splitlines():
            if 'read_markdown(path="' in line:
                candidate = line.strip()
                if len(candidate) <= 320:
                    pointer_line = candidate
                break
        preview = " ".join(msg.content[: self._preview_chars].split())
        parts = [f"[已折叠工具输出：{msg.tool_name}，原长 {len(msg.content)} 字符，原文完整保留]"]
        if preview:
            parts.append(f"{preview}…")
        if pointer_line:
            parts.append(pointer_line)
        folded_content = "\n".join(parts)
        if len(folded_content) >= len(msg.content):
            return msg
        return ToolResultMessage(
            tool_call_id=msg.tool_call_id,
            tool_name=msg.tool_name,
            content=folded_content,
            is_error=msg.is_error,
        )

    # 按粘性边界与动态滞回水位投影当前轮 Context：不改写原始 history，无实际折叠不推边界，折叠后抬高下次触发水位防每步抖动
    def __call__(self, context: Context) -> Context:
        msgs = list(context.messages)
        mutated = False
        locked = min(self._folded_until, len(msgs))
        for idx in range(locked):
            item = msgs[idx]
            if isinstance(item, ToolResultMessage):
                folded = self._fold_tool_result(item)
                if folded is not item:
                    msgs[idx] = folded
                    mutated = True

        total_chars = sum(_message_char_len(m) for m in msgs)
        if total_chars > self._next_trigger_chars:
            cutoff = max(0, len(msgs) - self._keep_tail_messages)
            if cutoff > self._folded_until:
                newly_folded = False
                for idx in range(self._folded_until, cutoff):
                    item = msgs[idx]
                    if isinstance(item, ToolResultMessage):
                        folded = self._fold_tool_result(item)
                        if folded is not item:
                            msgs[idx] = folded
                            mutated = True
                            newly_folded = True
                if newly_folded:
                    self._folded_until = cutoff
                    post_fold_chars = sum(_message_char_len(m) for m in msgs)
                    hysteresis = max(1, self._trigger_chars // 2)
                    self._next_trigger_chars = max(self._trigger_chars, post_fold_chars + hysteresis)

        if not mutated:
            return context
        return Context(
            system_prompt=context.system_prompt,
            messages=tuple(msgs),
            tools=context.tools,
        )


# 外层回合预算护栏包装器（实现 ModelInvoker）：在零改动 Layer 2 微内核的前提下执行步数/Token/墙钟/重复/截断熔断
class _GuardedTurnLLM:
    # 绑定底层 ModelInvoker、预算上限与单调时钟
    def __init__(
        self,
        inner: ModelInvoker,
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

    # 进入新一轮模型推理前核查预算护栏，命中任一阈值即返回 TurnBudgetExceeded
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

    # 构造不带工具调用的优雅降级收束 AssistantMessage，使 iter_react 自然退出并释放会话锁
    @staticmethod
    def _fallback_message(
        breach: TurnBudgetExceeded,
        *,
        usage: TokenUsage | None = None,
    ) -> AssistantMessage:
        return AssistantMessage.from_text(
            (
                f"抱歉，本轮工具调用达到安全预算上限（原因：{breach.reason}），"
                "已自动停止继续重试。请缩小问题范围或重试。"
            ),
            stop_reason="stop",
            usage=usage,
        )

    # 累加单次响应 TokenUsage 并同步 last_usage
    def _record_usage(self, usage: object) -> None:
        if isinstance(usage, TokenUsage):
            self.total_tokens += usage.prompt_tokens + usage.completion_tokens
            self.last_usage = usage
        else:
            self.last_usage = None

    # 流式调用前执行预算核查；检测到 length 截断且含破损工具 JSON 时立即就地熔断收束
    def stream(
        self,
        context: Context,
        options: StreamOptions | None = None,
    ) -> Iterator[AssistantMessageEvent]:
        breach = self._check_budget()
        if breach is not None:
            self.breach = breach
            self.last_usage = None
            fallback_msg = self._fallback_message(breach)
            fallback_text = fallback_msg.text
            yield StreamStart(partial=AssistantMessage())
            yield TextStart(content_index=0, partial=AssistantMessage())
            yield TextDelta(content_index=0, text=fallback_text, partial=fallback_msg)
            yield TextEnd(content_index=0, text=fallback_text, partial=fallback_msg)
            yield StreamDone(reason="stop", message=fallback_msg, usage=None, ttft_ms=None, latency_ms=0.0)
            return

        self.steps += 1
        self.last_usage = None
        raw_iter = self._inner.stream(context, options)
        try:
            for event in raw_iter:
                if isinstance(event, StreamError):
                    self.last_usage = None
                    yield event
                    return
                if isinstance(event, StreamDone):
                    usage = event.usage if isinstance(event.usage, TokenUsage) else self._inner.last_usage
                    self._record_usage(usage)
                    if event.reason == "length" and event.message.tool_calls:
                        self.truncated_tool_call = True
                        trunc_breach = TurnBudgetExceeded(
                            "truncated_tool_calls",
                            "模型输出因长度上限截断导致工具参数 JSON 不完整",
                        )
                        self.breach = trunc_breach
                        fallback_msg = self._fallback_message(trunc_breach, usage=usage)
                        fallback_text = fallback_msg.text
                        c_idx = len(event.message.blocks)
                        yield TextStart(content_index=c_idx, partial=event.message)
                        yield TextDelta(content_index=c_idx, text=fallback_text, partial=fallback_msg)
                        yield TextEnd(content_index=c_idx, text=fallback_text, partial=fallback_msg)
                        yield StreamDone(
                            reason="stop",
                            message=fallback_msg,
                            usage=usage,
                            ttft_ms=event.ttft_ms,
                            latency_ms=event.latency_ms,
                        )
                        return
                yield event
        finally:
            close_fn = getattr(raw_iter, "close", None)
            if callable(close_fn):
                close_fn()

    # 同步调用入口：确定性消费 self.stream() 并返回最终 AssistantMessage
    def complete(
        self,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessage:
        event_iter = self.stream(context, options)
        try:
            for event in event_iter:
                if isinstance(event, StreamError):
                    raise RuntimeError(event.error_message)
                if isinstance(event, StreamDone):
                    return event.message
        finally:
            close_fn = getattr(event_iter, "close", None)
            if callable(close_fn):
                close_fn()
        raise RuntimeError("模型流式响应异常终止，未收到 StreamDone 事件。")


# 向中立消息列表追加对话消息，遇到连续同角色纯文本消息时安全合并以防严格交替协议报错
def _append_alternating_message(
    messages: list[Message],
    message: Message,
) -> None:
    if not messages:
        messages.append(message)
        return
    prev = messages[-1]
    if isinstance(prev, UserMessage) and isinstance(message, UserMessage):
        merged = (
            f"{prev.content}\n\n{message.content}"
            if (prev.content and message.content)
            else (prev.content or message.content)
        )
        messages[-1] = UserMessage(content=merged)
        return
    if (
        isinstance(prev, AssistantMessage)
        and isinstance(message, AssistantMessage)
        and not prev.tool_calls
        and not message.tool_calls
        and not prev.thinking
        and not message.thinking
    ):
        prev_text = prev.text
        curr_text = message.text
        merged = f"{prev_text}\n\n{curr_text}" if (prev_text and curr_text) else (prev_text or curr_text)
        messages[-1] = AssistantMessage.from_text(merged)
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
        tools_config: ToolsConfig | None = None,
        llm: ModelInvoker | None = None,
        memory: MemoryStore | None = None,
        state: StateStore | None = None,
        consolidation_policy: ConsolidationPolicy | None = None,
        consolidator_llm: ModelInvoker | None = None,
        auto_consolidate: bool = True,
        today_fn: Callable[[], date] | None = None,
        model_name: str | None = None,
        consolidator_model: str | None = None,
        observers: Iterable[ObserverProtocol] | None = None,
        max_turn_steps: int = 24,
        max_turn_tokens: int = 250_000,
        max_turn_seconds: float = 180.0,
        max_repeated_tool_calls: int = 3,
        prompt_cache_ttl_seconds: float = DEFAULT_PROMPT_CACHE_TTL_SECONDS,
        max_turn_context_chars: int = DEFAULT_FOLD_TRIGGER_CHARS,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.max_turn_steps = require_positive_int(max_turn_steps, "max_turn_steps")
        self.max_turn_tokens = require_positive_int(max_turn_tokens, "max_turn_tokens")
        self.max_turn_seconds = _validate_non_negative_float(max_turn_seconds, "max_turn_seconds")
        self.max_repeated_tool_calls = require_positive_int(
            max_repeated_tool_calls, "max_repeated_tool_calls"
        )
        self.prompt_cache_ttl_seconds = _validate_non_negative_float(
            prompt_cache_ttl_seconds, "prompt_cache_ttl_seconds"
        )
        self.max_turn_context_chars = require_positive_int(
            max_turn_context_chars, "max_turn_context_chars"
        )
        self._clock: Callable[[], float] = clock or time.monotonic
        self.auto_consolidate = bool(auto_consolidate)
        self.data_root = Path(data_root).resolve()
        self.data_root.mkdir(parents=True, exist_ok=True)
        self._today_fn = today_fn

        self._owns_tools_config = tools_config is None
        if tools_config is not None:
            self.tools_config = tools_config
            self.config_path = (
                config_path.resolve() if config_path is not None else tools_config.config_path
            )
            self.project_root = (
                project_root.resolve() if project_root is not None else tools_config.project_root
            )
        else:
            self.config_path = config_path.resolve() if config_path is not None else None
            self.project_root = (
                project_root.resolve()
                if project_root is not None
                else (self.config_path.parent.parent if self.config_path else self.data_root.parent)
            )
            self.tools_config = (
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
        self._session_prompts: dict[str, tuple[str, float]] = {}
        self._history_trim_floors: dict[str, int] = {}
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
            "llm_configured": not isinstance(self.llm, NoopModelClient),
        }

    # 按最新 models.dsl / models.json 原子热切换当前运行中的主模型与后台沉淀 ModelInvoker（单次编译，失败零污染）
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
        with self._llm_lock:
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

    # 先等待后台沉淀收尾，再按所有权关闭已加载插件生命周期，最后关闭底层状态存储资源
    def close(self) -> None:
        try:
            self.background_consolidator.wait(timeout=15.0)
            if self._owns_tools_config:
                self.tools_config.plugin_manager.terminate_all()
        finally:
            self.state.close()

    # 返回当前逻辑日期（默认北京时间，评测时可通过 today_fn 注入历史日期）
    def _today(self) -> date:
        if self._today_fn is not None:
            return self._today_fn()
        return today_shanghai()

    # 单次采样本轮统一的逻辑自然日与持久化 ISO 时间戳，确保跨零点瞬间仍逐字节一致
    def _turn_clock_stamp(self) -> tuple[date, str]:
        if self._today_fn is not None:
            day = self._today_fn()
            return day, f"{day.isoformat()}T12:00:00+08:00"
        now = now_shanghai()
        return now.date(), now.isoformat()

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

    # 拼装纯静态/低频系统提示词（不含逐日日期与动态摘要），最大化前缀 KV Cache 复用
    def build_system_prompt(self, registry: ToolRegistry) -> str:
        return format_system_prompt(self.memory, tool_names=registry.names)

    # 获取指定会话冻结的系统提示词：热聊期（空闲 < TTL）死锁复用，空闲超时后零代价刷新最新记忆索引
    def get_session_system_prompt(self, chat_id: str, registry: ToolRegistry) -> str:
        now = self._clock()
        with self._prompt_lock:
            cached = self._session_prompts.pop(chat_id, None)
            if (
                cached is not None
                and self.prompt_cache_ttl_seconds > 0
                and (now - cached[1]) < self.prompt_cache_ttl_seconds
            ):
                prompt = cached[0]
            else:
                prompt = self.build_system_prompt(registry)
            if len(self._session_prompts) >= MAX_SESSION_PROMPTS:
                self._session_prompts.pop(next(iter(self._session_prompts)))
            self._session_prompts[chat_id] = (prompt, now)
            return prompt

    # 主动重置指定会话（或全部会话）冻结的系统提示词与历史切块边界，用于沉淀改写索引或手动刷新
    def reset_session_prompt(self, chat_id: str | None = None) -> None:
        with self._prompt_lock:
            if chat_id is None:
                self._session_prompts.clear()
                self._history_trim_floors.clear()
            else:
                self._session_prompts.pop(chat_id, None)
                self._history_trim_floors.pop(chat_id, None)

    # 仅重置指定会话的历史切块游标（例如后台沉淀推进了压缩边界但未改写 MEMORY.md 时保留热 system_prompt）
    def _clear_history_trim_floor(self, chat_id: str) -> None:
        with self._prompt_lock:
            self._history_trim_floors.pop(chat_id, None)

    # 按会话粘性边界裁剪活跃历史并按真 LRU 刷新命中会话热度（防止热会话在上限淘汰时被提前逐出）
    def _select_session_history(
        self,
        chat_id: str,
        active_all: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        with self._prompt_lock:
            existing = self._history_trim_floors.pop(chat_id, None)
            if existing is not None:
                self._history_trim_floors[chat_id] = existing
                floor_id = existing
            else:
                floor_id = 0
        selected, new_floor = _select_chunked_history(active_all, floor_id=floor_id)
        if new_floor != floor_id:
            with self._prompt_lock:
                self._history_trim_floors.pop(chat_id, None)
                if len(self._history_trim_floors) >= MAX_SESSION_PROMPTS:
                    self._history_trim_floors.pop(next(iter(self._history_trim_floors)))
                self._history_trim_floors[chat_id] = new_floor
        return selected

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

    # 将持久化历史转换为中立追加式消息列表：按记录日期还原用户信封并合并异常连续同角色，保持前缀字节级稳定
    @staticmethod
    def _build_history_messages(history: list[dict[str, object]]) -> list[Message]:
        result: list[Message] = []
        for msg in history:
            role = str(msg.get("role") or "")
            if role not in {"user", "assistant"}:
                continue
            content = str(msg.get("content") or "")
            if len(content) > MAX_HISTORY_ITEM_CHARS:
                content = content[:MAX_HISTORY_ITEM_CHARS] + "…"
            if role == "user":
                msg_day = _message_shanghai_date(msg.get("created_at"))
                rendered = format_user_envelope(content, today=msg_day) if msg_day is not None else content
                item: Message = UserMessage(content=rendered)
            else:
                item = AssistantMessage.from_text(content)
            _append_alternating_message(result, item)
        return result

    # 组装中立 Context：纯净静态 system_prompt + 位于 messages 头部的前情摘要 + 稳定历史 + 带日期信封的当前提问
    def _assemble_context(
        self,
        system_prompt: str,
        summary_text: str,
        history: list[dict[str, object]],
        clean_text: str,
        registry: ToolRegistry,
        *,
        today: date | None = None,
    ) -> Context:
        messages: list[Message] = []
        if summary_text:
            messages.append(UserMessage(content=f"【本会话已压缩前情摘要】\n{summary_text}"))
        for msg in self._build_history_messages(history):
            _append_alternating_message(messages, msg)
        envelope = format_user_envelope(clean_text, today=today or self._today())
        _append_alternating_message(messages, UserMessage(content=envelope))
        return Context(
            system_prompt=system_prompt,
            messages=tuple(messages),
            tools=registry.schemas(),
        )

    # 每个自然日首次对话时执行过期归档；仅在实际归档改动了记忆文件时才刷新会话提示词，无过期时零击穿缓存
    def _maybe_archive_expired(self, *, today: date | None = None) -> None:
        ref_day = today or self._today()
        today_str = ref_day.isoformat()
        with self._prompt_lock:
            if self._last_archive_day == today_str:
                return
        archived = self.memory.archive_expired_memories(today=ref_day)
        with self._prompt_lock:
            if archived:
                self._session_prompts.clear()
            if self._last_archive_day != today_str:
                self._last_archive_day = today_str

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
        if isinstance(active_llm, NoopModelClient):
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
            turn_day, turn_created_at = self._turn_clock_stamp()
            self._maybe_archive_expired(today=turn_day)
            compaction, active_all = self.state.get_context_snapshot(chat_id)
            history = self._select_session_history(chat_id, active_all)
            before_msg_id = self.state.max_message_id() + 1
            turn_tag = uuid.uuid4().hex[:12]
            fact_used = False

            turn_persisted = False
            turn_ended = False
            try:
                registry = self._tool_registry
                turn_ctx: dict[str, object] = {
                    "user_text": clean_text,
                    "before_msg_id": before_msg_id,
                    "turn_tag": turn_tag,
                    "untrusted_tool_seen": False,
                    "today": turn_day,
                    "history_db_path": self.state.db_path,
                }
                system_prompt = self.get_session_system_prompt(chat_id, registry)
                summary_text = str(compaction.get("summary") or "").strip()
                context = self._assemble_context(
                    system_prompt,
                    summary_text,
                    history,
                    clean_text,
                    registry,
                    today=turn_day,
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
                projector = _TurnContextProjector(trigger_chars=self.max_turn_context_chars)
                with closing(
                    iter_react(
                        guarded_llm,
                        context,
                        registry,
                        convert_to_llm=projector,
                        tool_context=turn_ctx,
                        options=StreamOptions(session_id=chat_id),
                    )
                ) as react_events:
                    for event in react_events:
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
                        if isinstance(event, StreamError):
                            raise RuntimeError(event.error_message)
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
                        elif isinstance(event, TextDelta) and event.text:
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
                            turn_ended = True
                            reply = event.reply

                user_msg_id, assistant_msg_id = self.state.append_turn(
                    chat_id,
                    clean_text,
                    reply,
                    created_at=turn_created_at,
                )
                turn_persisted = True
                if fact_used:
                    self.memory.finalize_turn_fact(turn_tag, message_id=user_msg_id)
            finally:
                if not turn_ended:
                    self._emit_event(
                        chat_id,
                        TaskEvent(kind="turn_aborted", duration_ms=0.0, error="turn_interrupted"),
                    )
                if not turn_persisted and fact_used:
                    self.memory.finalize_turn_fact(turn_tag, message_id=None)

            scheduled = (
                self.background_consolidator.maybe_schedule(
                    chat_id,
                    on_complete=lambda res: (
                        self.reset_session_prompt(res.chat_id)
                        if res.saved_files
                        else self._clear_history_trim_floor(res.chat_id)
                    ),
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
            "llm_configured": not isinstance(self.llm, NoopModelClient),
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
    llm: ModelInvoker | None = None,
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
