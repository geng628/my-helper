"""核心应用编排（AssistantApp）：串联记忆、白名单工具、SQLite 历史与 ReAct 循环。"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import closing, contextmanager
from datetime import date
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
    AbortController,
    AbortSignal,
    AssistantMessage,
    AssistantMessageEvent,
    CacheMonitor,
    Context,
    LLMNotConfigured,
    ModelInvoker,
    ModelsConfig,
    NoopModelClient,
    StreamAbortedError,
    StreamDone,
    StreamError,
    StreamOptions,
    StreamStart,
    TextDelta,
    TextEnd,
    TextStart,
    ThinkingDelta,
    TokenUsage,
)
from personal_assistant.assistant.builtin_tools import MemoryToolsetProvider
from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.assistant.common import (
    now_shanghai,
    require_non_negative_float,
    require_positive_int,
    today_shanghai,
)
from personal_assistant.assistant.consolidator import (
    BackgroundConsolidator,
    ConsolidationPolicy,
    MemoryConsolidator,
)
from personal_assistant.assistant.context import (
    DEFAULT_FOLD_TRIGGER_CHARS,
    DEFAULT_PROMPT_CACHE_TTL_SECONDS,
    ContextAssembler,
)
from personal_assistant.assistant.db import StateStore
from personal_assistant.assistant.files_policy import FilesPolicy
from personal_assistant.assistant.memory import MemoryStore

MAX_SESSION_LOCKS = 256


# 单轮推理或工具调用超出运行期安全预算时构造的结构化异常
class TurnBudgetExceeded(RuntimeError):
    # 记录熔断类别与人类可读原因说明
    def __init__(self, kind: str, reason: str) -> None:
        self.kind = kind
        self.reason = reason
        super().__init__(reason)


# 将工具调用原始 JSON 参数规范化为紧凑有序串，便于检测等价参数的死循环重复调用
def _canonical_tool_args(raw_arguments: str) -> str:
    text = (raw_arguments or "{}").strip() or "{}"
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return " ".join(text.split())
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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

    # 流式调用前优先响应协同取消再执行预算核查；检测到 length 截断且含破损工具 JSON 时立即就地熔断收束
    def stream(
        self,
        context: Context,
        options: StreamOptions | None = None,
    ) -> Iterator[AssistantMessageEvent]:
        abort_signal = options.abort_signal if options is not None else None
        if abort_signal is not None and abort_signal.aborted:
            self.last_usage = None
            reason = abort_signal.reason or "请求已取消"
            yield StreamError(
                reason="aborted",
                message=AssistantMessage(stop_reason="aborted", error_message=reason),
                error_message=reason,
                latency_ms=0.0,
            )
            return

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
                if abort_signal is not None and abort_signal.aborted:
                    self.last_usage = None
                    reason = abort_signal.reason or "请求已取消"
                    yield StreamError(
                        reason="aborted",
                        message=AssistantMessage(stop_reason="aborted", error_message=reason),
                        error_message=reason,
                        latency_ms=0.0,
                    )
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
                    if event.reason == "aborted":
                        raise StreamAbortedError(event.error_message or "请求已取消")
                    raise RuntimeError(event.error_message)
                if isinstance(event, StreamDone):
                    return event.message
        finally:
            close_fn = getattr(event_iter, "close", None)
            if callable(close_fn):
                close_fn()
        raise RuntimeError("模型流式响应异常终止，未收到 StreamDone 事件。")


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
        self.max_turn_seconds = require_non_negative_float(max_turn_seconds, "max_turn_seconds")
        self.max_repeated_tool_calls = require_positive_int(
            max_repeated_tool_calls, "max_repeated_tool_calls"
        )
        validated_ttl = require_non_negative_float(
            prompt_cache_ttl_seconds, "prompt_cache_ttl_seconds"
        )
        validated_fold_chars = require_positive_int(
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
        self.context_assembler = ContextAssembler(
            self.memory,
            prompt_cache_ttl_seconds=validated_ttl,
            max_turn_context_chars=validated_fold_chars,
            clock=self._clock,
        )
        self._chat_locks_guard = threading.Lock()
        self._chat_locks: dict[str, tuple[threading.Lock, int]] = {}
        self._abort_lock = threading.Lock()
        self._active_aborts: dict[str, AbortController] = {}
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

    # 主动向指定会话正在执行的前台回合发送协同取消信号；存在在途回合并首次触发取消时返回 True
    def abort_turn(self, chat_id: str, *, reason: str = "用户已取消当前请求") -> bool:
        with self._abort_lock:
            controller = self._active_aborts.get(chat_id)
        if controller is None:
            return False
        return controller.abort(reason)

    # 先中止所有在途前台回合、等待后台沉淀收尾，再按所有权关闭已加载插件生命周期，最后关闭底层状态存储资源
    def close(self) -> None:
        with self._abort_lock:
            controllers = list(self._active_aborts.values())
        for controller in controllers:
            controller.abort("服务正在关闭")
        try:
            self.background_consolidator.wait(timeout=15.0)
            self.memory.close()
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

    # 读取当前系统前缀缓存空闲 TTL（秒，单一事实来源位于 ContextAssembler）
    @property
    def prompt_cache_ttl_seconds(self) -> float:
        return self.context_assembler.prompt_cache_ttl_seconds

    # 动态更新系统前缀缓存空闲 TTL（秒，写入前校验非负有限数）
    @prompt_cache_ttl_seconds.setter
    def prompt_cache_ttl_seconds(self, value: float) -> None:
        self.context_assembler.prompt_cache_ttl_seconds = require_non_negative_float(
            value, "prompt_cache_ttl_seconds"
        )

    # 读取单轮上下文折叠触发高水位字符阈值（单一事实来源位于 ContextAssembler）
    @property
    def max_turn_context_chars(self) -> int:
        return self.context_assembler.max_turn_context_chars

    # 动态更新单轮上下文折叠触发高水位字符阈值（写入前校验正整数）
    @max_turn_context_chars.setter
    def max_turn_context_chars(self, value: int) -> None:
        self.context_assembler.max_turn_context_chars = require_positive_int(
            value, "max_turn_context_chars"
        )

    # 拼装纯静态/低频系统提示词（委托 ContextAssembler，不含逐日日期与动态摘要）
    def build_system_prompt(self, registry: ToolRegistry) -> str:
        return self.context_assembler.build_system_prompt(registry)

    # 获取指定会话冻结的系统提示词（委托 ContextAssembler 执行空闲 TTL 冻结与 LRU 管理）
    def get_session_system_prompt(self, chat_id: str, registry: ToolRegistry) -> str:
        return self.context_assembler.get_session_system_prompt(chat_id, registry)

    # 主动重置指定会话（或全部会话）冻结的系统提示词与历史切块边界
    def reset_session_prompt(self, chat_id: str | None = None) -> None:
        self.context_assembler.reset_session(chat_id)

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

    # 处理一轮流式对话（单一主流程）：先同步校验输入与模型就绪状态，再返回支持协同取消的事件帧生成器
    def stream_chat(
        self,
        text: str,
        chat_id: str = "default",
        *,
        on_event: Callable[[AgentEvent], None] | None = None,
        abort_signal: AbortSignal | None = None,
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
            abort_signal=abort_signal,
        )

    # 内部生成器：同会话加锁串行执行，支持 abort_turn 与外部信号协同取消；成功后原子成对落库，取消/中断零脏写
    def _run_stream_turn(
        self,
        clean_text: str,
        chat_id: str,
        *,
        on_event: Callable[[AgentEvent], None] | None,
        abort_signal: AbortSignal | None = None,
    ) -> Generator[dict[str, object], None, None]:
        turn_controller = AbortController()
        unsub_external = (
            abort_signal.subscribe(
                lambda: turn_controller.abort(abort_signal.reason or "外部已取消当前请求")
            )
            if abort_signal is not None
            else None
        )
        try:
            if turn_controller.signal.aborted:
                yield {
                    "type": "aborted",
                    "reason": turn_controller.signal.reason or "请求已取消",
                    "chat_id": chat_id,
                }
                return

            done_payload: dict[str, object] | None = None
            aborted_payload: dict[str, object] | None = None
            with self._chat_turn_lock(chat_id):
                with self._abort_lock:
                    self._active_aborts[chat_id] = turn_controller
                turn_persisted = False
                turn_ended = False
                fact_used = False
                turn_tag = uuid.uuid4().hex[:12]
                try:
                    if turn_controller.signal.aborted:
                        aborted_payload = {
                            "type": "aborted",
                            "reason": turn_controller.signal.reason or "请求已取消",
                            "chat_id": chat_id,
                        }
                    else:
                        turn_day, turn_created_at = self._turn_clock_stamp()
                        self.context_assembler.maybe_archive_expired(turn_day)
                        compaction, active_all = self.state.get_context_snapshot(chat_id)
                        history = self.context_assembler.select_history(chat_id, active_all)
                        before_msg_id = self.state.max_message_id() + 1

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
                        context = self.context_assembler.assemble(
                            system_prompt=system_prompt,
                            summary_text=summary_text,
                            history=history,
                            user_text=clean_text,
                            registry=registry,
                            today=turn_day,
                        )

                        traces: list[dict[str, object]] = []
                        thinking_parts: list[str] = []
                        reply = ""
                        aborted_reason: str | None = None
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
                        projector = self.context_assembler.create_turn_projector()
                        with closing(
                            iter_react(
                                guarded_llm,
                                context,
                                registry,
                                convert_to_llm=projector,
                                tool_context=turn_ctx,
                                options=StreamOptions(
                                    session_id=chat_id,
                                    abort_signal=turn_controller.signal,
                                ),
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
                                    if event.reason == "aborted":
                                        aborted_reason = (
                                            event.error_message
                                            or turn_controller.signal.reason
                                            or "请求已取消"
                                        )
                                        break
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

                        if aborted_reason is not None or turn_controller.signal.aborted:
                            aborted_payload = {
                                "type": "aborted",
                                "reason": aborted_reason or turn_controller.signal.reason or "请求已取消",
                                "chat_id": chat_id,
                            }
                        else:
                            user_msg_id, assistant_msg_id = self.state.append_turn(
                                chat_id,
                                clean_text,
                                reply,
                                created_at=turn_created_at,
                            )
                            turn_persisted = True
                            if fact_used:
                                self.memory.finalize_turn_fact(turn_tag, message_id=user_msg_id)
                            scheduled = (
                                self.background_consolidator.maybe_schedule(
                                    chat_id,
                                    on_complete=lambda res: (
                                        self.reset_session_prompt(res.chat_id)
                                        if res.saved_files
                                        else self.context_assembler.clear_history_floor(res.chat_id)
                                    ),
                                )
                                if self.auto_consolidate
                                else False
                            )
                            done_payload = {
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
                finally:
                    if not turn_ended:
                        turn_controller.abort("连接已断开")
                        self._emit_event(
                            chat_id,
                            TaskEvent(kind="turn_aborted", duration_ms=0.0, error="turn_interrupted"),
                        )
                    if not turn_persisted and fact_used:
                        self.memory.finalize_turn_fact(turn_tag, message_id=None)
                    with self._abort_lock:
                        if self._active_aborts.get(chat_id) is turn_controller:
                            self._active_aborts.pop(chat_id, None)

            if aborted_payload is not None:
                yield aborted_payload
            elif done_payload is not None:
                yield done_payload
        finally:
            if unsub_external is not None:
                unsub_external()

    # 处理一轮完整同步对话：直接消费 stream_chat 单一实现，被取消时抛 StreamAbortedError
    def chat(
        self,
        text: str,
        chat_id: str = "default",
        *,
        on_event: Callable[[AgentEvent], None] | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        with closing(
            self.stream_chat(
                text,
                chat_id=chat_id,
                on_event=on_event,
                abort_signal=abort_signal,
            )
        ) as frames:
            for frame in frames:
                if frame.get("type") == "aborted":
                    raise StreamAbortedError(str(frame.get("reason") or "请求已取消"))
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
