"""运行期可观测插件（零第三方依赖）：实现 ObserverProtocol，提供有界会话账本、单调游标、分阶段耗时拆解与后台沉淀统计。"""
from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import TypeVar

from personal_assistant.agent import Plugin, RuntimeEvent, TaskEvent, ToolStep, TurnEnd
from personal_assistant.ai import StreamDone, StreamError, TokenUsage

OBSERVER_NAME = "telemetry"
_T = TypeVar("_T")


# 安全解析正整数配置项（排除布尔值），非法或小于 1 时回退默认值
def _pos_int(raw: object, default: int) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        return default
    return raw if raw >= 1 else default


# 按最近访问顺序获取或创建有界字典项并在超限时淘汰最旧键，防止长挂内存泄漏
def _touch_bounded(mapping: dict[str, _T], key: str, factory: Callable[[], _T], limit: int) -> _T:
    val = mapping.pop(key, None)
    if val is None:
        if len(mapping) >= limit:
            mapping.pop(next(iter(mapping)))
        val = factory()
    mapping[key] = val
    return val


# 安全计算缓存命中率等比率指标（分母为 0 时返回 None）
def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den > 0 else None


# 运行期遥测配置：控制有界 MRU 会话账本容量上限
@dataclass(frozen=True)
class TelemetryConfig:
    max_sessions: int = 256

    # 校验正整数边界，防止外部直接构造 0 或负数配置引发运行时异常
    def __post_init__(self) -> None:
        if isinstance(self.max_sessions, bool) or self.max_sessions < 1:
            raise ValueError("max_sessions 必须是正整数。")

    # 从 config/tools.json 的 telemetry 字段解析配置并保障安全上下限
    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "TelemetryConfig":
        return cls(max_sessions=_pos_int(raw.get("max_sessions"), 256))


# 单轮完整对话遥测记录（含首字、模型推理、工具耗时与调度开销拆解）
@dataclass(frozen=True)
class TurnRecord:
    seq: int
    session_key: str
    total_ms: float
    ttft_ms: float | None
    llm_ms: float
    tool_ms: float
    overhead_ms: float
    llm_calls: int
    tool_calls: int
    tools: tuple[str, ...]
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int

    # 转换为便于接口序列化与前端展示的只读字典
    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "tools": list(self.tools),
            "cache_hit_rate": _rate(self.cached_tokens, self.prompt_tokens),
        }


# 后台异步任务遥测记录（例如后台记忆沉淀的耗时、Token 与错误信息）
@dataclass(frozen=True)
class TaskRecord:
    seq: int
    session_key: str
    kind: str
    duration_ms: float
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    error: str | None
    details: dict[str, object]


# 进行中回合与累计账本共享的阶段耗时和 Token 计数字段
@dataclass
class _TurnMetrics:
    llm_calls: int = 0
    tool_calls: int = 0
    llm_ms: float = 0.0
    tool_ms: float = 0.0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0

    # 将单次 TokenUsage 显式累加到指标计数器
    def add_usage(self, usage: object) -> None:
        if isinstance(usage, TokenUsage):
            self.prompt_tokens += usage.prompt_tokens
            self.cached_tokens += usage.cached_tokens
            self.completion_tokens += usage.completion_tokens


# 单会话进行中的前台回合暂存累加器（不与后台 TaskEvent 混用）
@dataclass
class _InFlightTurn(_TurnMetrics):
    ttft_ms: float | None = None
    tools: list[str] = field(default_factory=list)


# 共享累计统计器：统一用于全局 totals 与单会话 session 账本
@dataclass
class _CumulativeStats(_TurnMetrics):
    turns: int = 0
    total_ms: float = 0.0
    last_turn: TurnRecord | None = field(default=None, repr=False)

    # 显式累加已完结的单轮遥测记录并更新最近回合引用
    def add_turn(self, rec: TurnRecord) -> None:
        self.turns += 1
        self.total_ms = round(self.total_ms + rec.total_ms, 3)
        self.llm_ms = round(self.llm_ms + rec.llm_ms, 3)
        self.tool_ms = round(self.tool_ms + rec.tool_ms, 3)
        self.llm_calls += rec.llm_calls
        self.tool_calls += rec.tool_calls
        self.prompt_tokens += rec.prompt_tokens
        self.cached_tokens += rec.cached_tokens
        self.completion_tokens += rec.completion_tokens
        self.last_turn = rec

    # 导出累计统计字典（直接拼装字段避免展开 last_turn 产生无效深拷贝）
    def to_dict(self) -> dict[str, object]:
        return {
            "turns": self.turns,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "total_ms": self.total_ms,
            "llm_ms": self.llm_ms,
            "tool_ms": self.tool_ms,
            "mean_turn_ms": round(self.total_ms / self.turns, 3) if self.turns > 0 else None,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
            "cache_hit_rate": _rate(self.cached_tokens, self.prompt_tokens),
        }


# 内层可观测器实现（符合 ObserverProtocol）：线程安全收集回合耗时与后台任务账本
class TelemetryObserver:
    name = OBSERVER_NAME

    # 初始化线程互斥锁与有界会话隔离累加器
    def __init__(self, config: TelemetryConfig | None = None) -> None:
        self.config = config or TelemetryConfig()
        self._lock = threading.Lock()
        self._turn_seq = 0
        self._task_seq = 0
        self._bg_runs = 0
        self._bg_failures = 0
        self._bg_dur = 0.0
        self._bg_tokens = _TurnMetrics()
        self._bg_last: TaskRecord | None = None
        self._in_flight: dict[str, _InFlightTurn] = {}
        self._sessions: dict[str, _CumulativeStats] = {}
        self._totals = _CumulativeStats()

    # 接收微内核按 session_key 广播的运行期事件并归集分阶段指标（遇 StreamError 自动清理在途状态）
    def on_event(self, session_key: str, event: RuntimeEvent) -> None:
        if not isinstance(event, (StreamDone, StreamError, ToolStep, TurnEnd, TaskEvent)):
            return
        with self._lock:
            if isinstance(event, StreamError):
                self._in_flight.pop(session_key, None)
                return
            if isinstance(event, (StreamDone, ToolStep)):
                acc = _touch_bounded(self._in_flight, session_key, _InFlightTurn, self.config.max_sessions)
                if isinstance(event, StreamDone):
                    if acc.ttft_ms is None and event.ttft_ms is not None:
                        acc.ttft_ms = round(max(0.0, float(event.ttft_ms)), 3)
                    acc.llm_ms += max(0.0, float(event.latency_ms))
                    acc.llm_calls += 1
                    acc.add_usage(event.usage)
                else:
                    acc.tool_ms += max(0.0, float(event.duration_ms))
                    acc.tool_calls += 1
                    acc.tools.append(event.tool)
            elif isinstance(event, TurnEnd):
                acc = self._in_flight.pop(session_key, None) or _InFlightTurn()
                llm_ms = round(acc.llm_ms, 3)
                tool_ms = round(acc.tool_ms, 3)
                total_ms = round(max(0.0, float(event.total_ms)), 3)
                self._turn_seq += 1
                rec = TurnRecord(
                    seq=self._turn_seq,
                    session_key=session_key,
                    total_ms=total_ms,
                    ttft_ms=acc.ttft_ms,
                    llm_ms=llm_ms,
                    tool_ms=tool_ms,
                    overhead_ms=round(max(0.0, total_ms - llm_ms - tool_ms), 3),
                    llm_calls=acc.llm_calls,
                    tool_calls=acc.tool_calls,
                    tools=tuple(acc.tools),
                    prompt_tokens=acc.prompt_tokens,
                    cached_tokens=acc.cached_tokens,
                    completion_tokens=acc.completion_tokens,
                )
                self._totals.add_turn(rec)
                sess = _touch_bounded(self._sessions, session_key, _CumulativeStats, self.config.max_sessions)
                sess.add_turn(rec)
            else:
                if event.kind == "turn_aborted":
                    self._in_flight.pop(session_key, None)
                    return
                dur = round(max(0.0, float(event.duration_ms)), 3)
                usage = event.usage if isinstance(event.usage, TokenUsage) else None
                self._task_seq += 1
                t_rec = TaskRecord(
                    seq=self._task_seq,
                    session_key=session_key,
                    kind=event.kind,
                    duration_ms=dur,
                    prompt_tokens=usage.prompt_tokens if usage is not None else 0,
                    cached_tokens=usage.cached_tokens if usage is not None else 0,
                    completion_tokens=usage.completion_tokens if usage is not None else 0,
                    error=event.error,
                    details=dict(event.details),
                )
                self._bg_runs += 1
                if event.error is not None:
                    self._bg_failures += 1
                self._bg_dur = round(self._bg_dur + dur, 3)
                self._bg_last = t_rec
                self._bg_tokens.add_usage(usage)

    # 导出紧凑只读快照，供 AssistantApp.chat / /status / 前端展示
    def snapshot(self, *, session_key: str | None = None) -> dict[str, object]:
        with self._lock:
            sess = self._sessions.get(session_key) if session_key is not None else None
            if session_key is not None:
                last = sess.last_turn if sess is not None else None
            else:
                last = self._totals.last_turn
            data: dict[str, object] = {
                "cursor": self._turn_seq,
                "task_cursor": self._task_seq,
                "totals": self._totals.to_dict(),
                "last_turn": last.to_dict() if last is not None else None,
                "background": {
                    "runs": self._bg_runs,
                    "failures": self._bg_failures,
                    "total_duration_ms": self._bg_dur,
                    "last_duration_ms": self._bg_last.duration_ms if self._bg_last is not None else None,
                    "prompt_tokens": self._bg_tokens.prompt_tokens,
                    "cached_tokens": self._bg_tokens.cached_tokens,
                    "completion_tokens": self._bg_tokens.completion_tokens,
                    "last_error": self._bg_last.error if self._bg_last is not None else None,
                    "last_task": asdict(self._bg_last) if self._bg_last is not None else None,
                },
            }
            if session_key is not None:
                data["session"] = sess.to_dict() if sess is not None else _CumulativeStats().to_dict()
            return data


# 观测插件工厂函数：按 tools.json 的 telemetry 段创建独立 TelemetryObserver 实例
def create_observer(config: dict[str, object]) -> TelemetryObserver:
    return TelemetryObserver(TelemetryConfig.from_dict(config))


# 运行期观测主板插件（外层继承 Plugin ABC，内层向 observers 插槽注册 ObserverProtocol 工厂）
class TelemetryPlugin(Plugin):
    plugin_id = OBSERVER_NAME

    # 统一初始化生命周期：向主板插槽 5 注册 telemetry 观测器工厂
    def initialize(self) -> None:
        self.context.register_observer(self.plugin_id, create_observer)
