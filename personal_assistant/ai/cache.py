"""Layer 1 KV 缓存观测模块：面向中立 Context 与 TokenUsage 统计服务端真实命中率、本地前缀复用率、失配归因与上下文构成。"""
from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction

from personal_assistant.ai.protocol import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    ModelInvoker,
    StreamDone,
    StreamError,
    StreamOptions,
    TextBlock,
    ThinkingBlock,
    TokenUsage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


# 将单条中立消息转为稳定可序列化的规范字典，供前缀复用率与失配归因计算使用
def _canonical_message_dict(message: UserMessage | AssistantMessage | ToolResultMessage) -> dict[str, object]:
    if isinstance(message, UserMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, ToolResultMessage):
        return {
            "role": "tool_result",
            "tool_call_id": message.tool_call_id,
            "tool_name": message.tool_name,
            "content": message.content,
            "is_error": message.is_error,
        }
    blocks_out: list[dict[str, object]] = []
    for block in message.blocks:
        if isinstance(block, TextBlock):
            blocks_out.append({"type": "text", "text": block.text})
        elif isinstance(block, ThinkingBlock):
            blocks_out.append({"type": "thinking", "text": block.text, "signature": block.signature})
        elif isinstance(block, ToolCall):
            blocks_out.append(
                {
                    "type": "tool_call",
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.arguments,
                }
            )
    return {"role": "assistant", "blocks": blocks_out}


# 将中立 Context 拆解为带段标签的稳定规范行元组（tools / system / message[i]:role）
def _context_segments(context: Context) -> tuple[tuple[str, str], ...]:
    tools_payload = [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": dict(tool.parameters),
            "param_enums": (
                {k: list(v) for k, v in tool.param_enums.items()}
                if tool.param_enums is not None
                else None
            ),
            "required": list(tool.required_names),
        }
        for tool in context.tools
    ]
    segments: list[tuple[str, str]] = [
        ("tools", json.dumps({"tools": tools_payload}, ensure_ascii=False, sort_keys=True))
    ]
    if context.system_prompt:
        segments.append(
            (
                "system",
                json.dumps({"role": "system", "content": context.system_prompt}, ensure_ascii=False, sort_keys=True),
            )
        )
    for idx, msg in enumerate(context.messages):
        if isinstance(msg, UserMessage):
            role = "user"
        elif isinstance(msg, ToolResultMessage):
            role = "tool_result"
        else:
            role = "assistant"
        segments.append(
            (
                f"message[{idx}]:{role}",
                json.dumps(_canonical_message_dict(msg), ensure_ascii=False, sort_keys=True),
            )
        )
    return tuple(segments)


# 将带段标签的规范行元组拼接为与 serialize_context 完全一致的逐行文本
def _join_segments(segments: tuple[tuple[str, str], ...]) -> str:
    return "\n".join(line for _, line in segments) + "\n"


# 将中立 Context（工具签名、系统提示词与消息历史）序列化为逐行稳定文本，便于精确计算最长公共前缀
def serialize_context(context: Context) -> str:
    return _join_segments(_context_segments(context))


# 统计中立 Context 各逻辑段（tools / system / user / assistant / tool_results）的字符构成
def context_composition(context: Context) -> dict[str, int]:
    tools_chars = 0
    if context.tools:
        segments = _context_segments(Context(tools=context.tools))
        tools_chars = len(segments[0][1])
    user_chars = 0
    assistant_chars = 0
    tool_result_chars = 0
    for msg in context.messages:
        if isinstance(msg, UserMessage):
            user_chars += len(msg.content)
        elif isinstance(msg, ToolResultMessage):
            tool_result_chars += len(msg.content)
        else:
            for block in msg.blocks:
                if isinstance(block, TextBlock):
                    assistant_chars += len(block.text)
                elif isinstance(block, ThinkingBlock):
                    assistant_chars += len(block.text) + len(block.signature)
                elif isinstance(block, ToolCall):
                    assistant_chars += len(block.name) + len(block.arguments)
    return {
        "tools": tools_chars,
        "system": len(context.system_prompt),
        "user": user_chars,
        "assistant": assistant_chars,
        "tool_results": tool_result_chars,
    }


# 对比相邻两次调用的规范段序列，精准定位首次前缀失配来源（cold_start / append / tools / system / message[i]:role）
def _locate_prefix_break(
    prev_segments: tuple[tuple[str, str], ...] | None,
    curr_segments: tuple[tuple[str, str], ...],
) -> str:
    if prev_segments is None:
        return "cold_start"
    shared = min(len(prev_segments), len(curr_segments))
    for idx in range(shared):
        prev_label, prev_line = prev_segments[idx]
        curr_label, curr_line = curr_segments[idx]
        if prev_label != curr_label or prev_line != curr_line:
            if prev_label == "system" or curr_label == "system":
                return "system"
            return curr_label
    if len(curr_segments) >= len(prev_segments):
        return "append"
    return prev_segments[len(curr_segments)][0]


# 判断单次调用的 break_section 是否属于非预期前缀断档（冷启动与纯尾部追加不算断档）
def _is_prefix_break(break_section: str | None) -> bool:
    return break_section is not None and break_section not in {"cold_start", "append"}


# 计算相邻两次序列化请求的最长公共前缀字符数
def _common_prefix_len(prev: str, curr: str) -> int:
    for idx, (c1, c2) in enumerate(zip(prev, curr)):
        if c1 != c2:
            return idx
    return min(len(prev), len(curr))


# 单次模型调用的缓存观测记录（含本地前缀复用率、失配归因、上下文构成与可选服务端 TokenUsage）
@dataclass(frozen=True)
class CallCacheRecord:
    session_key: str
    prefix_chars: int
    total_chars: int
    prefix_reuse_rate: float | None
    usage: TokenUsage | None = None
    break_section: str | None = None
    composition: dict[str, int] | None = None


# 校验有界容量参数为正整数（排除布尔值）
def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} 必须是正整数。")
    return value


# 根据聚合后的标量计数构建标准缓存摘要字典
def _build_summary(
    *,
    llm_calls: int,
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
    warm_calls: int,
    prefix_rate_sum: Fraction,
    prefix_breaks: int,
    last_break_section: str | None,
    last_composition: dict[str, int] | None,
) -> dict[str, object]:
    server_rate = round(cached_tokens / prompt_tokens, 4) if prompt_tokens > 0 else None
    mean_prefix = round(float(prefix_rate_sum / warm_calls), 4) if warm_calls > 0 else None
    return {
        "llm_calls": llm_calls,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "server_cache_hit_rate": server_rate,
        "warm_calls": warm_calls,
        "mean_prefix_reuse_rate": mean_prefix,
        "prefix_breaks": prefix_breaks,
        "last_break_section": last_break_section,
        "last_composition": dict(last_composition) if last_composition is not None else None,
    }


# 汇总一组调用记录的服务端缓存命中率、本地平均前缀复用率、失配次数与最新上下文构成
def summarize_records(records: list[CallCacheRecord]) -> dict[str, object]:
    prompt_tokens = sum(r.usage.prompt_tokens for r in records if r.usage is not None)
    cached_tokens = sum(r.usage.cached_tokens for r in records if r.usage is not None)
    completion_tokens = sum(r.usage.completion_tokens for r in records if r.usage is not None)
    warm_rates = [r.prefix_reuse_rate for r in records if r.prefix_reuse_rate is not None]
    prefix_breaks = sum(1 for r in records if _is_prefix_break(r.break_section))
    last_break = records[-1].break_section if records else None
    last_comp = records[-1].composition if records else None
    return _build_summary(
        llm_calls=len(records),
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        completion_tokens=completion_tokens,
        warm_calls=len(warm_rates),
        prefix_rate_sum=sum((Fraction(r) for r in warm_rates), Fraction(0)),
        prefix_breaks=prefix_breaks,
        last_break_section=last_break,
        last_composition=last_comp,
    )


# 将缓存汇总字典格式化为终端单行可读摘要（发现前缀失配时追加归因提示）
def format_cache_summary(summary: dict[str, object]) -> str:
    calls = int(summary.get("llm_calls") or 0)
    prompt = int(summary.get("prompt_tokens") or 0)
    cached = int(summary.get("cached_tokens") or 0)
    server_rate = summary.get("server_cache_hit_rate")
    prefix_rate = summary.get("mean_prefix_reuse_rate")

    if isinstance(server_rate, (int, float)) and prompt > 0:
        server_part = f"服务端命中 {cached}/{prompt} ({server_rate * 100:.1f}%)"
    else:
        server_part = "服务端未返回 usage"

    if isinstance(prefix_rate, (int, float)):
        prefix_part = f"本地前缀复用 {prefix_rate * 100:.1f}%"
    else:
        prefix_part = "首次冷启动"

    base = f"{server_part} | {prefix_part} | 模型调用 {calls} 次"
    raw_breaks = summary.get("prefix_breaks")
    breaks = int(raw_breaks) if isinstance(raw_breaks, int) and not isinstance(raw_breaks, bool) else 0
    if breaks > 0:
        last_break = summary.get("last_break_section")
        detail = f"（最近：{last_break}）" if isinstance(last_break, str) and last_break else ""
        return f"{base} | 前缀失配 {breaks} 次{detail}"
    return base


# 缓存监控器：按会话隔离记录上一次请求前缀与段结构，并线程安全地维护有界窗口与全生命周期累计统计
class CacheMonitor:
    # 初始化线程互斥锁、有界会话前缀缓存、有界记录窗口与全量累计账本
    def __init__(self, *, max_records: int = 2000, max_sessions: int = 256) -> None:
        self.max_records = _validate_positive_int(max_records, "max_records")
        self.max_sessions = _validate_positive_int(max_sessions, "max_sessions")
        self._lock = threading.Lock()
        self._last_serialized: dict[str, str] = {}
        self._last_segments: dict[str, tuple[tuple[str, str], ...]] = {}
        self._records: deque[CallCacheRecord] = deque(maxlen=self.max_records)
        self._total_recorded = 0
        self._total_prompt_tokens = 0
        self._total_cached_tokens = 0
        self._total_completion_tokens = 0
        self._total_warm_calls = 0
        self._total_prefix_rate_sum = Fraction(0)
        self._total_prefix_breaks = 0
        self._last_break_section: str | None = None
        self._last_composition: dict[str, int] | None = None

    # 返回当前窗口内调用记录的只读列表快照
    @property
    def records(self) -> list[CallCacheRecord]:
        with self._lock:
            return list(self._records)

    # 返回单调递增的累计调用游标，用于截取单轮或单题区间的增量统计
    def snapshot_index(self) -> int:
        with self._lock:
            return self._total_recorded

    # 记录一次成功完成的模型调用，计算本地前缀复用率、失配归因并累加全量统计
    def record(
        self,
        session_key: str,
        serialized: str,
        usage: TokenUsage | None = None,
        *,
        segments: tuple[tuple[str, str], ...] | None = None,
        composition: dict[str, int] | None = None,
    ) -> CallCacheRecord:
        with self._lock:
            prev = self._last_serialized.pop(session_key, None)
            prev_segments = self._last_segments.pop(session_key, None)
            total = len(serialized)
            if prev is None or total == 0:
                prefix_len = 0
                reuse_rate = None
                break_section = "cold_start"
            else:
                prefix_len = _common_prefix_len(prev, serialized)
                reuse_rate = round(prefix_len / total, 4)
                if segments is not None and prev_segments is not None:
                    break_section = _locate_prefix_break(prev_segments, segments)
                else:
                    break_section = "append" if serialized.startswith(prev) else "prefix"

            if len(self._last_serialized) >= self.max_sessions:
                evicted = next(iter(self._last_serialized))
                self._last_serialized.pop(evicted, None)
                self._last_segments.pop(evicted, None)
            self._last_serialized[session_key] = serialized
            if segments is not None:
                self._last_segments[session_key] = segments

            comp_copy = dict(composition) if composition is not None else None
            entry = CallCacheRecord(
                session_key=session_key,
                prefix_chars=prefix_len,
                total_chars=total,
                prefix_reuse_rate=reuse_rate,
                usage=usage,
                break_section=break_section,
                composition=comp_copy,
            )
            self._records.append(entry)
            self._total_recorded += 1
            if usage is not None:
                self._total_prompt_tokens += usage.prompt_tokens
                self._total_cached_tokens += usage.cached_tokens
                self._total_completion_tokens += usage.completion_tokens
            if reuse_rate is not None:
                self._total_warm_calls += 1
                self._total_prefix_rate_sum += Fraction(reuse_rate)
            if _is_prefix_break(break_section):
                self._total_prefix_breaks += 1
            self._last_break_section = break_section
            self._last_composition = comp_copy
            return entry

    # 汇总从非负游标 since 位置起的调用统计（支持按 session_key 切片隔离；since=0 且未指定会话时走 O(1) 全局累计）
    def summary(
        self,
        since: int = 0,
        *,
        session_key: str | None = None,
    ) -> dict[str, object]:
        with self._lock:
            if since == 0 and session_key is None:
                return _build_summary(
                    llm_calls=self._total_recorded,
                    prompt_tokens=self._total_prompt_tokens,
                    cached_tokens=self._total_cached_tokens,
                    completion_tokens=self._total_completion_tokens,
                    warm_calls=self._total_warm_calls,
                    prefix_rate_sum=self._total_prefix_rate_sum,
                    prefix_breaks=self._total_prefix_breaks,
                    last_break_section=self._last_break_section,
                    last_composition=self._last_composition,
                )
            if since <= 0:
                snapshot = list(self._records)
            else:
                window_start = self._total_recorded - len(self._records)
                offset = max(0, since - window_start)
                snapshot = list(self._records)[offset:]
        if session_key is not None:
            snapshot = [rec for rec in snapshot if rec.session_key == session_key]
        return summarize_records(snapshot)

    # 包装任意 ModelInvoker，在不侵入上层循环的前提下自动采集缓存统计
    def wrap(self, invoker: ModelInvoker, *, session_key: str = "default") -> "TrackingModelClient":
        return TrackingModelClient(invoker, self, session_key=session_key)


# 透明代理客户端（实现 ModelInvoker）：在每次 stream/complete 前后采样中立 Context 前缀、失配归因与服务端 usage
class TrackingModelClient:
    # 绑定底层 ModelInvoker、监控器与默认所属会话标识
    def __init__(
        self,
        inner: ModelInvoker,
        monitor: CacheMonitor,
        *,
        session_key: str = "default",
    ) -> None:
        self.inner = inner
        self.monitor = monitor
        self.session_key = session_key
        self._usage_local = threading.local()

    # 读取当前线程最近一次被监控调用的 Token 统计
    @property
    def last_usage(self) -> TokenUsage | None:
        value = getattr(self._usage_local, "value", None)
        return value if isinstance(value, TokenUsage) else None

    # 设置当前线程最近一次被监控调用的 Token 统计
    @last_usage.setter
    def last_usage(self, value: TokenUsage | None) -> None:
        self._usage_local.value = value

    # 优先使用 StreamOptions.session_id 路由缓存统计会话，缺省回退至默认 session_key
    def _resolve_session_key(self, options: StreamOptions | None) -> str:
        if options is not None and options.session_id and options.session_id.strip():
            return options.session_id.strip()
        return self.session_key

    # 流式转发中立上下文给底层模型，并在 StreamDone 时单次记录本地前缀复用率、失配归因与服务端 usage
    def stream(
        self,
        context: Context,
        options: StreamOptions | None = None,
    ) -> Iterator[AssistantMessageEvent]:
        segments = _context_segments(context)
        serialized = _join_segments(segments)
        composition = context_composition(context)
        effective_session = self._resolve_session_key(options)
        self.last_usage = None
        raw_iter = self.inner.stream(context, options)

        # 消费底层事件流并在流完成时记录缓存指标，异常或 StreamError 不污染统计
        def _iter() -> Iterator[AssistantMessageEvent]:
            recorded = False
            try:
                for event in raw_iter:
                    if isinstance(event, StreamDone) and not recorded:
                        raw_usage = event.usage if isinstance(event.usage, TokenUsage) else self.inner.last_usage
                        usage = raw_usage if isinstance(raw_usage, TokenUsage) else None
                        self.last_usage = usage
                        self.monitor.record(
                            effective_session,
                            serialized,
                            usage,
                            segments=segments,
                            composition=composition,
                        )
                        recorded = True
                    elif isinstance(event, StreamError):
                        self.last_usage = None
                    yield event
            finally:
                close_fn = getattr(raw_iter, "close", None)
                if callable(close_fn):
                    close_fn()

        return _iter()

    # 同步转发中立上下文给底层模型，并在成功返回后记录本地前缀复用率、失配归因与服务端 usage
    def complete(
        self,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessage:
        segments = _context_segments(context)
        serialized = _join_segments(segments)
        composition = context_composition(context)
        effective_session = self._resolve_session_key(options)
        self.last_usage = None
        reply = self.inner.complete(context, options)
        raw_usage = reply.usage if isinstance(reply.usage, TokenUsage) else self.inner.last_usage
        usage = raw_usage if isinstance(raw_usage, TokenUsage) else None
        self.last_usage = usage
        self.monitor.record(
            effective_session,
            serialized,
            usage,
            segments=segments,
            composition=composition,
        )
        return reply
