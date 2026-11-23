"""Layer 1 KV 缓存观测模块：统计服务端真实 cached_tokens 命中率与本地前缀复用率。"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from statistics import mean

from personal_assistant.ai.client import LLMClient
from personal_assistant.ai.types import TokenUsage


# 将单次请求的 tools 与 messages 序列化为逐行稳定文本，便于精确计算最长公共前缀
def serialize_request(
    messages: list[dict[str, object]],
    tools: list[dict[str, object]] | None = None,
) -> str:
    lines = [json.dumps({"tools": tools or []}, ensure_ascii=False, sort_keys=True)]
    for msg in messages:
        lines.append(json.dumps(msg, ensure_ascii=False, sort_keys=True))
    return "\n".join(lines) + "\n"


# 计算相邻两次序列化请求的最长公共前缀字符数
def common_prefix_len(prev: str, curr: str) -> int:
    for idx, (c1, c2) in enumerate(zip(prev, curr)):
        if c1 != c2:
            return idx
    return min(len(prev), len(curr))


# 单次 LLM 调用的缓存观测记录（含本地前缀复用率与可选服务端 TokenUsage）
@dataclass(frozen=True)
class CallCacheRecord:
    session_key: str
    prefix_chars: int
    total_chars: int
    prefix_reuse_rate: float | None
    usage: TokenUsage | None = None

    # 转换为便于日志与评测报告输出的字典
    def to_dict(self) -> dict[str, object]:
        return {
            "session_key": self.session_key,
            "prefix_chars": self.prefix_chars,
            "total_chars": self.total_chars,
            "prefix_reuse_rate": self.prefix_reuse_rate,
            "usage": self.usage.to_dict() if self.usage is not None else None,
        }


# 汇总一组调用记录的服务端缓存命中率与本地平均前缀复用率
def summarize_records(records: list[CallCacheRecord]) -> dict[str, object]:
    prompt_tokens = sum(r.usage.prompt_tokens for r in records if r.usage is not None)
    cached_tokens = sum(r.usage.cached_tokens for r in records if r.usage is not None)
    completion_tokens = sum(r.usage.completion_tokens for r in records if r.usage is not None)
    warm_rates = [r.prefix_reuse_rate for r in records if r.prefix_reuse_rate is not None]
    server_rate = round(cached_tokens / prompt_tokens, 4) if prompt_tokens > 0 else None
    mean_prefix = round(mean(warm_rates), 4) if warm_rates else None
    return {
        "llm_calls": len(records),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "server_cache_hit_rate": server_rate,
        "warm_calls": len(warm_rates),
        "mean_prefix_reuse_rate": mean_prefix,
    }


# 将缓存汇总字典格式化为终端单行可读摘要
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

    return f"{server_part} | {prefix_part} | 模型调用 {calls} 次"


# 缓存监控器：按会话隔离记录上一次请求前缀，并线程安全地累积全部调用统计
class CacheMonitor:
    # 初始化线程互斥锁、会话前缀缓存与调用明细列表
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_serialized: dict[str, str] = {}
        self.records: list[CallCacheRecord] = []

    # 返回当前已记录条数，用于截取单轮或单题区间的增量统计
    def snapshot_index(self) -> int:
        with self._lock:
            return len(self.records)

    # 记录一次成功完成的 LLM 调用并计算其与同会话上一次请求的前缀复用率
    def record(
        self,
        session_key: str,
        serialized: str,
        usage: TokenUsage | None = None,
    ) -> CallCacheRecord:
        with self._lock:
            prev = self._last_serialized.get(session_key)
            total = len(serialized)
            if prev is None or total == 0:
                prefix_len = 0
                reuse_rate = None
            else:
                prefix_len = common_prefix_len(prev, serialized)
                reuse_rate = round(prefix_len / total, 4)
            self._last_serialized[session_key] = serialized
            entry = CallCacheRecord(
                session_key=session_key,
                prefix_chars=prefix_len,
                total_chars=total,
                prefix_reuse_rate=reuse_rate,
                usage=usage,
            )
            self.records.append(entry)
            return entry

    # 汇总从 since 位置起的调用统计（默认汇总全部历史调用）
    def summary(self, since: int = 0) -> dict[str, object]:
        with self._lock:
            snapshot = list(self.records[since:])
        return summarize_records(snapshot)

    # 包装任意 LLMClient，在不侵入上层循环的前提下自动采集缓存统计
    def wrap(self, llm: LLMClient, session_key: str = "default") -> "TrackingLLMClient":
        return TrackingLLMClient(llm, self, session_key=session_key)


# 透明代理客户端：在每次 respond 前后采样请求前缀与服务端 usage
class TrackingLLMClient(LLMClient):
    # 绑定底层模型客户端、监控器与所属会话标识
    def __init__(
        self,
        inner: LLMClient,
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
        return getattr(self._usage_local, "value", None)

    # 设置当前线程最近一次被监控调用的 Token 统计
    @last_usage.setter
    def last_usage(self, value: TokenUsage | None) -> None:
        self._usage_local.value = value

    # 转发请求给底层客户端，并在成功返回后记录本地前缀复用率与服务端 usage
    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        serialized = serialize_request(messages, tools)
        self.last_usage = None
        reply = self.inner.respond(messages, tools)
        raw_usage = getattr(self.inner, "last_usage", None)
        usage = raw_usage if isinstance(raw_usage, TokenUsage) else None
        self.last_usage = usage
        self.monitor.record(self.session_key, serialized, usage)
        return reply
