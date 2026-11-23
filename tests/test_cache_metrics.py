"""KV 缓存观测单元测试：验证各厂商 usage 解析、本地前缀复用率与回归检测。"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from personal_assistant.ai.cache import (
    CacheMonitor,
    format_cache_summary,
    serialize_request,
)
from personal_assistant.ai import (
    LLMClient,
    LLMNotConfigured,
    NoopLLMClient,
    OpenAICompatibleLLM,
    TokenUsage,
    extract_usage,
)


# 带预设 usage 的假模型，用于确定性验证服务端缓存命中与本地前缀统计
class UsageScriptedLLM(LLMClient):
    # 保存预设回复与对应 usage 列表
    def __init__(self, items: list[tuple[dict[str, object], TokenUsage | None]]) -> None:
        self._items = list(items)
        self.last_usage: TokenUsage | None = None

    # 弹出下一条回复并同步更新 last_usage
    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        reply, usage = self._items.pop(0)
        self.last_usage = usage
        return reply


# 测试 extract_usage 兼容 DeepSeek、OpenAI 与 Anthropic 代理响应格式
def test_extract_usage_supports_deepseek_openai_and_anthropic_formats() -> None:
    deepseek = extract_usage(
        {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 80,
                "prompt_cache_hit_tokens": 768,
                "prompt_cache_miss_tokens": 232,
            }
        }
    )
    assert deepseek == TokenUsage(prompt_tokens=1000, cached_tokens=768, completion_tokens=80)
    assert deepseek.cache_hit_rate == 0.768

    openai = extract_usage(
        {
            "usage": {
                "prompt_tokens": 2000,
                "completion_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 1536},
            }
        }
    )
    assert openai == TokenUsage(prompt_tokens=2000, cached_tokens=1536, completion_tokens=120)
    assert openai.cache_hit_rate == 0.768

    anthropic_proxy = extract_usage(
        {
            "usage": {
                "input_tokens": 500,
                "output_tokens": 40,
                "cache_read_input_tokens": 400,
            }
        }
    )
    assert anthropic_proxy == TokenUsage(prompt_tokens=500, cached_tokens=400, completion_tokens=40)
    assert extract_usage({"choices": []}) is None


# 测试 OpenAICompatibleLLM 在请求后自动记录响应中的 last_usage
def test_openai_compatible_llm_records_last_usage() -> None:
    # 模拟接口返回带 DeepSeek 缓存命中信息的响应体
    def fake_post(url: str, payload: dict[str, object], headers: dict[str, str]) -> dict[str, object]:
        return {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {
                "prompt_tokens": 1200,
                "completion_tokens": 30,
                "prompt_cache_hit_tokens": 1024,
            },
        }

    client = OpenAICompatibleLLM(
        api_key="sk-test",
        base_url="https://example.com/v1",
        model="demo-model",
        post_json=fake_post,
    )
    reply = client.respond([{"role": "user", "content": "hi"}])
    assert reply["content"] == "ok"
    assert client.last_usage == TokenUsage(prompt_tokens=1200, cached_tokens=1024, completion_tokens=30)


# 测试本地前缀复用率：追加式历史高命中、改动 system 前缀瞬间暴露失效、多会话隔离与异常安全
def test_cache_monitor_prefix_reuse_regression_and_session_isolation() -> None:
    monitor = CacheMonitor()
    tools = [{"type": "function", "function": {"name": "search_markdown"}}]
    sys_msg = {"role": "system", "content": "你是个人助手。" + "规则说明" * 40}

    llm = UsageScriptedLLM(
        [
            ({"role": "assistant", "content": "回答1"}, TokenUsage(prompt_tokens=1000, cached_tokens=0, completion_tokens=20)),
            ({"role": "assistant", "content": "回答2"}, TokenUsage(prompt_tokens=1100, cached_tokens=960, completion_tokens=25)),
            ({"role": "assistant", "content": "回答3"}, TokenUsage(prompt_tokens=1120, cached_tokens=0, completion_tokens=20)),
        ]
    )
    tracked_a = monitor.wrap(llm, session_key="chat-a")

    # 第 1 次冷启动：prefix_reuse_rate 为 None
    m1 = [sys_msg, {"role": "user", "content": "第一问"}]
    tracked_a.respond(m1, tools)
    assert monitor.records[0].prefix_reuse_rate is None

    # 第 2 次纯追加：前文完全复用，本地前缀复用率应很高（> 0.8）
    m2 = [sys_msg, {"role": "user", "content": "第一问"}, {"role": "assistant", "content": "回答1"}, {"role": "user", "content": "第二问"}]
    tracked_a.respond(m2, tools)
    assert monitor.records[1].prefix_reuse_rate is not None
    assert monitor.records[1].prefix_reuse_rate > 0.8

    # 第 3 次人为改动 system 开头：应立刻捕获前缀失效（复用率跌到极低）
    broken_sys = {"role": "system", "content": "【时间变了】你是个人助手。" + "规则说明" * 40}
    m3 = [broken_sys, {"role": "user", "content": "第三问"}]
    tracked_a.respond(m3, tools)
    assert monitor.records[2].prefix_reuse_rate is not None
    assert monitor.records[2].prefix_reuse_rate < 0.3

    # 未配置模型抛异常时，不应污染已有监控记录
    tracked_fail = monitor.wrap(NoopLLMClient(), session_key="chat-a")
    with pytest.raises(LLMNotConfigured):
        tracked_fail.respond(m2, tools)
    assert len(monitor.records) == 3

    # 另一会话 chat-b 独立冷启动，不与 chat-a 串前缀
    monitor.record("chat-b", serialize_request(m1, tools), None)
    assert monitor.records[3].prefix_reuse_rate is None

    # 验证汇总指标与终端格式化输出
    summary = monitor.summary()
    assert summary["llm_calls"] == 4
    assert summary["prompt_tokens"] == 3220
    assert summary["cached_tokens"] == 960
    assert summary["server_cache_hit_rate"] == round(960 / 3220, 4)
    text = format_cache_summary(summary)
    assert "服务端命中 960/3220" in text
    assert "本地前缀复用" in text

    cold_text = format_cache_summary({"llm_calls": 1, "prompt_tokens": 0, "cached_tokens": 0, "server_cache_hit_rate": None, "mean_prefix_reuse_rate": None})
    assert "服务端未返回 usage" in cold_text
    assert "首次冷启动" in cold_text


# 测试多线程共享 OpenAICompatibleLLM 与 CacheMonitor 时 last_usage 线程隔离、统计无丢数且异常自动清零
def test_openai_compatible_llm_last_usage_and_cache_monitor_thread_safe() -> None:
    n_threads = 8
    enter_barrier = threading.Barrier(n_threads, timeout=5.0)
    read_barrier = threading.Barrier(n_threads, timeout=5.0)
    should_fail = threading.local()

    # 强制 8 个并发请求交叠返回各线程专属 usage，验证不互相覆盖
    def concurrent_post(
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> dict[str, object]:
        if getattr(should_fail, "flag", False):
            raise RuntimeError("boom")
        msgs = payload["messages"]
        assert isinstance(msgs, list)
        tid = int(str(msgs[-1]["content"]).split("-")[-1])
        enter_barrier.wait()
        return {
            "choices": [{"message": {"role": "assistant", "content": f"reply-{tid}"}}],
            "usage": {
                "prompt_tokens": 1000 + tid,
                "prompt_cache_hit_tokens": 100 + tid,
                "completion_tokens": 10 + tid,
            },
        }

    client = OpenAICompatibleLLM(
        api_key="sk-test",
        base_url="https://example.com/v1",
        model="demo-model",
        post_json=concurrent_post,
    )

    # 先测裸 OpenAICompatibleLLM：全部 8 线程均写完后再同时读 last_usage，绝不串台
    def worker_direct(tid: int) -> TokenUsage | None:
        reply = client.respond([{"role": "user", "content": f"thread-{tid}"}])
        assert reply["content"] == f"reply-{tid}"
        read_barrier.wait()
        return client.last_usage

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        observed = list(pool.map(worker_direct, range(n_threads)))

    for tid, usage in enumerate(observed):
        assert usage == TokenUsage(
            prompt_tokens=1000 + tid,
            cached_tokens=100 + tid,
            completion_tokens=10 + tid,
        )

    # 再测共享 CacheMonitor + TrackingLLMClient 并发记录，全量统计零丢失、会话键不混用
    monitor = CacheMonitor()

    # 单线程并发触发 TrackingLLMClient.respond 与 monitor.record
    def worker_tracked(tid: int) -> TokenUsage | None:
        tracked = monitor.wrap(client, session_key=f"chat-{tid}")
        tracked.respond([{"role": "user", "content": f"thread-{tid}"}])
        return tracked.last_usage

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        tracked_usages = list(pool.map(worker_tracked, range(n_threads)))

    for tid, usage in enumerate(tracked_usages):
        assert usage == TokenUsage(
            prompt_tokens=1000 + tid,
            cached_tokens=100 + tid,
            completion_tokens=10 + tid,
        )

    summary = monitor.summary()
    assert summary["llm_calls"] == n_threads
    assert summary["prompt_tokens"] == sum(1000 + i for i in range(n_threads))
    assert summary["cached_tokens"] == sum(100 + i for i in range(n_threads))
    assert summary["completion_tokens"] == sum(10 + i for i in range(n_threads))
    by_session = {rec.session_key: rec.usage for rec in monitor.records}
    assert len(by_session) == n_threads
    for tid in range(n_threads):
        assert by_session[f"chat-{tid}"] == TokenUsage(
            prompt_tokens=1000 + tid,
            cached_tokens=100 + tid,
            completion_tokens=10 + tid,
        )

    # 同一线程随后一次 respond() 失败抛异常时，last_usage 必须清零而非残留旧值
    enter_barrier = threading.Barrier(1, timeout=5.0)
    client.respond([{"role": "user", "content": "thread-0"}])
    assert client.last_usage == TokenUsage(prompt_tokens=1000, cached_tokens=100, completion_tokens=10)
    should_fail.flag = True
    with pytest.raises(RuntimeError, match="boom"):
        client.respond([{"role": "user", "content": "thread-0"}])
    assert client.last_usage is None
