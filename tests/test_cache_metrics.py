"""KV 缓存观测单元测试：验证各厂商 usage 解析、中立 Context 前缀复用率与多线程安全。"""
from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest

from personal_assistant.ai import (
    ApiProviderRegistry,
    AssistantMessage,
    AssistantMessageEvent,
    CacheMonitor,
    Context,
    LLMNotConfigured,
    ModelClient,
    ModelProfile,
    NoopModelClient,
    OpenAICompletionsProvider,
    StreamDone,
    StreamOptions,
    TokenUsage,
    ToolSchema,
    UserMessage,
    extract_openai_usage,
    format_cache_summary,
    serialize_context,
)


# 带预设 usage 的假模型（实现 ModelInvoker），用于确定性验证服务端缓存命中与本地前缀统计
class UsageScriptedLLM:
    # 保存预设回复与对应 usage 列表
    def __init__(self, items: list[tuple[AssistantMessage, TokenUsage | None]]) -> None:
        self._items = list(items)
        self.last_usage: TokenUsage | None = None

    # 流式弹出下一条回复并同步更新 last_usage
    def stream(
        self,
        context: Context,
        options: StreamOptions | None = None,
    ) -> Iterator[AssistantMessageEvent]:
        del context, options
        reply, usage = self._items.pop(0)
        self.last_usage = usage
        yield StreamDone(reason="stop", message=reply, usage=usage)

    # 同步弹出下一条回复并同步更新 last_usage
    def complete(
        self,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessage:
        del context, options
        reply, usage = self._items.pop(0)
        self.last_usage = usage
        return reply


# 测试 extract_openai_usage 兼容 DeepSeek、OpenAI 与 Anthropic 代理响应格式
def test_extract_openai_usage_supports_deepseek_openai_and_anthropic_formats() -> None:
    deepseek = extract_openai_usage(
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

    openai = extract_openai_usage(
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

    anthropic_proxy = extract_openai_usage(
        {
            "usage": {
                "input_tokens": 500,
                "output_tokens": 40,
                "cache_read_input_tokens": 400,
            }
        }
    )
    assert anthropic_proxy == TokenUsage(prompt_tokens=500, cached_tokens=400, completion_tokens=40)
    assert extract_openai_usage({"choices": []}) is None


# 测试 ModelClient 在请求后自动记录响应中的 last_usage
def test_model_client_records_last_usage() -> None:
    # 模拟接口返回带 DeepSeek 缓存命中信息的 SSE 响应流
    def fake_stream(url: str, payload: dict[str, object], headers: dict[str, str]) -> Iterator[str]:
        yield 'data: {"choices":[{"delta":{"role":"assistant","content":"ok"}}]}\n'
        yield 'data: {"choices":[],"usage":{"prompt_tokens":1200,"completion_tokens":30,"prompt_cache_hit_tokens":1024}}\n'
        yield "data: [DONE]\n"

    reg = ApiProviderRegistry()
    reg.register(OpenAICompletionsProvider(post_stream=fake_stream))
    client = ModelClient(
        profile=ModelProfile(
            name="demo",
            api="openai-completions",
            base_url="https://example.com/v1",
            model="demo-model",
            api_key="sk-test",
        ),
        registry=reg,
    )
    reply = client.complete(Context(messages=(UserMessage("hi"),)))
    assert reply.text == "ok"
    assert client.last_usage == TokenUsage(prompt_tokens=1200, cached_tokens=1024, completion_tokens=30)


# 测试本地前缀复用率：追加式历史高命中、改动 system 前缀瞬间暴露失效、多会话隔离与异常安全
def test_cache_monitor_prefix_reuse_regression_and_session_isolation() -> None:
    monitor = CacheMonitor()
    tools = (ToolSchema(name="search_markdown", description="检索", parameters={"query": "词"}),)
    sys_prompt = "你是个人助手。" + "规则说明" * 80

    llm = UsageScriptedLLM(
        [
            (AssistantMessage.from_text("回答1"), TokenUsage(prompt_tokens=1000, cached_tokens=0, completion_tokens=20)),
            (AssistantMessage.from_text("回答2"), TokenUsage(prompt_tokens=1100, cached_tokens=960, completion_tokens=25)),
            (AssistantMessage.from_text("回答3"), TokenUsage(prompt_tokens=1120, cached_tokens=0, completion_tokens=20)),
        ]
    )
    tracked_a = monitor.wrap(llm, session_key="chat-a")

    # 第 1 次冷启动：prefix_reuse_rate 为 None
    c1 = Context(system_prompt=sys_prompt, messages=(UserMessage("第一问"),), tools=tools)
    tracked_a.complete(c1)
    assert monitor.records[0].prefix_reuse_rate is None

    # 第 2 次纯追加：前文完全复用，本地前缀复用率应很高（> 0.8）
    c2 = Context(
        system_prompt=sys_prompt,
        messages=(UserMessage("第一问"), AssistantMessage.from_text("回答1"), UserMessage("第二问")),
        tools=tools,
    )
    tracked_a.complete(c2)
    assert monitor.records[1].prefix_chars == len(serialize_context(c1))
    assert monitor.records[1].prefix_reuse_rate is not None
    assert monitor.records[1].prefix_reuse_rate > 0.8

    # 第 3 次人为改动 system 开头：应立刻捕获前缀失效（复用率跌到极低）
    c3 = Context(
        system_prompt="【时间变了】你是个人助手。" + "规则说明" * 80,
        messages=(UserMessage("第三问"),),
        tools=tools,
    )
    tracked_a.complete(c3)
    assert monitor.records[2].prefix_reuse_rate is not None
    assert monitor.records[2].prefix_reuse_rate < 0.3

    # 未配置模型抛异常时，不应污染已有监控记录
    tracked_fail = monitor.wrap(NoopModelClient(), session_key="chat-a")
    with pytest.raises(LLMNotConfigured):
        tracked_fail.complete(c2)
    assert len(monitor.records) == 3

    # 另一会话 chat-b 独立冷启动，不与 chat-a 串前缀
    monitor.record("chat-b", serialize_context(c1), None)
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

    cold_text = format_cache_summary(
        {
            "llm_calls": 1,
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "server_cache_hit_rate": None,
            "mean_prefix_reuse_rate": None,
        }
    )
    assert "服务端未返回 usage" in cold_text
    assert "首次冷启动" in cold_text


# 测试多线程共享 ModelClient 与 CacheMonitor 时 last_usage 线程隔离、统计无丢数且失败自动清零
def test_model_client_last_usage_and_cache_monitor_thread_safe() -> None:
    n_threads = 8
    enter_barrier = threading.Barrier(n_threads, timeout=5.0)
    read_barrier = threading.Barrier(n_threads, timeout=5.0)
    should_fail = threading.local()

    # 强制 8 个并发请求交叠返回各线程专属 usage，验证不互相覆盖
    def concurrent_stream(
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> Iterator[str]:
        if getattr(should_fail, "flag", False):
            raise RuntimeError("boom")
        msgs = payload["messages"]
        assert isinstance(msgs, list)
        tid = int(str(msgs[-1]["content"]).split("-")[-1])
        enter_barrier.wait()
        yield f'data: {json.dumps({"choices": [{"delta": {"content": f"reply-{tid}"}}]})}\n'
        yield (
            "data: "
            + json.dumps(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 1000 + tid,
                        "prompt_cache_hit_tokens": 100 + tid,
                        "completion_tokens": 10 + tid,
                    },
                }
            )
            + "\n"
        )
        yield "data: [DONE]\n"

    reg = ApiProviderRegistry()
    reg.register(OpenAICompletionsProvider(post_stream=concurrent_stream))
    client = ModelClient(
        profile=ModelProfile(
            name="demo",
            api="openai-completions",
            base_url="https://example.com/v1",
            model="demo-model",
            api_key="sk-test",
        ),
        registry=reg,
    )

    # 先测裸 ModelClient：全部 8 线程均写完后再同时读 last_usage，绝不串台
    def worker_direct(tid: int) -> TokenUsage | None:
        reply = client.complete(Context(messages=(UserMessage(f"thread-{tid}"),)))
        assert reply.text == f"reply-{tid}"
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

    # 再测共享 CacheMonitor + TrackingModelClient 并发记录，全量统计零丢失、会话键不混用
    monitor = CacheMonitor()

    # 单线程并发触发 TrackingModelClient.complete 与 monitor.record
    def worker_tracked(tid: int) -> TokenUsage | None:
        tracked = monitor.wrap(client, session_key=f"chat-{tid}")
        tracked.complete(Context(messages=(UserMessage(f"thread-{tid}"),)))
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

    # 同一线程随后一次调用失败抛异常时，last_usage 必须清零而非残留旧值
    enter_barrier = threading.Barrier(1, timeout=5.0)
    client.complete(Context(messages=(UserMessage("thread-0"),)))
    assert client.last_usage == TokenUsage(prompt_tokens=1000, cached_tokens=100, completion_tokens=10)
    should_fail.flag = True
    with pytest.raises(RuntimeError, match="boom"):
        client.complete(Context(messages=(UserMessage("thread-0"),)))
    assert client.last_usage is None


# 测试 CacheMonitor 的有界记录窗口、MRU 会话淘汰、单调游标增量汇总与非法容量校验
def test_cache_monitor_bounded_eviction_and_exact_summaries() -> None:
    for bad_val in (0, -1, True):
        with pytest.raises(ValueError):
            CacheMonitor(max_records=bad_val)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            CacheMonitor(max_sessions=bad_val)  # type: ignore[arg-type]

    unbounded = CacheMonitor()
    bounded = CacheMonitor(max_records=5, max_sessions=3)

    sequence = [
        ("s1", "sys\nu1\n", TokenUsage(prompt_tokens=100, cached_tokens=0, completion_tokens=10)),
        ("s2", "sys\nu2\n", TokenUsage(prompt_tokens=120, cached_tokens=20, completion_tokens=12)),
        ("s3", "sys\nu3\n", TokenUsage(prompt_tokens=140, cached_tokens=40, completion_tokens=14)),
        ("s1", "sys\nu1\na1\nu1b\n", TokenUsage(prompt_tokens=160, cached_tokens=96, completion_tokens=16)),
        ("s4", "sys\nu4\n", TokenUsage(prompt_tokens=110, cached_tokens=0, completion_tokens=11)),
        ("s4", "sys\nu4\na4\nu4b\n", TokenUsage(prompt_tokens=150, cached_tokens=90, completion_tokens=15)),
        ("s3", "sys\nu3\na3\nu3b\n", TokenUsage(prompt_tokens=180, cached_tokens=120, completion_tokens=18)),
        ("s1", "sys\nu1\na1\nu1b\na1b\nu1c\n", TokenUsage(prompt_tokens=210, cached_tokens=160, completion_tokens=20)),
        ("s5", "sys\nu5\n", None),
    ]
    for sk, serialized, usage in sequence:
        unbounded.record(sk, serialized, usage)
        bounded.record(sk, serialized, usage)

    assert len(bounded._last_serialized) == 3
    assert set(bounded._last_serialized) == {"s3", "s1", "s5"}
    assert len(bounded.records) == 5
    assert bounded.snapshot_index() == 9

    turn_idx = bounded.snapshot_index()
    assert turn_idx == unbounded.snapshot_index() == 9
    tail_sequence = [
        ("s5", "sys\nu5\na5\nu5b\n", TokenUsage(prompt_tokens=170, cached_tokens=100, completion_tokens=17)),
        ("s1", "sys\nu1\na1\nu1b\na1b\nu1c\na1c\nu1d\n", TokenUsage(prompt_tokens=240, cached_tokens=200, completion_tokens=22)),
        ("s3", "sys\nu3\na3\nu3b\na3b\nu3c\n", TokenUsage(prompt_tokens=200, cached_tokens=150, completion_tokens=19)),
    ]
    for sk, serialized, usage in tail_sequence:
        unbounded.record(sk, serialized, usage)
        bounded.record(sk, serialized, usage)

    assert len(bounded._last_serialized) == 3
    assert len(bounded.records) == 5
    assert bounded.snapshot_index() == 12
    assert isinstance(bounded.records, list)
    assert [r.session_key for r in bounded.records] == ["s1", "s5", "s5", "s1", "s3"]

    assert bounded.summary() == unbounded.summary()
    assert bounded.summary(since=0) == unbounded.summary(since=0)
    assert bounded.summary()["llm_calls"] == 12
    assert bounded.summary(since=turn_idx) == unbounded.summary(since=turn_idx)
    assert bounded.summary(since=turn_idx)["llm_calls"] == 3
    assert bounded.summary(since=bounded.snapshot_index())["llm_calls"] == 0

    revisited = bounded.record("s2", "sys\nu2\na2\n", TokenUsage(prompt_tokens=130, cached_tokens=0, completion_tokens=13))
    assert revisited.prefix_reuse_rate is None
    assert len(bounded._last_serialized) == 3
    assert len(bounded.records) == 5


# 测试 CacheMonitor.summary 按 session_key 精确切片隔离并发会话统计，不串台也不破坏全局汇总
def test_cache_monitor_summary_filters_by_session_key() -> None:
    monitor = CacheMonitor(max_records=10, max_sessions=10)
    monitor.record("chat-a", "sys\na1\n", TokenUsage(prompt_tokens=100, cached_tokens=0, completion_tokens=10))
    monitor.record("chat-b", "sys\nb1\n", TokenUsage(prompt_tokens=900, cached_tokens=500, completion_tokens=90))

    start_idx = monitor.snapshot_index()
    monitor.record("chat-a", "sys\na1\na2\n", TokenUsage(prompt_tokens=200, cached_tokens=100, completion_tokens=20))
    monitor.record("chat-b", "sys\nb1\nb2\n", TokenUsage(prompt_tokens=1000, cached_tokens=800, completion_tokens=50))
    monitor.record("chat-a", "sys\na1\na2\na3\n", TokenUsage(prompt_tokens=300, cached_tokens=240, completion_tokens=30))

    all_global = monitor.summary()
    assert all_global["llm_calls"] == 5
    assert all_global["prompt_tokens"] == 2500

    all_slice = monitor.summary(since=start_idx)
    assert all_slice["llm_calls"] == 3
    assert all_slice["prompt_tokens"] == 1500

    only_a_full = monitor.summary(session_key="chat-a")
    assert only_a_full["llm_calls"] == 3
    assert only_a_full["prompt_tokens"] == 600
    assert only_a_full["cached_tokens"] == 340
    assert only_a_full["completion_tokens"] == 60
    assert only_a_full["warm_calls"] == 2

    only_a_turn = monitor.summary(since=start_idx, session_key="chat-a")
    assert only_a_turn["llm_calls"] == 2
    assert only_a_turn["prompt_tokens"] == 500
    assert only_a_turn["cached_tokens"] == 340
    assert only_a_turn["completion_tokens"] == 50
    assert only_a_turn["warm_calls"] == 2

    only_b_turn = monitor.summary(since=start_idx, session_key="chat-b")
    assert only_b_turn["llm_calls"] == 1
    assert only_b_turn["prompt_tokens"] == 1000

    empty_s = monitor.summary(since=0, session_key="unknown-session")
    assert empty_s["llm_calls"] == 0
    assert empty_s["prompt_tokens"] == 0
