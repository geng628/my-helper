"""推理模型协议插件（Slot 4 LLMProviderProtocol）：支持 deepseek-reasoner / R1 / o1/o3，微内核零改动接入。"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from urllib.error import HTTPError

from personal_assistant.agent.plugins import Plugin
from personal_assistant.ai import (
    LLMClient,
    LLMStreamEvent,
    PostJson,
    PostStream,
    StreamDone,
    TextDelta,
    TokenUsage,
    default_post_json,
    default_post_stream,
    extract_usage,
)

PROVIDER_NAME = "openai_reasoning"


# 为推理模型准备请求消息副本：仅跨轮（最后一条 user 之前）剥离历史 reasoning_content，且不改动原消息
def _prepare_wire_messages(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    last_user_idx = -1
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            last_user_idx = i

    return [
        {k: v for k, v in msg.items() if k != "reasoning_content"}
        if i < last_user_idx and "reasoning_content" in msg
        else msg
        for i, msg in enumerate(messages)
    ]


# 解析推理模型 SSE 流：记录 reasoning_content 供轮内回传、以首包计入 ttft_ms、仅将 content 吐出为 TextDelta
def _consume_reasoning_sse(lines: Iterable[str], t0: float) -> Iterator[LLMStreamEvent]:
    role = "assistant"
    parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_acc: dict[int, dict[str, object]] = {}
    usage: TokenUsage | None = None
    ttft_ms: float | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if not data_str:
            continue
        if data_str == "[DONE]":
            break
        chunk = json.loads(data_str)
        if not isinstance(chunk, dict):
            continue
        if chunk.get("error") is not None:
            raise RuntimeError(f"流式响应返回错误：{chunk['error']}")
        usage = extract_usage(chunk) or usage
        choices = chunk.get("choices") or []
        delta = choices[0].get("delta") if choices and isinstance(choices[0], dict) else None
        if not isinstance(delta, dict):
            continue
        if isinstance(delta.get("role"), str) and delta["role"]:
            role = str(delta["role"])

        rc = delta.get("reasoning_content")
        if isinstance(rc, str) and rc:
            if ttft_ms is None:
                ttft_ms = round((time.perf_counter() - t0) * 1000, 3)
            reasoning_parts.append(rc)

        piece = delta.get("content")
        if isinstance(piece, str) and piece:
            if ttft_ms is None:
                ttft_ms = round((time.perf_counter() - t0) * 1000, 3)
            parts.append(piece)
            yield TextDelta(text=piece)

        for tc in delta.get("tool_calls") or ():
            if not isinstance(tc, dict):
                continue
            if ttft_ms is None:
                ttft_ms = round((time.perf_counter() - t0) * 1000, 3)
            raw_idx = tc.get("index")
            idx = raw_idx if isinstance(raw_idx, int) and not isinstance(raw_idx, bool) else 0
            acc = tool_acc.setdefault(idx, {"id": "", "type": "function", "name": "", "args": []})
            if isinstance(tc.get("id"), str) and tc["id"]:
                acc["id"] = tc["id"]
            if isinstance(tc.get("type"), str) and tc["type"]:
                acc["type"] = tc["type"]
            fn = tc.get("function") or {}
            if isinstance(fn, dict):
                if isinstance(fn.get("name"), str) and fn["name"]:
                    acc["name"] = f"{acc['name']}{fn['name']}"
                if isinstance(fn.get("arguments"), str) and fn["arguments"]:
                    acc["args"].append(fn["arguments"])  # type: ignore[attr-defined]

    full_text = "".join(parts)
    reasoning_text = "".join(reasoning_parts)
    calls = [
        {"id": a["id"], "type": a["type"], "function": {"name": a["name"], "arguments": "".join(a["args"])}}
        for _, a in sorted(tool_acc.items())
    ]
    msg: dict[str, object] = {"role": role, "content": full_text if (full_text or not calls) else None}
    if calls:
        msg["tool_calls"] = calls
    if reasoning_text or calls:
        msg["reasoning_content"] = reasoning_text
    yield StreamDone(
        message=msg,
        usage=usage,
        ttft_ms=ttft_ms,
        latency_ms=round((time.perf_counter() - t0) * 1000, 3),
    )


# 原生推理模型客户端：不发送 temperature 参数，按单轮生命周期管理 reasoning_content 的透传与剥离
@dataclass
class OpenAIReasoningLLM(LLMClient):
    api_key: str
    base_url: str
    model: str = "deepseek-reasoner"
    post_json: PostJson = default_post_json
    post_stream: PostStream = default_post_stream
    _usage_local: threading.local = field(default_factory=threading.local, init=False, repr=False)

    # 读取当前线程最近一次调用记录的 Token 统计（避免多线程共享客户端串台）
    @property
    def last_usage(self) -> TokenUsage | None:
        return getattr(self._usage_local, "value", None)

    # 设置当前线程最近一次调用的 Token 统计
    @last_usage.setter
    def last_usage(self, value: TokenUsage | None) -> None:
        self._usage_local.value = value

    # 组装推理模型请求体（省去不兼容的 temperature，并按轮次边界处理 reasoning_content）
    def _build_request(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None,
        *,
        stream: bool,
    ) -> tuple[str, dict[str, object], dict[str, str]]:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": _prepare_wire_messages(messages),
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return (
            f"{self.base_url.rstrip('/')}/chat/completions",
            payload,
            {"Authorization": f"Bearer {self.api_key}"},
        )

    # 同步调用推理接口并返回带 reasoning_content（工具续传时规范为字符串）的 assistant 消息
    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        url, payload, headers = self._build_request(messages, tools, stream=False)
        self.last_usage = None
        response = self.post_json(url, payload, headers)
        msg = dict(response["choices"][0]["message"])
        if msg.get("tool_calls") and msg.get("reasoning_content") is None:
            msg["reasoning_content"] = ""
        elif msg.get("reasoning_content") is None:
            msg.pop("reasoning_content", None)
        self.last_usage = extract_usage(response)
        return msg

    # 流式调用推理接口（兼容不支持 stream_options 返回 HTTP 400 的端点）
    def stream(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> Iterator[LLMStreamEvent]:
        url, payload, headers = self._build_request(messages, tools, stream=True)
        self.last_usage = None
        t0 = time.perf_counter()

        # 打开 SSE 响应行流；若首包前因 stream_options 返回 HTTP 400 则去掉后重试一次
        def _open_lines() -> Iterator[str]:
            yielded_any = False
            try:
                for line in self.post_stream(url, payload, headers):
                    yielded_any = True
                    yield line
            except HTTPError as error:
                if error.code != 400 or yielded_any or "stream_options" not in payload:
                    raise
                fallback_payload = {k: v for k, v in payload.items() if k != "stream_options"}
                yield from self.post_stream(url, fallback_payload, headers)

        for event in _consume_reasoning_sse(_open_lines(), t0):
            if isinstance(event, StreamDone):
                self.last_usage = event.usage
            yield event


# 实现内层 LLMProviderProtocol 协议的推理模型构造器
class OpenAIReasoningProvider:
    name = PROVIDER_NAME

    # 按统一环境变量/Profile 映射构建 OpenAIReasoningLLM 实例
    def build(self, env: Mapping[str, str]) -> LLMClient:
        return OpenAIReasoningLLM(
            api_key=env.get("LLM_API_KEY", "").strip(),
            base_url=env.get("LLM_BASE_URL", "").strip() or "https://api.deepseek.com/v1",
            model=env.get("LLM_MODEL", "").strip() or "deepseek-reasoner",
        )


# 推理模型主板插件：通过 Slot 4（LLMProviderProtocol）注册 openai_reasoning，实现微内核零改动扩展
class ReasoningProviderPlugin(Plugin):
    plugin_id = PROVIDER_NAME

    # 初始化生命周期：向主板注册 OpenAIReasoningProvider
    def initialize(self) -> None:
        self.context.register_provider(OpenAIReasoningProvider())
