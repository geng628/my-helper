"""Layer 1 OpenAI Chat Completions 协议翻译器（api = "openai-completions"）：双向翻译中立 Context 与 12 种统一事件。"""
from __future__ import annotations

import threading
import time
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from urllib.error import HTTPError

from personal_assistant.ai.protocol import (
    AssistantBlock,
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    DoneReason,
    ModelProfile,
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
    ThinkingEnd,
    ThinkingStart,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolResultMessage,
    ToolSchema,
    UserMessage,
    clamp_thinking_level,
)
from personal_assistant.ai.providers.http import (
    PostStream,
    default_post_stream,
    format_http_error,
    iter_sse_events,
    non_negative_int,
    redact_error_tokens,
)

_STREAM_OPTIONS_MARKERS = ("stream_options", "include_usage")
_CONTEXT_LIMIT_MARKERS = (
    "context_length",
    "maximum context",
    "too many tokens",
    "token limit",
    "max_tokens",
    "maximum token",
)


# 从 OpenAI / DeepSeek / 通义千问等兼容响应块中提取 Token 与缓存命中统计
def extract_openai_usage(response: Mapping[str, object]) -> TokenUsage | None:
    raw = response.get("usage")
    if not isinstance(raw, Mapping):
        return None
    prompt_tokens = non_negative_int(raw.get("prompt_tokens") or raw.get("input_tokens"))
    completion_tokens = non_negative_int(raw.get("completion_tokens") or raw.get("output_tokens"))

    cached_raw: object = raw.get("prompt_cache_hit_tokens")
    if cached_raw is None:
        details = raw.get("prompt_tokens_details")
        if isinstance(details, Mapping):
            cached_raw = details.get("cached_tokens")
    if cached_raw is None:
        cached_raw = raw.get("cache_read_input_tokens")
    cached_tokens = non_negative_int(cached_raw)
    if prompt_tokens > 0:
        cached_tokens = min(cached_tokens, prompt_tokens)

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        completion_tokens=completion_tokens,
    )


# 判断 HTTP 400 错误文本是否明确包含 stream_options / include_usage 不兼容标记（排除上下文超长错误）
def _is_stream_options_incompat(error_text: str) -> bool:
    lowered = error_text.lower()
    if any(marker in lowered for marker in _CONTEXT_LIMIT_MARKERS):
        return False
    return any(marker in lowered for marker in _STREAM_OPTIONS_MARKERS)


# 将中立 ToolSchema 翻译为 OpenAI Chat Completions Function Calling JSON 字典（仅限本翻译器内部使用）
def _translate_tool_schema(schema: ToolSchema) -> dict[str, object]:
    properties: dict[str, dict[str, object]] = {}
    for param, desc in schema.parameters.items():
        prop: dict[str, object] = {"type": "string", "description": desc}
        if schema.param_enums and param in schema.param_enums:
            prop["enum"] = list(schema.param_enums[param])
        properties[param] = prop
    return {
        "type": "function",
        "function": {
            "name": schema.name,
            "description": schema.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(schema.required_names),
                "additionalProperties": False,
            },
        },
    }


# 将 OpenAI finish_reason 规范化映射为中立 DoneReason
def _map_finish_reason(raw_reason: str | None, *, has_tool_calls: bool) -> DoneReason:
    if raw_reason == "length":
        return "length"
    if raw_reason in {"tool_calls", "function_call"}:
        return "toolUse"
    if raw_reason in {"stop", None}:
        return "toolUse" if has_tool_calls else "stop"
    return "toolUse" if has_tool_calls else "stop"


# 流式 SSE 增量工具调用分片累加器
@dataclass
class _ToolCallAcc:
    id: str = ""
    name: str = ""
    args: list[str] = field(default_factory=list)


# OpenAI Chat Completions 翻译器：提供清晰子类扩展点供内建推理方言（openai_reasoning）与外部插件复用
@dataclass
class OpenAICompletionsProvider:
    api: str = "openai-completions"
    post_stream: PostStream = field(default=default_post_stream, kw_only=True)
    _compat_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _stream_usage_unsupported: set[tuple[str, str]] = field(default_factory=set, init=False, repr=False)

    # 返回流式解析时需翻译为 ThinkingStart/Delta/End 的 delta 思维链字段名（默认为空，子类可重写）
    def _thinking_delta_fields(self) -> tuple[str, ...]:
        return ()

    # 将单条中立 AssistantMessage 翻译为 OpenAI assistant 消息字典（子类可按 is_current_turn 追加思维链字段）
    def _serialize_assistant_message(
        self,
        message: AssistantMessage,
        *,
        is_current_turn: bool,
    ) -> dict[str, object]:
        del is_current_turn
        text = message.text
        tool_calls = message.tool_calls
        wire: dict[str, object] = {
            "role": "assistant",
            "content": text if (text or not tool_calls) else None,
        }
        if tool_calls:
            wire["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in tool_calls
            ]
        return wire

    # 返回附加到 OpenAI 请求体顶层的采样/思考/上限参数（子类可按模型协议边界重写）
    def _extra_payload(
        self,
        model: ModelProfile,
        options: StreamOptions | None,
    ) -> dict[str, object]:
        extra: dict[str, object] = {}
        temp = options.temperature if (options is not None and options.temperature is not None) else model.temperature
        extra["temperature"] = temp
        max_tok = options.max_tokens if (options is not None and options.max_tokens is not None) else model.max_tokens
        if max_tok is not None:
            extra["max_tokens"] = max_tok
        reasoning = options.reasoning if (options is not None and options.reasoning is not None) else model.reasoning
        if reasoning != "off":
            extra["reasoning_effort"] = clamp_thinking_level(reasoning, ("low", "medium", "high"))
        return extra

    # 将中立 Context 完整翻译为 OpenAI Chat Completions messages 列表
    def _serialize_messages(self, context: Context) -> list[dict[str, object]]:
        wire: list[dict[str, object]] = []
        if context.system_prompt:
            wire.append({"role": "system", "content": context.system_prompt})
        last_user_idx = -1
        for idx, msg in enumerate(context.messages):
            if isinstance(msg, UserMessage):
                last_user_idx = idx
        for idx, msg in enumerate(context.messages):
            if isinstance(msg, UserMessage):
                wire.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AssistantMessage):
                wire.append(self._serialize_assistant_message(msg, is_current_turn=(idx > last_user_idx)))
            elif isinstance(msg, ToolResultMessage):
                wire.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                )
        return wire

    # 查询目标端点是否已记录不支持 stream_options
    def _supports_stream_usage(self, url: str, model_id: str) -> bool:
        with self._compat_lock:
            return (url, model_id) not in self._stream_usage_unsupported

    # 标记目标端点不支持 stream_options，后续调用直接跳过该字段
    def _mark_stream_usage_unsupported(self, url: str, model_id: str) -> None:
        with self._compat_lock:
            self._stream_usage_unsupported.add((url, model_id))

    # 组装 /chat/completions 的请求地址、JSON 载荷与鉴权头
    def _build_request(
        self,
        model: ModelProfile,
        context: Context,
        options: StreamOptions | None,
    ) -> tuple[str, dict[str, object], dict[str, str]]:
        url = f"{model.base_url.rstrip('/')}/chat/completions"
        payload: dict[str, object] = {
            "model": model.model,
            "messages": self._serialize_messages(context),
            **self._extra_payload(model, options),
            "stream": True,
        }
        if self._supports_stream_usage(url, model.model):
            payload["stream_options"] = {"include_usage": True}
        if context.tools:
            payload["tools"] = [_translate_tool_schema(schema) for schema in context.tools]
            payload["tool_choice"] = "auto"
        headers = {"Authorization": f"Bearer {model.api_key}"}
        return url, payload, headers

    # 打开 OpenAI SSE 行流；首包前若明确因 stream_options/include_usage 返回 400 则降级重试一次
    def _open_stream(
        self,
        url: str,
        model_id: str,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> Iterator[str]:
        yielded_any = False
        first_iter: Iterator[str] | None = None
        try:
            try:
                first_iter = self.post_stream(url, payload, headers)
                for line in first_iter:
                    yielded_any = True
                    yield line
                return
            finally:
                if first_iter is not None:
                    close_fn = getattr(first_iter, "close", None)
                    if callable(close_fn):
                        close_fn()
        except HTTPError as error:
            formatted_error, error_text = format_http_error(error)
            if (
                error.code != 400
                or yielded_any
                or "stream_options" not in payload
                or not _is_stream_options_incompat(error_text)
            ):
                if formatted_error is error:
                    raise
                raise formatted_error from error
            self._mark_stream_usage_unsupported(url, model_id)
            fallback_payload = {k: v for k, v in payload.items() if k != "stream_options"}
            second_iter: Iterator[str] | None = None
            try:
                try:
                    second_iter = self.post_stream(url, fallback_payload, headers)
                    yield from second_iter
                finally:
                    if second_iter is not None:
                        close_fn2 = getattr(second_iter, "close", None)
                        if callable(close_fn2):
                            close_fn2()
            except HTTPError as retry_error:
                formatted_retry, _ = format_http_error(retry_error)
                if formatted_retry is retry_error:
                    raise
                raise formatted_retry from retry_error

    # 执行完整 5 步翻译管道：发请求、读 SSE、按块状态机吐出 12 种统一事件，异常收敛为 StreamError
    def stream(
        self,
        model: ModelProfile,
        context: Context,
        options: StreamOptions | None = None,
    ) -> Iterator[AssistantMessageEvent]:
        t0 = time.perf_counter()
        blocks: list[AssistantBlock] = []
        usage: TokenUsage | None = None
        ttft_ms: float | None = None
        finish_reason: str | None = None
        thinking_fields = tuple(dict.fromkeys(self._thinking_delta_fields()))

        thinking_idx: int | None = None
        thinking_open = False
        thinking_buf: list[str] = []

        text_idx: int | None = None
        text_open = False
        text_buf: list[str] = []

        tool_slots: dict[int, tuple[int, _ToolCallAcc]] = {}

        # 生成当前不可变部分消息快照
        def snapshot() -> AssistantMessage:
            return AssistantMessage(blocks=tuple(blocks))

        # 闭合当前打开的思维链内容块并产出 ThinkingEnd
        def close_thinking() -> Iterator[AssistantMessageEvent]:
            nonlocal thinking_open
            if thinking_open and thinking_idx is not None:
                full = "".join(thinking_buf)
                blocks[thinking_idx] = ThinkingBlock(text=full)
                thinking_open = False
                yield ThinkingEnd(content_index=thinking_idx, text=full, partial=snapshot())

        # 闭合当前打开的正文内容块并产出 TextEnd
        def close_text() -> Iterator[AssistantMessageEvent]:
            nonlocal text_open
            if text_open and text_idx is not None:
                full = "".join(text_buf)
                blocks[text_idx] = TextBlock(text=full)
                text_open = False
                yield TextEnd(content_index=text_idx, text=full, partial=snapshot())

        yield StreamStart(partial=snapshot())

        line_iter: Iterator[str] | None = None
        try:
            url, payload, headers = self._build_request(model, context, options)
            line_iter = self._open_stream(url, model.model, payload, headers)
            for _, data_str in iter_sse_events(line_iter):
                if data_str == "[DONE]":
                    break
                chunk = json.loads(data_str)
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("error") is not None:
                    raise RuntimeError(f"流式响应返回错误：{chunk['error']}")
                usage = extract_openai_usage(chunk) or usage
                choices = chunk.get("choices") or []
                first_choice = choices[0] if choices and isinstance(choices[0], dict) else None
                if first_choice is not None:
                    raw_finish = first_choice.get("finish_reason")
                    if isinstance(raw_finish, str) and raw_finish:
                        finish_reason = raw_finish
                delta = first_choice.get("delta") if first_choice is not None else None
                if not isinstance(delta, dict):
                    continue

                for field_name in thinking_fields:
                    thinking_piece = delta.get(field_name)
                    if isinstance(thinking_piece, str) and thinking_piece:
                        if ttft_ms is None:
                            ttft_ms = round((time.perf_counter() - t0) * 1000, 3)
                        yield from close_text()
                        if not thinking_open:
                            thinking_idx = len(blocks)
                            thinking_buf = []
                            blocks.append(ThinkingBlock(text=""))
                            thinking_open = True
                            yield ThinkingStart(content_index=thinking_idx, partial=snapshot())
                        thinking_buf.append(thinking_piece)
                        assert thinking_idx is not None
                        blocks[thinking_idx] = ThinkingBlock(text="".join(thinking_buf))
                        yield ThinkingDelta(
                            content_index=thinking_idx,
                            text=thinking_piece,
                            partial=snapshot(),
                        )

                piece = delta.get("content")
                if isinstance(piece, str) and piece:
                    if ttft_ms is None:
                        ttft_ms = round((time.perf_counter() - t0) * 1000, 3)
                    yield from close_thinking()
                    if not text_open:
                        text_idx = len(blocks)
                        text_buf = []
                        blocks.append(TextBlock(text=""))
                        text_open = True
                        yield TextStart(content_index=text_idx, partial=snapshot())
                    text_buf.append(piece)
                    assert text_idx is not None
                    blocks[text_idx] = TextBlock(text="".join(text_buf))
                    yield TextDelta(content_index=text_idx, text=piece, partial=snapshot())

                raw_tool_calls = delta.get("tool_calls") or ()
                if isinstance(raw_tool_calls, (list, tuple)):
                    valid_tcs = [tc for tc in raw_tool_calls if isinstance(tc, dict)]
                    if valid_tcs:
                        yield from close_thinking()
                        yield from close_text()
                    for pos, tc in enumerate(valid_tcs):
                        if ttft_ms is None:
                            ttft_ms = round((time.perf_counter() - t0) * 1000, 3)
                        raw_idx = tc.get("index")
                        slot_idx = (
                            raw_idx
                            if (isinstance(raw_idx, int) and not isinstance(raw_idx, bool) and raw_idx >= 0)
                            else pos
                        )
                        tc_id = tc["id"] if isinstance(tc.get("id"), str) and tc["id"] else ""
                        fn = tc.get("function") or {}
                        fn_name = (
                            fn["name"]
                            if isinstance(fn, dict) and isinstance(fn.get("name"), str) and fn["name"]
                            else ""
                        )
                        fn_args = (
                            fn["arguments"]
                            if isinstance(fn, dict) and isinstance(fn.get("arguments"), str) and fn["arguments"]
                            else ""
                        )
                        if slot_idx not in tool_slots:
                            acc = _ToolCallAcc(id=tc_id or f"call_{slot_idx}", name=fn_name)
                            block_idx = len(blocks)
                            blocks.append(ToolCall(id=acc.id, name=acc.name, arguments=""))
                            tool_slots[slot_idx] = (block_idx, acc)
                            yield ToolCallStart(
                                content_index=block_idx,
                                call_id=acc.id,
                                name=acc.name,
                                partial=snapshot(),
                            )
                        else:
                            block_idx, acc = tool_slots[slot_idx]
                            if tc_id:
                                acc.id = tc_id
                            if fn_name:
                                acc.name = f"{acc.name}{fn_name}"
                        if fn_args:
                            acc.args.append(fn_args)
                            blocks[block_idx] = ToolCall(
                                id=acc.id,
                                name=acc.name,
                                arguments="".join(acc.args),
                            )
                            yield ToolCallDelta(
                                content_index=block_idx,
                                delta=fn_args,
                                partial=snapshot(),
                            )
                        else:
                            blocks[block_idx] = ToolCall(
                                id=acc.id,
                                name=acc.name,
                                arguments="".join(acc.args),
                            )

            yield from close_thinking()
            yield from close_text()
            for slot_idx, (block_idx, acc) in sorted(tool_slots.items(), key=lambda item: item[1][0]):
                final_call = ToolCall(
                    id=acc.id or f"call_{slot_idx}",
                    name=acc.name,
                    arguments="".join(acc.args) or "{}",
                )
                blocks[block_idx] = final_call
                yield ToolCallEnd(content_index=block_idx, tool_call=final_call, partial=snapshot())

            done_reason = _map_finish_reason(finish_reason, has_tool_calls=bool(tool_slots))
            final_msg = AssistantMessage(blocks=tuple(blocks), stop_reason=done_reason, usage=usage)
            yield StreamDone(
                reason=done_reason,
                message=final_msg,
                usage=usage,
                ttft_ms=ttft_ms,
                latency_ms=round((time.perf_counter() - t0) * 1000, 3),
            )
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, HTTPError):
                _, diag = format_http_error(exc)
                err_msg = f"HTTP {exc.code}: {diag}" if diag else f"HTTP {exc.code}"
            else:
                err_msg = redact_error_tokens(f"{type(exc).__name__}: {exc}")
            err_msg_obj = AssistantMessage(
                blocks=tuple(blocks),
                stop_reason="error",
                usage=usage,
                error_message=err_msg,
            )
            yield StreamError(
                reason="error",
                message=err_msg_obj,
                error_message=err_msg,
                latency_ms=round((time.perf_counter() - t0) * 1000, 3),
            )
        finally:
            if line_iter is not None:
                close_fn = getattr(line_iter, "close", None)
                if callable(close_fn):
                    close_fn()
