"""Layer 1 原生 Anthropic Messages API 翻译器（api = "anthropic-messages"）：支持 Prompt Caching 与扩展思考。"""
from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from urllib.error import HTTPError

from personal_assistant.ai.protocol import (
    THINKING_BUDGETS,
    AssistantBlock,
    AssistantMessage,
    AssistantMessageEvent,
    CacheRetention,
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

ANTHROPIC_API_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096


# 按 CacheRetention 生成 Anthropic Prompt Caching 断点标记字典
def _cache_control(retention: CacheRetention) -> dict[str, object] | None:
    if retention == "none":
        return None
    if retention == "long":
        return {"type": "ephemeral", "ttl": "1h"}
    return {"type": "ephemeral"}


# 将中立 ToolSchema 翻译为 Anthropic 工具定义对象（name / description / input_schema）
def _translate_tool_schema(schema: ToolSchema) -> dict[str, object]:
    properties: dict[str, dict[str, object]] = {}
    for param, desc in schema.parameters.items():
        prop: dict[str, object] = {"type": "string", "description": desc}
        if schema.param_enums and param in schema.param_enums:
            prop["enum"] = list(schema.param_enums[param])
        properties[param] = prop
    return {
        "name": schema.name,
        "description": schema.description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(schema.required_names),
            "additionalProperties": False,
        },
    }


# 将 JSON 字符串工具参数解析为对象字典（异常时安全回退为空对象）
def _parse_tool_input(raw_arguments: str) -> dict[str, object]:
    try:
        parsed = json.loads(raw_arguments or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


# 将中立消息序列翻译为 Anthropic messages 列表（跳过空文本/空助手消息，合并相邻同角色并按策略打缓存断点）
def _serialize_messages(
    context: Context,
    cache_control: dict[str, object] | None,
) -> list[dict[str, object]]:
    wire: list[dict[str, object]] = []
    for msg in context.messages:
        if isinstance(msg, UserMessage):
            if not msg.content:
                if wire and wire[-1]["role"] == "user":
                    continue
                block: dict[str, object] = {"type": "text", "text": " "}
            else:
                block = {"type": "text", "text": msg.content}
            if wire and wire[-1]["role"] == "user":
                content_list = wire[-1]["content"]
                assert isinstance(content_list, list)
                content_list.append(block)
            else:
                wire.append({"role": "user", "content": [block]})
        elif isinstance(msg, AssistantMessage):
            content_blocks: list[dict[str, object]] = []
            for item in msg.blocks:
                if isinstance(item, ThinkingBlock):
                    if item.signature:
                        content_blocks.append(
                            {
                                "type": "thinking",
                                "thinking": item.text,
                                "signature": item.signature,
                            }
                        )
                elif isinstance(item, TextBlock):
                    if item.text:
                        content_blocks.append({"type": "text", "text": item.text})
                elif isinstance(item, ToolCall):
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": item.id,
                            "name": item.name,
                            "input": _parse_tool_input(item.arguments),
                        }
                    )
            if not content_blocks:
                continue
            if wire and wire[-1]["role"] == "assistant":
                prev_blocks = wire[-1]["content"]
                assert isinstance(prev_blocks, list)
                prev_blocks.extend(content_blocks)
            else:
                wire.append({"role": "assistant", "content": content_blocks})
        elif isinstance(msg, ToolResultMessage):
            tr_block: dict[str, object] = {
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id,
                "content": msg.content,
                "is_error": msg.is_error,
            }
            if wire and wire[-1]["role"] == "user":
                content_list = wire[-1]["content"]
                assert isinstance(content_list, list)
                content_list.append(tr_block)
            else:
                wire.append({"role": "user", "content": [tr_block]})

    if cache_control is not None:
        for entry in reversed(wire):
            if entry["role"] == "user":
                blocks = entry["content"]
                if isinstance(blocks, list):
                    for candidate in reversed(blocks):
                        if not isinstance(candidate, dict):
                            continue
                        if candidate.get("type") == "text" and not str(candidate.get("text") or "").strip():
                            continue
                        candidate["cache_control"] = dict(cache_control)
                        break
                break
    return wire


# 将 Anthropic stop_reason 映射为中立 DoneReason
def _map_stop_reason(raw_reason: str | None, *, has_tool_calls: bool) -> DoneReason:
    if raw_reason == "max_tokens":
        return "length"
    if raw_reason == "tool_use":
        return "toolUse"
    if raw_reason in {"end_turn", "stop_sequence", "pause_turn"}:
        return "stop"
    return "toolUse" if has_tool_calls else "stop"


# 流式工具调用参数分片累加器
@dataclass
class _AnthropicToolAcc:
    id: str
    name: str
    args: list[str] = field(default_factory=list)


# 原生 Anthropic Messages API 翻译器（零第三方依赖）
@dataclass
class AnthropicMessagesProvider:
    api: str = "anthropic-messages"
    post_stream: PostStream = field(default=default_post_stream, kw_only=True)

    # 组装原生 /v1/messages 请求地址、载荷（含系统块/工具/思考预算/缓存断点）与鉴权头
    def _build_request(
        self,
        model: ModelProfile,
        context: Context,
        options: StreamOptions | None,
    ) -> tuple[str, dict[str, object], dict[str, str]]:
        base = model.base_url.rstrip("/")
        url = f"{base}/messages" if base.endswith("/v1") else f"{base}/v1/messages"
        retention = (
            options.cache_retention
            if (options is not None and options.cache_retention is not None)
            else model.cache_retention
        )
        cc = _cache_control(retention)

        reasoning = options.reasoning if (options is not None and options.reasoning is not None) else model.reasoning
        max_tokens = options.max_tokens if (options is not None and options.max_tokens is not None) else model.max_tokens

        payload: dict[str, object] = {
            "model": model.model,
            "messages": _serialize_messages(context, cc),
            "stream": True,
        }
        if context.system_prompt:
            sys_block: dict[str, object] = {"type": "text", "text": context.system_prompt}
            if cc is not None:
                sys_block["cache_control"] = dict(cc)
            payload["system"] = [sys_block]

        if context.tools:
            tool_defs = [_translate_tool_schema(schema) for schema in context.tools]
            if cc is not None:
                tool_defs[-1]["cache_control"] = dict(cc)
            payload["tools"] = tool_defs

        if reasoning != "off":
            clamped = clamp_thinking_level(reasoning, ("minimal", "low", "medium", "high", "xhigh"))
            budget_tokens = max(1024, THINKING_BUDGETS[clamped])
            if max_tokens is None:
                resolved_max = budget_tokens + 4096
            elif max_tokens <= budget_tokens:
                resolved_max = budget_tokens + 1024
            else:
                resolved_max = max_tokens
            payload["max_tokens"] = resolved_max
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
        else:
            temp = options.temperature if (options is not None and options.temperature is not None) else model.temperature
            payload["max_tokens"] = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
            payload["temperature"] = min(1.0, max(0.0, temp))

        headers = {
            "x-api-key": model.api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
        }
        return url, payload, headers

    # 发送原生 Anthropic 流式请求并将原生 SSE 事件翻译为 12 种统一事件（异常收敛为 StreamError）
    def stream(
        self,
        model: ModelProfile,
        context: Context,
        options: StreamOptions | None = None,
    ) -> Iterator[AssistantMessageEvent]:
        t0 = time.perf_counter()
        blocks: list[AssistantBlock] = []
        block_map: dict[int, tuple[int, str]] = {}
        open_indices: set[int] = set()
        text_bufs: dict[int, list[str]] = {}
        thinking_bufs: dict[int, list[str]] = {}
        signature_bufs: dict[int, list[str]] = {}
        tool_accs: dict[int, _AnthropicToolAcc] = {}

        input_tokens = 0
        cache_read = 0
        cache_creation = 0
        output_tokens = 0
        saw_usage = False
        ttft_ms: float | None = None
        stop_reason_raw: str | None = None

        # 生成当前不可变部分消息快照
        def snapshot() -> AssistantMessage:
            return AssistantMessage(blocks=tuple(blocks))

        # 汇总当前已收集到的 Anthropic TokenUsage
        def build_usage() -> TokenUsage | None:
            if not saw_usage:
                return None
            total_prompt = input_tokens + cache_read + cache_creation
            bounded_cache = min(cache_read, total_prompt) if total_prompt > 0 else cache_read
            return TokenUsage(
                prompt_tokens=total_prompt,
                cached_tokens=bounded_cache,
                completion_tokens=output_tokens,
            )

        # 闭合指定索引的活跃内容块并产出对应 End 事件
        def close_block(anth_idx: int) -> Iterator[AssistantMessageEvent]:
            if anth_idx not in open_indices or anth_idx not in block_map:
                return
            open_indices.discard(anth_idx)
            cidx, cb_type = block_map[anth_idx]
            if cb_type == "text":
                full = "".join(text_bufs.get(cidx, ()))
                blocks[cidx] = TextBlock(text=full)
                yield TextEnd(content_index=cidx, text=full, partial=snapshot())
            elif cb_type == "thinking":
                full = "".join(thinking_bufs.get(cidx, ()))
                sig = "".join(signature_bufs.get(cidx, ()))
                blocks[cidx] = ThinkingBlock(text=full, signature=sig)
                yield ThinkingEnd(content_index=cidx, text=full, partial=snapshot())
            elif cb_type == "tool_use":
                acc = tool_accs[cidx]
                final_call = ToolCall(
                    id=acc.id,
                    name=acc.name,
                    arguments="".join(acc.args) or "{}",
                )
                blocks[cidx] = final_call
                yield ToolCallEnd(content_index=cidx, tool_call=final_call, partial=snapshot())

        yield StreamStart(partial=snapshot())

        line_iter: Iterator[str] | None = None
        try:
            url, payload, headers = self._build_request(model, context, options)
            line_iter = self.post_stream(url, payload, headers)
            for sse_event, data_str in iter_sse_events(line_iter):
                data = json.loads(data_str)
                if not isinstance(data, dict):
                    continue
                ev_type = str(data.get("type") or sse_event or "")
                if ev_type == "ping":
                    continue
                if ev_type == "error":
                    raise RuntimeError(f"Anthropic 流式响应返回错误：{data.get('error') or data}")

                if ev_type == "message_start":
                    msg_obj = data.get("message")
                    raw_usage = msg_obj.get("usage") if isinstance(msg_obj, dict) else None
                    if isinstance(raw_usage, dict):
                        saw_usage = True
                        input_tokens = non_negative_int(raw_usage.get("input_tokens"))
                        cache_read = non_negative_int(raw_usage.get("cache_read_input_tokens"))
                        cache_creation = non_negative_int(raw_usage.get("cache_creation_input_tokens"))
                        output_tokens = non_negative_int(raw_usage.get("output_tokens"))
                    continue

                if ev_type == "content_block_start":
                    anth_idx = non_negative_int(data.get("index"))
                    cb = data.get("content_block")
                    if not isinstance(cb, dict):
                        continue
                    cb_type = str(cb.get("type") or "")
                    cidx = len(blocks)
                    if cb_type == "text":
                        init_text = str(cb.get("text") or "")
                        blocks.append(TextBlock(text=init_text))
                        text_bufs[cidx] = [init_text] if init_text else []
                        block_map[anth_idx] = (cidx, "text")
                        open_indices.add(anth_idx)
                        yield TextStart(content_index=cidx, partial=snapshot())
                        if init_text:
                            if ttft_ms is None:
                                ttft_ms = round((time.perf_counter() - t0) * 1000, 3)
                            yield TextDelta(content_index=cidx, text=init_text, partial=snapshot())
                    elif cb_type == "thinking":
                        init_think = str(cb.get("thinking") or "")
                        init_sig = str(cb.get("signature") or "")
                        blocks.append(ThinkingBlock(text=init_think, signature=init_sig))
                        thinking_bufs[cidx] = [init_think] if init_think else []
                        signature_bufs[cidx] = [init_sig] if init_sig else []
                        block_map[anth_idx] = (cidx, "thinking")
                        open_indices.add(anth_idx)
                        yield ThinkingStart(content_index=cidx, partial=snapshot())
                        if init_think:
                            if ttft_ms is None:
                                ttft_ms = round((time.perf_counter() - t0) * 1000, 3)
                            yield ThinkingDelta(content_index=cidx, text=init_think, partial=snapshot())
                    elif cb_type == "tool_use":
                        if ttft_ms is None:
                            ttft_ms = round((time.perf_counter() - t0) * 1000, 3)
                        call_id = str(cb.get("id") or f"call_{cidx}")
                        name = str(cb.get("name") or "")
                        acc = _AnthropicToolAcc(id=call_id, name=name)
                        init_input = cb.get("input")
                        if isinstance(init_input, dict) and init_input:
                            acc.args.append(json.dumps(init_input, ensure_ascii=False))
                        blocks.append(ToolCall(id=call_id, name=name, arguments="".join(acc.args)))
                        tool_accs[cidx] = acc
                        block_map[anth_idx] = (cidx, "tool_use")
                        open_indices.add(anth_idx)
                        yield ToolCallStart(
                            content_index=cidx,
                            call_id=call_id,
                            name=name,
                            partial=snapshot(),
                        )
                    continue

                if ev_type == "content_block_delta":
                    anth_idx = non_negative_int(data.get("index"))
                    if anth_idx not in block_map:
                        continue
                    cidx, _ = block_map[anth_idx]
                    delta = data.get("delta")
                    if not isinstance(delta, dict):
                        continue
                    d_type = str(delta.get("type") or "")
                    if d_type == "text_delta":
                        piece = str(delta.get("text") or "")
                        if piece:
                            if ttft_ms is None:
                                ttft_ms = round((time.perf_counter() - t0) * 1000, 3)
                            text_bufs[cidx].append(piece)
                            blocks[cidx] = TextBlock(text="".join(text_bufs[cidx]))
                            yield TextDelta(content_index=cidx, text=piece, partial=snapshot())
                    elif d_type == "thinking_delta":
                        piece = str(delta.get("thinking") or "")
                        if piece:
                            if ttft_ms is None:
                                ttft_ms = round((time.perf_counter() - t0) * 1000, 3)
                            thinking_bufs[cidx].append(piece)
                            blocks[cidx] = ThinkingBlock(
                                text="".join(thinking_bufs[cidx]),
                                signature="".join(signature_bufs[cidx]),
                            )
                            yield ThinkingDelta(content_index=cidx, text=piece, partial=snapshot())
                    elif d_type == "signature_delta":
                        sig = str(delta.get("signature") or "")
                        if sig:
                            signature_bufs[cidx].append(sig)
                            blocks[cidx] = ThinkingBlock(
                                text="".join(thinking_bufs[cidx]),
                                signature="".join(signature_bufs[cidx]),
                            )
                    elif d_type == "input_json_delta":
                        piece = str(delta.get("partial_json") or "")
                        if piece:
                            if ttft_ms is None:
                                ttft_ms = round((time.perf_counter() - t0) * 1000, 3)
                            acc = tool_accs[cidx]
                            acc.args.append(piece)
                            blocks[cidx] = ToolCall(
                                id=acc.id,
                                name=acc.name,
                                arguments="".join(acc.args),
                            )
                            yield ToolCallDelta(content_index=cidx, delta=piece, partial=snapshot())
                    continue

                if ev_type == "content_block_stop":
                    anth_idx = non_negative_int(data.get("index"))
                    yield from close_block(anth_idx)
                    continue

                if ev_type == "message_delta":
                    delta = data.get("delta")
                    if isinstance(delta, dict):
                        raw_stop = delta.get("stop_reason")
                        if isinstance(raw_stop, str) and raw_stop:
                            stop_reason_raw = raw_stop
                    raw_usage = data.get("usage")
                    if isinstance(raw_usage, dict):
                        saw_usage = True
                        if raw_usage.get("output_tokens") is not None:
                            output_tokens = non_negative_int(raw_usage.get("output_tokens"))
                        if raw_usage.get("input_tokens") is not None:
                            input_tokens = non_negative_int(raw_usage.get("input_tokens"))
                        if raw_usage.get("cache_read_input_tokens") is not None:
                            cache_read = non_negative_int(raw_usage.get("cache_read_input_tokens"))
                        if raw_usage.get("cache_creation_input_tokens") is not None:
                            cache_creation = non_negative_int(raw_usage.get("cache_creation_input_tokens"))
                    continue

                if ev_type == "message_stop":
                    break

            for anth_idx in sorted(list(open_indices)):
                yield from close_block(anth_idx)

            usage = build_usage()
            has_tools = any(isinstance(b, ToolCall) for b in blocks)
            done_reason = _map_stop_reason(stop_reason_raw, has_tool_calls=has_tools)
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
                usage=build_usage(),
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
