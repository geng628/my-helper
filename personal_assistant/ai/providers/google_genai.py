"""Layer 1 原生 Google Gemini API 翻译器（api = "google-generative-ai"）：支持原生思考与函数调用。"""
from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from urllib.error import HTTPError
from urllib.parse import quote

from personal_assistant.ai.protocol import (
    THINKING_BUDGETS,
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


# 将中立 ToolSchema 翻译为 Gemini functionDeclarations 条目（零参数工具省略空 OBJECT.properties 防 400）
def _translate_tool_declaration(schema: ToolSchema) -> dict[str, object]:
    properties: dict[str, dict[str, object]] = {}
    for param, desc in schema.parameters.items():
        prop: dict[str, object] = {"type": "STRING", "description": desc}
        if schema.param_enums and param in schema.param_enums:
            prop["enum"] = list(schema.param_enums[param])
        properties[param] = prop
    decl: dict[str, object] = {
        "name": schema.name,
        "description": schema.description,
    }
    if properties:
        params: dict[str, object] = {
            "type": "OBJECT",
            "properties": properties,
        }
        if schema.required_names:
            params["required"] = list(schema.required_names)
        decl["parameters"] = params
    return decl


# 将工具调用 JSON 字符串安全解析为参数字典
def _parse_function_args(raw_arguments: str) -> dict[str, object]:
    try:
        parsed = json.loads(raw_arguments or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


# 将中立消息序列翻译为 Gemini contents 数组（合并连续 ToolResultMessage 为单条 user functionResponse）
def _serialize_contents(context: Context) -> list[dict[str, object]]:
    contents: list[dict[str, object]] = []
    for msg in context.messages:
        if isinstance(msg, UserMessage):
            part: dict[str, object] = {"text": msg.content}
            if contents and contents[-1]["role"] == "user":
                parts_list = contents[-1]["parts"]
                assert isinstance(parts_list, list)
                parts_list.append(part)
            else:
                contents.append({"role": "user", "parts": [part]})
        elif isinstance(msg, AssistantMessage):
            model_parts: list[dict[str, object]] = []
            for block in msg.blocks:
                if isinstance(block, ThinkingBlock):
                    if block.text:
                        t_part: dict[str, object] = {"thought": True, "text": block.text}
                        if block.signature:
                            t_part["thoughtSignature"] = block.signature
                        model_parts.append(t_part)
                elif isinstance(block, TextBlock):
                    if block.text:
                        model_parts.append({"text": block.text})
                elif isinstance(block, ToolCall):
                    model_parts.append(
                        {
                            "functionCall": {
                                "name": block.name,
                                "args": _parse_function_args(block.arguments),
                            }
                        }
                    )
            if model_parts:
                contents.append({"role": "model", "parts": model_parts})
        elif isinstance(msg, ToolResultMessage):
            response_obj: dict[str, object] = (
                {"error": msg.content} if msg.is_error else {"result": msg.content}
            )
            fr_part: dict[str, object] = {
                "functionResponse": {
                    "name": msg.tool_name,
                    "response": response_obj,
                }
            }
            if contents and contents[-1]["role"] == "user":
                parts_list = contents[-1]["parts"]
                assert isinstance(parts_list, list)
                parts_list.append(fr_part)
            else:
                contents.append({"role": "user", "parts": [fr_part]})
    return contents


# 将 Gemini finishReason 映射为中立 DoneReason
def _map_finish_reason(raw_reason: str | None, *, has_tool_calls: bool) -> DoneReason:
    if raw_reason == "MAX_TOKENS":
        return "length"
    if has_tool_calls:
        return "toolUse"
    return "stop"


# 原生 Google Gemini API 翻译器（零第三方依赖）
@dataclass
class GoogleGenAIProvider:
    api: str = "google-generative-ai"
    post_stream: PostStream = field(default=default_post_stream, kw_only=True)

    # 组装 Gemini :streamGenerateContent?alt=sse 请求地址、JSON 载荷与 x-goog-api-key 鉴权头
    def _build_request(
        self,
        model: ModelProfile,
        context: Context,
        options: StreamOptions | None,
    ) -> tuple[str, dict[str, object], dict[str, str]]:
        base = model.base_url.rstrip("/")
        prefix = base if base.endswith(("/v1beta", "/v1", "/v1alpha")) else f"{base}/v1beta"
        raw_model = model.model.strip()
        model_id = raw_model[len("models/") :] if raw_model.startswith("models/") else raw_model
        url = f"{prefix}/models/{quote(model_id, safe='')}:streamGenerateContent?alt=sse"

        gen_cfg: dict[str, object] = {}
        temp = options.temperature if (options is not None and options.temperature is not None) else model.temperature
        gen_cfg["temperature"] = temp
        max_tok = options.max_tokens if (options is not None and options.max_tokens is not None) else model.max_tokens
        if max_tok is not None:
            gen_cfg["maxOutputTokens"] = max_tok
        reasoning = options.reasoning if (options is not None and options.reasoning is not None) else model.reasoning
        if reasoning != "off":
            clamped = clamp_thinking_level(reasoning, ("minimal", "low", "medium", "high", "xhigh"))
            gen_cfg["thinkingConfig"] = {
                "includeThoughts": True,
                "thinkingBudget": THINKING_BUDGETS[clamped],
            }

        payload: dict[str, object] = {
            "contents": _serialize_contents(context),
            "generationConfig": gen_cfg,
        }
        if context.system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": context.system_prompt}]}
        if context.tools:
            payload["tools"] = [
                {"functionDeclarations": [_translate_tool_declaration(schema) for schema in context.tools]}
            ]

        headers = {"x-goog-api-key": model.api_key}
        return url, payload, headers

    # 发送原生 Gemini 流式请求并解析 SSE 分片吐出 12 种统一事件（异常收敛为 StreamError）
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
        finish_reason_raw: str | None = None

        thinking_idx: int | None = None
        thinking_open = False
        thinking_buf: list[str] = []
        thinking_sig = ""

        text_idx: int | None = None
        text_open = False
        text_buf: list[str] = []

        call_seq = 0

        # 生成当前不可变部分消息快照
        def snapshot() -> AssistantMessage:
            return AssistantMessage(blocks=tuple(blocks))

        # 闭合当前打开的思维链内容块并产出 ThinkingEnd（同步清零当前块签名防泄漏至后续块）
        def close_thinking() -> Iterator[AssistantMessageEvent]:
            nonlocal thinking_open, thinking_sig
            if thinking_open and thinking_idx is not None:
                full = "".join(thinking_buf)
                blocks[thinking_idx] = ThinkingBlock(text=full, signature=thinking_sig)
                thinking_open = False
                thinking_sig = ""
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
            line_iter = self.post_stream(url, payload, headers)
            for _, data_str in iter_sse_events(line_iter):
                if data_str == "[DONE]":
                    break
                chunk = json.loads(data_str)
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("error") is not None:
                    raise RuntimeError(f"Gemini 流式响应返回错误：{chunk['error']}")

                meta = chunk.get("usageMetadata")
                if isinstance(meta, dict):
                    prompt_tokens = non_negative_int(meta.get("promptTokenCount"))
                    cached_tokens = non_negative_int(meta.get("cachedContentTokenCount"))
                    completion_tokens = non_negative_int(meta.get("candidatesTokenCount")) + non_negative_int(
                        meta.get("thoughtsTokenCount")
                    )
                    if prompt_tokens > 0:
                        cached_tokens = min(cached_tokens, prompt_tokens)
                    usage = TokenUsage(
                        prompt_tokens=prompt_tokens,
                        cached_tokens=cached_tokens,
                        completion_tokens=completion_tokens,
                    )

                candidates = chunk.get("candidates")
                if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
                    continue
                first = candidates[0]
                raw_finish = first.get("finishReason")
                if isinstance(raw_finish, str) and raw_finish:
                    finish_reason_raw = raw_finish
                content_obj = first.get("content")
                parts = content_obj.get("parts") if isinstance(content_obj, dict) else None
                if not isinstance(parts, list):
                    continue

                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    if part.get("thought") is True:
                        piece = str(part.get("text") or "")
                        sig = str(part.get("thoughtSignature") or "")
                        if sig:
                            thinking_sig = sig
                        if not piece and not sig:
                            continue
                        yield from close_text()
                        if not thinking_open:
                            thinking_idx = len(blocks)
                            thinking_buf = []
                            blocks.append(ThinkingBlock(text="", signature=thinking_sig))
                            thinking_open = True
                            yield ThinkingStart(content_index=thinking_idx, partial=snapshot())
                        assert thinking_idx is not None
                        if piece:
                            if ttft_ms is None:
                                ttft_ms = round((time.perf_counter() - t0) * 1000, 3)
                            thinking_buf.append(piece)
                            blocks[thinking_idx] = ThinkingBlock(
                                text="".join(thinking_buf),
                                signature=thinking_sig,
                            )
                            yield ThinkingDelta(
                                content_index=thinking_idx,
                                text=piece,
                                partial=snapshot(),
                            )
                        else:
                            blocks[thinking_idx] = ThinkingBlock(
                                text="".join(thinking_buf),
                                signature=thinking_sig,
                            )
                        continue

                    if "functionCall" in part and isinstance(part["functionCall"], dict):
                        if ttft_ms is None:
                            ttft_ms = round((time.perf_counter() - t0) * 1000, 3)
                        yield from close_thinking()
                        yield from close_text()
                        fc = part["functionCall"]
                        call_seq += 1
                        call_id = str(fc.get("id") or f"gemini_call_{call_seq}")
                        fn_name = str(fc.get("name") or "")
                        raw_args = fc.get("args")
                        args_json = json.dumps(raw_args if isinstance(raw_args, dict) else {}, ensure_ascii=False)
                        cidx = len(blocks)
                        blocks.append(ToolCall(id=call_id, name=fn_name, arguments=""))
                        yield ToolCallStart(
                            content_index=cidx,
                            call_id=call_id,
                            name=fn_name,
                            partial=snapshot(),
                        )
                        final_tc = ToolCall(id=call_id, name=fn_name, arguments=args_json)
                        blocks[cidx] = final_tc
                        yield ToolCallDelta(content_index=cidx, delta=args_json, partial=snapshot())
                        yield ToolCallEnd(content_index=cidx, tool_call=final_tc, partial=snapshot())
                        continue

                    raw_text = part.get("text")
                    if isinstance(raw_text, str) and raw_text:
                        if ttft_ms is None:
                            ttft_ms = round((time.perf_counter() - t0) * 1000, 3)
                        yield from close_thinking()
                        if not text_open:
                            text_idx = len(blocks)
                            text_buf = []
                            blocks.append(TextBlock(text=""))
                            text_open = True
                            yield TextStart(content_index=text_idx, partial=snapshot())
                        text_buf.append(raw_text)
                        assert text_idx is not None
                        blocks[text_idx] = TextBlock(text="".join(text_buf))
                        yield TextDelta(content_index=text_idx, text=raw_text, partial=snapshot())

            yield from close_thinking()
            yield from close_text()
            has_tools = any(isinstance(b, ToolCall) for b in blocks)
            done_reason = _map_finish_reason(finish_reason_raw, has_tool_calls=has_tools)
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
