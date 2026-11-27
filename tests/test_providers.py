"""Layer 1 Pi ch04 三层翻译架构专项单元测试：覆盖协议契约、注册表、OpenAI / Anthropic / Gemini 原生翻译器。"""
from __future__ import annotations

import io
import json
from collections.abc import Iterator
from urllib.error import HTTPError, URLError

import pytest

from personal_assistant.ai import (
    AnthropicMessagesProvider,
    ApiProviderRegistry,
    AssistantMessage,
    Context,
    GoogleGenAIProvider,
    LLMNotConfigured,
    ModelProfile,
    NoopModelClient,
    OpenAICompletionsProvider,
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
    available_api_providers,
    clamp_thinking_level,
    complete,
    from_env,
    from_model_profile,
    iter_sse_events,
    stream,
    stream_simple,
)
from personal_assistant.ai.providers.http import redact_error_tokens


# 测试语义思考级别夹紧规则（先向上找、找不到再向下找）与 ModelProfile.from_dict 严格校验
def test_protocol_clamp_thinking_and_model_profile_validation() -> None:
    assert clamp_thinking_level("minimal", ("low", "medium", "high")) == "low"
    assert clamp_thinking_level("xhigh", ("low", "medium", "high")) == "high"
    assert clamp_thinking_level("medium", ("low", "high")) == "high"
    assert clamp_thinking_level("high", ("minimal", "low")) == "low"
    assert clamp_thinking_level("off", ("off", "low", "high")) == "off"
    assert clamp_thinking_level("high", ()) == "off"
    with pytest.raises(ValueError, match="思考级别"):
        clamp_thinking_level("ultra", ("low", "high"))  # type: ignore[arg-type]

    # 严格拒绝已废弃的旧 provider 字段、NaN/Inf/bool 温度与非法扩展字段
    with pytest.raises(ValueError, match="provider"):
        ModelProfile.from_dict(
            "legacy",
            {"provider": "openai_compatible", "base_url": "https://api.example.com/v1", "model": "m"},
        )
    for bad_temp in (True, False, float("nan"), float("inf"), float("-inf"), -0.1, 2.5, "nan", "abc"):
        with pytest.raises(ValueError, match="temperature"):
            ModelProfile.from_dict(
                "bad_t",
                {"base_url": "https://api.example.com/v1", "model": "m", "temperature": bad_temp},
            )
    for bad_max in (0, -5, True, 1.5, "abc"):
        with pytest.raises(ValueError, match="max_tokens"):
            ModelProfile.from_dict(
                "bad_m",
                {"base_url": "https://api.example.com/v1", "model": "m", "max_tokens": bad_max},
            )
    with pytest.raises(ValueError, match="reasoning"):
        ModelProfile.from_dict(
            "bad_r",
            {"base_url": "https://api.example.com/v1", "model": "m", "reasoning": "extreme"},
        )
    with pytest.raises(ValueError, match="cache_retention"):
        ModelProfile.from_dict(
            "bad_c",
            {"base_url": "https://api.example.com/v1", "model": "m", "cache_retention": "forever"},
        )

    ok = ModelProfile.from_dict(
        "claude",
        {
            "api": "Anthropic-Messages",
            "base_url": "https://api.anthropic.com",
            "model": "claude-sonnet-4-5",
            "api_key": "sk-ant-secret",
            "temperature": 0.4,
            "max_tokens": 8192,
            "reasoning": "MEDIUM",
            "cache_retention": "LONG",
        },
    )
    assert ok.api == "anthropic-messages"
    assert ok.max_tokens == 8192
    assert ok.reasoning == "medium"
    assert ok.cache_retention == "long"
    redacted = ok.redacted_dict()
    assert "sk-ant-secret" not in json.dumps(redacted)
    assert redacted["api"] == "anthropic-messages"
    assert redacted["reasoning"] == "medium"
    assert redacted["cache_retention"] == "long"


# 测试全局默认注册表三大内建翻译器、同步抛 LLMNotConfigured、stream_simple 合并与 complete() 语义
def test_registry_builtin_providers_sync_auth_and_stream_simple() -> None:
    assert set(available_api_providers()) >= {
        "anthropic-messages",
        "google-generative-ai",
        "openai-completions",
    }

    # 缺密钥时 stream() 必须同步抛出 LLMNotConfigured（不等消费迭代器）
    unauth = ModelProfile(name="no-key", api="openai-completions", base_url="https://x/v1", model="m")
    with pytest.raises(LLMNotConfigured):
        stream(unauth, Context(messages=(UserMessage("hi"),)))
    assert isinstance(from_model_profile(unauth), NoopModelClient)

    # 未知 api 协议立即抛带可用列表的 ValueError
    unknown = ModelProfile(name="bad", api="nonexistent-api", base_url="https://x/v1", model="m", api_key="k")
    with pytest.raises(ValueError, match="nonexistent-api"):
        from_model_profile(unknown)

    # stream_simple 合并模型默认项与显式 StreamOptions（开启 reasoning 且未指定 max_tokens 自动补足上限）
    captured_options: list[StreamOptions | None] = []

    # 记录收到的合并选项并返回文本或错误事件的探针翻译器
    class _SpyProvider:
        api = "spy-api"

        # 构造探针事件队列
        def __init__(self, fail: bool = False) -> None:
            self.fail = fail

        # 产出成功或错误事件流
        def stream(
            self,
            model: ModelProfile,
            context: Context,
            options: StreamOptions | None = None,
        ) -> Iterator[object]:
            captured_options.append(options)
            yield StreamStart(partial=AssistantMessage())
            if self.fail:
                err_msg = AssistantMessage(stop_reason="error", error_message="upstream 503")
                yield StreamError(reason="error", message=err_msg, error_message="upstream 503")
                return
            msg = AssistantMessage.from_text("ok", stop_reason="stop", usage=TokenUsage(10, 5, 2))
            yield TextStart(content_index=0, partial=AssistantMessage())
            yield TextDelta(content_index=0, text="ok", partial=msg)
            yield TextEnd(content_index=0, text="ok", partial=msg)
            yield StreamDone(reason="stop", message=msg, usage=msg.usage)

    reg = ApiProviderRegistry()
    reg.register(_SpyProvider(fail=False))
    prof = ModelProfile(
        name="thinker",
        api="spy-api",
        base_url="https://x/v1",
        model="m",
        api_key="k",
        temperature=0.3,
        reasoning="high",
        cache_retention="long",
    )
    msg = complete(prof, Context(messages=(UserMessage("q"),)), registry=reg)
    assert msg.text == "ok"
    assert captured_options[-1] == StreamOptions(
        temperature=0.3,
        max_tokens=16384 + 4096,
        reasoning="high",
        cache_retention="long",
        session_id=None,
    )

    # complete() 遇到 StreamError 时抛出 RuntimeError(event.error_message)
    err_reg = ApiProviderRegistry()
    err_reg.register(_SpyProvider(fail=True))
    with pytest.raises(RuntimeError, match="upstream 503"):
        complete(prof, Context(messages=(UserMessage("q"),)), registry=err_reg)

    # from_env 复用 ModelProfile.from_dict 校验（拒绝 NaN 温度）
    with pytest.raises(ValueError, match="temperature"):
        from_env({"LLM_MODE": "on", "LLM_API_KEY": "k", "LLM_TEMPERATURE": "nan"})


# 测试 OpenAI Completions 翻译器：双向请求翻译（含 reasoning_effort/max_tokens/tools）、12 事件与异常转 StreamError
def test_openai_completions_provider_bidirectional_and_stream_error() -> None:
    captured: dict[str, object] = {}

    # 模拟 OpenAI SSE 流式输出文本与工具调用
    def fake_stream(url: str, payload: dict[str, object], headers: dict[str, str]) -> Iterator[str]:
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        return iter(
            [
                'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"准备查询"}}]}\n',
                'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_","arguments":"{\\"path\\":"}}]}}]}\n',
                'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"file","arguments":"\\"a.md\\"}"}}]}}]}\n',
                'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n',
                'data: {"choices":[],"usage":{"prompt_tokens":150,"prompt_tokens_details":{"cached_tokens":96},"completion_tokens":20}}\n',
                "data: [DONE]\n",
            ]
        )

    provider = OpenAICompletionsProvider(post_stream=fake_stream)
    model = ModelProfile(
        name="oa",
        api="openai-completions",
        base_url="https://api.openai.com/v1/",
        model="gpt-4o",
        api_key="sk-oa-key",
        temperature=0.5,
        max_tokens=2048,
        reasoning="xhigh",
    )
    schema = ToolSchema(
        name="read_file",
        description="读取文件",
        parameters={"path": "路径", "mode": "模式"},
        param_enums={"mode": ("text", "raw")},
        required=("path",),
    )
    ctx = Context(
        system_prompt="你是助手",
        messages=(
            UserMessage("第一句"),
            AssistantMessage(blocks=(ToolCall(id="old_1", name="read_file", arguments='{"path":"x.md"}'),)),
            ToolResultMessage(tool_call_id="old_1", tool_name="read_file", content="内容X"),
            UserMessage("再查 a.md"),
        ),
        tools=(schema,),
    )
    events = list(provider.stream(model, ctx, StreamOptions(temperature=0.5, max_tokens=2048, reasoning="xhigh")))

    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"] == {"Authorization": "Bearer sk-oa-key"}
    sent = captured["payload"]
    assert isinstance(sent, dict)
    assert sent["model"] == "gpt-4o"
    assert sent["temperature"] == 0.5
    assert sent["max_tokens"] == 2048
    assert sent["reasoning_effort"] == "high"
    assert sent["tools"][0]["function"]["parameters"]["required"] == ["path"]  # type: ignore[index]
    assert sent["tools"][0]["function"]["parameters"]["properties"]["mode"]["enum"] == ["text", "raw"]  # type: ignore[index]
    assert [m["role"] for m in sent["messages"]] == ["system", "user", "assistant", "tool", "user"]  # type: ignore[index]

    assert [type(ev) for ev in events] == [
        StreamStart,
        TextStart,
        TextDelta,
        TextEnd,
        ToolCallStart,
        ToolCallDelta,
        ToolCallDelta,
        ToolCallEnd,
        StreamDone,
    ]
    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.reason == "toolUse"
    assert done.message.text == "准备查询"
    assert done.message.tool_calls == (ToolCall(id="call_1", name="read_file", arguments='{"path":"a.md"}'),)
    assert done.usage == TokenUsage(prompt_tokens=150, cached_tokens=96, completion_tokens=20)

    # HTTP/传输异常必须收敛为 StreamError 且脱敏凭据
    def fail_stream(url: str, payload: dict[str, object], headers: dict[str, str]) -> Iterator[str]:
        raise HTTPError(
            url,
            401,
            "Unauthorized",
            {},
            io.BytesIO(b'{"error":"bad key sk-secret123 Bearer tok_xyz"}'),
        )

    err_events = list(OpenAICompletionsProvider(post_stream=fail_stream).stream(model, ctx))
    assert [type(ev) for ev in err_events] == [StreamStart, StreamError]
    err_ev = err_events[-1]
    assert isinstance(err_ev, StreamError)
    assert err_ev.reason == "error"
    assert "401" in err_ev.error_message
    assert "sk-secret123" not in err_ev.error_message
    assert "tok_xyz" not in err_ev.error_message
    assert "sk-***" in err_ev.error_message


# 测试原生 Anthropic Messages 翻译器：请求翻译（缓存断点/工具/思考预算/合并 tool_result）、原生 SSE 12 事件与异常收敛
def test_anthropic_messages_provider_request_sse_caching_thinking_and_errors() -> None:
    captured: dict[str, object] = {}

    # 模拟 Anthropic 原生 SSE 完整流（message_start -> thinking+signature -> text -> tool_use -> message_delta -> message_stop）
    def fake_anthropic_stream(url: str, payload: dict[str, object], headers: dict[str, str]) -> Iterator[str]:
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        lines = [
            "event: message_start\n",
            'data: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":100,"cache_read_input_tokens":400,"cache_creation_input_tokens":50,"output_tokens":1}}}\n',
            "\n",
            "event: content_block_start\n",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}\n',
            "\n",
            "event: content_block_delta\n",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"先分析"}}\n',
            "\n",
            "event: content_block_delta\n",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig_abc"}}\n',
            "\n",
            "event: content_block_stop\n",
            'data: {"type":"content_block_stop","index":0}\n',
            "\n",
            "event: content_block_start\n",
            'data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}\n',
            "\n",
            "event: content_block_delta\n",
            'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"正在查询"}}\n',
            "\n",
            "event: content_block_stop\n",
            'data: {"type":"content_block_stop","index":1}\n',
            "\n",
            "event: content_block_start\n",
            'data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_1","name":"search_docs","input":{}}}\n',
            "\n",
            "event: content_block_delta\n",
            'data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\\"q\\": \\"pi\\"}"}}\n',
            "\n",
            "event: content_block_stop\n",
            'data: {"type":"content_block_stop","index":2}\n',
            "\n",
            "event: message_delta\n",
            'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":45}}\n',
            "\n",
            "event: message_stop\n",
            'data: {"type":"message_stop"}\n',
            "\n",
        ]
        return iter(lines)

    provider = AnthropicMessagesProvider(post_stream=fake_anthropic_stream)
    model = ModelProfile(
        name="claude",
        api="anthropic-messages",
        base_url="https://api.anthropic.com",
        model="claude-sonnet-4-5",
        api_key="sk-ant-live",
        reasoning="low",
        cache_retention="long",
    )
    ctx = Context(
        system_prompt="你是严谨助手",
        messages=(
            UserMessage("帮我查文档"),
            AssistantMessage(
                blocks=(
                    ThinkingBlock(text="之前思考", signature="sig_prev"),
                    TextBlock(text="调用两个工具"),
                    ToolCall(id="t1", name="search_docs", arguments='{"q":"a"}'),
                    ToolCall(id="t2", name="search_docs", arguments='{"q":"b"}'),
                )
            ),
            ToolResultMessage(tool_call_id="t1", tool_name="search_docs", content="结果A", is_error=False),
            ToolResultMessage(tool_call_id="t2", tool_name="search_docs", content="错误：超时", is_error=True),
        ),
        tools=(
            ToolSchema(name="search_docs", description="检索文档", parameters={"q": "关键词"}),
            ToolSchema(name="read_doc", description="读文档", parameters={"path": "路径"}),
        ),
    )
    events = list(
        provider.stream(
            model,
            ctx,
            StreamOptions(max_tokens=1024, reasoning="low", cache_retention="long"),
        )
    )

    # 1. 验证请求地址、鉴权头、Prompt Caching 断点、思考预算与 max_tokens > budget_tokens 自动保障
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"] == {"x-api-key": "sk-ant-live", "anthropic-version": "2023-06-01"}
    sent = captured["payload"]
    assert isinstance(sent, dict)
    assert sent["system"] == [
        {"type": "text", "text": "你是严谨助手", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
    ]
    assert "cache_control" not in sent["tools"][0]  # type: ignore[index]
    assert sent["tools"][1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}  # type: ignore[index]
    assert sent["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert int(sent["max_tokens"]) > 2048  # type: ignore[arg-type]
    assert "temperature" not in sent

    # 连续两条 ToolResultMessage 合并为单条 user 消息并打缓存断点；前轮 assistant 保留 signature
    wire_msgs = sent["messages"]
    assert isinstance(wire_msgs, list)
    assert [m["role"] for m in wire_msgs] == ["user", "assistant", "user"]
    assert wire_msgs[1]["content"][0] == {"type": "thinking", "thinking": "之前思考", "signature": "sig_prev"}
    assert wire_msgs[1]["content"][2] == {"type": "tool_use", "id": "t1", "name": "search_docs", "input": {"q": "a"}}
    assert len(wire_msgs[2]["content"]) == 2
    assert wire_msgs[2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "t1",
        "content": "结果A",
        "is_error": False,
    }
    assert wire_msgs[2]["content"][1] == {
        "type": "tool_result",
        "tool_use_id": "t2",
        "content": "错误：超时",
        "is_error": True,
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    }

    # 2. 验证 12 种规范流事件、ThinkingBlock.signature 与 TokenUsage 合并口径
    assert [type(ev) for ev in events] == [
        StreamStart,
        ThinkingStart,
        ThinkingDelta,
        ThinkingEnd,
        TextStart,
        TextDelta,
        TextEnd,
        ToolCallStart,
        ToolCallDelta,
        ToolCallEnd,
        StreamDone,
    ]
    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.reason == "toolUse"
    assert done.message.blocks == (
        ThinkingBlock(text="先分析", signature="sig_abc"),
        TextBlock(text="正在查询"),
        ToolCall(id="toolu_1", name="search_docs", arguments='{"q": "pi"}'),
    )
    assert done.usage == TokenUsage(prompt_tokens=550, cached_tokens=400, completion_tokens=45)

    # 3. cache_retention="none" 不打任何 cache_control；原生 error 事件转 StreamError
    none_captured: dict[str, object] = {}

    # 模拟关闭缓存并在流内返回原生 error 帧
    def err_anthropic_stream(url: str, payload: dict[str, object], headers: dict[str, str]) -> Iterator[str]:
        none_captured["payload"] = payload
        return iter(
            [
                "event: error\n",
                'data: {"type":"error","error":{"type":"overloaded_error","message":"x-api-key: ant-secret-999"}}\n',
                "\n",
            ]
        )

    err_evs = list(
        AnthropicMessagesProvider(post_stream=err_anthropic_stream).stream(
            model,
            ctx,
            StreamOptions(reasoning="off", cache_retention="none"),
        )
    )
    none_payload = none_captured["payload"]
    assert isinstance(none_payload, dict)
    assert "cache_control" not in json.dumps(none_payload)
    assert "temperature" in none_payload
    assert isinstance(err_evs[-1], StreamError)
    assert "ant-secret-999" not in err_evs[-1].error_message


# 测试原生 Google Gemini 翻译器：请求翻译（systemInstruction/contents/functionDeclarations/thinkingConfig）、SSE 与错误收敛
def test_google_genai_provider_request_sse_and_errors() -> None:
    captured: dict[str, object] = {}

    # 模拟 Gemini 原生 SSE 流（思考 -> 文本 -> functionCall + usageMetadata）
    def fake_gemini_stream(url: str, payload: dict[str, object], headers: dict[str, str]) -> Iterator[str]:
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        lines = [
            'data: {"candidates":[{"content":{"role":"model","parts":[{"thought":true,"text":"先推理"},{"text":"你好"}]}}]}\n',
            "\n",
            'data: {"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"city":"北京"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":300,"cachedContentTokenCount":120,"candidatesTokenCount":30,"thoughtsTokenCount":10}}\n',
            "\n",
        ]
        return iter(lines)

    provider = GoogleGenAIProvider(post_stream=fake_gemini_stream)
    model = ModelProfile(
        name="gem",
        api="google-generative-ai",
        base_url="https://generativelanguage.googleapis.com",
        model="gemini-2.5-flash",
        api_key="AIzaSySecretKey123456789",
        temperature=0.3,
        max_tokens=4096,
        reasoning="medium",
    )
    ctx = Context(
        system_prompt="你是旅行助手",
        messages=(
            UserMessage("北京天气？"),
            AssistantMessage(
                blocks=(
                    ThinkingBlock(text="旧思考"),
                    ToolCall(id="g1", name="get_weather", arguments='{"city":"上海"}'),
                )
            ),
            ToolResultMessage(tool_call_id="g1", tool_name="get_weather", content="上海多云", is_error=False),
            ToolResultMessage(tool_call_id="g2", tool_name="get_weather", content="错误：限流", is_error=True),
        ),
        tools=(ToolSchema(name="get_weather", description="查天气", parameters={"city": "城市"}),),
    )
    events = list(
        provider.stream(
            model,
            ctx,
            StreamOptions(temperature=0.3, max_tokens=4096, reasoning="medium"),
        )
    )

    # 1. 验证 URL、鉴权头与 Gemini 请求载荷结构
    assert captured["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
    assert captured["headers"] == {"x-goog-api-key": "AIzaSySecretKey123456789"}
    sent = captured["payload"]
    assert isinstance(sent, dict)
    assert sent["systemInstruction"] == {"parts": [{"text": "你是旅行助手"}]}
    assert sent["generationConfig"] == {
        "temperature": 0.3,
        "maxOutputTokens": 4096,
        "thinkingConfig": {"includeThoughts": True, "thinkingBudget": 8192},
    }
    assert sent["tools"][0]["functionDeclarations"][0]["name"] == "get_weather"  # type: ignore[index]
    contents = sent["contents"]
    assert isinstance(contents, list)
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert contents[1]["parts"][0] == {"thought": True, "text": "旧思考"}
    assert contents[1]["parts"][1] == {"functionCall": {"name": "get_weather", "args": {"city": "上海"}}}
    assert contents[2]["parts"] == [
        {"functionResponse": {"name": "get_weather", "response": {"result": "上海多云"}}},
        {"functionResponse": {"name": "get_weather", "response": {"error": "错误：限流"}}},
    ]

    # 2. 验证 12 种规范事件流、存在 functionCall 时 reason=="toolUse" 与 usageMetadata 映射
    assert [type(ev) for ev in events] == [
        StreamStart,
        ThinkingStart,
        ThinkingDelta,
        ThinkingEnd,
        TextStart,
        TextDelta,
        TextEnd,
        ToolCallStart,
        ToolCallDelta,
        ToolCallEnd,
        StreamDone,
    ]
    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.reason == "toolUse"
    assert done.message.thinking == "先推理"
    assert done.message.text == "你好"
    assert len(done.message.tool_calls) == 1
    assert done.message.tool_calls[0].name == "get_weather"
    assert json.loads(done.message.tool_calls[0].arguments) == {"city": "北京"}
    assert done.usage == TokenUsage(prompt_tokens=300, cached_tokens=120, completion_tokens=40)

    # 3. 验证网络异常转为 StreamError 且脱敏 Google API Key
    def fail_net(url: str, payload: dict[str, object], headers: dict[str, str]) -> Iterator[str]:
        raise URLError("x-goog-api-key: AIzaSyLeakedToken9999999")

    err_evs = list(GoogleGenAIProvider(post_stream=fail_net).stream(model, ctx))
    assert isinstance(err_evs[-1], StreamError)
    assert "AIzaSyLeakedToken9999999" not in err_evs[-1].error_message

    # 4. 验证脱敏函数覆盖全部常见密钥头与查询参数
    redacted = redact_error_tokens("Bearer abc sk-123 x-api-key: ant99 x-goog-api-key=g99 ?key=q99 AIzaSyAbCdEf123456")
    for secret in ("abc", "sk-123", "ant99", "g99", "q99", "AIzaSyAbCdEf123456"):
        assert secret not in redacted

    # 5. 验证通用 iter_sse_events 同时兼容标准空行分隔、多行 data 拼接、无空行紧凑流与提前中断关流
    parsed_sse = list(
        iter_sse_events(
            [
                ": ping\n",
                "event: msg\n",
                'data: {"a":\n',
                "data: 1}\n",
                "\n",
                'data: {"b": 2}\n',
                "data: [DONE]\n",
            ]
        )
    )
    assert parsed_sse == [("msg", '{"a":\n1}'), ("", '{"b": 2}'), ("", "[DONE]")]

    sse_closed: list[bool] = []

    # 追踪提前关闭时底层 SSE 行生成器是否确定性清理
    def _closing_lines() -> Iterator[str]:
        try:
            yield 'data: {"k": 1}\n'
            yield "\n"
            yield 'data: {"k": 2}\n'
            yield "\n"
        finally:
            sse_closed.append(True)

    sse_iter = iter_sse_events(_closing_lines())
    assert next(sse_iter) == ("", '{"k": 1}')
    sse_iter.close()
    assert sse_closed == [True]


# 测试审查补齐的实网线协议细节：Gemini 零参工具与 URL 归一化、Anthropic null usage/非空文本块/温度夹紧、StreamOptions 覆写与 URL 密钥脱敏
def test_provider_wire_protocol_edge_cases_and_stream_options_override() -> None:
    import io
    from urllib.error import HTTPError

    from personal_assistant.ai.providers.http import format_http_error
    from personal_assistant.ai.registry import _merge_stream_options

    # 1. Gemini：零参数工具省略空 OBJECT.properties；剥离 models/ 前缀、URL 编码且尊重 /v1 与 /v1alpha
    gem_url_calls: list[tuple[str, dict[str, object]]] = []

    # 捕获 Gemini 请求地址与载荷并返回最小终止流
    def _cap_gem(url: str, payload: dict[str, object], headers: dict[str, str]) -> Iterator[str]:
        gem_url_calls.append((url, payload))
        return iter(['data: {"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}\n\n'])

    gem_provider = GoogleGenAIProvider(post_stream=_cap_gem)
    gem_model = ModelProfile(
        name="gem_v1",
        api="google-generative-ai",
        base_url="https://generativelanguage.googleapis.com/v1",
        model="models/gemini-2.5-flash",
        api_key="AIzaSyEdgeTestKey123456",
        reasoning="high",
    )
    gem_ctx = Context(
        messages=(UserMessage("现在几点"),),
        tools=(
            ToolSchema(name="get_current_time", description="获取时间", parameters={}),
            ToolSchema(name="echo", description="回显", parameters={"msg": "消息"}, required=()),
        ),
    )
    list(gem_provider.stream(gem_model, gem_ctx, StreamOptions(temperature=0.4)))
    gem_url, gem_payload = gem_url_calls[0]
    assert gem_url == "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
    decls = gem_payload["tools"][0]["functionDeclarations"]  # type: ignore[index]
    assert "parameters" not in decls[0]
    assert "required" not in decls[1]["parameters"]
    # 仅传 temperature 时不抹掉 model.reasoning="high"
    assert "thinkingConfig" in gem_payload["generationConfig"]  # type: ignore[operator]

    # 2. Anthropic：message_delta.usage 含 null 时保留 message_start 的输入与缓存 token；绝不发空文本块；temperature 夹紧 <=1.0
    anth_captured: dict[str, object] = {}

    # 模拟真实 Anthropic SSE（message_delta.usage 含 null 输入字段）
    def _cap_anth(url: str, payload: dict[str, object], headers: dict[str, str]) -> Iterator[str]:
        anth_captured["payload"] = payload
        return iter(
            [
                "event: message_start\n",
                'data: {"type":"message_start","message":{"usage":{"input_tokens":120,"cache_read_input_tokens":300,"cache_creation_input_tokens":50}}}\n',
                "\n",
                "event: content_block_start\n",
                'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n',
                "\n",
                "event: content_block_delta\n",
                'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"答复"}}\n',
                "\n",
                "event: content_block_stop\n",
                'data: {"type":"content_block_stop","index":0}\n',
                "\n",
                "event: message_delta\n",
                'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":null,"cache_read_input_tokens":null,"cache_creation_input_tokens":null,"output_tokens":18}}\n',
                "\n",
                "event: message_stop\n",
                'data: {"type":"message_stop"}\n',
                "\n",
            ]
        )

    anth_model = ModelProfile(
        name="claude_edge",
        api="anthropic-messages",
        base_url="https://api.anthropic.com",
        model="claude-sonnet-4-5",
        api_key="sk-ant-edge",
        temperature=1.8,
        reasoning="high",
        cache_retention="long",
    )
    anth_ctx = Context(
        messages=(
            UserMessage("第一句"),
            AssistantMessage(blocks=(ThinkingBlock(text="无签名旧思考"),)),
            UserMessage("第二句"),
        ),
    )
    # 显式传 reasoning="off", cache_retention="short" 必须真正覆写 model 默认的 high/long
    merged_override = _merge_stream_options(anth_model, StreamOptions(reasoning="off", cache_retention="short"))
    assert merged_override.reasoning == "off"
    assert merged_override.cache_retention == "short"

    anth_evs = list(AnthropicMessagesProvider(post_stream=_cap_anth).stream(anth_model, anth_ctx, merged_override))
    anth_payload = anth_captured["payload"]
    assert isinstance(anth_payload, dict)
    assert anth_payload["temperature"] == 1.0
    assert "thinking" not in anth_payload
    for wire_msg in anth_payload["messages"]:  # type: ignore[union-attr]
        for blk in wire_msg["content"]:
            if blk.get("type") == "text":
                assert blk["text"] != ""
    anth_done = anth_evs[-1]
    assert isinstance(anth_done, StreamDone)
    assert anth_done.usage == TokenUsage(prompt_tokens=470, cached_tokens=300, completion_tokens=18)

    # 3. format_http_error 空响应体时仍必须脱敏 error.url 中的 ?key=... 及 JSON 引号形式头密钥
    raw_err = HTTPError(
        "https://generativelanguage.googleapis.com/v1beta/models/m:stream?key=AIzaSyUrlSecret999",
        403,
        "Forbidden",
        {},
        io.BytesIO(b""),
    )
    fmt_err, _ = format_http_error(raw_err)
    assert "AIzaSyUrlSecret999" not in str(fmt_err.url)
    assert "proxy-secret-xyz" not in redact_error_tokens('{"x-api-key": "proxy-secret-xyz"}')
    assert clamp_thinking_level("off", ("low", "medium", "high")) == "off"
