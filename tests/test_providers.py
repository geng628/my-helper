"""Layer 1 Pi ch04 三层翻译架构专项单元测试：覆盖协议契约、注册表、OpenAI / DeepSeek 推理 / Anthropic / Gemini 四大原生翻译器。"""
from __future__ import annotations

import io
import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from personal_assistant.agent import ToolRegistry, TurnEnd, iter_react, tool
from personal_assistant.ai import (
    AnthropicMessagesProvider,
    ApiProviderRegistry,
    AssistantMessage,
    CacheMonitor,
    Context,
    GoogleGenAIProvider,
    LLMNotConfigured,
    Message,
    ModelClient,
    ModelProfile,
    NoopModelClient,
    OpenAICompletionsProvider,
    OpenAIReasoningProvider,
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
from personal_assistant.assistant.app import build_app
from personal_assistant.assistant.builtin_tools import MemoryToolsetProvider
from personal_assistant.assistant.common import REPO_ROOT
from personal_assistant.assistant.memory import MemoryStore
from plugins.telemetry import TelemetryObserver


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


# 测试全局默认注册表四大内建翻译器、同步抛 LLMNotConfigured、stream_simple 合并与 complete() 语义
def test_registry_builtin_providers_sync_auth_and_stream_simple() -> None:
    assert set(available_api_providers()) >= {
        "anthropic-messages",
        "google-generative-ai",
        "openai-completions",
        "openai-reasoning",
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


# 端到端跨协议热切换契约测试：同一条对话历史轨迹依次无缝穿越 Anthropic / OpenAI / Gemini / Reasoner 再回到 Anthropic
def test_mid_conversation_cross_protocol_hot_swap_end_to_end(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "data")
    registry = ToolRegistry(allowed_names=["remember_fact", "search_markdown", "add_numbers"])
    registry.register_provider(MemoryToolsetProvider(memory))

    # 自定义两数相加工具，验证真实工具执行观察在跨协议往返中保持无损
    @tool(description="将两个整数相加并返回结果", parameters={"a": "加数 A", "b": "加数 B"})
    def add_numbers(a: str, b: str) -> str:
        return str(int(a) + int(b))

    registry.register(add_numbers)

    # 1. 四套真实翻译器仅注入伪造 SSE 流并记录每次发往上游的完整载荷
    anth_payloads: list[dict[str, object]] = []
    anth_streams: list[list[str]] = [
        # Round 1 Step 1：Anthropic 产出带签名的 ThinkingBlock + 文本 + remember_fact 工具调用
        [
            "event: message_start\n",
            'data: {"type":"message_start","message":{"usage":{"input_tokens":150,"cache_read_input_tokens":0,"cache_creation_input_tokens":50,"output_tokens":1}}}\n',
            "\n",
            "event: content_block_start\n",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}\n',
            "\n",
            "event: content_block_delta\n",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"先记下用户的偏好"}}\n',
            "\n",
            "event: content_block_delta\n",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig_anth_r1"}}\n',
            "\n",
            "event: content_block_stop\n",
            'data: {"type":"content_block_stop","index":0}\n',
            "\n",
            "event: content_block_start\n",
            'data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}\n',
            "\n",
            "event: content_block_delta\n",
            'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"好的，我来记下。"}}\n',
            "\n",
            "event: content_block_stop\n",
            'data: {"type":"content_block_stop","index":1}\n',
            "\n",
            "event: content_block_start\n",
            'data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"anth_tc_1","name":"remember_fact","input":{}}}\n',
            "\n",
            "event: content_block_stop\n",
            'data: {"type":"content_block_stop","index":2}\n',
            "\n",
            "event: message_delta\n",
            'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":30}}\n',
            "\n",
            "event: message_stop\n",
            'data: {"type":"message_stop"}\n',
            "\n",
        ],
        # Round 1 Step 2：Anthropic 收到 tool_result 后给出本轮最终回复
        [
            "event: message_start\n",
            'data: {"type":"message_start","message":{"usage":{"input_tokens":100,"cache_read_input_tokens":200,"cache_creation_input_tokens":0,"output_tokens":1}}}\n',
            "\n",
            "event: content_block_start\n",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n',
            "\n",
            "event: content_block_delta\n",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"已记住你偏好深色主题。"}}\n',
            "\n",
            "event: content_block_stop\n",
            'data: {"type":"content_block_stop","index":0}\n',
            "\n",
            "event: message_delta\n",
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":20}}\n',
            "\n",
            "event: message_stop\n",
            'data: {"type":"message_stop"}\n',
            "\n",
        ],
        # Round 5 Step 1：切回 Anthropic 验证跳过 Gemini / Reasoner 无签名思考且零空文本块
        [
            "event: message_start\n",
            'data: {"type":"message_start","message":{"usage":{"input_tokens":120,"cache_read_input_tokens":700,"cache_creation_input_tokens":0,"output_tokens":1}}}\n',
            "\n",
            "event: content_block_start\n",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n',
            "\n",
            "event: content_block_delta\n",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"跨四协议轨迹确认无误。"}}\n',
            "\n",
            "event: content_block_stop\n",
            'data: {"type":"content_block_stop","index":0}\n',
            "\n",
            "event: message_delta\n",
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":20}}\n',
            "\n",
            "event: message_stop\n",
            'data: {"type":"message_stop"}\n',
            "\n",
        ],
    ]

    # 捕获 Anthropic 上游请求载荷并返回预设原生 SSE 帧
    def _fake_anth_stream(url: str, payload: dict[str, object], headers: dict[str, str]) -> Iterator[str]:
        del url, headers
        anth_payloads.append(payload)
        return iter(anth_streams.pop(0))

    oa_payloads: list[dict[str, object]] = []
    oa_streams: list[list[str]] = [
        # Round 2 Step 1：OpenAI Completions 发起 add_numbers 工具调用
        [
            'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"我来计算"}}]}\n',
            'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"oa_tc_1","type":"function","function":{"name":"add_numbers","arguments":"{\\"a\\":\\"6\\",\\"b\\":\\"7\\"}"}}]}}]}\n',
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n',
            'data: {"choices":[],"usage":{"prompt_tokens":380,"prompt_tokens_details":{"cached_tokens":256},"completion_tokens":25}}\n',
            "data: [DONE]\n",
        ],
        # Round 2 Step 2：OpenAI Completions 收到工具结果 13 后作答
        [
            'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"6 + 7 = 13。"}}]}\n',
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n',
            'data: {"choices":[],"usage":{"prompt_tokens":440,"prompt_tokens_details":{"cached_tokens":320},"completion_tokens":18}}\n',
            "data: [DONE]\n",
        ],
    ]

    # 捕获 OpenAI Completions 上游请求载荷并返回预设 SSE 帧
    def _fake_oa_stream(url: str, payload: dict[str, object], headers: dict[str, str]) -> Iterator[str]:
        del url, headers
        oa_payloads.append(payload)
        return iter(oa_streams.pop(0))

    gem_payloads: list[dict[str, object]] = []
    gem_streams: list[list[str]] = [
        # Round 3 Step 1：Google Gemini 产出原生 thought 与 search_markdown 函数调用
        [
            'data: {"candidates":[{"content":{"role":"model","parts":[{"thought":true,"text":"去记忆库检索用户偏好"},{"functionCall":{"id":"gem_tc_1","name":"search_markdown","args":{"query":"深色主题"}}}]}}],"usageMetadata":{"promptTokenCount":520,"cachedContentTokenCount":350,"candidatesTokenCount":20,"thoughtsTokenCount":15}}\n\n',
        ],
        # Round 3 Step 2：Google Gemini 收到 functionResponse 后作答
        [
            'data: {"candidates":[{"content":{"role":"model","parts":[{"thought":true,"text":"已找到深色主题记忆"},{"text":"查到了，你偏好深色主题。"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":610,"cachedContentTokenCount":450,"candidatesTokenCount":25,"thoughtsTokenCount":10}}\n\n',
        ],
    ]

    # 捕获 Gemini 上游请求载荷并返回预设原生 SSE 帧
    def _fake_gem_stream(url: str, payload: dict[str, object], headers: dict[str, str]) -> Iterator[str]:
        del url, headers
        gem_payloads.append(payload)
        return iter(gem_streams.pop(0))

    r1_payloads: list[dict[str, object]] = []
    r1_streams: list[list[str]] = [
        # Round 4 Step 1：OpenAI Reasoning 产出 reasoning_content + add_numbers 工具调用
        [
            'data: {"choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"先算 13 加 29"}}]}\n',
            'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"r1_tc_1","type":"function","function":{"name":"add_numbers","arguments":"{\\"a\\":\\"13\\",\\"b\\":\\"29\\"}"}}]}}]}\n',
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n',
            'data: {"choices":[],"usage":{"prompt_tokens":700,"prompt_cache_hit_tokens":512,"completion_tokens":40}}\n',
            "data: [DONE]\n",
        ],
        # Round 4 Step 2：OpenAI Reasoning 收到工具结果 42 后给出总结
        [
            'data: {"choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"核对最终结果"}}]}\n',
            'data: {"choices":[{"index":0,"delta":{"content":"13 + 29 = 42，偏好已核实。"}}]}\n',
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n',
            'data: {"choices":[],"usage":{"prompt_tokens":780,"prompt_cache_hit_tokens":640,"completion_tokens":32}}\n',
            "data: [DONE]\n",
        ],
    ]

    # 捕获 OpenAI Reasoning 上游请求载荷并返回预设 SSE 帧
    def _fake_r1_stream(url: str, payload: dict[str, object], headers: dict[str, str]) -> Iterator[str]:
        del url, headers
        r1_payloads.append(payload)
        return iter(r1_streams.pop(0))

    local_reg = ApiProviderRegistry()
    local_reg.register(AnthropicMessagesProvider(post_stream=_fake_anth_stream))
    local_reg.register(OpenAICompletionsProvider(post_stream=_fake_oa_stream))
    local_reg.register(GoogleGenAIProvider(post_stream=_fake_gem_stream))
    local_reg.register(OpenAIReasoningProvider(post_stream=_fake_r1_stream))

    profiles = {
        "anthropic": ModelProfile(
            name="claude",
            api="anthropic-messages",
            base_url="https://api.anthropic.com",
            model="claude-sonnet-4-5",
            api_key="sk-ant-swap",
            reasoning="low",
            cache_retention="short",
        ),
        "openai": ModelProfile(
            name="gpt",
            api="openai-completions",
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
            api_key="sk-oa-swap",
            reasoning="medium",
        ),
        "gemini": ModelProfile(
            name="gem",
            api="google-generative-ai",
            base_url="https://generativelanguage.googleapis.com",
            model="gemini-2.5-flash",
            api_key="AIzaSySwapTestKey123456",
            reasoning="low",
        ),
        "reasoner": ModelProfile(
            name="r1",
            api="openai-reasoning",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-reasoner",
            api_key="sk-r1-swap",
        ),
    }

    monitor = CacheMonitor()
    observer = TelemetryObserver()
    session_key = "hot-swap-session"
    system_prompt = "你是一个跨协议无缝协作的个人助手。"
    trajectory: list[Message] = []

    # 沿同一条中立对话轨迹热切换目标协议并执行一轮完整 iter_react 推理与观测
    def _run_round(provider_key: str, user_text: str) -> str:
        trajectory.append(UserMessage(content=user_text))
        client = ModelClient(profile=profiles[provider_key], registry=local_reg)
        tracked = monitor.wrap(client, session_key=session_key)
        ctx = Context(system_prompt=system_prompt, messages=tuple(trajectory), tools=registry.schemas())
        reply = ""
        for ev in iter_react(tracked, ctx, registry, tool_context={"user_text": user_text}):
            observer.on_event(session_key, ev)
            if isinstance(ev, TurnEnd):
                reply = ev.reply
                trajectory[:] = list(ev.messages)
        return reply

    # Round 1：Anthropic（带签名思考 + 真实 remember_fact 工具写入 facts.md）
    r1_reply = _run_round("anthropic", "请记住我偏好深色主题")
    assert r1_reply == "已记住你偏好深色主题。"
    assert "偏好深色主题" in (tmp_path / "data" / "memory" / "facts.md").read_text(encoding="utf-8")
    assert anth_payloads[0]["system"] == [
        {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
    ]
    anth_r1_tools = anth_payloads[0]["tools"]
    assert isinstance(anth_r1_tools, list) and all("input_schema" in t for t in anth_r1_tools)
    anth_step2_msgs = anth_payloads[1]["messages"]
    assert isinstance(anth_step2_msgs, list)
    assert anth_step2_msgs[1]["content"][0] == {
        "type": "thinking",
        "thinking": "先记下用户的偏好",
        "signature": "sig_anth_r1",
    }
    assert anth_step2_msgs[2]["content"][0]["type"] == "tool_result"
    r1_last = observer.snapshot(session_key=session_key)["last_turn"]
    assert isinstance(r1_last, dict) and r1_last["tools"] == ["remember_fact"]

    # Round 2：热切换至 OpenAI Completions（接收 Anthropic 产出的中立轨迹 + 执行真实 add_numbers）
    r2_reply = _run_round("openai", "帮我算一下 6 加 7")
    assert r2_reply == "6 + 7 = 13。"
    oa_step1_msgs = oa_payloads[0]["messages"]
    assert isinstance(oa_step1_msgs, list)
    assert [m["role"] for m in oa_step1_msgs] == ["system", "user", "assistant", "tool", "assistant", "user"]
    assert oa_step1_msgs[2]["tool_calls"][0]["id"] == "anth_tc_1"
    assert oa_step1_msgs[2]["tool_calls"][0]["function"]["name"] == "remember_fact"
    assert oa_step1_msgs[3]["tool_call_id"] == "anth_tc_1"
    assert "facts.md" in str(oa_step1_msgs[3]["content"])
    assert all("reasoning_content" not in m for m in oa_step1_msgs)
    oa_step2_msgs = oa_payloads[1]["messages"]
    assert isinstance(oa_step2_msgs, list)
    assert oa_step2_msgs[-1] == {"role": "tool", "tool_call_id": "oa_tc_1", "content": "13"}
    r2_last = observer.snapshot(session_key=session_key)["last_turn"]
    assert isinstance(r2_last, dict) and r2_last["tools"] == ["add_numbers"]

    # Round 3：热切换至 Google Gemini（接收 Anthropic + OpenAI 轨迹，校验 model/parts/functionResponse + 真实 search_markdown）
    r3_reply = _run_round("gemini", "查一下我刚才记了什么偏好")
    assert r3_reply == "查到了，你偏好深色主题。"
    gem_step1_payload = gem_payloads[0]
    assert gem_step1_payload["systemInstruction"] == {"parts": [{"text": system_prompt}]}
    gem_tools = gem_step1_payload["tools"]
    assert isinstance(gem_tools, list)
    gem_decls = gem_tools[0]["functionDeclarations"]
    remember_decl = next(d for d in gem_decls if d["name"] == "remember_fact")
    assert "parameters" not in remember_decl
    gem_contents = gem_step1_payload["contents"]
    assert isinstance(gem_contents, list)
    assert [c["role"] for c in gem_contents] == [
        "user",
        "model",
        "user",
        "model",
        "user",
        "model",
        "user",
        "model",
        "user",
    ]
    assert gem_contents[1]["parts"][0] == {
        "thought": True,
        "text": "先记下用户的偏好",
        "thoughtSignature": "sig_anth_r1",
    }
    assert gem_contents[5]["parts"][1] == {"functionCall": {"name": "add_numbers", "args": {"a": "6", "b": "7"}}}
    assert gem_contents[6]["parts"][0] == {
        "functionResponse": {"name": "add_numbers", "response": {"result": "13"}}
    }
    # Step 2 收到真实 search_markdown 命中的 facts.md 内容
    gem_step2_contents = gem_payloads[1]["contents"]
    assert isinstance(gem_step2_contents, list)
    gem_fr = gem_step2_contents[-1]["parts"][0]["functionResponse"]
    assert gem_fr["name"] == "search_markdown"
    assert "偏好深色主题" in str(gem_fr["response"]["result"])
    r3_last = observer.snapshot(session_key=session_key)["last_turn"]
    assert isinstance(r3_last, dict) and r3_last["tools"] == ["search_markdown"]

    # Round 4：热切换至 OpenAI Reasoning（接收前 3 厂商轨迹，校验跨轮剥离旧思维链、轮内续传当前 reasoning_content）
    r4_reply = _run_round("reasoner", "再算一下 13 加 29")
    assert r4_reply == "13 + 29 = 42，偏好已核实。"
    assert "temperature" not in r1_payloads[0]
    r1_step1_msgs = r1_payloads[0]["messages"]
    assert isinstance(r1_step1_msgs, list)
    assert all("reasoning_content" not in m for m in r1_step1_msgs)
    assert any(
        m.get("role") == "tool" and m.get("tool_call_id") == "gem_tc_1" and "偏好深色主题" in str(m.get("content"))
        for m in r1_step1_msgs
    )
    r1_step2_msgs = r1_payloads[1]["messages"]
    assert isinstance(r1_step2_msgs, list)
    # 跨轮历史中的旧 assistant 消息不带 reasoning_content；仅当前轮工具调用消息携带本轮 reasoning_content
    reasoning_msgs = [m for m in r1_step2_msgs if "reasoning_content" in m]
    assert len(reasoning_msgs) == 1
    assert reasoning_msgs[0]["reasoning_content"] == "先算 13 加 29"
    assert r1_step2_msgs[-1] == {"role": "tool", "tool_call_id": "r1_tc_1", "content": "42"}
    r4_last = observer.snapshot(session_key=session_key)["last_turn"]
    assert isinstance(r4_last, dict) and r4_last["tools"] == ["add_numbers"]

    # Round 5：切回 Anthropic，验证跨协议无签名思考（来自 Gemini / Reasoner）被安全跳过、有签名思考保留且绝不发出空文本块
    r5_reply = _run_round("anthropic", "最后确认一下前面所有记录")
    assert r5_reply == "跨四协议轨迹确认无误。"
    anth_r5_msgs = anth_payloads[2]["messages"]
    assert isinstance(anth_r5_msgs, list)
    thinking_blocks_sent: list[dict[str, object]] = []
    for wire_msg in anth_r5_msgs:
        content_list = wire_msg["content"]
        assert isinstance(content_list, list) and len(content_list) > 0
        for blk in content_list:
            if blk.get("type") == "text":
                assert blk["text"] != ""
            if blk.get("type") == "thinking":
                assert blk.get("signature")
                thinking_blocks_sent.append(blk)
    # 仅有 Round 1 Anthropic 自带签名的思考块被保留，Gemini / Reasoner 的无签名思考全部跳过
    assert thinking_blocks_sent == [
        {"type": "thinking", "thinking": "先记下用户的偏好", "signature": "sig_anth_r1"}
    ]

    # 2. 验证同一套 CacheMonitor 与 TelemetryObserver 全程跨 4 协议统计完全一致
    cache_summary = monitor.summary(session_key=session_key)
    assert cache_summary["llm_calls"] == 9
    assert cache_summary["warm_calls"] == 8
    assert isinstance(cache_summary["mean_prefix_reuse_rate"], float)
    assert cache_summary["mean_prefix_reuse_rate"] > 0.5
    assert cache_summary["prompt_tokens"] == 4750
    assert cache_summary["cached_tokens"] == 3428
    assert cache_summary["completion_tokens"] == 255

    obs_snap = observer.snapshot(session_key=session_key)
    assert obs_snap["cursor"] == 5
    totals = obs_snap["totals"]
    assert isinstance(totals, dict)
    assert totals["turns"] == 5
    assert totals["llm_calls"] == 9
    assert totals["tool_calls"] == 4
    assert totals["prompt_tokens"] == 4750
    assert totals["cached_tokens"] == 3428
    assert totals["completion_tokens"] == 255


# 单点扩展契约测试：新增第 5 套私有协议（custom-acme-rpc）仅需 1 个 Slot 4 翻译器插件 + models.dsl，零核心改动跑通全链路
def test_new_fifth_protocol_extension_requires_only_one_translator_and_registration(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    shutil.copy2(REPO_ROOT / "plugins" / "models_dsl.py", plugins_dir / "models_dsl.py")
    shutil.copy2(REPO_ROOT / "plugins" / "telemetry.py", plugins_dir / "telemetry.py")

    # 动态写入全新的第 5 套私有协议翻译器（Slot 4 主板插件）
    (plugins_dir / "acme_provider.py").write_text(
        '''"""第 5 套虚构私有协议翻译器插件（custom-acme-rpc）：验证零核心改动单点扩展。"""
from __future__ import annotations

from collections.abc import Iterator

from personal_assistant.agent.plugins import Plugin
from personal_assistant.ai import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
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
)


# 虚构 ACME RPC 协议翻译器：产出全部 12 种规范流事件，覆盖正常工具流、上游 StreamError 与恢复对话
class AcmeRpcProvider:
    api = "custom-acme-rpc"

    # 初始化调用计数器与上下文/选项捕获列表
    def __init__(self) -> None:
        self.call_count = 0
        self.recorded_contexts: list[Context] = []
        self.recorded_options: list[StreamOptions | None] = []

    # 产出规范 12 事件流（第 1 轮记忆工具调用成功 -> 第 2 轮工具后遇 StreamError -> 第 3 轮恢复正常）
    def stream(
        self,
        model: ModelProfile,
        context: Context,
        options: StreamOptions | None = None,
    ) -> Iterator[AssistantMessageEvent]:
        del model
        self.call_count += 1
        self.recorded_contexts.append(context)
        self.recorded_options.append(options)
        step = self.call_count

        yield StreamStart(partial=AssistantMessage())

        if step == 1:
            think = ThinkingBlock(text="ACME 正在规划记忆写入")
            tc = ToolCall(id="acme_tc_1", name="remember_fact", arguments="{}")
            msg = AssistantMessage(blocks=(think, tc), stop_reason="toolUse", usage=TokenUsage(120, 0, 20))
            yield ThinkingStart(content_index=0, partial=AssistantMessage())
            yield ThinkingDelta(content_index=0, text=think.text, partial=AssistantMessage(blocks=(think,)))
            yield ThinkingEnd(content_index=0, text=think.text, partial=AssistantMessage(blocks=(think,)))
            yield ToolCallStart(content_index=1, call_id=tc.id, name=tc.name, partial=AssistantMessage(blocks=(think,)))
            yield ToolCallDelta(content_index=1, delta="{}", partial=msg)
            yield ToolCallEnd(content_index=1, tool_call=tc, partial=msg)
            yield StreamDone(reason="toolUse", message=msg, usage=msg.usage, ttft_ms=5.0, latency_ms=12.0)
            return

        if step == 2:
            think = ThinkingBlock(text="已收到工具观察")
            text = TextBlock(text="ACME 已记住项目代号北极星。")
            msg = AssistantMessage(blocks=(think, text), stop_reason="stop", usage=TokenUsage(180, 120, 15))
            yield ThinkingStart(content_index=0, partial=AssistantMessage())
            yield ThinkingDelta(content_index=0, text=think.text, partial=AssistantMessage(blocks=(think,)))
            yield ThinkingEnd(content_index=0, text=think.text, partial=AssistantMessage(blocks=(think,)))
            yield TextStart(content_index=1, partial=AssistantMessage(blocks=(think,)))
            yield TextDelta(content_index=1, text=text.text, partial=msg)
            yield TextEnd(content_index=1, text=text.text, partial=msg)
            yield StreamDone(reason="stop", message=msg, usage=msg.usage, ttft_ms=4.0, latency_ms=10.0)
            return

        if step == 3:
            tc = ToolCall(id="acme_tc_2", name="remember_fact", arguments="{}")
            msg = AssistantMessage(blocks=(tc,), stop_reason="toolUse", usage=TokenUsage(220, 160, 18))
            yield ToolCallStart(content_index=0, call_id=tc.id, name=tc.name, partial=AssistantMessage())
            yield ToolCallDelta(content_index=0, delta="{}", partial=msg)
            yield ToolCallEnd(content_index=0, tool_call=tc, partial=msg)
            yield StreamDone(reason="toolUse", message=msg, usage=msg.usage, ttft_ms=3.0, latency_ms=9.0)
            return

        if step == 4:
            err_msg = AssistantMessage(stop_reason="error", error_message="ACME RPC 503 upstream overloaded")
            yield StreamError(
                reason="error",
                message=err_msg,
                error_message="ACME RPC 503 upstream overloaded",
                latency_ms=6.0,
            )
            return

        think = ThinkingBlock(text="确认 ACME 连接已恢复")
        text = TextBlock(text="ACME 协议已恢复正常。")
        msg = AssistantMessage(blocks=(think, text), stop_reason="stop", usage=TokenUsage(250, 180, 12))
        yield ThinkingStart(content_index=0, partial=AssistantMessage())
        yield ThinkingDelta(content_index=0, text=think.text, partial=AssistantMessage(blocks=(think,)))
        yield ThinkingEnd(content_index=0, text=think.text, partial=AssistantMessage(blocks=(think,)))
        yield TextStart(content_index=1, partial=AssistantMessage(blocks=(think,)))
        yield TextDelta(content_index=1, text=text.text, partial=msg)
        yield TextEnd(content_index=1, text=text.text, partial=msg)
        yield StreamDone(reason="stop", message=msg, usage=msg.usage, ttft_ms=2.0, latency_ms=8.0)


# ACME 私有协议主板插件：经 Slot 4 register_provider 单点接入
class AcmeProviderPlugin(Plugin):
    plugin_id = "custom-acme-rpc"

    # 注册第 5 套私有协议翻译器
    def initialize(self) -> None:
        self.context.register_provider(AcmeRpcProvider())
''',
        encoding="utf-8",
    )

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "tools.json").write_text(
        json.dumps(
            {
                "enabled_tools": ["remember_fact", "search_markdown", "read_markdown"],
                "plugins": ["plugins"],
                "telemetry": {"enabled": True, "max_sessions": 16},
                "files": {"read_roots": ["data/memory", "data/workspace"], "write_root": "data/workspace"},
                "commands": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (cfg_dir / "models.dsl").write_text(
        "default acme\n\n"
        "model acme {\n"
        "  api             custom-acme-rpc\n"
        "  base_url        https://rpc.acme.example/v1\n"
        "  model           acme-v1\n"
        "  api_key         acme-secret-key\n"
        "  max_tokens      4096\n"
        "  reasoning       medium\n"
        "  cache_retention long\n"
        "}\n",
        encoding="utf-8",
    )

    app = build_app(
        data_root=tmp_path / "data",
        config_root=cfg_dir,
        project_root=tmp_path,
        auto_consolidate=False,
    )
    try:
        # 1. 确认由 models.dsl + Slot 4 插件自动装配，且未污染进程全局默认注册表
        assert "custom-acme-rpc" in app.tools_config.plugin_manager.providers
        assert "custom-acme-rpc" not in available_api_providers()
        assert isinstance(app.llm, ModelClient)
        assert app.llm.api == "custom-acme-rpc"
        assert app.llm.model == "acme-v1"
        acme_provider = app.tools_config.plugin_manager.providers["custom-acme-rpc"]

        # 2. Turn 1：思考流 + 真实 remember_fact 工具调用 + 正常落库与观测
        frames1 = list(app.stream_chat("请记住项目代号是北极星", chat_id="acme-chat"))
        assert [f["text"] for f in frames1 if f["type"] == "thinking_delta"] == [
            "ACME 正在规划记忆写入",
            "已收到工具观察",
        ]
        assert any(f["type"] == "tool_start" and f["tool"] == "remember_fact" for f in frames1)
        assert any(f["type"] == "tool_step" and f["tool"] == "remember_fact" for f in frames1)
        done1 = next(f for f in frames1 if f["type"] == "done")
        assert done1["reply"] == "ACME 已记住项目代号北极星。"
        assert done1["thinking"] == "ACME 正在规划记忆写入已收到工具观察"

        facts_path = tmp_path / "data" / "memory" / "facts.md"
        facts_after_t1 = facts_path.read_text(encoding="utf-8")
        assert "项目代号是北极星" in facts_after_t1
        assert "(msg#" in facts_after_t1
        assert "(turn#" not in facts_after_t1
        assert len(app.state.active_messages("acme-chat")) == 2

        # 3. Turn 2：Step 1 写暂存事实，Step 2 触发 StreamError -> 必须完整回滚消息与暂存事实、不污染遥测
        with pytest.raises(RuntimeError, match="ACME RPC 503 upstream overloaded"):
            list(app.stream_chat("请记住这条脏事实必须回滚", chat_id="acme-chat"))

        assert len(app.state.active_messages("acme-chat")) == 2
        facts_after_t2 = facts_path.read_text(encoding="utf-8")
        assert "这条脏事实必须回滚" not in facts_after_t2
        assert "(turn#" not in facts_after_t2
        obs_after_t2 = app.observers[0].snapshot(session_key="acme-chat")
        assert obs_after_t2["cursor"] == 1
        sess_after_t2 = obs_after_t2["session"]
        assert isinstance(sess_after_t2, dict)
        assert sess_after_t2["turns"] == 1
        assert sess_after_t2["llm_calls"] == 2
        assert sess_after_t2["tool_calls"] == 1
        # StreamError 步骤不写入 CacheMonitor（此时仅记录 Turn 1 的 2 次成功调用 + Turn 2 Step 1 的 1 次成功调用）
        assert app.cache_monitor.summary(session_key="acme-chat")["llm_calls"] == 3

        # 4. Turn 3：正常恢复对话，校验历史无脏消息且遥测精确仅含 Turn 3 指标
        res3 = app.chat("现在恢复正常了吗？", chat_id="acme-chat")
        assert res3["reply"] == "ACME 协议已恢复正常。"
        assert res3["thinking"] == "确认 ACME 连接已恢复"

        persisted = app.state.active_messages("acme-chat")
        assert [m["content"] for m in persisted] == [
            "请记住项目代号是北极星",
            "ACME 已记住项目代号北极星。",
            "现在恢复正常了吗？",
            "ACME 协议已恢复正常。",
        ]

        # 验证 Turn 3 传给翻译器的中立 Context 无任何 Turn 2 残留，且 StreamOptions 继承 models.dsl 配置并透传 session_id
        recorded_contexts = getattr(acme_provider, "recorded_contexts")
        recorded_options = getattr(acme_provider, "recorded_options")
        t3_msgs = recorded_contexts[-1].messages
        assert len(t3_msgs) == 3
        assert isinstance(t3_msgs[0], UserMessage) and t3_msgs[0].content.endswith("\n请记住项目代号是北极星")
        assert t3_msgs[1] == AssistantMessage.from_text("ACME 已记住项目代号北极星。")
        assert isinstance(t3_msgs[2], UserMessage) and t3_msgs[2].content.endswith("\n现在恢复正常了吗？")
        assert t3_msgs[0] == recorded_contexts[0].messages[0]
        assert recorded_options[-1] == StreamOptions(
            temperature=0.2,
            max_tokens=4096,
            reasoning="medium",
            cache_retention="long",
            session_id="acme-chat",
        )

        # 验证 Turn 3 遥测未被失败 Turn 2 污染，且 CacheMonitor 记录完整 4 次成功调用
        assert app.cache_monitor.summary(session_key="acme-chat")["llm_calls"] == 4
        t3_tel_root = res3["telemetry"]
        assert isinstance(t3_tel_root, dict)
        t3_tel = t3_tel_root["telemetry"]
        assert isinstance(t3_tel, dict)
        t3_last = t3_tel["last_turn"]
        assert isinstance(t3_last, dict)
        assert t3_last["seq"] == 2
        assert t3_last["llm_calls"] == 1
        assert t3_last["tool_calls"] == 0
        assert t3_last["prompt_tokens"] == 250
        assert t3_last["cached_tokens"] == 180
        assert t3_last["completion_tokens"] == 12
        t3_totals = t3_tel["totals"]
        assert isinstance(t3_totals, dict)
        assert t3_totals["turns"] == 2
        assert t3_totals["llm_calls"] == 3
        assert t3_totals["tool_calls"] == 1
        assert t3_totals["prompt_tokens"] == 550
    finally:
        app.close()
