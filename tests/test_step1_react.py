"""Unit tests for Step 1: minimal ReAct loop, ToolRegistry, and LLM client."""
from __future__ import annotations

import json

import pytest

from personal_assistant.cli import build_demo_tools
from personal_assistant.llm import (
    LLMClient,
    LLMNotConfigured,
    NoopLLMClient,
    OpenAICompatibleLLM,
    from_env,
)
from personal_assistant.react import run_react
from personal_assistant.tools import Tool, ToolRegistry


class ScriptedLLM(LLMClient):
    """Deterministic fake LLM that snapshots every request and replays scripted replies."""

    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = list(responses)
        self.requests: list[tuple[list[dict[str, object]], list[dict[str, object]] | None]] = []

    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        self.requests.append(([dict(msg) for msg in messages], tools))
        if not self._responses:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        return self._responses.pop(0)


def _tool_call(call_id: str, name: str, arguments: str) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def test_direct_answer_without_tool_calls() -> None:
    llm = ScriptedLLM([{"role": "assistant", "content": "你好，有什么可以帮你？"}])
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "你好"}]

    reply = run_react(llm, messages, build_demo_tools())

    assert reply == "你好，有什么可以帮你？"
    assert len(llm.requests) == 1
    assert llm.requests[0][1] is not None
    assert messages[-1] == {"role": "assistant", "content": "你好，有什么可以帮你？"}


def test_single_step_react_loop() -> None:
    steps: list[tuple[str, str, str]] = []
    tools = ToolRegistry(
        Tool(
            name="lookup_weather",
            description="查询城市天气",
            parameters={"city": "城市名称"},
            handler=lambda city: f"{city}：晴，22°C",
        )
    )
    llm = ScriptedLLM(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("call_1", "lookup_weather", '{"city": "上海"}')],
            },
            {"role": "assistant", "content": "上海今天晴，气温 22°C。"},
        ]
    )
    messages: list[dict[str, object]] = [{"role": "user", "content": "上海今天天气怎么样？"}]

    reply = run_react(
        llm,
        messages,
        tools,
        on_step=lambda name, args, obs: steps.append((name, args, obs)),
    )

    assert reply == "上海今天晴，气温 22°C。"
    assert steps == [("lookup_weather", '{"city": "上海"}', "上海：晴，22°C")]
    second_msgs = llm.requests[1][0]
    assert second_msgs[-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "上海：晴，22°C",
    }


def test_multi_step_chained_react_loop() -> None:
    steps: list[tuple[str, str, str]] = []
    tools = ToolRegistry(
        Tool(
            name="find_note",
            description="根据主题查找笔记路径",
            parameters={"topic": "主题"},
            handler=lambda topic: f"notes/{topic}.md",
        ),
        Tool(
            name="read_file",
            description="读取笔记内容",
            parameters={"path": "文件路径"},
            handler=lambda path: f"[{path}] 持有自旋锁时禁止 sleep。",
        ),
    )
    llm = ScriptedLLM(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("c1", "find_note", '{"topic": "spinlock"}')],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("c2", "read_file", '{"path": "notes/spinlock.md"}')],
            },
            {"role": "assistant", "content": "根据 notes/spinlock.md，持有自旋锁时禁止 sleep。"},
        ]
    )
    messages: list[dict[str, object]] = [{"role": "user", "content": "自旋锁能 sleep 吗？"}]

    reply = run_react(
        llm,
        messages,
        tools,
        on_step=lambda name, args, obs: steps.append((name, args, obs)),
    )

    assert "禁止 sleep" in reply
    assert [name for name, _, _ in steps] == ["find_note", "read_file"]
    assert len(llm.requests) == 3


def test_tool_error_as_observation_self_healing() -> None:
    steps: list[tuple[str, str, str]] = []
    tools = ToolRegistry(
        Tool(
            name="add_numbers",
            description="两数相加",
            parameters={"a": "第一个整数", "b": "第二个整数"},
            handler=lambda a, b: str(int(a) + int(b)),
        )
    )
    llm = ScriptedLLM(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _tool_call("bad_tool", "nonexistent", "{}"),
                    _tool_call("bad_json", "add_numbers", "{not-json"),
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("good", "add_numbers", '{"a": "19", "b": "23"}')],
            },
            {"role": "assistant", "content": "19 + 23 = 42。"},
        ]
    )
    messages: list[dict[str, object]] = [{"role": "user", "content": "算一下 19 加 23"}]

    reply = run_react(
        llm,
        messages,
        tools,
        on_step=lambda name, args, obs: steps.append((name, args, obs)),
    )

    assert reply == "19 + 23 = 42。"
    assert "未知工具" in steps[0][2]
    assert "JSON 解析失败" in steps[1][2]
    assert steps[2][2] == "42"


def test_max_rounds_limit_enforced() -> None:
    steps: list[tuple[str, str, str]] = []
    tools = ToolRegistry(
        Tool(
            name="ping",
            description="探测工具",
            parameters={},
            handler=lambda: "pong",
        )
    )
    llm = ScriptedLLM(
        [
            {"role": "assistant", "content": None, "tool_calls": [_tool_call("c1", "ping", "{}")]},
            {"role": "assistant", "content": None, "tool_calls": [_tool_call("c2", "ping", "{}")]},
            {"role": "assistant", "content": None, "tool_calls": [_tool_call("c3", "ping", "{}")]},
            {"role": "assistant", "content": None, "tool_calls": [_tool_call("c4", "ping", "{}")]},
            {"role": "assistant", "content": "已完成 4 轮探测，总结如下。"},
        ]
    )
    messages: list[dict[str, object]] = [{"role": "user", "content": "一直探测"}]

    reply = run_react(
        llm,
        messages,
        tools,
        max_rounds=4,
        on_step=lambda name, args, obs: steps.append((name, args, obs)),
    )

    assert reply == "已完成 4 轮探测，总结如下。"
    assert len(steps) == 4
    assert len(llm.requests) == 5
    assert all(llm.requests[i][1] is not None for i in range(4))
    assert llm.requests[4][1] is None
    last_tool_msg = llm.requests[4][0][-1]
    assert last_tool_msg["role"] == "tool"
    assert "已达到工具调用轮次上限" in str(last_tool_msg["content"])


def test_llm_client_from_env_and_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODE", "off")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    client = from_env()
    assert isinstance(client, NoopLLMClient)
    with pytest.raises(LLMNotConfigured):
        client.respond([{"role": "user", "content": "hi"}])

    captured: dict[str, object] = {}

    def fake_post(url: str, payload: dict[str, object], headers: dict[str, str]) -> dict[str, object]:
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    real_client = OpenAICompatibleLLM(
        api_key="sk-test",
        base_url="https://example.com/v1/",
        model="demo-model",
        post_json=fake_post,
    )
    msg = real_client.respond(
        [{"role": "user", "content": "hi"}],
        tools=build_demo_tools().definitions(),
    )
    assert msg == {"role": "assistant", "content": "ok"}
    assert captured["url"] == "https://example.com/v1/chat/completions"
    assert captured["headers"] == {"Authorization": "Bearer sk-test"}
    sent_payload = captured["payload"]
    assert isinstance(sent_payload, dict)
    assert sent_payload["model"] == "demo-model"
    assert sent_payload["tool_choice"] == "auto"
    assert json.loads(json.dumps(sent_payload["tools"]))[0]["function"]["name"] == "get_current_time"
