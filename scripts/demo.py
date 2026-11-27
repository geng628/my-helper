"""便捷测试与演示脚本：支持离线零 Token 演示与真实模型单句实测。"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from personal_assistant.agent.plugins import PluginManager
from personal_assistant.agent.react import AgentEvent, ToolStep, TurnEnd, run_react
from personal_assistant.agent.tools import ToolRegistry
from personal_assistant.ai import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    LLMNotConfigured,
    Message,
    StreamDone,
    StreamOptions,
    StreamStart,
    TextBlock,
    TextDelta,
    TextEnd,
    TextStart,
    ThinkingBlock,
    TokenUsage,
    ToolCall,
    ToolCallEnd,
    ToolCallStart,
    ToolResultMessage,
    UserMessage,
)
from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.assistant.common import load_dotenv
from personal_assistant.assistant.prompts import SYSTEM_PROMPT


# 从插件目录加载单句提问演示所需的时间工具注册表
def build_demo_tools() -> ToolRegistry:
    plugin_tools = PluginManager.from_entries(["plugins/tools"], ROOT).build_tools()
    return ToolRegistry(*plugin_tools, allowed_names=["get_current_time"])


# 离线脚本化模拟模型（实现 ModelInvoker）：按预设顺序返回中立 AssistantMessage，用于零成本演示 ReAct 循环
class DemoScriptedLLM:
    # 保存预设回复队列并初始化 last_usage
    def __init__(self, responses: list[AssistantMessage]) -> None:
        self._responses = list(responses)
        self.last_usage: TokenUsage | None = None

    # 弹出下一条预设中立消息并按统一流式事件协议产出
    def stream(
        self,
        context: Context,
        options: StreamOptions | None = None,
    ) -> Iterator[AssistantMessageEvent]:
        del context, options
        if not self._responses:
            raise RuntimeError("DemoScriptedLLM ran out of responses")
        msg = self._responses.pop(0)
        yield StreamStart(partial=AssistantMessage())
        for idx, block in enumerate(msg.blocks):
            if isinstance(block, TextBlock):
                yield TextStart(content_index=idx, partial=AssistantMessage())
                yield TextDelta(content_index=idx, text=block.text, partial=msg)
                yield TextEnd(content_index=idx, text=block.text, partial=msg)
            elif isinstance(block, ToolCall):
                yield ToolCallStart(content_index=idx, call_id=block.id, name=block.name, partial=msg)
                yield ToolCallEnd(content_index=idx, tool_call=block, partial=msg)
        reason = "toolUse" if msg.tool_calls else "stop"
        yield StreamDone(reason=reason, message=msg, usage=msg.usage)

    # 同步返回下一条预设中立消息
    def complete(
        self,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessage:
        for event in self.stream(context, options):
            if isinstance(event, StreamDone):
                return event.message
        raise RuntimeError("DemoScriptedLLM did not emit StreamDone")


# 将中立消息轨迹序列化为可读 JSON 列表，供 --show-messages 演示打印
def _format_messages_for_display(system_prompt: str, messages: tuple[Message, ...]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        if isinstance(msg, UserMessage):
            out.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AssistantMessage):
            blocks: list[dict[str, object]] = []
            for block in msg.blocks:
                if isinstance(block, TextBlock):
                    blocks.append({"type": "text", "text": block.text})
                elif isinstance(block, ThinkingBlock):
                    blocks.append({"type": "thinking", "text": block.text})
                elif isinstance(block, ToolCall):
                    blocks.append(
                        {"type": "tool_call", "id": block.id, "name": block.name, "arguments": block.arguments}
                    )
            out.append({"role": "assistant", "blocks": blocks})
        elif isinstance(msg, ToolResultMessage):
            out.append(
                {
                    "role": "tool_result",
                    "tool_call_id": msg.tool_call_id,
                    "tool_name": msg.tool_name,
                    "content": msg.content,
                    "is_error": msg.is_error,
                }
            )
    return out


# 创建事件监听器：实时打印工具步骤并在回合结束时捕获最终中立消息轨迹
def _make_event_logger(captured_messages: list[Message]) -> Callable[[AgentEvent], None]:
    # 处理单个 AgentEvent 事件
    def _on_event(event: AgentEvent) -> None:
        if isinstance(event, ToolStep):
            print(f"  [Act]     {event.tool}({event.arguments}) ({event.duration_ms:.1f}ms)")
            print(f"  [Observe] {event.observation}")
        elif isinstance(event, TurnEnd):
            captured_messages[:] = list(event.messages)

    return _on_event


# 运行两个离线演示场景：多轮链式工具调用 + 工具参数出错自愈
def run_offline_demo(show_messages: bool = False) -> None:
    tools = ToolRegistry(
        *PluginManager.from_entries(["plugins/tools"], ROOT).build_tools(),
        allowed_names=["get_current_time", "add_numbers"],
    )

    print("=== 场景 1：多轮 ReAct（先查时间，再做计算）===")
    llm1 = DemoScriptedLLM(
        [
            AssistantMessage(
                blocks=(ToolCall(id="call_1", name="get_current_time", arguments="{}"),),
                stop_reason="toolUse",
            ),
            AssistantMessage(
                blocks=(ToolCall(id="call_2", name="add_numbers", arguments='{"a": "19", "b": "23"}'),),
                stop_reason="toolUse",
            ),
            AssistantMessage.from_text(
                "现在是北京时间（见上方工具观察时间），另外 19 + 23 = 42。",
                stop_reason="stop",
            ),
        ]
    )
    q1 = "现在几点了？顺便帮我算一下 19 加 23。"
    ctx1 = Context(system_prompt=SYSTEM_PROMPT, messages=(UserMessage(content=q1),))
    traj1: list[Message] = []
    print(f"你> {q1}")
    reply1 = run_react(llm1, ctx1, tools, on_event=_make_event_logger(traj1))
    print(f"助手> {reply1}")
    if show_messages:
        print("\n--- 完整 messages 轨迹 ---")
        print(json.dumps(_format_messages_for_display(SYSTEM_PROMPT, tuple(traj1)), ensure_ascii=False, indent=2))

    print("\n=== 场景 2：工具参数出错后自愈（Error as Observation）===")
    llm2 = DemoScriptedLLM(
        [
            AssistantMessage(
                blocks=(ToolCall(id="bad_1", name="add_numbers", arguments='{"x": "10"}'),),
                stop_reason="toolUse",
            ),
            AssistantMessage(
                blocks=(ToolCall(id="fix_1", name="add_numbers", arguments='{"a": "10", "b": "32"}'),),
                stop_reason="toolUse",
            ),
            AssistantMessage.from_text("已修正参数并计算完成：10 + 32 = 42。", stop_reason="stop"),
        ]
    )
    q2 = "帮我算 10 加 32。"
    ctx2 = Context(system_prompt=SYSTEM_PROMPT, messages=(UserMessage(content=q2),))
    traj2: list[Message] = []
    print(f"你> {q2}")
    reply2 = run_react(llm2, ctx2, tools, on_event=_make_event_logger(traj2))
    print(f"助手> {reply2}")
    if show_messages:
        print("\n--- 完整 messages 轨迹 ---")
        print(json.dumps(_format_messages_for_display(SYSTEM_PROMPT, tuple(traj2)), ensure_ascii=False, indent=2))


# 统一通过 ToolsConfig 加载模型（支持 models.dsl / models.json / Slot 4 翻译器与 .env）并向真实模型发起单句提问
def run_live_ask(
    question: str,
    show_messages: bool = False,
    *,
    model_name: str | None = None,
) -> int:
    load_dotenv()
    cfg_path = ROOT / "config" / "tools.json"
    cfg = ToolsConfig.load(cfg_path, project_root=ROOT) if cfg_path.is_file() else ToolsConfig.empty(ROOT)
    try:
        client = cfg.build_llm(cfg_path.parent if cfg_path.is_file() else None, model_name=model_name)
        tools = (
            ToolRegistry(*cfg.plugin_manager.build_tools(), allowed_names=["get_current_time"])
            if cfg_path.is_file()
            else build_demo_tools()
        )
        context = Context(
            system_prompt=SYSTEM_PROMPT,
            messages=(UserMessage(content=question),),
        )
        traj: list[Message] = []
        print(f"你> {question}")
        try:
            reply = run_react(client, context, tools, on_event=_make_event_logger(traj))
        except LLMNotConfigured as error:
            print(f"助手> {error}")
            print("提示：不配置 API Key 也可直接运行 `uv run python scripts/demo.py` 查看离线演示。")
            return 1
        print(f"助手> {reply}")
        if show_messages:
            print("\n--- 完整 messages 轨迹 ---")
            print(json.dumps(_format_messages_for_display(SYSTEM_PROMPT, tuple(traj)), ensure_ascii=False, indent=2))
        return 0
    finally:
        cfg.plugin_manager.terminate_all()


# 演示脚本命令行入口：不传 --ask 跑离线演示，传 --ask 跑真实模型提问
def main() -> int:
    parser = argparse.ArgumentParser(description="测试与演示最小 ReAct 循环")
    parser.add_argument("--ask", help="使用 config/models.dsl 或 .env 中配置的真实模型测试单句提问")
    parser.add_argument("--model", default=None, help="指定 models.dsl 档案名、预设名或模型 ID")
    parser.add_argument(
        "--show-messages",
        action="store_true",
        help="打印完整中立 messages 对话上下文（含 tool_call 与 tool_result 观察消息）",
    )
    args = parser.parse_args()
    if args.ask:
        return run_live_ask(args.ask, show_messages=args.show_messages, model_name=args.model)
    run_offline_demo(show_messages=args.show_messages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
