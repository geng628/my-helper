"""全链路 KV Cache 架构升级单元测试（Issue #95）：失配归因、选项透传、动静分离、空闲 TTL、按块裁剪与非破坏性投影。"""
from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path

from personal_assistant.agent import (
    Tool,
    ToolRegistry,
    TurnEnd,
    iter_react,
    run_react,
)
from personal_assistant.ai import (
    AssistantMessage,
    AssistantMessageEvent,
    CacheMonitor,
    Context,
    StreamDone,
    StreamOptions,
    TextBlock,
    TokenUsage,
    ToolCall,
    ToolResultMessage,
    ToolSchema,
    UserMessage,
    context_composition,
    format_cache_summary,
)
from personal_assistant.assistant.app import AssistantApp, _select_chunked_history
from personal_assistant.assistant.consolidator import ConsolidationPolicy
from personal_assistant.assistant.prompts import format_system_prompt, format_user_envelope


# 录制型测试模型：保存每轮收到的 Context 与 StreamOptions，并按剧本回放流式响应
class _OptionsRecordingLLM:
    # 初始化预设回复序列并记录每次请求上下文与选项
    def __init__(self, replies: list[AssistantMessage]) -> None:
        self._replies = list(replies)
        self.requests: list[Context] = []
        self.options_seen: list[StreamOptions | None] = []
        self.last_usage: TokenUsage | None = None

    # 弹出下一条预设回复（列表耗尽时返回简单兜底文本）
    def _next_reply(self) -> AssistantMessage:
        if self._replies:
            return self._replies.pop(0)
        return AssistantMessage.from_text("收到", stop_reason="stop")

    # 记录上下文与选项并以流式 StreamDone 返回
    def stream(
        self,
        context: Context,
        options: StreamOptions | None = None,
    ) -> Iterator[AssistantMessageEvent]:
        self.requests.append(context)
        self.options_seen.append(options)
        msg = self._next_reply()
        self.last_usage = msg.usage
        reason = "toolUse" if msg.tool_calls else "stop"
        yield StreamDone(reason=reason, message=msg, usage=msg.usage)

    # 同步调用封装
    def complete(
        self,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessage:
        for ev in self.stream(context, options):
            if isinstance(ev, StreamDone):
                return ev.message
        raise AssertionError("未收到 StreamDone")


# 验收 1：CacheMonitor 精准归因 tools / system / message[0]:user 失配、统计上下文构成，且 TrackingModelClient 优先遵循 options.session_id
def test_cache_monitor_attributes_prefix_breaks_and_context_composition() -> None:
    monitor = CacheMonitor()
    t1 = (ToolSchema(name="read_file", description="读文件", parameters={"path": "路径"}),)
    t2 = (
        ToolSchema(name="read_file", description="读文件", parameters={"path": "路径"}),
        ToolSchema(name="search", description="检索", parameters={"q": "词"}),
    )
    sys_a = "你是个人助手。" + "静态指令" * 30
    sys_b = "你是个人助手（改了）。" + "静态指令" * 30

    llm = _OptionsRecordingLLM(
        [
            AssistantMessage.from_text("答1", usage=TokenUsage(100, 0, 10)),
            AssistantMessage.from_text("答2", usage=TokenUsage(140, 100, 10)),
            AssistantMessage.from_text("答3", usage=TokenUsage(150, 40, 10)),
            AssistantMessage.from_text("答4", usage=TokenUsage(160, 0, 10)),
            AssistantMessage.from_text("答5", usage=TokenUsage(170, 80, 10)),
            AssistantMessage.from_text("另会话答1", usage=TokenUsage(100, 0, 10)),
        ]
    )
    tracked = monitor.wrap(llm, session_key="default-session")

    c1 = Context(system_prompt=sys_a, messages=(UserMessage("第一问"),), tools=t1)
    tracked.complete(c1)
    assert monitor.records[0].break_section == "cold_start"

    # 纯追加：break_section == "append"，不计入 prefix_breaks
    c2 = Context(
        system_prompt=sys_a,
        messages=(
            UserMessage("第一问"),
            AssistantMessage(blocks=(TextBlock("查一下"), ToolCall("c1", "read_file", '{"path":"a.md"}'))),
            ToolResultMessage("c1", "read_file", "文件正文内容"),
        ),
        tools=t1,
    )
    tracked.complete(c2)
    assert monitor.records[1].break_section == "append"
    comp = context_composition(c2)
    assert comp["tools"] > 0
    assert comp["system"] == len(sys_a)
    assert comp["user"] == len("第一问")
    assert comp["assistant"] > 0
    assert comp["tool_results"] == len("文件正文内容")
    assert monitor.records[1].composition == comp

    # 改动 system -> 精准指认 "system"
    c3 = Context(system_prompt=sys_b, messages=c2.messages, tools=t1)
    tracked.complete(c3)
    assert monitor.records[2].break_section == "system"

    # 改动 tools -> 精准指认 "tools"
    c4 = Context(system_prompt=sys_b, messages=c2.messages, tools=t2)
    tracked.complete(c4)
    assert monitor.records[3].break_section == "tools"

    # 改动队头消息（模拟队头滑窗）-> 精准指认 "message[0]:user"
    c5 = Context(
        system_prompt=sys_b,
        messages=(UserMessage("队头变了"), *c2.messages[1:]),
        tools=t2,
    )
    tracked.complete(c5)
    assert monitor.records[4].break_section == "message[0]:user"

    # 通过 StreamOptions.session_id 覆盖会话键时独立冷启动，不污染 default-session
    tracked.complete(c1, options=StreamOptions(session_id="override-session"))
    assert monitor.records[5].session_key == "override-session"
    assert monitor.records[5].break_section == "cold_start"

    summary = monitor.summary(session_key="default-session")
    assert summary["prefix_breaks"] == 3
    assert summary["last_break_section"] == "message[0]:user"
    assert summary["last_composition"] == context_composition(c5)
    text = format_cache_summary(summary)
    assert "前缀失配 3 次" in text
    assert "message[0]:user" in text


# 验收 2：Layer 2 iter_react 与 run_react 完整透传 StreamOptions 到 ModelInvoker.stream
def test_iter_react_and_run_react_forward_stream_options() -> None:
    opts = StreamOptions(session_id="sess-42", cache_retention="long", reasoning="low")
    llm = _OptionsRecordingLLM(
        [
            AssistantMessage(blocks=(ToolCall("t1", "echo", '{"text":"hi"}'),), stop_reason="toolUse"),
            AssistantMessage.from_text("完成", stop_reason="stop"),
            AssistantMessage.from_text("二次完成", stop_reason="stop"),
        ]
    )
    reg = ToolRegistry(
        Tool(name="echo", description="回显", parameters={"text": "文本"}, handler=lambda text: f"ok:{text}")
    )
    ctx = Context(system_prompt="sys", messages=(UserMessage("测选项"),))

    events = list(iter_react(llm, ctx, reg, options=opts))
    assert any(isinstance(ev, TurnEnd) and ev.reply == "完成" for ev in events)
    assert llm.options_seen == [opts, opts]

    reply = run_react(llm, ctx, reg, options=opts)
    assert reply == "二次完成"
    assert llm.options_seen[-1] == opts


# 验收 3：动静彻底分离——日期移出 system_prompt 进入用户消息信封，跨自然日（无过期归档）零击穿缓存
def test_static_system_prompt_and_cross_midnight_append_cache_hit(tmp_path: Path) -> None:
    current_day = date(2026, 11, 26)
    fake_now = [100.0]
    llm = _OptionsRecordingLLM(
        [
            AssistantMessage.from_text("第一天回复", stop_reason="stop"),
            AssistantMessage.from_text("第二天回复", stop_reason="stop"),
        ]
    )
    app = AssistantApp(
        data_root=tmp_path / "data",
        llm=llm,
        today_fn=lambda: current_day,
        clock=lambda: fake_now[0],
        auto_consolidate=False,
    )
    try:
        sys_text = format_system_prompt(app.memory, tool_names=app.tools.names)
        assert "今天是" not in sys_text
        assert "2026-11-26" not in sys_text
        assert "2026-11-26（星期四，北京时间）" in format_user_envelope("你好", today=current_day)

        r1 = app.chat("第一天提问", chat_id="cross-day")
        assert r1["reply"] == "第一天回复"
        assert llm.options_seen[0] == StreamOptions(session_id="cross-day")
        req1 = llm.requests[0]
        assert "2026-11-26" not in req1.system_prompt
        assert isinstance(req1.messages[-1], UserMessage)
        assert "2026-11-26（星期四，北京时间）" in req1.messages[-1].content
        assert "第一天提问" in req1.messages[-1].content

        # 数据库中保存的仍是干净原话，不带日期信封外壳
        saved = app.state.active_messages("cross-day")
        assert saved[0]["content"] == "第一天提问"

        # 跨自然日且仍在热缓存窗口内（+60s）、无过期记忆需归档：system_prompt 零变化且历史用户消息逐字节一致！
        current_day = date(2026, 11, 27)
        fake_now[0] += 60.0
        r2 = app.chat("第二天追问", chat_id="cross-day")
        assert r2["reply"] == "第二天回复"
        req2 = llm.requests[1]
        assert req2.system_prompt == req1.system_prompt
        assert req2.messages[0] == req1.messages[0]
        assert isinstance(req2.messages[-1], UserMessage)
        assert "2026-11-27（星期五，北京时间）" in req2.messages[-1].content

        cache2 = r2["cache"]
        assert isinstance(cache2, dict)
        assert cache2["last_break_section"] == "append"
        assert cache2["prefix_breaks"] == 0
        assert float(cache2["mean_prefix_reuse_rate"]) > 0.75  # type: ignore[arg-type]
    finally:
        app.close()


# 验收 4：前情摘要迁出 system_prompt（压缩变更摘要不伤 system 缓存）+ 5 分钟空闲 TTL 自动免费刷新
def test_compaction_summary_kept_out_of_system_prompt_and_idle_ttl_refresh(tmp_path: Path) -> None:
    fake_now = [1000.0]
    llm = _OptionsRecordingLLM(
        [
            AssistantMessage.from_text("答1", stop_reason="stop"),
            AssistantMessage.from_text("答2", stop_reason="stop"),
            AssistantMessage.from_text("答3", stop_reason="stop"),
            AssistantMessage.from_text("答4", stop_reason="stop"),
        ]
    )
    app = AssistantApp(
        data_root=tmp_path / "data",
        llm=llm,
        clock=lambda: fake_now[0],
        prompt_cache_ttl_seconds=300.0,
        auto_consolidate=False,
    )
    try:
        app.chat("第一句", chat_id="ttl-chat")
        sys_turn1 = llm.requests[0].system_prompt

        # 手动写入会话摘要：摘要应位于 messages 头部，system_prompt 必须 100% 保持原样！
        app.state.save_compaction("ttl-chat", "旧对话已总结：用户在学分布式系统。", compacted_until_id=0)
        fake_now[0] += 60.0
        app.chat("第二句", chat_id="ttl-chat")
        req2 = llm.requests[1]
        assert req2.system_prompt == sys_turn1
        assert "【本会话已压缩前情摘要】" not in req2.system_prompt
        assert isinstance(req2.messages[0], UserMessage)
        assert "旧对话已总结：用户在学分布式系统。" in req2.messages[0].content

        # 热聊期内（+200s < 300s）记忆文件有更新：仍死锁 system_prompt 保护热缓存
        app.memory.remember_user_statement("新增一条偏好事实", message_id=999)
        fake_now[0] += 200.0
        app.chat("第三句（热聊期）", chat_id="ttl-chat")
        assert llm.requests[2].system_prompt == sys_turn1
        assert "data/memory/facts.md" not in llm.requests[2].system_prompt

        # 空闲超过 300s（服务端缓存已自然过期）：自动免费刷新最新 MEMORY.md 索引
        fake_now[0] += 301.0
        app.chat("第四句（空闲归来）", chat_id="ttl-chat")
        assert "data/memory/facts.md" in llm.requests[3].system_prompt
    finally:
        app.close()


# 验收 5：按块粘性边界裁剪彻底消灭超窗后“每轮滑窗 0% 历史命中”
def test_chunked_history_eliminates_per_turn_head_slide_cache_miss(tmp_path: Path) -> None:
    # 先测纯函数 _select_chunked_history：未超上限保持 floor 不动；超上限一次切块并推进 floor
    msgs = [
        {"id": i + 1, "role": "user" if i % 2 == 0 else "assistant", "content": f"M{i+1}:" + ("a" * 300)}
        for i in range(12)
    ]
    kept1, floor1 = _select_chunked_history(msgs[:8], floor_id=0, base_messages=4, max_messages=8, max_chars=20_000)
    assert len(kept1) == 8 and floor1 == 0

    kept2, floor2 = _select_chunked_history(msgs[:10], floor_id=floor1, base_messages=4, max_messages=8, max_chars=20_000)
    assert len(kept2) == 4 and kept2[0]["role"] == "user"
    assert floor2 == int(kept2[0]["id"]) - 1

    # 继续追加 2 条（总数 6 <= max_messages=8）：保持同一 floor2，前缀 100% 稳定！
    kept3, floor3 = _select_chunked_history(msgs[:12], floor_id=floor2, base_messages=4, max_messages=8, max_chars=20_000)
    assert floor3 == floor2
    assert kept3[: len(kept2)] == kept2

    # 端到端连聊 40 轮（关闭后台沉淀，强行撞穿 MAX_ADAPTIVE_HISTORY_MESSAGES=60 上限）：
    # 绝大部分轮次必须维持 append 高命中，断档次数极少（<= 2 次切块，而不是每次撞上限后每轮都断）
    fake_now = [0.0]
    llm = _OptionsRecordingLLM([AssistantMessage.from_text(f"回复{i}", stop_reason="stop") for i in range(40)])
    app = AssistantApp(
        data_root=tmp_path / "long_data",
        llm=llm,
        clock=lambda: fake_now[0],
        consolidation_policy=ConsolidationPolicy(max_active_messages=500, keep_recent_messages=6, min_batch_messages=4),
        auto_consolidate=False,
    )
    try:
        for i in range(40):
            fake_now[0] += 5.0
            app.chat(f"第{i+1}轮问题", chat_id="long-chunk")
        summary = app.cache_monitor.summary(session_key="long-chunk")
        assert summary["llm_calls"] == 40
        assert int(summary["prefix_breaks"]) <= 2  # type: ignore[arg-type]
        assert float(summary["mean_prefix_reuse_rate"]) > 0.85  # type: ignore[arg-type]
    finally:
        app.close()


# 验收 6：激活 convert_to_llm 粘性高水位非破坏性投影——低水位 0 改动，超水位折叠旧大输出并锁定边界，原文完整保留在 TurnEnd.messages
def test_sticky_high_water_turn_projector_folds_nondestructively_and_recovers_append(tmp_path: Path) -> None:
    offload_hint = (
        '…（输出已截断：以上为第 1-20 行预览，完整共 500 行，已保存至 data/workspace/tool_outputs/demo.txt；'
        '可用 read_markdown(path="data/workspace/tool_outputs/demo.txt", start_line=21, end_line=220) 续读）'
    )
    big_payloads = {
        "1": ("日志第一段\n" + "A" * 2400 + "\n" + offload_hint),
        "2": ("日志第二段\n" + "B" * 2400),
        "3": ("日志第三段\n" + "C" * 2400),
        "4": ("日志第四段\n" + "D" * 500),
        "5": ("日志第五段\n" + "E" * 500),
    }

    # 返回按批次号索引的模拟大输出工具
    def dump_log(batch: str) -> str:
        return big_payloads[batch]

    llm = _OptionsRecordingLLM(
        [
            AssistantMessage(blocks=(ToolCall("c1", "dump_log", '{"batch":"1"}'),), stop_reason="toolUse"),
            AssistantMessage(blocks=(ToolCall("c2", "dump_log", '{"batch":"2"}'),), stop_reason="toolUse"),
            AssistantMessage(blocks=(ToolCall("c3", "dump_log", '{"batch":"3"}'),), stop_reason="toolUse"),
            AssistantMessage(blocks=(ToolCall("c4", "dump_log", '{"batch":"4"}'),), stop_reason="toolUse"),
            AssistantMessage(blocks=(ToolCall("c5", "dump_log", '{"batch":"5"}'),), stop_reason="toolUse"),
            AssistantMessage.from_text("五步排查完毕", stop_reason="stop"),
        ]
    )
    app = AssistantApp(
        data_root=tmp_path / "fold_data",
        llm=llm,
        auto_consolidate=False,
        max_turn_context_chars=5500,
    )
    app.tools.register(
        Tool(name="dump_log", description="输出大段日志", parameters={"batch": "批次号"}, handler=dump_log)
    )
    observed_turn_end: list[TurnEnd] = []
    try:
        res = app.chat(
            "帮我连续查 5 批日志",
            chat_id="fold-sess",
            on_event=lambda ev: observed_turn_end.append(ev) if isinstance(ev, TurnEnd) else None,
        )
        assert res["reply"] == "五步排查完毕"
        assert len(llm.requests) == 6

        # 第 2 步未超水位：第 1 批大日志原文完整透传，0 改动
        step2_tr = [m for m in llm.requests[1].messages if isinstance(m, ToolResultMessage)]
        assert len(step2_tr) == 1 and "A" * 2000 in step2_tr[0].content

        # 超水位后（第 6 步）：早期大工具输出已被折叠为紧凑摘要，且保留原有的 read_markdown 回读指针
        step6_tr = [m for m in llm.requests[5].messages if isinstance(m, ToolResultMessage)]
        assert len(step6_tr) == 5
        assert len(step6_tr[0].content) < 600
        assert "已折叠工具输出" in step6_tr[0].content
        assert 'read_markdown(path="data/workspace/tool_outputs/demo.txt"' in step6_tr[0].content
        # 最近尾部工具结果保持原文不折叠
        assert "E" * 400 in step6_tr[-1].content

        # 粘性锁定：最后两步在折叠之后恢复纯 append，整轮非预期前缀断档 <= 2 次
        recs = app.cache_monitor.records
        assert recs[-1].break_section == "append"
        assert int(res["cache"]["prefix_breaks"]) <= 2  # type: ignore[index,arg-type]

        # 非破坏性证明：TurnEnd.messages 里的原始工具结果 100% 完整保留未改写
        assert len(observed_turn_end) == 1
        raw_trs = [m for m in observed_turn_end[0].messages if isinstance(m, ToolResultMessage)]
        assert len(raw_trs) == 5
        assert "A" * 2000 in raw_trs[0].content
        assert "B" * 2000 in raw_trs[1].content
    finally:
        app.close()
