"""自动记忆沉淀与后台上下文压缩单元测试。"""
from __future__ import annotations

import json
import threading
from datetime import date
from pathlib import Path

import pytest

from personal_assistant.assistant.app import AssistantApp
from personal_assistant.assistant.common import format_date_with_weekday
from personal_assistant.assistant.consolidator import (
    CONSOLIDATION_SYSTEM_PROMPT,
    BackgroundConsolidator,
    ConsolidationPolicy,
    MemoryConsolidator,
    _parse_json_object,
)
from personal_assistant.assistant.db import StateStore
from personal_assistant.ai import LLMClient
from personal_assistant.assistant.memory import MemoryStore
from personal_assistant.assistant.prompts import SYSTEM_PROMPT, format_system_prompt


# 可录制请求并按预设回复（支持门控同步）的假模型
class ControlledLLM(LLMClient):
    # 保存预设文本列表、可选门控事件与请求历史
    def __init__(
        self,
        contents: list[str],
        *,
        gate: threading.Event | None = None,
        started: threading.Event | None = None,
    ) -> None:
        self._contents = list(contents)
        self._gate = gate
        self._started = started
        self.requests: list[list[dict[str, object]]] = []

    # 记录请求，在设置了门控事件时等待放行后再返回下一条回复
    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        self.requests.append([dict(m) for m in messages])
        if self._started is not None:
            self._started.set()
        if self._gate is not None:
            self._gate.wait(timeout=5.0)
        return {"role": "assistant", "content": self._contents.pop(0)}


# 测试同步沉淀核心：达阈值提炼长期记忆写盘、去重、更新索引并原子缩减活跃上下文
def test_consolidator_distills_memories_deduplicates_and_compacts_context(tmp_path: Path) -> None:
    state = StateStore(tmp_path / "data" / "state" / "chat.sqlite3")
    memory = MemoryStore(tmp_path / "data")
    try:
        for i in range(2):
            state.append_message("s1", "user", f"早期提问 {i}")
            state.append_message("s1", "assistant", f"早期回答 {i}")

        policy = ConsolidationPolicy(
            max_active_messages=6,
            max_active_chars=10000,
            keep_recent_messages=2,
            min_batch_messages=4,
        )
        llm = ControlledLLM(
            [
                # 兼容模型用 ```json 代码块包裹的返回格式
                "```json\n"
                + json.dumps(
                    {
                        "session_summary": "用户正在搭建纯 Python Agent，并决定采用索引式 Markdown 记忆。",
                        "memories": [
                            {
                                "topic": "projects",
                                "title": "项目决策",
                                "index_summary": "架构选择与关键决策",
                                "operations": [
                                    {"op": "add", "text": "使用纯 Python 标准库搭建个人 Agent"},
                                    {"op": "add", "text": "采用索引式 Markdown 记忆"},
                                ],
                            },
                            {
                                "topic": "memory",  # 保留字主题名应被安全跳过，绝不覆盖 MEMORY.md
                                "title": "非法主题",
                                "index_summary": "应被拦截",
                                "operations": [{"op": "add", "text": "不应写入"}],
                            },
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n```",
                # 第二轮重复提到相同事实时应自动去重，不重复写入
                json.dumps(
                    {
                        "session_summary": "延续纯 Python Agent 开发，补充了后台沉淀与压缩。",
                        "memories": [
                            {
                                "topic": "projects",
                                "title": "项目决策",
                                "index_summary": "架构选择与关键决策",
                                "operations": [
                                    {"op": "add", "text": "使用纯 Python 标准库搭建个人 Agent"},
                                    {"op": "add", "text": "启用阈值触发的后台自动沉淀"},
                                ],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        consolidator = MemoryConsolidator(state, memory, llm, policy=policy)

        # 未达 6 条消息阈值时不触发
        assert consolidator.select_batch("s1") is None
        assert consolidator.consolidate_once("s1") is None

        # 再加 2 条达到阈值（共 6 条：应沉淀前 4 条，保留最近 2 条）
        state.append_message("s1", "user", "最近提问")
        state.append_message("s1", "assistant", "最近回答")

        res1 = consolidator.consolidate_once("s1")
        assert res1 is not None
        assert res1.compacted_messages == 4
        assert res1.saved_files == ("data/memory/projects.md",)
        assert "纯 Python 标准库" in (memory.memory_dir / "projects.md").read_text(encoding="utf-8")
        assert "data/memory/projects.md" in memory.read_memory_index()
        assert "不应写入" not in memory.read_memory_index()
        # 活跃消息从 6 条缩减为 2 条，但 SQLite 原始表仍完整保留全部 6 条
        assert len(state.active_messages("s1")) == 2
        assert state.count_messages("s1") == 6

        # 再追加 4 条再次达到阈值，验证滚动摘要更新与记忆条目去重
        for i in range(2):
            state.append_message("s1", "user", f"后续提问 {i}")
            state.append_message("s1", "assistant", f"后续回答 {i}")
        res2 = consolidator.consolidate_once("s1")
        assert res2 is not None
        projects_body = (memory.memory_dir / "projects.md").read_text(encoding="utf-8")
        assert projects_body.count("使用纯 Python 标准库搭建个人 Agent") == 1
        assert "启用阈值触发的后台自动沉淀" in projects_body
        assert memory.read_memory_index().count("data/memory/projects.md") == 1
    finally:
        state.close()


# 测试沉淀解析失败时绝不推进压缩游标，后台线程捕获异常不崩溃主流程
def test_consolidator_failure_never_advances_cursor_or_loses_context(tmp_path: Path) -> None:
    state = StateStore(tmp_path / "data" / "state" / "chat.sqlite3")
    memory = MemoryStore(tmp_path / "data")
    try:
        for i in range(3):
            state.append_message("s2", "user", f"u-{i}")
            state.append_message("s2", "assistant", f"a-{i}")
        policy = ConsolidationPolicy(max_active_messages=6, keep_recent_messages=2, min_batch_messages=4)
        bad_llm = ControlledLLM(["这不是合法 JSON"])
        bg = BackgroundConsolidator(MemoryConsolidator(state, memory, bad_llm, policy=policy))

        assert bg.maybe_schedule("s2") is True
        bg.wait(timeout=5.0)

        # 游标保持 0，所有 6 条活跃消息完整保留，且错误记录在状态中
        assert state.get_compaction("s2")["compacted_until_id"] == 0
        assert len(state.active_messages("s2")) == 6
        assert bg.status()["last_error"] is not None
    finally:
        state.close()


# 测试 AssistantApp 端到端自动后台沉淀：不阻塞当前回复、后台落盘后下一轮上下文明显缩减
def test_assistant_app_auto_background_consolidation_and_context_shrinking(tmp_path: Path) -> None:
    started = threading.Event()
    gate = threading.Event()

    chat_llm = ControlledLLM(["答1", "答2", "答3", "答4"])
    consolidator_llm = ControlledLLM(
        [
            json.dumps(
                {
                    "session_summary": "用户偏好简洁中文回答，正在开发 my-helper 助手。",
                    "memories": [
                        {
                            "topic": "preferences",
                            "title": "个人偏好",
                            "index_summary": "沟通与代码风格偏好",
                            "operations": [{"op": "add", "text": "偏好简洁中文回答"}],
                        }
                    ],
                },
                ensure_ascii=False,
            )
        ],
        gate=gate,
        started=started,
    )
    policy = ConsolidationPolicy(
        max_active_messages=6,
        max_active_chars=10000,
        keep_recent_messages=2,
        min_batch_messages=4,
    )
    app = AssistantApp(
        data_root=tmp_path / "data",
        llm=chat_llm,
        consolidator_llm=consolidator_llm,
        consolidation_policy=policy,
    )
    try:
        r1 = app.chat("第1轮：我喜欢简洁中文回答", chat_id="auto-1")
        assert r1["consolidation_scheduled"] is False
        r2 = app.chat("第2轮：我在开发 my-helper", chat_id="auto-1")
        assert r2["consolidation_scheduled"] is False

        # 第 3 轮结束后共 6 条活跃消息，立刻在后台拉起沉淀，且主线程不等待门控事件就直接返回
        r3 = app.chat("第3轮：继续推进", chat_id="auto-1")
        assert r3["reply"] == "答3"
        assert r3["consolidation_scheduled"] is True
        assert started.wait(timeout=3.0)

        # 放行后台线程并等待其完成沉淀与压缩
        gate.set()
        app.wait_for_consolidation(timeout=5.0)

        assert "data/memory/preferences.md" in app.memory.read_memory_index()
        assert app.status()["consolidation"]["last_result"]["compacted_messages"] == 4

        # 第 4 轮验证上下文已缩减：system 更新了索引 + 前情摘要承接对 + 仅保留第3轮 2 条原话 + 本轮提问
        r4 = app.chat("第4轮：现在还记得吗？", chat_id="auto-1")
        assert r4["reply"] == "答4"
        turn4_msgs = chat_llm.requests[3]
        assert "data/memory/preferences.md" in str(turn4_msgs[0]["content"])
        assert "【本会话已压缩前情摘要" in str(turn4_msgs[1]["content"])
        assert turn4_msgs[2]["role"] == "assistant"
        # 仅保留最近 2 条（第3轮一问一答）+ 本轮提问，第1/2轮原话已缩减掉
        contents = [str(m["content"]) for m in turn4_msgs[3:]]
        assert contents == ["第3轮：继续推进", "答3", "第4轮：现在还记得吗？"]
        # 原始 SQLite 历史仍然完整保留全部 8 条消息
        assert app.state.count_messages("auto-1") == 8
    finally:
        gate.set()
        app.close()


# 测试 _parse_json_object 稳健容忍尾部多余文本与重复输出第二个 JSON，同时拒绝非法内容
def test_parse_json_object_tolerates_trailing_text_and_duplicate_json() -> None:
    # 复现真实评测 q016：模型输出合法 JSON 后多吐了一段换行说明或第二个 JSON 对象
    dup = '{"correct": true, "reason": "all matched"}\n\n{"correct": true, "reason": "duplicate"}'
    assert _parse_json_object(dup) == {"correct": True, "reason": "all matched"}

    trailing = '以下为结果（参考 {非JSON}）：\n{"session_summary": "ok", "memories": []}\n说明：整理完成。'
    assert _parse_json_object(trailing) == {"session_summary": "ok", "memories": []}

    fenced_extra = '```json\n{"a": {"b": 1}}\n```\n额外备注'
    assert _parse_json_object(fenced_extra) == {"a": {"b": 1}}

    with pytest.raises((ValueError, json.JSONDecodeError)):
        _parse_json_object("[1, 2, 3]")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        _parse_json_object("纯文本无 JSON")


# 测试对话系统提示词与后台记忆沉淀器均统一注入带中文星期几的北京时间日期
def test_weekday_injected_consistently_in_system_prompt_and_consolidator(tmp_path: Path) -> None:
    monday = date(2026, 11, 16)
    sunday = date(2026, 11, 22)
    assert format_date_with_weekday(monday) == "2026-11-16（星期一，北京时间）"
    assert format_date_with_weekday(sunday) == "2026-11-22（星期日，北京时间）"

    memory = MemoryStore(tmp_path / "data")
    prompt = format_system_prompt(memory, today=monday)
    assert "今天是 2026-11-16（星期一，北京时间）。" in prompt

    state = StateStore(tmp_path / "data" / "state" / "chat.sqlite3")
    try:
        state.append_message("s-day", "user", "下周三提交季度架构评审报告。")
        state.append_message("s-day", "assistant", "收到。")
        llm = ControlledLLM([json.dumps({"session_summary": "ok", "memories": []})])
        policy = ConsolidationPolicy(max_active_messages=2, keep_recent_messages=0, min_batch_messages=2)
        runner = MemoryConsolidator(state, memory, llm, policy=policy)
        runner.consolidate_once("s-day", today=monday)
        assert "【当前日期】2026-11-16（星期一，北京时间）" in str(llm.requests[0][1]["content"])
    finally:
        state.close()


# 守卫系统提示词字数硬上限（<= 300 字符，原基线 261 字符）、防幻觉红线与沉淀洁净通用规范
def test_prompt_and_consolidation_hygiene_guardrails() -> None:
    assert len(SYSTEM_PROMPT) <= 300, f"SYSTEM_PROMPT 超出 300 字符精简预算：{len(SYSTEM_PROMPT)}"
    assert "不要编造未观测到的结果" in SYSTEM_PROMPT
    assert "建议/推荐/规划" in SYSTEM_PROMPT

    assert "只写当前最新有效状态" in CONSOLIDATION_SYSTEM_PROMPT
    assert "一律用 op=add 保留" in CONSOLIDATION_SYSTEM_PROMPT
    assert "精确绝对日期" in CONSOLIDATION_SYSTEM_PROMPT
    assert "专有名词" in CONSOLIDATION_SYSTEM_PROMPT
    assert "优先完整记录用户自身的事实" in CONSOLIDATION_SYSTEM_PROMPT
    assert "不抄录长段通用科普或教程" in CONSOLIDATION_SYSTEM_PROMPT


# 测试 CONSOLIDATION_SYSTEM_PROMPT 正式声明 op=merge 且沉淀器端到端无损合并互补维度，不覆盖也不触发 archive
def test_consolidator_executes_op_merge_end_to_end_and_prompt_declares_merge(
    tmp_path: Path,
) -> None:
    assert '"op": "merge"' in CONSOLIDATION_SYSTEM_PROMPT
    assert "优先用 op=merge" in CONSOLIDATION_SYSTEM_PROMPT
    assert len(CONSOLIDATION_SYSTEM_PROMPT) <= 1350

    state = StateStore(tmp_path / "data" / "state" / "chat.sqlite3")
    memory = MemoryStore(tmp_path / "data")
    try:
        memory.apply_memory_operations(
            "fitness",
            "运动健康",
            "马拉松成绩与目标",
            [{"op": "add", "text": "Completed first full marathon in 4:22:00."}],
            today=date(2023, 5, 21),
        )
        state.append_message("m-merge", "user", "By the way, my marathon target time was 4:10:00.")
        state.append_message("m-merge", "assistant", "Got it, thanks for sharing your target time.")

        llm = ControlledLLM(
            [
                json.dumps(
                    {
                        "session_summary": "User noted marathon target time was 4:10:00.",
                        "memories": [
                            {
                                "topic": "fitness",
                                "title": "运动健康",
                                "index_summary": "马拉松成绩与目标",
                                "operations": [
                                    {
                                        "op": "merge",
                                        "old": "4:22:00",
                                        "text": "Target time was 4:10:00.",
                                    }
                                ],
                            }
                        ],
                    }
                )
            ]
        )
        policy = ConsolidationPolicy(max_active_messages=2, keep_recent_messages=0, min_batch_messages=2)
        runner = MemoryConsolidator(state, memory, llm, policy=policy)
        result = runner.consolidate_once("m-merge", today=date(2023, 5, 24))
        assert result is not None
        assert result.saved_files == ("data/memory/fitness.md",)

        content = (memory.memory_dir / "fitness.md").read_text(encoding="utf-8")
        assert "4:22:00" in content
        assert "4:10:00" in content
        assert not (memory.memory_dir / "archive").exists()
    finally:
        state.close()


# 测试 op=merge 沉淀后同样触发 prune_consolidated_raw_facts 清理裸原话幽灵，且 written_texts 与 text/new 优先级一致
def test_consolidator_op_merge_prunes_consolidated_raw_facts_ghost(tmp_path: Path) -> None:
    state = StateStore(tmp_path / "data" / "state" / "chat.sqlite3")
    memory = MemoryStore(tmp_path / "data")
    try:
        memory.apply_memory_operations(
            "fitness",
            "运动健康",
            "马拉松成绩与目标",
            [{"op": "add", "text": "Completed first full marathon in 4:22:00."}],
            today=date(2023, 5, 21),
        )
        m1 = state.append_message("merge-prune", "user", "记住，我养了一只橘猫。")
        memory.remember_user_statement("记住，我养了一只橘猫。", message_id=m1)
        state.append_message("merge-prune", "assistant", "好的，记下了。")
        m3 = state.append_message("merge-prune", "user", "记住，我的马拉松目标成绩是 4:10:00。")
        memory.remember_user_statement("记住，我的马拉松目标成绩是 4:10:00。", message_id=m3)
        state.append_message("merge-prune", "assistant", "记下了。")

        llm = ControlledLLM(
            [
                json.dumps(
                    {
                        "session_summary": "用户补充马拉松目标成绩为 4:10:00。",
                        "memories": [
                            {
                                "topic": "fitness",
                                "title": "运动健康",
                                "index_summary": "马拉松成绩与目标",
                                "operations": [
                                    {
                                        "op": "merge",
                                        "old": "marathon",
                                        "text": "马拉松目标成绩是 4:10:00。",
                                        "new": "不应优先使用的备用字段。",
                                    }
                                ],
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            ]
        )
        policy = ConsolidationPolicy(max_active_messages=4, keep_recent_messages=0, min_batch_messages=4)
        runner = MemoryConsolidator(state, memory, llm, policy=policy)
        runner.consolidate_once("merge-prune", today=date(2023, 5, 24))

        fitness_text = (memory.memory_dir / "fitness.md").read_text(encoding="utf-8")
        assert "Completed first full marathon in 4:22:00；马拉松目标成绩是 4:10:00。" in fitness_text
        facts_text = (memory.memory_dir / "facts.md").read_text(encoding="utf-8")
        assert f"(msg#{m3})" not in facts_text
        assert "橘猫" in facts_text
        assert f"(msg#{m1})" in facts_text
    finally:
        state.close()


# 测试记忆沉淀仅接受结构化 operations 协议，忽略旧版 items 兼容格式
def test_consolidator_ignores_legacy_items_format(tmp_path: Path) -> None:
    state = StateStore(tmp_path / "data" / "state" / "chat.sqlite3")
    memory = MemoryStore(tmp_path / "data")
    try:
        state.append_message("legacy-items", "user", "我喜欢乌龙茶。")
        state.append_message("legacy-items", "assistant", "记下了。")
        llm = ControlledLLM(
            [
                json.dumps(
                    {
                        "session_summary": "用户喜欢乌龙茶。",
                        "memories": [
                            {
                                "topic": "preferences",
                                "title": "个人偏好",
                                "index_summary": "饮品偏好",
                                "items": ["喜欢乌龙茶"],
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            ]
        )
        policy = ConsolidationPolicy(max_active_messages=2, keep_recent_messages=0, min_batch_messages=2)
        runner = MemoryConsolidator(state, memory, llm, policy=policy)
        result = runner.consolidate_once("legacy-items", today=date(2026, 11, 20))
        assert result is not None
        assert result.saved_files == ()
        assert not (memory.memory_dir / "preferences.md").exists()
    finally:
        state.close()


# 测试沉淀分批上限与偶数对对齐、today_fn 逻辑日期注入、保护在途 (turn#...) 暂存行以及失败指数退避与恢复
def test_consolidator_batch_cap_today_fn_turn_tag_protection_and_backoff(tmp_path: Path) -> None:
    state = StateStore(tmp_path / "data" / "state" / "chat.sqlite3")
    memory = MemoryStore(tmp_path / "data")
    try:
        # 1. 验证 max_batch_messages=24（默认）与奇数配额时均按偶数对截取且不劈开 user+assistant
        for i in range(18):
            state.append_turn("s-batch", f"u-{i}", f"a-{i}")
        policy_default = ConsolidationPolicy(max_active_messages=16, keep_recent_messages=6, min_batch_messages=4)
        assert policy_default.max_batch_messages == 24
        llm_noop = ControlledLLM([])
        cons_default = MemoryConsolidator(state, memory, llm_noop, policy=policy_default)
        selected_default = cons_default.select_batch("s-batch")
        assert selected_default is not None
        assert len(selected_default[1]) == 24
        assert selected_default[1][0]["role"] == "user"
        assert selected_default[1][-1]["role"] == "assistant"

        policy_odd = ConsolidationPolicy(
            max_active_messages=10,
            keep_recent_messages=2,
            min_batch_messages=4,
            max_batch_messages=7,
        )
        cons_odd = MemoryConsolidator(state, memory, llm_noop, policy=policy_odd)
        selected_odd = cons_odd.select_batch("s-batch")
        assert selected_odd is not None
        assert len(selected_odd[1]) == 6
        assert selected_odd[1][-1]["role"] == "assistant"

        # 2. 验证 today_fn 注入、read_topic_snapshots 跳过仅含在途行空块、未落盘操作绝不误删 (msg#...) 事实、以及在途回滚不丢已沉淀同文事实
        only_inflight_mem = MemoryStore(tmp_path / "only_inflight_data")
        only_inflight_mem.remember_user_statement("仅在途未提交的唯一条目", turn_tag="onlyturn01")
        assert only_inflight_mem.read_topic_snapshots() == ""

        m_mouse, _ = state.append_turn("s-turn-protect", "前天买过旧鼠标", "记下了。")
        state.append_turn("s-turn-protect", "昨天我买了新键盘", "记下了。")
        memory.remember_user_statement("前天买过旧鼠标", message_id=m_mouse)
        memory.remember_user_statement("前台尚未落库的在途事实，顺便引用 (msg#37) 昨天我买了新键盘", turn_tag="deadbeef1234")

        logical_day = date(2025, 3, 15)
        cons_llm = ControlledLLM(
            [
                json.dumps(
                    {
                        "session_summary": "用户昨天买了新键盘。",
                        "memories": [
                            {
                                "topic": "facts",
                                "title": "事实备忘",
                                "index_summary": "近期事实",
                                "operations": [
                                    {"op": "remove", "old": "(turn#deadbeef1234)"},
                                    {"op": "replace", "old": "(turn#deadbeef", "new": "被篡改"},
                                    {"op": "remove", "old": "在途事实"},
                                    {"op": "replace", "old": "尚未落库", "new": "前天买过旧鼠标（未落盘不应触发删除）"},
                                    {"op": "add", "text": "前天买过旧鼠标（已过期被拒，不应删 msg）", "until": "2025-03-01"},
                                    {"op": "add", "text": "伪造 (turn#fake999) 标签不应写入"},
                                    {"op": "add", "text": "尚未落库的在途事实"},
                                    {"op": "add", "text": "2025-03-14 买了新键盘。"},
                                ],
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            ]
        )
        cons_day = MemoryConsolidator(
            state,
            memory,
            cons_llm,
            policy=ConsolidationPolicy(max_active_messages=4, keep_recent_messages=0, min_batch_messages=4),
            today_fn=lambda: logical_day,
        )
        res_day = cons_day.consolidate_once("s-turn-protect")
        assert res_day is not None
        prompt_sent = str(cons_llm.requests[0][1]["content"])
        assert "【当前日期】2025-03-15" in prompt_sent
        assert "(turn#deadbeef1234)" not in prompt_sent
        assert "前台尚未落库的在途事实" not in prompt_sent

        facts_body = (memory.memory_dir / "facts.md").read_text(encoding="utf-8")
        assert "(turn#deadbeef1234) 前台尚未落库的在途事实" in facts_body
        assert "被篡改" not in facts_body
        assert "(turn#fake999)" not in facts_body
        assert f"(msg#{m_mouse}) 前天买过旧鼠标" in facts_body
        assert "[2025-03-15] 2025-03-14 买了新键盘。" in facts_body
        assert "[2025-03-15] 尚未落库的在途事实" in facts_body

        # 前台流异常回滚移除 (turn#deadbeef1234) 后，后台已沉淀落盘的同文正式条目依然保留不丢失
        memory.finalize_turn_fact("deadbeef1234", message_id=None)
        facts_after_rollback = (memory.memory_dir / "facts.md").read_text(encoding="utf-8")
        assert "(turn#deadbeef1234)" not in facts_after_rollback
        assert "[2025-03-15] 尚未落库的在途事实" in facts_after_rollback
        assert f"(msg#{m_mouse}) 前天买过旧鼠标" in facts_after_rollback

        # 3. 验证 BackgroundConsolidator 按 chat_id 失败计数与指数退避（注入确定性时钟零等待）
        for i in range(3):
            state.append_turn("s-backoff", f"bu-{i}", f"ba-{i}")
            state.append_turn("s-other", f"ou-{i}", f"oa-{i}")

        now_ts = [1000.0]
        # 记录并响应不同会话请求的受控模型：s-backoff 前两次失败、第三次成功；s-other 立即成功
        class BackoffProbeLLM(LLMClient):
            # 初始化状态计数器
            def __init__(self) -> None:
                self.calls: list[str] = []

            # 根据批次内容决定抛错或返回合法 JSON
            def respond(
                self,
                messages: list[dict[str, object]],
                tools: list[dict[str, object]] | None = None,
            ) -> dict[str, object]:
                content = str(messages[1]["content"])
                if "bu-" in content:
                    self.calls.append("s-backoff")
                    if self.calls.count("s-backoff") <= 2:
                        raise RuntimeError("transient llm error")
                    return {"role": "assistant", "content": json.dumps({"session_summary": "backoff recovered", "memories": []})}
                self.calls.append("s-other")
                return {"role": "assistant", "content": json.dumps({"session_summary": "other ok", "memories": []})}

        probe_llm = BackoffProbeLLM()
        bg = BackgroundConsolidator(
            MemoryConsolidator(
                state,
                memory,
                probe_llm,
                policy=ConsolidationPolicy(max_active_messages=6, keep_recent_messages=2, min_batch_messages=4),
            ),
            clock=lambda: now_ts[0],
            base_backoff_seconds=2.0,
        )
        assert bg.status()["backoff_sessions"] == 0

        # 第 1 次失败：退避 2.0s（1000 -> 1002）
        assert bg.maybe_schedule("s-backoff") is True
        bg.wait(timeout=5.0)
        assert bg.status()["backoff_sessions"] == 1
        assert probe_llm.calls == ["s-backoff"]

        # 冷却期内同会话直接跳过，但另一会话 s-other 不受任何影响可正常调度
        now_ts[0] = 1001.5
        assert bg.maybe_schedule("s-backoff") is False
        assert bg.maybe_schedule("s-other") is True
        bg.wait(timeout=5.0)
        assert probe_llm.calls == ["s-backoff", "s-other"]

        # 到达 1002.0s 后第 2 次重试仍失败：退避翻倍至 4.0s（1002 -> 1006）
        now_ts[0] = 1002.0
        assert bg.maybe_schedule("s-backoff") is True
        bg.wait(timeout=5.0)
        assert bg.status()["backoff_sessions"] == 1

        now_ts[0] = 1005.9
        assert bg.maybe_schedule("s-backoff") is False

        # 到达 1006.0s 后第 3 次重试成功：立即清除失败计数
        now_ts[0] = 1006.0
        assert bg.maybe_schedule("s-backoff") is True
        bg.wait(timeout=5.0)
        assert bg.status()["backoff_sessions"] == 0
        assert bg.status()["last_error"] is None
    finally:
        state.close()
