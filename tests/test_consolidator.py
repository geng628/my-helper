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
                                "items": ["使用纯 Python 标准库搭建个人 Agent", "采用索引式 Markdown 记忆"],
                            },
                            {
                                "topic": "memory",  # 保留字主题名应被安全跳过，绝不覆盖 MEMORY.md
                                "title": "非法主题",
                                "index_summary": "应被拦截",
                                "items": ["不应写入"],
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
                                "items": ["使用纯 Python 标准库搭建个人 Agent", "启用阈值触发的后台自动沉淀"],
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
                            "items": ["偏好简洁中文回答"],
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


# 守卫记忆沉淀提示词的随口细节/问答索引、互补属性合并保留与事件实际发生时间规范
def test_consolidation_prompt_covers_complementary_merge_event_dates_and_qa_entities() -> None:
    # 规则 1：随口提及细节与聚焦问答的一句话核心实体/推荐/数值索引
    assert "随口提及的具体细节、数值、物件" in CONSOLIDATION_SYSTEM_PROMPT
    assert "咨询特定主题" in CONSOLIDATION_SYSTEM_PROMPT
    assert "一句话" in CONSOLIDATION_SYSTEM_PROMPT
    assert "关键实体" in CONSOLIDATION_SYSTEM_PROMPT
    assert "推荐" in CONSOLIDATION_SYSTEM_PROMPT
    assert "数值" in CONSOLIDATION_SYSTEM_PROMPT
    assert "不抄录长段通用科普或教程" in CONSOLIDATION_SYSTEM_PROMPT

    # 规则 2：真失效 replace vs 互补属性合并保留，不确定时保守用 op=add
    assert "互补属性" in CONSOLIDATION_SYSTEM_PROMPT
    assert "目标时间与实际成绩" in CONSOLIDATION_SYSTEM_PROMPT
    assert "进球数与助攻数" in CONSOLIDATION_SYSTEM_PROMPT
    assert "绝不丢弃仍有效的互补维度" in CONSOLIDATION_SYSTEM_PROMPT
    assert "拿不准是否冲突时保守使用 op=add" in CONSOLIDATION_SYSTEM_PROMPT

    # 规则 4：[YYYY-MM-DD] 仅为记录日期，相对时间与早于今天的显式过往日期/年份须按实际发生时间写入正文
    assert "仅为记录日期" in CONSOLIDATION_SYSTEM_PROMPT
    assert "实际发生时间" in CONSOLIDATION_SYSTEM_PROMPT
    assert "显式过往日期" in CONSOLIDATION_SYSTEM_PROMPT
    assert "2月20日" in CONSOLIDATION_SYSTEM_PROMPT
    assert "2022年" in CONSOLIDATION_SYSTEM_PROMPT
    assert "写入正文" in CONSOLIDATION_SYSTEM_PROMPT

    # 规则 3、5 与 JSON 输出契约保持完整
    assert '"session_summary"' in CONSOLIDATION_SYSTEM_PROMPT
    assert '"memories"' in CONSOLIDATION_SYSTEM_PROMPT
    assert "data/memory/facts.md" in CONSOLIDATION_SYSTEM_PROMPT
    assert "session_summary 必须简洁客观" in CONSOLIDATION_SYSTEM_PROMPT
