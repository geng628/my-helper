"""记忆与多轮端到端评测器（memory_eval）自身单元测试（TDD）。"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from urllib.error import URLError

import pytest

from evals import evaluate_memory, judge_answer, load_scenarios, main, write_scenarios_jsonl
from evals.memory_eval import JUDGE_SYSTEM_PROMPT
from personal_assistant.ai import (
    ApiProviderRegistry,
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    ModelClient,
    ModelProfile,
    NoopModelClient,
    OpenAICompletionsProvider,
    StreamDone,
    StreamOptions,
    TextDelta,
    TokenUsage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


# 可按序返回预设消息与 usage 的脚本化假模型（实现 ModelInvoker），用于确定性测试 memory_eval
class ScriptedMemoryEvalLLM:
    # 保存预设回复与可选 usage 队列
    def __init__(self, responses: list[tuple[AssistantMessage, TokenUsage | None]]) -> None:
        self._responses = list(responses)
        self.last_usage: TokenUsage | None = None

    # 弹出下一条预设回复并更新 last_usage
    def complete(
        self,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessage:
        del context, options
        reply, usage = self._responses.pop(0)
        self.last_usage = usage
        return reply

    # 流式弹出下一条预设回复并更新 last_usage
    def stream(
        self,
        context: Context,
        options: StreamOptions | None = None,
    ) -> Iterator[AssistantMessageEvent]:
        reply = self.complete(context, options)
        if reply.text:
            yield TextDelta(content_index=0, text=reply.text, partial=reply)
        reason = "toolUse" if reply.tool_calls else "stop"
        yield StreamDone(reason=reason, message=reply, usage=self.last_usage)


# 测试 memory_eval 完整流程：多轮聊天 -> 沉淀冲突替换 -> 过期归档 -> 跨会话核验与缓存统计
def test_evaluate_memory_runs_lifecycle_scenario_and_checks_assertions() -> None:
    scenarios = [
        {
            "id": "sc-conflict-and-expiry",
            "description": "验证冲突替换与临时事实过期归档",
            "policy": {"max_active_messages": 4, "keep_recent_messages": 0, "min_batch_messages": 2},
            "seed_memories": [
                {
                    "topic": "projects",
                    "title": "项目备忘",
                    "index_summary": "项目端口与日程",
                    "today": "2026-11-19",
                    "operations": [{"op": "add", "text": "北极星端口定为 9091。"}],
                }
            ],
            "steps": [
                {"action": "chat", "chat_id": "s1", "message": "北极星端口改成 9092，明天开演练会。"},
                {"action": "chat", "chat_id": "s1", "message": "好的先这样。"},
                {"action": "consolidate", "chat_id": "s1", "today": "2026-11-19"},
                {
                    "action": "check_search",
                    "query": "演练会",
                    "today": "2026-11-25",
                    "expect_empty": True,
                },
                {"action": "archive", "today": "2026-11-25"},
                {
                    "action": "ask",
                    "chat_id": "s2",
                    "message": "北极星端口是多少？还有演练会吗？",
                    "expect_tools_any": ["search_markdown", "read_markdown"],
                    "must_include": ["9092"],
                    "must_not_include": ["9091"],
                    "file_checks": [
                        {
                            "path": "data/memory/projects.md",
                            "must_include": ["9092"],
                            "must_not_include": ["9091", "演练会"],
                        }
                    ],
                },
            ],
        }
    ]

    llm = ScriptedMemoryEvalLLM(
        [
            # chat 1
            (AssistantMessage.from_text("明白，端口 9092，明天演练会。", stop_reason="stop"), TokenUsage(1000, 0, 20)),
            # chat 2
            (AssistantMessage.from_text("好的。", stop_reason="stop"), TokenUsage(1100, 960, 10)),
            # consolidate
            (
                AssistantMessage.from_text(
                    json.dumps(
                        {
                            "session_summary": "北极星端口改为 9092。",
                            "memories": [
                                {
                                    "topic": "projects",
                                    "title": "项目备忘",
                                    "index_summary": "项目端口与日程",
                                    "operations": [
                                        {"op": "replace", "old": "9091", "new": "北极星端口改为 9092。"},
                                        {"op": "add", "text": "2026-11-20 召开演练会。", "until": "2026-11-20"},
                                    ],
                                }
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    stop_reason="stop",
                ),
                TokenUsage(900, 0, 80),
            ),
            # ask (s2) -> tool call read_markdown -> final reply
            (
                AssistantMessage(
                    blocks=(
                        ToolCall(
                            id="c1",
                            name="read_markdown",
                            arguments='{"path": "data/memory/projects.md", "start_line": "1", "end_line": "20"}',
                        ),
                    ),
                    stop_reason="toolUse",
                ),
                TokenUsage(1200, 768, 25),
            ),
            (
                AssistantMessage.from_text("北极星当前端口是 9092，目前没有待办演练会记录。", stop_reason="stop"),
                TokenUsage(1300, 1152, 25),
            ),
        ]
    )

    report = evaluate_memory(llm, scenarios, git_commit="eval-sha")
    assert report["mode"] == "memory"
    assert report["git_commit"] == "eval-sha"
    assert report["summary"]["total_scenarios"] == 1
    assert report["summary"]["passed_scenarios"] == 1
    assert report["summary"]["failed_scenarios"] == 0
    assert report["summary"]["pass_rate"] == 1.0
    assert report["summary"]["cache"]["cached_tokens"] > 0
    assert report["failures"] == []


# 测试 memory_eval 忠实捕获并保留回答错误、工具未调用与磁盘校验失败，绝不虚报通过
def test_evaluate_memory_faithfully_reports_failures() -> None:
    scenarios = [
        {
            "id": "sc-hallucination-fail",
            "description": "未查记忆且编造旧值应判失败",
            "steps": [
                {
                    "action": "ask",
                    "chat_id": "s1",
                    "message": "我的显示器型号是什么？",
                    "expect_tools_any": ["search_markdown", "read_markdown"],
                    "must_include_any": ["没有", "未找到", "无记录"],
                    "must_not_include": ["Dell U2723QE"],
                }
            ],
        }
    ]
    llm = ScriptedMemoryEvalLLM(
        [(AssistantMessage.from_text("你的显示器是 Dell U2723QE。", stop_reason="stop"), None)]
    )
    report = evaluate_memory(llm, scenarios, git_commit="fail-sha")
    assert report["summary"]["passed_scenarios"] == 0
    assert report["summary"]["failed_scenarios"] == 1
    assert len(report["failures"]) == 1
    reasons = str(report["failures"][0]["reasons"])
    assert "expect_tools_any" in reasons
    assert "must_include_any" in reasons
    assert "must_not_include" in reasons


# 测试 eval CLI 支持 --mode memory 并能加载公开场景文件校验结构
def test_memory_scenarios_jsonl_loads_cleanly() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    scenarios_path = repo_root / "evals" / "memory_scenarios.jsonl"
    assert scenarios_path.is_file()
    lines = [line.strip() for line in scenarios_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= 15
    ids = set()
    categories = set()
    for line in lines:
        item = json.loads(line)
        assert item["id"] not in ids
        ids.add(item["id"])
        assert isinstance(item.get("category"), str) and item["category"].strip()
        categories.add(item["category"])
        assert isinstance(item.get("steps"), list) and item["steps"]
    assert categories >= {
        "extraction",
        "knowledge-update",
        "temporal",
        "multi-session",
        "abstention",
        "compaction-cache",
    }


# 测试 seed_history 批量注入历史、步骤级 today 时钟覆盖、分类汇总与隔离式 LLM-as-a-Judge 评分
def test_evaluate_memory_supports_seed_history_today_categories_and_llm_judge() -> None:
    scenarios = [
        {
            "id": "lme-sample-1",
            "category": "knowledge-update",
            "description": "使用历史对话注入、历史日期时钟与 LLM Judge 评测公开题",
            "policy": {"max_active_messages": 2, "keep_recent_messages": 0, "min_batch_messages": 2},
            "steps": [
                {
                    "action": "seed_history",
                    "chat_id": "hist-1",
                    "messages": [
                        {"role": "user", "content": "My personal best 5K time is now 25:50."},
                        {"role": "assistant", "content": "Congrats on your new 25:50 5K PR!"},
                    ],
                },
                {"action": "consolidate", "chat_id": "hist-1", "today": "2023-05-27"},
                {
                    "action": "ask",
                    "chat_id": "eval-q",
                    "today": "2023-06-01",
                    "message": "What is my personal best 5K time?",
                    "expect_tools_any": ["search_markdown", "read_markdown"],
                    "expected_answer": "25 minutes and 50 seconds (or 25:50)",
                    "is_abstention": False,
                },
            ],
        }
    ]

    seen_ask_system_prompts: list[str] = []

    # 拦截 ask 阶段的 system prompt 以验证 today 时钟确实生效
    class ClockCapturingLLM(ScriptedMemoryEvalLLM):
        # 记录带工具的前台调用 system prompt 并返回预设回复
        def complete(
            self,
            context: Context,
            options: StreamOptions | None = None,
        ) -> AssistantMessage:
            if context.tools:
                seen_ask_system_prompts.append(context.system_prompt)
            return super().complete(context, options)

    agent_llm = ClockCapturingLLM(
        [
            # consolidate
            (
                AssistantMessage.from_text(
                    json.dumps(
                        {
                            "session_summary": "User updated 5K personal best to 25:50.",
                            "memories": [
                                {
                                    "topic": "fitness",
                                    "title": "运动记录",
                                    "index_summary": "5K 跑步成绩",
                                    "operations": [
                                        {"op": "add", "text": "5K personal best time is 25:50."}
                                    ],
                                }
                            ],
                        }
                    ),
                    stop_reason="stop",
                ),
                TokenUsage(500, 0, 60),
            ),
            # ask -> search_markdown
            (
                AssistantMessage(
                    blocks=(ToolCall(id="c1", name="search_markdown", arguments='{"query": "5K", "root": "data/memory"}'),),
                    stop_reason="toolUse",
                ),
                TokenUsage(900, 512, 20),
            ),
            # ask -> final reply
            (
                AssistantMessage.from_text("Your personal best 5K time is 25:50.", stop_reason="stop"),
                TokenUsage(1050, 896, 20),
            ),
        ]
    )

    judge_llm = ScriptedMemoryEvalLLM(
        [
            (
                AssistantMessage.from_text(
                    '{"correct": true, "reason": "Answer matches 25:50 exactly."}',
                    stop_reason="stop",
                ),
                TokenUsage(300, 0, 15),
            )
        ]
    )

    report = evaluate_memory(agent_llm, scenarios, judge_llm=judge_llm, git_commit="judge-sha")
    assert report["summary"]["passed_scenarios"] == 1
    assert report["summary"]["by_category"]["knowledge-update"]["passed_scenarios"] == 1
    assert report["summary"]["by_category"]["knowledge-update"]["pass_rate"] == 1.0
    # 确认 ask 阶段注入的是历史日期 2023-06-01
    assert seen_ask_system_prompts and "2023-06-01" in seen_ask_system_prompts[0]
    # 确认 Judge 的 LLM 调用没有污染 Agent 的 KV 缓存统计（agent 共 3 次调用）
    assert report["summary"]["cache"]["llm_calls"] == 3
    ask_step = report["scenarios"][0]["steps"][-1]
    assert ask_step["judge"]["correct"] is True


# 测试 LLM-as-a-Judge 判错时忠实记录失败原因，不虚报通过
def test_evaluate_memory_records_judge_failure() -> None:
    scenarios = [
        {
            "id": "lme-abs-fail",
            "category": "abstention",
            "description": "应当拒答却胡编，Judge 判错",
            "steps": [
                {
                    "action": "ask",
                    "chat_id": "s1",
                    "message": "How many cows did I buy from Peter?",
                    "expected_answer": "The information provided is not enough.",
                    "is_abstention": True,
                }
            ],
        }
    ]
    agent_llm = ScriptedMemoryEvalLLM(
        [(AssistantMessage.from_text("You bought three cows from Peter.", stop_reason="stop"), None)]
    )
    judge_llm = ScriptedMemoryEvalLLM(
        [(AssistantMessage.from_text('{"correct": false, "reason": "Hallucinated instead of abstaining."}', stop_reason="stop"), None)]
    )
    report = evaluate_memory(agent_llm, scenarios, judge_llm=judge_llm, git_commit="judge-fail")
    assert report["summary"]["passed_scenarios"] == 0
    assert report["summary"]["failed_scenarios"] == 1
    assert "judge" in str(report["failures"][0]["reason"]).lower()


# 测试 evaluate_memory 按序执行全部场景并触发单题实时进度回调
def test_evaluate_memory_reports_progress_in_order() -> None:
    scenarios = [
        {
            "id": f"sc-{idx}",
            "category": "extraction",
            "steps": [
                {
                    "action": "ask",
                    "chat_id": f"c-{idx}",
                    "message": f"question #{idx}",
                    "must_include": [f"ok-{idx}"],
                }
            ],
        }
        for idx in range(3)
    ]
    llm = ScriptedMemoryEvalLLM(
        [
            (AssistantMessage.from_text(f"ok-{idx}", stop_reason="stop"), TokenUsage(100 + idx, 50, 10))
            for idx in range(3)
        ]
    )

    progress_events: list[tuple[int, int, str]] = []

    # 收集每题完成时的进度回调事件
    def on_progress(done: int, total: int, record: dict[str, object]) -> None:
        progress_events.append((done, total, str(record["id"])))

    report = evaluate_memory(llm, scenarios, on_progress=on_progress, git_commit="prog-sha")
    assert [item["id"] for item in report["scenarios"]] == ["sc-0", "sc-1", "sc-2"]
    assert report["summary"]["passed_scenarios"] == 3
    assert report["summary"]["cache"]["llm_calls"] == 3
    assert report["summary"]["cache"]["prompt_tokens"] == 303
    assert progress_events == [(1, 3, "sc-0"), (2, 3, "sc-1"), (3, 3, "sc-2")]


# 测试 shared_group 共享长记忆模式：同一对话只沉淀一次，每道 QA 在独立隔离副本上运行且不串状态
def test_evaluate_memory_shared_group_consolidates_once_and_isolates_each_qa(tmp_path: Path) -> None:
    consolidate_calls = 0

    # 根据请求内容区分沉淀调用与 QA 问答调用的确定性假模型
    class SharedGroupMockLLM:
        # 初始化最近一次 usage 记录
        def __init__(self) -> None:
            self.last_usage: TokenUsage | None = None

        # 沉淀阶段写入 pets.md；QA1 故意调 remember_fact 写入隔离标记；QA2 仅读 pets.md 验证无串台
        def complete(
            self,
            context: Context,
            options: StreamOptions | None = None,
        ) -> AssistantMessage:
            del options
            nonlocal consolidate_calls
            if not context.tools and "后台记忆整理器" in context.system_prompt:
                consolidate_calls += 1
                self.last_usage = TokenUsage(prompt_tokens=400, cached_tokens=0, completion_tokens=50)
                return AssistantMessage.from_text(
                    json.dumps(
                        {
                            "session_summary": "用户养了一只叫 Max 的金毛。",
                            "memories": [
                                {
                                    "topic": "pets",
                                    "title": "宠物记录",
                                    "index_summary": "宠物名字与品种",
                                    "operations": [{"op": "add", "text": "Alice adopted a golden retriever named Max."}],
                                }
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    stop_reason="stop",
                )

            last_msg = context.messages[-1] if context.messages else None
            user_prompt = next(
                (m.content for m in reversed(context.messages) if isinstance(m, UserMessage)),
                "",
            )
            self.last_usage = TokenUsage(prompt_tokens=300, cached_tokens=150, completion_tokens=20)
            if "QA1_MUTATE" in user_prompt and isinstance(last_msg, UserMessage):
                return AssistantMessage(
                    blocks=(ToolCall(id="call_mut", name="remember_fact", arguments="{}"),),
                    stop_reason="toolUse",
                )
            return AssistantMessage.from_text("The golden retriever is named Max.", stop_reason="stop")

        # 流式接口复用 complete
        def stream(
            self,
            context: Context,
            options: StreamOptions | None = None,
        ) -> Iterator[AssistantMessageEvent]:
            msg = self.complete(context, options)
            if msg.text:
                yield TextDelta(content_index=0, text=msg.text, partial=msg)
            reason = "toolUse" if msg.tool_calls else "stop"
            yield StreamDone(reason=reason, message=msg, usage=self.last_usage)

    setup_steps = [
        {
            "action": "seed_history",
            "chat_id": "conv-s1",
            "messages": [
                {"role": "user", "content": "I adopted a golden retriever named Max."},
                {"role": "assistant", "content": "Max is a great name!"},
            ],
        },
        {"action": "consolidate", "chat_id": "conv-s1", "today": "2023-05-01"},
    ]
    shared_policy = {"max_active_messages": 1, "keep_recent_messages": 0, "min_batch_messages": 1}
    shared_setup = {
        "record_type": "shared_setup",
        "shared_group": "conv-shared",
        "policy": shared_policy,
        "steps": setup_steps,
    }
    raw_scenarios = [
        {
            "id": "qa-1",
            "category": "single-hop",
            "shared_group": "conv-shared",
            "shared_setup": shared_setup,
            "policy": shared_policy,
            "steps": [
                {
                    "action": "ask",
                    "chat_id": "eval-query",
                    "today": "2023-05-10",
                    "message": "QA1_MUTATE: What is Alice's dog's name?",
                    "must_include": ["Max"],
                    "file_checks": [
                        {"path": "data/memory/pets.md", "must_include": ["Max"]},
                        {"path": "data/memory/facts.md", "must_include": ["QA1_MUTATE"]},
                    ],
                },
            ],
        },
        {
            "id": "qa-2",
            "category": "multi-hop",
            "shared_group": "conv-shared",
            "shared_setup": shared_setup,
            "policy": shared_policy,
            "steps": [
                {
                    "action": "ask",
                    "chat_id": "eval-query",
                    "today": "2023-05-10",
                    "message": "QA2_CLEAN: What breed is Max?",
                    "must_include": ["Max"],
                    "file_checks": [
                        {"path": "data/memory/pets.md", "must_include": ["Max"]},
                        {"path": "data/memory/facts.md", "must_not_include": ["QA1_MUTATE"]},
                    ],
                },
            ],
        },
    ]

    # 先通过紧凑 JSONL 写出再读回，验证 shared_setup 序列化与加载往返
    jsonl_path = tmp_path / "shared_scenarios.jsonl"
    write_scenarios_jsonl(jsonl_path, raw_scenarios)
    loaded = load_scenarios(jsonl_path)
    assert len(loaded) == 2
    assert loaded[0]["shared_group"] == "conv-shared"
    assert "shared_setup" in loaded[0]

    setup_events: list[str] = []
    report = evaluate_memory(
        SharedGroupMockLLM(),
        loaded,
        on_setup_done=lambda rep: setup_events.append(str(rep["shared_group"])),
        git_commit="shared-sha",
    )
    assert consolidate_calls == 1
    assert setup_events == ["conv-shared"]
    assert report["summary"]["passed_scenarios"] == 2
    assert report["summary"]["failed_scenarios"] == 0
    assert "conv-shared" in report["shared_setups"]
    # 缓存统计：setup 1 次 + QA1 2 次（含工具调用） + QA2 1 次 = 4 次，无重复计入
    assert report["summary"]["cache"]["llm_calls"] == 4
    assert report["summary"]["cache"]["prompt_tokens"] == 400 + 300 + 300 + 300

    # 引用不存在的 shared_group 应在加载阶段立刻报错
    bad_path = tmp_path / "bad_shared.jsonl"
    bad_path.write_text(
        json.dumps({"id": "orphan", "shared_group": "missing-group", "steps": [{"action": "ask", "message": "hi"}]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing-group"):
        load_scenarios(bad_path)


# 测试 judge_answer 与 JUDGE_SYSTEM_PROMPT 在 is_abstention=True 下允许有效前提纠偏而不误判为编造
def test_judge_answer_prompts_allow_valid_premise_correction_for_abstention() -> None:
    captured_contexts: list[Context] = []

    # 捕获传给 judge_llm 的完整 Context 并按序返回预设判决
    class CapturingJudgeLLM(ScriptedMemoryEvalLLM):
        # 校验未传工具并记录 Context
        def complete(
            self,
            context: Context,
            options: StreamOptions | None = None,
        ) -> AssistantMessage:
            assert not context.tools
            captured_contexts.append(context)
            return super().complete(context, options)

    question = "When did Caroline go camping?"
    expected_answer = "Not enough information / not mentioned in the conversation (do not answer 'last year')."
    correction_reply = "Caroline has no camping record; it was Melanie's family who went camping last year."
    fabricated_reply = "Caroline went camping with her family last year."

    judge_llm = CapturingJudgeLLM(
        [
            (
                AssistantMessage.from_text(
                    '{"correct": true, "reason": "Valid premise correction: Caroline has no camping record and Melanie\'s family is noted from memory."}',
                    stop_reason="stop",
                ),
                TokenUsage(260, 0, 24),
            ),
            (
                AssistantMessage.from_text(
                    '{"correct": false, "reason": "Adopted the false premise about Caroline."}',
                    stop_reason="stop",
                ),
                TokenUsage(250, 0, 18),
            ),
        ]
    )

    ok_result = judge_answer(
        judge_llm,
        question,
        expected_answer,
        correction_reply,
        is_abstention=True,
    )
    bad_result = judge_answer(
        judge_llm,
        question,
        expected_answer,
        fabricated_reply,
        is_abstention=True,
    )

    assert ok_result == {
        "mode": "llm_judge",
        "correct": True,
        "reason": "Valid premise correction: Caroline has no camping record and Melanie's family is noted from memory.",
    }
    assert bad_result == {
        "mode": "llm_judge",
        "correct": False,
        "reason": "Adopted the false premise about Caroline.",
    }

    assert len(captured_contexts) == 2
    for ctx, expected_reply in zip(captured_contexts, [correction_reply, fabricated_reply], strict=True):
        assert ctx.system_prompt == JUDGE_SYSTEM_PROMPT
        assert len(ctx.messages) == 1
        assert isinstance(ctx.messages[0], UserMessage)
        user_payload = ctx.messages[0].content
        assert f"Question: {question}" in user_payload
        assert f"Reference Answer: {expected_answer}" in user_payload
        assert "is_abstention: true" in user_payload
        assert f"Assistant Reply: {expected_reply}" in user_payload

    system_prompt = captured_contexts[0].system_prompt

    # 规则 1 对正常问答（is_abstention=false）保持原有严格事实校验不变
    assert (
        "1. For normal questions (is_abstention=false): mark correct=true ONLY if the assistant's response "
        "clearly states the essential factual content of the reference answer (equivalent wording, units, "
        "or date formatting are acceptable; extra harmless context is fine; wrong numbers/entities or claiming "
        "not to know is false)."
    ) in system_prompt

    # 规则 2 必须保留 ONLY if 合取约束，防止“先拒偏、后猜测/编造”的混合回答误判通过，同时允许基于记忆的前提纠偏
    assert "2. For abstention questions (is_abstention=true):" in system_prompt
    rule2 = system_prompt.split("2. For abstention questions (is_abstention=true):", maxsplit=1)[1]
    rule2_lower = rule2.lower()
    assert "mark correct=true ONLY if" in rule2
    assert "does not adopt the false premise" in rule2_lower
    assert "does not fabricate unrecorded details" in rule2_lower
    assert "and clearly rejects the false premise" in rule2_lower
    assert "valid premise correction" in rule2_lower
    assert "from memory" in rule2_lower
    assert "different person/entity" in rule2_lower
    assert "counts as valid premise rejection (correct=true), not fabrication" in rule2_lower

    # 规则 2 仍严格把顺从错误前提或猜测/编造未记录事实判为 correct=false
    assert "correct=false" in rule2_lower
    assert "false premise were true" in rule2_lower
    assert "speculating/fabricating unrecorded facts is correct=false" in rule2_lower


# 构造测试用 OpenAI /choices 模拟响应字典
def _mock_choice_resp(content: str | None, prompt: int, cached: int, completion: int, tool_calls: list[dict[str, object]] | None = None) -> dict[str, object]:
    msg: dict[str, object] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {
        "choices": [
            {"message": msg}
        ],
        "usage": {"prompt_tokens": prompt, "prompt_cache_hit_tokens": cached, "completion_tokens": completion},
    }


# 将测试用 /choices 字典转换为标准 OpenAI SSE 行流，供 OpenAICompletionsProvider 真流式并发测试复用
def _mock_sse_from_choice_resp(resp: dict[str, object]) -> list[str]:
    choices = resp["choices"]
    assert isinstance(choices, list)
    msg = choices[0]["message"]
    assert isinstance(msg, dict)
    delta: dict[str, object] = {"role": msg.get("role", "assistant")}
    if msg.get("content"):
        delta["content"] = msg["content"]
    if msg.get("tool_calls"):
        delta["tool_calls"] = [
            {"index": i, **tc} for i, tc in enumerate(msg["tool_calls"])  # type: ignore[arg-type]
        ]
    lines = [f"data: {json.dumps({'choices': [{'index': 0, 'delta': delta}]}, ensure_ascii=False)}\n"]
    if "usage" in resp:
        lines.append(f"data: {json.dumps({'choices': [], 'usage': resp['usage']}, ensure_ascii=False)}\n")
    lines.append("data: [DONE]\n")
    return lines


# 测试 evaluate_memory(workers>1) 参数校验、真并发执行、乱序完成保序汇总、shared_group 隔离与 CLI --workers 透传
def test_evaluate_memory_concurrent_workers_preserves_order_isolation_and_cli_arg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for bad in (0, -1, True):
        with pytest.raises(ValueError, match="workers"):
            evaluate_memory(ScriptedMemoryEvalLLM([]), [], workers=bad)  # type: ignore[arg-type]

    # 1. 4 场景共享单个 ModelClient 真并发（Barrier(4) 强证）且逆序完成仍严格按原序输出
    scenarios = [
        {"id": f"sc-{i}", "category": "extraction", "steps": [{"action": "ask", "chat_id": f"c-{i}", "message": f"q #{i}", "must_include": ["nonexistent"] if i == 1 else [f"ok-{i}"]}]}
        for i in range(4)
    ]
    barrier = threading.Barrier(4, timeout=5.0)

    # 通过 Barrier(4) 与逆序休眠迫使 sc-3 先完、sc-0 后完
    def concurrent_post(url: str, payload: dict[str, object], headers: dict[str, str]) -> dict[str, object]:
        del url, headers
        msgs = payload["messages"]
        assert isinstance(msgs, list)
        idx = int(str(msgs[-1]["content"]).split("#")[-1])
        barrier.wait()
        time.sleep((3 - idx) * 0.03)
        return _mock_choice_resp(f"ok-{idx}", 200 + idx, 50 + idx, 10 + idx)

    reg1 = ApiProviderRegistry()
    reg1.register(OpenAICompletionsProvider(post_stream=lambda u, p, h: iter(_mock_sse_from_choice_resp(concurrent_post(u, p, h)))))
    shared_llm = ModelClient(
        profile=ModelProfile(
            name="demo",
            api="openai-completions",
            base_url="https://example.com/v1",
            model="demo-model",
            api_key="sk-test",
        ),
        registry=reg1,
    )
    main_thread = threading.current_thread()
    progress_events: list[tuple[int, int, str, threading.Thread]] = []
    report = evaluate_memory(
        shared_llm,
        scenarios,
        workers=4,
        on_progress=lambda d, t, r: progress_events.append((d, t, str(r["id"]), threading.current_thread())),
        git_commit="conc-sha",
    )
    assert [item["id"] for item in report["scenarios"]] == ["sc-0", "sc-1", "sc-2", "sc-3"]
    assert report["summary"]["passed_scenarios"] == 3 and report["summary"]["failed_scenarios"] == 1
    assert [item["id"] for item in report["failures"]] == ["sc-1"]
    assert report["summary"]["cache"]["llm_calls"] == 4
    assert report["summary"]["cache"]["prompt_tokens"] == 806
    assert report["summary"]["cache"]["cached_tokens"] == 206
    assert report["summary"]["cache"]["completion_tokens"] == 46
    assert [ev[:2] for ev in progress_events] == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert all(ev[3] is main_thread for ev in progress_events)

    # 2. shared_group + workers=3：前置沉淀只执行 1 次且先于 QA 完成，3 路 QA 并发且文件写入互不污染
    setup_ready, setup_lock, consolidate_calls = threading.Event(), threading.Lock(), 0
    qa_barrier = threading.Barrier(3, timeout=5.0)
    setup_steps = [
        {"action": "seed_history", "chat_id": "conv-s1", "messages": [{"role": "user", "content": "Alice adopted Max."}, {"role": "assistant", "content": "Nice."}]},
        {"action": "consolidate", "chat_id": "conv-s1", "today": "2023-05-01"},
    ]
    shared_conc_policy = {"max_active_messages": 1, "keep_recent_messages": 0, "min_batch_messages": 1}
    shared_conc_setup = {
        "record_type": "shared_setup",
        "shared_group": "conv-shared-conc",
        "policy": shared_conc_policy,
        "steps": setup_steps,
    }
    shared_scenarios = [
        {
            "id": f"shared-qa-{i}",
            "category": "single-hop",
            "shared_group": "conv-shared-conc",
            "shared_setup": shared_conc_setup,
            "policy": shared_conc_policy,
            "steps": [
                {
                    "action": "ask", "chat_id": "eval-query", "today": "2023-05-10", "message": f"QA{i}_TAG: dog name?", "must_include": ["Max"],
                    "file_checks": [{"path": "data/memory/pets.md", "must_include": ["Max"]}, {"path": "data/memory/facts.md", "must_include": ["QA0_TAG"] if i == 0 else [], "must_not_include": [] if i == 0 else ["QA0_TAG"]}],
                }
            ],
        }
        for i in range(3)
    ]

    # 处理共享前置沉淀与 3 路并发 QA 请求的模拟 POST
    def shared_post(url: str, payload: dict[str, object], headers: dict[str, str]) -> dict[str, object]:
        del url, headers
        nonlocal consolidate_calls
        msgs = payload["messages"]
        assert isinstance(msgs, list)
        if payload.get("tools") is None and "后台记忆整理器" in str(msgs[0].get("content") or ""):
            with setup_lock:
                consolidate_calls += 1
            setup_ready.set()
            body = json.dumps({"session_summary": "Max.", "memories": [{"topic": "pets", "title": "宠物", "index_summary": "宠物", "operations": [{"op": "add", "text": "Alice adopted Max."}]}]}, ensure_ascii=False)
            return _mock_choice_resp(body, 400, 0, 40)
        last_user = next(str(m.get("content") or "") for m in reversed(msgs) if m.get("role") == "user")
        if not any(m.get("role") == "tool" for m in msgs):
            assert setup_ready.is_set()
            qa_barrier.wait()
            if "QA0_TAG" in last_user:
                return _mock_choice_resp(None, 300, 0, 15, tool_calls=[{"id": "call-qa0", "type": "function", "function": {"name": "remember_fact", "arguments": "{}"}}])
        return _mock_choice_resp("Alice's dog is named Max.", 310, 120, 12)

    reg2 = ApiProviderRegistry()
    reg2.register(OpenAICompletionsProvider(post_stream=lambda u, p, h: iter(_mock_sse_from_choice_resp(shared_post(u, p, h)))))
    shared_report = evaluate_memory(
        ModelClient(
            profile=ModelProfile(
                name="demo",
                api="openai-completions",
                base_url="https://example.com/v1",
                model="demo-model",
                api_key="sk-test",
            ),
            registry=reg2,
        ),
        shared_scenarios,
        workers=3,
        git_commit="shared-conc-sha",
    )
    assert consolidate_calls == 1 and shared_report["summary"]["passed_scenarios"] == 3
    assert [item["id"] for item in shared_report["scenarios"]] == ["shared-qa-0", "shared-qa-1", "shared-qa-2"]

    # 3. 验证 CLI --workers 校验及透传
    with pytest.raises(ValueError, match="workers"):
        main(["--mode", "memory", "--workers", "0"])
    cli_cases = tmp_path / "cli_cases.jsonl"
    cli_cases.write_text(json.dumps(scenarios[0], ensure_ascii=False) + "\n", encoding="utf-8")
    captured: list[int] = []
    monkeypatch.setattr(
        "evals.cli.evaluate_memory",
        lambda llm, sc, **kw: captured.append(int(kw["workers"])) or {"mode": "memory", "git_commit": "cli", "summary": {}, "failures": [], "scenarios": []},
    )
    assert main(["--mode", "memory", "--cases", str(cli_cases), "--workers", "4", "--no-judge"]) == 0
    assert captured == [4]


# 静态守卫：JUDGE_SYSTEM_PROMPT 不得泄漏 LoCoMo / LongMemEval 测试集专有实体与场景名，且须暴露版本与 SHA256 指纹
def test_judge_system_prompt_has_no_benchmark_entity_leakage_and_exports_fingerprint() -> None:
    from evals.memory_eval import JUDGE_PROMPT_SHA256, JUDGE_PROMPT_VERSION

    forbidden_entities = (
        "caroline",
        "melanie",
        "jon",
        "gina",
        "joanna",
        "nate",
        "tim",
        "maria",
        "camping",
    )
    prompt_lower = JUDGE_SYSTEM_PROMPT.lower()
    for entity in forbidden_entities:
        assert not re.search(rf"(?<![a-z]){re.escape(entity)}(?![a-z])", prompt_lower), (
            f"JUDGE_SYSTEM_PROMPT 泄漏测试集实体：{entity!r}"
        )

    assert JUDGE_PROMPT_VERSION == "v2-neutral"
    expected_sha = hashlib.sha256(JUDGE_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]
    assert JUDGE_PROMPT_SHA256 == expected_sha


# 测试 evaluate_memory 报告顶层记录 judge 元数据（含禁用、Noop、自评与独立评判四种组合）
def test_evaluate_memory_records_judge_metadata() -> None:
    from evals.memory_eval import JUDGE_PROMPT_SHA256, JUDGE_PROMPT_VERSION

    # 带 model 属性的脚本化假模型，便于验证 judge.model 记录
    class NamedScriptedLLM(ScriptedMemoryEvalLLM):
        # 初始化模型名称与预设回复队列
        def __init__(self, model: str, responses: list[tuple[AssistantMessage, TokenUsage | None]]) -> None:
            super().__init__(responses)
            self.model = model

    scenarios = [
        {
            "id": "sc-judge-meta",
            "category": "extraction",
            "steps": [{"action": "ask", "chat_id": "s1", "message": "hi", "must_include": ["ok"]}],
        }
    ]

    rep_none = evaluate_memory(
        NamedScriptedLLM("agent-m", [(AssistantMessage.from_text("ok", stop_reason="stop"), None)]),
        scenarios,
        judge_llm=None,
        git_commit="meta-1",
    )
    assert rep_none["judge"] == {
        "enabled": False,
        "model": None,
        "self_judged": False,
        "prompt_version": None,
        "prompt_sha256": None,
    }

    rep_noop = evaluate_memory(
        NamedScriptedLLM("agent-m", [(AssistantMessage.from_text("ok", stop_reason="stop"), None)]),
        scenarios,
        judge_llm=NoopModelClient(),
        git_commit="meta-2",
    )
    assert rep_noop["judge"] == {
        "enabled": False,
        "model": None,
        "self_judged": False,
        "prompt_version": None,
        "prompt_sha256": None,
    }

    self_llm = NamedScriptedLLM("deepseek-chat", [(AssistantMessage.from_text("ok", stop_reason="stop"), None)])
    rep_self = evaluate_memory(self_llm, scenarios, judge_llm=self_llm, git_commit="meta-3")
    assert rep_self["judge"] == {
        "enabled": True,
        "model": "deepseek-chat",
        "self_judged": True,
        "prompt_version": JUDGE_PROMPT_VERSION,
        "prompt_sha256": JUDGE_PROMPT_SHA256,
    }

    agent_llm = NamedScriptedLLM("deepseek-reasoner", [(AssistantMessage.from_text("ok", stop_reason="stop"), None)])
    judge_llm = NamedScriptedLLM("deepseek-chat", [])
    rep_sep = evaluate_memory(agent_llm, scenarios, judge_llm=judge_llm, git_commit="meta-4")
    assert rep_sep["judge"] == {
        "enabled": True,
        "model": "deepseek-chat",
        "self_judged": False,
        "prompt_version": JUDGE_PROMPT_VERSION,
        "prompt_sha256": JUDGE_PROMPT_SHA256,
    }


# 测试单场景与 shared_setup 异常容错隔离：并发中单题抛网络/运行时异常不拖垮其余场景，KeyboardInterrupt 照常抛出
def test_evaluate_memory_isolates_scenario_and_shared_setup_exceptions() -> None:
    # 根据消息关键词按需抛出指定异常的并发假模型
    class FaultyScenarioLLM:
        last_usage: TokenUsage | None = None

        # 按场景内容决定返回正常答案、触发沉淀失败或抛出网络异常
        def complete(
            self,
            context: Context,
            options: StreamOptions | None = None,
        ) -> AssistantMessage:
            del options
            joined = "\n".join(
                m.content if isinstance(m, UserMessage) else (m.text if isinstance(m, AssistantMessage) else m.content)
                for m in context.messages
            )
            if not context.tools and "后台记忆整理器" in context.system_prompt:
                if "FAIL_SETUP" in joined:
                    raise RuntimeError("shared setup boom")
                return AssistantMessage.from_text(
                    json.dumps(
                        {
                            "session_summary": "ok",
                            "memories": [
                                {
                                    "topic": "facts",
                                    "title": "事实",
                                    "index_summary": "摘要",
                                    "operations": [{"op": "add", "text": "fact-ok"}],
                                }
                            ],
                        }
                    ),
                    stop_reason="stop",
                )
            if "RAISE_URL_ERROR" in joined:
                raise URLError("temporary network glitch")
            if "RAISE_KB_INT" in joined:
                raise KeyboardInterrupt("user aborted")
            return AssistantMessage.from_text("ok-reply", stop_reason="stop")

        # 流式入口复用 complete
        def stream(
            self,
            context: Context,
            options: StreamOptions | None = None,
        ) -> Iterator[AssistantMessageEvent]:
            msg = self.complete(context, options)
            if msg.text:
                yield TextDelta(content_index=0, text=msg.text, partial=msg)
            yield StreamDone(reason="stop", message=msg)

    scenarios = [
        {"id": "sc-ok-1", "category": "extraction", "steps": [{"action": "ask", "message": "hello 1", "must_include": ["ok-reply"]}]},
        {"id": "sc-net-err", "category": "extraction", "steps": [{"action": "ask", "message": "RAISE_URL_ERROR", "must_include": ["ok-reply"]}]},
        {"id": "sc-ok-2", "category": "extraction", "steps": [{"action": "ask", "message": "hello 2", "must_include": ["ok-reply"]}]},
        {
            "id": "sc-bad-setup",
            "category": "multi-session",
            "shared_group": "bad-grp",
            "shared_setup": {
                "record_type": "shared_setup",
                "shared_group": "bad-grp",
                "policy": {"max_active_messages": 1, "keep_recent_messages": 0, "min_batch_messages": 1},
                "steps": [
                    {"action": "seed_history", "chat_id": "b1", "messages": [{"role": "user", "content": "FAIL_SETUP"}, {"role": "assistant", "content": "ack"}]},
                    {"action": "consolidate", "chat_id": "b1", "today": "2026-11-20"},
                ],
            },
            "policy": {"max_active_messages": 1, "keep_recent_messages": 0, "min_batch_messages": 1},
            "steps": [
                {"action": "ask", "chat_id": "b2", "message": "should skip", "must_include": ["ok-reply"]},
            ],
        },
        {
            "id": "sc-good-setup",
            "category": "multi-session",
            "shared_group": "good-grp",
            "shared_setup": {
                "record_type": "shared_setup",
                "shared_group": "good-grp",
                "policy": {"max_active_messages": 1, "keep_recent_messages": 0, "min_batch_messages": 1},
                "steps": [
                    {"action": "seed_history", "chat_id": "g1", "messages": [{"role": "user", "content": "GOOD_SETUP"}, {"role": "assistant", "content": "ack"}]},
                    {"action": "consolidate", "chat_id": "g1", "today": "2026-11-20"},
                ],
            },
            "policy": {"max_active_messages": 1, "keep_recent_messages": 0, "min_batch_messages": 1},
            "steps": [
                {"action": "ask", "chat_id": "g2", "message": "ask good", "must_include": ["ok-reply"]},
            ],
        },
    ]

    report = evaluate_memory(FaultyScenarioLLM(), scenarios, workers=3, git_commit="err-iso")
    assert report["summary"]["total_scenarios"] == 5
    assert report["summary"]["passed_scenarios"] == 3
    assert report["summary"]["failed_scenarios"] == 2
    by_id = {item["id"]: item for item in report["scenarios"]}
    assert by_id["sc-ok-1"]["passed"] is True
    assert by_id["sc-ok-2"]["passed"] is True
    assert by_id["sc-good-setup"]["passed"] is True
    assert by_id["sc-net-err"]["passed"] is False
    assert by_id["sc-net-err"]["reasons"] == ["scenario_exception: URLError: <urlopen error temporary network glitch>"]
    assert by_id["sc-bad-setup"]["passed"] is False
    assert any("scenario_exception" in str(r) and "RuntimeError" in str(r) for r in by_id["sc-bad-setup"]["reasons"])

    # KeyboardInterrupt 绝不能被容错层吞掉
    with pytest.raises(KeyboardInterrupt):
        evaluate_memory(
            FaultyScenarioLLM(),
            [{"id": "sc-kb", "category": "extraction", "steps": [{"action": "ask", "message": "RAISE_KB_INT"}]}],
            workers=1,
        )


# 测试词面校验对短 ASCII 词（如 Go / 90）执行词边界匹配防止 Google/900 假阳性，并检查场景集收紧规则
def test_lexical_word_boundary_and_tightened_memory_scenarios() -> None:
    scenarios = [
        {
            "id": "boundary-check",
            "category": "compaction-cache",
            "steps": [
                {
                    "action": "ask",
                    "chat_id": "s1",
                    "message": "总结选型",
                    "must_include": ["Go", "90", "Parquet"],
                    "must_include_any": ["企微", "企业微信"],
                }
            ],
        }
    ]

    # 1. Google/good 不应误命中 Go，900 不应误命中 90
    false_pos_llm = ScriptedMemoryEvalLLM(
        [
            (AssistantMessage.from_text("Google 的 good 方案，企业微信通知，90 天，Parquet。", stop_reason="stop"), None),
            (AssistantMessage.from_text("采用 Go 语言，企业微信通知，900 天，Parquet。", stop_reason="stop"), None),
        ]
    )
    rep_go_fail = evaluate_memory(false_pos_llm, scenarios, git_commit="wb-1")
    assert rep_go_fail["summary"]["passed_scenarios"] == 0
    assert "must_include 缺失：'Go'" in str(rep_go_fail["failures"][0]["reasons"])

    rep_90_fail = evaluate_memory(false_pos_llm, scenarios, git_commit="wb-2")
    assert rep_90_fail["summary"]["passed_scenarios"] == 0
    assert "must_include 缺失：'90'" in str(rep_90_fail["failures"][0]["reasons"])

    # 2. 中文紧贴边界（如 Go语言 / 90天）与企业微信全称应正常通过
    valid_llm = ScriptedMemoryEvalLLM(
        [
            (AssistantMessage.from_text("采用Go语言编写，接入企业微信机器人，冷数据归档90天，格式为Parquet。", stop_reason="stop"), None),
        ]
    )
    rep_pass = evaluate_memory(valid_llm, scenarios, git_commit="wb-3")
    assert rep_pass["summary"]["passed_scenarios"] == 1

    # 3. 校验 evals/memory_scenarios.jsonl 已收紧过短数字与全简称兼容
    repo_root = Path(__file__).resolve().parent.parent
    loaded = {item["id"]: item for item in load_scenarios(repo_root / "evals" / "memory_scenarios.jsonl")}

    rolling_ask = loaded["rolling-multi-compaction"]["steps"][-1]
    assert rolling_ask["must_include"] == ["Go", "90", "Parquet"]
    assert rolling_ask["must_include_any"] == ["企微", "企业微信"]

    synth_ask = loaded["cross-session-synthesis"]["steps"][-1]
    assert synth_ask["must_include"] == ["京都", "12000", "祇园"]
    assert synth_ask["must_include_any"] == ["4 晚", "4晚", "四晚"]

    partial_ask = loaded["partial-attribute-update"]["steps"][-1]
    assert partial_ask["must_include"] == ["法兰克福", "8080"]
    assert partial_ask["must_include_any"] == ["30 秒", "30秒", "30s"]

    accum_ask = loaded["multi-session-accumulation"]["steps"][-1]
    assert accum_ask["must_include"] == ["DDIA", "SICP", "CSAPP"]
    assert "3" not in accum_ask["must_include_any"]
    assert "三" not in accum_ask["must_include_any"]

    # 4. 校验已移除 len<=3 魔法阈值：任意长度纯 ASCII 词/数字均受同类边界保护，中文或标点走子串匹配
    from evals.memory_eval import _contains_keyword

    assert not _contains_keyword("Google is great", "Go")
    assert not _contains_keyword("JavaScript runtime", "Java")
    assert not _contains_keyword("端口为 10000", "1000")
    assert not _contains_keyword("100 天", "10")
    assert _contains_keyword("使用Java开发，端口1000元", "Java")
    assert _contains_keyword("使用Java开发，端口1000元", "1000")
    assert _contains_keyword("归档周期 10天", "10")
    assert _contains_keyword("采用 Go语言", "Go")
    assert _contains_keyword("编号 ok-0 已完成", "ok-0")
    assert _contains_keyword("接入企业微信通知", "企业微信")


# 测试 memory_eval 复用的 AssistantApp.consolidator 真实路由到双模型 consolidator_llm 且 CLI 仅加载一次 ToolsConfig
def test_memory_eval_uses_dual_model_consolidator_and_cli_loads_tools_config_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from personal_assistant.assistant.commands import ToolsConfig
    from personal_assistant.assistant.common import REPO_ROOT

    main_calls: list[str] = []
    cons_calls: list[str] = []

    # 仅处理前台问答，若误入沉淀提示词则直接报错
    class MainOnlyLLM:
        last_usage: TokenUsage | None = None

        # 记录前台问答请求并返回预设答复
        def complete(
            self,
            context: Context,
            options: StreamOptions | None = None,
        ) -> AssistantMessage:
            del options
            assert "后台记忆整理器" not in context.system_prompt
            last = context.messages[-1]
            assert isinstance(last, UserMessage)
            main_calls.append(last.content)
            return AssistantMessage.from_text("星云端口为 7788。", stop_reason="stop")

        # 流式入口复用 complete
        def stream(
            self,
            context: Context,
            options: StreamOptions | None = None,
        ) -> Iterator[AssistantMessageEvent]:
            msg = self.complete(context, options)
            yield TextDelta(content_index=0, text=msg.text, partial=msg)
            yield StreamDone(reason="stop", message=msg)

    # 仅处理后台记忆沉淀，验证 consolidate 步骤真正经由 app.consolidator 调用双模型
    class ConsolidatorOnlyLLM:
        last_usage: TokenUsage | None = None

        # 记录沉淀请求并返回结构化 JSON 记忆操作
        def complete(
            self,
            context: Context,
            options: StreamOptions | None = None,
        ) -> AssistantMessage:
            del options
            assert "后台记忆整理器" in context.system_prompt
            cons_calls.append("consolidate")
            return AssistantMessage.from_text(
                json.dumps(
                    {
                        "session_summary": "星云服务监听端口 7788。",
                        "memories": [
                            {
                                "topic": "projects",
                                "title": "项目",
                                "index_summary": "星云端口",
                                "operations": [{"op": "add", "text": "星云服务监听端口 7788。"}],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                stop_reason="stop",
            )

        # 流式入口复用 complete
        def stream(
            self,
            context: Context,
            options: StreamOptions | None = None,
        ) -> Iterator[AssistantMessageEvent]:
            msg = self.complete(context, options)
            yield StreamDone(reason="stop", message=msg)

    scenarios = [
        {
            "id": "dual-consolidator-check",
            "category": "extraction",
            "policy": {"max_active_messages": 2, "keep_recent_messages": 0, "min_batch_messages": 2},
            "steps": [
                {
                    "action": "seed_history",
                    "chat_id": "s1",
                    "messages": [
                        {"role": "user", "content": "星云服务端口 7788。"},
                        {"role": "assistant", "content": "已收到。"},
                    ],
                },
                {"action": "consolidate", "chat_id": "s1", "today": "2026-11-20"},
                {
                    "action": "ask",
                    "chat_id": "s2",
                    "message": "星云端口是多少？",
                    "must_include": ["7788"],
                    "file_checks": [{"path": "data/memory/projects.md", "must_include": ["7788"]}],
                },
            ],
        }
    ]
    report = evaluate_memory(
        MainOnlyLLM(),
        scenarios,
        consolidator_llm=ConsolidatorOnlyLLM(),
        git_commit="dual-sha",
    )
    assert report["summary"]["passed_scenarios"] == 1
    assert cons_calls == ["consolidate"]
    assert len(main_calls) == 1

    # 验证 evals/cli.py 单次加载 ToolsConfig 并透传 tools_config 与 models.dsl 中的 consolidator 模型
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    tools_json = cfg_dir / "tools.json"
    tools_json.write_text((REPO_ROOT / "config" / "tools.json").read_text(encoding="utf-8"), encoding="utf-8")
    (cfg_dir / "models.dsl").write_text(
        "use reasoner sk-r1\nadd deepseek sk-ds\nconsolidator deepseek\n",
        encoding="utf-8",
    )
    sc_file = tmp_path / "sc.jsonl"
    sc_file.write_text(
        json.dumps({"id": "m1", "steps": [{"action": "ask", "message": "hi"}]}) + "\n",
        encoding="utf-8",
    )

    load_count = 0
    orig_load = ToolsConfig.load

    # 统计 ToolsConfig.load 调用次数
    def counting_load(cls, config_path: Path, project_root: Path | None = None) -> ToolsConfig:
        nonlocal load_count
        load_count += 1
        return orig_load(config_path, project_root=project_root)

    captured_kwargs: dict[str, object] = {}
    monkeypatch.setattr(ToolsConfig, "load", classmethod(counting_load))
    monkeypatch.setattr(
        "evals.cli.evaluate_memory",
        lambda llm, _sc, **kw: captured_kwargs.update({"llm": llm, **kw})
        or {"mode": "memory", "git_commit": "t", "summary": {}, "failures": [], "scenarios": []},
    )
    assert main(["--mode", "memory", "--config", str(tools_json), "--cases", str(sc_file), "--no-judge"]) == 0
    assert load_count == 1
    assert isinstance(captured_kwargs.get("tools_config"), ToolsConfig)
    cons_obj = captured_kwargs.get("consolidator_llm")
    assert isinstance(cons_obj, ModelClient)
    assert cons_obj.api == "openai-completions"
    assert cons_obj.model == "deepseek-chat"
