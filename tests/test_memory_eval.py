"""记忆与多轮端到端评测器（memory_eval）自身单元测试（TDD）。"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from evals import evaluate_memory, load_scenarios, main, write_scenarios_jsonl
from personal_assistant.ai import LLMClient, TokenUsage


# 可按序返回预设消息与 usage 的脚本化假模型，用于确定性测试 memory_eval
class ScriptedMemoryEvalLLM(LLMClient):
    # 保存预设回复与可选 usage 队列
    def __init__(self, responses: list[tuple[dict[str, object], TokenUsage | None]]) -> None:
        self._responses = list(responses)
        self.last_usage: TokenUsage | None = None

    # 弹出下一条预设回复并更新 last_usage
    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        reply, usage = self._responses.pop(0)
        self.last_usage = usage
        return reply


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
            ({"role": "assistant", "content": "明白，端口 9092，明天演练会。"}, TokenUsage(1000, 0, 20)),
            # chat 2
            ({"role": "assistant", "content": "好的。"}, TokenUsage(1100, 960, 10)),
            # consolidate
            (
                {
                    "role": "assistant",
                    "content": json.dumps(
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
                },
                TokenUsage(900, 0, 80),
            ),
            # ask (s2) -> tool call read_markdown -> final reply
            (
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "read_markdown",
                                "arguments": '{"path": "data/memory/projects.md", "start_line": "1", "end_line": "20"}',
                            },
                        }
                    ],
                },
                TokenUsage(1200, 768, 25),
            ),
            ({"role": "assistant", "content": "北极星当前端口是 9092，目前没有待办演练会记录。"}, TokenUsage(1300, 1152, 25)),
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
        [({"role": "assistant", "content": "你的显示器是 Dell U2723QE。"}, None)]
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
        # 记录非沉淀阶段的 system prompt 并返回预设回复
        def respond(
            self,
            messages: list[dict[str, object]],
            tools: list[dict[str, object]] | None = None,
        ) -> dict[str, object]:
            if tools:
                seen_ask_system_prompts.append(str(messages[0].get("content") or ""))
            return super().respond(messages, tools)

    agent_llm = ClockCapturingLLM(
        [
            # consolidate
            (
                {
                    "role": "assistant",
                    "content": json.dumps(
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
                },
                TokenUsage(500, 0, 60),
            ),
            # ask -> search_markdown
            (
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "search_markdown",
                                "arguments": '{"query": "5K", "root": "data/memory"}',
                            },
                        }
                    ],
                },
                TokenUsage(900, 512, 20),
            ),
            # ask -> final reply
            (
                {"role": "assistant", "content": "Your personal best 5K time is 25:50."},
                TokenUsage(1050, 896, 20),
            ),
        ]
    )

    judge_llm = ScriptedMemoryEvalLLM(
        [
            (
                {
                    "role": "assistant",
                    "content": '{"correct": true, "reason": "Answer matches 25:50 exactly."}',
                },
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
        [({"role": "assistant", "content": "You bought three cows from Peter."}, None)]
    )
    judge_llm = ScriptedMemoryEvalLLM(
        [({"role": "assistant", "content": '{"correct": false, "reason": "Hallucinated instead of abstaining."}'}, None)]
    )
    report = evaluate_memory(agent_llm, scenarios, judge_llm=judge_llm, git_commit="judge-fail")
    assert report["summary"]["passed_scenarios"] == 0
    assert report["summary"]["failed_scenarios"] == 1
    assert "judge" in str(report["failures"][0]["reason"]).lower()


# 测试 evaluate_memory 支持 workers 多线程并发，严格保持原始场景顺序并触发实时进度回调
def test_evaluate_memory_parallel_workers_preserves_order_and_reports_progress() -> None:
    # 线程安全假模型：前几题人为延迟更长，验证并发乱序完成但报告顺序严格稳定
    class DelayedParallelLLM(LLMClient):
        # 初始化线程局部 usage 存储
        def __init__(self) -> None:
            self._local = threading.local()

        # 读取当前线程的最近一次 usage
        @property
        def last_usage(self) -> TokenUsage | None:
            return getattr(self._local, "usage", None)

        # 按题目编号反向延迟后返回确定性答复
        def respond(
            self,
            messages: list[dict[str, object]],
            tools: list[dict[str, object]] | None = None,
        ) -> dict[str, object]:
            user_text = str(messages[-1].get("content") or "")
            idx = int(user_text.split("#")[-1])
            time.sleep(0.04 * (3 - idx))
            self._local.usage = TokenUsage(prompt_tokens=100 + idx, cached_tokens=50, completion_tokens=10)
            return {"role": "assistant", "content": f"ok-{idx}"}

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
        for idx in range(4)
    ]

    with pytest.raises(ValueError, match="workers"):
        evaluate_memory(DelayedParallelLLM(), scenarios, workers=0)

    progress_events: list[tuple[int, int, str]] = []

    # 收集并发回调的完成进度事件
    def on_progress(done: int, total: int, record: dict[str, object]) -> None:
        progress_events.append((done, total, str(record["id"])))

    report = evaluate_memory(
        DelayedParallelLLM(),
        scenarios,
        workers=3,
        on_progress=on_progress,
        git_commit="par-sha",
    )
    assert report["workers"] == 3
    assert [item["id"] for item in report["scenarios"]] == ["sc-0", "sc-1", "sc-2", "sc-3"]
    assert report["summary"]["passed_scenarios"] == 4
    assert report["summary"]["cache"]["llm_calls"] == 4
    assert report["summary"]["cache"]["prompt_tokens"] == 100 + 101 + 102 + 103
    assert len(progress_events) == 4
    assert [ev[0] for ev in progress_events] == [1, 2, 3, 4]
    assert all(ev[1] == 4 for ev in progress_events)


# 测试 shared_group 共享长记忆模式：同一对话只沉淀一次，每道 QA 在独立隔离副本上并发运行不串状态
def test_evaluate_memory_shared_group_consolidates_once_and_isolates_each_qa(tmp_path: Path) -> None:
    lock = threading.Lock()
    consolidate_calls = 0

    # 根据请求内容区分沉淀调用与 QA 问答调用的线程安全假模型
    class SharedGroupMockLLM(LLMClient):
        # 初始化线程局部 usage
        def __init__(self) -> None:
            self._local = threading.local()

        # 读取当前线程最近一次 usage
        @property
        def last_usage(self) -> TokenUsage | None:
            return getattr(self._local, "usage", None)

        # 沉淀阶段写入 pets.md；QA1 故意调 remember_fact 写入隔离标记；QA2 仅读 pets.md 验证无串台
        def respond(
            self,
            messages: list[dict[str, object]],
            tools: list[dict[str, object]] | None = None,
        ) -> dict[str, object]:
            nonlocal consolidate_calls
            sys_text = str(messages[0].get("content") or "")
            if tools is None and "后台记忆整理器" in sys_text:
                with lock:
                    consolidate_calls += 1
                self._local.usage = TokenUsage(prompt_tokens=400, cached_tokens=0, completion_tokens=50)
                return {
                    "role": "assistant",
                    "content": json.dumps(
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
                }

            last_role = str(messages[-1].get("role") or "")
            user_prompt = next(
                (str(m.get("content") or "") for m in reversed(messages) if m.get("role") == "user"),
                "",
            )
            self._local.usage = TokenUsage(prompt_tokens=300, cached_tokens=150, completion_tokens=20)
            if "QA1_MUTATE" in user_prompt and last_role == "user":
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_mut",
                            "type": "function",
                            "function": {"name": "remember_fact", "arguments": "{}"},
                        }
                    ],
                }
            return {"role": "assistant", "content": "The golden retriever is named Max."}

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
    raw_scenarios = [
        {
            "id": "qa-1",
            "category": "single-hop",
            "shared_group": "conv-shared",
            "setup_step_count": len(setup_steps),
            "policy": {"max_active_messages": 1, "keep_recent_messages": 0, "min_batch_messages": 1},
            "steps": [
                *setup_steps,
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
            "setup_step_count": len(setup_steps),
            "policy": {"max_active_messages": 1, "keep_recent_messages": 0, "min_batch_messages": 1},
            "steps": [
                *setup_steps,
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

    report = evaluate_memory(SharedGroupMockLLM(), loaded, workers=2, git_commit="shared-sha")
    assert consolidate_calls == 1
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
