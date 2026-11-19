"""记忆与多轮端到端评测器（memory_eval）自身单元测试（TDD）。"""
from __future__ import annotations

import json
from pathlib import Path

from personal_assistant.eval import evaluate_memory, main
from personal_assistant.llm import LLMClient, TokenUsage


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
    assert len(lines) >= 4
    ids = set()
    for line in lines:
        item = json.loads(line)
        assert item["id"] not in ids
        ids.add(item["id"])
        assert isinstance(item.get("steps"), list) and item["steps"]
