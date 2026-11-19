"""评测模块自身单元测试：验证离线检索评测、隔离 Agent 评测与报告路径保护。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from personal_assistant.eval import evaluate_agent, evaluate_search, load_cases, main
from personal_assistant.llm import LLMClient
from personal_assistant.memory import MemoryStore


# 预录回复的假模型，用于确定性测试端到端 Agent 评测器
class ScriptedLLM(LLMClient):
    # 保存预设回复队列
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = list(responses)

    # 按序弹出下一条预设回复
    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return self._responses.pop(0)


# 测试公开样本集离线检索评测，并确认不会改写样本目录任何文件
def test_sample_search_eval_runs_without_mutating_data_dir() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    sample_data = repo_root / "evals" / "sample_data"
    cases = load_cases(repo_root / "evals" / "sample_cases.jsonl")

    before_entries = sorted(p.relative_to(sample_data).as_posix() for p in sample_data.rglob("*"))
    memory = MemoryStore(sample_data, create_dirs=False)
    report = evaluate_search(memory, cases, limit=5, git_commit="test-sha")
    after_entries = sorted(p.relative_to(sample_data).as_posix() for p in sample_data.rglob("*"))

    assert before_entries == after_entries
    assert report["git_commit"] == "test-sha"
    assert report["summary"]["total_cases"] == 4
    assert report["summary"]["mean_source_recall"] == 1.0
    assert report["summary"]["mean_quote_coverage"] == 1.0
    assert report["summary"]["negative_clean_rate"] == 1.0
    assert report["failures"] == []


# 测试检索评测的金标原文校验与未命中失败案例保留机制
def test_search_eval_validates_gold_quotes_and_reports_failures(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "memory").mkdir(parents=True)
    (data_dir / "memory" / "a.md").write_text("# 标题\n真实存在的原文句子。\n", encoding="utf-8")
    memory = MemoryStore(data_dir, create_dirs=False)

    # 金标引文在原文中不存在时应立即报错
    with pytest.raises(ValueError, match="未在原文中找到"):
        evaluate_search(
            memory,
            [
                {
                    "id": "bad-gold",
                    "query": "标题",
                    "expected_sources": [{"path": "data/memory/a.md", "quote": "不存在的引文"}],
                }
            ],
        )

    # 检索漏召回时应完整记录到 failures
    report = evaluate_search(
        memory,
        [
            {
                "id": "miss-case",
                "query": "完全无关的检索词xyz",
                "expected_sources": [{"path": "data/memory/a.md", "quote": "真实存在的原文句子"}],
            }
        ],
    )
    assert report["summary"]["failed_cases"] == 1
    assert report["failures"][0]["id"] == "miss-case"


# 测试 Agent 端到端评测的数据目录隔离性与工具调用轨迹记录
def test_agent_eval_isolates_data_dir_and_records_tool_trace(tmp_path: Path) -> None:
    data_dir = tmp_path / "real_data"
    (data_dir / "memory").mkdir(parents=True)
    (data_dir / "memory" / "note.md").write_text("# 规范\n使用固定 argv 防止注入。\n", encoding="utf-8")
    cases = [
        {
            "id": "c1",
            "query": "如何防止命令注入？记住这个问题。",
            "answerable": True,
            "expected_sources": [{"path": "data/memory/note.md", "quote": "使用固定 argv 防止注入"}],
            "expected_tools": ["search_markdown"],
        }
    ]
    llm = ScriptedLLM(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_markdown",
                            "arguments": '{"query": "固定 argv 注入", "root": "data/memory"}',
                        },
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "remember_fact", "arguments": "{}"},
                    },
                ],
            },
            {"role": "assistant", "content": "规范指出应使用固定 argv 防止注入。"},
        ]
    )

    report = evaluate_agent(llm, data_dir, cases, git_commit="sha123")

    assert report["summary"]["expected_tools_all_hit"] == 1
    assert report["summary"]["cache"]["llm_calls"] == 2
    assert float(report["summary"]["cache"]["mean_prefix_reuse_rate"]) > 0.7
    assert report["cases"][0]["called_tools"] == ["search_markdown", "remember_fact"]
    assert report["cases"][0]["cache"]["llm_calls"] == 2
    assert report["cases"][0]["review_status"] == "needs_human_review"
    # 即使评测过程中调用了 remember_fact，原始数据目录也不应被写入 MEMORY.md 或 facts.md
    assert not (data_dir / "memory" / "MEMORY.md").exists()
    assert not (data_dir / "memory" / "facts.md").exists()


# 测试评测 CLI 输出报告以及拦截把报告写进 --data-dir 内部的行为
def test_eval_cli_writes_report_and_blocks_output_inside_data_dir(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    sample_data = repo_root / "evals" / "sample_data"
    sample_cases = repo_root / "evals" / "sample_cases.jsonl"

    with pytest.raises(ValueError, match="不能放在 --data-dir"):
        main(
            [
                "--mode",
                "search",
                "--data-dir",
                str(sample_data),
                "--cases",
                str(sample_cases),
                "--output",
                str(sample_data / "report.json"),
            ]
        )

    out = tmp_path / "report.json"
    code = main(
        [
            "--mode",
            "search",
            "--data-dir",
            str(sample_data),
            "--cases",
            str(sample_cases),
            "--output",
            str(out),
        ]
    )
    assert code == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["mode"] == "search"
    assert saved["summary"]["total_cases"] == 4
