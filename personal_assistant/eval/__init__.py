"""评测包统一导出接口，方便外部按 `from personal_assistant.eval import ...` 引用。"""
from personal_assistant.eval.agent_eval import evaluate_agent
from personal_assistant.eval.cli import main
from personal_assistant.eval.common import current_git_commit, load_cases, safe_write_report
from personal_assistant.eval.memory_eval import evaluate_memory, judge_answer, load_scenarios
from personal_assistant.eval.public_benchmarks import (
    convert_locomo_for_memory,
    convert_longmemeval_dataset,
    convert_longmemeval_item,
    export_locomo_for_search,
)
from personal_assistant.eval.search_eval import evaluate_search

__all__ = [
    "convert_locomo_for_memory",
    "convert_longmemeval_dataset",
    "convert_longmemeval_item",
    "current_git_commit",
    "evaluate_agent",
    "evaluate_memory",
    "evaluate_search",
    "export_locomo_for_search",
    "judge_answer",
    "load_cases",
    "load_scenarios",
    "main",
    "safe_write_report",
]
