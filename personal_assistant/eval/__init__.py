"""评测包统一导出接口，方便外部按 `from personal_assistant.eval import ...` 引用。"""
from personal_assistant.eval.agent_eval import evaluate_agent
from personal_assistant.eval.cli import main
from personal_assistant.eval.common import current_git_commit, load_cases, safe_write_report
from personal_assistant.eval.search_eval import evaluate_search

__all__ = [
    "current_git_commit",
    "evaluate_agent",
    "evaluate_search",
    "load_cases",
    "main",
    "safe_write_report",
]
