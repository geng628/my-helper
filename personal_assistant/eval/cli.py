"""评测命令行入口：按 --mode 参数分别调度离线检索评测或端到端 Agent 评测。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from personal_assistant.cli import REPO_ROOT, load_dotenv
from personal_assistant.eval.agent_eval import evaluate_agent
from personal_assistant.eval.common import load_cases, safe_write_report
from personal_assistant.eval.memory_eval import evaluate_memory, load_scenarios
from personal_assistant.eval.search_eval import evaluate_search
from personal_assistant.llm import from_env
from personal_assistant.memory import MemoryStore


# 解析命令行参数、执行对应模式评测、输出汇总并按需保存 JSON 报告
def main(argv: list[str] | None = None) -> int:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description="My Helper 评测运行器（search 离线检索 / agent 单轮评测 / memory 多轮记忆生命周期评测）")
    parser.add_argument("--mode", choices=["search", "agent", "memory"], default="search")
    parser.add_argument("--data-dir", default="evals/sample_data", help="评测数据目录（含 memory/、workspace/）")
    parser.add_argument("--cases", default=None, help="JSONL 评测集路径（search/agent 默认 sample_cases.jsonl，memory 默认 memory_scenarios.jsonl）")
    parser.add_argument("--root", default="data/memory", help="search 模式检索根目录")
    parser.add_argument("--limit", type=int, default=5, help="search 模式 Top-K 数量")
    parser.add_argument("--config", default=None, help="tools.json 路径（agent/memory 模式使用）")
    parser.add_argument("--output", default=None, help="保存完整 JSON 评测报告的路径")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir).resolve()
    default_cases = "evals/memory_scenarios.jsonl" if args.mode == "memory" else "evals/sample_cases.jsonl"
    cases_path = Path(args.cases or default_cases).resolve()
    cfg_path = Path(args.config) if args.config else None

    if args.mode == "search":
        cases = load_cases(cases_path)
        memory = MemoryStore(data_dir, create_dirs=False)
        report = evaluate_search(memory, cases, root=args.root, limit=args.limit)
    elif args.mode == "agent":
        cases = load_cases(cases_path)
        report = evaluate_agent(from_env(), data_dir, cases, config_path=cfg_path)
    else:
        scenarios = load_scenarios(cases_path)
        report = evaluate_memory(from_env(), scenarios, config_path=cfg_path)

    if args.output:
        safe_write_report(Path(args.output), data_dir, report)

    print(f"Eval mode={report['mode']}  git={report['git_commit']}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if report["failures"]:
        print("\nFailed cases:")
        for item in report["failures"]:
            print(f"  - {item['id']}: {item['reason']}")
    return 0
