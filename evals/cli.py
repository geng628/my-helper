"""评测命令行入口：按 --mode 参数统一调度检索评测、单轮 Agent 评测、多轮记忆评测与公开数据集下载。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from personal_assistant.ai.client import from_env
from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.assistant.common import REPO_ROOT, load_dotenv
from personal_assistant.assistant.memory import MemoryStore
from evals.agent_eval import evaluate_agent
from evals.common import load_cases, safe_write_report
from evals.memory_eval import evaluate_memory, load_scenarios
from evals.public_benchmarks import prepare_public_datasets
from evals.search_eval import evaluate_search


# 按可选 category 与 max_cases 对案例/场景列表做筛选与截断
def _filter_items(
    items: list[dict[str, object]],
    *,
    category: str | None = None,
    max_cases: int | None = None,
) -> list[dict[str, object]]:
    filtered = items
    if category:
        target = category.strip().lower()
        filtered = [
            item
            for item in filtered
            if str(item.get("category") or "uncategorized").strip().lower() == target
        ]
    if max_cases is not None:
        if max_cases < 1:
            raise ValueError("--max-cases 必须 >= 1。")
        filtered = filtered[:max_cases]
    if not filtered:
        raise ValueError("过滤后的评测案例列表为空，请检查 --category 或案例文件。")
    return filtered


# 解析命令行参数、执行对应模式评测或公开数据集下载，输出汇总并按需保存 JSON 报告
def main(argv: list[str] | None = None) -> int:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="My Helper 评测运行器（search 离线检索 / agent 单轮 / memory 多轮记忆 / download 下载公开基准）"
    )
    parser.add_argument("--mode", choices=["search", "agent", "memory", "download"], default="search")
    parser.add_argument("--data-dir", default="evals/sample_data", help="评测数据目录（含 memory/、workspace/）")
    parser.add_argument(
        "--cases",
        default=None,
        help="JSONL 评测集路径（search/agent 默认 sample_cases.jsonl，memory 默认 memory_scenarios.jsonl）",
    )
    parser.add_argument("--root", default="data/memory", help="search 模式检索根目录")
    parser.add_argument("--limit", type=int, default=5, help="search 模式 Top-K 数量")
    parser.add_argument("--retriever", default="auto", help="search 模式检索后端：auto（按 tools.json）/ linear / 插件名如 fts5_bm25")
    parser.add_argument("--category", default=None, help="仅运行指定 category 分类的案例/场景")
    parser.add_argument("--max-cases", type=int, default=None, help="最多运行的案例/场景条数（便于抽样快测）")
    parser.add_argument("--no-judge", action="store_true", help="memory 模式关闭 LLM-as-a-Judge，改用离线词面兜底评分")
    parser.add_argument("--config", default=None, help="tools.json 路径（agent/memory 模式使用）")
    parser.add_argument("--output", default=None, help="保存完整 JSON 评测报告的路径")
    # 以下参数专用于 --mode download 下载并转换公开学术基准（默认落盘到 evals/public）
    parser.add_argument("--dataset", choices=["longmemeval", "locomo", "all"], default="all", help="download 模式目标公开集")
    parser.add_argument("--out-dir", default="evals/public", help="download 模式输出目录（默认 evals/public，已由 .gitignore 忽略）")
    parser.add_argument("--longmemeval-limit", type=int, default=35, help="LongMemEval 分层抽样题数（0 表示全量 500 题）")
    parser.add_argument("--locomo-search-limit", type=int, default=200, help="LoCoMo 离线检索抽样题数（0 表示全量）")
    parser.add_argument("--locomo-memory-limit", type=int, default=25, help="LoCoMo 多轮记忆抽样题数（0 表示全量）")
    parser.add_argument("--locomo-max-sessions", type=int, default=0, help="LoCoMo 每段对话最多保留的会话数（0 表示保留完整全部会话）")
    parser.add_argument("--locomo-max-convs", type=int, default=0, help="LoCoMo 多轮记忆最多导出的对话段数（0 表示全部对话）")
    parser.add_argument("--seed", type=int, default=42, help="download 模式分层抽样随机种子")
    parser.add_argument("--force", action="store_true", help="download 模式强制重新下载原始 JSON")
    args = parser.parse_args(argv)

    if args.mode == "download":
        summary = prepare_public_datasets(
            Path(args.out_dir),
            dataset=args.dataset,
            longmemeval_limit=args.longmemeval_limit,
            locomo_search_limit=args.locomo_search_limit,
            locomo_memory_limit=args.locomo_memory_limit,
            locomo_max_sessions=args.locomo_max_sessions,
            locomo_max_convs=args.locomo_max_convs,
            seed=args.seed,
            force=args.force,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    data_dir = Path(args.data_dir).resolve()
    default_cases = "evals/memory_scenarios.jsonl" if args.mode == "memory" else "evals/sample_cases.jsonl"
    cases_path = Path(args.cases or default_cases).resolve()
    cfg_path = Path(args.config) if args.config else None

    if args.mode == "search":
        cases = _filter_items(load_cases(cases_path), category=args.category, max_cases=args.max_cases)
        retriever = None
        if args.retriever != "linear":
            tools_cfg_path = (cfg_path or (REPO_ROOT / "config" / "tools.json")).resolve()
            if tools_cfg_path.is_file():
                proj = tools_cfg_path.parent.parent
                loaded_cfg = ToolsConfig.load(tools_cfg_path, project_root=proj)
                override = None if args.retriever == "auto" else args.retriever
                retriever = loaded_cfg.build_retriever(override_name=override)
            elif args.retriever != "auto":
                raise ValueError(f"无法加载检索插件 {args.retriever!r}：未找到配置文件 {tools_cfg_path}")
        memory = MemoryStore(data_dir, create_dirs=False, retriever=retriever)
        report = evaluate_search(memory, cases, root=args.root, limit=args.limit)
    elif args.mode == "agent":
        cases = _filter_items(load_cases(cases_path), category=args.category, max_cases=args.max_cases)
        report = evaluate_agent(from_env(), data_dir, cases, config_path=cfg_path)
    else:
        scenarios = _filter_items(load_scenarios(cases_path), category=args.category, max_cases=args.max_cases)
        llm = from_env()
        judge = None if args.no_judge else llm

        # 向 stderr 输出共享长对话沉淀完成进度
        def _report_setup_done(setup_rep: dict[str, object]) -> None:
            sec = float(setup_rep.get("latency_ms") or 0.0) / 1000.0
            gid = setup_rep.get("shared_group")
            steps_n = setup_rep.get("steps_executed")
            print(f"[setup] {gid} ready ({steps_n} steps, {sec:.1f}s)", file=sys.stderr, flush=True)

        # 向 stderr 输出单题完成进度，避免长跑大样本评测时终端无反馈且不污染 stdout 汇总
        def _report_progress(done: int, total: int, rec: dict[str, object]) -> None:
            status = "PASS" if rec.get("passed") else "FAIL"
            sec = float(rec.get("latency_ms") or 0.0) / 1000.0
            print(f"[{done}/{total}] {status} {rec.get('id')} ({sec:.1f}s)", file=sys.stderr, flush=True)

        report = evaluate_memory(
            llm,
            scenarios,
            config_path=cfg_path,
            judge_llm=judge,
            on_progress=_report_progress,
            on_setup_done=_report_setup_done,
        )

    if args.output:
        safe_write_report(Path(args.output), data_dir, report)

    extra = f"  retriever={report['retriever']}" if "retriever" in report else ""
    print(f"Eval mode={report['mode']}{extra}  git={report['git_commit']}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if report["failures"]:
        print("\nFailed cases:")
        for item in report["failures"]:
            cat = f"[{item['category']}] " if item.get("category") else ""
            print(f"  - {cat}{item['id']}: {item['reason']}")
    return 0
