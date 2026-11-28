"""评测命令行入口：按 --mode 参数统一调度检索评测、单轮 Agent 评测、多轮记忆评测与公开数据集下载。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from personal_assistant.ai import ModelInvoker, from_env
from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.assistant.common import load_dotenv
from personal_assistant.assistant.memory import MemoryStore
from evals.agent_eval import evaluate_agent
from evals.common import load_cases, resolve_eval_paths, safe_write_report
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


# 按 --config 或默认 config/tools.json 解析路径并加载工具与主板插件
def _load_tools_config(cfg_path: Path | None) -> tuple[Path, ToolsConfig | None]:
    tools_cfg_path, project_root = resolve_eval_paths(cfg_path)
    if not tools_cfg_path.is_file():
        return tools_cfg_path, None
    return tools_cfg_path, ToolsConfig.load(tools_cfg_path, project_root=project_root)


# 统一通过 ToolsConfig.build_llm 构造评测模型（支持 models.dsl 档案、内置预设与环境变量）
def _build_eval_llm(
    loaded_cfg: ToolsConfig | None,
    tools_cfg_path: Path,
    model_name: str | None = None,
) -> ModelInvoker:
    clean_name = (model_name or "").strip() or None
    if loaded_cfg is not None:
        return loaded_cfg.build_llm(tools_cfg_path.parent, model_name=clean_name)
    if clean_name is None:
        return from_env()
    env = dict(os.environ)
    env["LLM_MODEL"] = clean_name
    return from_env(env)


# 解析命令行参数、执行对应模式评测或公开数据集下载，输出汇总并按需保存 JSON 报告
def main(argv: list[str] | None = None) -> int:
    load_dotenv()
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
    parser.add_argument("--retriever", default="auto", help="search 模式检索后端：auto（按 tools.json）/ linear / 插件名如 rag")
    parser.add_argument("--embedder", default=None, help="search 模式可选向量嵌入模型（models.dsl 档案名或环境变量；省略时默认零网络纯词面评测）")
    parser.add_argument("--category", default=None, help="仅运行指定 category 分类的案例/场景")
    parser.add_argument("--max-cases", type=int, default=None, help="最多运行的案例/场景条数（便于抽样快测）")
    parser.add_argument("--no-judge", action="store_true", help="memory 模式关闭 LLM-as-a-Judge，改用离线词面兜底评分")
    parser.add_argument("--model", default=None, help="agent/memory 模式指定待测模型（models.dsl 档案名、预设名如 reasoner 或模型 ID）")
    parser.add_argument("--consolidator-model", default=None, help="memory 模式指定后台沉淀专用模型（缺省按 models.dsl 的 consolidator 或复用待测模型）")
    parser.add_argument("--judge-model", default=None, help="memory 模式指定独立 Judge 模型（同上；未指定时默认复用待测模型）")
    parser.add_argument("--workers", type=int, default=1, help="memory 模式并发场景数（默认 1 串行；>1 并发执行各独立场景）")
    parser.add_argument("--config", default=None, help="tools.json 路径（默认 config/tools.json）")
    parser.add_argument("--output", default=None, help="保存完整 JSON 评测报告的路径")
    parser.add_argument("--fail-under", type=float, default=None, help="search/memory 模式最低通过率阈值（0.0~1.0），低于则返回退出码 1")
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
    if args.workers < 1:
        raise ValueError("--workers 必须 >= 1。")
    if args.fail_under is not None and not (0.0 <= args.fail_under <= 1.0):
        raise ValueError("--fail-under 必须在 0.0 到 1.0 之间。")

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

    loaded_cfg: ToolsConfig | None = None
    try:
        if args.mode == "search":
            cases = _filter_items(load_cases(cases_path), category=args.category, max_cases=args.max_cases)
            retriever = None
            if args.retriever != "linear":
                tools_cfg_path, loaded_cfg = _load_tools_config(cfg_path)
                if loaded_cfg is not None:
                    override = None if args.retriever == "auto" else args.retriever
                    retriever = loaded_cfg.build_retriever(override_name=override)
                    if args.embedder and callable(set_emb := getattr(retriever, "set_embedder", None)):
                        embedder = loaded_cfg.build_embedder(tools_cfg_path.parent, model_name=args.embedder)
                        if embedder is None:
                            raise ValueError(f"无法加载向量嵌入模型：{args.embedder!r}")
                        set_emb(embedder)
                elif args.retriever != "auto":
                    raise ValueError(f"无法加载检索插件 {args.retriever!r}：未找到配置文件 {tools_cfg_path}")
            memory = MemoryStore(data_dir, create_dirs=False, retriever=retriever)
            report = evaluate_search(memory, cases, root=args.root, limit=args.limit)
        elif args.mode == "agent":
            cases = _filter_items(load_cases(cases_path), category=args.category, max_cases=args.max_cases)
            tools_cfg_path, loaded_cfg = _load_tools_config(cfg_path)
            llm = _build_eval_llm(loaded_cfg, tools_cfg_path, args.model)
            report = evaluate_agent(
                llm,
                data_dir,
                cases,
                config_path=cfg_path,
                tools_config=loaded_cfg,
            )
        else:
            scenarios = _filter_items(load_scenarios(cases_path), category=args.category, max_cases=args.max_cases)
            tools_cfg_path, loaded_cfg = _load_tools_config(cfg_path)
            llm = _build_eval_llm(loaded_cfg, tools_cfg_path, args.model)
            cons_name = (args.consolidator_model or "").strip() or None
            if loaded_cfg is not None:
                consolidator_llm: ModelInvoker | None = loaded_cfg.build_consolidator_llm(
                    tools_cfg_path.parent,
                    fallback_llm=llm,
                    model_name=cons_name,
                )
            elif cons_name is not None:
                consolidator_llm = _build_eval_llm(None, tools_cfg_path, cons_name)
            else:
                consolidator_llm = None
            judge_name = (args.judge_model or "").strip() or None
            if args.no_judge:
                judge: ModelInvoker | None = None
            elif judge_name is not None:
                judge = _build_eval_llm(loaded_cfg, tools_cfg_path, judge_name)
            else:
                judge = llm

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
                tools_config=loaded_cfg,
                consolidator_llm=consolidator_llm,
                judge_llm=judge,
                workers=args.workers,
                on_progress=_report_progress,
                on_setup_done=_report_setup_done,
            )
    finally:
        if loaded_cfg is not None:
            loaded_cfg.plugin_manager.terminate_all()

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
    if args.fail_under is not None and args.mode in {"search", "memory"}:
        summary_obj = report.get("summary")
        pass_rate = summary_obj.get("pass_rate") if isinstance(summary_obj, dict) else None
        if not isinstance(pass_rate, (int, float)) or float(pass_rate) < args.fail_under:
            print(
                f"FAIL: pass_rate={pass_rate} 低于阈值 --fail-under={args.fail_under}",
                file=sys.stderr,
            )
            return 1
    return 0
