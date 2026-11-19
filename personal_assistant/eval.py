"""轻量评测运行器：支持 Markdown 离线检索评测与真实 Agent 端到端评测。"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter

from personal_assistant.cli import REPO_ROOT, format_system_prompt, load_dotenv
from personal_assistant.commands import ToolsConfig
from personal_assistant.llm import LLMClient, from_env
from personal_assistant.memory import FilesPolicy, MemoryStore, memory_tools
from personal_assistant.react import MAX_TOOL_ROUNDS, run_react


# 获取当前仓库短 Git Commit SHA（若有未提交改动则追加 -dirty），用于绑定评测基线
def current_git_commit(repo_root: Path = REPO_ROOT) -> str:
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repo_root), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout.strip()
        )
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# 从 JSONL 文件逐行加载评测案例，并校验案例 id 唯一且包含非空 query
def load_cases(cases_path: Path) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for line_no, raw_line in enumerate(cases_path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw_line.strip()
        if not text:
            continue
        item = json.loads(text)
        if not isinstance(item, dict):
            raise ValueError(f"{cases_path.name}:{line_no} 必须是 JSON 对象。")
        case_id = str(item.get("id") or f"case-{line_no}").strip()
        if case_id in seen_ids:
            raise ValueError(f"重复的案例 id：{case_id}")
        seen_ids.add(case_id)
        query = str(item.get("query") or "").strip()
        if not query:
            raise ValueError(f"案例 {case_id} 缺少非空 query。")
        item["id"] = case_id
        item["query"] = query
        cases.append(item)
    if not cases:
        raise ValueError("评测案例文件为空。")
    return cases


# 校验每条期望来源（金标）文件可读且引文确实存在于原文，并规范化路径展示格式
def _normalize_expected_sources(
    memory: MemoryStore, raw_sources: object
) -> list[dict[str, str]]:
    if not raw_sources:
        return []
    if not isinstance(raw_sources, list):
        raise ValueError("expected_sources 必须是数组。")
    normalized: list[dict[str, str]] = []
    file_cache: dict[Path, str] = {}
    for entry in raw_sources:
        if not isinstance(entry, dict):
            raise ValueError("expected_sources 项必须是对象。")
        raw_path = str(entry.get("path") or "").strip()
        quote = str(entry.get("quote") or "").strip()
        if not raw_path or not quote:
            raise ValueError("expected_sources 每项都需要非空 path 和 quote。")
        resolved = memory.resolve_path(raw_path, write=False)
        if not resolved.is_file():
            raise ValueError(f"金标来源文件不存在：{raw_path}")
        if resolved not in file_cache:
            file_cache[resolved] = resolved.read_text(encoding="utf-8")
        if quote not in file_cache[resolved]:
            raise ValueError(f"金标 quote 未在原文中找到：{raw_path}")
        normalized.append({"path": memory.display_path(resolved), "quote": quote})
    return normalized


# 离线检索评测（零模型调用）：统计来源召回率、MRR、引文覆盖率、负例表现与失败列表
def evaluate_search(
    memory: MemoryStore,
    cases: list[dict[str, object]],
    *,
    root: str = "data/knowledge",
    limit: int = 5,
    git_commit: str | None = None,
) -> dict[str, object]:
    if limit < 1:
        raise ValueError("limit 必须 >= 1。")

    per_case: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    recalls: list[float] = []
    mrrs: list[float] = []
    quote_covs: list[float] = []
    neg_total = 0
    neg_clean = 0
    latencies: list[float] = []

    for case in cases:
        case_id = str(case["id"])
        query = str(case["query"])
        expected = _normalize_expected_sources(memory, case.get("expected_sources"))
        answerable = bool(case.get("answerable", bool(expected)))
        if answerable and not expected:
            raise ValueError(f"可回答案例 {case_id} 必须标注 expected_sources。")

        t0 = perf_counter()
        hits = memory.search_hits(query, root=root, limit=limit)
        elapsed_ms = round((perf_counter() - t0) * 1000, 3)
        latencies.append(elapsed_ms)

        returned_paths = [hit.path for hit in hits]
        visible_text = "\n".join(hit.snippet for hit in hits)

        if answerable:
            gold_paths = list(dict.fromkeys(item["path"] for item in expected))
            matched_paths = [p for p in gold_paths if p in returned_paths]
            recall = round(len(matched_paths) / len(gold_paths), 4)
            rank = next((idx for idx, p in enumerate(returned_paths, start=1) if p in gold_paths), 0)
            mrr = round(1.0 / rank, 4) if rank else 0.0
            matched_quotes = [item["quote"] for item in expected if item["quote"] in visible_text]
            quote_cov = round(len(matched_quotes) / len(expected), 4)
            passed = recall == 1.0 and quote_cov == 1.0
            recalls.append(recall)
            mrrs.append(mrr)
            quote_covs.append(quote_cov)
            record: dict[str, object] = {
                "id": case_id,
                "query": query,
                "answerable": True,
                "passed": passed,
                "source_recall": recall,
                "mrr": mrr,
                "quote_coverage": quote_cov,
                "returned_paths": returned_paths,
                "expected_paths": gold_paths,
                "latency_ms": elapsed_ms,
            }
            if not passed:
                failures.append(
                    {
                        "id": case_id,
                        "query": query,
                        "reason": f"recall={recall}, quote_coverage={quote_cov}",
                        "returned_paths": returned_paths,
                        "expected_paths": gold_paths,
                    }
                )
        else:
            neg_total += 1
            passed = len(hits) == 0
            if passed:
                neg_clean += 1
            record = {
                "id": case_id,
                "query": query,
                "answerable": False,
                "passed": passed,
                "hit_count": len(hits),
                "returned_paths": returned_paths,
                "latency_ms": elapsed_ms,
            }
            if not passed:
                failures.append(
                    {
                        "id": case_id,
                        "query": query,
                        "reason": f"negative case returned {len(hits)} candidate hit(s)",
                        "returned_paths": returned_paths,
                    }
                )

        per_case.append(record)

    summary = {
        "total_cases": len(cases),
        "answerable_cases": len(recalls),
        "negative_cases": neg_total,
        "passed_cases": sum(1 for item in per_case if item["passed"]),
        "failed_cases": len(failures),
        "mean_source_recall": round(mean(recalls), 4) if recalls else None,
        "mean_mrr": round(mean(mrrs), 4) if mrrs else None,
        "mean_quote_coverage": round(mean(quote_covs), 4) if quote_covs else None,
        "negative_clean_rate": round(neg_clean / neg_total, 4) if neg_total else None,
        "mean_latency_ms": round(mean(latencies), 3) if latencies else 0.0,
    }
    return {
        "mode": "search",
        "git_commit": git_commit or current_git_commit(),
        "timestamp": datetime.now(UTC).isoformat(),
        "root": root,
        "limit": limit,
        "summary": summary,
        "failures": failures,
        "cases": per_case,
    }


# 端到端 Agent 评测：在临时隔离副本上跑完整 ReAct 循环，记录工具轨迹与最终回答供复核
def evaluate_agent(
    llm: LLMClient,
    data_dir: Path,
    cases: list[dict[str, object]],
    *,
    config_path: Path | None = None,
    max_rounds: int = MAX_TOOL_ROUNDS,
    git_commit: str | None = None,
) -> dict[str, object]:
    cfg_path = (config_path or (REPO_ROOT / "config" / "tools.json")).resolve()
    project_root = cfg_path.parent.parent if cfg_path.is_file() else REPO_ROOT
    cfg = ToolsConfig.load(cfg_path, project_root=project_root) if cfg_path.is_file() else None

    per_case: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    latencies: list[float] = []

    with tempfile.TemporaryDirectory(prefix="my-helper-eval-") as tmp:
        isolated_data = Path(tmp) / "data"
        isolated_data.mkdir(parents=True)
        for sub in ("knowledge", "memory"):
            src = data_dir.resolve() / sub
            if src.exists():
                shutil.copytree(src, isolated_data / sub, symlinks=False)

        policy = (
            FilesPolicy.from_dict(cfg.files, project_root=project_root, data_root=isolated_data)
            if cfg is not None
            else None
        )
        memory = MemoryStore(isolated_data, policy=policy, create_dirs=True)

        current_user: list[tuple[str, int | None]] = [("", None)]
        builtins = memory_tools(memory, get_user_message=lambda: current_user[0])
        registry = cfg.build_registry(*builtins) if cfg is not None else None
        if registry is None:
            from personal_assistant.tools import ToolRegistry

            registry = ToolRegistry(*builtins)

        for idx, case in enumerate(cases, start=1):
            case_id = str(case["id"])
            query = str(case["query"])
            expected = _normalize_expected_sources(memory, case.get("expected_sources"))
            expected_tools = [str(t) for t in (case.get("expected_tools") or [])]

            current_user[0] = (query, idx)
            messages: list[dict[str, object]] = [
                {"role": "system", "content": format_system_prompt(memory)},
                {"role": "user", "content": query},
            ]
            steps: list[dict[str, str]] = []

            # 记录当前题目的每一次工具调用及前 300 字符观察结果
            def on_step(name: str, args: str, obs: str) -> None:
                steps.append(
                    {
                        "tool": name,
                        "arguments": args,
                        "observation_preview": obs[:300],
                    }
                )

            t0 = perf_counter()
            reply = run_react(llm, messages, registry, max_rounds=max_rounds, on_step=on_step)
            elapsed_ms = round((perf_counter() - t0) * 1000, 3)
            latencies.append(elapsed_ms)

            called_tools = [step["tool"] for step in steps]
            tools_ok = all(req in called_tools for req in expected_tools)
            if not tools_ok:
                failures.append(
                    {
                        "id": case_id,
                        "query": query,
                        "reason": f"expected tools {expected_tools}, got {called_tools}",
                    }
                )
            per_case.append(
                {
                    "id": case_id,
                    "query": query,
                    "answerable": bool(case.get("answerable", bool(expected))),
                    "expected_sources": expected,
                    "expected_tools": expected_tools,
                    "called_tools": called_tools,
                    "expected_tools_hit": tools_ok,
                    "tool_call_count": len(steps),
                    "tool_trace": steps,
                    "reply": reply,
                    "latency_ms": elapsed_ms,
                    "review_status": "needs_human_review",
                }
            )

    summary = {
        "total_cases": len(cases),
        "expected_tools_all_hit": sum(1 for item in per_case if item["expected_tools_hit"]),
        "tool_check_failures": len(failures),
        "mean_tool_calls": round(mean([int(item["tool_call_count"]) for item in per_case]), 2),
        "mean_latency_ms": round(mean(latencies), 3) if latencies else 0.0,
        "note": "工具调用命中不代表回答事实正确，请结合 tool_trace 与 reply 逐例人工核验。",
    }
    return {
        "mode": "agent",
        "git_commit": git_commit or current_git_commit(),
        "timestamp": datetime.now(UTC).isoformat(),
        "summary": summary,
        "failures": failures,
        "cases": per_case,
    }


# 将评测报告写入指定 JSON 文件，禁止输出到 --data-dir 内部以防污染资料库
def _safe_write_report(output_path: Path, data_dir: Path, report: dict[str, object]) -> None:
    resolved_out = output_path.resolve()
    resolved_data = data_dir.resolve()
    if resolved_out.is_relative_to(resolved_data):
        raise ValueError("评测报告输出路径不能放在 --data-dir 目录内。")
    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    resolved_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# 评测命令行入口：解析 --mode/--data-dir/--cases/--output 参数并打印汇总结果
def main(argv: list[str] | None = None) -> int:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description="My Helper v2 评测运行器（检索离线评测 / Agent 端到端评测）")
    parser.add_argument("--mode", choices=["search", "agent"], default="search")
    parser.add_argument("--data-dir", default="evals/sample_data", help="资料目录（含 knowledge/、memory/）")
    parser.add_argument("--cases", default="evals/sample_cases.jsonl", help="JSONL 评测集路径")
    parser.add_argument("--root", default="data/knowledge", help="search 模式检索根目录")
    parser.add_argument("--limit", type=int, default=5, help="search 模式 Top-K 数量")
    parser.add_argument("--config", default=None, help="tools.json 路径（agent 模式使用）")
    parser.add_argument("--output", default=None, help="保存完整 JSON 评测报告的路径")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir).resolve()
    cases = load_cases(Path(args.cases).resolve())

    if args.mode == "search":
        memory = MemoryStore(data_dir, create_dirs=False)
        report = evaluate_search(memory, cases, root=args.root, limit=args.limit)
    else:
        cfg_path = Path(args.config) if args.config else None
        report = evaluate_agent(from_env(), data_dir, cases, config_path=cfg_path)

    if args.output:
        _safe_write_report(Path(args.output), data_dir, report)

    print(f"Eval mode={report['mode']}  git={report['git_commit']}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if report["failures"]:
        print("\nFailed cases:")
        for item in report["failures"]:
            print(f"  - {item['id']}: {item['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
