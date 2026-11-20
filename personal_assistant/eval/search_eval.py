"""离线 Markdown 检索评测（零模型调用）：评估来源召回率、MRR、引文覆盖、分类指标与负例干净度。"""
from __future__ import annotations

from datetime import UTC, date, datetime
from statistics import mean
from time import perf_counter

from personal_assistant.eval.common import current_git_commit, normalize_expected_sources
from personal_assistant.memory import MemoryStore


# 安全解析可选的 ISO 日期字符串（YYYY-MM-DD），供单题时间过期过滤使用
def _parse_optional_date(raw: object) -> date | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    return date.fromisoformat(raw.strip())


# 评测单道检索题（区分可回答正例与无命中负例），返回明细记录与可选失败原因
def _score_search_case(
    memory: MemoryStore,
    case: dict[str, object],
    *,
    root: str,
    limit: int,
) -> tuple[dict[str, object], dict[str, object] | None]:
    case_id = str(case["id"])
    query = str(case["query"])
    category = str(case.get("category") or "uncategorized").strip() or "uncategorized"
    ref_day = _parse_optional_date(case.get("today"))
    expected = normalize_expected_sources(memory, case.get("expected_sources"))
    answerable = bool(case.get("answerable", bool(expected)))
    if answerable and not expected:
        raise ValueError(f"可回答案例 {case_id} 必须标注 expected_sources。")

    t0 = perf_counter()
    hits = memory.search_hits(query, root=root, limit=limit, today=ref_day)
    elapsed_ms = round((perf_counter() - t0) * 1000, 3)

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
        record: dict[str, object] = {
            "id": case_id,
            "category": category,
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
        failure = None if passed else {
            "id": case_id,
            "category": category,
            "query": query,
            "reason": f"recall={recall}, quote_coverage={quote_cov}",
            "returned_paths": returned_paths,
            "expected_paths": gold_paths,
        }
        return record, failure

    passed = len(hits) == 0
    record = {
        "id": case_id,
        "category": category,
        "query": query,
        "answerable": False,
        "passed": passed,
        "hit_count": len(hits),
        "returned_paths": returned_paths,
        "latency_ms": elapsed_ms,
    }
    failure = None if passed else {
        "id": case_id,
        "category": category,
        "query": query,
        "reason": f"negative case returned {len(hits)} candidate hit(s)",
        "returned_paths": returned_paths,
    }
    return record, failure


# 汇总一组检索明细记录的核心指标（同时服务于总体 summary 与按 category 分类统计）
def _aggregate_search_records(records: list[dict[str, object]]) -> dict[str, object]:
    recalls = [float(item["source_recall"]) for item in records if item["answerable"]]
    mrrs = [float(item["mrr"]) for item in records if item["answerable"]]
    quote_covs = [float(item["quote_coverage"]) for item in records if item["answerable"]]
    latencies = [float(item["latency_ms"]) for item in records]
    neg_items = [item for item in records if not item["answerable"]]
    neg_clean = sum(1 for item in neg_items if item["passed"])
    passed = sum(1 for item in records if item["passed"])
    total = len(records)
    return {
        "total_cases": total,
        "answerable_cases": len(recalls),
        "negative_cases": len(neg_items),
        "passed_cases": passed,
        "failed_cases": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "mean_source_recall": round(mean(recalls), 4) if recalls else None,
        "mean_mrr": round(mean(mrrs), 4) if mrrs else None,
        "mean_quote_coverage": round(mean(quote_covs), 4) if quote_covs else None,
        "negative_clean_rate": round(neg_clean / len(neg_items), 4) if neg_items else None,
        "mean_latency_ms": round(mean(latencies), 3) if latencies else 0.0,
    }


# 批量运行离线 Markdown 检索评测，汇总总体与分类型指标，保留全部逐例结果与失败列表
def evaluate_search(
    memory: MemoryStore,
    cases: list[dict[str, object]],
    *,
    root: str = "data/memory",
    limit: int = 5,
    git_commit: str | None = None,
) -> dict[str, object]:
    if limit < 1:
        raise ValueError("limit 必须 >= 1。")

    per_case: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = {}

    for case in cases:
        record, failure = _score_search_case(memory, case, root=root, limit=limit)
        per_case.append(record)
        grouped.setdefault(str(record["category"]), []).append(record)
        if failure is not None:
            failures.append(failure)

    summary = _aggregate_search_records(per_case)
    summary["by_category"] = {
        cat: _aggregate_search_records(items) for cat, items in sorted(grouped.items())
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
