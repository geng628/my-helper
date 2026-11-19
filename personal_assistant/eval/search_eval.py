"""离线 Markdown 检索评测（零模型调用）：评估来源召回率、MRR、引文覆盖与负例干净度。"""
from __future__ import annotations

from datetime import UTC, datetime
from statistics import mean
from time import perf_counter

from personal_assistant.eval.common import current_git_commit, normalize_expected_sources
from personal_assistant.memory import MemoryStore


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
    expected = normalize_expected_sources(memory, case.get("expected_sources"))
    answerable = bool(case.get("answerable", bool(expected)))
    if answerable and not expected:
        raise ValueError(f"可回答案例 {case_id} 必须标注 expected_sources。")

    t0 = perf_counter()
    hits = memory.search_hits(query, root=root, limit=limit)
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
            "query": query,
            "reason": f"recall={recall}, quote_coverage={quote_cov}",
            "returned_paths": returned_paths,
            "expected_paths": gold_paths,
        }
        return record, failure

    passed = len(hits) == 0
    record = {
        "id": case_id,
        "query": query,
        "answerable": False,
        "passed": passed,
        "hit_count": len(hits),
        "returned_paths": returned_paths,
        "latency_ms": elapsed_ms,
    }
    failure = None if passed else {
        "id": case_id,
        "query": query,
        "reason": f"negative case returned {len(hits)} candidate hit(s)",
        "returned_paths": returned_paths,
    }
    return record, failure


# 批量运行离线 Markdown 检索评测，汇总指标并保留全部逐例结果与失败列表
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
    recalls: list[float] = []
    mrrs: list[float] = []
    quote_covs: list[float] = []
    latencies: list[float] = []
    neg_total = 0
    neg_clean = 0

    for case in cases:
        record, failure = _score_search_case(memory, case, root=root, limit=limit)
        per_case.append(record)
        latencies.append(float(record["latency_ms"]))
        if failure is not None:
            failures.append(failure)
        if record["answerable"]:
            recalls.append(float(record["source_recall"]))
            mrrs.append(float(record["mrr"]))
            quote_covs.append(float(record["quote_coverage"]))
        else:
            neg_total += 1
            if record["passed"]:
                neg_clean += 1

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
