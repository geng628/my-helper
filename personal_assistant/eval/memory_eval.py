"""多轮记忆与生命周期端到端评测：验证跨会话沉淀回忆、冲突替换、临时事实过期、防幻觉与 KV 缓存。"""
from __future__ import annotations

import json
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter

from personal_assistant.app import AssistantApp
from personal_assistant.cache_metrics import CallCacheRecord, summarize_records
from personal_assistant.cli import REPO_ROOT
from personal_assistant.consolidator import ConsolidationPolicy, MemoryConsolidator
from personal_assistant.eval.common import current_git_commit
from personal_assistant.llm import LLMClient, NoopLLMClient


# 从 JSONL 文件读取多轮记忆评测场景，校验场景 id 唯一且包含非空 steps
def load_scenarios(path: Path) -> list[dict[str, object]]:
    scenarios: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw_line.strip()
        if not text:
            continue
        item = json.loads(text)
        if not isinstance(item, dict):
            raise ValueError(f"{path.name}:{line_no} 必须是 JSON 对象。")
        sc_id = str(item.get("id") or f"scenario-{line_no}").strip()
        if sc_id in seen_ids:
            raise ValueError(f"重复的场景 id：{sc_id}")
        seen_ids.add(sc_id)
        steps = item.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"场景 {sc_id} 必须包含非空 steps 数组。")
        item["id"] = sc_id
        scenarios.append(item)
    if not scenarios:
        raise ValueError("场景文件为空。")
    return scenarios


# 安全解析可选的 ISO 日期字符串（YYYY-MM-DD）
def _parse_optional_date(raw: object) -> date | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    return date.fromisoformat(raw.strip())


# 根据场景配置构造 ConsolidationPolicy
def _build_policy(raw: object) -> ConsolidationPolicy:
    if not isinstance(raw, dict):
        return ConsolidationPolicy()
    return ConsolidationPolicy(
        max_active_messages=int(raw.get("max_active_messages", 16)),
        max_active_chars=int(raw.get("max_active_chars", 8000)),
        keep_recent_messages=int(raw.get("keep_recent_messages", 6)),
        min_batch_messages=int(raw.get("min_batch_messages", 4)),
    )


# 校验单步 ask 的回复内容、工具调用、磁盘记忆文件与缓存命中阈值
def _check_ask_assertions(
    app: AssistantApp,
    step: dict[str, object],
    result: dict[str, object],
) -> list[str]:
    reasons: list[str] = []
    reply = str(result.get("reply") or "")
    reply_lower = reply.lower()
    called_tools = [str(item.get("tool") or "") for item in (result.get("tool_calls") or [])]

    expect_tools_any = step.get("expect_tools_any")
    if isinstance(expect_tools_any, list) and expect_tools_any:
        if not any(str(t) in called_tools for t in expect_tools_any):
            reasons.append(f"expect_tools_any 未命中：期望其一 {expect_tools_any}，实际 {called_tools}")

    for kw in step.get("must_include") or []:
        token = str(kw)
        if token.lower() not in reply_lower:
            reasons.append(f"must_include 缺失：{token!r}")

    must_include_any = step.get("must_include_any")
    if isinstance(must_include_any, list) and must_include_any:
        if not any(str(kw).lower() in reply_lower for kw in must_include_any):
            reasons.append(f"must_include_any 未命中任何词：{must_include_any}")

    for bad in step.get("must_not_include") or []:
        token = str(bad)
        if token.lower() in reply_lower:
            reasons.append(f"must_not_include 违规出现：{token!r}")

    for fc in step.get("file_checks") or []:
        if not isinstance(fc, dict):
            continue
        rel_path = str(fc.get("path") or "")
        try:
            target = app.memory.resolve_path(rel_path, write=False)
            content = target.read_text(encoding="utf-8") if target.is_file() else ""
        except ValueError as error:
            reasons.append(f"file_checks 路径非法 {rel_path}: {error}")
            continue
        for kw in fc.get("must_include") or []:
            if str(kw).lower() not in content.lower():
                reasons.append(f"file_checks({rel_path}) 缺少 {kw!r}")
        for bad in fc.get("must_not_include") or []:
            if str(bad).lower() in content.lower():
                reasons.append(f"file_checks({rel_path}) 残留禁用内容 {bad!r}")

    cache = app.cache_monitor.summary()
    min_prefix = step.get("min_prefix_reuse_rate")
    if isinstance(min_prefix, (int, float)):
        actual_prefix = cache.get("mean_prefix_reuse_rate")
        if not isinstance(actual_prefix, (int, float)) or actual_prefix < float(min_prefix):
            reasons.append(f"min_prefix_reuse_rate 不达标：期望 >= {min_prefix}，实际 {actual_prefix}")

    min_server = step.get("min_server_cache_hit_rate")
    if isinstance(min_server, (int, float)) and int(cache.get("prompt_tokens") or 0) > 0:
        actual_server = cache.get("server_cache_hit_rate")
        if not isinstance(actual_server, (int, float)) or actual_server < float(min_server):
            reasons.append(f"min_server_cache_hit_rate 不达标：期望 >= {min_server}，实际 {actual_server}")

    return reasons


# 在完全隔离的临时目录中执行单个端到端记忆场景，返回步骤明细、失败原因与缓存统计
def _run_single_scenario(
    llm: LLMClient,
    scenario: dict[str, object],
    cfg_path: Path | None,
    project_root: Path,
) -> tuple[dict[str, object], list[CallCacheRecord]]:
    sc_id = str(scenario["id"])
    policy = _build_policy(scenario.get("policy"))
    steps = list(scenario.get("steps") or [])
    has_explicit_consolidate = any(
        isinstance(s, dict) and s.get("action") == "consolidate" for s in steps
    )

    t0 = perf_counter()
    step_logs: list[dict[str, object]] = []
    reasons: list[str] = []

    with tempfile.TemporaryDirectory(prefix=f"my-helper-memeval-{sc_id}-") as tmp:
        data_root = Path(tmp) / "data"
        app = AssistantApp(
            data_root=data_root,
            config_path=cfg_path if (cfg_path and cfg_path.is_file()) else None,
            project_root=project_root,
            llm=llm,
            consolidation_policy=policy,
            # 含显式 consolidate 步骤时关闭自动后台线程，保证步骤顺序确定不竞争
            consolidator_llm=NoopLLMClient() if has_explicit_consolidate else llm,
        )
        try:
            # 注入场景预设的初始记忆
            for seed in scenario.get("seed_memories") or []:
                if not isinstance(seed, dict):
                    continue
                app.memory.apply_memory_operations(
                    str(seed.get("topic") or "facts"),
                    str(seed.get("title") or "预置记忆"),
                    str(seed.get("index_summary") or "预置主题记忆"),
                    list(seed.get("operations") or []),
                    today=_parse_optional_date(seed.get("today")),
                )

            for idx, raw_step in enumerate(steps, start=1):
                if not isinstance(raw_step, dict):
                    continue
                action = str(raw_step.get("action") or "").strip().lower()
                chat_id = str(raw_step.get("chat_id") or "default")

                if action == "chat":
                    res = app.chat(str(raw_step.get("message") or ""), chat_id=chat_id)
                    step_logs.append(
                        {
                            "step": idx,
                            "action": "chat",
                            "chat_id": chat_id,
                            "tool_calls": [t["tool"] for t in res["tool_calls"]],
                            "reply": res["reply"],
                        }
                    )
                elif action == "consolidate":
                    ref_day = _parse_optional_date(raw_step.get("today"))
                    tracked = app.cache_monitor.wrap(llm, session_key=f"consolidate-{chat_id}")
                    runner = MemoryConsolidator(app.state, app.memory, tracked, policy=policy)
                    c_res = runner.consolidate_once(chat_id, today=ref_day)
                    app.reset_session_prompt(chat_id)
                    if c_res is None:
                        reasons.append(f"step#{idx} consolidate 未触发（请检查消息数与 policy 阈值）")
                    step_logs.append(
                        {
                            "step": idx,
                            "action": "consolidate",
                            "chat_id": chat_id,
                            "result": c_res.to_dict() if c_res else None,
                        }
                    )
                elif action == "wait_consolidation":
                    app.wait_for_consolidation(timeout=30.0)
                    step_logs.append({"step": idx, "action": "wait_consolidation"})
                elif action == "archive":
                    ref_day = _parse_optional_date(raw_step.get("today"))
                    archived = app.memory.archive_expired_memories(today=ref_day)
                    app.reset_session_prompt()
                    step_logs.append({"step": idx, "action": "archive", "archived": archived})
                elif action == "check_search":
                    ref_day = _parse_optional_date(raw_step.get("today"))
                    root = str(raw_step.get("root") or "data/memory")
                    query = str(raw_step.get("query") or "")
                    hits = app.memory.search_hits(query, root=root, today=ref_day)
                    expect_empty = bool(raw_step.get("expect_empty", False))
                    if expect_empty and hits:
                        reasons.append(f"step#{idx} check_search({query!r}) 期望为空，实际命中 {len(hits)} 条")
                    must_hit = raw_step.get("must_include")
                    if isinstance(must_hit, str) and must_hit:
                        joined = "\n".join(h.snippet for h in hits)
                        if must_hit.lower() not in joined.lower():
                            reasons.append(f"step#{idx} check_search({query!r}) 未包含 {must_hit!r}")
                    step_logs.append(
                        {
                            "step": idx,
                            "action": "check_search",
                            "query": query,
                            "hit_count": len(hits),
                        }
                    )
                elif action == "ask":
                    res = app.chat(str(raw_step.get("message") or ""), chat_id=chat_id)
                    step_reasons = _check_ask_assertions(app, raw_step, res)
                    reasons.extend(f"step#{idx} {r}" for r in step_reasons)
                    step_logs.append(
                        {
                            "step": idx,
                            "action": "ask",
                            "chat_id": chat_id,
                            "tool_calls": [t["tool"] for t in res["tool_calls"]],
                            "reply": res["reply"],
                            "cache": res["cache"],
                            "check_failures": step_reasons,
                        }
                    )
                else:
                    reasons.append(f"step#{idx} 未知 action：{action}")

            records = list(app.cache_monitor.records)
            cache_summary = app.cache_monitor.summary()
        finally:
            app.close()

    elapsed_ms = round((perf_counter() - t0) * 1000, 3)
    record = {
        "id": sc_id,
        "description": str(scenario.get("description") or ""),
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "latency_ms": elapsed_ms,
        "cache": cache_summary,
        "steps": step_logs,
    }
    return record, records


# 批量运行端到端记忆与生命周期评测，汇总通过率、失败明细与 KV 缓存统计
def evaluate_memory(
    llm: LLMClient,
    scenarios: list[dict[str, object]],
    *,
    config_path: Path | None = None,
    git_commit: str | None = None,
) -> dict[str, object]:
    cfg_path = (config_path or (REPO_ROOT / "config" / "tools.json")).resolve()
    project_root = cfg_path.parent.parent if cfg_path.is_file() else REPO_ROOT

    per_scenario: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    all_cache_records: list[CallCacheRecord] = []
    latencies: list[float] = []

    for scenario in scenarios:
        record, cache_records = _run_single_scenario(llm, scenario, cfg_path, project_root)
        per_scenario.append(record)
        all_cache_records.extend(cache_records)
        latencies.append(float(record["latency_ms"]))
        if not record["passed"]:
            failures.append(
                {
                    "id": record["id"],
                    "reason": "; ".join(record["reasons"]),
                    "reasons": record["reasons"],
                }
            )

    passed = sum(1 for item in per_scenario if item["passed"])
    total = len(per_scenario)
    summary = {
        "total_scenarios": total,
        "passed_scenarios": passed,
        "failed_scenarios": len(failures),
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "mean_latency_ms": round(mean(latencies), 3) if latencies else 0.0,
        "cache": summarize_records(all_cache_records),
    }
    return {
        "mode": "memory",
        "git_commit": git_commit or current_git_commit(),
        "timestamp": datetime.now(UTC).isoformat(),
        "summary": summary,
        "failures": failures,
        "scenarios": per_scenario,
    }
