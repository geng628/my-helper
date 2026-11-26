"""端到端 Agent 评测：在逐题隔离临时副本上直接复用 AssistantApp 运行单轮对话，记录工具链与回答供人工复核。"""
from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter

from evals.common import (
    current_git_commit,
    normalize_expected_sources,
    parse_optional_date,
    resolve_eval_tools_config,
)
from personal_assistant.ai.cache import CallCacheRecord, summarize_records
from personal_assistant.ai.client import LLMClient
from personal_assistant.assistant.app import AssistantApp
from personal_assistant.assistant.commands import ToolsConfig


# 将原始数据目录的 memory/ 与 workspace/ 复制到隔离临时目录，确保评测绝不改写生产数据
def _copy_isolated_data(data_dir: Path, isolated_data: Path) -> None:
    isolated_data.mkdir(parents=True, exist_ok=True)
    for sub in ("memory", "workspace"):
        src = data_dir.resolve() / sub
        if src.exists():
            shutil.copytree(src, isolated_data / sub, symlinks=False)


# 构造单题逻辑日期回调（显式绑定值拷贝，避免闭包变量漂移）
def _make_today_fn(ref_day: date) -> Callable[[], date]:
    # 返回该题绑定的固定历史日期
    def _fixed_today() -> date:
        return ref_day

    return _fixed_today


# 批量运行端到端 Agent 评测（逐题复用 AssistantApp 沙箱、逻辑日期与安全护栏），保存工具调用轨迹与回答
def evaluate_agent(
    llm: LLMClient,
    data_dir: Path,
    cases: list[dict[str, object]],
    *,
    config_path: Path | None = None,
    tools_config: ToolsConfig | None = None,
    git_commit: str | None = None,
) -> dict[str, object]:
    cfg_path, project_root, loaded_cfg, owns_cfg = resolve_eval_tools_config(
        config_path, tools_config
    )

    per_case: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    latencies: list[float] = []
    all_cache_records: list[CallCacheRecord] = []

    try:
        for idx, case in enumerate(cases, start=1):
            case_id = str(case["id"])
            query = str(case["query"])
            raw_expected_tools = case.get("expected_tools")
            expected_tools = (
                [str(t) for t in raw_expected_tools]
                if isinstance(raw_expected_tools, list)
                else []
            )
            case_day = parse_optional_date(case.get("today"))
            safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", case_id) or f"case_{idx}"

            with tempfile.TemporaryDirectory(prefix=f"my-helper-eval-{safe_id}-") as tmp:
                isolated_data = Path(tmp) / "data"
                _copy_isolated_data(data_dir, isolated_data)
                app = AssistantApp(
                    data_root=isolated_data,
                    config_path=cfg_path if loaded_cfg is not None else None,
                    project_root=project_root,
                    tools_config=loaded_cfg,
                    llm=llm,
                    auto_consolidate=False,
                    today_fn=_make_today_fn(case_day) if case_day is not None else None,
                )
                try:
                    expected = normalize_expected_sources(app.memory, case.get("expected_sources"))
                    t0 = perf_counter()
                    result = app.chat(query, chat_id=f"agent-eval-{case_id}")
                    elapsed_ms = round((perf_counter() - t0) * 1000, 3)
                    all_cache_records.extend(app.cache_monitor.records)
                finally:
                    app.close()

            raw_calls = result.get("tool_calls")
            tool_calls = raw_calls if isinstance(raw_calls, list) else []
            steps = [
                {
                    "tool": str(step.get("tool") or ""),
                    "arguments": str(step.get("arguments") or ""),
                    "observation_preview": str(step.get("observation") or "")[:300],
                    "duration_ms": step.get("duration_ms", 0.0),
                }
                for step in tool_calls
                if isinstance(step, dict)
            ]
            reply = str(result.get("reply") or "")
            latencies.append(elapsed_ms)
            called_tools = [str(step["tool"]) for step in steps]
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
                    "cache": result.get("cache") or summarize_records([]),
                    "review_status": "needs_human_review",
                }
            )
    finally:
        if owns_cfg and loaded_cfg is not None:
            loaded_cfg.plugin_manager.terminate_all()

    summary = {
        "total_cases": len(cases),
        "expected_tools_all_hit": sum(1 for item in per_case if item["expected_tools_hit"]),
        "tool_check_failures": len(failures),
        "mean_tool_calls": (
            round(mean([int(item["tool_call_count"]) for item in per_case]), 2) if per_case else 0.0
        ),
        "mean_latency_ms": round(mean(latencies), 3) if latencies else 0.0,
        "cache": summarize_records(all_cache_records),
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
