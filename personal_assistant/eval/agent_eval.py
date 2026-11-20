"""端到端 Agent 评测：在临时隔离副本上运行真实 ReAct 循环，记录工具链与回答供人工复核。"""
from __future__ import annotations

import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter

from personal_assistant.agent.react import MAX_TOOL_ROUNDS, run_react
from personal_assistant.agent.tools import ToolRegistry
from personal_assistant.ai.cache import CacheMonitor
from personal_assistant.ai.client import LLMClient
from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.assistant.common import REPO_ROOT
from personal_assistant.assistant.memory import FilesPolicy, MemoryStore, memory_tools
from personal_assistant.assistant.prompts import format_system_prompt
from personal_assistant.eval.common import current_git_commit, normalize_expected_sources


# 将原始数据目录的 memory/ 与 workspace/ 复制到临时目录，确保评测过程绝不改写生产数据
def _prepare_isolated_memory(
    data_dir: Path, tmp_dir: Path, cfg: ToolsConfig | None, project_root: Path
) -> MemoryStore:
    isolated_data = tmp_dir / "data"
    isolated_data.mkdir(parents=True)
    for sub in ("memory", "workspace"):
        src = data_dir.resolve() / sub
        if src.exists():
            shutil.copytree(src, isolated_data / sub, symlinks=False)
    policy = (
        FilesPolicy.from_dict(cfg.files, project_root=project_root, data_root=isolated_data)
        if cfg is not None
        else None
    )
    return MemoryStore(isolated_data, policy=policy, create_dirs=True)


# 批量运行端到端 Agent 评测，保存每题的工具调用轨迹、最终回答与耗时
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
    monitor = CacheMonitor()
    tracked_llm = monitor.wrap(llm, session_key="agent-eval")

    with tempfile.TemporaryDirectory(prefix="my-helper-eval-") as tmp:
        memory = _prepare_isolated_memory(data_dir, Path(tmp), cfg, project_root)
        current_user: list[tuple[str, int | None]] = [("", None)]
        builtins = memory_tools(memory, get_user_message=lambda: current_user[0])
        registry = cfg.build_registry(*builtins) if cfg is not None else ToolRegistry(*builtins)
        # 评测批次内冻结系统提示词，确保跨题前缀一致且与真实会话行为对齐
        frozen_system_prompt = format_system_prompt(memory)

        for idx, case in enumerate(cases, start=1):
            case_id = str(case["id"])
            query = str(case["query"])
            expected = normalize_expected_sources(memory, case.get("expected_sources"))
            expected_tools = [str(t) for t in (case.get("expected_tools") or [])]

            current_user[0] = (query, idx)
            messages: list[dict[str, object]] = [
                {"role": "system", "content": frozen_system_prompt},
                {"role": "user", "content": query},
            ]
            steps: list[dict[str, str]] = []

            # 记录单题执行过程中的每一次工具调用与前 300 字符观察结果
            def on_step(name: str, args: str, obs: str) -> None:
                steps.append(
                    {
                        "tool": name,
                        "arguments": args,
                        "observation_preview": obs[:300],
                    }
                )

            start_idx = monitor.snapshot_index()
            t0 = perf_counter()
            reply = run_react(tracked_llm, messages, registry, max_rounds=max_rounds, on_step=on_step)
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
                    "cache": monitor.summary(since=start_idx),
                    "review_status": "needs_human_review",
                }
            )

    summary = {
        "total_cases": len(cases),
        "expected_tools_all_hit": sum(1 for item in per_case if item["expected_tools_hit"]),
        "tool_check_failures": len(failures),
        "mean_tool_calls": round(mean([int(item["tool_call_count"]) for item in per_case]), 2),
        "mean_latency_ms": round(mean(latencies), 3) if latencies else 0.0,
        "cache": monitor.summary(),
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
