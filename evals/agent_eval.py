"""端到端 Agent 评测：在逐题隔离临时副本上运行真实 ReAct 循环，记录工具链与回答供人工复核。"""
from __future__ import annotations

import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter

from personal_assistant.agent.react import AgentEvent, ToolStep, run_react
from personal_assistant.agent.tools import ToolRegistry
from personal_assistant.ai.cache import CacheMonitor
from personal_assistant.ai.client import LLMClient
from personal_assistant.assistant.builtin_tools import MemoryToolsetProvider
from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.assistant.files_policy import FilesPolicy
from personal_assistant.assistant.memory import MemoryStore
from personal_assistant.assistant.prompts import format_system_prompt
from evals.common import current_git_commit, normalize_expected_sources, resolve_eval_paths


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
    retriever = cfg.build_retriever() if cfg is not None else None
    return MemoryStore(isolated_data, policy=policy, create_dirs=True, retriever=retriever)


# 批量运行端到端 Agent 评测（逐题独立临时目录、独立注册表与独立 KV 会话键），保存工具调用轨迹与回答
def evaluate_agent(
    llm: LLMClient,
    data_dir: Path,
    cases: list[dict[str, object]],
    *,
    config_path: Path | None = None,
    git_commit: str | None = None,
) -> dict[str, object]:
    cfg_path, project_root = resolve_eval_paths(config_path)
    cfg = ToolsConfig.load(cfg_path, project_root=project_root) if cfg_path.is_file() else None

    per_case: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    latencies: list[float] = []
    monitor = CacheMonitor()

    for idx, case in enumerate(cases, start=1):
        case_id = str(case["id"])
        query = str(case["query"])
        expected_tools = [str(t) for t in (case.get("expected_tools") or [])]
        safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", case_id) or f"case_{idx}"

        with tempfile.TemporaryDirectory(prefix=f"my-helper-eval-{safe_id}-") as tmp:
            memory = _prepare_isolated_memory(data_dir, Path(tmp), cfg, project_root)
            expected = normalize_expected_sources(memory, case.get("expected_sources"))
            provider = MemoryToolsetProvider(memory)
            if cfg is not None:
                registry = cfg.build_registry(
                    providers=[provider],
                    data_root=memory.data_root,
                )
            else:
                registry = ToolRegistry()
                registry.register_provider(provider)
            system_prompt = format_system_prompt(memory, tool_names=registry.names)
            tracked_llm = monitor.wrap(llm, session_key=f"agent-eval-{case_id}")

            messages: list[dict[str, object]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]
            steps: list[dict[str, object]] = []

            # 记录单题执行过程中的每一次工具调用、耗时与前 300 字符观察结果
            def on_event(event: AgentEvent) -> None:
                if isinstance(event, ToolStep):
                    steps.append(
                        {
                            "tool": event.tool,
                            "arguments": event.arguments,
                            "observation_preview": event.observation[:300],
                            "duration_ms": event.duration_ms,
                        }
                    )

            start_idx = monitor.snapshot_index()
            t0 = perf_counter()
            reply = run_react(
                tracked_llm,
                messages,
                registry,
                on_event=on_event,
                tool_context={"user_text": query, "turn_tag": f"eval-{idx}", "before_msg_id": idx},
            )
            elapsed_ms = round((perf_counter() - t0) * 1000, 3)

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
                "cache": monitor.summary(since=start_idx),
                "review_status": "needs_human_review",
            }
        )

    summary = {
        "total_cases": len(cases),
        "expected_tools_all_hit": sum(1 for item in per_case if item["expected_tools_hit"]),
        "tool_check_failures": len(failures),
        "mean_tool_calls": round(mean([int(item["tool_call_count"]) for item in per_case]), 2) if per_case else 0.0,
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
