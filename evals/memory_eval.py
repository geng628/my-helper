"""多轮记忆与生命周期端到端评测：验证跨会话沉淀、冲突替换、过期归档、公开基准与 KV 缓存。"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import TypeVar, cast

from personal_assistant.ai import (
    CallCacheRecord,
    Context,
    ModelInvoker,
    NoopModelClient,
    UserMessage,
    summarize_records,
)
from personal_assistant.assistant.app import AssistantApp
from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.assistant.common import today_shanghai
from personal_assistant.assistant.consolidator import ConsolidationPolicy, _parse_json_object
from evals.common import current_git_commit, parse_optional_date, resolve_eval_tools_config

ProgressCallback = Callable[[int, int, dict[str, object]], None]
SetupDoneCallback = Callable[[dict[str, object]], None]

JUDGE_SYSTEM_PROMPT = (
    "You are a strict, impartial evaluator grading an AI assistant's answer against a reference answer.\n"
    "Output ONLY a valid JSON object: {\"correct\": true/false, \"reason\": \"one concise sentence\"}.\n"
    "Grading rules:\n"
    "1. For normal questions (is_abstention=false): mark correct=true ONLY if the assistant's response "
    "clearly states the essential factual content of the reference answer (equivalent wording, units, "
    "or date formatting are acceptable; extra harmless context is fine; wrong numbers/entities or claiming "
    "not to know is false).\n"
    "2. For abstention questions (is_abstention=true): mark correct=true ONLY if the assistant does NOT "
    "adopt the false premise for the asked-about person/entity and does NOT fabricate unrecorded details, "
    "AND clearly rejects the false premise or states that the required information is missing/not recorded/"
    "insufficient. Clarifying from memory that the event actually happened to a different person/entity "
    "(valid premise correction, e.g., stating Person A has no record of the asked activity and clarifying "
    "from memory that the activity actually involved Person B) counts as valid premise rejection "
    "(correct=true), NOT fabrication. Answering as if the false premise were true or speculating/"
    "fabricating unrecorded facts is correct=false."
)
JUDGE_PROMPT_VERSION = "v2-neutral"
JUDGE_PROMPT_SHA256 = hashlib.sha256(JUDGE_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]

ABSTENTION_MARKERS = (
    "没有",
    "未找到",
    "查不到",
    "没查到",
    "无记录",
    "未记录",
    "没有记录",
    "不知道",
    "未提及",
    "not enough",
    "no record",
    "no information",
    "don't have",
    "do not have",
    "didn't mention",
    "did not mention",
    "not mentioned",
    "cannot find",
    "couldn't find",
    "could not find",
    "unable to find",
    "not recorded",
    "insufficient",
)


# 从 JSONL 文件读取多轮记忆评测场景（支持 record_type=shared_setup 共享长记忆前缀绑定）
def load_scenarios(path: Path) -> list[dict[str, object]]:
    scenarios: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    shared_setups: dict[str, dict[str, object]] = {}
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw_line.strip()
        if not text:
            continue
        item = json.loads(text)
        if not isinstance(item, dict):
            raise ValueError(f"{path.name}:{line_no} 必须是 JSON 对象。")
        if str(item.get("record_type") or "").strip().lower() == "shared_setup":
            group_id = str(item.get("shared_group") or "").strip()
            setup_steps = item.get("steps")
            if not group_id:
                raise ValueError(f"{path.name}:{line_no} shared_setup 缺少非空 shared_group。")
            if group_id in shared_setups:
                raise ValueError(f"重复的 shared_group 定义：{group_id}")
            if not isinstance(setup_steps, list) or not setup_steps:
                raise ValueError(f"shared_setup {group_id} 必须包含非空 steps 数组。")
            shared_setups[group_id] = item
            continue

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

    # 将紧凑格式中引用 shared_group 的场景与对应 shared_setup 定义绑定，便于后续过滤与并发复用
    for item in scenarios:
        group_id = str(item.get("shared_group") or "").strip()
        if not group_id or isinstance(item.get("shared_setup"), dict):
            continue
        if group_id not in shared_setups:
            raise ValueError(f"场景 {item['id']} 引用了未定义的 shared_group：{group_id}")
        item["shared_setup"] = shared_setups[group_id]
    return scenarios


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


# 提取中英文词元，供无 Judge 时的离线词面兜底评分使用
def _normalize_tokens(text: str) -> set[str]:
    lowered = text.lower()
    tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", lowered))
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "at", "for", "and", "or"}
    return {t for t in tokens if t not in stopwords}


# 使用独立 LLM-as-a-Judge（或离线词面兜底）评判开放式问答的事实正确性与拒答表现
def judge_answer(
    judge_llm: ModelInvoker | None,
    question: str,
    expected_answer: str,
    reply: str,
    *,
    is_abstention: bool = False,
) -> dict[str, object]:
    if judge_llm is not None and not isinstance(judge_llm, NoopModelClient):
        payload = (
            f"Question: {question}\n"
            f"Reference Answer: {expected_answer}\n"
            f"is_abstention: {str(is_abstention).lower()}\n"
            f"Assistant Reply: {reply}"
        )
        msg = judge_llm.complete(
            Context(
                system_prompt=JUDGE_SYSTEM_PROMPT,
                messages=(UserMessage(content=payload),),
            )
        )
        raw_content = msg.text
        try:
            parsed = _parse_json_object(raw_content)
            correct = bool(parsed.get("correct"))
            reason = str(parsed.get("reason") or "").strip() or "LLM judge verdict"
            return {"mode": "llm_judge", "correct": correct, "reason": reason}
        except (ValueError, json.JSONDecodeError) as error:
            return {
                "mode": "llm_judge",
                "correct": False,
                "reason": f"Judge 输出解析失败：{error} ({raw_content[:120]})",
            }

    # 无 Judge 时退化为透明的词面兜底校验
    reply_lower = reply.lower()
    if is_abstention:
        matched = any(marker in reply_lower for marker in ABSTENTION_MARKERS)
        return {
            "mode": "lexical_fallback",
            "correct": matched,
            "reason": "检测到拒答表述" if matched else "未检测到明确拒答表述",
        }
    expected_clean = expected_answer.strip().lower()
    if expected_clean and expected_clean in reply_lower:
        return {"mode": "lexical_fallback", "correct": True, "reason": "参考答案原文命中"}
    exp_tokens = _normalize_tokens(expected_answer)
    rep_tokens = _normalize_tokens(reply)
    overlap = len(exp_tokens & rep_tokens) / len(exp_tokens) if exp_tokens else 0.0
    return {
        "mode": "lexical_fallback",
        "correct": overlap >= 0.6,
        "reason": f"词元覆盖率={overlap:.2f}",
    }


# 判断文本是否包含目标词条：纯 ASCII 词/数字关键字使用同类 ASCII 字符边界匹配，含中文或标点时执行规范子串匹配
def _contains_keyword(text: str, keyword: str) -> bool:
    if not keyword:
        return True
    if keyword.isascii() and keyword.isalnum():
        if keyword.isalpha():
            boundary = "[A-Za-z]"
        elif keyword.isdigit():
            boundary = "[0-9]"
        else:
            boundary = "[A-Za-z0-9]"
        return bool(re.search(rf"(?<!{boundary}){re.escape(keyword)}(?!{boundary})", text, re.IGNORECASE))
    return keyword.lower() in text.lower()


# 校验单步 ask 的回复内容、工具调用、磁盘记忆文件、缓存命中阈值与可选 Judge 评判
def _check_ask_assertions(
    app: AssistantApp,
    step: dict[str, object],
    result: dict[str, object],
    *,
    judge_llm: ModelInvoker | None = None,
) -> tuple[list[str], dict[str, object] | None]:
    reasons: list[str] = []
    reply = str(result.get("reply") or "")
    called_tools = [str(item.get("tool") or "") for item in (result.get("tool_calls") or [])]

    expect_tools_any = step.get("expect_tools_any")
    if isinstance(expect_tools_any, list) and expect_tools_any:
        if not any(str(t) in called_tools for t in expect_tools_any):
            reasons.append(f"expect_tools_any 未命中：期望其一 {expect_tools_any}，实际 {called_tools}")

    for kw in step.get("must_include") or []:
        token = str(kw)
        if not _contains_keyword(reply, token):
            reasons.append(f"must_include 缺失：{token!r}")

    must_include_any = step.get("must_include_any")
    if isinstance(must_include_any, list) and must_include_any:
        if not any(_contains_keyword(reply, str(kw)) for kw in must_include_any):
            reasons.append(f"must_include_any 未命中任何词：{must_include_any}")

    for bad in step.get("must_not_include") or []:
        token = str(bad)
        if _contains_keyword(reply, token):
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
            if not _contains_keyword(content, str(kw)):
                reasons.append(f"file_checks({rel_path}) 缺少 {kw!r}")
        fc_any = fc.get("must_include_any")
        if isinstance(fc_any, list) and fc_any:
            if not any(_contains_keyword(content, str(kw)) for kw in fc_any):
                reasons.append(f"file_checks({rel_path}) 未命中任何词：{fc_any}")
        for bad in fc.get("must_not_include") or []:
            if _contains_keyword(content, str(bad)):
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

    judge_detail: dict[str, object] | None = None
    expected_answer = str(step.get("expected_answer") or "").strip()
    if expected_answer:
        is_abstention = bool(step.get("is_abstention", False))
        judge_detail = judge_answer(
            judge_llm,
            str(step.get("message") or ""),
            expected_answer,
            reply,
            is_abstention=is_abstention,
        )
        if not judge_detail["correct"]:
            reasons.append(f"judge 判定未通过（{judge_detail['mode']}）：{judge_detail['reason']}")

    return reasons, judge_detail


# 将已完成沉淀的共享基准 data/ 目录完整复制到独立临时目录（隔离记忆、工作区与状态库）
def _copy_data_snapshot(src_data: Path, dst_data: Path) -> None:
    dst_data.mkdir(parents=True, exist_ok=True)
    for sub in ("memory", "workspace", "state"):
        src_sub = src_data / sub
        if src_sub.is_dir():
            shutil.copytree(src_sub, dst_data / sub, symlinks=False)


# 提取场景的共享前缀信息（统一仅接受 shared_setup 格式），返回 (group_id, setup_spec, qa_steps)
def _split_shared_setup(
    scenario: dict[str, object],
) -> tuple[str | None, dict[str, object] | None, list[dict[str, object]]]:
    steps = list(scenario.get("steps") or [])
    group_id = str(scenario.get("shared_group") or "").strip()
    raw_setup = scenario.get("shared_setup")
    if group_id and isinstance(raw_setup, dict):
        return group_id, raw_setup, steps
    return None, None, steps


# 在指定 AssistantApp 实例上顺序执行一组评测步骤（直接复用 app.consolidator 确保双模型生效）
def _execute_steps(
    app: AssistantApp,
    steps: list[dict[str, object]],
    clock_day: list[date | None],
    current_today: Callable[[], date],
    *,
    judge_llm: ModelInvoker | None = None,
) -> tuple[list[dict[str, object]], list[str]]:
    step_logs: list[dict[str, object]] = []
    reasons: list[str] = []

    for idx, raw_step in enumerate(steps, start=1):
        if not isinstance(raw_step, dict):
            continue
        action = str(raw_step.get("action") or "").strip().lower()
        chat_id = str(raw_step.get("chat_id") or "default")
        step_day = parse_optional_date(raw_step.get("today"))
        if step_day is not None and step_day != clock_day[0]:
            clock_day[0] = step_day
            app.reset_session_prompt(chat_id)

        if action == "seed_history":
            seeded = 0
            seed_day = step_day or clock_day[0]
            seed_created_at = f"{seed_day.isoformat()}T12:00:00+08:00" if seed_day is not None else None
            for msg in raw_step.get("messages") or []:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role") or "").strip().lower()
                content = str(msg.get("content") or "").strip()
                if role in {"user", "assistant"} and content:
                    app.state.append_message(chat_id, role, content, created_at=seed_created_at)
                    seeded += 1
            step_logs.append(
                {
                    "step": idx,
                    "action": "seed_history",
                    "chat_id": chat_id,
                    "seeded_messages": seeded,
                }
            )
        elif action == "chat":
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
            ref_day = step_day or current_today()
            orig_cons_llm = app.consolidator.llm
            app.consolidator.llm = app.cache_monitor.wrap(
                orig_cons_llm, session_key=f"consolidate-{chat_id}"
            )
            try:
                c_res = app.consolidator.consolidate_once(chat_id, today=ref_day)
            finally:
                app.consolidator.llm = orig_cons_llm
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
            ref_day = step_day or current_today()
            archived = app.memory.archive_expired_memories(today=ref_day)
            app.reset_session_prompt()
            step_logs.append({"step": idx, "action": "archive", "archived": archived})
        elif action == "check_search":
            ref_day = step_day or current_today()
            root = str(raw_step.get("root") or "data/memory")
            query = str(raw_step.get("query") or "")
            hits = app.memory.search_hits(query, root=root, today=ref_day)
            expect_empty = bool(raw_step.get("expect_empty", False))
            if expect_empty and hits:
                reasons.append(f"step#{idx} check_search({query!r}) 期望为空，实际命中 {len(hits)} 条")
            must_hit = raw_step.get("must_include")
            if isinstance(must_hit, str) and must_hit:
                joined = "\n".join(h.snippet for h in hits)
                if not _contains_keyword(joined, must_hit):
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
            step_reasons, judge_detail = _check_ask_assertions(
                app, raw_step, res, judge_llm=judge_llm
            )
            reasons.extend(f"step#{idx} {r}" for r in step_reasons)
            entry: dict[str, object] = {
                "step": idx,
                "action": "ask",
                "chat_id": chat_id,
                "tool_calls": [t["tool"] for t in res["tool_calls"]],
                "reply": res["reply"],
                "cache": res["cache"],
                "check_failures": step_reasons,
            }
            if judge_detail is not None:
                entry["judge"] = judge_detail
            step_logs.append(entry)
        else:
            reasons.append(f"step#{idx} 未知 action：{action}")

    return step_logs, reasons


# 公共沙箱会话构造与执行辅助函数：统一基于 AssistantApp(auto_consolidate=False) 执行步骤并回收资源
def _run_eval_session(
    *,
    data_root: Path,
    llm: ModelInvoker,
    consolidator_llm: ModelInvoker | None,
    cfg_path: Path | None,
    project_root: Path,
    tools_config: ToolsConfig | None,
    spec: dict[str, object],
    steps: list[dict[str, object]],
    base_data_dir: Path | None = None,
    judge_llm: ModelInvoker | None = None,
) -> tuple[list[dict[str, object]], list[str], list[CallCacheRecord], dict[str, object], str | None]:
    policy = _build_policy(spec.get("policy"))
    clock_day: list[date | None] = [parse_optional_date(spec.get("today"))]

    # 返回当前评测沙箱的逻辑日期（未指定时回退北京时间当日）
    def current_today() -> date:
        return clock_day[0] or today_shanghai()

    app: AssistantApp | None = None
    step_logs: list[dict[str, object]] = []
    reasons: list[str] = []
    records: list[CallCacheRecord] = []
    cache_summary = summarize_records([])
    exc_text: str | None = None
    try:
        if base_data_dir is not None:
            _copy_data_snapshot(base_data_dir, data_root)
        app = AssistantApp(
            data_root=data_root,
            config_path=cfg_path if (cfg_path and cfg_path.is_file()) else None,
            project_root=project_root,
            tools_config=tools_config,
            llm=llm,
            consolidator_llm=consolidator_llm,
            consolidation_policy=policy,
            auto_consolidate=False,
            today_fn=current_today,
        )
        if base_data_dir is None:
            for seed in spec.get("seed_memories") or []:
                if not isinstance(seed, dict):
                    continue
                app.memory.apply_memory_operations(
                    str(seed.get("topic") or "facts"),
                    str(seed.get("title") or "预置记忆"),
                    str(seed.get("index_summary") or "预置主题记忆"),
                    list(seed.get("operations") or []),
                    today=parse_optional_date(seed.get("today")),
                )
        step_logs, reasons = _execute_steps(
            app, steps, clock_day, current_today, judge_llm=judge_llm
        )
        records = list(app.cache_monitor.records)
        cache_summary = app.cache_monitor.summary()
    except Exception as exc:  # noqa: BLE001
        exc_text = f"{type(exc).__name__}: {exc}"
        if app is not None:
            records = list(app.cache_monitor.records)
            cache_summary = app.cache_monitor.summary()
    finally:
        if app is not None:
            app.close()
    return step_logs, reasons, records, cache_summary, exc_text


# 为共享长记忆组执行一次前置历史注入与沉淀，产出可供多道 QA 并发拷贝复用的基准目录快照
def _build_shared_setup(
    llm: ModelInvoker,
    group_id: str,
    setup_spec: dict[str, object],
    base_data_dir: Path,
    cfg_path: Path | None,
    project_root: Path,
    *,
    tools_config: ToolsConfig | None = None,
    consolidator_llm: ModelInvoker | None = None,
) -> tuple[dict[str, object], list[CallCacheRecord]]:
    steps = list(setup_spec.get("steps") or [])
    t0 = perf_counter()
    step_logs, reasons, records, cache_summary, setup_exception = _run_eval_session(
        data_root=base_data_dir,
        llm=llm,
        consolidator_llm=consolidator_llm,
        cfg_path=cfg_path,
        project_root=project_root,
        tools_config=tools_config,
        spec=setup_spec,
        steps=steps,
    )
    if setup_exception is not None:
        reasons = [f"shared_setup_exception: {setup_exception}"]

    elapsed_ms = round((perf_counter() - t0) * 1000, 3)
    summary: dict[str, object] = {
        "shared_group": group_id,
        "steps_executed": len(step_logs),
        "errors": reasons,
        "latency_ms": elapsed_ms,
        "cache": cache_summary,
    }
    if setup_exception is not None:
        summary["exception"] = setup_exception
    return summary, records


# 在完全隔离的临时目录中执行单个端到端记忆场景（支持从共享沉淀快照拷贝初始状态，并隔离单题运行时异常）
def _run_single_scenario(
    llm: ModelInvoker,
    scenario: dict[str, object],
    cfg_path: Path | None,
    project_root: Path,
    *,
    tools_config: ToolsConfig | None = None,
    consolidator_llm: ModelInvoker | None = None,
    judge_llm: ModelInvoker | None = None,
    base_data_dir: Path | None = None,
    steps_override: list[dict[str, object]] | None = None,
    setup_error: str | None = None,
) -> tuple[dict[str, object], list[CallCacheRecord]]:
    sc_id = str(scenario["id"])
    category = str(scenario.get("category") or "uncategorized").strip() or "uncategorized"
    group_id = str(scenario.get("shared_group") or "").strip()
    if setup_error is not None:
        fail_record: dict[str, object] = {
            "id": sc_id,
            "category": category,
            "description": str(scenario.get("description") or ""),
            "passed": False,
            "reasons": [f"scenario_exception: shared_setup({group_id}) failed: {setup_error}"],
            "latency_ms": 0.0,
            "cache": summarize_records([]),
            "steps": [],
        }
        if group_id:
            fail_record["shared_group"] = group_id
        return fail_record, []

    steps = list(steps_override) if steps_override is not None else list(scenario.get("steps") or [])
    t0 = perf_counter()
    safe_sc_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", sc_id) or "scenario"
    with tempfile.TemporaryDirectory(prefix=f"my-helper-memeval-{safe_sc_id}-") as tmp:
        step_logs, reasons, records, cache_summary, exc_text = _run_eval_session(
            data_root=Path(tmp) / "data",
            llm=llm,
            consolidator_llm=consolidator_llm,
            cfg_path=cfg_path,
            project_root=project_root,
            tools_config=tools_config,
            spec=scenario,
            steps=steps,
            base_data_dir=base_data_dir,
            judge_llm=judge_llm,
        )
    if exc_text is not None:
        reasons = [f"scenario_exception: {exc_text}"]

    elapsed_ms = round((perf_counter() - t0) * 1000, 3)
    record: dict[str, object] = {
        "id": sc_id,
        "category": category,
        "description": str(scenario.get("description") or ""),
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "latency_ms": elapsed_ms,
        "cache": cache_summary,
        "steps": step_logs,
    }
    if group_id:
        record["shared_group"] = group_id
    return record, records


# 汇总一组多轮场景的通过率，用于总体 summary 与按 category 分类统计
def _aggregate_scenario_records(records: list[dict[str, object]]) -> dict[str, object]:
    total = len(records)
    passed = sum(1 for item in records if item["passed"])
    return {
        "total_scenarios": total,
        "passed_scenarios": passed,
        "failed_scenarios": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
    }


_T = TypeVar("_T")
_R = TypeVar("_R")


# 按输入顺序或线程池并发执行独立作业：workers>1 时主线程依次回调 on_done，并严格按原序返回
def _run_ordered_jobs(
    items: list[_T],
    worker_fn: Callable[[_T], _R],
    *,
    workers: int,
    on_done: Callable[[int, int, _R], None] | None = None,
) -> list[_R]:
    total = len(items)
    if total == 0:
        return []
    if workers == 1 or total == 1:
        seq_results: list[_R] = []
        for idx, item in enumerate(items, start=1):
            res = worker_fn(item)
            seq_results.append(res)
            if on_done is not None:
                on_done(idx, total, res)
        return seq_results

    results: list[_R | None] = [None] * total
    with ThreadPoolExecutor(max_workers=min(workers, total)) as pool:
        future_to_idx: dict[Future[_R], int] = {
            pool.submit(worker_fn, item): idx for idx, item in enumerate(items)
        }
        try:
            done = 0
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                res = fut.result()
                results[idx] = res
                done += 1
                if on_done is not None:
                    on_done(done, total, res)
        except Exception:
            for pending in future_to_idx:
                pending.cancel()
            raise
    return cast(list[_R], results)


# 批量运行端到端记忆与生命周期评测（默认 workers=1 串行；workers>1 使用线程池并发独立场景并按输入序汇总）
def evaluate_memory(
    llm: ModelInvoker,
    scenarios: list[dict[str, object]],
    *,
    config_path: Path | None = None,
    tools_config: ToolsConfig | None = None,
    consolidator_llm: ModelInvoker | None = None,
    judge_llm: ModelInvoker | None = None,
    git_commit: str | None = None,
    workers: int = 1,
    on_progress: ProgressCallback | None = None,
    on_setup_done: SetupDoneCallback | None = None,
) -> dict[str, object]:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers 必须为 >= 1 的整数。")

    cfg_path, project_root, loaded_cfg, owns_cfg = resolve_eval_tools_config(
        config_path, tools_config
    )

    # 识别本轮所有待构建的共享长记忆组，并保留每个场景对应的独立执行步骤
    shared_specs: dict[str, dict[str, object]] = {}
    prepared_items: list[tuple[dict[str, object], str | None, list[dict[str, object]]]] = []
    for scenario in scenarios:
        group_id, setup_spec, qa_steps = _split_shared_setup(scenario)
        if group_id and setup_spec is not None:
            shared_specs.setdefault(group_id, setup_spec)
        prepared_items.append((scenario, group_id if setup_spec is not None else None, qa_steps))

    shared_setups_report: dict[str, dict[str, object]] = {}
    setup_cache_records: list[CallCacheRecord] = []
    per_scenario: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = {}
    all_cache_records: list[CallCacheRecord] = []
    latencies: list[float] = []

    try:
        with tempfile.TemporaryDirectory(prefix="my-helper-shared-eval-") as shared_tmp:
            shared_dirs: dict[str, Path] = {}
            setup_items = list(shared_specs.items())
            for idx, (gid, _) in enumerate(setup_items):
                safe_dir = re.sub(r"[^a-zA-Z0-9_-]+", "_", gid) or f"group_{idx}"
                shared_dirs[gid] = Path(shared_tmp) / f"{idx:03d}_{safe_dir}"

            # Phase 1：先构建全部共享长记忆只读快照（workers>1 且存在多组时并发）
            setup_results = _run_ordered_jobs(
                setup_items,
                lambda item: _build_shared_setup(
                    llm,
                    item[0],
                    item[1],
                    shared_dirs[item[0]],
                    cfg_path,
                    project_root,
                    tools_config=loaded_cfg,
                    consolidator_llm=consolidator_llm,
                ),
                workers=workers,
                on_done=(
                    (lambda _i, _n, item: on_setup_done(item[0]))
                    if on_setup_done is not None
                    else None
                ),
            )
            for (gid, _), (s_rep, s_recs) in zip(setup_items, setup_results, strict=True):
                shared_setups_report[gid] = s_rep
                setup_cache_records.extend(s_recs)
            all_cache_records.extend(setup_cache_records)

            # Phase 2：并发或串行执行各隔离场景并按输入原序汇总
            run_jobs = [
                (
                    scenario,
                    qa_steps,
                    shared_dirs.get(gid) if gid else None,
                    cast(str | None, shared_setups_report.get(gid, {}).get("exception"))
                    if gid
                    else None,
                )
                for scenario, gid, qa_steps in prepared_items
            ]
            scenario_results = _run_ordered_jobs(
                run_jobs,
                lambda job: _run_single_scenario(
                    llm,
                    job[0],
                    cfg_path,
                    project_root,
                    tools_config=loaded_cfg,
                    consolidator_llm=consolidator_llm,
                    judge_llm=judge_llm,
                    base_data_dir=job[2],
                    steps_override=job[1],
                    setup_error=job[3],
                ),
                workers=workers,
                on_done=(
                    (lambda done, total_n, item: on_progress(done, total_n, item[0]))
                    if on_progress is not None
                    else None
                ),
            )

            for record, cache_records in scenario_results:
                per_scenario.append(record)
                grouped.setdefault(str(record["category"]), []).append(record)
                all_cache_records.extend(cache_records)
                latencies.append(float(record["latency_ms"]))
                if not record["passed"]:
                    failures.append(
                        {
                            "id": record["id"],
                            "category": record["category"],
                            "reason": "; ".join(record["reasons"]),
                            "reasons": record["reasons"],
                        }
                    )
    finally:
        if owns_cfg and loaded_cfg is not None:
            loaded_cfg.plugin_manager.terminate_all()

    effective_judge = judge_llm
    use_judge = effective_judge is not None and not isinstance(effective_judge, NoopModelClient)
    judge_meta: dict[str, object] = {
        "enabled": use_judge,
        "model": (
            getattr(effective_judge, "model", None) or type(effective_judge).__name__
            if use_judge
            else None
        ),
        "self_judged": bool(use_judge and (judge_llm is None or judge_llm is llm)),
        "prompt_version": JUDGE_PROMPT_VERSION if use_judge else None,
        "prompt_sha256": JUDGE_PROMPT_SHA256 if use_judge else None,
    }
    summary = _aggregate_scenario_records(per_scenario)
    summary["mean_latency_ms"] = round(mean(latencies), 3) if latencies else 0.0
    summary["by_category"] = {
        cat: _aggregate_scenario_records(items) for cat, items in sorted(grouped.items())
    }
    summary["cache"] = summarize_records(all_cache_records)
    report: dict[str, object] = {
        "mode": "memory",
        "git_commit": git_commit or current_git_commit(),
        "timestamp": datetime.now(UTC).isoformat(),
        "judge": judge_meta,
        "summary": summary,
        "failures": failures,
        "scenarios": per_scenario,
    }
    if shared_setups_report:
        report["shared_setups"] = shared_setups_report
    return report
