"""公开学术记忆与检索基准适配器：支持零依赖下载并转换 LongMemEval（ICLR 2025）与 LoCoMo（ACL 2024）。"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import urllib.request
from datetime import date, datetime
from pathlib import Path

LONGMEMEVAL_ORACLE_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json"
)
LOCOMO10_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

LOCOMO_CATEGORY_NAMES: dict[int, str] = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

_LME_DATE_RE = re.compile(r"(\d{4})[/-](\d{2})[/-](\d{2})(?:.*?(\d{2}):(\d{2}))?")
_LOCOMO_DATE_RE = re.compile(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})")
_SESSION_KEY_RE = re.compile(r"^session_(\d+)$")
_DIA_ID_RE = re.compile(r"D(\d+):(\d+)")
_MONTH_MAP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


# 解析 LongMemEval 时间戳字符串（如 "2023/04/10 (Mon) 17:50"）为可排序的 datetime
def parse_longmemeval_datetime(raw: str) -> datetime:
    match = _LME_DATE_RE.search(str(raw or ""))
    if not match:
        raise ValueError(f"无法解析 LongMemEval 日期：{raw!r}")
    year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
    hour = int(match.group(4)) if match.group(4) else 0
    minute = int(match.group(5)) if match.group(5) else 0
    return datetime(year, month, day, hour, minute)


# 解析 LoCoMo 会话日期字符串（如 "1:56 pm on 8 May, 2023"）为 date，无法识别时返回 None
def parse_locomo_date(raw: str) -> date | None:
    match = _LOCOMO_DATE_RE.search(str(raw or ""))
    if not match:
        return None
    day = int(match.group(1))
    month = _MONTH_MAP.get(match.group(2).lower())
    year = int(match.group(3))
    if month is None:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


# 按 category 字段执行确定性分层轮询抽样，确保小样本也均衡覆盖各类题型
def stratified_sample(
    items: list[dict[str, object]],
    *,
    limit: int | None = None,
    seed: int = 42,
) -> list[dict[str, object]]:
    if limit is None or limit >= len(items):
        return list(items)
    if limit < 1:
        raise ValueError("limit 必须 >= 1。")
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, object]]] = {}
    for item in items:
        cat = str(item.get("category") or "uncategorized")
        buckets.setdefault(cat, []).append(item)
    ordered_cats = sorted(buckets.keys())
    for cat in ordered_cats:
        rng.shuffle(buckets[cat])

    selected: list[dict[str, object]] = []
    while len(selected) < limit:
        progressed = False
        for cat in ordered_cats:
            if buckets[cat] and len(selected) < limit:
                selected.append(buckets[cat].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


# 将单条 LongMemEval 样本转换为 memory_eval 可直接运行的端到端多轮场景
def convert_longmemeval_item(item: dict[str, object]) -> dict[str, object]:
    qid = str(item.get("question_id") or "").strip()
    question = str(item.get("question") or "").strip()
    answer = str(item.get("answer") if item.get("answer") is not None else "").strip()
    if not qid or not question or not answer:
        raise ValueError(f"LongMemEval 样本缺少必要字段：{qid!r}")

    is_abstention = qid.endswith("_abs")
    qtype = str(item.get("question_type") or "unknown").strip()
    category = "abstention" if is_abstention else qtype
    question_dt = parse_longmemeval_datetime(str(item.get("question_date") or ""))

    sessions = list(item.get("haystack_sessions") or [])
    dates = list(item.get("haystack_dates") or [])
    if len(sessions) != len(dates):
        raise ValueError(f"LongMemEval 样本 {qid} 的 haystack_sessions 与 haystack_dates 长度不一致。")

    # 按会话发生时间先后排序，避免乱序沉淀破坏知识更新与时间推理顺序
    indexed_sessions: list[tuple[datetime, int, list[dict[str, object]]]] = []
    for idx, (raw_dt, raw_turns) in enumerate(zip(dates, sessions, strict=True)):
        if not isinstance(raw_turns, list) or not raw_turns:
            continue
        parsed_dt = parse_longmemeval_datetime(str(raw_dt))
        indexed_sessions.append((parsed_dt, idx, raw_turns))
    indexed_sessions.sort(key=lambda entry: (entry[0], entry[1]))

    steps: list[dict[str, object]] = []
    for order_idx, (sess_dt, _, raw_turns) in enumerate(indexed_sessions, start=1):
        clean_msgs: list[dict[str, str]] = []
        for turn in raw_turns:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role") or "").strip().lower()
            content = str(turn.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                clean_msgs.append({"role": role, "content": content})
        if not clean_msgs:
            continue
        chat_id = f"lme-s{order_idx}"
        day_iso = sess_dt.date().isoformat()
        steps.append({"action": "seed_history", "chat_id": chat_id, "messages": clean_msgs})
        steps.append({"action": "consolidate", "chat_id": chat_id, "today": day_iso})

    steps.append(
        {
            "action": "ask",
            "chat_id": "eval-query",
            "today": question_dt.date().isoformat(),
            "message": question,
            "expect_tools_any": ["search_markdown", "read_markdown", "list_files"],
            "expected_answer": answer,
            "is_abstention": is_abstention,
        }
    )

    return {
        "id": f"lme-{qid}",
        "source": "longmemeval",
        "category": category,
        "question_type": qtype,
        "description": f"LongMemEval ({category}): {question[:80]}",
        "policy": {
            "max_active_messages": 1,
            "keep_recent_messages": 0,
            "min_batch_messages": 1,
        },
        "steps": steps,
    }


# 批量转换 LongMemEval 原始样本并支持按类别分层抽样
def convert_longmemeval_dataset(
    items: list[dict[str, object]],
    *,
    limit: int | None = None,
    seed: int = 42,
) -> list[dict[str, object]]:
    converted = [convert_longmemeval_item(item) for item in items if isinstance(item, dict)]
    return stratified_sample(converted, limit=limit, seed=seed)


# 提取单个 LoCoMo 对话中按编号升序排列的会话列表：(会话序号, 日期文本, 发言列表)
def _extract_locomo_sessions(
    conversation: dict[str, object],
) -> list[tuple[int, str, list[dict[str, object]]]]:
    sessions: list[tuple[int, str, list[dict[str, object]]]] = []
    for key, value in conversation.items():
        match = _SESSION_KEY_RE.match(key)
        if not match or not isinstance(value, list):
            continue
        num = int(match.group(1))
        dt_str = str(conversation.get(f"session_{num}_date_time") or "").strip()
        turns = [t for t in value if isinstance(t, dict)]
        if turns:
            sessions.append((num, dt_str, turns))
    sessions.sort(key=lambda item: item[0])
    return sessions


# 从 LoCoMo QA 的 evidence 列表提取标准 dia_id 列表（兼容分号/逗号拼接格式）
def _parse_locomo_evidence(raw_evidence: object) -> list[str]:
    if not isinstance(raw_evidence, list):
        return []
    found: list[str] = []
    for entry in raw_evidence:
        for match in _DIA_ID_RE.finditer(str(entry)):
            dia_id = f"D{match.group(1)}:{match.group(2)}"
            if dia_id not in found:
                found.append(dia_id)
    return found


# 清理对话标识以便用作安全文件名
def _safe_slug(raw: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(raw).strip()).strip("_").lower()
    return slug or "conv"


# 将 LoCoMo 数据集导出为可离线评测的 Markdown 语料与检索案例集（零 API 消耗）
def export_locomo_for_search(
    conversations: list[dict[str, object]],
    output_data_dir: Path,
    *,
    limit: int | None = None,
    seed: int = 42,
) -> list[dict[str, object]]:
    memory_dir = output_data_dir.resolve() / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    all_cases: list[dict[str, object]] = []
    for conv_idx, item in enumerate(conversations, start=1):
        if not isinstance(item, dict):
            continue
        sample_id = _safe_slug(str(item.get("sample_id") or f"conv_{conv_idx}"))
        conv = item.get("conversation")
        if not isinstance(conv, dict):
            continue
        sessions = _extract_locomo_sessions(conv)
        dia_lookup: dict[str, tuple[str, str]] = {}

        for sess_num, dt_str, turns in sessions:
            filename = f"locomo_{sample_id}_s{sess_num:02d}.md"
            display_path = f"data/memory/{filename}"
            lines = [f"# LoCoMo {sample_id} Session {sess_num} ({dt_str or 'unknown date'})", ""]
            for turn in turns:
                dia_id = str(turn.get("dia_id") or "").strip()
                speaker = str(turn.get("speaker") or "Speaker").strip()
                raw_text = " ".join(str(turn.get("text") or "").split())
                if not dia_id or not raw_text:
                    continue
                line = f"- [{dia_id}] {speaker}: {raw_text}"
                lines.append(line)
                quote = f"[{dia_id}] {speaker}: {raw_text[:60]}".rstrip()
                dia_lookup[dia_id] = (display_path, quote)
            (memory_dir / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")

        for qa_idx, qa in enumerate(item.get("qa") or [], start=1):
            if not isinstance(qa, dict):
                continue
            cat_num = int(qa.get("category") or 0)
            # 类别 5 是对抗拒答题，不适合作为离线关键词检索的正例/零命中负例
            if cat_num not in {1, 2, 3, 4}:
                continue
            question = str(qa.get("question") or "").strip()
            evidence_ids = _parse_locomo_evidence(qa.get("evidence"))
            sources = [
                {"path": dia_lookup[eid][0], "quote": dia_lookup[eid][1]}
                for eid in evidence_ids
                if eid in dia_lookup
            ]
            if not question or not sources:
                continue
            all_cases.append(
                {
                    "id": f"locomo-{sample_id}-q{qa_idx:03d}",
                    "source": "locomo",
                    "category": LOCOMO_CATEGORY_NAMES[cat_num],
                    "query": question,
                    "answerable": True,
                    "expected_sources": sources,
                    "expected_tools": ["search_markdown"],
                }
            )

    return stratified_sample(all_cases, limit=limit, seed=seed)


# 将 LoCoMo 对话与 QA 转换为多轮记忆端到端场景（按最大会话数控制成本并保留全部 5 类题型）
def convert_locomo_for_memory(
    conversations: list[dict[str, object]],
    *,
    limit: int | None = None,
    max_sessions_per_conv: int = 4,
    seed: int = 42,
) -> list[dict[str, object]]:
    scenarios: list[dict[str, object]] = []
    for conv_idx, item in enumerate(conversations, start=1):
        if not isinstance(item, dict):
            continue
        sample_id = _safe_slug(str(item.get("sample_id") or f"conv_{conv_idx}"))
        conv = item.get("conversation")
        if not isinstance(conv, dict):
            continue
        speaker_a = str(conv.get("speaker_a") or "").strip()
        sessions = _extract_locomo_sessions(conv)
        if not sessions:
            continue
        kept_sessions = sessions[:max_sessions_per_conv]
        kept_nums = {num for num, _, _ in kept_sessions}

        session_steps: list[dict[str, object]] = []
        last_day: date | None = None
        for sess_num, dt_str, turns in kept_sessions:
            parsed_day = parse_locomo_date(dt_str)
            if parsed_day is not None:
                last_day = parsed_day
            day_iso = (parsed_day or last_day or date(2023, 5, 1)).isoformat()
            msgs: list[dict[str, str]] = []
            for turn in turns:
                speaker = str(turn.get("speaker") or "Speaker").strip()
                text = " ".join(str(turn.get("text") or "").split())
                if not text:
                    continue
                role = "user" if (not speaker_a or speaker == speaker_a) else "assistant"
                msgs.append({"role": role, "content": f"[{speaker}] {text}"})
            if not msgs:
                continue
            chat_id = f"locomo-s{sess_num}"
            session_steps.append({"action": "seed_history", "chat_id": chat_id, "messages": msgs})
            session_steps.append({"action": "consolidate", "chat_id": chat_id, "today": day_iso})

        if not session_steps:
            continue
        eval_day = (last_day or date(2023, 6, 1)).isoformat()

        for qa_idx, qa in enumerate(item.get("qa") or [], start=1):
            if not isinstance(qa, dict):
                continue
            cat_num = int(qa.get("category") or 0)
            category = LOCOMO_CATEGORY_NAMES.get(cat_num)
            if category is None:
                continue
            evidence_ids = _parse_locomo_evidence(qa.get("evidence"))
            ev_sessions = {int(m.group(1)) for eid in evidence_ids if (m := _DIA_ID_RE.match(eid))}
            if ev_sessions and not ev_sessions.issubset(kept_nums):
                continue
            question = str(qa.get("question") or "").strip()
            if not question:
                continue
            is_abs = cat_num == 5
            if is_abs:
                adv = str(qa.get("adversarial_answer") or "").strip()
                expected_answer = (
                    f"Not enough information / not mentioned in the conversation (do not answer '{adv}')."
                    if adv
                    else "Not enough information / not mentioned in the conversation."
                )
            else:
                expected_answer = str(qa.get("answer") if qa.get("answer") is not None else "").strip()
                if not expected_answer:
                    continue

            scenarios.append(
                {
                    "id": f"locomo-{sample_id}-q{qa_idx:03d}",
                    "source": "locomo",
                    "category": category,
                    "description": f"LoCoMo ({category}): {question[:80]}",
                    "policy": {
                        "max_active_messages": 1,
                        "keep_recent_messages": 0,
                        "min_batch_messages": 1,
                    },
                    "steps": [
                        *session_steps,
                        {
                            "action": "ask",
                            "chat_id": "eval-query",
                            "today": eval_day,
                            "message": question,
                            "expect_tools_any": ["search_markdown", "read_markdown", "list_files"],
                            "expected_answer": expected_answer,
                            "is_abstention": is_abs,
                        },
                    ],
                }
            )

    return stratified_sample(scenarios, limit=limit, seed=seed)


# 将评测案例或场景列表写出为标准 UTF-8 JSONL 文件
def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n"
    path.write_text(content, encoding="utf-8")


# 使用纯标准库安全下载远端公开数据集到本地缓存（原子落盘，避免残缺文件）
def download_dataset(url: str, dest: Path, *, force: bool = False, timeout: float = 60.0) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and not force:
        return dest
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "my-helper-eval/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, tmp_path.open("wb") as handle:
            shutil.copyfileobj(resp, handle)
        os.replace(tmp_path, dest)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return dest
