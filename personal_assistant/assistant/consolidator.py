"""自动记忆沉淀与上下文压缩模块：达阈值后后台提炼长期记忆并缩减活跃上下文。"""
from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from time import perf_counter

from personal_assistant.agent.plugins import TaskEvent
from personal_assistant.ai import Context, ModelInvoker, NoopModelClient, TokenUsage, UserMessage
from personal_assistant.assistant.common import format_date_with_weekday, today_shanghai
from personal_assistant.assistant.db import StateStore
from personal_assistant.assistant.memory import MemoryStore

CONSOLIDATION_SYSTEM_PROMPT = (
    "你是个人助手的后台记忆整理器。你会收到当前日期、已有主题记忆内容、MEMORY.md 索引、前情摘要，以及一段即将从活跃窗口移除的历史对话。\n"
    "请严格输出一个合法 JSON 对象（不要输出任何额外解释），结构如下：\n"
    "{\n"
    '  "session_summary": "合并旧摘要与本批对话后的精简前情摘要（3-6 条要点，保留未完事项与关键上下文）",\n'
    '  "memories": [\n'
    "    {\n"
    '      "topic": "仅含小写字母/数字/连字符的主题文件名，例如 preferences、projects、facts",\n'
    '      "title": "简短中文标题，如 个人偏好、项目备忘",\n'
    '      "index_summary": "一句简短索引说明（20字以内）",\n'
    '      "operations": [\n'
    '        {"op": "add", "text": "新增的明确事实/偏好/决定（可选填 until: YYYY-MM-DD）"},\n'
    '        {"op": "merge", "old": "已有记忆中需补充维度的旧句关键词", "text": "追加合并的互补细节（程序保留旧正文，不覆盖）"},\n'
    '        {"op": "replace", "old": "已有记忆中被更新的旧句关键词", "new": "更新后的完整陈述"},\n'
    '        {"op": "remove", "old": "已作废或用户明确要求忘掉的旧句关键词"}\n'
    "      ]\n"
    "    }\n"
    "  ]\n"
    "}\n"
    "严格要求：\n"
    "1. 严禁编造对话中未明确出现的信息；纯寒暄不要写入 memories（返回空数组 []）；优先完整记录用户自身的事实、决策、偏好及具体专有名词（人名、书名、作品、地点、数值）；助手给出的具体推荐或结论仅用一句话提炼关键实体与结论，不抄录长段通用科普或教程。\n"
    "2. 仅当同一属性/决策的旧值被新值推翻时（如修改端口、更换方案），用 op=replace 替换旧条目，且 new 只写当前最新有效状态，严禁括号附带或复述“从旧值改为/原方案已放弃”等废弃信息；同一活动/主题补充互补细节（如目标值与实际值、进球与助攻）优先用 op=merge 追加保留旧维度；新增的独立事件或累加记录一律用 op=add 保留；op=remove 直接删除作废条目，不另写“已取消”冗余句。\n"
    "3. 若 data/memory/facts.md 中已有带 (msg#...) 标记的原始记录对应本次整理内容，必须用 op=replace 原地规范化（或迁至其他主题后对 facts.md 执行 op=remove），绝不同时保留原始裸句与规范化副本。\n"
    "4. 条目开头的 [YYYY-MM-DD] 仅为记录日期；凡对话提及相对时间（如昨天、上周六、明天、下周三），均须结合【当前日期】换算为精确绝对日期（YYYY-MM-DD）写入正文，不要模糊成“某月初/某月”；仅对有明确未来截止日的临时日程/待办填写 until（YYYY-MM-DD），过往事件与永久偏好不要填 until。\n"
    "5. session_summary 必须简洁客观，便于后续轮次无缝承接上下文。"
)


# 自动沉淀与上下文缩减的阈值策略配置
@dataclass(frozen=True)
class ConsolidationPolicy:
    max_active_messages: int = 16
    max_active_chars: int = 8000
    keep_recent_messages: int = 6
    min_batch_messages: int = 4
    max_batch_messages: int = 24


# 单次沉淀与压缩完成后的结构化结果
@dataclass(frozen=True)
class ConsolidationResult:
    chat_id: str
    compacted_messages: int
    compacted_until_id: int
    saved_files: tuple[str, ...]
    summary: str

    # 转换为便于 /status 展示与日志记录的字典
    def to_dict(self) -> dict[str, object]:
        return {
            "chat_id": self.chat_id,
            "compacted_messages": self.compacted_messages,
            "compacted_until_id": self.compacted_until_id,
            "saved_files": list(self.saved_files),
            "summary_chars": len(self.summary),
        }


# 从模型输出文本中稳健提取首个完整 JSON 对象（兼容 Markdown 代码块、前置说明与尾部多余内容）
def _parse_json_object(raw: str) -> dict[str, object]:
    text = raw.strip()
    candidates: list[str] = []
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fence:
        candidates.append(fence.group(1).strip())
    candidates.append(text)

    decoder = json.JSONDecoder()
    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        pos = 0
        while True:
            start = candidate.find("{", pos)
            if start == -1:
                break
            try:
                parsed, _ = decoder.raw_decode(candidate, start)
            except json.JSONDecodeError as error:
                last_error = error
                pos = start + 1
                continue
            if isinstance(parsed, dict):
                return parsed
            pos = start + 1

    if last_error is not None:
        raise last_error
    raise ValueError("沉淀结果必须是 JSON 对象。")


# 将待沉淀的历史消息格式化为紧凑文本，供整理模型阅读
def _format_transcript(batch: list[dict[str, object]], max_item_chars: int = 1000) -> str:
    lines: list[str] = []
    for msg in batch:
        role = "用户" if msg.get("role") == "user" else "助手"
        content = str(msg.get("content") or "").strip()
        if len(content) > max_item_chars:
            content = content[:max_item_chars] + "…"
        lines.append(f"[{role} #{msg.get('id')}] {content}")
    return "\n".join(lines)


# 纯同步记忆沉淀与上下文压缩核心（不绑定线程，未来可直接挂载到独立 worker）
class MemoryConsolidator:
    # 绑定状态存储、Markdown 记忆库、模型调用句柄、触发策略与可选逻辑时钟
    def __init__(
        self,
        state: StateStore,
        memory: MemoryStore,
        llm: ModelInvoker,
        *,
        policy: ConsolidationPolicy | None = None,
        today_fn: Callable[[], date] | None = None,
    ) -> None:
        self.state = state
        self.memory = memory
        self.llm = llm
        self.policy = policy or ConsolidationPolicy()
        self._today_fn = today_fn
        self._usage_local = threading.local()

    # 读取当前线程最近一次沉淀调用记录的 TokenUsage（隔离热重载与并发调用）
    @property
    def last_usage(self) -> TokenUsage | None:
        value = getattr(self._usage_local, "value", None)
        return value if isinstance(value, TokenUsage) else None

    # 检查指定会话是否达到沉淀阈值；若达到则返回（旧摘要，待沉淀旧消息批次），否则返回 None
    def select_batch(self, chat_id: str) -> tuple[str, list[dict[str, object]]] | None:
        compaction, active = self.state.get_context_snapshot(chat_id)
        if not active:
            return None
        total_chars = sum(len(str(msg.get("content") or "")) for msg in active)
        reached = (
            len(active) >= self.policy.max_active_messages
            or total_chars >= self.policy.max_active_chars
        )
        if not reached:
            return None
        cutoff = len(active) - max(0, self.policy.keep_recent_messages)
        if cutoff < self.policy.min_batch_messages:
            return None
        max_batch = self.policy.max_batch_messages
        if max_batch > 0 and cutoff > max_batch:
            cap = max_batch - (max_batch % 2) if max_batch >= 2 else max_batch
            if (
                cap > 1
                and cap < len(active)
                and active[cap - 1].get("role") == "user"
                and active[cap].get("role") == "assistant"
            ):
                cap -= 1
            cutoff = max(1, cap)
        existing_summary = str(compaction.get("summary") or "")
        return existing_summary, active[:cutoff]

    # 对指定会话执行一次完整沉淀：归档过期记忆、读旧改旧消冲突、生成滚动摘要并原子推进游标
    def consolidate_once(
        self,
        chat_id: str,
        *,
        today: date | None = None,
    ) -> ConsolidationResult | None:
        llm = self.llm
        self._usage_local.value = None
        if isinstance(llm, NoopModelClient):
            return None
        selected = self.select_batch(chat_id)
        if selected is None:
            return None
        existing_summary, batch = selected
        ref_day = (
            today
            if today is not None
            else (self._today_fn() if self._today_fn is not None else today_shanghai())
        )

        # 沉淀前先零成本归档已过期的临时记忆，避免把过期条目再喂给模型
        self.memory.archive_expired_memories(today=ref_day)

        index_text = self.memory.read_memory_index() or "（暂无记忆索引）"
        topic_snapshots = self.memory.read_topic_snapshots() or "（暂无已有主题记忆）"
        transcript = _format_transcript(batch)
        user_payload = (
            f"【当前日期】{format_date_with_weekday(ref_day)}\n\n"
            f"【当前 MEMORY.md 索引】\n{index_text}\n\n"
            f"【已有主题记忆（用于识别冲突与 merge/replace/remove）】\n{topic_snapshots}\n\n"
            f"【已有前情摘要】\n{existing_summary or '（无）'}\n\n"
            f"【待沉淀与压缩的早期对话（共 {len(batch)} 条）】\n{transcript}"
        )
        context = Context(
            system_prompt=CONSOLIDATION_SYSTEM_PROMPT,
            messages=(UserMessage(content=user_payload),),
        )

        reply_msg = llm.complete(context)
        raw_usage = reply_msg.usage if isinstance(reply_msg.usage, TokenUsage) else llm.last_usage
        self._usage_local.value = raw_usage if isinstance(raw_usage, TokenUsage) else None
        parsed = _parse_json_object(reply_msg.text)

        summary = str(parsed.get("session_summary") or "").strip()
        if not summary:
            raise ValueError("模型未返回有效的 session_summary，取消推进压缩游标。")

        saved_files: list[str] = []
        applied_texts: list[str] = []
        raw_memories = parsed.get("memories")
        if isinstance(raw_memories, list):
            for entry in raw_memories:
                if not isinstance(entry, dict):
                    continue
                topic = str(entry.get("topic") or "").strip()
                title = str(entry.get("title") or topic).strip()
                index_summary = str(entry.get("index_summary") or "").strip()
                if not topic:
                    continue

                raw_ops = entry.get("operations")
                if not isinstance(raw_ops, list):
                    continue
                ops = [op for op in raw_ops if isinstance(op, dict)]
                if not ops:
                    continue
                try:
                    saved = self.memory.apply_memory_operations(
                        topic,
                        title,
                        index_summary,
                        ops,
                        today=ref_day,
                        applied_texts=applied_texts,
                    )
                except ValueError:
                    continue
                if saved and saved not in saved_files:
                    saved_files.append(saved)

        # 兜底清理 facts.md 中已被本次沉淀实际落盘吸收的原始 (msg#...) 幽灵行，避免临时事实漏过期
        batch_ids = {int(msg["id"]) for msg in batch if isinstance(msg.get("id"), int)}
        if applied_texts:
            self.memory.prune_consolidated_raw_facts(batch_ids, applied_texts)

        # 只有长期记忆文件落盘与摘要解析均成功后，才推进 SQLite 压缩游标
        last_id = batch[-1].get("id")
        assert isinstance(last_id, int)
        self.state.save_compaction(chat_id, summary, last_id)
        return ConsolidationResult(
            chat_id=chat_id,
            compacted_messages=len(batch),
            compacted_until_id=last_id,
            saved_files=tuple(saved_files),
            summary=summary,
        )


# 进程内后台调度器：用标准库线程异步执行沉淀，不阻塞对话回复并确保同会话互斥与失败指数退避
class BackgroundConsolidator:
    # 初始化后台调度锁、在途会话集合、按会话失败退避表与可选事件回调；回调异常由调用方隔离（AssistantApp._emit_event 已统一兜底）
    def __init__(
        self,
        consolidator: MemoryConsolidator,
        *,
        on_event: Callable[[str, TaskEvent], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        base_backoff_seconds: float = 2.0,
        max_failures_tracked: int = 256,
    ) -> None:
        self.consolidator = consolidator
        self.on_event = on_event
        self._clock = clock
        self._base_backoff_seconds = max(0.0, float(base_backoff_seconds))
        self._max_failures_tracked = max(1, int(max_failures_tracked))
        self._lock = threading.Lock()
        self._in_flight: set[str] = set()
        self._threads: dict[str, threading.Thread] = {}
        self._failures: dict[str, tuple[int, float]] = {}
        self.last_result: dict[str, object] | None = None
        self.last_error: str | None = None

    # 判断指定会话当前是否处于失败冷却期（需在持有 self._lock 时调用）
    def _is_in_cooldown(self, chat_id: str) -> bool:
        entry = self._failures.get(chat_id)
        if entry is None:
            return False
        return self._clock() < entry[1]

    # 检查阈值与冷却期后按需拉起后台线程执行沉淀；已触发返回 True，未达阈值/在途/冷却中返回 False
    def maybe_schedule(
        self,
        chat_id: str,
        *,
        on_complete: Callable[[ConsolidationResult], None] | None = None,
    ) -> bool:
        if isinstance(self.consolidator.llm, NoopModelClient):
            return False
        with self._lock:
            if chat_id in self._in_flight or self._is_in_cooldown(chat_id):
                return False
        if self.consolidator.select_batch(chat_id) is None:
            return False

        with self._lock:
            if chat_id in self._in_flight or self._is_in_cooldown(chat_id):
                return False
            self._in_flight.add(chat_id)
            worker = threading.Thread(
                target=self._run_worker,
                args=(chat_id, on_complete),
                name=f"consolidate-{chat_id}",
                daemon=True,
            )
            self._threads[chat_id] = worker
            worker.start()
        return True

    # 后台工作线程主体：捕获异常并按 chat_id 记录指数退避，完成后记录状态、单点广播事件并触发回调
    def _run_worker(
        self,
        chat_id: str,
        on_complete: Callable[[ConsolidationResult], None] | None,
    ) -> None:
        t0 = perf_counter()
        result: ConsolidationResult | None = None
        err_text: str | None = None
        try:
            result = self.consolidator.consolidate_once(chat_id)
            with self._lock:
                self._failures.pop(chat_id, None)
                if result is not None:
                    self.last_result = result.to_dict()
                    self.last_error = None
            if result is not None and on_complete is not None:
                on_complete(result)
        except Exception as error:  # noqa: BLE001
            err_text = f"{type(error).__name__}: {error}"
            now = self._clock()
            with self._lock:
                self.last_error = err_text
                prev = self._failures.pop(chat_id, None)
                count = (prev[0] if prev is not None else 0) + 1
                backoff = min(300.0, self._base_backoff_seconds * (2 ** min(count - 1, 7)))
                if len(self._failures) >= self._max_failures_tracked:
                    self._failures.pop(next(iter(self._failures)))
                self._failures[chat_id] = (count, now + backoff)
        finally:
            try:
                if self.on_event is not None and (result is not None or err_text is not None):
                    raw_usage = self.consolidator.last_usage
                    details = (
                        {
                            "compacted_messages": result.compacted_messages,
                            "compacted_until_id": result.compacted_until_id,
                            "saved_files": list(result.saved_files),
                        }
                        if result is not None
                        else {}
                    )
                    self.on_event(
                        chat_id,
                        TaskEvent(
                            kind="consolidation",
                            duration_ms=round((perf_counter() - t0) * 1000, 3),
                            usage=raw_usage if isinstance(raw_usage, TokenUsage) else None,
                            details=details,
                            error=err_text,
                        ),
                    )
            finally:
                with self._lock:
                    self._in_flight.discard(chat_id)
                    self._threads.pop(chat_id, None)

    # 等待所有在途后台沉淀线程结束（供程序退出或单元测试收尾调用）
    def wait(self, timeout: float = 15.0) -> None:
        with self._lock:
            threads = list(self._threads.values())
        for thread in threads:
            thread.join(timeout=timeout)

    # 返回当前后台沉淀运行状态、当前处于冷却期内的会话数与最近一次结果/错误信息，供 /status 展示
    def status(self) -> dict[str, object]:
        now = self._clock()
        with self._lock:
            active_backoffs = sum(1 for _, until in self._failures.values() if until > now)
            return {
                "running_sessions": sorted(self._in_flight),
                "backoff_sessions": active_backoffs,
                "last_result": self.last_result,
                "last_error": self.last_error,
            }
