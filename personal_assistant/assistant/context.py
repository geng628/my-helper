"""Layer 3 上下文装配与 KV Cache 生命周期引擎：动静分层组装、空闲 TTL 冻结、粘性按块裁剪、信源隔离与轮内非破坏性投影。"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import date, datetime

from personal_assistant.agent.tools import ToolRegistry
from personal_assistant.ai import (
    AssistantMessage,
    Context,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from personal_assistant.assistant.common import (
    SHANGHAI_TZ,
    parse_iso_date,
    require_non_negative_float,
    require_positive_int,
    today_shanghai,
)
from personal_assistant.assistant.memory import MemoryStore
from personal_assistant.assistant.prompts import (
    format_compaction_summary,
    format_system_prompt,
    format_user_envelope,
)

MAX_HISTORY_MESSAGES = 20
MAX_ADAPTIVE_HISTORY_MESSAGES = 60
MAX_ADAPTIVE_HISTORY_CHARS = 24_000
MAX_HISTORY_ITEM_CHARS = 2000
MAX_SESSION_PROMPTS = 256
DEFAULT_PROMPT_CACHE_TTL_SECONDS = 300.0
DEFAULT_FOLD_TRIGGER_CHARS = 48_000
DEFAULT_FOLD_KEEP_TAIL_MESSAGES = 4
DEFAULT_FOLD_MIN_RESULT_CHARS = 800
DEFAULT_FOLD_PREVIEW_CHARS = 240


# 判断消息字典的 id 字段是否为有效整数（排除布尔值）
def _valid_msg_id(raw_id: object) -> int | None:
    if isinstance(raw_id, int) and not isinstance(raw_id, bool):
        return raw_id
    return None


# 粘性边界按块裁剪活跃历史：未超上限保持队头纹丝不动，仅在超限时整块切至目标水位并推进 floor_id 以守护 KV Cache
def select_chunked_history(
    active_all: list[dict[str, object]],
    *,
    floor_id: int = 0,
    base_messages: int = MAX_HISTORY_MESSAGES,
    max_messages: int = MAX_ADAPTIVE_HISTORY_MESSAGES,
    max_chars: int = MAX_ADAPTIVE_HISTORY_CHARS,
) -> tuple[list[dict[str, object]], int]:
    if floor_id > 0:
        eligible = [
            msg
            for msg in active_all
            if (mid := _valid_msg_id(msg.get("id"))) is None or mid > floor_id
        ]
    else:
        eligible = list(active_all)

    total_chars = sum(min(len(str(msg.get("content") or "")), MAX_HISTORY_ITEM_CHARS) for msg in eligible)
    if len(eligible) <= max_messages and total_chars <= max_chars:
        while len(eligible) > 1 and eligible[0].get("role") == "assistant":
            eligible.pop(0)
        return eligible, floor_id

    target_msgs = min(max(2, base_messages), max_messages)
    target_chars = min(max_chars, max(MAX_HISTORY_ITEM_CHARS * 2, max_chars // 2))
    selected_rev: list[dict[str, object]] = []
    used_chars = 0
    for msg in reversed(eligible):
        item_len = min(len(str(msg.get("content") or "")), MAX_HISTORY_ITEM_CHARS)
        if selected_rev and (len(selected_rev) >= target_msgs or used_chars + item_len > target_chars):
            break
        selected_rev.append(msg)
        used_chars += item_len
    while len(selected_rev) > 1 and selected_rev[-1].get("role") == "assistant":
        selected_rev.pop()
    selected_rev.reverse()

    new_floor = floor_id
    if selected_rev:
        first_id = _valid_msg_id(selected_rev[0].get("id"))
        if first_id is not None:
            new_floor = max(floor_id, first_id - 1)
    return selected_rev, new_floor


# 从 SQLite 消息的 created_at 字段解析北京时间自然日（兼容带时区 datetime / ISO 与纯日期字符串）
def message_shanghai_date(raw_created_at: object) -> date | None:
    if isinstance(raw_created_at, datetime):
        return raw_created_at.astimezone(SHANGHAI_TZ).date() if raw_created_at.tzinfo is not None else raw_created_at.date()
    if isinstance(raw_created_at, str):
        text = raw_created_at.strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return dt.astimezone(SHANGHAI_TZ).date() if dt.tzinfo is not None else dt.date()
        except ValueError:
            return parse_iso_date(text)
    return parse_iso_date(raw_created_at)


# 向中立消息列表追加对话消息，遇到连续同角色纯文本消息时安全合并以防严格交替协议报错
def append_alternating_message(
    messages: list[Message],
    message: Message,
) -> None:
    if not messages:
        messages.append(message)
        return
    prev = messages[-1]
    if isinstance(prev, UserMessage) and isinstance(message, UserMessage):
        merged = (
            f"{prev.content}\n\n{message.content}"
            if (prev.content and message.content)
            else (prev.content or message.content)
        )
        messages[-1] = UserMessage(content=merged)
        return
    if (
        isinstance(prev, AssistantMessage)
        and isinstance(message, AssistantMessage)
        and not prev.tool_calls
        and not message.tool_calls
        and not prev.thinking
        and not message.thinking
    ):
        prev_text = prev.text
        curr_text = message.text
        merged = f"{prev_text}\n\n{curr_text}" if (prev_text and curr_text) else (prev_text or curr_text)
        messages[-1] = AssistantMessage.from_text(merged)
        return
    messages.append(message)


# 将持久化历史记录转换为中立消息列表：按记录日期还原用户信封并合并异常连续同角色，保持前缀字节级稳定
def build_history_messages(history: list[dict[str, object]]) -> list[Message]:
    result: list[Message] = []
    for msg in history:
        role = str(msg.get("role") or "")
        if role not in {"user", "assistant"}:
            continue
        content = str(msg.get("content") or "")
        if len(content) > MAX_HISTORY_ITEM_CHARS:
            content = content[:MAX_HISTORY_ITEM_CHARS] + "…"
        if role == "user":
            msg_day = message_shanghai_date(msg.get("created_at"))
            rendered = format_user_envelope(content, today=msg_day) if msg_day is not None else content
            item: Message = UserMessage(content=rendered)
        else:
            item = AssistantMessage.from_text(content)
        append_alternating_message(result, item)
    return result


# 计算单条中立消息的大致字符长度，供轮内高水位投影阈值判断
def _message_char_len(msg: Message) -> int:
    if isinstance(msg, UserMessage):
        return len(msg.content)
    if isinstance(msg, ToolResultMessage):
        return len(msg.content)
    total = 0
    for block in msg.blocks:
        if isinstance(block, TextBlock):
            total += len(block.text)
        elif isinstance(block, ThinkingBlock):
            total += len(block.text) + len(block.signature)
        elif isinstance(block, ToolCall):
            total += len(block.name) + len(block.arguments)
    return total


# 严格校验参数为非负整数（>= 0 且拒绝布尔值）
def _require_non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} 必须是非负整数。")
    return value


# 轮内粘性高水位非破坏性上下文投影器（接入 react.iter_react 的 convert_to_llm 钩子）：带动态滞回水位防抖，绝不放大原输出
class TurnContextProjector:
    # 初始化触发高水位字符阈值、动态下次触发水位、尾部原文保护条数与单条大输出折叠阈值
    def __init__(
        self,
        *,
        trigger_chars: int = DEFAULT_FOLD_TRIGGER_CHARS,
        keep_tail_messages: int = DEFAULT_FOLD_KEEP_TAIL_MESSAGES,
        min_result_chars: int = DEFAULT_FOLD_MIN_RESULT_CHARS,
        preview_chars: int = DEFAULT_FOLD_PREVIEW_CHARS,
    ) -> None:
        self._trigger_chars = require_positive_int(trigger_chars, "trigger_chars")
        self._next_trigger_chars = self._trigger_chars
        self._keep_tail_messages = _require_non_negative_int(keep_tail_messages, "keep_tail_messages")
        self._min_result_chars = _require_non_negative_int(min_result_chars, "min_result_chars")
        self._preview_chars = _require_non_negative_int(preview_chars, "preview_chars")
        self._folded_until = 0

    # 将单条超长 ToolResultMessage 确定性地投影为紧凑卡片（仅提取有界 TOAST 回读指针，并严格保证折叠后更短）
    def _fold_tool_result(self, msg: ToolResultMessage) -> ToolResultMessage:
        if len(msg.content) <= self._min_result_chars:
            return msg
        pointer_line = ""
        for line in msg.content.splitlines():
            if 'read_markdown(path="' in line:
                candidate = line.strip()
                if len(candidate) <= 320:
                    pointer_line = candidate
                break
        preview = " ".join(msg.content[: self._preview_chars].split())
        parts = [f"[已折叠工具输出：{msg.tool_name}，原长 {len(msg.content)} 字符，原文完整保留]"]
        if preview:
            parts.append(f"{preview}…")
        if pointer_line:
            parts.append(pointer_line)
        folded_content = "\n".join(parts)
        if len(folded_content) >= len(msg.content):
            return msg
        return ToolResultMessage(
            tool_call_id=msg.tool_call_id,
            tool_name=msg.tool_name,
            content=folded_content,
            is_error=msg.is_error,
        )

    # 按粘性边界与动态滞回水位投影当前轮 Context：不改写原始 history，无实际折叠不推边界，折叠后抬高下次触发水位防每步抖动
    def __call__(self, context: Context) -> Context:
        msgs = list(context.messages)
        mutated = False
        locked = min(self._folded_until, len(msgs))
        for idx in range(locked):
            item = msgs[idx]
            if isinstance(item, ToolResultMessage):
                folded = self._fold_tool_result(item)
                if folded is not item:
                    msgs[idx] = folded
                    mutated = True

        total_chars = sum(_message_char_len(m) for m in msgs)
        if total_chars > self._next_trigger_chars:
            cutoff = min(len(msgs), max(0, len(msgs) - self._keep_tail_messages))
            if cutoff > self._folded_until:
                newly_folded = False
                for idx in range(self._folded_until, cutoff):
                    item = msgs[idx]
                    if isinstance(item, ToolResultMessage):
                        folded = self._fold_tool_result(item)
                        if folded is not item:
                            msgs[idx] = folded
                            mutated = True
                            newly_folded = True
                if newly_folded:
                    self._folded_until = cutoff
                    post_fold_chars = sum(_message_char_len(m) for m in msgs)
                    hysteresis = max(1, self._trigger_chars // 2)
                    self._next_trigger_chars = max(self._trigger_chars, post_fold_chars + hysteresis)

        if not mutated:
            return context
        return Context(
            system_prompt=context.system_prompt,
            messages=tuple(msgs),
            tools=context.tools,
        )


# Layer 3 统一上下文装配器：单一持有会话前缀 TTL 冻结、每日过期归档失效、粘性历史切块与三层动静分离组装
class ContextAssembler:
    # 初始化记忆库引用、前缀空闲 TTL、轮内折叠高水位阈值、会话缓存上限与单调时钟
    def __init__(
        self,
        memory: MemoryStore,
        *,
        prompt_cache_ttl_seconds: float = DEFAULT_PROMPT_CACHE_TTL_SECONDS,
        max_turn_context_chars: int = DEFAULT_FOLD_TRIGGER_CHARS,
        max_sessions: int | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._memory = memory
        self.prompt_cache_ttl_seconds = require_non_negative_float(
            prompt_cache_ttl_seconds, "prompt_cache_ttl_seconds"
        )
        self.max_turn_context_chars = require_positive_int(
            max_turn_context_chars, "max_turn_context_chars"
        )
        self._max_sessions = (
            require_positive_int(max_sessions, "max_sessions") if max_sessions is not None else None
        )
        self._clock: Callable[[], float] = clock or time.monotonic
        self._lock = threading.Lock()
        self._session_prompts: dict[str, tuple[str, float]] = {}
        self._history_trim_floors: dict[str, int] = {}
        self._last_archive_day = ""

    # 解析当前生效的会话 LRU 容量上限（未显式指定时读取模块默认常量）
    def _session_cap(self) -> int:
        return self._max_sessions if self._max_sessions is not None else max(1, MAX_SESSION_PROMPTS)

    # 拼装纯静态/低频系统提示词（不含逐日日期与动态摘要），最大化前缀 KV Cache 复用
    def build_system_prompt(self, registry: ToolRegistry) -> str:
        return format_system_prompt(self._memory, tool_names=registry.names)

    # 获取指定会话冻结的系统提示词：热聊期（空闲 < TTL）死锁复用，空闲超时后零代价刷新最新记忆索引
    def get_session_system_prompt(self, chat_id: str, registry: ToolRegistry) -> str:
        cap = self._session_cap()
        with self._lock:
            now = self._clock()
            cached = self._session_prompts.pop(chat_id, None)
            if (
                cached is not None
                and self.prompt_cache_ttl_seconds > 0
                and (now - cached[1]) < self.prompt_cache_ttl_seconds
            ):
                prompt = cached[0]
            else:
                prompt = self.build_system_prompt(registry)
            while len(self._session_prompts) >= cap:
                self._session_prompts.pop(next(iter(self._session_prompts)))
            self._session_prompts[chat_id] = (prompt, now)
            return prompt

    # 主动重置指定会话（或全部会话）冻结的系统提示词与历史切块边界，用于沉淀改写索引或手动刷新
    def reset_session(self, chat_id: str | None = None) -> None:
        with self._lock:
            if chat_id is None:
                self._session_prompts.clear()
                self._history_trim_floors.clear()
            else:
                self._session_prompts.pop(chat_id, None)
                self._history_trim_floors.pop(chat_id, None)

    # 仅重置指定会话的历史切块游标（例如后台沉淀推进了压缩边界但未改写 MEMORY.md 时保留热 system_prompt）
    def clear_history_floor(self, chat_id: str) -> None:
        with self._lock:
            self._history_trim_floors.pop(chat_id, None)

    # 每个自然日首次对话时原子执行过期归档：持锁防止并发跨午夜读到过期旧前缀，并单调推进日期防迟到线程回退
    def maybe_archive_expired(self, today: date) -> dict[str, int]:
        today_str = today.isoformat()
        with self._lock:
            if self._last_archive_day >= today_str:
                return {}
            archived = self._memory.archive_expired_memories(today=today)
            if archived:
                self._session_prompts.clear()
            self._last_archive_day = max(self._last_archive_day, today_str)
            return archived

    # 按会话粘性边界裁剪活跃历史并按真 LRU 刷新命中会话热度（全程持锁避免并发后台回调清空 floor 时被旧结果覆写）
    def select_history(
        self,
        chat_id: str,
        active_all: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        cap = self._session_cap()
        with self._lock:
            existing = self._history_trim_floors.pop(chat_id, None)
            if existing is not None:
                self._history_trim_floors[chat_id] = existing
                floor_id = existing
            else:
                floor_id = 0
            selected, new_floor = select_chunked_history(active_all, floor_id=floor_id)
            if new_floor != floor_id:
                self._history_trim_floors.pop(chat_id, None)
                while len(self._history_trim_floors) >= cap:
                    self._history_trim_floors.pop(next(iter(self._history_trim_floors)))
                self._history_trim_floors[chat_id] = new_floor
            return selected

    # 按变化频率分层组装中立 Context：静态 system 前缀 -> <system-reminder> 隔离前情摘要 + 稳定历史 -> 带日期信封的当前提问
    def assemble(
        self,
        *,
        system_prompt: str,
        summary_text: str,
        history: list[dict[str, object]],
        user_text: str,
        registry: ToolRegistry,
        today: date | None = None,
    ) -> Context:
        messages: list[Message] = []
        rendered_summary = format_compaction_summary(summary_text)
        if rendered_summary:
            messages.append(UserMessage(content=rendered_summary))
        for msg in build_history_messages(history):
            append_alternating_message(messages, msg)
        envelope = format_user_envelope(user_text, today=today or today_shanghai())
        append_alternating_message(messages, UserMessage(content=envelope))
        return Context(
            system_prompt=system_prompt,
            messages=tuple(messages),
            tools=registry.schemas(),
        )

    # 为单轮 ReAct 循环构造独立非破坏性轮内上下文投影器
    def create_turn_projector(self) -> TurnContextProjector:
        return TurnContextProjector(trigger_chars=self.max_turn_context_chars)
