"""RAG 文档元数据解析与前置硬过滤模型（零第三方依赖）：支持 YAML Frontmatter 提取、时间/产品/部门/标签过滤与系统级权限防提级。"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from personal_assistant.assistant.common import parse_iso_date

MAX_FRONTMATTER_LINES = 256
_RESTRICTED_FALLBACK_ACCESS = "__restricted__"
_KEY_VAL_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.-]*)\s*:\s*(.*)$")
_LIST_ITEM_RE = re.compile(r"^[ \t]*-[ \t]*(.*)$")
_FILTER_SPLIT_RE = re.compile(r"[\n;；]+")
_ALLOWED_FILTER_KEYS = frozenset({"date_from", "date_to", "product", "department", "tags", "access"})
_ACCESS_KEYS = ("access", "permission", "visibility")


# 规范化单个标量元数据值（剥离首尾空白与成对装饰引号并转小写）
def _norm_scalar(raw: str) -> str:
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1]
    return text.strip().lower()


# 解析单个 YAML 列表项或逗号分隔字符串为去重排序后的小写标签元组（剔除逗号确保 CSV 边界匹配无歧义）
def _norm_tags(items: Sequence[str]) -> tuple[str, ...]:
    cleaned: set[str] = set()
    for raw in items:
        for part in str(raw).split(","):
            tok = _norm_scalar(part).replace(",", "")
            if tok:
                cleaned.add(tok)
    return tuple(sorted(cleaned))


# 将 YYYY-MM-DD 或 ISO 时间戳前缀解析为标准 ISO 日期串；空串返回 ""，非法格式返回 ""（文档容错）或抛 ValueError（过滤严格校验）
def _parse_date_field(raw: str, *, strict: bool) -> str:
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    if not text:
        return ""
    candidate = text[:10] if len(text) >= 10 else text
    try:
        parsed = date.fromisoformat(candidate)
    except ValueError as error:
        if strict:
            raise ValueError(f"日期格式不合法（应为 YYYY-MM-DD）：{raw!r}") from error
        parsed_opt = parse_iso_date(text)
        return parsed_opt.isoformat() if parsed_opt is not None else ""
    return parsed.isoformat()


# 校验并归一化允许访问权限集合（默认 ("public",)，允许空列表 [] 表示全拒绝，拒绝布尔值与非字符串项）
def parse_allowed_access(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ("public",)
    if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
        raise ValueError("allowed_access 必须是字符串列表（如 ['public']）。")
    picked: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("allowed_access 中的每一项必须是非空字符串。")
        norm = item.strip().lower()
        if norm not in picked:
            picked.append(norm)
    return tuple(picked)


# 从解析出的 Frontmatter 标量与列表字典中按 Fail-Closed 原则提取访问权限（显式声明非空/列表/空权限绝不静默降级为 public）
def _extract_access_level(scalars: dict[str, str], lists: dict[str, list[str]]) -> str:
    declared_any = False
    for key in _ACCESS_KEYS:
        if key in scalars:
            declared_any = True
            val = _norm_scalar(scalars[key])
            if val:
                return val
        if key in lists:
            declared_any = True
            vals = _norm_tags(lists[key])
            if len(vals) == 1:
                return vals[0]
            if len(vals) > 1:
                return _RESTRICTED_FALLBACK_ACCESS
    return _RESTRICTED_FALLBACK_ACCESS if declared_any else "public"


# 将单行或跨行方括号列表片段拆分为条目列表（遇 ']' 即闭合，自动忽略闭合符后的 YAML 行内注释如 `] # comment`）
def _parse_bracket_list_segment(segment: str) -> tuple[list[str], bool]:
    close_idx = segment.find("]")
    if close_idx != -1:
        body = segment[:close_idx].strip()
        items = [item.strip() for item in body.split(",") if item.strip()]
        return items, True
    items = [item.strip() for item in segment.strip().split(",") if item.strip()]
    return items, False


# 单篇 Markdown 文档顶部提取出的结构化元数据（默认 access='public'，确保无头部旧文档零破坏）
@dataclass(frozen=True)
class DocumentMetadata:
    doc_date: str = ""
    product: str = ""
    department: str = ""
    access: str = "public"
    tags: tuple[str, ...] = ()
    body_start_line: int = 1

    # 导出逗号包裹的标签序列化串（供 SQLite INSTR(',tag,') 精确前置过滤，杜绝子串误中）
    @property
    def tags_csv(self) -> str:
        return f",{','.join(self.tags)}," if self.tags else ""

    # 判断文档是否显式声明了任意非默认元数据（供 /rag status 统计）
    @property
    def has_metadata(self) -> bool:
        return bool(self.doc_date or self.product or self.department or self.tags or self.access != "public")


# 解析 Markdown 顶部 YAML Frontmatter 并返回 (DocumentMetadata, 等行号屏蔽正文)；支持嵌套/块标量/多行列表并对权限声明严格 Fail-Closed
def parse_markdown_frontmatter(text: str) -> tuple[DocumentMetadata, str]:
    lines = text.splitlines()
    if len(lines) < 2:
        return DocumentMetadata(), text
    first = lines[0].lstrip("\ufeff").strip()
    if first != "---":
        return DocumentMetadata(), text

    closing_idx = -1
    limit = min(len(lines), MAX_FRONTMATTER_LINES + 1)
    for idx in range(1, limit):
        if lines[idx].strip() in {"---", "..."}:
            closing_idx = idx
            break

    # 若未在扫描上限内找到闭合符，检查是否属于超长未闭合 Frontmatter 且声明了受限权限（Fail-Closed，绝不降级 public）
    if closing_idx == -1:
        saw_first_kv = False
        unclosed_scalars: dict[str, str] = {}
        unclosed_lists: dict[str, list[str]] = {}
        cur_k: str | None = None
        for raw_line in lines[1:limit]:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not saw_first_kv:
                kv0 = _KEY_VAL_RE.match(raw_line)
                if kv0 is None:
                    return DocumentMetadata(), text
                saw_first_kv = True
            if raw_line[0] in {" ", "\t"}:
                lm = _LIST_ITEM_RE.match(raw_line)
                if lm is not None and cur_k in _ACCESS_KEYS and lm.group(1).strip():
                    unclosed_lists.setdefault(cur_k, []).append(lm.group(1).strip())
                continue
            kv = _KEY_VAL_RE.match(raw_line)
            if kv is None:
                continue
            cur_k = kv.group(1).strip().lower()
            if cur_k in _ACCESS_KEYS:
                val = kv.group(2).strip()
                if val.startswith("["):
                    items, _ = _parse_bracket_list_segment(val[1:])
                    unclosed_lists.setdefault(cur_k, []).extend(items)
                elif not val:
                    unclosed_lists.setdefault(cur_k, [])
                else:
                    unclosed_scalars[cur_k] = val
        extracted = _extract_access_level(unclosed_scalars, unclosed_lists)
        if extracted != "public":
            return DocumentMetadata(access=extracted), text
        return DocumentMetadata(), text

    scalars: dict[str, str] = {}
    lists: dict[str, list[str]] = {}
    saw_key = False
    current_key: str | None = None
    in_block_scalar = False
    multiline_bracket_key: str | None = None

    for raw_line in lines[1:closing_idx]:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if multiline_bracket_key is not None:
            # 纵深防御：若遇到顶格新键值对且非 `]` 闭合行（如未闭合的 `title: [Draft Q4`），立即结束列表态以防吞噬后续 `access` 声明
            if raw_line[0] not in {" ", "\t"} and not stripped.startswith("]") and _KEY_VAL_RE.match(raw_line) is not None:
                multiline_bracket_key = None
            else:
                items, closed = _parse_bracket_list_segment(stripped)
                lists.setdefault(multiline_bracket_key, []).extend(items)
                if closed:
                    multiline_bracket_key = None
                continue
        if raw_line[0] in {" ", "\t"}:
            if not saw_key:
                return DocumentMetadata(), text
            if in_block_scalar:
                continue
            list_m = _LIST_ITEM_RE.match(raw_line)
            if list_m is not None and current_key is not None:
                item_val = list_m.group(1).strip()
                if item_val:
                    lists.setdefault(current_key, []).append(item_val)
            # 其余缩进行属于嵌套 YAML 映射（如 meta:\n  owner: hr），安全跳过而不丢弃已解析权限
            continue

        in_block_scalar = False
        zero_list = _LIST_ITEM_RE.match(raw_line)
        if zero_list is not None and current_key is not None:
            item_val = zero_list.group(1).strip()
            if item_val:
                lists.setdefault(current_key, []).append(item_val)
            continue

        kv_m = _KEY_VAL_RE.match(raw_line)
        if kv_m is None:
            # 若尚未解析出任何合法 YAML 键，判定为顶部 `---` 水平分割线并原样返回；若已解析出键值，禁止 Fail-Open 丢弃已有权限
            if not saw_key:
                return DocumentMetadata(), text
            continue

        saw_key = True
        key = kv_m.group(1).strip().lower()
        val = kv_m.group(2).strip()
        current_key = key
        if val in {"|", ">", "|-", ">-", "|+", ">+"}:
            in_block_scalar = True
        elif not val:
            lists.setdefault(key, [])
        elif val.startswith("["):
            close_idx = val.find("]", 1)
            if close_idx != -1:
                tail = val[close_idx + 1 :].strip()
                if not tail or tail.startswith("#"):
                    items, _ = _parse_bracket_list_segment(val[1 : close_idx + 1])
                    lists.setdefault(key, []).extend(items)
                else:
                    # 如 `title: [Internal] Q4 Salary` 属于带方括号前缀的普通标量文本，绝不进入多行列表态
                    scalars[key] = val
            else:
                items, _ = _parse_bracket_list_segment(val[1:])
                lists.setdefault(key, []).extend(items)
                multiline_bracket_key = key
        else:
            scalars[key] = val

    if not saw_key:
        return DocumentMetadata(), text

    doc_date = ""
    for date_key in ("date", "updated_at", "created_at"):
        if date_key in scalars:
            doc_date = _parse_date_field(scalars[date_key], strict=False)
            if doc_date:
                break

    product = _norm_scalar(scalars.get("product", ""))
    department = _norm_scalar(scalars.get("department", ""))
    access = _extract_access_level(scalars, lists)

    raw_tags: list[str] = list(lists.get("tags", []))
    if "tags" in scalars:
        raw_tags.append(scalars["tags"])
    tags = _norm_tags(raw_tags)

    body_offset = closing_idx + 1
    if body_offset >= len(lines):
        masked_text = ""
    else:
        # 用等量前导空行替代 Frontmatter 行，既防止元数据词污染正文 BM25/向量，又确保分块行号与磁盘文件 1:1 精确对齐
        masked_text = ("\n" * body_offset) + "\n".join(lines[body_offset:])

    meta = DocumentMetadata(
        doc_date=doc_date,
        product=product,
        department=department,
        access=access,
        tags=tags,
        body_start_line=body_offset + 1,
    )
    return meta, masked_text


# 检索前置元数据硬过滤条件：在 BM25 与稠密向量相似度计算前过滤时间范围、产品线、部门、标签与权限边界
@dataclass(frozen=True)
class MetadataFilter:
    date_from: str = ""
    date_to: str = ""
    product: str = ""
    department: str = ""
    tags: tuple[str, ...] = ()
    allowed_access: tuple[str, ...] = ("public",)

    # 归一化并严格校验日期范围、标签与权限集合
    def __post_init__(self) -> None:
        d_from = _parse_date_field(self.date_from, strict=True) if self.date_from else ""
        d_to = _parse_date_field(self.date_to, strict=True) if self.date_to else ""
        if d_from and d_to and d_from > d_to:
            raise ValueError(f"时间过滤范围非法：date_from ({d_from}) 不能晚于 date_to ({d_to})。")
        norm_prod = _norm_scalar(self.product)
        norm_dept = _norm_scalar(self.department)
        norm_tags = _norm_tags(self.tags)
        norm_acc = parse_allowed_access(self.allowed_access)
        object.__setattr__(self, "date_from", d_from)
        object.__setattr__(self, "date_to", d_to)
        object.__setattr__(self, "product", norm_prod)
        object.__setattr__(self, "department", norm_dept)
        object.__setattr__(self, "tags", norm_tags)
        object.__setattr__(self, "allowed_access", norm_acc)

    # 按系统级权限上限钳制当前过滤器的 allowed_access（严格取交集，杜绝显式入参越权突破应用权限边界）
    def clamp_access(self, ceiling: Sequence[str] | None) -> "MetadataFilter":
        norm_ceiling = parse_allowed_access(ceiling)
        if "*" in norm_ceiling:
            return self
        if self.allows_all_access:
            clamped = norm_ceiling
        else:
            ceiling_set = set(norm_ceiling)
            clamped = tuple(acc for acc in self.allowed_access if acc in ceiling_set)
        return MetadataFilter(
            date_from=self.date_from,
            date_to=self.date_to,
            product=self.product,
            department=self.department,
            tags=self.tags,
            allowed_access=clamped,
        )

    # 判断权限集合是否为空（不可满足时直接短路返回空结果，零 SQL 与零向量开销）
    @property
    def is_unsatisfiable(self) -> bool:
        return len(self.allowed_access) == 0

    # 判断是否包含管理员全权通配符 '*'
    @property
    def allows_all_access(self) -> bool:
        return "*" in self.allowed_access

    # 判断是否存在除默认 public 权限外的显式筛选条件
    @property
    def has_explicit_criteria(self) -> bool:
        return bool(self.date_from or self.date_to or self.product or self.department or self.tags)

    # 内存端逐文档判定（供 LinearRetriever 与单测复用，与 SQLite WHERE 子句保持 100% 等价语义）
    def matches(self, meta: DocumentMetadata) -> bool:
        if self.is_unsatisfiable:
            return False
        if not self.allows_all_access and meta.access not in self.allowed_access:
            return False
        if self.date_from and (not meta.doc_date or meta.doc_date < self.date_from):
            return False
        if self.date_to and (not meta.doc_date or meta.doc_date > self.date_to):
            return False
        if self.product and meta.product != self.product:
            return False
        if self.department and meta.department != self.department:
            return False
        if self.tags:
            doc_tag_set = set(meta.tags)
            if any(tag not in doc_tag_set for tag in self.tags):
                return False
        return True

    # 生成参数化 SQLite WHERE 前置过滤片段（用于在 BM25 与余弦相似度计算前硬筛除越权与不匹配候选）
    def to_sql_clause(self, alias: str = "c") -> tuple[str, list[object]]:
        if self.is_unsatisfiable:
            return "1 = 0", []
        clauses: list[str] = []
        params: list[object] = []
        if not self.allows_all_access:
            placeholders = ", ".join("?" for _ in self.allowed_access)
            clauses.append(f"{alias}.access_level IN ({placeholders})")
            params.extend(self.allowed_access)
        if self.date_from:
            clauses.append(f"({alias}.doc_date != '' AND {alias}.doc_date >= ?)")
            params.append(self.date_from)
        if self.date_to:
            clauses.append(f"({alias}.doc_date != '' AND {alias}.doc_date <= ?)")
            params.append(self.date_to)
        if self.product:
            clauses.append(f"{alias}.product = ?")
            params.append(self.product)
        if self.department:
            clauses.append(f"{alias}.department = ?")
            params.append(self.department)
        for tag in self.tags:
            clauses.append(f"INSTR({alias}.tags_csv, ?) > 0")
            params.append(f",{tag},")
        if not clauses:
            return "1 = 1", []
        return " AND ".join(clauses), params

    # 生成人类可读的紧凑过滤摘要（供 CLI /rag query 展示）
    def describe(self) -> str:
        parts: list[str] = []
        if self.date_from or self.date_to:
            parts.append(f"date={self.date_from or '*'}..{self.date_to or '*'}")
        if self.product:
            parts.append(f"product={self.product}")
        if self.department:
            parts.append(f"department={self.department}")
        if self.tags:
            parts.append(f"tags={','.join(self.tags)}")
        if not self.allows_all_access:
            parts.append(f"access={'|'.join(self.allowed_access) or 'NONE'}")
        return "; ".join(parts)


# 解析工具或命令行传入的紧凑过滤串（如 "product=billing;date_from=2026-01-01;tags=退款,年卡"），并严格钳制在系统允许权限内防提级
def parse_filter_string(
    raw: str,
    *,
    allowed_access: Sequence[str] | None = None,
) -> MetadataFilter:
    ceiling = parse_allowed_access(allowed_access)
    if not isinstance(raw, str):
        raise ValueError("filters 必须是字符串（如 product=billing;date_from=2026-01-01）。")
    text = raw.strip()
    if not text:
        return MetadataFilter(allowed_access=ceiling)

    date_from = ""
    date_to = ""
    product = ""
    department = ""
    tags_raw: list[str] = []
    requested_access: list[str] = []
    seen_scalar_keys: set[str] = set()

    for token in _FILTER_SPLIT_RE.split(text):
        piece = token.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"过滤条件缺少 '='：{piece!r}（示例：product=billing;date_from=2026-01-01）。")
        key_raw, val_raw = piece.split("=", 1)
        key = key_raw.strip().lower()
        val = val_raw.strip()
        if key not in _ALLOWED_FILTER_KEYS:
            allowed_list = ", ".join(sorted(_ALLOWED_FILTER_KEYS))
            raise ValueError(f"不支持的过滤字段 {key!r}，仅支持：{allowed_list}。")
        if key in {"date_from", "date_to", "product", "department"}:
            if key in seen_scalar_keys:
                raise ValueError(f"过滤字段 {key!r} 不能重复指定。")
            seen_scalar_keys.add(key)
        if not _norm_scalar(val):
            raise ValueError(f"过滤字段 {key!r} 的值不能为空。")
        if key == "date_from":
            date_from = val
        elif key == "date_to":
            date_to = val
        elif key == "product":
            product = val
        elif key == "department":
            department = val
        elif key == "tags":
            parsed_tags = _norm_tags([val])
            if not parsed_tags:
                raise ValueError("过滤字段 'tags' 至少需要一个非空标签。")
            tags_raw.extend(parsed_tags)
        elif key == "access":
            part_added = False
            for acc_part in val.split(","):
                acc = _norm_scalar(acc_part)
                if not acc or acc == "*":
                    raise ValueError("过滤条件中的 access 不允许为空或通配符 '*'。")
                part_added = True
                if acc not in requested_access:
                    requested_access.append(acc)
            if not part_added:
                raise ValueError("过滤条件中的 access 不能为空。")

    base = MetadataFilter(
        date_from=date_from,
        date_to=date_to,
        product=product,
        department=department,
        tags=tuple(tags_raw),
        allowed_access=tuple(requested_access) if requested_access else ("*",),
    )
    return base.clamp_access(ceiling)
