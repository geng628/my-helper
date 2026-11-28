"""RAG 文档元数据解析与前置硬过滤模型（零第三方依赖）：支持 YAML Frontmatter 提取、时间/产品/部门/标签过滤与系统级权限防提级。"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from personal_assistant.assistant.common import parse_iso_date

MAX_FRONTMATTER_LINES = 64
_KEY_VAL_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.*)$")
_LIST_ITEM_RE = re.compile(r"^[ \t]*-[ \t]+(.+)$")
_FILTER_SPLIT_RE = re.compile(r"[\n;；]+")
_ALLOWED_FILTER_KEYS = frozenset({"date_from", "date_to", "product", "department", "tags", "access"})


# 规范化单个标量元数据值（剥离首尾空白与装饰引号并转小写）
def _norm_scalar(raw: str) -> str:
    return raw.strip().strip("\"'").strip().lower()


# 解析单个 YAML 列表项或逗号分隔字符串为去重排序后的小写标签元组（剔除逗号确保 CSV 边界匹配无歧义）
def _norm_tags(items: Sequence[str]) -> tuple[str, ...]:
    cleaned: set[str] = set()
    for raw in items:
        for part in str(raw).split(","):
            tok = part.strip().strip("\"'").strip().lower().replace(",", "")
            if tok:
                cleaned.add(tok)
    return tuple(sorted(cleaned))


# 将 YYYY-MM-DD 或 ISO 时间戳前缀解析为标准 ISO 日期串；空串返回 ""，非法格式返回 ""（文档容错）或抛 ValueError（过滤严格校验）
def _parse_date_field(raw: str, *, strict: bool) -> str:
    text = raw.strip().strip("\"'").strip()
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


# 校验并归一化允许访问权限集合（默认 ("public",)，拒绝布尔值与非字符串项）
def parse_allowed_access(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ("public",)
    if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
        raise ValueError("allowed_access 必须是非空字符串列表（如 ['public']）。")
    picked: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("allowed_access 中的每一项必须是非空字符串。")
        norm = item.strip().lower()
        if norm not in picked:
            picked.append(norm)
    return tuple(picked)


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


# 解析 Markdown 顶部 YAML Frontmatter 并返回 (DocumentMetadata, 等行号屏蔽正文)；非 Frontmatter（如顶部分割线）原样返回
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
        stripped = lines[idx].strip()
        if stripped in {"---", "..."}:
            closing_idx = idx
            break
    if closing_idx == -1:
        return DocumentMetadata(), text

    scalars: dict[str, str] = {}
    lists: dict[str, list[str]] = {}
    current_list_key: str | None = None

    for raw_line in lines[1:closing_idx]:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        list_m = _LIST_ITEM_RE.match(raw_line)
        if list_m is not None and current_list_key is not None:
            lists.setdefault(current_list_key, []).append(list_m.group(1).strip())
            continue
        kv_m = _KEY_VAL_RE.match(raw_line)
        if kv_m is None:
            # 出现非 YAML 键值行的普通正文，说明顶部 `---` 仅为水平分割线，安全回退为无 Frontmatter
            return DocumentMetadata(), text
        key = kv_m.group(1).strip().lower()
        val = kv_m.group(2).strip()
        if not val:
            current_list_key = key
            lists.setdefault(key, [])
        else:
            current_list_key = None
            if val.startswith("[") and val.endswith("]"):
                inner = val[1:-1].strip()
                lists[key] = [item.strip() for item in inner.split(",") if item.strip()]
            else:
                scalars[key] = val

    doc_date = ""
    for date_key in ("date", "updated_at", "created_at"):
        if date_key in scalars:
            doc_date = _parse_date_field(scalars[date_key], strict=False)
            if doc_date:
                break

    product = _norm_scalar(scalars.get("product", ""))
    department = _norm_scalar(scalars.get("department", ""))
    raw_access = scalars.get("access") or scalars.get("permission") or scalars.get("visibility") or "public"
    access = _norm_scalar(raw_access) or "public"

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
        if not val:
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
            tags_raw.append(val)
        elif key == "access":
            for acc_part in val.split(","):
                acc = acc_part.strip().lower()
                if not acc or acc == "*":
                    raise ValueError("过滤条件中的 access 不允许为空或通配符 '*'。")
                if acc not in requested_access:
                    requested_access.append(acc)

    # 权限取交集：仅允许在系统授予的 allowed_access 范围内收窄，绝对杜绝模型或用户借 filters 越权提级
    if requested_access:
        if "*" in ceiling:
            effective_access = tuple(requested_access)
        else:
            ceiling_set = set(ceiling)
            effective_access = tuple(acc for acc in requested_access if acc in ceiling_set)
    else:
        effective_access = ceiling

    return MetadataFilter(
        date_from=date_from,
        date_to=date_to,
        product=product,
        department=department,
        tags=_norm_tags(tags_raw),
        allowed_access=effective_access,
    )
