"""垂直领域 Embedding 微调三元组与 BM25 难负样本导出器（纯标准库，零第三方依赖）。"""
from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from personal_assistant.assistant.chunker import chunk_markdown_hierarchy
from personal_assistant.assistant.common import today_shanghai
from personal_assistant.assistant.memory import MemoryStore
from personal_assistant.assistant.rag_metadata import (
    MetadataFilter,
    parse_allowed_access,
    parse_markdown_frontmatter,
)
from evals.common import load_cases, normalize_expected_sources
from plugins.rag_retriever import RAGConfig, RAGRetriever

MIN_UNSUPERVISED_POSITIVE_CHARS = 24


# 校验输出文件路径严禁落在 data_dir 内部，避免导出训练集污染知识库目录
def _validate_output_path(output_path: Path, data_dir: Path) -> Path:
    resolved_out = output_path.resolve()
    resolved_data = data_dir.resolve()
    if resolved_out.is_relative_to(resolved_data):
        raise ValueError("微调训练集输出路径不能放在 --data-dir 目录内。")
    return resolved_out


# 为指定 (anchor, gold_paths) 挖掘同库 BM25 难负样本（优先取高词面重叠但非目标来源的干扰块，不足时跨文件兜底）
def _mine_hard_negatives(
    memory: MemoryStore,
    anchor: str,
    *,
    gold_paths: set[str],
    fallback_pool: list[tuple[str, str]],
    root: str,
    ref_day: date,
    meta_filter: MetadataFilter,
    max_negatives: int,
) -> tuple[list[tuple[str, str, float]], int]:
    negatives: list[tuple[str, str, float]] = []
    seen_texts: set[str] = set()
    hard_hits = 0

    try:
        candidates = memory.search_hits(
            anchor,
            root=root,
            limit=max(10, max_negatives * 6),
            today=ref_day,
            metadata_filter=meta_filter,
        )
    except ValueError:
        candidates = []

    for hit in candidates:
        if hit.path in gold_paths:
            continue
        clean_snip = " ".join(hit.snippet.split())
        if not clean_snip or clean_snip in seen_texts:
            continue
        seen_texts.add(clean_snip)
        negatives.append((hit.path, hit.snippet, float(hit.score)))
        hard_hits += 1
        if len(negatives) >= max_negatives:
            return negatives, hard_hits

    # BM25 未返回跨文件干扰时，从其他授权文档块按确定性顺序兜底补齐负样本
    for cand_path, cand_text in fallback_pool:
        if cand_path in gold_paths:
            continue
        clean_cand = " ".join(cand_text.split())
        if not clean_cand or clean_cand in seen_texts:
            continue
        seen_texts.add(clean_cand)
        negatives.append((cand_path, cand_text, 0.0))
        if len(negatives) >= max_negatives:
            break
    return negatives, hard_hits


# 基于业务 Markdown 目录（及可选标注问答 JSONL）自动导出标准 (anchor, positive, negative) 微调 JSONL
def export_embedding_triplets(
    data_dir: Path,
    *,
    root: str = "data/memory",
    cases_path: Path | None = None,
    output_path: Path | None = None,
    max_negatives_per_pair: int = 1,
    allowed_access: Sequence[str] = ("public",),
    today: date | None = None,
) -> dict[str, object]:
    if isinstance(max_negatives_per_pair, bool) or not isinstance(max_negatives_per_pair, int) or max_negatives_per_pair < 1:
        raise ValueError("max_negatives_per_pair 必须是正整数。")
    resolved_data = data_dir.resolve()
    resolved_out = _validate_output_path(output_path, resolved_data) if output_path is not None else None
    ref_day = today or today_shanghai()
    ref_day_iso = ref_day.isoformat()
    meta_filter = MetadataFilter(allowed_access=parse_allowed_access(allowed_access))

    # 使用纯内存 BM25 派生索引（零写盘污染 data_dir）进行难负样本挖掘
    retriever = RAGRetriever(RAGConfig(index_path=None))
    memory = MemoryStore(resolved_data, create_dirs=False, retriever=retriever)
    try:
        searchable, _ = memory.iter_searchable_files(root)
        active_chunks_by_path: dict[str, list[tuple[str, str]]] = {}
        fallback_pool: list[tuple[str, str]] = []
        files_scanned = 0

        for resolved, display_path, is_memory in searchable:
            try:
                raw_text = resolved.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            doc_meta, masked_text = parse_markdown_frontmatter(raw_text)
            if not meta_filter.matches(doc_meta):
                continue
            files_scanned += 1
            parents, children = chunk_markdown_hierarchy(
                display_path,
                masked_text,
                track_expiry=is_memory,
            )
            parent_sec = {p.parent_index: p.section for p in parents}
            for child in children:
                if child.until_date and child.until_date < ref_day_iso:
                    continue
                body = child.text.strip()
                if not body:
                    continue
                sec = parent_sec.get(child.parent_index, child.section)
                active_chunks_by_path.setdefault(display_path, []).append((sec, body))
                fallback_pool.append((display_path, body))

        triplets: list[dict[str, object]] = []
        hard_negative_hits = 0
        fallback_negatives = 0

        if cases_path is not None:
            cases = load_cases(cases_path.resolve())
            for case in cases:
                expected = normalize_expected_sources(memory, case.get("expected_sources"))
                if not expected or not bool(case.get("answerable", bool(expected))):
                    continue
                anchor = str(case["query"]).strip()
                gold_paths = {item["path"] for item in expected}
                for src in expected:
                    path = src["path"]
                    quote = src["quote"]
                    chunks = active_chunks_by_path.get(path, [])
                    positive = next((body for _, body in chunks if quote in body), "")
                    if not positive:
                        continue
                    negs, hard_cnt = _mine_hard_negatives(
                        memory,
                        anchor,
                        gold_paths=gold_paths,
                        fallback_pool=fallback_pool,
                        root=root,
                        ref_day=ref_day,
                        meta_filter=meta_filter,
                        max_negatives=max_negatives_per_pair,
                    )
                    hard_negative_hits += hard_cnt
                    fallback_negatives += len(negs) - hard_cnt
                    for neg_path, neg_text, neg_score in negs:
                        triplets.append(
                            {
                                "anchor": anchor,
                                "positive": positive,
                                "negative": neg_text,
                                "source": path,
                                "negative_source": neg_path,
                                "hard_negative_score": round(neg_score, 4),
                            }
                        )
        else:
            # 无标注自监督模式：用“文件名/章节标题”作查询锚点、非过期子块正文作正样本，并跨文件挖掘 BM25 难负样本
            for path, chunks in active_chunks_by_path.items():
                stem = Path(path).stem
                for section, body in chunks:
                    if len(body) < MIN_UNSUPERVISED_POSITIVE_CHARS:
                        continue
                    anchor = f"{stem} {section}".strip() if section else body.splitlines()[0][:80].strip()
                    if not anchor or anchor == body:
                        continue
                    negs, hard_cnt = _mine_hard_negatives(
                        memory,
                        anchor,
                        gold_paths={path},
                        fallback_pool=fallback_pool,
                        root=root,
                        ref_day=ref_day,
                        meta_filter=meta_filter,
                        max_negatives=max_negatives_per_pair,
                    )
                    hard_negative_hits += hard_cnt
                    fallback_negatives += len(negs) - hard_cnt
                    for neg_path, neg_text, neg_score in negs:
                        triplets.append(
                            {
                                "anchor": anchor,
                                "positive": body,
                                "negative": neg_text,
                                "source": path,
                                "negative_source": neg_path,
                                "hard_negative_score": round(neg_score, 4),
                            }
                        )

        if resolved_out is not None:
            resolved_out.parent.mkdir(parents=True, exist_ok=True)
            payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in triplets)
            resolved_out.write_text((payload + "\n") if payload else "", encoding="utf-8")

        return {
            "mode": "export-triplets",
            "root": root,
            "files_scanned": files_scanned,
            "triplets_exported": len(triplets),
            "hard_negative_hits": hard_negative_hits,
            "fallback_negatives": fallback_negatives,
            "allowed_access": list(meta_filter.allowed_access),
            "output_path": str(resolved_out) if resolved_out is not None else None,
        }
    finally:
        memory.close()
