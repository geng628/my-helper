"""一键下载并转换公开学术评测集（LongMemEval / LoCoMo）到 gitignore 保护的 data/evals/ 目录。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from personal_assistant.eval.public_benchmarks import (  # noqa: E402
    LOCOMO10_URL,
    LONGMEMEVAL_ORACLE_URL,
    convert_locomo_for_memory,
    convert_longmemeval_dataset,
    download_dataset,
    export_locomo_for_search,
    write_jsonl,
)


# 校验输出目录必须位于仓库外的自定义路径或 data/ 内，避免把大型第三方数据集误提交到 Git
def _validate_out_dir(out_dir: Path) -> Path:
    resolved = out_dir.resolve()
    evals_tracked = (REPO_ROOT / "evals").resolve()
    if resolved == evals_tracked or resolved.is_relative_to(evals_tracked):
        raise ValueError("公开大型数据集禁止直接写入受 Git 追踪的 evals/ 目录，请使用 data/evals。")
    return resolved


# 解析命令行参数，按需下载并生成 LongMemEval 与 LoCoMo 的本地评测集
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="下载并转换 LongMemEval（ICLR 2025）与 LoCoMo（ACL 2024）公开评测集到 data/evals/"
    )
    parser.add_argument(
        "--dataset",
        choices=["longmemeval", "locomo", "all"],
        default="all",
        help="要下载转换的数据集（默认 all）",
    )
    parser.add_argument("--out-dir", default="data/evals", help="输出目录（默认 data/evals，已由 .gitignore 忽略）")
    parser.add_argument("--longmemeval-limit", type=int, default=35, help="LongMemEval 分层抽样题数（0 表示全量 500 题）")
    parser.add_argument("--locomo-search-limit", type=int, default=200, help="LoCoMo 离线检索抽样题数（0 表示全量）")
    parser.add_argument("--locomo-memory-limit", type=int, default=25, help="LoCoMo 多轮记忆抽样题数（0 表示全量）")
    parser.add_argument("--seed", type=int, default=42, help="分层抽样随机种子")
    parser.add_argument("--force", action="store_true", help="强制重新下载原始 JSON 文件")
    args = parser.parse_args(argv)

    out_dir = _validate_out_dir(Path(args.out_dir))
    raw_dir = out_dir / "raw"

    if args.dataset in {"longmemeval", "all"}:
        raw_lme = download_dataset(
            LONGMEMEVAL_ORACLE_URL, raw_dir / "longmemeval_oracle.json", force=args.force
        )
        raw_items = json.loads(raw_lme.read_text(encoding="utf-8"))
        limit = None if args.longmemeval_limit == 0 else args.longmemeval_limit
        scenarios = convert_longmemeval_dataset(raw_items, limit=limit, seed=args.seed)
        lme_out = out_dir / "longmemeval_scenarios.jsonl"
        write_jsonl(lme_out, scenarios)
        dist = dict(sorted(Counter(s["category"] for s in scenarios).items()))
        print(f"[LongMemEval] 原始 {len(raw_items)} 题 -> 导出 {len(scenarios)} 题到 {lme_out}")
        print(f"  分类分布: {dist}")
        print(f"  运行命令: uv run python -m personal_assistant.eval --mode memory --cases {lme_out} --max-cases 14 --output /tmp/eval-lme.json")

    if args.dataset in {"locomo", "all"}:
        raw_loc = download_dataset(LOCOMO10_URL, raw_dir / "locomo10.json", force=args.force)
        raw_convs = json.loads(raw_loc.read_text(encoding="utf-8"))

        search_data_dir = out_dir / "locomo_search_data"
        search_limit = None if args.locomo_search_limit == 0 else args.locomo_search_limit
        search_cases = export_locomo_for_search(
            raw_convs, search_data_dir, limit=search_limit, seed=args.seed
        )
        search_out = out_dir / "locomo_search_cases.jsonl"
        write_jsonl(search_out, search_cases)
        s_dist = dict(sorted(Counter(c["category"] for c in search_cases).items()))
        print(f"[LoCoMo-Search] 导出 {len(search_cases)} 题检索集到 {search_out}（语料：{search_data_dir}）")
        print(f"  分类分布: {s_dist}")
        print(
            f"  运行命令: uv run python -m personal_assistant.eval --mode search "
            f"--data-dir {search_data_dir} --cases {search_out} --output /tmp/eval-locomo-search.json"
        )

        mem_limit = None if args.locomo_memory_limit == 0 else args.locomo_memory_limit
        mem_scenarios = convert_locomo_for_memory(raw_convs, limit=mem_limit, seed=args.seed)
        mem_out = out_dir / "locomo_memory_scenarios.jsonl"
        write_jsonl(mem_out, mem_scenarios)
        m_dist = dict(sorted(Counter(s["category"] for s in mem_scenarios).items()))
        print(f"[LoCoMo-Memory] 导出 {len(mem_scenarios)} 题多轮记忆场景到 {mem_out}")
        print(f"  分类分布: {m_dist}")
        print(
            f"  运行命令: uv run python -m personal_assistant.eval --mode memory "
            f"--cases {mem_out} --max-cases 10 --output /tmp/eval-locomo-memory.json"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
