"""独立外部 Embedding 领域微调脚本模板（按需在有 GPU 环境运行，不引入主包运行依赖）。

用法示例：
  1) 先用纯标准库导出带 BM25 难负样本的三元组：
     uv run python -m evals --mode export-triplets --data-dir data --output /tmp/triplets.jsonl
  2) 在安装有 sentence-transformers 的训练环境运行本脚本微调 BGE / E5 等模型：
     python scripts/finetune_embedding.py --triplets /tmp/triplets.jsonl \\
       --base-model BAAI/bge-small-zh-v1.5 --output-dir /tmp/my-bge-finetuned
  3) 将微调产物挂到 Ollama / vLLM / TEI 的 /v1/embeddings，并在 config/models.dsl 绑定：
     embedder my_bge_finetuned
  4) 用离线评测对比微调前后 Recall / MRR：
     uv run python -m evals --mode search --retriever rag --embedder my_bge_finetuned
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


# 读取由 `evals --mode export-triplets` 导出的标准 JSONL 三元组文件
def load_triplets(path: Path) -> list[tuple[str, str, str]]:
    items: list[tuple[str, str, str]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw.strip()
        if not text:
            continue
        row = json.loads(text)
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{line_no} 必须是 JSON 对象。")
        anchor = str(row.get("anchor") or "").strip()
        positive = str(row.get("positive") or "").strip()
        negative = str(row.get("negative") or "").strip()
        if not anchor or not positive or not negative:
            raise ValueError(f"{path.name}:{line_no} 缺少非空 anchor/positive/negative 字段。")
        items.append((anchor, positive, negative))
    if not items:
        raise ValueError("三元组训练集为空。")
    return items


# 解析命令行参数并执行 sentence-transformers MultipleNegativesRankingLoss 领域微调
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="使用导出的 (anchor, positive, negative) 三元组微调垂直领域 Embedding 模型"
    )
    parser.add_argument("--triplets", required=True, help="JSONL 三元组路径（由 evals --mode export-triplets 生成）")
    parser.add_argument("--base-model", default="BAAI/bge-small-zh-v1.5", help="基座向量模型名称或本地路径")
    parser.add_argument("--output-dir", required=True, help="微调后模型保存目录")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数（默认 3）")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小（默认 32）")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率（默认 2e-5）")
    args = parser.parse_args(argv)
    if args.epochs < 1:
        raise ValueError("--epochs 必须是正整数。")
    if args.batch_size < 1:
        raise ValueError("--batch-size 必须是正整数。")
    if not math.isfinite(args.lr) or args.lr <= 0.0:
        raise ValueError("--lr 必须是有限正浮点数。")

    triplets = load_triplets(Path(args.triplets).resolve())
    try:
        from datasets import Dataset
        from sentence_transformers import (
            SentenceTransformer,
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
        )
        from sentence_transformers.losses import MultipleNegativesRankingLoss
    except ImportError:
        print(
            "未检测到外部训练依赖。请在独立训练环境安装后重试：\n"
            "  pip install sentence-transformers datasets accelerate\n"
            f"（已校验训练集：{len(triplets)} 条三元组格式合法）",
            file=sys.stderr,
        )
        return 1

    dataset = Dataset.from_dict(
        {
            "anchor": [a for a, _, _ in triplets],
            "positive": [p for _, p, _ in triplets],
            "negative": [n for _, _, n in triplets],
        }
    )
    model = SentenceTransformer(args.base_model)
    loss = MultipleNegativesRankingLoss(model)
    train_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        save_strategy="epoch",
        logging_steps=10,
    )
    trainer = SentenceTransformerTrainer(model=model, args=train_args, train_dataset=dataset, loss=loss)
    trainer.train()
    model.save_pretrained(args.output_dir)
    print(f"微调完成！模型已保存至：{args.output_dir}")
    print("后续接入步骤：")
    print("  1. 使用 Ollama / vLLM / TEI 暴露该模型的 /v1/embeddings 接口；")
    print("  2. 在 config/models.dsl 绑定 `embedder <你的模型名>`；")
    print("  3. 运行 `uv run python -m evals --mode search --retriever rag --embedder <你的模型名>` 对比 Recall / MRR。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
