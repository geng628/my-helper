"""公开学术评测集（LongMemEval / LoCoMo）离线适配器单元测试（TDD）。"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from evals import main as eval_main
from evals.memory_eval import load_scenarios
from evals.public_benchmarks import (
    convert_locomo_for_memory,
    convert_longmemeval_dataset,
    convert_longmemeval_item,
    export_locomo_for_search,
    parse_locomo_date,
    parse_longmemeval_datetime,
    prepare_public_datasets,
    validate_public_out_dir,
    write_jsonl,
    write_scenarios_jsonl,
)
from evals.search_eval import evaluate_search
from personal_assistant.assistant.memory import MemoryStore


# 测试 LongMemEval 与 LoCoMo 历史日期字符串解析准确性
def test_public_benchmark_date_parsers() -> None:
    dt = parse_longmemeval_datetime("2023/04/10 (Mon) 17:50")
    assert dt.year == 2023 and dt.month == 4 and dt.day == 10
    assert dt.hour == 17 and dt.minute == 50

    assert parse_locomo_date("1:56 pm on 8 May, 2023") == date(2023, 5, 8)
    assert parse_locomo_date("10:05 am on 21 October, 2023") == date(2023, 10, 21)
    assert parse_locomo_date("invalid date") is None


# 测试 LongMemEval 单条转换：按时间戳重排乱序会话、生成 seed_history+consolidate+ask 并正确识别拒答题
def test_convert_longmemeval_item_sorts_sessions_and_detects_abstention() -> None:
    raw_normal = {
        "question_id": "q_ku_1",
        "question_type": "knowledge-update",
        "question": "What is my 5K personal best?",
        "answer": "25:50",
        "question_date": "2023/06/01 (Thu) 09:00",
        "haystack_dates": [
            "2023/05/27 (Sat) 10:20",
            "2023/05/25 (Thu) 20:21",
        ],
        "haystack_session_ids": ["s_later", "s_earlier"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I beat my record today with 25:50!"},
                {"role": "assistant", "content": "Awesome 25:50 PR!"},
            ],
            [
                {"role": "user", "content": "My 5K personal best is 27:10."},
                {"role": "assistant", "content": "Nice 27:10 time."},
            ],
        ],
    }

    sc = convert_longmemeval_item(raw_normal)
    assert sc["id"] == "lme-q_ku_1"
    assert sc["category"] == "knowledge-update"
    steps = sc["steps"]
    assert isinstance(steps, list)
    # 2 个会话各产生 seed_history + consolidate，再加最后 1 步 ask，共 5 步
    assert len(steps) == 5
    # 验证乱序会话已按时间先后重排（05-25 在前，05-27 在后）
    assert "27:10" in steps[0]["messages"][0]["content"]
    assert steps[1]["action"] == "consolidate" and steps[1]["today"] == "2023-05-25"
    assert "25:50" in steps[2]["messages"][0]["content"]
    assert steps[3]["action"] == "consolidate" and steps[3]["today"] == "2023-05-27"
    assert steps[4]["action"] == "ask"
    assert steps[4]["today"] == "2023-06-01"
    assert steps[4]["expected_answer"] == "25:50"
    assert steps[4]["is_abstention"] is False

    # _abs 后缀题目应自动归类为 abstention 且 is_abstention=True
    raw_abs = dict(raw_normal, question_id="q_temp_99_abs", question_type="temporal-reasoning")
    sc_abs = convert_longmemeval_item(raw_abs)
    assert sc_abs["category"] == "abstention"
    assert sc_abs["steps"][-1]["is_abstention"] is True


# 测试 LongMemEval 分层均衡抽样，确保小样本也覆盖全部题型类别
def test_convert_longmemeval_dataset_stratified_sampling(tmp_path: Path) -> None:
    items = []
    for cat in ("single-session-user", "knowledge-update", "temporal-reasoning", "multi-session"):
        for idx in range(3):
            items.append(
                {
                    "question_id": f"{cat}_{idx}",
                    "question_type": cat,
                    "question": f"Q {cat} {idx}",
                    "answer": f"A {idx}",
                    "question_date": "2023/06/01 (Thu) 09:00",
                    "haystack_dates": ["2023/05/20 (Sat) 10:00"],
                    "haystack_sessions": [[{"role": "user", "content": f"Fact {cat} {idx}"}]],
                }
            )

    sampled = convert_longmemeval_dataset(items, limit=4, seed=7)
    assert len(sampled) == 4
    assert {item["category"] for item in sampled} == {
        "single-session-user",
        "knowledge-update",
        "temporal-reasoning",
        "multi-session",
    }

    out = tmp_path / "lme.jsonl"
    write_jsonl(out, sampled)
    reloaded = load_scenarios(out)
    assert len(reloaded) == 4


# 测试 LoCoMo 导出为离线检索语料与检索案例，并可直接通过 evaluate_search 校验金标原文
def test_export_locomo_for_search_produces_valid_runnable_benchmark(tmp_path: Path) -> None:
    fake_locomo = [
        {
            "sample_id": "conv-26",
            "conversation": {
                "speaker_a": "Caroline",
                "speaker_b": "Melanie",
                "session_1_date_time": "1:56 pm on 8 May, 2023",
                "session_1": [
                    {"speaker": "Caroline", "dia_id": "D1:1", "text": "Yesterday I attended the LGBTQ support group."},
                    {"speaker": "Melanie", "dia_id": "D1:2", "text": "That sounds wonderful, Caroline!"},
                ],
                "session_2_date_time": "3:10 pm on 15 May, 2023",
                "session_2": [
                    {"speaker": "Caroline", "dia_id": "D2:1", "text": "I ran a charity race for mental health awareness."},
                    {"speaker": "Melanie", "dia_id": "D2:2", "text": "Congrats! You also researched adoption agencies right?"},
                ],
            },
            "qa": [
                {
                    "question": "What did the charity race raise awareness for?",
                    "answer": "mental health",
                    "evidence": ["D2:1"],
                    "category": 4,
                },
                {
                    "question": "When did Caroline attend the LGBTQ support group?",
                    "answer": "7 May 2023",
                    "evidence": ["D1:1"],
                    "category": 2,
                },
                {
                    "question": "What did Caroline realize after the charity race?",
                    "adversarial_answer": "self-care",
                    "evidence": ["D2:1"],
                    "category": 5,
                },
            ],
        }
    ]

    data_dir = tmp_path / "locomo_data"
    cases = export_locomo_for_search(fake_locomo, data_dir, limit=10, seed=42)
    # 类别 5 属于拒答/对抗题，不应误混入正例检索集
    assert len(cases) == 2
    assert {c["category"] for c in cases} == {"single-hop", "temporal"}

    memory = MemoryStore(data_dir, create_dirs=False)
    report = evaluate_search(memory, cases, root="data/memory", limit=5, git_commit="locomo-test")
    assert report["summary"]["total_cases"] == 2
    assert report["summary"]["passed_cases"] == 2
    assert report["summary"]["mean_source_recall"] == 1.0


# 测试 LoCoMo 转换为多轮记忆评测场景（保留说话人标识并支持对抗拒答题）
def test_convert_locomo_for_memory_handles_speakers_and_adversarial() -> None:
    fake_locomo = [
        {
            "sample_id": "conv-1",
            "conversation": {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session_1_date_time": "10:00 am on 1 May, 2023",
                "session_1": [
                    {"speaker": "Alice", "dia_id": "D1:1", "text": "I adopted a golden retriever named Max."},
                    {"speaker": "Bob", "dia_id": "D1:2", "text": "Max is a great name!"},
                ],
            },
            "qa": [
                {
                    "question": "What is Alice's dog's name?",
                    "answer": "Max",
                    "evidence": ["D1:1"],
                    "category": 4,
                },
                {
                    "question": "What is Alice's cat's name?",
                    "adversarial_answer": "Whiskers",
                    "evidence": ["D1:1"],
                    "category": 5,
                },
            ],
        }
    ]

    scenarios = convert_locomo_for_memory(fake_locomo, limit=10, max_sessions_per_conv=2, seed=42)
    assert len(scenarios) == 2
    by_cat = {s["category"]: s for s in scenarios}
    assert by_cat["single-hop"]["steps"][-1]["expected_answer"] == "Max"
    assert by_cat["single-hop"]["steps"][-1]["is_abstention"] is False
    assert "[Alice]" in by_cat["single-hop"]["shared_setup"]["steps"][0]["messages"][0]["content"]
    assert by_cat["adversarial"]["steps"][-1]["is_abstention"] is True


# 测试公开数据集路径校验：允许 evals/public，拒绝写入受 Git 追踪的 evals/ 根目录或 sample_data
def test_eval_cli_download_mode_validates_out_dir_and_prepares_datasets(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    # 允许 gitignore 保护的 evals/public 目录
    assert validate_public_out_dir(repo_root / "evals" / "public", repo_root=repo_root) == (
        repo_root / "evals" / "public"
    ).resolve()
    # 拒绝直接写入受 Git 追踪的 evals/ 或 evals/sample_data
    with pytest.raises(ValueError, match="evals/public"):
        eval_main(["--mode", "download", "--out-dir", str(repo_root / "evals")])
    with pytest.raises(ValueError, match="evals/public"):
        eval_main(["--mode", "download", "--out-dir", str(repo_root / "evals" / "sample_data")])

    # 使用预先放入 raw/ 的本地样例验证 prepare_public_datasets 与 --mode download 闭环
    out_dir = tmp_path / "public"
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "longmemeval_oracle.json").write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question_type": "single-session-user",
                    "question": "What color is my bike?",
                    "answer": "blue",
                    "question_date": "2023/06/01 (Thu) 09:00",
                    "haystack_dates": ["2023/05/20 (Sat) 10:00"],
                    "haystack_sessions": [[{"role": "user", "content": "I bought a blue bike."}]],
                }
            ]
        ),
        encoding="utf-8",
    )
    (raw_dir / "locomo10.json").write_text(
        json.dumps(
            [
                {
                    "sample_id": "conv-1",
                    "conversation": {
                        "speaker_a": "Alice",
                        "speaker_b": "Bob",
                        "session_1_date_time": "10:00 am on 1 May, 2023",
                        "session_1": [
                            {"speaker": "Alice", "dia_id": "D1:1", "text": "I love hiking in Kyoto."},
                        ],
                    },
                    "qa": [
                        {
                            "question": "Where does Alice love hiking?",
                            "answer": "Kyoto",
                            "evidence": ["D1:1"],
                            "category": 4,
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    summary = prepare_public_datasets(
        out_dir,
        dataset="all",
        longmemeval_limit=5,
        locomo_search_limit=5,
        locomo_memory_limit=5,
        repo_root=repo_root,
    )
    assert summary["longmemeval"]["exported"] == 1
    assert summary["locomo_search"]["exported"] == 1
    assert summary["locomo_memory"]["exported"] == 1
    assert (out_dir / "longmemeval_scenarios.jsonl").is_file()
    assert (out_dir / "locomo_search_cases.jsonl").is_file()
    assert (out_dir / "locomo_memory_scenarios.jsonl").is_file()

    # 通过统一 CLI --mode download 再跑一次（命中已有缓存文件，不联网）
    code = eval_main(["--mode", "download", "--dataset", "all", "--out-dir", str(out_dir)])
    assert code == 0


# 测试 LoCoMo 共享长记忆转换：默认保留全量会话、支持 max_convs 限制、紧凑写出 shared_setup 且支持按分类过滤
def test_locomo_shared_memory_full_sessions_max_convs_and_compact_jsonl_roundtrip(tmp_path: Path) -> None:
    # 构造 2 段各含 6 个会话的 LoCoMo 对话，其中 QA 证据落在第 6 个会话（验证不再硬截断前 4 会话）
    def _make_conv(sample_id: str, fact_word: str) -> dict[str, object]:
        conv: dict[str, object] = {"speaker_a": "Alice", "speaker_b": "Bob"}
        for s in range(1, 7):
            conv[f"session_{s}_date_time"] = f"10:00 am on {s} May, 2023"
            text = f"I finally visited {fact_word} today." if s == 6 else f"Casual chat session {s}."
            conv[f"session_{s}"] = [{"speaker": "Alice", "dia_id": f"D{s}:1", "text": text}]
        return {
            "sample_id": sample_id,
            "conversation": conv,
            "qa": [
                {"question": f"Where did Alice visit in {sample_id}?", "answer": fact_word, "evidence": ["D6:1"], "category": 4},
                {"question": f"What did Alice cook in {sample_id}?", "adversarial_answer": "pasta", "evidence": ["D6:1"], "category": 5},
            ],
        }

    fake_locomo = [_make_conv("conv-1", "Kyoto"), _make_conv("conv-2", "Reykjavik")]

    # max_convs=1 且不截断会话（None）时，只取 conv-1 但完整保留全部 6 个会话与第 6 会话证据题（统一使用 shared_setup）
    scenarios = convert_locomo_for_memory(fake_locomo, limit=10, max_sessions_per_conv=None, max_convs=1, seed=42)
    assert len(scenarios) == 2
    assert all(sc["shared_group"] == "locomo-conv-1" for sc in scenarios)
    assert all("setup_step_count" not in sc for sc in scenarios)
    assert all(len(sc["shared_setup"]["steps"]) == 12 for sc in scenarios)
    assert all(len(sc["steps"]) == 1 and sc["steps"][0]["action"] == "ask" for sc in scenarios)

    out_path = tmp_path / "locomo_compact.jsonl"
    write_scenarios_jsonl(out_path, scenarios)
    raw_records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # 首行为 1 条 shared_setup，后接 2 条轻量 QA（仅含 1 步 ask，不再重复 12 步历史）
    assert len(raw_records) == 3
    assert raw_records[0]["record_type"] == "shared_setup"
    assert raw_records[0]["shared_group"] == "locomo-conv-1"
    assert len(raw_records[0]["steps"]) == 12
    assert all(len(rec["steps"]) == 1 and rec["steps"][0]["action"] == "ask" for rec in raw_records[1:])

    reloaded = load_scenarios(out_path)
    assert len(reloaded) == 2
    assert reloaded[0]["shared_setup"]["shared_group"] == "locomo-conv-1"
    assert len(reloaded[0]["shared_setup"]["steps"]) == 12

    # 通过 CLI 参数 --locomo-max-convs 与 --locomo-max-sessions 验证端到端下载导出路径
    out_dir = tmp_path / "pub"
    (out_dir / "raw").mkdir(parents=True)
    (out_dir / "raw" / "locomo10.json").write_text(json.dumps(fake_locomo), encoding="utf-8")
    code = eval_main(
        [
            "--mode",
            "download",
            "--dataset",
            "locomo",
            "--out-dir",
            str(out_dir),
            "--locomo-max-convs",
            "1",
            "--locomo-max-sessions",
            "0",
            "--locomo-memory-limit",
            "0",
        ]
    )
    assert code == 0
    exported = load_scenarios(out_dir / "locomo_memory_scenarios.jsonl")
    assert len(exported) == 2
    assert all(sc["shared_group"] == "locomo-conv-1" for sc in exported)
