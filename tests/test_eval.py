"""评测模块自身单元测试：验证离线检索评测、隔离 Agent 评测与报告路径保护。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import evaluate_agent, evaluate_search, load_cases, main
from personal_assistant.ai import LLMClient
from personal_assistant.assistant.memory import MemoryStore


# 预录回复的假模型，用于确定性测试端到端 Agent 评测器
class ScriptedLLM(LLMClient):
    # 保存预设回复队列
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = list(responses)

    # 按序弹出下一条预设回复
    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return self._responses.pop(0)


# 测试公开样本集离线检索评测，并确认不会改写样本目录任何文件
def test_sample_search_eval_runs_without_mutating_data_dir() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    sample_data = repo_root / "evals" / "sample_data"
    cases = load_cases(repo_root / "evals" / "sample_cases.jsonl")

    before_entries = sorted(p.relative_to(sample_data).as_posix() for p in sample_data.rglob("*"))
    memory = MemoryStore(sample_data, create_dirs=False)
    report = evaluate_search(memory, cases, limit=5, git_commit="test-sha")
    after_entries = sorted(p.relative_to(sample_data).as_posix() for p in sample_data.rglob("*"))

    assert before_entries == after_entries
    assert report["git_commit"] == "test-sha"
    assert report["summary"]["total_cases"] == len(cases)
    assert report["summary"]["total_cases"] >= 35
    assert set(report["summary"]["by_category"].keys()) == {
        "exact",
        "multi_source",
        "negative",
        "paraphrase",
        "temporal",
    }
    assert report["summary"]["by_category"]["temporal"]["pass_rate"] == 1.0
    assert report["summary"]["mean_source_recall"] >= 0.85


# 测试检索评测的金标原文校验与未命中失败案例保留机制
def test_search_eval_validates_gold_quotes_and_reports_failures(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "memory").mkdir(parents=True)
    (data_dir / "memory" / "a.md").write_text("# 标题\n真实存在的原文句子。\n", encoding="utf-8")
    memory = MemoryStore(data_dir, create_dirs=False)

    # 金标引文在原文中不存在时应立即报错
    with pytest.raises(ValueError, match="未在原文中找到"):
        evaluate_search(
            memory,
            [
                {
                    "id": "bad-gold",
                    "query": "标题",
                    "expected_sources": [{"path": "data/memory/a.md", "quote": "不存在的引文"}],
                }
            ],
        )

    # 检索漏召回时应完整记录到 failures
    report = evaluate_search(
        memory,
        [
            {
                "id": "miss-case",
                "query": "完全无关的检索词xyz",
                "expected_sources": [{"path": "data/memory/a.md", "quote": "真实存在的原文句子"}],
            }
        ],
    )
    assert report["summary"]["failed_cases"] == 1
    assert report["failures"][0]["id"] == "miss-case"


# 测试 Agent 端到端评测的数据目录隔离性与工具调用轨迹记录
def test_agent_eval_isolates_data_dir_and_records_tool_trace(tmp_path: Path) -> None:
    data_dir = tmp_path / "real_data"
    (data_dir / "memory").mkdir(parents=True)
    (data_dir / "memory" / "note.md").write_text("# 规范\n使用固定 argv 防止注入。\n", encoding="utf-8")
    cases = [
        {
            "id": "c1",
            "query": "如何防止命令注入？记住这个问题。",
            "answerable": True,
            "expected_sources": [{"path": "data/memory/note.md", "quote": "使用固定 argv 防止注入"}],
            "expected_tools": ["search_markdown"],
        }
    ]
    llm = ScriptedLLM(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_markdown",
                            "arguments": '{"query": "固定 argv 注入", "root": "data/memory"}',
                        },
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "remember_fact", "arguments": "{}"},
                    },
                ],
            },
            {"role": "assistant", "content": "规范指出应使用固定 argv 防止注入。"},
        ]
    )

    report = evaluate_agent(llm, data_dir, cases, git_commit="sha123")

    assert report["summary"]["expected_tools_all_hit"] == 1
    assert report["summary"]["cache"]["llm_calls"] == 2
    assert float(report["summary"]["cache"]["mean_prefix_reuse_rate"]) > 0.7
    assert report["cases"][0]["called_tools"] == ["search_markdown", "remember_fact"]
    assert report["cases"][0]["cache"]["llm_calls"] == 2
    assert report["cases"][0]["review_status"] == "needs_human_review"
    # 即使评测过程中调用了 remember_fact，原始数据目录也不应被写入 MEMORY.md 或 facts.md
    assert not (data_dir / "memory" / "MEMORY.md").exists()
    assert not (data_dir / "memory" / "facts.md").exists()


# 测试评测 CLI 输出报告以及拦截把报告写进 --data-dir 内部的行为
def test_eval_cli_writes_report_and_blocks_output_inside_data_dir(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    sample_data = repo_root / "evals" / "sample_data"
    sample_cases = repo_root / "evals" / "sample_cases.jsonl"

    with pytest.raises(ValueError, match="不能放在 --data-dir"):
        main(
            [
                "--mode",
                "search",
                "--data-dir",
                str(sample_data),
                "--cases",
                str(sample_cases),
                "--output",
                str(sample_data / "report.json"),
            ]
        )

    out = tmp_path / "report.json"
    code = main(
        [
            "--mode",
            "search",
            "--data-dir",
            str(sample_data),
            "--cases",
            str(sample_cases),
            "--output",
            str(out),
        ]
    )
    assert code == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["mode"] == "search"
    assert saved["summary"]["total_cases"] >= 35


# 测试离线检索评测支持按 category 分组统计与单题 today 过期过滤
def test_search_eval_supports_categories_and_temporal_today(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "memory").mkdir(parents=True)
    (data_dir / "memory" / "schedule.md").write_text(
        "# 日程备忘\n"
        "- [2026-10-01] 参加旧版架构评审会 (until: 2026-10-10)\n"
        "- [2026-11-15] 参加冬季发布演练会 (until: 2026-11-30)\n",
        encoding="utf-8",
    )
    memory = MemoryStore(data_dir, create_dirs=False)
    cases = [
        {
            "id": "temp-active",
            "category": "temporal",
            "query": "冬季发布演练会",
            "today": "2026-11-20",
            "answerable": True,
            "expected_sources": [{"path": "data/memory/schedule.md", "quote": "冬季发布演练会"}],
        },
        {
            "id": "temp-expired",
            "category": "temporal",
            "query": "旧版架构评审会",
            "today": "2026-11-20",
            "answerable": False,
            "expected_sources": [],
        },
        {
            "id": "neg-other",
            "category": "negative",
            "query": "火星殖民基地预算",
            "answerable": False,
            "expected_sources": [],
        },
    ]

    report = evaluate_search(memory, cases, limit=5, git_commit="cat-sha")
    assert report["summary"]["total_cases"] == 3
    assert report["summary"]["passed_cases"] == 3
    by_cat = report["summary"]["by_category"]
    assert by_cat["temporal"]["total_cases"] == 2
    assert by_cat["temporal"]["passed_cases"] == 2
    assert by_cat["temporal"]["pass_rate"] == 1.0
    assert by_cat["negative"]["total_cases"] == 1
    assert by_cat["negative"]["negative_clean_rate"] == 1.0
    assert report["cases"][0]["category"] == "temporal"


# 测试 agent_eval 自动挂载 tools.json 配置的检索插件（fts5_bm25）并将 TOAST 落盘隔离在临时目录内
def test_agent_eval_mounts_configured_retriever_and_isolates_toast_offload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evals.agent_eval import _prepare_isolated_memory
    from personal_assistant.assistant.commands import ToolsConfig
    from personal_assistant.assistant.common import REPO_ROOT

    monkeypatch.delenv("MY_HELPER_DATA", raising=False)
    cfg = ToolsConfig.load(REPO_ROOT / "config" / "tools.json", project_root=REPO_ROOT)
    mem = _prepare_isolated_memory(REPO_ROOT / "evals" / "sample_data", tmp_path, cfg, REPO_ROOT)
    assert getattr(mem.retriever, "name", None) == "fts5_bm25"


# 测试 evals CLI 统一复用 ToolsConfig.build_llm：支持外部 --config、models.dsl、预设名/模型 ID 与 --judge-model
def test_eval_cli_unifies_build_llm_with_models_dsl_presets_and_judge_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    from personal_assistant.ai import OpenAIChatClient, OpenAICompatibleLLM
    from personal_assistant.ai import client as client_mod
    from personal_assistant.assistant.commands import ToolsConfig
    from personal_assistant.assistant.common import REPO_ROOT

    # 隔离宿主 .env 与全局 _PROVIDERS 引用，模拟干净 CLI 进程且退出后零泄漏
    monkeypatch.setattr("evals.cli.load_dotenv", lambda *_a, **_kw: None)
    isolated_providers = {k: v for k, v in client_mod._PROVIDERS.items() if k != "openai_reasoning"}
    monkeypatch.setattr(client_mod, "_PROVIDERS", isolated_providers)

    # 在仓库外临时目录放置独立 tools.json（验证非 repo/config 层级下仍能回退加载仓库 plugins）
    ext_cfg_dir = tmp_path / "ext_cfg"
    ext_cfg_dir.mkdir()
    ext_tools_json = ext_cfg_dir / "tools.json"
    ext_tools_json.write_text((REPO_ROOT / "config" / "tools.json").read_text(encoding="utf-8"), encoding="utf-8")

    agent_cases = tmp_path / "agent_cases.jsonl"
    agent_cases.write_text(
        json.dumps({"id": "a1", "query": "hi", "answerable": True, "expected_sources": []}) + "\n",
        encoding="utf-8",
    )
    mem_cases = tmp_path / "mem_cases.jsonl"
    mem_cases.write_text(
        json.dumps({"id": "m1", "steps": [{"action": "ask", "message": "hi"}]}) + "\n",
        encoding="utf-8",
    )

    captured_agent: list[LLMClient] = []
    monkeypatch.setattr(
        "evals.cli.evaluate_agent",
        lambda llm, *_a, **_kw: captured_agent.append(llm)
        or {"mode": "agent", "git_commit": "t", "summary": {}, "failures": [], "cases": []},
    )
    captured_mem: list[tuple[LLMClient, LLMClient | None]] = []
    monkeypatch.setattr(
        "evals.cli.evaluate_memory",
        lambda llm, _sc, **kw: captured_mem.append((llm, kw.get("judge_llm")))
        or {"mode": "memory", "git_commit": "t", "summary": {}, "failures": [], "scenarios": []},
    )

    # 1. 环境变量模式：外部 --config 自动注册 Slot-4，--model / --judge-model 支持预设别名（reasoner / deepseek）
    monkeypatch.setenv("LLM_MODE", "on")
    monkeypatch.setenv("LLM_API_KEY", "sk-eval-cli-test")
    monkeypatch.setenv("LLM_BASE_URL", "https://proxy.example.com/v1")
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    assert main(["--mode", "agent", "--config", str(ext_tools_json), "--cases", str(agent_cases), "--model", "reasoner"]) == 0
    assert isinstance(captured_agent[-1], OpenAIChatClient)
    assert type(captured_agent[-1]).__name__ == "OpenAIReasoningLLM"
    assert captured_agent[-1].model == "deepseek-reasoner"  # type: ignore[attr-defined]
    assert captured_agent[-1].base_url == "https://proxy.example.com/v1"  # type: ignore[attr-defined]

    # 2. 即使环境中设了 LLM_PROVIDER=openai_reasoning，--judge-model deepseek 仍按预设自身协议构造 OpenAICompatibleLLM
    isolated_providers.pop("openai_reasoning", None)
    monkeypatch.setenv("LLM_PROVIDER", "openai_reasoning")
    monkeypatch.setenv("LLM_MODEL", "deepseek-reasoner")
    assert (
        main(["--mode", "memory", "--config", str(ext_tools_json), "--cases", str(mem_cases), "--judge-model", "deepseek"])
        == 0
    )
    eval_llm, judge_llm = captured_mem[-1]
    assert type(eval_llm).__name__ == "OpenAIReasoningLLM"
    assert eval_llm.model == "deepseek-reasoner"  # type: ignore[attr-defined]
    assert isinstance(judge_llm, OpenAICompatibleLLM)
    assert judge_llm.model == "deepseek-chat"
    assert judge_llm.api_key == "sk-eval-cli-test"
    assert judge_llm.base_url == "https://proxy.example.com/v1"
    assert os.environ["LLM_PROVIDER"] == "openai_reasoning"
    assert os.environ["LLM_MODEL"] == "deepseek-reasoner"

    # 3. 非预设自定义模型名绝不偷偷篡改当前 LLM_PROVIDER（无暗箱 or "openai_compatible" 劫持）
    assert (
        main(
            [
                "--mode",
                "memory",
                "--config",
                str(ext_tools_json),
                "--cases",
                str(mem_cases),
                "--model",
                "custom-r1-v2",
                "--judge-model",
                "custom-r1-judge",
            ]
        )
        == 0
    )
    assert type(captured_mem[-1][0]).__name__ == "OpenAIReasoningLLM"
    assert captured_mem[-1][0].model == "custom-r1-v2"  # type: ignore[attr-defined]
    assert type(captured_mem[-1][1]).__name__ == "OpenAIReasoningLLM"
    assert captured_mem[-1][1].model == "custom-r1-judge"  # type: ignore[attr-defined]

    # 4. default_model 匹配、缺省 base_url 补全、ollama 免密钥补全与缺失 tools.json 降级路径
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    assert (
        main(
            [
                "--mode",
                "memory",
                "--config",
                str(ext_tools_json),
                "--cases",
                str(mem_cases),
                "--model",
                "deepseek-reasoner",
                "--judge-model",
                "deepseek-chat",
            ]
        )
        == 0
    )
    dm_eval, dm_judge = captured_mem[-1]
    assert type(dm_eval).__name__ == "OpenAIReasoningLLM"
    assert dm_eval.base_url == "https://api.deepseek.com/v1"  # type: ignore[attr-defined]
    assert isinstance(dm_judge, OpenAICompatibleLLM)
    assert dm_judge.base_url == "https://api.deepseek.com/v1"

    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert main(["--mode", "agent", "--config", str(ext_tools_json), "--cases", str(agent_cases), "--model", "ollama"]) == 0
    assert isinstance(captured_agent[-1], OpenAICompatibleLLM)
    assert captured_agent[-1].base_url == "http://127.0.0.1:11434/v1"
    assert captured_agent[-1].model == "qwen2.5:7b"
    assert captured_agent[-1].api_key == "ollama"

    monkeypatch.setenv("LLM_API_KEY", "sk-fallback")
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    missing_cfg = tmp_path / "missing_tools.json"
    assert main(["--mode", "agent", "--config", str(missing_cfg), "--cases", str(agent_cases)]) == 0
    assert isinstance(captured_agent[-1], OpenAICompatibleLLM)
    assert main(["--mode", "agent", "--config", str(missing_cfg), "--cases", str(agent_cases), "--model", "m-override"]) == 0
    assert captured_agent[-1].model == "m-override"  # type: ignore[attr-defined]

    # 5. 空白 --judge-model 视为未传（复用待测模型）；--no-judge 优先关闭 Judge
    assert main(["--mode", "memory", "--config", str(ext_tools_json), "--cases", str(mem_cases), "--judge-model", "   "]) == 0
    assert captured_mem[-1][0] is captured_mem[-1][1]
    assert (
        main(["--mode", "memory", "--config", str(ext_tools_json), "--cases", str(mem_cases), "--judge-model", "deepseek", "--no-judge"])
        == 0
    )
    assert captured_mem[-1][1] is None

    # 6. 纯 models.dsl 模式：优先读 DSL（未定义档案显式报错不静默回退 .env），并支持预设别名/default_model 映射与 env= 透传
    (ext_cfg_dir / "models.dsl").write_text(
        "use deepseek sk-dsl-chat\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="未找到模型配置"):
        main(["--mode", "agent", "--config", str(ext_tools_json), "--cases", str(agent_cases), "--model", "reasoner"])

    monkeypatch.delenv("LLM_MODE", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    (ext_cfg_dir / "models.dsl").write_text(
        "use deepseek sk-dsl-chat\n"
        "add reasoner sk-dsl-r1\n"
        "model env_backed {\n"
        "  base_url https://api.deepseek.com/v1\n"
        "  model    deepseek-chat\n"
        "  key_env  CUSTOM_DSL_KEY\n"
        "}\n",
        encoding="utf-8",
    )
    assert (
        main(
            [
                "--mode",
                "memory",
                "--config",
                str(ext_tools_json),
                "--cases",
                str(mem_cases),
                "--model",
                "reasoner",
                "--judge-model",
                "deepseek-chat",
            ]
        )
        == 0
    )
    dsl_eval_llm, dsl_judge_llm = captured_mem[-1]
    assert type(dsl_eval_llm).__name__ == "OpenAIReasoningLLM"
    assert dsl_eval_llm.api_key == "sk-dsl-r1"  # type: ignore[attr-defined]
    assert isinstance(dsl_judge_llm, OpenAICompatibleLLM)
    assert dsl_judge_llm.api_key == "sk-dsl-chat"

    tcfg = ToolsConfig.load(ext_tools_json, project_root=REPO_ROOT)
    assert tcfg._match_preset("   ") is None
    env_llm = tcfg.build_llm(ext_cfg_dir, model_name="env_backed", env={"CUSTOM_DSL_KEY": "sk-passed-via-env"})
    assert isinstance(env_llm, OpenAICompatibleLLM)
    assert env_llm.api_key == "sk-passed-via-env"
