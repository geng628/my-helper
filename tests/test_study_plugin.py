"""学习助手知识卡片与间隔复习插件（plugins/tools/study_tools.py）单元测试（TDD）。"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pytest

from personal_assistant.agent.plugins import PluginManager
from personal_assistant.agent.tools import ToolDocument, ToolMetadata
from personal_assistant.ai import LLMClient
from personal_assistant.assistant.app import AssistantApp
from personal_assistant.assistant.commands import ToolsConfig
from personal_assistant.assistant.common import REPO_ROOT
from personal_assistant.assistant.memory import MemoryStore


# 测试用脚本化模拟模型
class ScriptedStudyLLM(LLMClient):
    # 初始化预设回复序列
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = list(responses)

    # 弹出并返回下一条预设回复
    def respond(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return self._responses.pop(0)


# 测试 ToolsPlugin 自动发现、默认 tools.json 白名单启用、Schema 隐藏注入参数与关闭拦截
def test_study_plugin_discovery_allowlist_and_schema_isolation(tmp_path: Path) -> None:
    manager = PluginManager.from_entries(["plugins"], REPO_ROOT)
    tool_names = {item.name for item in manager.default_tools}
    assert {"save_study_card", "list_due_reviews"}.issubset(tool_names)

    cfg = ToolsConfig.load(REPO_ROOT / "config" / "tools.json", project_root=REPO_ROOT)
    assert "save_study_card" in cfg.enabled_tools
    assert "list_due_reviews" in cfg.enabled_tools

    registry = cfg.build_registry(
        data_root=tmp_path / "data",
        tool_context={"today": date(2026, 11, 26)},
    )
    assert "save_study_card" in registry.names
    assert "list_due_reviews" in registry.names
    assert registry.metadata("save_study_card") == ToolMetadata(risk="write", always_on=False)
    assert registry.metadata("list_due_reviews") == ToolMetadata(risk="read", always_on=False)
    assert registry.document("save_study_card") == ToolDocument(
        source_type="plugin",
        source_name="study_tools",
    )
    assert registry.document("list_due_reviews") == ToolDocument(
        source_type="plugin",
        source_name="study_tools",
    )

    save_tool = next(t for t in registry if t.name == "save_study_card")
    list_tool = next(t for t in registry if t.name == "list_due_reviews")
    assert set(save_tool.parameters) == {"topic", "concept", "summary", "grade"}
    assert save_tool.param_enums == {"grade": ("again", "hard", "good", "easy")}
    assert set(list_tool.parameters) == {"scope"}
    assert "切勿调用此工具" in save_tool.description

    # 从 enabled_tools 移除后应立即被注册表白名单拦截
    disabled_cfg_path = tmp_path / "tools_disabled.json"
    disabled_cfg_path.write_text(
        json.dumps({"enabled_tools": ["list_files"], "plugins": ["plugins"], "commands": []}),
        encoding="utf-8",
    )
    disabled_reg = ToolsConfig.load(disabled_cfg_path, project_root=REPO_ROOT).build_registry(
        data_root=tmp_path / "data",
        tool_context={"today": date(2026, 11, 26)},
    )
    assert "不在白名单中" in disabled_reg.execute(
        "save_study_card",
        json.dumps(
            {
                "topic": "python",
                "concept": "GIL",
                "summary": "全局解释器锁",
                "grade": "good",
            }
        ),
    )
    assert "不在白名单中" in disabled_reg.execute("list_due_reviews", '{"scope": "due"}')


# 测试逻辑日期注入下同一概念演进、幂等更新及掌握度单调性与上限封顶
def test_study_card_idempotent_update_and_spaced_repetition_progression(tmp_path: Path) -> None:
    from plugins.tools.study_tools import _compute_schedule, parse_study_file

    cfg = ToolsConfig.load(REPO_ROOT / "config" / "tools.json", project_root=REPO_ROOT)
    data_root = tmp_path / "data"

    # 借助统一日期序列验证 good -> good -> easy -> again -> hard 的精确状态转移
    steps = [
        (date(2026, 11, 26), "  Attention   Mechanism ", "good", 1, 1, "2026-11-27"),
        (date(2026, 11, 27), "attention mechanism", "good", 2, 3, "2026-11-30"),
        (date(2026, 11, 30), "ATTENTION  mechanism", "easy", 3, 8, "2026-12-08"),
        (date(2026, 12, 8), "Attention Mechanism", "again", 0, 1, "2026-12-09"),
        (date(2026, 12, 9), "attention mechanism", "hard", 1, 1, "2026-12-10"),
        (date(2026, 12, 10), "Attention Mechanism", "hard", 2, 2, "2026-12-12"),
    ]
    for idx, (day, concept_variant, grade, exp_reps, exp_interval, exp_due) in enumerate(steps, start=1):
        registry = cfg.build_registry(
            data_root=data_root,
            tool_context={"today": day},
        )
        res = registry.execute(
            "save_study_card",
            json.dumps(
                {
                    "topic": "Transformer Basics",
                    "concept": concept_variant,
                    "summary": f"第 {idx} 轮复习总结：QKV 缩放点积除以 sqrt(d_k)。",
                    "grade": grade,
                },
                ensure_ascii=False,
            ),
        )
        assert "data/workspace/study/transformer-basics.md" in res
        assert f"复习次数: {exp_reps}" in res
        assert f"间隔天数: {exp_interval}" in res
        assert f"下次复习: {exp_due}" in res

    card_file = data_root / "workspace" / "study" / "transformer-basics.md"
    assert card_file.is_file()
    topic_title, cards = parse_study_file(card_file.read_text(encoding="utf-8"))
    assert topic_title == "Transformer Basics"
    assert len(cards) == 1
    card = cards[0]
    assert card.grade == "hard"
    assert card.reps == 2
    assert card.interval_days == 2
    assert card.due.isoformat() == "2026-12-12"
    assert card.updated.isoformat() == "2026-12-10"
    assert "第 6 轮复习总结" in card.summary

    # 验证单调性边界：good(1)->easy(4) 不倒挂，easy(4)->hard(5)/good(8)/easy(10) 严格递增，且 hard/good<=180、easy<=365
    ref = date(2026, 11, 26)
    assert _compute_schedule("easy", 1, 1, ref)[1] == 4
    assert _compute_schedule("hard", 1, 4, ref)[1] == 5
    assert _compute_schedule("good", 1, 4, ref)[1] == 8
    assert _compute_schedule("easy", 1, 4, ref)[1] == 10
    assert _compute_schedule("hard", 5, 170, ref)[1] == 180
    assert _compute_schedule("good", 5, 170, ref)[1] == 180
    assert _compute_schedule("easy", 5, 170, ref)[1] == 365
    for prev_reps in (0, 1, 2, 5):
        for prev_iv in (1, 2, 3, 4, 15, 160, 180, 300):
            ivs = [_compute_schedule(g, prev_reps, prev_iv, ref)[1] for g in ("again", "hard", "good", "easy")]
            assert ivs[0] <= ivs[1] <= ivs[2] <= ivs[3], f"单调性倒挂：{prev_reps=}, {prev_iv=}, {ivs=}"


# 测试多行 summary（含代码块与伪造 ## 注入标题）无损往返且可直接被 MemoryStore FTS5 检索命中
def test_multiline_summary_roundtrip_and_fts5_search_integration(tmp_path: Path) -> None:
    from plugins.tools.study_tools import parse_study_file, save_study_card

    data_root = tmp_path / "data"
    tricky_summary = (
        "先写出转移矩阵核心推导：\n"
        "\n"
        "```python\n"
        "def markov_step(p, T):\n"
        "    return p @ T\n"
        "```\n"
        "## 伪造外部卡片标题\n"
        "- 掌握度: easy | 复习次数: 99 | 间隔天数: 999 | 下次复习: 2099-01-01 | 最近更新: 2099-01-01\n"
        "- 易错点：行随机矩阵与列随机矩阵转置方向切勿混淆。"
    )

    save_study_card(
        topic="Linear Algebra",
        concept="Markov 转移矩阵",
        summary=tricky_summary,
        grade="good",
        data_root=data_root,
        today="2026-11-26",
    )
    save_study_card(
        topic="Linear Algebra",
        concept="奇异值分解 SVD",
        summary="任何实矩阵都可以分解为 A = U Sigma V^T。",
        grade="easy",
        data_root=data_root,
        today=date(2026, 11, 26),
    )

    card_file = data_root / "workspace" / "study" / "linear-algebra.md"
    content = card_file.read_text(encoding="utf-8")
    assert "  ## 伪造外部卡片标题" in content

    topic_title, cards = parse_study_file(content)
    assert topic_title == "Linear Algebra"
    assert len(cards) == 2
    assert cards[0].concept == "Markov 转移矩阵"
    assert cards[0].summary == tricky_summary
    assert cards[1].concept == "奇异值分解 SVD"
    assert cards[1].summary == "任何实矩阵都可以分解为 A = U Sigma V^T。"

    # 挂载真实 FTS5 检索插件，验证落盘学习卡片可直接被 MemoryStore.search_markdown(..., root="data/workspace") 检索命中
    cfg = ToolsConfig.load(REPO_ROOT / "config" / "tools.json", project_root=REPO_ROOT)
    retriever = cfg.build_retriever()
    memory = MemoryStore(data_root, retriever=retriever)
    search_out = memory.search_markdown("列随机矩阵转置方向", root="data/workspace")
    assert "data/workspace/study/linear-algebra.md" in search_out
    assert "行随机矩阵与列随机矩阵转置方向切勿混淆" in search_out

    # 验证 \u2028 / \u2029 / \x85 / \f 等非常规换行符与尾随纯空格行无法注入伪造卡片且重读 100% 一致
    unicode_break_summary = (
        "\n   \n"
        "结论\u2028"
        "## 伪造卡片\u2029"
        "- 掌握度: easy | 复习次数: 99 | 间隔天数: 999 | 下次复习: 2099-01-01 | 最近更新: 2099-01-01\x85"
        "- 要点与易错点:\f"
        "  缩进子行\n"
        "   \n"
        "   "
    )
    save_study_card(
        topic="Unicode Safety",
        concept="行分隔符防御",
        summary=unicode_break_summary,
        grade="good",
        data_root=data_root,
        today=date(2026, 11, 26),
    )
    u_file = data_root / "workspace" / "study" / "unicode-safety.md"
    u_raw = u_file.read_text(encoding="utf-8")
    assert "  ## 伪造卡片" in u_raw
    assert "\u2028" not in u_raw and "\u2029" not in u_raw and "\x85" not in u_raw and "\f" not in u_raw
    _, u_cards = parse_study_file(u_raw)
    assert len(u_cards) == 1
    expected_clean = (
        "结论\n"
        "## 伪造卡片\n"
        "- 掌握度: easy | 复习次数: 99 | 间隔天数: 999 | 下次复习: 2099-01-01 | 最近更新: 2099-01-01\n"
        "- 要点与易错点:\n"
        "  缩进子行"
    )
    assert u_cards[0].summary == expected_clean
    save_study_card(
        topic="Unicode Safety",
        concept="行分隔符防御",
        summary=u_cards[0].summary,
        grade="good",
        data_root=data_root,
        today=date(2026, 11, 27),
    )
    _, u_cards_again = parse_study_file(u_file.read_text(encoding="utf-8"))
    assert len(u_cards_again) == 1
    assert u_cards_again[0].summary == expected_clean


# 测试 list_due_reviews 在空目录、due 排序过滤、all 概览以及指定专题名筛选下的输出
def test_list_due_reviews_modes_sorting_and_empty_states(tmp_path: Path) -> None:
    from plugins.tools.study_tools import list_due_reviews, save_study_card

    data_root = tmp_path / "data"
    ref_day = date(2026, 11, 26)

    # 1. 空目录友好提示
    empty_due = list_due_reviews("due", data_root=data_root, today=ref_day)
    empty_all = list_due_reviews("all", data_root=data_root, today=ref_day)
    empty_topic = list_due_reviews("agent-loop", data_root=data_root, today=ref_day)
    assert "暂无" in empty_due or "为空" in empty_due or "没有" in empty_due
    assert "暂无" in empty_all or "为空" in empty_all or "没有" in empty_all
    assert "暂无" in empty_topic or "为空" in empty_topic or "未找到" in empty_topic

    # 2. 构造多专题、不同到期时间与间隔天数的卡片
    # 卡1：2026-11-20 创建 good -> due 2026-11-21（相对 11-26 逾期 5 天，interval=1）
    save_study_card("Agent Loop", "ReAct 循环边界", "控制最大步数防止死循环", "good", data_root=data_root, today="2026-11-20")
    # 卡2：2026-11-24 创建 good (due 11-25) -> 11-22? 用 11-22 good (11-23), 再 11-22 good -> due 11-25（逾期 1 天，interval=3）
    save_study_card("Agent Loop", "KV Cache 前缀冻结", "系统提示词稳定不变", "good", data_root=data_root, today="2026-11-22")
    save_study_card("Agent Loop", "KV Cache 前缀冻结", "系统提示词稳定不变", "good", data_root=data_root, today="2026-11-22")
    # 卡3：2026-11-24 创建 again -> due 2026-11-25（逾期 1 天，interval=1，同天到期应排在 interval=3 之前）
    save_study_card("数据库索引", "B+树聚簇索引", "叶子节点存储完整行记录", "again", data_root=data_root, today="2026-11-24")
    # 卡4：2026-11-25 创建 good -> due 2026-11-26（今日到期，interval=1）
    save_study_card("数据库索引", "MVCC 快照读", "ReadView 判断事务可见性", "good", data_root=data_root, today="2026-11-25")
    # 卡5：2026-11-26 创建 easy -> due 2026-11-30（未到期，不应出现在 due 中）
    save_study_card("数据库索引", "WAL 预写日志", "先写日志再刷脏页", "easy", data_root=data_root, today="2026-11-26")

    due_out = list_due_reviews("due", data_root=data_root, today=ref_day)
    assert "ReAct 循环边界" in due_out
    assert "B+树聚簇索引" in due_out
    assert "KV Cache 前缀冻结" in due_out
    assert "MVCC 快照读" in due_out
    assert "WAL 预写日志" not in due_out
    assert "逾期 5 天" in due_out
    assert "逾期 1 天" in due_out
    assert "今日到期" in due_out

    # 验证排序：逾期 5 天 < (逾期 1 天, interval=1) < (逾期 1 天, interval=3) < 今日到期
    pos_react = due_out.index("ReAct 循环边界")
    pos_btree = due_out.index("B+树聚簇索引")
    pos_kv = due_out.index("KV Cache 前缀冻结")
    pos_mvcc = due_out.index("MVCC 快照读")
    assert pos_react < pos_btree < pos_kv < pos_mvcc

    # 3. 全部未到期时 due 返回空队列友好提示
    early_due = list_due_reviews("due", data_root=data_root, today=date(2026, 11, 19))
    assert "没有" in early_due or "暂无" in early_due
    assert "ReAct 循环边界" not in early_due

    # 4. all 概览汇总全部 5 张卡片与统计数字
    all_out = list_due_reviews("all", data_root=data_root, today=ref_day)
    assert "5" in all_out
    assert "4" in all_out
    assert "WAL 预写日志" in all_out
    assert "ReAct 循环边界" in all_out

    # 5. 指定专题名筛选（同时支持 slug 风格与原始专题名风格）
    topic_out = list_due_reviews("数据库索引", data_root=data_root, today=ref_day)
    assert "B+树聚簇索引" in topic_out
    assert "MVCC 快照读" in topic_out
    assert "WAL 预写日志" in topic_out
    assert "ReAct 循环边界" not in topic_out

    slug_out = list_due_reviews("agent-loop", data_root=data_root, today=ref_day)
    assert "ReAct 循环边界" in slug_out
    assert "KV Cache 前缀冻结" in slug_out
    assert "MVCC 快照读" not in slug_out


# 测试非法 topic 拦截、符号链接逃逸拦截与 8 线程并发写入同专题不同概念零丢失
def test_security_validation_symlink_escape_and_concurrent_writes(tmp_path: Path) -> None:
    from plugins.tools.study_tools import list_due_reviews, parse_study_file, save_study_card

    data_root = tmp_path / "data"
    cfg = ToolsConfig.load(REPO_ROOT / "config" / "tools.json", project_root=REPO_ROOT)
    registry = cfg.build_registry(data_root=data_root, tool_context={"today": date(2026, 11, 26)})

    # 1. 非法 topic、概念、要点、评分与日期格式拦截
    for bad_topic in ("", "   ", "!@#$%^&*", ".", "..", "...", "../secret", "a/b", "a\\b", "___---"):
        with pytest.raises(ValueError):
            save_study_card(bad_topic, "概念", "总结", "good", data_root=data_root)
        err = registry.execute(
            "save_study_card",
            json.dumps({"topic": bad_topic, "concept": "概念", "summary": "总结", "grade": "good"}),
        )
        assert err.startswith("错误：")
    with pytest.raises(ValueError):
        save_study_card("python", "  ", "总结", "good", data_root=data_root)
    with pytest.raises(ValueError):
        save_study_card("python", "GIL", "  \n ", "good", data_root=data_root)
    with pytest.raises(ValueError):
        save_study_card("python", "GIL", "总结", "invalid", data_root=data_root)
    with pytest.raises(ValueError):
        save_study_card("python", "GIL", "总结", "good", data_root=data_root, today="2026-99-99")
    with pytest.raises(ValueError):
        list_due_reviews("../secret", data_root=data_root)
    with pytest.raises(ValueError):
        list_due_reviews("   ", data_root=data_root)
    for bad_dot_scope in (".", "..", "..."):
        with pytest.raises(ValueError):
            list_due_reviews(bad_dot_scope, data_root=data_root)
    with pytest.raises(ValueError, match="过长"):
        save_study_card("超长专题" * 15, "概念", "总结", "good", data_root=data_root)

    # 禁止覆盖已存在且非空、非学习卡片格式的普通 Markdown 笔记
    study_dir = data_root / "workspace" / "study"
    study_dir.mkdir(parents=True, exist_ok=True)
    plain_note = study_dir / "plain-note.md"
    plain_note.write_text("# 我的日常随笔\n请勿覆盖我的私人笔记本内容。", encoding="utf-8")
    with pytest.raises(ValueError, match="拒绝覆盖"):
        save_study_card("plain-note", "新概念", "新总结", "good", data_root=data_root)
    assert plain_note.read_text(encoding="utf-8") == "# 我的日常随笔\n请勿覆盖我的私人笔记本内容。"
    plain_note.unlink()

    # 2. 符号链接与非普通文件逃逸拦截
    outside_file = tmp_path / "outside_secret.md"
    outside_file.write_text("# 机密\n绝不可被读取或覆写", encoding="utf-8")
    (study_dir / "escape-file.md").symlink_to(outside_file)

    with pytest.raises(ValueError, match="符号链接"):
        save_study_card("escape-file", "注入", "覆盖内容", "again", data_root=data_root)
    with pytest.raises(ValueError, match="符号链接"):
        list_due_reviews("escape-file", data_root=data_root)
    with pytest.raises(ValueError, match="符号链接"):
        list_due_reviews("all", data_root=data_root)
    assert outside_file.read_text(encoding="utf-8") == "# 机密\n绝不可被读取或覆写"
    (study_dir / "escape-file.md").unlink()

    # 3. 父目录符号链接逃逸拦截
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    symlink_data = tmp_path / "sym_data"
    (symlink_data / "workspace").mkdir(parents=True)
    (symlink_data / "workspace" / "study").symlink_to(outside_dir)
    with pytest.raises(ValueError, match="符号链接"):
        save_study_card("python", "装饰器", "高阶函数包装", "good", data_root=symlink_data)
    with pytest.raises(ValueError, match="符号链接"):
        list_due_reviews("due", data_root=symlink_data)
    assert list(outside_dir.iterdir()) == []

    # 4. 8 线程并发写入同专题 24 个不同概念卡片零丢失、零半截文件
    total_cards = 24

    # 并发工作线程：写入指定索引的学习卡片
    def _worker(idx: int) -> str:
        return save_study_card(
            topic="Concurrency Lab",
            concept=f"并发概念-{idx:02d}",
            summary=f"要点 {idx:02d}：\n  锁与原子替换保障零丢卡。",
            grade="good",
            data_root=data_root,
            today=date(2026, 11, 26),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_worker, range(total_cards)))

    assert len(results) == total_cards
    concurrent_file = study_dir / "concurrency-lab.md"
    assert concurrent_file.is_file()
    assert not list(study_dir.glob("*.tmp"))
    _, parsed = parse_study_file(concurrent_file.read_text(encoding="utf-8"))
    assert len(parsed) == total_cards
    assert {c.concept for c in parsed} == {f"并发概念-{idx:02d}" for idx in range(total_cards)}


# 测试 AssistantApp 在零微内核改动下自动加载学习助手工具并完成一次 ReAct 调用
def test_assistant_app_integrates_study_plugin_end_to_end(tmp_path: Path) -> None:
    llm = ScriptedStudyLLM(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_study_save",
                        "type": "function",
                        "function": {
                            "name": "save_study_card",
                            "arguments": json.dumps(
                                {
                                    "topic": "algorithms",
                                    "concept": "二分查找边界",
                                    "summary": "左闭右开区间 [l, r) 下终止条件为 l < r。",
                                    "grade": "good",
                                }
                            ),
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "已为你记录「二分查找边界」学习卡片，明天（2026-11-27）记得复习。",
            },
        ]
    )
    app = AssistantApp(
        data_root=tmp_path / "data",
        config_path=REPO_ROOT / "config" / "tools.json",
        project_root=REPO_ROOT,
        llm=llm,
        today_fn=lambda: date(2026, 11, 26),
    )
    try:
        registry = app.build_tools("帮我记录今天刚学的二分查找边界，我已经掌握了")
        assert {"save_study_card", "list_due_reviews"}.issubset(set(registry.names))
        result = app.chat("帮我记录今天刚学的二分查找边界，我已经掌握了", chat_id="s1")
        assert "二分查找边界" in str(result["reply"])
        saved = tmp_path / "data" / "workspace" / "study" / "algorithms.md"
        assert saved.is_file()
        assert "左闭右开区间 [l, r) 下终止条件为 l < r。" in saved.read_text(encoding="utf-8")
    finally:
        app.close()
