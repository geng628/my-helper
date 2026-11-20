## 关联 Issue
Closes #

## 核心改动摘要
- 改动动机与关键取舍：
- 涉及文件/插件：

## 验证与同口径 Benchmark 对比

| 评测项 | 改前基线（SHA: `...`） | 改后结果（本 PR） | 变化 / 备注 |
| --- | --- | --- | --- |
| 单元测试 (`uv run pytest -v`) | | | |
| 40 题自建检索 (`--mode search`) | | | |
| LoCoMo 200 题检索（如涉及） | | | |
| 16 场景多轮记忆（如涉及） | | | |
| LongMemEval 14 题（如涉及） | | | |

- **退步与未验证项（如实填写）**：

## 合并前自检 Checklist
- [ ] 严格遵守零第三方运行依赖与三层依赖漏斗（`tests/test_architecture.py` 通过）
- [ ] 所有新增 `class` / `def` 上方均含简洁中文 `#` 注释
- [ ] 未泄露 `.env`、`config/models.dsl`、`config/models.json`、`data/` 或 `evals/public/`
- [ ] 已按 `AGENTS.md` 将本次思路、踩坑与评测数字追加到 `docs/dev-log.md`
- [ ] GitHub Actions CI 全绿
