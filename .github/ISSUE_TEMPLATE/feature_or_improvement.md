---
name: 功能新增 / 效果改进
about: 规划一个新插件、架构重构或检索/记忆优化任务
title: "[Feat] "
labels: enhancement
---

## 背景与痛点
描述当前遇到的问题或为什么需要这项能力。

## 设计思路与落点
- 是否做成 `plugins/` 独立插件？（是 / 否，说明原因）
- 涉及哪些层级？（Layer 1 `ai` / Layer 2 `agent` / Layer 3 `assistant` / `eval`）
- 潜在安全或 KV Cache 风险评估：

## 改前基线（必填，涉及检索/记忆/提示词/性能时）
- 对应 Git Commit SHA：
- 当前指标（单元测试 / 40 题检索 / LoCoMo 200 题 / 16 场景记忆 / LongMemEval）：

## 验收标准
- [ ] 新增 TDD 单元测试并全量 `uv run pytest -v` 通过（含三层架构守卫）
- [ ] 跑完同口径改后 Benchmark，确认无未解释退步
- [ ] 更新 `docs/dev-log.md` 交接记录与必要文档
