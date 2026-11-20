# My Helper v2 协作约定（跨模型交接必读）

任何新会话或切换模型后，在动手改代码前必须先阅读本文件与 [docs/dev-log.md](docs/dev-log.md)，沿着既有架构、基线与演进路线继续推进。

## 核心红线

1. **零第三方运行依赖**：生产代码仅使用 Python 3.12+ 标准库（`sqlite3`、`urllib`、`threading`、`http.server` 等）；`pytest` 仅作为开发测试依赖。
2. **插件优先与三层漏斗架构**：
   - 严格守住三层单向依赖：`ai`（纯 LLM 协议，零向上依赖） ← `agent`（通用 ReAct 与插件发现，只依赖 `ai`） ← `assistant`（个人助手业务） + `eval`（不反向依赖 `cli.py`），每次改动必须通过 `tests/test_architecture.py` 架构守卫测试。
   - 新增工具（`TOOLS` / `@tool`）、检索后端（`RETRIEVERS`）、模型厂商适配（`LLM_PROVIDERS`）或配置编译器（`CONFIG_COMPILERS`）优先写成 `plugins/*.py` 独立插件；非必要不动核心 `personal_assistant/agent/react.py`。
3. **安全与隐私边界**：
   - 保持 `write_root = data/workspace`、只读 Shell 白名单、防 SSRF 校验；**禁止**开放让 LLM 无人工审批直接写入 `plugins/` 或 `personal_assistant/`。
   - 严禁提交 `.env`、`data/`、`evals/public/`；不得修改父目录中的旧版仓库。
   - 不引入 Telegram、旧版知识库或手动 `/compact`、`/yes` 队列。
4. **编码与测试规范**：
   - 严格遵循 **TDD（Red → Green → Refactor）**：先写失败单元测试并运行 `uv run pytest` 确认红灯，再写最小实现跑绿灯。
   - 所有 Python 文件的每个 `class` 和 `def` 上方写一句简洁中文 `#` 注释。
   - 保持改动小步、清晰并及时 Git 提交（SSH 推送：`GIT_SSH_COMMAND="ssh -o BatchMode=yes" git push`）。

## 评测与诚实汇报纪律

1. **改前基线 → 改后同口径对比**：涉及检索、记忆、提示词、上下文或模型的改动，必须先留改前基线，再用同一组案例和可比配置跑改后评测，并绑定具体 Git Commit SHA。
2. **不挑拣、不删失败题**：完整报告改善、持平、退步与未验证项；检索 benchmark（`--mode search`）与真实模型多轮评测（`--mode memory`）分别验收，发现真退步先定位修复再重跑。
3. **评测路径统一**：所有评测数据放 `evals/`（公开大型数据集由 `python -m personal_assistant.eval --mode download` 下载到 gitignore 的 `evals/public/`），所有评测命令统一走 `python -m personal_assistant.eval`。

## 标准 GitHub 研发流程（Issue → 分支 → TDD 与评测 → PR → 合并）

每次新增功能、修 Bug、做架构重构或优化检索/记忆，必须走正规协作流，**禁止直接在 `main` 上开发功能并裸推**：

1. **开 Issue**：先建 GitHub Issue（使用 `.github/ISSUE_TEMPLATE/` 模板），写清背景、设计落点、改前基线与验收标准。
   - 若本机已安装并登录 `geng628` 的 `gh` CLI，先用 `gh auth status` 确认当前账号为 `geng628` 后执行 `gh issue create`。
2. **切功能分支**：从最新 `main` 切分支，命名规范：`feat/issue-<编号>-<简短英文>`、`fix/issue-<编号>-<简短英文>`、`refactor/issue-<编号>-<简短英文>`、`chore/<简短英文>`。
3. **TDD 开发 + 同口径评测**：
   - 先写失败单测（红灯）→ 最小实现（绿灯）→ 跑全量 `uv run pytest -v`；
   - 涉及检索/记忆/提示词/性能的改动，必须跑同口径改后 Benchmark 并保留逐例失败与退步项；
   - 追加交接记录到 [docs/dev-log.md](docs/dev-log.md)。
4. **提 PR + 等 CI + 确认合并**：
   - 推送分支至远端（`GIT_SSH_COMMAND="ssh -o BatchMode=yes" git push -u origin <branch>`）并创建 PR（正文按 `.github/PULL_REQUEST_TEMPLATE.md` 填写 `Closes #<编号>` 与改前/改后对比表）；
   - **只有当 GitHub Actions CI 全绿、评测结果达标且用户确认后**，才合并至 `main`（推荐 Squash Merge）并清理已合并分支。

## 跨模型交接日志协议（每次改动必做）

每次完成功能开发、Bug 修复、架构重构或评测对比并提交前，**必须**按 [docs/dev-log.md](docs/dev-log.md) 文末模板追加一条记录（含 Commit SHA、设计思路、踩坑根因与修复、评测前后数字、更新后的下一步待办），确保下一次换模型可以直接无缝续写。
