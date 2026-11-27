# My Helper v2 协作约定（跨模型交接必读）

任何新会话或切换模型后，在动手改代码前必须先阅读本文件与 [docs/dev-log.md](docs/dev-log.md)，沿着既有架构、基线与演进路线继续推进。

## 核心红线

1. **零第三方运行依赖**：生产代码仅使用 Python 3.12+ 标准库（`sqlite3`、`urllib`、`http.client`、`threading`、`http.server` 等）；`pytest` 仅作为开发测试依赖。
2. **插件优先、三层漏斗与 Pi ch04 三层翻译架构**：
   - 严格守住三层单向依赖：`personal_assistant/` 仅保留纯净三层产品代码——`ai`（Layer 1：零向上依赖，内部遵循 Pi ch04 三层解耦：`protocol.py` 中立 12 事件协议 ← `registry.py` 统一前台 `stream`/`stream_simple`/`complete` 与 `ApiProviderRegistry` ← `providers/` 原生翻译器，加 `cache.py` 中立缓存观测） ← `agent`（Layer 2：通用 ReAct 循环、纯值对象 `Tool(ToolSchema)` 与零业务字段插件主板，只依赖 `ai`） ← `assistant`（Layer 3：个人助手业务）；顶层评测包 `evals/` 与 `tests/`、`plugins/` 平级（`personal_assistant/` 绝不反向依赖 `evals/`，且 `evals/` 绝不反向依赖 `assistant/cli.py`）。
   - Layer 2（`agent/`）与 Layer 3（`assistant/`）**严禁重新引入 OpenAI 方言渗透**（禁止 `to_openai_schema` / `to_openai_dict` / `extract_tool_calls` 或裸厂商 wire 字典）；统一消费中立 `Context` / `Message` / `ToolSchema` / `ModelInvoker`。每次改动必须通过 [tests/test_architecture.py](tests/test_architecture.py) 架构守卫（三层漏斗、`ai/` 内部三层解耦、零 OpenAI 方言符号、零业务字面量与四棵受管目录树 `0` 处 `# type: ignore`）。
   - 新增工具优先写入 [plugins/tools/](plugins/tools/) 下的 `@tool` 模块并由 [plugins/tools/plugin.py](plugins/tools/plugin.py)（`ToolsPlugin`）统一接入 `Tool` 协议；新增检索引擎、配置编译器、模型协议翻译器或运行观测器则实现为继承外层统一主板基类 `Plugin(ABC)` 的独立插件 `plugins/*.py`，分别接入 `RetrieverProtocol`、`ConfigCompilerProtocol`、`ApiProvider`（Slot 4，按 `models.dsl` 的 `api` 字段单点扩展新厂商，无需改动核心代码）或 `ObserverProtocol`；核心 [personal_assistant/agent/react.py](personal_assistant/agent/react.py) 可按通用引擎演进需要直接修改（如并行只读工具、保序提交等），只需保持 Layer 2 纯净（零业务耦合、零厂商方言）。
3. **安全与隐私边界**：
   - 保持 `write_root = data/workspace`、APFS Unicode/Inode 文件隔离、只读 Shell 白名单（含 `find` 原语白名单与 `git` 逐子命令只读白名单）、DNS Pinning 防 SSRF 与同轮联网锁死 `save_rule`；**禁止**开放让 LLM 无人工审批直接写入 `plugins/` 或 `personal_assistant/`。
   - 严禁提交 `.env`、`data/`、`evals/public/`；不得修改父目录中的旧版仓库。
   - 不引入 Telegram、旧版知识库或手动 `/compact`、`/yes` 队列。
4. **编码与测试规范**：
   - 严格遵循 **TDD（Red → Green → Refactor）**：先写失败单元测试并运行 `uv run pytest` 确认红灯，再写最小实现跑绿灯。
   - 所有 Python 文件（含 `tests/`）的每个 `class`、`def` 和 `async def`（含嵌套类/函数）上方写一句含中文字符的简洁 `#` 注释；`personal_assistant/`、`plugins/`、`scripts/`、`evals/` 四棵目录树保持 `0` 处 `# type: ignore`。
   - 保持改动小步、清晰并及时 Git 提交（SSH 推送：`GIT_SSH_COMMAND="ssh -o BatchMode=yes" git push`）。

## 评测与诚实汇报纪律

1. **改前基线 → 改后同口径对比**：涉及检索、记忆、提示词、上下文或模型的改动，必须先留改前基线，再用同一组案例和可比配置跑改后评测，并绑定具体 Git Commit SHA。
2. **不挑拣、不删失败题**：完整报告改善、持平、退步与未验证项；检索 benchmark（`--mode search`）与真实模型多轮评测（`--mode memory`）分别验收，发现真退步先定位修复再重跑。
3. **评测路径统一**：评测引擎代码与评测数据统一收拢在顶层 `evals/`（公开大型数据集由 `python -m evals --mode download` 下载到 gitignore 的 `evals/public/`），所有评测命令统一走 `uv run python -m evals`。

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

## 小 PR + 并行 Worktree 与三层接力 Review 规范

为提升研发效率并同时守住架构质量与审阅清晰度，多任务协作默认遵循以下受控并行流程：

1. **先拆方案、列文件清单、确认后再派活**：
   - 收到多项需求时先读相关代码，列出**任务拆分方案、每个任务拟改文件列表、依赖关系及串/并行判定**，经用户确认后再启动编码；
   - 一两行单文件的极小改动直接在单分支快做，不强行拆分。
2. **串并行边界与并发限流（上限 2~3 路）**：
   - **必须串行**：核心架构重构、公共抽象与协议定义、改动同一批文件（尤其注意避让 `memory.py`、`commands.py` 等高碰撞模块）或有先后依赖的任务，老实按顺序串行推进（后项可叠分支基于前项分支）。
   - **允许并行**：边界清晰、文件零交集的独立小任务（如独立插件、独立 Bug、单测补全、`evals/` 改进），最多同时开 **2~3 个**子 Agent 在独立 git worktree（`workspace="share"`）与独立分支（`feat/issue-<N>-xxx`、`fix/issue-<N>-xxx`、`docs/issue-<N>-xxx`、`test/issue-<N>-xxx`）上开发，绝不直接污染主工作目录。
3. **单 PR 单一职责与精简体积**：每个 Issue / PR 严守**只做一件事**，不顺手修改无关文件，保持 Diff 小而易审，提交前必须先 `rebase` 到最新 `main` 解决冲突。
4. **并行防撞补丁（避开本仓库两类必撞点）**：
   - **`docs/dev-log.md` 延后追加**：并行子 Agent 在开发阶段只改代码与测试，不在各自并行分支提前抢写 [docs/dev-log.md](docs/dev-log.md)；待各 PR 依次 `rebase origin/main` 准备合并前，再按合并顺序严格 `append-only` 追加，彻底消灭尾部冲突；
   - **`.env` 与 `evals/public/` 只读复用**：独立 worktree 可零阻力跑 `uv run pytest` 与 40 题本地离线检索；跑实模 Benchmark 时只读复用主目录 `.env` 与 `evals/public/`（严禁复制提交）；若多条并行 PR 都会影响同一套实模评测指标，合入第 1 个后，后续 PR 必须先 `rebase` 最新 `main` 并补跑合流验收以排查跨 PR 语义交互。
5. **三层接力 Review（严禁写代码子 Agent 自审）**：
   - **第 1 层（机器硬守卫）**：`uv run pytest -v`（含 `tests/test_architecture.py`）、AST 中文 `#` 注释 0 缺失、GitHub Actions CI、同口径 Benchmark；
   - **第 2 层（独立只读 Reviewer Agent）**：另开全新上下文的只读 Reviewer 子 Agent 对照 Issue 验收标准与 `git diff main...HEAD` 做对抗式审查（重点查：**范围漂移、跨 PR 语义冲突、重复造轮子、测试放水、安全/KV Cache 不变量破坏**），输出 `Blocker / Nit / Pass`；凡存在 `Blocker` 直接打回对应编码子 Agent 返工重测，修净后才端到用户面前；
   - **第 3 层（项目 Owner 终审拍板）**：全部完成后集中汇总每个 PR 的分支/链接、测试与评测结果、Reviewer 结论及失败/遗留项（失败任务如实写明卡点，绝不假装完成），由用户终审架构边界、核心 Diff 与真实评测账本后拍板 Squash Merge。

## 跨模型交接日志协议（每次改动必做）

每次完成功能开发、Bug 修复、架构重构或评测对比并提交前，**必须**按 [docs/dev-log.md](docs/dev-log.md) 文末模板追加一条记录（含 Commit SHA、设计思路、踩坑根因与修复、评测前后数字、更新后的下一步待办），确保下一次换模型可以直接无缝续写。
