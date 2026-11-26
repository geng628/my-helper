# My Helper v2 架构基线、工程权衡与演进日志（跨模型交接文档）

> 本文档按「当前架构快照 → 关键工程权衡（ADR） → 真实排坑与红队修复档案 → 同口径评测总账 → 近期迭代明细」组织。
> 新会话或切换模型后，先通读本文件第 1–4 节与根目录 [AGENTS.md](../AGENTS.md)，即可准确掌握最新代码状态与设计边界。

---

## 一、当前架构快照与核心不变量（最新主干基线）

1. **零第三方运行依赖**：生产代码仅使用 Python 3.12+ 标准库（`urllib`、`http.client`、`sqlite3`、`threading`、`http.server`、`shlex`、`ipaddress`、`unicodedata` 等）；`pytest` 仅作开发测试依赖。
2. **三层单向依赖漏斗（`ai ← agent ← assistant`）**：
   - `personal_assistant/ai/`（Layer 1）：零内部依赖的 LLM 传输底座（`OpenAIChatClient` + `ModelProfile` 直通工厂 `from_model_profile`）、强类型流式事件（含 `StreamDone.finish_reason`）、通用命名角色配置表 `ModelsConfig.roles` 与按会话隔离的有界 `CacheMonitor`；
   - `personal_assistant/agent/`（Layer 2）：仅依赖 `ai/` 的 30 余行单层 ReAct 核心循环（`iter_react` / `run_react`，单步 `convert_to_llm` 边界钩子）、纯值对象工具递进类型（`Tool(ToolSchema)` + `bind_tool_context`，自动推断默认可选参数并支持执行期 `tool_context`）与零业务字段 `PluginContext` 五大插件主板插槽（`Plugin(ABC)`），零业务路径、插件名或配置段硬编码；
   - `personal_assistant/assistant/`（Layer 3）：Markdown 记忆与内置零索引 `LinearRetriever` 兜底（`memory.py` / `memory_lifecycle.py` / `builtin_tools.py`）、零内部依赖底座（`common.py` / `text_search.py` / `files_policy.py`）、SQLite 会话存储（`db.py`）、外层回合预算与统一编排（`app.py`，暴露常驻 `app.tools` 并支撑 `evals/` 统一 `AssistantApp(auto_consolidate=False)` 装配）、后台沉淀（`consolidator.py`）及终端入口（`cli.py`）；
   - 顶层 `evals/` 与 `plugins/`、`tests/` 平级，双向隔离守卫及 `personal_assistant/`、`plugins/`、`scripts/`、`evals/` 四棵目录树零 `# type: ignore` 静态硬守卫由 [../tests/test_architecture.py](../tests/test_architecture.py) 强制执行。
3. **五大主板插件插槽与生命周期**：
   - Slot 1 `Tool`（`plugins/tools/plugin.py` 扫描命名空间子包 `custom_tools`、`shell_tools`、`web_tools`、`history_tools`、`study_tools`，单轨消费主板注入的配置与 `data_root`）；
   - Slot 2 `RetrieverProtocol`（`plugins/fts5_retriever.py`，未挂载时由 `LinearRetriever` 兜底）；
   - Slot 3 `ConfigCompilerProtocol`（`plugins/models_dsl.py`，`render` 采用显式强类型具名参数，核心参数对齐纯通用协议、业务角色参数收敛于插件侧）；
   - Slot 4 `LLMProviderProtocol`（`plugins/reasoning_provider.py`，经 `ModelProfile` 直通工厂装配）；
   - Slot 5 `ObserverProtocol`（`plugins/telemetry.py`，显式字段累加与有界 MRU 会话账本）；
   - 所有插件在启动期仅执行一次 `initialize()`，零跨插件横向导入，并在 `AssistantApp.close()` 时经 `terminate_all()` 统一释放资源。
4. **纵深安全与测试隔离基线**：
   - 文件层：零内部依赖的 `files_policy.py` 执行 `NFKC + casefold()` + 零宽格式符剥离 + 轻量单层敏感目标 `(st_dev, st_ino)` Inode 校验，写权限严格收敛于 `data/workspace/`；
   - Shell 层：`shell=False`、隔离空 `HOME`、语法级逐子命令只读 `git` 白名单、`pytest` 强制限定 `tests/` 与 `--ignore=data`、1 MB 流式管道上限与 `os.killpg` 进程组强杀；
   - 网络与防注入层：直连 `http.client` + `socket.create_connection((pinned_ip, port))` + TLS SNI/证书校验防 DNS Rebinding（天然不读环境代理）、`allow_fake_ip` 默认关闭、同轮触发 `risk="network"` 工具后硬性锁死 `save_rule`；
   - 单测与静态守卫层：[../tests/conftest.py](../tests/conftest.py) 全局 `autouse` fixture 默认隔离宿主机 `.env`、`LLM_*` 环境变量、`config/models.*` 与 `_PROVIDERS` 注册表（当前确定性单测 `202 passed`，四棵非测试目录树 `# type: ignore` 均为 `0`）。

---

## 二、关键架构决策与工程权衡（ADRs & Trade-offs）

### ADR-1：保持 30 余行单层 `react.py` 核心循环纯净，将安全预算下放至 Layer 3 `_GuardedTurnLLM`
- **背景**：移除早期硬编码 `MAX_TOOL_ROUNDS = 4` 与 `tools=None` 强行总结后，内核回归自然终止，但在工具持续报错或参数被 `max_tokens` 截断时存在无限重试消耗费用与锁死会话的风险。
- **决策**：不在 Layer 2 `agent/react.py` 重新塞回业务计数器或中文降级文案，而是在 Layer 3 `AssistantApp._run_stream_turn` 外层用 `_GuardedTurnLLM` 包装 `LLMClient.stream`，实施单轮步数（`24`）、累计 Token（`250,000`）、墙钟（`180s`）、连续等价工具调用（`3` 次）与 `finish_reason == "length"` 截断工具调用五项熔断。
- **权衡**：通用内核保持零侵入；直接绕过 `AssistantApp` 调用底层 `run_react()` 的外部调用方则需自行负责超时控制。

### ADR-2：会话内冻结 `MEMORY.md` 索引，跨轮仅持久化问答正文（剥离轮内工具轨迹）
- **背景**：个人伴读场景多轮连续追问频繁，若每轮动态改写 `system` 提示词或把上轮几千字网页/命令原始 `tool` 观察全量重放进历史，会迅速撑爆上下文并破坏服务端 Prefix Caching。
- **决策**：系统提示词按 `(today_str, chat_id)` 在会话内冻结；`StateStore.messages` 仅保存每轮 `user` 与 `assistant` 最终回复，跨轮剥离中间 `tool_calls` 与 `role="tool"` 观察。
- **权衡**：实测多轮服务端 KV Cache 命中率稳定在 `65%–82%+`，大幅节省输入 Token；代价是若上一轮工具返回的细枝末节未被助手写进回复正文，下一轮追问该细节时模型需重新调用读取工具。

### ADR-3：成对原子落库、单锁上下文快照与分层回滚语义
- **背景**：流式输出中途断网若残留单条 `user` 消息会引发后续推理模型连续同角色 HTTP 400；而分两次查询 `session_compactions` 与 `active_messages` 在后台压缩并发提交时会出现窗口断层。
- **决策**：`StateStore.append_turn` 在单事务内成对写入问答消息（只追不删）；`get_context_snapshot` 在单把锁内原子读取 `(compaction, active_messages)` 二元组（快照上界由 `max_message_id()` 提供）。
- **权衡与边界**：回合异常中断时，未完成的问答对不会入库，`remember_fact` 的 `(turn#<tag>)` 暂存行会在流生成器的 `finally` 分支自动回滚清理；但回合内已执行的 `write_file` 与 `save_study_card` 属于即时工作区文件写入，不做跨文件事务回滚（覆盖模式依赖 `data/workspace/.backups/` 时间戳备份恢复）。

### ADR-4：前后台双模型分槽、自适应滑窗与沉淀指数退避
- **背景**：推理模型（如 `deepseek-reasoner`）在复杂多步推理上准确率更高，但后台输出 JSON 摘要与记忆操作时消耗大量思维链输出 Token；此外后台沉淀若遇上游故障，固定 `20` 条历史截断会导致超出窗口的未压缩消息从上下文中暂时不可见。
- **决策**：`models.dsl` 支持 `consolidator` 独立轻量模型槽位；当未压缩活跃消息因沉淀繁忙或失败退避超过 `20` 条时，`_select_adaptive_history` 在 `60` 条 / `24,000` 字符上限内自动拓宽保留未压缩历史（对齐首条 `user` 边界）；`BackgroundConsolidator` 按会话实施 `2s -> 4s -> ... -> 256s`（上限封顶 `300s`）指数退避，单批上限 `max_batch_messages=24`，在途 `(turn#...)` 暂存事实在 `memory_lifecycle.py` 存储层通过单点 `_is_inflight_turn_line` 原生隔离。

### ADR-5：零依赖词面检索（SQLite FTS5 + BM25）优先，保留 `RetrieverProtocol` 扩展口
- **决策**：主线采用零依赖中英双语分词 + SQLite FTS5 BM25，不内置重型本地向量模型依赖。
- **权衡**：单跳精确事实与显式时间线检索延迟仅数毫秒且可解释性强；但面对零词面重叠的同义改写与跨会话隐式多跳问题（如 LoCoMo Search 的 `multi-hop` 与 `open-domain` 子类）存在明确词面上限。

---

## 三、典型排坑根因与红队修复档案（防重踩清单）

1. **`MEMORY.md` 逐轮改写击穿 KV Cache（`7f32529`）**：早期每次 `remember_fact` 直接追加 `MEMORY.md`，导致系统前缀每轮变化、缓存命中率接近 0%。改为 `MEMORY.md` 限长索引会话内冻结 + `facts.md` 按需检索。
2. **`facts.md` 幽灵裸行导致过期事件复活（`be1325b`）**：后台沉淀把事件带 `(until: YYYY-MM-DD)` 写入主题文件后未清理 `facts.md` 中的原始裸句。新增 `prune_raw_facts_for_batch()` 自动按批次消息 ID 与文本重叠清理已吸收裸行。
3. **模型在 `replace.new` 中复读旧时间戳（`f08b187`）**：沉淀输出常自带 `[YYYY-MM-DD]` 造成双时间戳落盘。新增 `clean_memory_body()` 正则剥离复读前缀。
4. **Python 3.14 动态加载 `@dataclass` 插件崩溃（`5dbed3f`）**：`exec_module` 前未写入 `sys.modules[module_name]` 导致注解反射失败；改为执行前先登记、异常时回滚移除。
5. **`list_files("data/memory")` 暴露 `archive/` 目录（`1343b0d`）**：模型通过列目录绕过热检索过滤直读已过期事件；改为默认隐藏 `archive/`，仅显式传 `root="data/memory/archive"` 时开放。
6. **极小语料（`N < 10`）下 SQLite FTS5 `bm25() == 0` 漏召回（`d2e20e3`）**：原生 IDF 公式在 2 行文档命中 1 行时输出 `0.0`；改为仅在语料块数 `>= 10` 时启用 `min_bm25_score` 门控。
7. **`Path.rglob().resolve()` 拖慢检索延迟（`d2e20e3`）**：272 个文件每次查询触发大量 `lstat` 占去 80% 耗时；改用 `os.scandir` 遍历后 LoCoMo 单题检索耗时由 ~29 ms 降至 ~9–14 ms。
8. **流前预写 `user` + 异常 `DELETE` 制造内存 FTS5 幽灵索引（`932a72c9`）**：`history_tools` 以 `(max_id, total_count)` 判断增量同步，删行重插会导致缓存失步；彻底改为流完成后 `append_turn` 单事务成对只追不删。
9. **macOS APFS Unicode 大小写折叠绕过路径黑名单（`1ad1e0d`）**：APFS 将拉丁长 s `ſ`（U+017F）折叠为 `s`，旧版仅做 `.lower()`（`'ſtate'.lower() == 'ſtate'`），实测 `ls data/ſtate` 与 `cat config/modelſ.json` 可绕过敏感段黑名单；修复为剥离 `Cf` 格式符 + `NFKC` + `.casefold()`（`'ſ'.casefold() == 's'`）并叠加受保护文件 `(st_dev, st_ino)` 硬链接/软链接拦截（经全量 111 万 Unicode 码点对 APFS 实测验证）。
10. **`git branch` 写操作与裸 `<blob>`/`<tree>` 历史泄密（`1ad1e0d`）**：旧版仅放行 `git` 子命令名而未逐子命令锁死写标志，导致 `git branch -D` 可删分支、`--edit-description` 拉起编辑器挂起，且 `git show <blob>` 或 `git diff 4b825dc6... HEAD`（Git 内置空树 SHA）可读取历史提交中的敏感文件；修复为逐子命令严格只读参数白名单、`git cat-file -t` 拦截 `blob`/`tree` 端点、`git diff` 自动注入 `:(exclude,glob,icase)` 负向排除、隔离空 `HOME` 与 `os.killpg` 整组回收。
11. **`urllib` 二次 DNS 解析重绑定与默认 Fake-IP 放行风险（`1ad1e0d`）**：校验阶段与实际建连各自解析 DNS 存在 TOCTOU 窗口，且默认走系统环境代理并放行 `198.18.0.0/15`；修复为自定义 `_PinnedHTTPConnection` / `_PinnedHTTPSConnection` 直连已校验 IP、`ProxyHandler({})` 禁代理、`allow_fake_ip` 默认改为 `False` 并补齐 NAT64/6to4/Teredo 校验。
12. **间接提示词注入持久化篡改 `rules/`（`1ad1e0d` + `1b421d2`）**：恶意网页可诱导模型调用 `save_rule` 写入全局系统提示词；修复为同轮调用任何 `risk="network"` 工具后经 `turn_ctx["untrusted_tool_seen"]` 硬性锁死 `save_rule`，且无网络工具时亦校验用户原话中的显式长期规则意图词。
13. **评测 Judge 提示词混入测试集人物实体与短词子串误判（`c761c81`）**：早期 `memory_eval.py` 的 `JUDGE_SYSTEM_PROMPT` 针对 LoCoMo 失败题写死测试集人物示例（`Caroline` / `Melanie`），且期望词匹配对 `"Go"` 使用普通子串包含可误中 `"Google"`；已清除全部具名示例升级为 `JUDGE_PROMPT_VERSION = "v2-neutral"`（记录 `prompt_sha256`），先对 `<= 3` 字符 ASCII 期望词强制非字母数字词边界匹配，后于 Wave 7（`#84`）进一步移除长度魔法阈值，统一为全长度纯 ASCII 词/数字字符边界匹配。

---

## 四、Benchmark 演进总账（均绑定 Git SHA 与评测口径）

### 1. 离线检索评测（确定性，零 API 调用）

| Commit | 检索后端 | 仓库内回归集 `sample_cases.jsonl`（40 题） | 外部公开集 `LoCoMo Search`（ACL 2024，200 题 / 272 篇 `.md`） |
| --- | --- | --- | --- |
| `552863d` | `linear` | 通过 `32/40 (80.0%)`；Recall `96.6%`，MRR `0.908`，负例干净度 `63.6%` | 通过 `57/200 (28.5%)`；Recall `42.75%`，MRR `0.3587` |
| `d2e20e3` ~ `1b421d2` | `fts5_bm25` | 通过 `38/40 (95.0%)`；Recall `94.83%`，MRR `0.9483`，Quote `94.83%`，负例干净度 `100%` | 通过 `105/200 (52.5%)`；Recall `68.51%`（`single-hop` `94.0%` / `temporal` `92.0%` / `open-domain` `44.2%` / `multi-hop` `43.9%`），MRR `0.6221`，Quote `60.17%` |

### 2. 真实模型多轮记忆评测（区分历史 `v1` Judge 与去泄漏 `v2-neutral` Judge）

> **口径备忘**：
> - `LongMemEval_oracle`（`70/500` 或早期 `35/500`，`seed=42`）仅含证据会话，测多会话提炼与时序问答准确率，不含海量无关会话检索；
> - `LoCoMo Memory`（100 题）全部出自单段含 19 个会话的长对话 `conv-26`（`shared_setup` 共享一次沉淀后逐题隔离问答）。

| Commit | Judge 版本 | `memory_scenarios.jsonl`（16 场景，开发回归集） | `LoCoMo conv-26`（100 题，单段 19 会话长对话） | `LongMemEval_oracle`（证据会话分层抽样） | 关键说明 |
| --- | --- | --- | --- | --- | --- |
| `cffbf26` | `v1` | `deepseek-chat`：`13/16 (81.25%)`，KV 60.4% | — | `10/14 (71.43%)`（早期 14 题子集） | 初始多轮记忆基线 |
| `dba72a3` / `16ebd78` | `v1`（含 LoCoMo 具名示例） | `deepseek-chat`：`16/16`；`deepseek-reasoner`：`15/16` | `deepseek-chat`：`84/100 (84.0%)`，95% CI `[75.6%, 89.9%]` | `35/500` 子集（固定 `deepseek-chat` Judge）：`deepseek-chat` `30/35 (85.7%)`，CI `[70.6%, 93.7%]`；`deepseek-reasoner` `34/35 (97.1%)`，CI `[85.5%, 99.5%]` | `reasoner` 输出 Token 增至约 `4.2×`，推动前后台 `consolidator` 双模型分槽；`n=35` 置信区间重叠 |
| `f039de1`（Wave 4 合流） | `v1` | `deepseek-chat`：`15/16 (93.75%)`，KV 命中 `82.37%`，输出 Token `9,919` | — | — | 精简默认参数 `required` 后输出 Token 较 `623f357`（`12,674`）下降 `21.7%`；唯一未过项为 `企业微信` 未命中缩写 `企微` |
| `1b421d2`（Wave 5 合流） | **`v2-neutral`**（`sha256=08e7c0adcb56`，零测试集实体） | `deepseek-chat`：`15/16 (93.8%)`，CI `[71.7%, 98.9%]`，KV 命中 `81.31%`，前缀复用 `92.91%` | `deepseek-chat`：`80/100 (80.0%)`，CI `[71.1%, 86.7%]`（`adversarial` `19/22`、`single-hop` `19/22`、`multi-hop` `17/22`、`temporal` `16/21`、`open-domain` `9/13`） | **`70/500` 扩充子集**（每类 10 题，`deepseek-chat`）：`63/70 (90.0%)`，CI `[80.8%, 95.1%]`（`abstention`/`single-session-user`/`assistant` 各 `10/10`，`temporal-reasoning` `7/10`） | 彻底剥离评分提示词泄漏后，LoCoMo 较旧 `v1` 回落 `4` 题，LongMemEval 在双倍样本量（70 题）且零具名提示下拒答类 `10/10` 全过 |

---

## 五、分阶段演进里程碑与近期迭代记录

### 5.1 早期至中期关键里程碑摘要

| 阶段 / Commit | 核心变更 | 结果与意义 |
| --- | --- | --- |
| `8ea727e` ~ `383d956` | Step 1–4 最小 ReAct 循环、白名单 `ToolRegistry`、纯 Markdown 记忆与 SQLite `StateStore` | 确立零第三方依赖、工具异常转观察回传模型的自愈闭环 |
| `7f32529` ~ `f08b187` | `MEMORY.md` 限长索引会话内冻结 + 阈值后台沉淀 `consolidator.py` + `memory_lifecycle.py` 冲突替换与过期归档 | 解决前缀膨胀击穿 KV Cache 与旧记忆冲突、过期日程污染问题 |
| `d2e20e3` ~ `263cf37` | 实现 SQLite FTS5 + BM25 双语检索插件，并将评测代码与数据统一收拢至顶层 `evals/` | LoCoMo Search 200 题 Recall 由 `42.8%` 提升至 `68.5%`，产品包与评测包双向解耦 |
| `3510a16` ~ `cf85539` | 落地 `ai ← agent ← assistant` 三层单向依赖漏斗、五大 `Plugin(ABC)` 插槽、`models_dsl` 与 `reasoning_provider` | 通过 AST 与子进程断层测试验证每层可剥离上层独立运行 |
| `ee36ba3` ~ `119cd07` | 移除内核 `MAX_TOOL_ROUNDS`、激活强类型 `ToolCall` / `ThinkingDelta` / `ToolStart`、加固 `web_chat.py` 防 CSRF 与 `shell_tools.py` 防 `pytest` 两步执行 | 打通流式推导协议并封堵首批本地 Web 与命令行高危向量 |
| `932a72c9` ~ `623f357` | 插件单次 `initialize()` 解耦、`StateStore.append_turn` 成对原子落库、`study_tools.py` 间隔复习卡片与前后台 `consolidator` 双模型分槽 | 完成个人学习助手核心功能叠加 |

---

### [2026-11-26] 插件正交解耦、Layer 2 洋葱微内核提纯与默认参数推断（Issues `#68` / `#69`，Commits `d15ed09`、`f039de1`）
- **目标与思路**：
  1. 消除 `plugins/tools/history_tools.py -> plugins.fts5_retriever` 跨插件导入耦合，并将 `972` 行的 `memory.py` 拆分为单一职责模块；
  2. 清除 Layer 2 `agent/plugins.py` 中残留的业务目录字面量、具名插件配置键、动态 `type()` 造类与 `object.__setattr__`，实现 `AssistantApp._tool_registry` 启动期单次构建与 `ToolSchema.required` 默认参数自动推断。
- **核心改动**：
  - 新增零内部依赖底座 [../personal_assistant/assistant/text_search.py](../personal_assistant/assistant/text_search.py) 与 [../personal_assistant/assistant/files_policy.py](../personal_assistant/assistant/files_policy.py)，新增 [../personal_assistant/assistant/builtin_tools.py](../personal_assistant/assistant/builtin_tools.py)，`memory.py` 精简至 `570` 行；
  - `ToolSchema` 支持精确 `required`，`build_tool` 自动识别 Python 默认参数与执行期 `accepted_context_keys`，`iter_react(..., tool_context=turn_ctx)` 按轮注入动态上下文，`plugins.py` 精简至 `431` 行并由 `test_architecture.py` AST 硬守卫锁死零业务字面量。
- **验证结果（`cede4bb` → `f039de1`）**：确定性单测 `167 → 179 passed`；16 场景实模配对回归保持 `15/16 (93.75%)`，因省略默认参数重复输出，`completion_tokens` 由 `12,674` 降至 `9,919`（`-21.7%`），服务端 KV Cache 命中率升至 `82.37%`。

---

### [2026-11-26] 红队安全闭环、外层回合预算与沉淀韧性、评测去泄漏及文档口径修订（Issues `#72` / `#73` / `#74`，Commits `c761c81`、`1ad1e0d`、`1b421d2`）
- **背景与目标**：针对独立四路红队与代码审查暴露的安全绕过、无限循环/沉淀退避缺失及评测可信度问题，按文件零交集拆分为三条正交 PR 依次通过独立二审合入主干，全程保持 Layer 2 `personal_assistant/agent/react.py` **0 行改动**。
- **核心改动**：
  1. **评测去泄漏与单测宿主隔离（Issue `#74` / PR `#75`，`c761c81`）**：
     - 清除 `evals/memory_eval.py` 评分提示词中的公开集人物名（`Caroline` / `Melanie`），升级为 `JUDGE_PROMPT_VERSION = "v2-neutral"` 并在报告中记录 `prompt_sha256`、`model` 与 `self_judged`；补齐单场景异常隔离与短 ASCII 词边界匹配（避免 `"Go"` 误中 `"Google"`），收紧 `memory_scenarios.jsonl` 断言并兼容 `"企微"` / `"企业微信"`；
     - `evals/agent_eval.py` 改为逐题独立临时 `MemoryStore`；`evals/cli.py` 与 `.github/workflows/test.yml` 增加 `--fail-under 0.90` 与 `timeout-minutes: 10`；新增 [../tests/conftest.py](../tests/conftest.py) 全局隔离宿主机 `.env`、环境变量与 `models.*`。
  2. **沙箱绕过、SSRF 二次解析与规则注入封堵（Issue `#72` / PR `#77`，`1ad1e0d`）**：
     - [../personal_assistant/assistant/files_policy.py](../personal_assistant/assistant/files_policy.py) 统一实现 `Cf` 格式符剥离 + `NFKC + casefold()` 与受保护目录有界递归 `(st_dev, st_ino)` 采集，彻底拦截 macOS APFS `ſ -> s` 折叠与硬链接绕过；
     - [../plugins/tools/shell_tools.py](../plugins/tools/shell_tools.py) 实装逐子命令只读 `git` 白名单（封堵 `git branch -D/-m/--edit-description`、`git log -p` 及 `git show`/`diff` 裸 `<blob>`/`<tree>`/空树 SHA，自动为 `git diff` 注入 `:(exclude,glob,icase)` 敏感路径排除），启用隔离空 `HOME`、1 MB 流式管道上限与 `os.killpg` 整组强杀（含父进程秒退后残留持管道孙进程回收）；
     - [../plugins/tools/web_tools.py](../plugins/tools/web_tools.py) 实装 `_PinnedHTTPConnection` / `_PinnedHTTPSConnection` 直连已校验 IP 消除 DNS Rebinding、`ProxyHandler({})` 禁系统代理、`allow_fake_ip` 默认关闭、NAT64/6to4/Teredo 私网拦截与单调总截止时间；
     - [../personal_assistant/assistant/builtin_tools.py](../personal_assistant/assistant/builtin_tools.py) 与 [../personal_assistant/assistant/prompts.py](../personal_assistant/assistant/prompts.py) 增加 `rules/` 注入非越权边界声明与 `save_rule` 双重门控（同轮运行过 `risk="network"` 工具即硬拒，且校验显式长期规则设定意图）；[../scripts/web_chat.py](../scripts/web_chat.py) 补齐安全响应头、SSE `closing()` 与 `--auth-token` / `MY_HELPER_WEB_TOKEN` 常量时间 UTF-8 Bearer 鉴权。
  3. **外层回合预算、原子上下文快照与沉淀退避（Issue `#73` / PR `#76`，`1b421d2`）**：
     - [../personal_assistant/assistant/app.py](../personal_assistant/assistant/app.py) 新增 `_GuardedTurnLLM` 五维预算熔断、沉淀滞后自适应历史滑窗（`20 -> 60` 条 / `24,000` 字符，首条对齐 `user`）、`finally` 分支（覆盖 `GeneratorExit` 等 `BaseException`）流中断 `turn_tag` 回滚、`untrusted_tool_seen` 联动注入、`_llm_lock` 串行热重载与 `close()` 插件 `terminate_all()`；
     - [../personal_assistant/assistant/db.py](../personal_assistant/assistant/db.py) 新增 `get_context_snapshot()` 单锁原子读取二元组；[../personal_assistant/assistant/consolidator.py](../personal_assistant/assistant/consolidator.py) 新增 `max_batch_messages=24`、按会话指数退避冷却（默认 `2s` 起步至 `256s`）与写锁内 `_filter_safe_operations` 在途 `(turn#...)` 事实保护；
     - [../personal_assistant/ai/client.py](../personal_assistant/ai/client.py) 与 [../personal_assistant/ai/cache.py](../personal_assistant/ai/cache.py) 补齐无 `index` SSE 工具流切槽、外发剥离 `finish_reason`、`HTTPError` 脱敏错误体保留与按 `session_key` 隔离的缓存统计。
  4. **诚实文档改写与去泄漏复测**：修订 [../README.md](../README.md)、[mvp-design.md](mvp-design.md) 与本文件，去除夸大措辞，明确披露 `sample_cases.jsonl` 开发集属性、`LongMemEval_oracle` 证据会话子集属性、`LoCoMo conv-26` 单段 19 会话结构、置信区间及跨轮剥离工具观察/非事务写文件边界。
- **验证与复测对比（`f039de1` → `1b421d2`）**：
  - 确定性单元测试：`179 passed → 198 passed`（在 `tests/conftest.py` 宿主隔离下全绿，`react.py` 0 行改动），AST 中文 `#` 注释缺失为 `0`；
  - 离线检索基准（`fts5_bm25`）：40 题 `38/40 (95.0%)`，LoCoMo Search 200 题 `105/200 (52.5%)`（零退步）；
  - 去泄漏 `v2-neutral` 实模复测（`1b421d2`，`prompt_sha256="08e7c0adcb56"`）：16 场景开发集 `15/16 (93.8%)`（KV 命中 `81.31%`，唯一未过项 `chained-three-step-update` 磁盘旧词检查残留 `PostgreSQL`）；`LoCoMo conv-26` 100 题 `80/100 (80.0%)`（95% CI `[71.1%, 86.7%]`，较含具名提示泄漏的旧 `v1` 回落 `4` 题）；`LongMemEval_oracle` 70 题扩充抽样 `63/70 (90.0%)`（95% CI `[80.8%, 95.1%]`，`abstention` 拒答类在零具名示例下 `10/10` 全过，未过项主要为时序先后与日期换算 `temporal-reasoning` `7/10`）。
- **下一步**：当前安全边界、运行期有界性、测试隔离与评测口径均已收敛完备；后续仅在真实伴读场景遇到跨主题零词面重叠召回瓶颈时，按需通过 `RetrieverProtocol` 扩展可选混合检索插件。

---

### [2026-11-26] 去补丁与去双轨兼容提纯：回归单一契约与精简外围沙箱插件（Issues `#79` / `#80`，Commits `956e739`、`6bfe67b`）
- **背景与目标**：清理前期快速演进与红队攻防中遗留的「新旧双轨兼容壳」与「特判补丁堆叠」，使全仓代码回归从数据结构自然生长出来的简洁对称形态，保持核心微内核 `react.py` **0 行改动** 与全部安全防线不退步。
- **核心改动（两路合计净删生产代码逾 810 行，全仓净减近 900 行）**：
  1. **核心层单轨契约提纯（Issue `#79` / PR `#81`，`956e739`）**：
     - `StreamDone` 原生携带强类型 `finish_reason: str | None = None`，彻底拔除往业务 `message` 字典塞 `"finish_reason"` 及下游 `client.py` / `app.py` 三处外发剥离补丁；SSE `tool_calls` 拔除 60 余行跨帧猜槽状态机，回归 7 行标准 `index` 归槽；
     - 删除废弃 `AssistantApp.build_tools()`、`cli.py` 零调用 `build_repl_tools` / `run_repl(tools=...)` 分支及 `memory.py` 搬家重导出兼容壳；`AssistantApp` 暴露只读属性 `app.tools`，`MemoryToolsetProvider` 彻底移除全部构造期闭包回调（`get_user_message` / `get_today` / `turn_tag` / `on_remember`），动态回合状态 100% 统一经 `tool_context` 注入；
     - 删除 `builtin_tools.py` 硬编码关键词表 `RULE_INTENT_KEYWORDS`（保留正交的 `untrusted_tool_seen` 联网轮锁死 `save_rule` 机制）；将未提交 `(turn#...)` 暂存行隔离全部内聚至 `memory_lifecycle.py` 单点 `_is_inflight_turn_line` 与真实落盘 `applied_texts`，删除 `consolidator.py` 模糊子串过滤补丁（`consolidator.py` 对 `turn#` 100% 零感知）。
  2. **外围沙箱与联网插件去臃肿提纯（Issue `#80` / PR `#82`，`6bfe67b`）**：
     - `web_tools.py`（`824 -> 573` 行，`-30.5%`，`0` 处 `# type: ignore`）：删除 5 个自定义 `urllib.request` 子类（约 180 行），改用标准库 `http.client` + `socket.create_connection((pinned_ip, port))` + `ssl` 直接实现重定向循环（天然不读环境代理、天然钉住已校验 IP 并确定性关闭 `HTTPResponse` 套接字）；
     - `shell_tools.py`（`942 -> 713` 行，`-24.3%`）：删除预派生 `git cat-file -t` 子进程补丁与启发式修订号切分，在语法层直接拦截冒号对象引用 `:`、解引用 `^{`/`@{` 与裸对象哈希（含 4 位起短哈希与 `describe -g<sha>`），无显式安全文件时自动为 `git show`/`git diff` 补 `--stat`；
     - `files_policy.py`（`474 -> 337` 行，`-28.9%`）：移除 512 项 DFS 目录栈与双父链拼装，收敛为轻量单层敏感 Inode 采集与单一父链回溯。
- **验证对比**：确定性单测保持 `198 passed`，离线检索保持 `38/40 (95.0%)`，AST 中文 `#` 注释与架构纯度守卫 `0` 违规。

---

### [2026-11-27] 三层核心净洁提纯、外围插件单轨化、评测统一装配与全仓零 `# type: ignore` 收官（Issues `#83` / `#84`，Commits `a775b7e` + 本次提交）
- **背景与目标**：在 Wave 6 清除高频补丁后，继续沿「最小对称原语」收官两类残余耦合：其一是三层核心中仍存在的配置散包传参、`ModelsConfig` 业务角色字段、`PluginContext` 业务化绑定与 `transform_context` 冗余钩子；其二是外围 `plugins/` 的旧配置文件回退与重复工具函数、`evals/` 手搓装配（导致 `agent_eval.py` 漏传 `case["today"]` 存在时效用例到期隐患）、`scripts/` 测试迁就补丁及外围目录残余 `# type: ignore`。
- **核心改动**：
  1. **阶段一：三层净洁核心提纯（Issue `#83` / PR `#85`，`a775b7e`）**：
     - Layer 1 `ai/`：`ProviderFactory` 改为 `ModelProfile` 直通工厂（`from_model_profile(profile)`）；`ModelsConfig` 以通用 `roles: dict[str, str]` 取代硬编码业务角色字段；
     - Layer 2 `agent/`：`Tool(ToolSchema)` 回归纯值对象并提炼 `bind_tool_context()`；`react.py` 删除未使用的 `transform_context`，仅保留单步 `convert_to_llm`；`PluginContext` 清零业务字段与 `tool_bindings()`，由 Layer 3 `commands.py._build_plugin_bindings()` 注入运行时绑定；
     - Layer 3 `assistant/`：`MemoryStore` 抽出内置零索引 `LinearRetriever` 兜底；`common.py` 沉淀 `today_shanghai`、`parse_iso_date`、`require_positive_int`、`atomic_write_text`；`assistant/__init__.py` 改为按需懒加载导出；`personal_assistant/` 全目录清零 `# type: ignore`。
  2. **阶段二：外围插件单轨化、评测统一装配与四树零 `# type: ignore` 收官（Issue `#84`，本次提交）**：
     - **外围插件 `plugins/`**：`shell_tools.py` 与 `web_tools.py` 移除 `from_config` 与 `config_path` 回退，统一单轨消费主板注入配置字典；`study_tools.py` 复用 `common.py` 原子写盘与日期工具，移除冗余 `project_root` 与隐式 CWD `Path("data")` 兜底（缺省显式报错）；`fts5_retriever.py` 与 `history_tools.py` 复用 `text_search.py` 新增的 `validate_retrieval_thresholds` / `parse_retrieval_thresholds`，删除 `_HistoryThresholdConfig` 鸭子协议与未使用分词重导出；`ConfigCompilerProtocol.render` 与 `ModelsDSLCompiler.render` 改为显式强类型具名参数（Layer 2 协议零 `consolidator` 业务名泄漏，具体插件以带默认值可选参数扩展）；`telemetry.py` 移除 `getattr`/`setattr` 反射累加及 0 生产调用的 `recent_turns`/`recent_tasks`（`config/tools.json` 同步切换为 `max_sessions`）；删除冗余 `plugins/tools/__init__.py`；
     - **评测统一装配 `evals/`**：`AssistantApp` 支持接收外部预加载 `tools_config`（按所有权释放）与 `auto_consolidate=False`；`evals/agent_eval.py` 改为基于 `AssistantApp(..., auto_consolidate=False, today_fn=...)` 执行并透传每题 `case["today"]` 修复 2026-11-30 时效到期隐患；`evals/memory_eval.py` 与 `evals/public_benchmarks.py` 移除 `setup_step_count` 旧格式双轨（统一 `shared_setup`）、通过 `AssistantApp` + `app.consolidator` 使双模型沉淀真实生效，并将 `_contains_keyword` 升级为无长度魔法阈值的全长度纯 ASCII 字母数字边界匹配；`evals/cli.py` 单次加载 `ToolsConfig` 向下透传、支持 `--consolidator-model` 并清理未使用导入；
     - **入口脚本与静态守卫**：`scripts/web_chat.py` 移除迁就旧测试桩的 `auth_token` 三元分支，复用 `_is_loopback_host` 与 `StateStore.get_context_snapshot(chat_id, limit=40)`；`scripts/demo.py` 的 `run_live_ask` 对齐 `ToolsConfig` 加载；`tests/test_architecture.py` 将 `0` 处 `# type: ignore` 静态硬守卫扩大覆盖 `personal_assistant/`、`plugins/`、`scripts/`、`evals/` 全部四棵目录树。
- **验证与同口径说明（`1b421d2` → 本次提交）**：
  - 确定性单元测试：`198 passed → 202 passed`（100% 通过）；
  - 离线检索基准（确定性）：`fts5_bm25` 保持 `38/40 (95.0%)`（Recall `0.9483`，MRR `0.9483`，负例干净度 `100%`）；内置 `linear` 保持 `32/40 (80.0%)`（Recall `0.9655`，MRR `0.908`）；
  - 离线演示与静态硬守卫：`scripts/demo.py` 离线场景全部通过；AST 全仓（含 `tests/`）中文 `#` 注释缺失 `0`；`personal_assistant/`、`plugins/`、`scripts/`、`evals/` 四棵目录树 `# type: ignore` 命中 `0`；
  - 实模评测口径：Wave 6 / Wave 7 均为架构提纯与装配统一，未发起新的实模 API 跑分，表四中的 `v2-neutral`（`sha256=08e7c0adcb56`）实模基线保持锚定于 `1b421d2`（`15/16`、`80/100`、`63/70`）。

---

## 六、后续迭代追加记录模板

```markdown
### [YYYY-MM-DD] <简短标题>（Issue `#N`，Commit `<short-sha>`）
- **背景与目标**：说明问题根因、设计取舍与涉及模块。
- **核心改动**：列出关键文件与核心机制变化。
- **验证与同口径对比（`<old-sha>` → `<new-sha>`）**：
  - 单元测试：`XX passed`
  - 离线/实模评测（注明数据集口径、Judge 版本与失败/退步项）：...
- **已知边界与下一步**：如实说明残余限制与后续跟进项。
```
