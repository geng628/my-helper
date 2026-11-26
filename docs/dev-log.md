# My Helper v2 工程演进、排坑与评测日志（跨模型交接文档）

> 本文档记录项目从零搭建至今的全部**核心设计思路、架构重构、真实踩坑根因与修复、同口径 Benchmark 基线演进**以及**下一步待办路线**。
> 换新模型或开新会话时，先通读本文件与根目录 [AGENTS.md](../AGENTS.md)，即可无缝沿着最新提交继续开发。

---

## 一、核心架构定调（不要破坏的设计原则）

1. **纯 Python 标准库零运行依赖**：生产环境 0 第三方库（`urllib`、`sqlite3`、`threading`、`http.server` 等），`pytest` 仅作开发测试依赖。
2. **纯 Markdown 为记忆唯一事实源**：`data/memory/MEMORY.md` 只做限长目录索引（100 行 / 4000 字符），详细记忆分主题落盘为 `.md` 文件，人类随时可直接打开编辑，不锁死在黑盒向量库里。
3. **KV Cache 友好优先**：会话内冻结一次系统提示词（静态指令 → 规则 → `MEMORY.md` 索引 → 日期），历史对话采用标准追加式 `[system, user_1, assistant_1, ...]`，记忆按需由工具检索而不每轮动态塞入前缀。
4. **外围插件化扩展**：`plugins/*.py` 同时支持两类插件扩展点——**工具插件**（`@tool` / `TOOLS`，受 `enabled_tools` 白名单管控）与**检索后端插件**（`RETRIEVERS`，受 `config/tools.json` 的 `retrieval` 段热切换）。
5. **安全沙箱与受控演化**：Agent 可自主改写记忆与规则，但对 `plugins/`、`config/`、`personal_assistant/` 无写权限（防联网 `fetch_webpage` 触发间接提示词注入导致 RCE）。
6. **严格 TDD + 诚实评测**：所有新功能/修 Bug 先红后绿；所有改进必须带同口径改前/改后数据与 Git SHA，如实汇报失败项与退步项。

---

## 二、架构演进与关键重构时间线

| 阶段 / Commit | 做了什么 | 当时的核心思路 / 取舍 |
| --- | --- | --- |
| `8ea727e` ~ `f4ce909` | Step 1 最小 ReAct 循环 + `ToolRegistry` 拆分 + 离线演示脚本 | 坚持原生 OpenAI 兼容 Function Calling，工具异常作为 Observation 回传模型自愈，不套框架 |
| `918cdb3` ~ `b592b26` | Step 2 纯白名单工具系统 + 固定 `argv` 命令模板 + `@tool` 插件加载器 | 把 `tools.py` 与 `commands.py` 拆清职责；强制 `shell=False` 与环境变量白名单 |
| `5d379cf` ~ `383d956` | Step 3 纯 Markdown 记忆与文件边界 + Eval 包拆分 + Step 4 SQLite 历史与 `AssistantApp` | 评测器独立成 `personal_assistant/eval/` 包且绝不污染原数据目录 |
| `2a62faf`、`c47d212` | 彻底移除 Telegram、旧版知识库与 v1 历史耦合 | 聚焦纯本地 CLI / Web / API 个人 Agent，保持仓库干净可公开展示 |
| `7f32529`、`85907b6` | 重构记忆为 `MEMORY.md` 限长索引 + `facts.md` + 前缀冻结 + KV Cache 遥测（`cache_metrics.py`） | 解决记忆增长撑爆提示词与打穿服务端 KV Cache 的核心痛点；同时记录服务端 `cached_tokens` 与本地前缀复用率 |
| `5d6a14d` | 阈值触发后台记忆沉淀与上下文压缩（`consolidator.py` + `session_compactions`） | 放弃手动 `/compact`，达 16 条或 8000 字符自动后台线程提炼长期记忆并原子推进压缩游标 |
| `f08b187` | 抽离 `memory_lifecycle.py`：读旧改旧冲突替换（`add/replace/remove`）+ `(until: YYYY-MM-DD)` 过期过滤与按月归档 | 解决记忆“只增不改、旧值打架、过期日程污染检索”三大长期运行顽疾 |
| `293d099`、`be1325b` | 新增 `--mode memory` 多轮端到端评测，并基于真模型失败修复 3 个真实 Bug | 用实测闭环把记忆评测从 4/5 拉到 5/5（详见第三节排坑记录） |
| `5dbed3f`、`6dbf02b`、`1343b0d` | 补齐 3 个高实用外围能力：档位 1 白名单 Shell 插件、零依赖本地网页端、联网搜索与防 SSRF 抓网页插件 | 全部采用外围插件或独立脚本形态，保持核心零膨胀 |
| `cffbf26`、`552863d` | 扩建四层评测体系（40 题检索 + 16 场景多轮记忆 + LongMemEval/LoCoMo 适配器），并将公开数据与下载入口统一收拢至 `evals/public/` 与 `--mode download` | 原 4 题/5 场景太玩具；扩充后主动保留高难改写、干扰负例与学术公开集，真实暴露短板；修正散落目录，恢复 `data/` 为纯运行时目录 |
| `d2e20e3` | 扩展 `plugins.py` 支持 `RETRIEVERS` 检索插件，实现零依赖 `plugins/fts5_retriever.py`（SQLite FTS5 + BM25）与 `--retriever` A/B 对比 | 对标重型向量方案后采取“先零依赖 FTS5 BM25 插件打底、后续按需叠 Embedding+RRF 插件”的渐进路线，LoCoMo 200 题 Recall 从 42.8% 跃升至 68.5%，延迟降 3 倍 |
| `8d1271c` | 新增 `AGENTS.md` 与 `docs/dev-log.md` 跨模型交接文档 | 沉淀全量设计取舍、12 个真实踩坑根因与 Benchmark 总账，支持换任意模型无缝续写 |
| `3510a16` | 借鉴 Pi 三层架构方法论完成 `ai → agent → assistant + eval` 三层重构，引入 `ToolSchema → Tool` 类型递进扩展、`LLM_PROVIDERS` 插件口与 `tests/test_architecture.py` 守卫测试 | 彻底消灭 `eval/* → cli.py` 反向依赖、`memory.py → commands.py` 横向串台与 `react.py` 夹带个人业务提示词的问题；通过 AST + 子进程 `sys.modules` 隔离测试验证三层均可剥离上层独立运行 |
| `35cc8ea` | 新增外部配置 DSL 编译插件 `plugins/models_dsl.py` + `CONFIG_COMPILERS` 第 4 类插件扩展点 + 系统级零心智防泄漏保护 | 核心层依然只认识标准 JSON（`ModelProfile` / `ModelsConfig`）；用户可在 `config/models.dsl` 直接写 `api_key` 免配环境变量，防泄漏由系统默认兜底（`.gitignore`、`0600` 落盘权限、文件与 Shell 沙箱屏蔽、全自动脱敏） |
| 本次提交 | 面向零代码用户的极简一行预设语法（`use deepseek sk-...`）+ 终端一键向导 `setup` + 网页端「模型设置」自动生成与热切换 | 把“编程味”彻底降为零：内置 7 大主流厂商预设自动补全 URL/模型名/协议/温度，终端与网页均可 10 秒点选自动生成 `models.dsl` 与 `models.json`（均 0600 权限）并即时热生效 |
| `efe8568` | 正规化 GitHub Issue → 分支 → CI → PR 研发流程（`#1` / `#2`） | 补齐 Issue/PR 模板与 CI 三步守卫（`pytest` + 40 题 `fts5_bm25` + AST 中文注释校验），彻底告别裸推 `main` |
| `263cf37` | 将 `personal_assistant/eval/` 整体迁移合并至顶层 `evals/` 包并统一 `python -m evals`（`#3` / `#4`） | 让 `personal_assistant/` 只保留纯净三层产品代码（`ai`/`agent`/`assistant`），评测代码与评测数据统一收拢至顶层 `evals/`，并用 AST 守卫双向防反向依赖 |
| Issue `#5` | LoCoMo 共享长记忆评测模式、紧凑 `shared_setup` JSONL、完整会话导出与实时进度（纯 `evals/` 单线程，零改动 `personal_assistant/`） | 消除 LoCoMo 多轮评测的重复沉淀浪费与 4 会话硬截断，保持极简单线程确定性且绝不因评测触碰产品核心代码 |

---

## 三、真实踩坑、根因分析与修复记录（防重踩清单）

### 坑 1：`MEMORY.md` 无限追加 + 历史打包改写打穿 KV Cache（`7f32529`）
- **现象**：早期每调一次 `remember_fact` 就往 `MEMORY.md` 追加一行，且把多轮历史拼成单条 JSON 塞进 `user` 消息，导致每轮前缀全变、缓存命中率接近 0%。
- **根因**：系统提示词中途变动 + 非追加式历史结构破坏了模型服务端的前缀缓存（Prefix Caching）。
- **修复**：`MEMORY.md` 严控为前 100 行 / 4000 字符的索引；`remember_fact` 改写 `facts.md` 按需检索；会话内冻结系统提示词，历史改用标准追加式 `user/assistant` 消息列表。实测 DeepSeek 服务端缓存命中率稳定在 **60%–80%+**。

### 坑 2：幽灵裸事实行导致临时记忆永不过期（`be1325b`）
- **现象**：首次跑 `--mode memory` 时 `temp-fact-expiry` 失败（4/5）：过期日期后依然搜得到“明天演练会”。
- **根因**：对话时模型调用 `remember_fact` 把原话带 `(msg#3)` 写进了 `facts.md`；后台沉淀器后来把规范化带 `(until: 2026-11-20)` 的条目写进了主题文件，但忘了删 `facts.md` 里的原始裸句，导致裸句没有 `until` 永远不过期。
- **修复**：提示词明确要求规范化或删除对应 `(msg#...)` 行 + 在 `memory_lifecycle.py` 中增加确定性兜底函数 `prune_raw_facts_for_batch()`，按批次消息 ID 与文本重叠自动清理已被沉淀吸收的原始行。

### 坑 3：大模型在 `replace.new` 中复读旧时间戳前缀（`f08b187`）
- **现象**：实测 DeepSeek 输出的替换内容自带 `[2026-11-19]`，落盘后变成 `- [2026-11-19] [2026-11-19] ...`；且早期单测只用子串断言漏报了该问题。
- **修复**：在 `memory_lifecycle.py` 增加 `clean_memory_body()` 正则剥离复读的列表符、日期与 `until` 前缀；单测改用精确 `splitlines()` 断言。

### 坑 4：`read_markdown` 拒绝模型传入整数行号参数（`be1325b`）
- **现象**：DeepSeek 频繁传入 `{"start_line": 115, "end_line": 200}`（JSON 整数而非字符串），虽能靠 Observation 自愈，但每次白白浪费一轮 ReAct 往返。
- **修复**：在 `tools.py` 参数校验中允许非 `bool` 的 `int`/`float` 自动转为字符串（显式拒绝 `bool`，防止 `True` 变 `"True"`）。

### 坑 5：Python 3.14 动态插件 `@dataclass` 崩溃（`5dbed3f`）
- **现象**：引入 `plugins/shell_tools.py`（使用了 `from __future__ import annotations` + `@dataclass`）后，动态加载报 `AttributeError: 'NoneType' object has no attribute '__dict__'`。
- **根因**：`plugins.py` 调用 `spec.loader.exec_module(module)` 前未将模块注册进 `sys.modules[module_name]`，Python 3.14 的 `dataclasses` 反射查不到模块命名空间。
- **修复**：加载前先写 `sys.modules[module_name] = module`，失败时 `pop` 清理。

### 坑 6：DeepSeek 偶发 `ssl.SSLEOFError` 中断长评测（`5dbed3f`）
- **现象**：多轮评测跑几十次 API 请求时偶发 TLS 握手 EOF 直接整轮崩掉。
- **修复**：在 `llm.py` 的 `default_post_json()` 中加入最多 2 次指数退避重试，仅对瞬时网络/TLS/5xx/429 重试，跳过 400/401/403/404 等确定性客户端错误。

### 坑 7：macOS TUN Fake-IP 代理误触 SSRF 防护（`1343b0d`）
- **现象**：在开启本地 TUN 代理的 macOS 上，`docs.python.org` 被 DNS 解析成 `198.18.1.170`（RFC 2544 保留网段），被 `ip.is_global` 拦成 SSRF。
- **修复**：在 `plugins/web_tools.py` 中，对**非字面 IP 的域名 DNS 解析结果**允许 `198.18.0.0/15`，但对用户直接输入的字面 IP 以及 `127/8`、`10/8`、`172.16/12`、`192.168/16`、`169.254.169.254` 等私网/元数据地址继续硬拦截，并在每次重定向前重新校验。

### 坑 8：Bing RSS 免费搜索相关性太差（`1343b0d`）
- **现象**：零密钥查技术问题（如 Python 3.14 发布说明），Bing RSS 频繁返回 `Welcome to Python.org` 等泛主页。
- **修复**：改为 DuckDuckGo HTML 为默认零密钥主通道（拆包 `uddg` 重定向真实链接），失败或空结果时自动回退 Bing RSS，并支持配置 Key 走 Tavily / Brave。

### 坑 9：`list_files("data/memory")` 暴露归档目录导致过期事件复活（`1343b0d`）
- **现象**：尽管 `search_markdown` 屏蔽了 `archive/`，模型先调 `list_files("data/memory")` 看到 `archive/2026-11.md` 后直接 `read_markdown` 读出来，把已过期演练会当成待办。
- **修复**：`MemoryStore.list_files()` 默认排除 `data/memory/archive/`，仅在显式指定 `root="data/memory/archive"` 时列出。

### 坑 10：公开数据集与下载脚本散落到 `data/evals/` 与 `scripts/`（`552863d`）
- **现象**：图省事复用 `.gitignore` 的 `data/` 规则导致评测数据拆散在 `evals/` 和 `data/evals/`，且下载脚本另挂在 `scripts/`。
- **修复**：统一收拢到 `evals/public/`（加入 `.gitignore`）并合并下载命令进 `python -m personal_assistant.eval --mode download`，删除多余脚本，`data/` 恢复为纯运行时目录。

### 坑 11：SQLite FTS5 在极小语料（N < 10）下 `bm25() == 0` 导致漏召回（`d2e20e3`）
- **现象**：单测只有 2 行文档时，SQLite FTS5 `bm25()` 返回 `0.0`，被 `min_bm25_score=0.1` 全部滤光。
- **根因**：SQLite FTS5 原生 IDF 公式为 $\ln((N - n + 0.5)/(n + 0.5))$（未加 Lucene +1 平滑），当 $N=2, n=1$ 时 $\ln(1.5/1.5) = 0$。
- **修复**：仅在语料块数 `>= 10` 时启用 `min_bm25_score` 过滤，极小语料走覆盖率与词数门控。

### 坑 12：FTS5 窗口去重误杀 + 中文 Bigram 分母膨胀 + `rglob().resolve()` 性能瓶颈（`d2e20e3`）
- **现象**：首版 FTS5 在 40 题上出现 2 例退步（`para-ssrf-internal-ip`、`multi-memory-index-arch`），且 LoCoMo 200 题单题耗时 15.56 ms（实测其中 12.27 ms 竟花在遍历 272 个文件的 `Path.rglob("*").resolve()` 上，SQLite 本身仅 1.39 ms）。
- **根因**：
  1. 摘要窗口只展示 `[line-2, line+2]`，去重阈值写成 `abs(line - prev) <= 4` 把窗外第 3–4 行的有效命中误删了；
  2. 17 字中文句子会切出 16 个重叠 bigram，导致命中 3 个稀有词（BM25=11.29）却因 `3/16 < 0.22` 被误拦；
  3. macOS 上每查询一次就对 272 个文件调 `Path.resolve()` 触发大量 `lstat`。
- **修复**：去重改为 `<= 2`；增加“命中 ≥3 词且 BM25 ≥ 10”高置信放行；`MemoryStore.iter_searchable_files()` 改用 `os.scandir`（保留完整软链接与 `reject_segments` 校验）。40 题退步归零（38/40，95%），LoCoMo 延迟降至 **9.51 ms**（相比 `552863d` 提速 **~3×**）。

---

## 四、Benchmark 演进总账（均绑定 Git SHA）

### 1. 离线检索评测（零 Token）
| Commit | 检索后端 | 40 题自建集（`sample_cases.jsonl`） | 200 题公开 LoCoMo（ACL 2024，272 篇 `.md`） |
| --- | --- | --- | --- |
| `cffbf26` / `552863d` | 内置 `linear` | 通过 **32/40 (80.0%)**；Recall 96.6%，MRR 0.908，Quote 87.9%，负例干净 **63.6%**，0.72 ms | 全过 **57/200 (28.5%)**；Recall **42.75%**，MRR **0.3587**，Quote 33.55%，29.0 ms |
| `d2e20e3` | 插件 `fts5_bm25` | 通过 **38/40 (95.0%)**；Recall 94.8%，MRR **0.948**，Quote **94.8%**，负例干净 **100%**，0.53 ms（+6 / 0 退步） | 全过 **105/200 (52.5%)**；Recall **68.51%**（`single-hop` 94% / `temporal` 92%），MRR **0.6221**，Quote **60.17%**，**9.51 ms**（+51 / -3） |

### 2. 真实模型多轮记忆评测（`deepseek-chat`）
| Commit | 16 场景自建记忆集（`memory_scenarios.jsonl`） | 14 题公开 LongMemEval（ICLR 2025，LLM Judge） | 备注 / 暴露的真实问题 |
| --- | --- | --- | --- |
| `cffbf26` | **13/16 (81.25%)**；服务端 KV 命中 60.4% / 前缀复用 90.4% | **10/14 (71.43%)**；服务端 KV 命中 26.2% / 前缀复用 82.6% | 暴露：无星期致相对日期算错；替换句残留旧词；LongMemEval `single-session-preference` 0/2 未主动查记忆 |
| `d2e20e3`（启用 `fts5_bm25`） | **13/16 (81.25%)**；服务端 KV 命中 **77.0%** / 前缀复用 87.9% | 未重跑（仅改检索后端） | 无检索退步；`temporal` 3/3；`rolling-multi-compaction` 实答“企业微信机器人”正确但未命中严苛子串“企微” |
| `263cf37`（`evals/` 统一 + `fts5_bm25`） | **14/16 (87.5%)**；服务端 KV 命中 **76.1%** / 前缀复用 87.2% | **12/14 (85.71%)**；服务端 KV 命中 **55.1%** / 前缀复用 83.9% | 16 场景仅剩 2 例 `op=replace` 残留旧词（`chained-three-step-update`、`partial-attribute-update`）；LongMemEval 仅剩 `lme-35a27287`（未查偏好记忆）与 `lme-gpt4_2312f94c`（先后口径矛盾） |

---

## 五、当前已知问题与下一步路线图（供下个模型直接接力）

按“投入产出比 + 简历含金量”排序：

### P1：修复实测暴露的 3 个多轮记忆与提示词问题（目标：拉高 16 场景与 LongMemEval）
1. **日期注入补上星期几**：[prompts.py](../personal_assistant/assistant/prompts.py) 与 [consolidator.py](../personal_assistant/assistant/consolidator.py) 的日期格式从 `2026-11-16` 改为 `2026-11-16（星期一）`，根治相对日期（如“下周三”）换算不稳。
2. **沉淀替换句去旧词残留**：在 `CONSOLIDATION_SYSTEM_PROMPT` 中明确 `op=replace` 的 `new` 字段只写**当前最新状态**，不要在括号里附带已废弃旧值（根治 `chained-three-step-update`、`partial-attribute-update` 磁盘旧词污染）；同时酌情保留助手曾给出的关键具体建议（改善 LongMemEval `single-session-assistant`）。
3. **隐式个性化提问主动查记忆**：优化 [prompts.py](../personal_assistant/assistant/prompts.py) 的 `SYSTEM_PROMPT`（注意最小改动、验收 KV Cache 不退步），当用户询问带个人偏好/经验色彩的建议（如烘焙建议、活动推荐）且 `MEMORY.md` 存在相关主题时，先检索对应记忆再个性化回答（直击 LongMemEval `single-session-preference` 0/2）。
- **验收标准**：同口径重跑 16 场景 `memory_scenarios.jsonl` + 14 题 `evals/public/longmemeval_scenarios.jsonl`，对比 `cffbf26` / `d2e20e3` 并追加记录到本文档。

### P2：新增可选插件 `plugins/hybrid_rrf_retriever.py`（FTS5 BM25 + Embedding API 向量 + RRF 融合）
- **动机**：`fts5_bm25` 仍有两类纯词面天花板——40 题里仅剩的 `para-search-fallback`（零共享词同义改写）以及 LoCoMo `multi-hop` (43.9%) / `open-domain` (44.2%)。
- **设计约束**：继续纯 stdlib（`urllib` 调 OpenAI 兼容 `/embeddings` 接口，SQLite 缓存向量，纯 Python 余弦相似度 + RRF 融合，无 Key 时自动退化为 `fts5_bm25`），做成独立插件挂进 `RETRIEVERS`，在 `config/tools.json` 中一键切换并用 `--retriever` 跑 40 题 + LoCoMo 200 题 A/B 对比。

### P3（可选亮点）：Human-in-the-Loop 受控插件草稿审批与热重载
- Agent 用 `write_file` 在 `data/workspace/plugin_drafts/` 起草插件 → AST 安全扫描 + `pytest` 验证 → 人工审批后由宿主（非 LLM）安装至 `plugins/` 并触发 `reload_tools()`。

---

## 六、后续迭代追加记录模板

> 新模型每次完成迭代后，请在下方按此模板追加（保持倒序或正序一致即可，推荐按时间顺序往末尾追加）：

```markdown
### [YYYY-MM-DD] <简短标题>（Commit `<short-sha>`）
- **目标与思路**：为什么改、为什么选这个方案。
- **改动文件**：列出核心文件（附关键设计点）。
- **踩坑与修复**：过程中遇到的真实 Bug、根因与测试补强。
- **同口径评测对比（改前 `<old-sha>` → 改后 `<new-sha>`）**：
  - 单元测试：`XX/XX passed`
  - 离线检索（40 题 / LoCoMo 200 题）：...
  - 实模多轮记忆（16 场景 / LongMemEval 14 题）：...
  - 退步与未验证项：如实列出。
- **下一步**：更新最新优先级。
```

---

### [2026-11-21] 三层依赖漏斗架构重构（Pi 方法论落地：`ai → agent → assistant + eval`）
- **目标与思路**：按“依赖漏斗分层法 + 类型递进扩展 + 可独立使用测试”把原本平铺在 `personal_assistant/` 的 12 个模块拆清边界，消灭 AST 扫出的 `eval/* → cli.py` 反向依赖、`memory.py → commands.py` 横向串台以及 `react.py` 夹带个人记忆提示词的问题；同时补齐 `LLM_PROVIDERS` 插件口，让未来切换非 OpenAI 原生厂商也不触碰通用循环。
- **改动文件**：
  - Layer 1 (`personal_assistant/ai/`)：`types.py`（原子 `ToolSchema`、`ToolCall`、`TokenUsage`）、`client.py`（`LLMClient` + `register_provider` / `from_env`）、`cache.py`。
  - Layer 2 (`personal_assistant/agent/`)：`tools.py`（`Tool(ToolSchema)` 递进继承，`param_enums` 设 `kw_only=True` 保位置参数兼容）、`react.py`（纯通用无业务提示词）、`plugins.py`（一次扫描发现 `TOOLS` / `RETRIEVERS` / `LLM_PROVIDERS`）。
  - Layer 3 (`personal_assistant/assistant/`)：`common.py`（无依赖底座）、`prompts.py`（个人助手提示词）、`memory.py`、`memory_lifecycle.py`、`db.py`、`consolidator.py`、`commands.py`、`app.py`、`cli.py`。
  - 架构守卫测试：`tests/test_architecture.py`（AST 依赖漏斗 + 环检测 + 类型递进断言 + 子进程 `sys.modules` 断层隔离实测 + Provider 插件分发）。
- **踩坑与修复**：`@dataclass(frozen=True)` 继承时，父类带默认值的 `param_enums` 若不设为 `kw_only=True` 会阻塞子类添加位置字段 `handler`；设为 `kw_only=True` 后兼顾了真继承与全量已有调用方兼容。
- **同口径评测对比（改前 `d2e20e3` → 改后本次提交）**：
  - 单元测试：**68/68 passed**（+4 条三层架构守卫与子进程隔离测试）。
  - 离线检索（40 题 / LoCoMo 200 题，`fts5_bm25`）：40 题 **38/40 (95.0%)**、MRR **0.9483**、0.52 ms；LoCoMo 200 题 **105/200 (52.5%)**、Recall **68.51%**、MRR **0.6221**、9.19 ms —— **与 `d2e20e3` 完全持平，零退步**。
  - 实模多轮记忆：本次为纯模块分层重构，未改动提示词语义、检索算法或沉淀逻辑，未重复消耗 API 重跑（可在 P1 改提示词时一并验收）。
- **下一步**：继续按第五节 P1（补星期 + 沉淀去旧词 + 隐式个性化主动查记忆）或 P2（`hybrid_rrf_retriever` 向量融合插件）推进。

---

### [2026-11-21] 极简模型配置 DSL 外部编译插件与系统级防泄漏保护
- **目标与思路**：支持通过 JSON 管理多模型配置档案（`models.json`）并一键切换 `--model`，同时解决“手写 JSON 繁琐 + 普通用户不懂配环境变量”的痛点：在 `plugins/models_dsl.py` 手写零依赖编译器前端（Lexer → 递归下降 AST → 语义校验 → 标准 JSON），允许用户直接把 `api_key` 写进极简 `config/models.dsl`；把防泄漏职责完全收归系统，而不是甩给用户。
- **改动文件**：
  - Layer 1 (`personal_assistant/ai/types.py`, `client.py`, `__init__.py`)：新增 `ModelProfile`、`ModelsConfig`（带 `redacted_dict()` 脱敏）、`from_model_profile()`、`from_models_dict()`，仅识别标准 JSON 字典，对 DSL 零感知。
  - Layer 2 (`personal_assistant/agent/plugins.py`, `__init__.py`)：新增第 4 类统一插件扩展点 `CONFIG_COMPILERS` / `load_plugin_compilers()`。
  - Layer 3 (`personal_assistant/assistant/commands.py`, `app.py`, `cli.py`, `memory.py`)：`ToolsConfig.load_models_config()`（优先编译 `models.dsl`，支持原子写入 `0600` 权限 `models.json`，无 DSL 时直读 `models.json`，均无时兼容回退 `.env`）、新增 CLI `--model` 与 `compile-config` 子命令；`scripts/web_chat.py` 同步支持 `--model`。
  - 外部编译插件与安全防护：`plugins/models_dsl.py`、`config/models.example.dsl`、`.gitignore` 屏蔽 `config/models.dsl` 与 `config/models.json`、`config/tools.json` + `memory.py` + `plugins/shell_tools.py` 默认拦截读取密钥配置文件。
  - 单元测试：`tests/test_models_dsl.py`（4 条完整测试，覆盖语法/语义行列报错、直写密钥零环境变量启动、`0600` 权限、脱敏、`.gitignore` 与文件/Shell 双沙箱拦截）。
- **验证与回归**：
  - 单元测试：**72/72 passed**（含三层漏斗守卫 `tests/test_architecture.py` 全绿），AST 中文注释零缺失。
  - 离线检索（40 题 `fts5_bm25`）：保持 **38/40 (95.0%)**、MRR **0.9483**，零退步。
  - 实模多轮记忆：本次未改动检索算法与提示词语义，未重复消耗 API 重跑。
- **下一步**：继续按第五节 P1（补星期 + 沉淀去旧词 + 隐式个性化主动查记忆）或 P2（`hybrid_rrf_retriever` 向量融合插件）推进。

---

### [2026-11-21] 零代码模型配置体验升级（一行预设语法 + 终端 `setup` 向导 + 网页端热切换表单）
- **目标与思路**：站在完全不懂代码的普通用户视角，去掉 `provider`、`base_url`、`temp` 和 `{ }` 花括号的编程心智负担；把主流厂商预设与自动生成能力全部收敛在 `plugins/models_dsl.py` 内，做到“能点就不写，要写也只写一行 `use deepseek sk-xxx`”。
- **改动文件**：
  - `plugins/models_dsl.py`：新增 7 大厂商内置预设（DeepSeek / 通义千问 / Kimi / 智谱 GLM / OpenAI / 硅基流动 / 本地免密钥 Ollama）、中英文别名识别、极简一行语句 `use/add <厂商> [模型名] [api_key]`，以及挂在编译器上的 `presets()` 与 `render()` 自动生成器（核心层零直接 import 插件）。
  - `personal_assistant/assistant/commands.py` & `app.py`：新增 `save_model_setup()`（先内存编译校验再 0600 原子写入 `models.dsl` + `models.json`）、`reload_llm()`、`configure_model()`、`models_overview()`。
  - `personal_assistant/assistant/cli.py`：新增 `run_setup_wizard()` 与 `setup` 子命令（终端选编号 + 隐藏输入粘贴 Key 自动生成配置），并在未配模型运行 `chat` 时给出一步引导提示。
  - `scripts/web_chat.py` & `scripts/web_ui.html`：新增 `GET /api/models`、`POST /api/models/setup` 与右上角「模型设置」非技术表单弹窗（选厂商、贴 Key、保存即热生效，Ollama 自动隐藏 Key 输入框）。
  - `config/models.example.dsl`：更新为一行 `use deepseek` 极简示例。
- **验证与回归**：
  - 单元测试：**75/75 passed**（含 `tests/test_models_dsl.py`、`tests/test_web_chat.py`、`tests/test_architecture.py`），AST 中文注释零缺失。
  - 离线检索（40 题 `fts5_bm25`）：保持 **38/40 (95.0%)**、MRR **0.9483**，零退步。
- **下一步**：继续按第五节 P1（补星期 + 沉淀去旧词 + 隐式个性化主动查记忆）或 P2（`hybrid_rrf_retriever` 向量融合插件）推进。




---

### [2026-11-21] 正规化 GitHub Issue → 分支 → CI → PR 研发流程
- **目标与思路**：从直接推送 `main` 切换为工业级“Issue 立项 → 功能分支开发 → TDD 与同口径 Benchmark 验收 → PR + CI 守卫 → 合并”流程，既保障 `main` 稳定，也在 GitHub 仓库留下完整可查的工程协作记录。
- **改动文件**：
  - `.github/ISSUE_TEMPLATE/feature_or_improvement.md` & `bug_report.md`：标准化功能/优化与缺陷立项模板（强制含改前基线与验收标准）。
  - `.github/PULL_REQUEST_TEMPLATE.md`：标准化 PR 模板（`Closes #`、改前→改后 Benchmark 对比表、安全与架构 Checklist）。
  - `.github/workflows/test.yml`：CI 升级为三步守卫（`pytest -v` 含三层架构测试 + 40 题 `fts5_bm25` 离线检索验证 + AST 中文注释校验）。
  - `AGENTS.md`：固化标准 Issue/Branch/PR 流程（所有未来模型自动遵守）。
- **验证**：75/75 单元测试通过，40 题检索 38/40，AST 注释零缺失；本改动本身以分支 `chore/pr-workflow-templates` 推送供发起首个正式 PR。
- **下一步**：合并本 PR 并安装配置 `gh` CLI（登录大号 `geng628`），随后按正规流程为 P1（补星期 + 沉淀去旧词 + 隐式个性化主动查记忆）开 Issue #1 并提交 PR。

---

### [2026-11-21] 将 `personal_assistant/eval/` 评测引擎统一迁移至顶层 `evals/` 包（Issue `#3`）
- **目标与思路**：消除评测引擎代码（原 `personal_assistant/eval/`）与评测数据（顶层 `evals/`）分离的割裂感，让 `personal_assistant/` 成为完全纯净的三层产品包（`ai/`、`agent/`、`assistant/`），顶层 `evals/` 与 `tests/`、`plugins/` 平级并统一通过 `uv run python -m evals ...` 调用；同步强化 AST 架构守卫，同时封死 `personal_assistant/ -> evals/` 与 `evals/ -> personal_assistant/assistant/cli.py` 双向反向依赖。
- **改动文件**：
  - 顶层 `evals/`：通过 `git mv` 迁入 `__init__.py`、`__main__.py`、`cli.py`、`common.py`、`search_eval.py`、`agent_eval.py`、`memory_eval.py`、`public_benchmarks.py`，更新包内导入为 `evals.*`；完整保留 `evals/sample_data/`、`evals/sample_cases.jsonl`、`evals/memory_scenarios.jsonl` 与 gitignore 的 `evals/public/` 数据不变。
  - 架构与单元测试：`tests/test_architecture.py`（新增纯三层子包断言、产品包零依赖 `evals/` 断言、`evals/` 零依赖 `assistant/cli.py` 断言、`from personal_assistant.assistant import cli` 子模块级 AST 提取校验、`python -m evals` 子进程隔离验证）、`tests/test_eval.py`、`tests/test_memory_eval.py`、`tests/test_public_benchmarks.py`、`tests/test_fts5_plugin.py`。
  - CI 与文档：`.github/workflows/test.yml`（改为 `uv run python -m evals` 并将 `evals/` 纳入 AST 中文注释校验）、`.github/ISSUE_TEMPLATE/feature_or_improvement.md`、`README.md`、`AGENTS.md`、`docs/dev-log.md`。
- **踩坑与修复**：AST 依赖提取器原本只读取 `ImportFrom.module`，若有人写 `from personal_assistant.assistant import cli` 则 `node.module` 仅为 `personal_assistant.assistant` 从而漏检；本次升级为同时检测 `ImportFrom` 的子模块名（核对 `.py` 文件或目录存在性）与相对导入，并用独立单测 `test_architecture_import_extractor_catches_evals_and_cli_backedges` 锁死该守卫。
- **同口径评测对比（改前 `efe8568` → 改后 `refactor/issue-3-unify-evals`）**：
  - 单元测试：**75/75 → 76/76 passed**（+1 条 AST 反向依赖提取守卫单测，并扩展了子进程 `python -m evals` 与三层隔离校验）。
  - 离线检索（40 题 `fts5_bm25`）：保持 **38/40 (95.0%)**、Recall **0.9483**、MRR **0.9483**、Quote **0.9483**、负例干净 **100%**、0.522 ms（失败项仍为既有 `para-search-fallback`、`multi-honest-eval-discipline`，零退步）。
  - 公开检索（LoCoMo 200 题 `fts5_bm25`）：保持 **105/200 (52.5%)**、Recall **0.6851**、MRR **0.6221**、Quote **0.6017**、8.911 ms（与 `efe8568` 完全持平，零退步）。
  - AST 中文注释校验（`personal_assistant/`、`evals/`、`plugins/`、`scripts/`）：**0 缺失**。
  - 退步与未验证项：无退步；本次为纯目录与包边界重构，未改动提示词、记忆生命周期或检索算法，未重复消耗 API 重跑 16 场景 / LongMemEval 实模多轮记忆评测。
- **下一步**：待 PR CI 全绿并确认合并后，按第五节继续推进 P1（补星期 + 沉淀去旧词 + 隐式个性化主动查记忆）或 P2（`hybrid_rrf_retriever` 向量融合插件）。

---

### [2026-11-21] LoCoMo 共享长记忆评测模式与全量会话支持（Issue `#5`，纯 `evals/` 改动）
- **目标与思路**：解决 LoCoMo 多轮记忆原先硬截断 4 个会话且每道 QA 重复从头沉淀导致的 10×+ 算力与 Token 浪费，同时为大样本评测补齐终端 `stderr` 实时进度反馈；坚持极简单线程原则，**零改动 `personal_assistant/` 产品核心代码**。
- **改动文件**：
  - 顶层 `evals/`：`evals/memory_eval.py`（`evaluate_memory()` 支持 `shared_group` 共享长记忆快照一次沉淀+逐 QA 拷贝独立隔离目录复用，新增 `on_progress` / `on_setup_done` 实时进度回调）、`evals/public_benchmarks.py`（`convert_locomo_for_memory()` 默认保留完整 19–32 会话并支持 `max_convs` 与 `shared_group`，新增紧凑去重写出 `write_scenarios_jsonl()`，100 题完整 19 会话仅 136 KB）、`evals/cli.py`（新增 `--locomo-max-sessions`、`--locomo-max-convs` 与 `stderr` 实时进度）、`evals/__init__.py`。
  - 测试与文档：`tests/test_memory_eval.py`、`tests/test_public_benchmarks.py`、`README.md`、`docs/dev-log.md`。
- **踩坑与修复**：
  - **共享记忆 QA 误触发后台沉淀**：TDD 单测 `test_evaluate_memory_shared_group_consolidates_once_and_isolates_each_qa` 捕获到 `consolidate_calls == 3`（预期 1）——剥离 `setup_steps` 后，QA 侧剩余步骤只有 `ask`，`_run_single_scenario` 误判“无显式 `consolidate`”而按 `max_active_messages=1` 开启了后台沉淀；修复为将共享前缀是否含显式 `consolidate` 透传给 `disable_auto_consolidation`，彻底消除冗余调用与状态污染。
- **同口径评测对比（改前 `263cf37` → 改后 `feat/issue-5-eval-workers-shared-locomo`）**：
  - 单元测试：**76/76 → 79/79 passed**（+3 条进度回调、共享记忆隔离复用与紧凑 JSONL 导出单测，含三层架构守卫）。
  - 离线检索（40 题 `fts5_bm25`）：保持 **38/40 (95.0%)**、MRR **0.9483**，零退步。
  - 公开检索（LoCoMo 200 题 `fts5_bm25`）：保持 **105/200 (52.5%)**、Recall **0.6851**、MRR **0.6221**，零退步。
  - AST 中文注释校验：**0 缺失**；`personal_assistant/` 产品代码 **0 行改动**。
  - 退步与未验证项：无退步；本次仅升级顶层 `evals/` 评测器，未改动任何提示词、检索算法与记忆沉淀逻辑。
- **下一步**：合并本 PR 后，基于已确认的 `263cf37` 实模基线（16 场景 14/16 + LongMemEval 12/14，可按需扩大至 70 题或 LoCoMo 共享记忆）推进 P1 记忆生命周期与提示词优化。

---

### [2026-11-21] PR A：稳健化 JSON 解析与日期注入统一补星期几（Issue `#7`）
- **目标与思路**：按“先纯代码确定性小修、再做极简提示词微调”的正交拆分路线，先修 `2ca2580` LoCoMo 50 题基线（39/50）与多轮评测暴露的两处确定性代码缺陷，不改动任何业务提示词语义规则，避免多变量混杂。
- **改动文件**：
  - `personal_assistant/assistant/common.py`：新增零内部依赖函数 `format_date_with_weekday(day)`（输出 `2026-11-16（星期一，北京时间）`）。
  - `personal_assistant/assistant/prompts.py` & `personal_assistant/assistant/consolidator.py`：对话系统提示词与后台沉淀器统一注入带星期几的日期；`_parse_json_object()` 升级为标准库 `json.JSONDecoder().raw_decode()`，稳健提取首个完整 JSON 对象，容忍模型尾部多余说明或重复输出（根治 `q016` `Extra data` 解析崩溃）。
  - `tests/test_consolidator.py`：新增 2 条 TDD 单测覆盖 `raw_decode` 边界与星期几一致性注入。
- **同口径评测对比（改前 `2ca2580` → 改后 `fix/issue-7-json-parse-and-weekday`）**：
  - 单元测试：**79/79 → 81/81 passed**（+2 条 TDD 单测，含三层架构守卫）。
  - 离线检索（40 题 / LoCoMo 200 题 `fts5_bm25`）：保持 **38/40 (95.0%)** 与 **105/200 (52.5%)**，零退步。
  - 16 场景实模回归（`deepseek-chat`）：严格字面判定 **12/16**（实质语义正确仍为 **14/16**，与 `263cf37` 一致；2 例字面未命中经逐例核查均为纯排版/同义波动：`expiry-boundary-same-day` 实答“下午 3 点”因空格未命中 `3点`，`rolling-multi-compaction` 实答“企业微信机器人”未命中缩写 `企微`；`relative-date-resolution` 准确算出 `2026-11-25（周三）`）；本地前缀复用率 **90.41%**（改前 87.24%），服务端 KV 命中 59.54%（新日期后缀首次冷启动正常波动）。
  - AST 中文注释校验：**0 缺失**。
  - 退步与未验证项：无真实行为退步；既有 2 例 `op=replace` 旧词残留留待下一步 PR B 解决；LongMemEval / LoCoMo 50 题 + 盲测集将在 PR B 完成后一并做全套同口径实模对比。
- **下一步**：合并 PR A 后推进 **PR B**（极简沉淀洁净与提示词微调 + 5 项制衡指标与 Held-out 盲测验收）。

---

### [2026-11-21] 新增极简 TOAST 工具输出卸载原语 `truncate_or_offload`（Issue `#10`）
- **目标与思路**：在不新增工具、不碰 Layer 1/2 核心循环、保持零第三方依赖的前提下，先在 Layer 3 零内部依赖底座 `common.py` 提供通用的极简 TOAST 输出卸载原语，为后续 Issue `#11`（`fetch_webpage`）与 Issue `#12`（`run_command` / 固定命令）统一复用，做到每个 PR 只做一件事。
- **改动文件**：
  - `personal_assistant/assistant/common.py`：新增 `truncate_or_offload()` 及 `default_offload_dir()`（仅依赖标准库 `hashlib`、`os`、`re`、`tempfile`、`time`），支持短文本零落盘直通、超长单行软折行、严格 1 MB 落盘上限（对齐 `MemoryStore.MAX_SEARCH_FILE_BYTES`）、原子写盘 + 懒清理（保留最近 20 个 / 24h TTL）、tag 路径穿越清洗、`head` / `head_tail` 整行边界预览（带精确行号、`read_markdown` 续读起点与 `search_markdown(root="data/workspace/tool_outputs")` 指引），以及磁盘不可写时安全回退 `truncate_utf8`。
  - `tests/test_toast_offload.py`：新增 7 条 TDD 单元测试（覆盖直通校验、`head` 模式与 `MemoryStore` + `FTS5Retriever` 联调、`head_tail` 首尾摘要保留与中间行回查、单行软折行与 1 MB 落盘硬上限、懒清理与 tag 清洗、超预算多字节首末行安全切片、`MY_HELPER_DATA` 路径隔离与 `OSError` 优雅回退）。
- **踩坑与修复**：
  - 提示语初版写了 `query=...`，其中 `...` 包含子串 `..`，误触路径穿越断言；改为明确的中文占位 `query="关键词"`，对模型更清晰且避免歧义。
  - 懒清理单测最初将 `utime` 设到未来时间戳导致新文件反而比旧文件更旧，改为严格递增的过去时间戳并用正则核对刚写入的文件内容，准确验证保留最新 2 个批次。
- **同口径评测对比（改前 `debeda2` → 改后 `feat/issue-10-toast-offload-primitive`）**：
  - 单元测试：**81/81 → 88/88 passed**（+7 条 TDD 单测，含 `tests/test_architecture.py` 对 `common.py` 零内部依赖守卫）。
  - 离线检索（40 题 `fts5_bm25`）：保持 **38/40 (95.0%)**、Recall **0.9483**、MRR **0.9483**、Quote **0.9483**、负例干净 **100%**、0.527 ms（零退步）。
  - 公开检索（LoCoMo 200 题 `fts5_bm25`）：保持 **105/200 (52.5%)**、Recall **0.6851**、MRR **0.6221**、Quote **0.6017**、9.274 ms（零退步）。
  - AST 中文注释校验：**0 缺失**。
  - 退步与未验证项：无退步；本 PR 仅新增底层原语，未改动任何插件行为、提示词或记忆沉淀逻辑，未重复消耗 API 重跑实模多轮记忆评测。
- **下一步**：合并本 PR 后，依次推进 Issue `#11`（`fetch_webpage` 接入 TOAST 并解除 512 KB 静默截断）与 Issue `#12`（`run_command` 与固定命令接入 Head+Tail TOAST）。

---

### [2026-11-21] `fetch_webpage` 接入 TOAST 输出卸载并解除 512 KB 静默截断（Issue `#11`）
- **目标与思路**：解决 `fetch_webpage` 抓长文章的两道硬截断痛点——第一道 512 KB HTML 静默截断、第二道 8 KB UTF-8（中文约 2,600 字）直接丢弃后半篇且无法续读；复用 `#10` 的 `truncate_or_offload(..., mode="head")`，零改动核心 Layer 1/2/3 代码。
- **改动文件**：
  - `config/tools.json`：`web.max_download_bytes` 从 `524288`（512 KB）提升至 `2097152`（2 MB）。
  - `plugins/web_tools.py`：`WebPolicy` 同步提升默认下载上限至 2 MB 并支持可选 `offload_dir`；`_default_http_get` 多读 1 字节探测超限并在响应头部显式标注源码截断；新增 `_format_json_text()` 美化压缩 JSON 便于按行分页；`execute_fetch_webpage` 接入 `truncate_or_offload(..., tag="webpage", mode="head")` 并按 `min(DEFAULT_SOFT_WRAP_CHARS, max(32, max_output_bytes // 6))` 保障单段落长中文在预览中也稳定可见。
  - `tests/test_web_plugin.py`：新增 `test_fetch_webpage_toast_offload_download_cap_notice_and_json_pretty`（覆盖 2 MB 配置、长文章 TOAST 落盘 + `MemoryStore.read_markdown` 分页续读 + `FTS5Retriever.search_markdown` 检索、超下载上限显式提示与压缩 JSON 自动换行），并隔离已有测试的 `offload_dir`。
- **踩坑与修复**：
  - 既有单测 `max_output_bytes=200` 且 `<p>` 单段中文 960 字节（320 字 < 1000 字符默认折行阈值），整行贪心选取在前 3 行头部后即停止，导致正文预览为空；通过按 `max_output_bytes // 6` 动态收窄软折行字符上限（保证 3 字节汉字单行不超过预算一半），8 KB 生产配置不受影响而小预算/超长单段场景均稳定保留正文预览。
- **同口径评测对比（改前 `d674d33` → 改后 `feat/issue-11-web-toast-offload`）**：
  - 单元测试：**88/88 → 89/89 passed**（+1 项多场景网页 TOAST 集成单测，三层架构守卫全绿）。
  - 离线检索（40 题 `fts5_bm25`）：保持 **38/40 (95.0%)**、MRR **0.9483**、0.533 ms（零退步）。
  - 公开检索（LoCoMo 200 题 `fts5_bm25`）：保持 **105/200 (52.5%)**、Recall **0.6851**、MRR **0.6221**、9.078 ms（零退步）。
  - AST 中文注释校验：**0 缺失**。
  - 退步与未验证项：无退步；仅改动外围网页抓取插件，未改检索算法或记忆提示词，未重复消耗 API 重跑实模多轮记忆评测。
- **下一步**：与并行 PR（Issue `#12`）一并合并至 `main`，随后按需排期 Issue `#13`（活跃上下文与工具轮次调优）。

---

### [2026-11-21] `run_command` 与固定白名单命令接入 Head+Tail TOAST 输出卸载（Issue `#12`）
- **目标与思路**：解决 `pytest`、`git diff`、`rg` 等命令超 8 KB 输出时直接截断后半段导致模型看不到尾部失败汇总（`short test summary`）与完整日志的痛点；统一接入 `#10` 的 `truncate_or_offload(..., mode="head_tail")`，主输出始终保留退出码与头尾预览，中间省略行号与落盘路径供 `read_markdown` / `search_markdown` 按需深挖。
- **改动文件**：
  - `plugins/shell_tools.py`：`ShellPolicy` 新增可选 `offload_dir`；抽离 `_format_command_output()` 合并 `stdout` / `stderr` 并接入 `truncate_or_offload(..., mode="head_tail", tag=f"shell-{exe}")`，保留首行 `退出码` 不被切碎。
  - `personal_assistant/assistant/commands.py`：`CommandSpec` 新增可选 `offload_dir`（默认值 `None` 完全兼容现有位置/关键字构造），`CommandSpec.run()` 同步接入 `truncate_or_offload(..., mode="head_tail", tag=f"cmd-{self.name}")`。
  - `tests/test_shell_plugin.py` & `tests/test_step2_tools.py`：新增 2 条 TDD 单测验证 `run_command` 与固定命令超长 `stdout` / `stderr` 的 Head+Tail 首尾预览保留、中间省略行经 `MemoryStore.read_markdown` 完整回查及短输出零落盘，并为已有截断测试做好 `offload_dir` / `MY_HELPER_DATA` 隔离。
- **踩坑与修复**：
  - `CommandSpec` 的安全校验禁止 `argv` 中包含换行符，TDD 单测里的内联 Python `-c` 脚本改为单行语句即可真实复现多行 `stdout` + `stderr` 合并卸载场景。
- **同口径评测对比（改前 `f44392b` → 改后 `feat/issue-12-shell-toast-offload`）**：
  - 单元测试：**89/89 → 91/91 passed**（+2 条命令行 Head+Tail TOAST 集成单测，三层架构守卫全绿）。
  - 离线检索（40 题 `fts5_bm25`）：保持 **38/40 (95.0%)**、MRR **0.9483**（零退步）。
  - 公开检索（LoCoMo 200 题 `fts5_bm25`）：保持 **105/200 (52.5%)**、Recall **0.6851**、MRR **0.6221**（零退步）。
  - AST 中文注释校验：**0 缺失**。
  - 退步与未验证项：无退步；仅改动命令行输出格式化路径，未改检索算法或记忆提示词，未重复消耗 API 重跑实模多轮记忆评测。
- **下一步**：TOAST 输出卸载三部曲（`#10` / `#11` / `#12`）全部落地；待 Issue `#9` 合并后，再独立推进 Issue `#13`（活跃上下文与工具轮次调优）。

---

### [2026-11-21] 修复 `run_command` 在软链虚拟环境中找不到 `pytest` 的老 Bug（Issue `#17`）
- **目标与思路**：合并后本地端到端实测 `execute_safe_command("pytest -v", ...)` 暴露了 `139dc32e` 引入的历史 Bug——`Path(sys.executable).resolve().parent` 穿透 `.venv/bin/python` 软链接跳出虚拟环境目录，导致找不到 `.venv/bin/pytest`。
- **改动文件**：
  - `plugins/shell_tools.py`：将 `Path(sys.executable).resolve().parent` 改为 `Path(sys.executable).absolute().parent`，保留软链接所在的 `.venv/bin/` 目录以便正确定位虚拟环境工具。
  - `tests/test_shell_plugin.py`：新增 TDD 单测 `test_shell_plugin_finds_pytest_inside_symlinked_venv`（构造软链虚拟环境 + 真实 `.venv` 实跑 `pytest --version`）。
- **同口径评测对比（改前 `973d4f0` → 改后 `fix/issue-17-venv-pytest-path`）**：
  - 单元测试：**91/91 → 92/92 passed**（+1 条软链 venv 定位回归单测，三层架构守卫通过）。
  - 离线检索（40 题 `fts5_bm25`）：保持 **38/40 (95.0%)**、MRR **0.9483**（零退步）。
  - 真机端到端验证：`execute_safe_command("pytest -v", ...)` 成功跑完全部 92 项测试并按 8 KB 预算触发 Head+Tail TOAST 卸载（保留首部环境头与尾部 `92 passed` 汇总，中间省略 14 行落盘可查）。
  - AST 中文注释校验：**0 缺失**。
- **下一步**：CI 全绿并经确认合并后，待 `#9` 收尾再按需推进 `#13`。

---

### [2026-11-21] PR B：极简提示词与记忆沉淀洁净规范（Issue `#9`）
- **目标与思路**：基于真模型多套评测暴露的通用工程缺陷，以“极简通用、绝不针对测试集打补丁、5 项制衡指标防回退、Dev + Held-out 盲测分层验收”为红线，仅做 4 行通用规范微调，不触碰核心架构与检索代码。
- **改动文件**：
  - `personal_assistant/assistant/prompts.py`：`SYSTEM_PROMPT` 仅增 23 字符（261 → **284 字符**，单测硬守卫 `<= 300`），增加通用查阅触发条件“或用户寻求建议/推荐/规划且索引存在相关主题时，请先调用 `search_markdown` 或 `read_markdown` 查阅”，防幻觉红线原样保留。
  - `personal_assistant/assistant/consolidator.py`：`CONSOLIDATION_SYSTEM_PROMPT` 精修第 1、2、4 条通用规则：
    1. 规则 1：优先完整记录用户自身事实、决策、偏好与具体专有名词（人名、书名、作品、地点、数值）；助手给出的具体推荐/结论仅一句话提炼关键实体与结论，**不抄录长段通用科普或教程**。
    2. 规则 2：严格区分 `op=replace`（仅旧值被推翻时使用，且 `new` 只写最新有效状态、严禁括号附带或复述“从旧值改为/原方案已放弃”等废弃信息）与 `op=add`（独立事件/累加记录），`op=remove` 直接删除不写“已取消”冗余句。
    3. 规则 4：明确条目开头 `[YYYY-MM-DD]` 仅为记录日期，所有相对时间（过往与未来）均须结合 `【当前日期】` 换算为精确绝对日期写入正文（不模糊成“某月初/某月”）；`until` 仅用于有明确未来截止日的临时日程/待办。
  - `tests/test_consolidator.py`：新增 `test_prompt_and_consolidation_hygiene_guardrails()` 守卫 `SYSTEM_PROMPT <= 300` 字符、防幻觉红线与沉淀通用规范。
- **踩坑与多轮防回退复盘**：
  1. **长对话自回归日期模仿导致 LoCoMo `temporal` 崩盘（v1/v2 `8/10 → 3/10`）**：19 轮共享沉淀中，前几轮若把相对时间模糊为“某月初”，后续轮次读到 `【已有主题记忆】` 会模仿省略正文日期。在规则 4 补上“`[YYYY-MM-DD]` 仅为记录日期、相对时间须换算精确绝对日期写入正文”后，LoCoMo `temporal` 反超基线至 **9/10（90%，文本可答题 9/9 全对）**。
  2. **抄录助手长科普引发记忆膨胀（v3 LongMemEval `completion_tokens +40%`、Dev 14 `12/14 → 9/14`）**：规则 1 未限制助手建议篇幅时，沉淀器将大段通用教程原样抄入记忆，反而挤掉用户自身事实（已买 5 件上衣、全马目标 4:10 等）。收紧为“优先记录用户事实，助手建议仅一句话提炼实体与结论、不抄长科普”后，`completion_tokens` 迅速回落（16 场景 **-1.5%**、LME14 **+8.3%**），LME14 `knowledge-update` 与 `multi-session` 均恢复 **100%**。
  3. **规则 1/2 冲突导致 `op=replace` 括号残留旧值（v4 `memory16` `knowledge-update 2/4`）**：规则 2 补回通用反例约束“严禁括号附带或复述‘从旧值改为/原方案已放弃’等废弃信息”后，16 场景 `knowledge-update` 升至 **4/4（100%）**。
- **同口径全量评测对比（单一最终版 `feat/issue-9-prompt-memory-hygiene` 对比基线 `263cf37` / `2ca2580` / `debeda2`，零跨版本拼接）**：
  - 单元测试：**92/92 → 93/93 passed**（含架构依赖守卫与新增提示词/沉淀守卫单测）；AST 中文注释校验：**0 缺失**。
  - 离线检索（`fts5_bm25`）：40 题保持 **38/40 (95.0%)**、MRR **0.9483**；LoCoMo 200 题保持 **105/200 (52.5%)**、Recall **0.6851**、MRR **0.6221**（零退步）。
  - 实模 4 套 101 题汇总（`deepseek-chat`，同一版代码一次跑完）：
    - **16 场景 Dev**：`14/16 (87.5%) → 15/16 (93.75%)`（**+6.25pp**；`knowledge-update` **2/4 → 4/4**，`temporal` **3/3**，`abstention` **2/2**；唯一字面未命中为 `rolling-multi-compaction` 实答“企业微信机器人”未命中缩写 `企微`，实质语义 **16/16**）；KV 命中 `76.09% → 76.88%`，本地前缀复用 `87.24% → 90.43%`，`completion_tokens` **-1.5%**。
    - **LongMemEval Dev 14**：`12/14 (85.71%) → 11/14 (78.57%)`（`abstention` 2/2、`knowledge-update` 2/2、`multi-session` 2/2、`single-session-user` 2/2 均 100%；唯一 `Pass → Fail` 为 `lme-1d4da289`，改前改后均未沉淀该段纯科普对话且均诚实声明无记忆，仅改前 Judge 放行了通用 2FA 回答，实质表现持平）；本地前缀复用 `82.46%`，`completion_tokens +8.3%`。
    - **LongMemEval 盲测 Held-out 21（独立第 15–35 题，未用于调参）**：`14/21 (66.67%) → 15/21 (71.43%)`（**+4.76pp，0 例 `Pass → Fail` 退步**，`single-session-preference` `1/3 → 2/3`，`abstention` **3/3**）；KV 命中 `33.11% → 36.92%`，`completion_tokens +12.2%`。
    - **LoCoMo 50 共享长记忆（19 轮完整对话）**：自动 Judge `39/50 (78.0%) → 38/50 (76.0%)`（`temporal` **8/10 → 9/10**、`open-domain` **6/10 → 7/10**，仍严格拒绝猜测政治倾向 `q051` 与宗教程度 `q060`；本地前缀复用 `49.70% → 49.83%`）；逐例复核 4 例 `Pass → Fail`：`q182`（实答“是 Caroline 为教堂做彩绘玻璃窗，并非 Melanie”）与 `q163`（实答“Caroline 本人没有露营记录，露营的是 Melanie 一家”）均准确纠偏、零幻觉，属 Judge 对纠偏表述误判，`q004` 同时答出心理咨询与收养机构仅主次排序差异，仅 `q129`（遗漏 rainbow sidewalk）为真实召回遗漏；实质有效准确率为 **40/50（80.0%）**，净增 `q016`、`q045`、`q015` 3 例修复。
  - **5 项防回退制衡指标核验**：① 拒答/对抗零真实退步（4 套拒答 100%，LoCoMo 有效对抗 9/9 零幻觉）；② 闲聊零多余工具调用（`kv-cache-efficiency` 第 1/3/4 轮 `tool_calls == []`）；③ KV/前缀复用健康（16 场景 90.43%、盲测 KV +3.81pp、LoCoMo 49.83%）；④ 记忆体积受控（16 场景 -1.5%、LME +8.3%–12.2%）；⑤ `SYSTEM_PROMPT` 仅 284 字符，零测试集特化词。
- **遗留问题与下一步**：不放宽防幻觉红线去迎合主观猜测题；纯百科问答类 `single-session-assistant` 与记忆精简的天然权衡留作后续可选策略；LoCoMo 对抗题 LLM-as-a-Judge 误伤“明确否定并纠正主语”的现象可在后续评测工具改进中单独修正。合并 PR B 后继续推进 P2（如 `hybrid_rrf_retriever` 检索增强插件）。


---

### [2026-11-22] 合流后 `main` (`4d18252`) 全量实模重测与“外层 `Plugin(ABC)` 主板基类 + 内层 4 子协议 + 统一 `ToolRegistry`”架构重构（Issue `#20`）
- **目标与思路**：在两条开发线（P1 记忆/提示词优化 + TOAST 输出卸载三部曲 `#10/#11/#12/#17`）合入 `main`（`4d18252`）后，首先对 `4d18252` 执行零复用的全量 5 套 141 题实模重测与跨模块联调；针对联调暴露的“各插件散乱无统一契约、内置工具与插件工具走两条不同配置/目录路径”以及 `ContextVar` / `os.getenv` 暗道问题，将插件与工具体系重构为**“外层 1 个统一主板基类 + 内层 4 个极简强类型子协议 + 统一 `ToolRegistry` 多源汇聚”**架构：
  - **第一层：外层统一外壳 `Plugin(ABC)`**（`personal_assistant/agent/plugins.py`）：主板上仅挂载 **3 个同级 `Plugin(ABC)` 插件**（分别对应内层子协议：`plugins/tools/plugin.py` 的 `ToolsPlugin`、`plugins/fts5_retriever.py` 的 `FTS5RetrieverPlugin`、`plugins/models_dsl.py` 的 `ModelsDSLPlugin`），只负责 3 件统一的事：① `__init_subclass__` 自动发现与注册；② 统一持有系统资源 `self.context: PluginContext`（`project_root` / `data_root` / `config_path` / `offload_dir` / `raw_config` / `tool_registry` 及 4 类标准插槽注册方法）；③ 统一生命周期 `initialize()` / `terminate()`。
  - **第二层：内层 4 个极简强类型子协议**（核心运行时各子系统只认自己的小协议，不感知实现是否来自插件）：
    1. **LLM 工具协议 `Tool(ToolSchema)`**（统一经 `build_tool()` 构建并汇聚至 `ToolRegistry._tools[name]` + `_metadata[name]` (`risk`, `always_on`) + `_documents[name]` (`source_type`, `source_name`)）：
       - 内核内置工具：`MemoryToolsetProvider(ToolsetProvider)` ──▶ `BuiltinTool(Tool)` 子类
       - 插件工具包（`plugins/tools/` 下的 `custom_tools.py`、`shell_tools.py`、`web_tools.py` 只写纯 `@tool` 函数，由唯一的 `ToolsPlugin(Plugin)` 统一扫描装配）──▶ 动态 `type()` `Tool` 子类
       - 配置命令包装：`CommandToolsetProvider(ToolsetProvider)` / `CommandSpec.to_tool()` ──▶ `CommandToolWrapper(Tool)` 子类
    2. **全文检索协议 `RetrieverProtocol`**（`name` + `search_hits(...)`，由单文件插件 `FTS5RetrieverPlugin` 注册 `FTS5Retriever`）
    3. **配置编译器协议 `ConfigCompilerProtocol`**（`name` + `compile(...)` / `presets()` / `render(...)`，由单文件插件 `ModelsDSLPlugin` 注册 `ModelsDSLCompiler`）
    4. **模型底座协议 `LLMProviderProtocol`**（`name` + `build(env)`）
- **`main` (`4d18252`) 全量 5 套 141 题实模基线重测记录**：
  - 16 场景多轮记忆（`memory_scenarios.jsonl`）：**16/16 (100.0%)**（全类 100%，0 失败），KV 命中 `73.13%`、本地前缀复用 `87.44%`。
  - LongMemEval Dev 14：**10/14 (71.43%)**（`abstention` 2/2、`knowledge-update` 2/2、`single-session-user` 2/2），KV 命中 `54.52%`、本地前缀复用 `82.48%`。
  - LongMemEval 盲测 Held-out 21：**18/21 (85.71%)**（较 `debeda2` 基线 `14/21` **+19.04pp**，`abstention` 3/3、`single-session-preference` 3/3、`single-session-assistant` 2/3），KV 命中 `57.77%`。
  - LoCoMo 50 题 19 会话共享记忆：自动 Judge **38/50 (76.0%)**（`temporal` **9/10**、`open-domain` **7/10**、`completion_tokens` 20,075 较基线 `-2.1%`；`q176` 实答“Caroline 本人没有露营记录，露营的是 Melanie 一家”属准确纠偏零幻觉，核查有效 **39/50**）。
  - 全量 40 题 `--mode agent`（`sample_cases.jsonl`）：**31/40 (77.5%)**，KV 命中 `77.85%`、本地前缀复用 `77.07%`（暴露 `agent_eval.py` 漏挂 `fts5_bm25` 检索插件）。
- **改动文件（Issue `#20`）**：
  - `personal_assistant/agent/tools.py` & `plugins.py` & `__init__.py`：新增外层主板基类 `Plugin(ABC)` + `PluginContext`（同时支持 `plugins/*/plugin.py` 文件夹插件与 `plugins/*.py` 单文件插件发现）与内层 4 子协议（`Tool`、`RetrieverProtocol`、`ConfigCompilerProtocol`、`LLMProviderProtocol`）；新增唯一统一的 `build_tool()` 构建器（将任意 callable 适配为指定 `Tool` 子类或动态 `type()` `Tool` 子类，自动按签名绑定上下文参数并挂载 `ToolMetadata(risk, always_on)` 与 `ToolDocument(source_type, source_name)`）；`MemoryToolsetProvider`、`PluginManager`、`CommandToolsetProvider` 全部继承自统一基类 `ToolsetProvider`；`ToolRegistry` 统一维护 `_all_tools` / `_tools` / `_metadata` / `_documents` 并支持 `register_provider()` 与 `activate()` 按需加载。
  - `plugins/tools/`（`plugin.py`、`__init__.py`、`custom_tools.py`、`shell_tools.py`、`web_tools.py`）与 `plugins/fts5_retriever.py`、`plugins/models_dsl.py`：把同类工具模块归拢至 `plugins/tools/` 目录，各工具文件只写纯 `@tool` 函数，由唯一的 `ToolsPlugin(Plugin)`（`plugin_id = "tools"`）统一扫描注册，与单文件检索插件 `FTS5RetrieverPlugin(Plugin)`（`plugin_id = "fts5_bm25"`）及单文件编译插件 `ModelsDSLPlugin(Plugin)`（`plugin_id = "models_dsl"`）严格同级；`execute_safe_command()` 子进程 `TMPDIR` 改为 `tempfile.gettempdir()`。
  - `personal_assistant/assistant/memory.py`、`commands.py`、`app.py`、`cli.py`：新增 `BuiltinTool(Tool)` / `MemoryToolsetProvider(ToolsetProvider)` 与 `CommandToolWrapper(Tool)` / `CommandToolsetProvider(ToolsetProvider)`；`ToolsConfig.build_registry(providers=..., data_root=...)` 通过统一的 `register_provider(provider, context=build_ctx)` 汇聚三路 `ToolsetProvider`；`CommandSpec.run()` 子进程 `TMPDIR` 改为 `tempfile.gettempdir()`。
  - `evals/agent_eval.py`：挂载 `retriever = cfg.build_retriever()` 并通过 `MemoryToolsetProvider` 统一构建工具表。
  - `tests/test_plugins.py`、`tests/test_step2_tools.py`、`tests/test_toast_offload.py`、`tests/test_shell_plugin.py`、`tests/test_eval.py`：新增 5 条 TDD 单测覆盖 `Plugin(ABC)` 外层主板基类 + 内层 4 子协议校验、三源 `Tool` 子类统一构建汇聚与 `_metadata`/`_documents` 校验、无环境变量自定义 `data_root` 下 TOAST 落盘续读、`pytest` 子进程 `TMPDIR` 零工作区污染与 `agent_eval` 检索插件挂载。
- **同口径评测对比（改前 `4d18252` → 改后 `931514e`）**：
  - 单元测试：**93/93 → 98/98 passed (3.54s)**（+5 条 TDD 单测，三层架构守卫全绿）；AST 中文注释校验：**0 缺失**。
  - 离线检索（`fts5_bm25`）：40 题保持 **38/40 (95.0%)**、MRR **0.9483**；LoCoMo 200 题保持 **105/200 (52.5%)**、Recall **0.6851**（零退步）。
  - 全量 40 题 `--mode agent` 实模端到端（`deepseek-chat`，`git=931514e`）：工具命中 **31/40 (77.5%) → 35/40 (87.5%)**（**+10.0pp / +4 题**，`0` 例 `Pass → Fail` 退步；新增命中 `para-crossling-injection`、`para-tdd-workflow`、`multi-benchmarks-and-rules`、`neg-ood-mars`；29 道可答题目标源文档命中率从 **25/29 (86.2%) → 27/29 (93.1%)**），KV 命中 **77.85% → 79.14%**，本地前缀复用 **77.07% → 78.58%**，`completion_tokens` **-1.8%**。
  - 无 `MY_HELPER_DATA` 环境变量下的 TOAST + ReAct 实模联调（`git=931514e`）：模型 2 轮调用 `run_command("pytest -v")` → `read_markdown("data/workspace/tool_outputs/20261122-130428-shell-pytest-b8524bb3.txt")` 精准读出被省略的第 30 行测试名 `test_fts5_retriever_bm25_ranking_hard_negatives_and_diversity`，落盘完全隔离于 `custom_data` 且仓库根目录 `pytest-of-*` 残留为 `[]`。
  - 退步与未验证项：无退步；`--mode agent` 剩余 5 例未命中 `expected_tools=["search_markdown"]` 中，1 例（`exact-pref-style`）为模型根据 `MEMORY.md` 索引直接一步调 `read_markdown` 读对文件，3 例为 `answerable=False` 的领域外负例直接拒答，仅 `para-tool-self-heal` 1 例未查记忆直接按通用知识作答。
- **下一步**：合并本 PR 后，视需要评估 Issue `#13`（活跃上下文与工具轮次调优）或下一阶段检索/记忆优化。


---

### [2026-11-22] `op="replace"` / `op="remove"` 软退休归档与 `superseded_by` 替代链（Issue `#22`）
- **目标与思路**：按小步单一职责原则，聚焦解决 `apply_topic_operations()` 执行 `op="replace"` / `op="remove"` 时直接物理硬删除旧事实、导致“历史初值 vs 当前状态”类问题（如 LongMemEval Held-out 21 的 `lme-031748ae`）丢失旧值且误删不可回溯的缺陷；**本次不改动 `CONSOLIDATION_SYSTEM_PROMPT` 与 `SYSTEM_PROMPT`**，以便独立检验冷账本机制的纯增量收益：
  1. **热记忆层保持纯净**：`data/memory/<topic>.md` 仍只原地保留最新有效事实，默认 `search_markdown(root="data/memory")` 继续隔离 `archive/`（零旧值干扰）。
  2. **冷账本层自动软退休 + 替代链**：`replace` 退下的旧行追加写入 `data/memory/archive/YYYY-MM.md`，格式为 `- [superseded YYYY-MM-DD | from: <topic>.md] <old_line> -> superseded_by: <new_line>`；`remove` 标记 `-> superseded_by: (removed)`；并自动跳过 `facts.md` 中正在规范化的 `(msg#...)` 临时裸原话，防止污染归档。
  3. **KV Cache 友好的幂等索引与工具可发现性**：首次向 `data/memory/archive/` 写入时在 `MEMORY.md` 幂等登记一次固定冷账本入口指针，并在 `list_files` / `search_markdown` 的 `root` 参数描述中注明 `data/memory/archive`。
- **改动文件（Issue `#22`，代码提交 `ed9cdd8`）**：
  - `personal_assistant/assistant/memory_lifecycle.py`：新增 `ARCHIVE_INDEX_PATH` / `ARCHIVE_INDEX_LINE` 与 `_append_superseded_records()`，`apply_topic_operations()` 接收可选 `archive_dir` 并在发生实质替换/删除时写入替代链及登记索引；`archive_expired_topics()` 同步支持首次归档时幂等登记索引。
  - `personal_assistant/assistant/memory.py`：收紧 `_ensure_index_entry()` 幂等判定（匹配 `target_path` 或 `/{name})`）；`apply_memory_operations()` / `archive_expired_memories()` 传入 `archive_dir` 与 `_ensure_index_entry`；更新 `MemoryToolsetProvider.tools()` 中 `list_files` 与 `search_markdown` 的 `root` 说明。
  - `tests/test_memory_lifecycle.py`：新增 `test_replace_and_remove_soft_retire_with_superseded_chain_and_fts5_traceability`，覆盖 `replace`/`remove` 替代链格式、`MEMORY.md` 二次归档零改写（KV Cache 稳定）、FTS5 与线性默认热检索零污染、显式 `root="data/memory/archive"` 追溯以及 `(msg#...)` 跳过保护。
- **同口径评测对比（改前 `main` `4d18252` / `b6b93f4` → 改后 `ed9cdd8`）**：
  - 单元测试：**98/98 → 99/99 passed (2.91s)**（+1 条 TDD 单测，三层架构守卫全绿）；AST 中文注释校验：**0 缺失**。
  - 离线检索（`fts5_bm25`）：40 题保持 **38/40 (95.0%)**、MRR **0.9483**；LoCoMo 200 题保持 **105/200 (52.5%)**、Recall **0.6851**（零退步）。
  - 16 场景多轮记忆 Dev（`issue22_memory16.json`）：**16/16 (100.0%) → 15/16 (93.75%)**；全部 4 例 `knowledge-update`（含 `conflict-update`、`chained-three-step-update`、`explicit-forget-remove`）与 `abstention` / `extraction` / `multi-session` / `temporal` 均 **100%**；唯一未命中字面词为 `rolling-multi-compaction` 实答“企业微信机器人”未命中缩写 `'企微'`；`completion_tokens` **12,778 → 11,897 (-6.9%)**，本地前缀复用 **87.44% → 86.54%**，服务端 KV 命中 `73.13% → 64.06%`。
  - **LongMemEval 35 题（Dev 14 + 盲测 Held-out 21，同一代码 `ed9cdd8` 实跑，`issue22_lme35.json`）**：
    - 总计：**28/35 (80.00%) → 29/35 (82.86%)**（**+2.86pp**），服务端 KV 命中 **56.49% → 57.41%**，本地前缀复用 **82.94%**，总 `completion_tokens` **34,706 → 34,827 (+0.3%)**。
    - 分集：**Dev 14 `10/14 (71.43%) → 11/14 (78.57%)`（+7.14pp）**；**Held-out 21 `18/21 (85.71%) → 18/21 (85.71%)`**。
    - 分类（35 题）：`knowledge-update` **5/5 (100%)**、`abstention` **5/5 (100%)**、`single-session-user` **5/5 (100%)**、`multi-session` **4/5 (80%)**、`temporal-reasoning` **4/5 (80%)**、`single-session-assistant` **3/5 (60%)**、`single-session-preference` **3/5 (60%)**。
    - 改善（`Fail → Pass`，3 例）：① **`lme-031748ae`（Held-out 21 `knowledge-update` 目标修复题）**：助手经由 `MEMORY.md` 索引与 `search_markdown`/`list_files` 成功查回冷账本，精准答出“May 11, 2023 入职时带领 **4 名工程师**，Oct 24, 2023 更新后带领 **5 名工程师**”；② `lme-1d4da289`（Dev 14）；③ `lme-4adc0475`（Held-out 21）。
    - 退步与遗留（`Pass → Fail`，2 例，均未触发 `archive/`，属未改沉淀提示词前的沉淀提炼随机波动）：`lme-0e5e2d1a`（`single-session-assistant`，对话沉淀返回 `saved_files: []`，漏提助手答复的 38 名受试者）与 `lme-caf03d32`（`single-session-preference`，`preferences.md` 沉淀侧重植物饮食，漏提慢炖锅炖牛肉与酸奶需求）；另 `lme-a346bb18`、`lme-gpt4_2312f94c`、`lme-35a27287`、`lme-cc539528` 待下一独立 Issue 定向优化沉淀提示词与检索策略。
- **下一步**：合并本 PR 后，单开下一个小 Issue 集中解决 `CONSOLIDATION_SYSTEM_PROMPT` 的“局部互补 `merge` vs 完全失效、显式事件发生日时间锚点与问答核心实体索引”。


---

### [2026-11-23] 新增《小 PR + 并行 Worktree 与三层接力 Review 规范》（Issue `#24`）
- **目标与思路**：基于已成型的三层漏斗架构与 99 条 TDD 单测基线，在不牺牲代码品味、KV Cache 与架构不变量的前提下引入受控并行开发：将“先拆方案列文件清单确认再派活、2~3 路并发限流、核心架构与高碰撞文件串行、`docs/dev-log.md` 延后追加 + `.env`/`evals/public/` 只读复用防撞补丁、以及禁止作者自审的三层接力 Review（机器守卫 → 独立只读 Reviewer Agent → Owner 终审）”写入长期协作契约，并顺手对齐 `AGENTS.md` 中的外层 `Plugin(ABC)` 与内层 4 子协议描述。
- **改动文件（Issue `#24`）**：
  - `AGENTS.md`：同步最新插件主板/子协议描述，新增 `## 小 PR + 并行 Worktree 与三层接力 Review 规范` 一节。
  - `docs/dev-log.md`：追加本次协作规范变更记录（严格 append-only）。
- **验证**：纯文档改动，全量单测保持 **99/99 passed**；未改动任何 Python 源码，无需重跑实模 Benchmark。
- **下一步**：合并本 PR 后，按新流程先提交首批 2 路低耦合 Eval 改进任务的拆分与拟改文件清单，经确认后再派独立 worktree 并行实施。


---

### [2026-11-23] 修正对抗题 Judge 误伤“有效纠偏声明”问题与并行复盘（Issue `#27`）
- **目标与思路**：按第 1 批并行任务试跑结果：
  - **PR `#28`（Issue `#27`）通过验收并合并**：修正 `evals/memory_eval.py` 中 `JUDGE_SYSTEM_PROMPT` 规则 2（`is_abstention=true`），保留 `mark correct=true ONLY if` 严格合取防线（不顺从错误前提、不编造未记录细节、且明确拒绝前提或声明缺失），同时明确指出从记忆中澄清事件实际涉及另一人物/实体属于有效的纠偏声明（`correct=true`），而非编造；规则 1 逐字不变。
  - **同期纯 Prompt 实验 PR `#29`（Issue `#26`）按纪律主动关闭**：逐例核对表明仅调 `CONSOLIDATION_SYSTEM_PROMPT` 在 Held-out 21 上 `18/21 → 18/21` 零真实提升（`lme-a346bb18` 实靠 PR `#23` 的 `archive/` 兜底、`lme-gpt4_2312f94c` 仍未稳定保留发生日），故改由独立小 Issue 以**代码级结构化机制（`op="merge"` 操作码 + `event_date` 结构化字段）**分别推进。
- **改动文件（Issue `#27`，代码提交 `e2ccc25`）**：
  - `evals/memory_eval.py`：收紧并明确 `JUDGE_SYSTEM_PROMPT` 规则 2 的纠偏与防混合猜测判定。
  - `tests/test_memory_eval.py`：新增 `test_judge_answer_prompts_allow_valid_premise_correction_for_abstention()` 验证提示词契约、Rule 1 不变性与解析流程。
- **三层审查与同口径验证**：
  - 单元测试 & AST 中文注释：**100/100 passed**，0 缺失；离线检索 `38/40` 持平。
  - 第 2 层独立只读 Reviewer：**`Pass`**（0 Blocker）。
  - 实模 LLM-as-a-Judge 5 题专项验证（`deepseek-chat`，`e2ccc25`）：**5/5 全部符合预期**（`q176`/`q182` 纠偏均判 `true`；`q162` 幻觉、先拒答后猜测混合回复、普通数字错答均严格判 `false`）。
- **下一步**：合并 PR `#28` 后，串行推进记忆闭环第 3 块拼图——代码级一等公民 `op="merge"` 操作码，并独立以实模 Benchmark 验收。


---

### [2026-11-23] 新增一等公民 `op="merge"` 无损合并互补记忆维度（Issue `#30`）
- **目标与思路**：延续第 3 块拼图与“结构化代码机制优先于纯 Prompt 微调”的路线，在 `memory_lifecycle.py` 与 `consolidator.py` 引入代码级一等公民 **`op="merge"`**：当同一主题/活动补充互补属性（如马拉松完赛成绩与目标成绩、足球进球与助攻）时，原地无损拼接保留原时间戳、原正文与既有 `(until: ...)` 标记，不破坏原文且不误触发 `replace`/`remove` 的 `superseded` 软退休。
- **改动文件（Issue `#30`，代码提交 `c39268b` → 审阅修复 `238c6f4`）**：
  - `personal_assistant/assistant/memory_lifecycle.py`：新增 `_find_matching_indices()`、`_bullet_contains_clause()`（子句边界幂等去重，兼容多次 merge 剥离尾部标点，同时防 `"12."` 误命中 `"12.5"`）、`_merge_memory_bullet()`（中英文分号拼接，支持 `HH:MM` 时间戳与无日期行回退使用 `ref_day` 盖戳），以及 `apply_topic_operations()` 的 `op="merge"` 分支（未命中旧句时安全回退为 `add` 形态）。
  - `personal_assistant/assistant/consolidator.py`：`CONSOLIDATION_SYSTEM_PROMPT` 声明 `"op": "merge"` 与规则 2（保持 `1343 <= 1350` 字符、完整保留全部既有守卫词）；`consolidate_once()` 对 `written_texts` 纳入 `merge`（与 `memory_lifecycle.py` 一致优先取 `text` 再回退 `new`）以触发 `prune_consolidated_raw_facts()`。
  - `tests/test_memory_lifecycle.py` & `tests/test_consolidator.py`：新增 4 条 TDD 单测覆盖中英文多次 merge 幂等去重、边界防误判、`HH:MM` 与无日期 `until` 盖戳、`matched_indices[0]`、过期跳过、端到端合并与 `facts.md` `(msg#...)` 幽灵清理。
- **Layer 2 独立只读审查 trace**：
  - **Round 1（`c39268b`）→ `Blocker`（1 Blocker + 4 Nits）**：捕获“3 次及以上连续 `merge` 时中间子句末尾句号被 `rstrip` 剥离导致原句幂等去重失效 (`target time was 4:10:00.` 重复拼接)”，以及 `HH:MM` 正则漏匹配、`written_texts` `new`/`text` 优先级与边界单测缺失。
  - **Round 2（`238c6f4`）→ `Pass`（0 Blocker）**：逐项核验 `_bullet_contains_clause` 边界、无日期行 `ref_day` 盖戳、`written_texts` 优先级一致与 104/104 全量单测通过。
- **同口径评测对比（改前 `main` `24a2156` → 改后 `238c6f4`，同一套 PR `#28` Judge）**：
  - 单元测试：**100/100 → 104/104 passed**；AST 中文注释：业务/评测核心 0 缺失（既有 `plugins/tools/` 闭包除外）。
  - 离线检索（`fts5_bm25`）：40 题保持 **38/40 (95.0%)**；LoCoMo 200 题保持 **105/200 (52.5%)**，零退步。
  - 16 场景多轮记忆 Dev（`issue30_v2_memory16.json`）：**15/16 (93.75%)**（`knowledge-update 4/4`、`temporal 3/3`、`abstention 2/2`、`extraction 3/3`、`multi-session 2/2`；服务端 KV 命中 **80.97%**、本地前缀复用 **90.15%**；唯一字面失败 `rolling-multi-compaction` 实答“企业微信机器人”缺缩写 `'企微'`，无记忆丢失）。
  - LongMemEval 35 题（`deepseek-chat`，`main_24a2156_lme35.json` vs `issue30_v2_lme35.json`）：
    - 总计：**28/35 (80.00%) → 31/35 (88.57%)**（Dev 14: `11/14 → 12/14`；Held-out 21: `17/21 → 19/21`）；本地前缀复用 **80.29% → 81.52%**；服务端 KV 命中 `61.74% → 37.91%`（注：`completion_tokens` 36,385 → 36,870 +1.3%，属单次评测前缀冷热批间波动）。
    - **`op="merge"` 目标验证**：`lme-a346bb18` 单次 `read_markdown` 即精准答出 `4:22` 与 `4:10` 差值 12 分钟（无需回查 `archive/`）；`lme-4adc0475`（`3 goals + 2 assists = 5`）与 `lme-gpt4_194be4b3`（全部 4 件乐器，含 Pearl Export 鼓组）均稳定通过。
    - **非 `merge` 题改善与退步（如实区分波动）**：改善 4 例（`lme-35a27287`、`lme-c7cf7dfd`、`lme-0e5e2d1a`、`lme-gpt4_9a159967`）均属非 `merge` 题的模型检索/沉淀提炼命中；退步 1 例（`lme-5a7937c8`，实答 2 天漏算 Dec 10 教会募捐食盒活动）为模型归类到 `volunteering.md` 后检索遗漏波动，非 `op="merge"` 代码缺陷。
    - **遗留失败（4 例）**：`lme-gpt4_2312f94c`（`temporal-reasoning`，S22 与 XPS 13 先后顺序仍待结构化 `event_date` 解决）、`lme-1d4da289`、`lme-cc539528`（科普推荐未沉淀进长期记忆）、`lme-5a7937c8`（同上波动）。
    - **未验证项**：LoCoMo 50 题未在本小 PR 重跑。
- **下一步**：待 PR 合入 `main` 后，串行推进第 4 块拼图——在 `operations` 与 `format_memory_bullet()` 引入代码级一等公民 **`event_date`** 结构化字段（独立渲染 `(event: YYYY-MM-DD)` 与写入日期 `[YYYY-MM-DD]` 隔离，直击 `lme-gpt4_2312f94c`）。


---

### [2026-11-23] 清理过渡期历史兼容胶水，统一现代插件主板与记忆变更协议（Issue `#34`）
- **目标与思路**：基于 `main` (`9b4b7f3`) 做架构去债，砍掉演进过程中遗留的 5 类过渡期兼容胶水，彻底收敛到“只有一套现代写法（Single Way）”：
  1. 清理 `plugins/<x>.py → plugins/tools/<x>.py` 隐式路径兜底，显式统一为 `plugins/tools`。
  2. 清理旧式魔法全局变量插件加载（`_extract_legacy_factories`、`TOOLS = [...]` 列表扫描）、`register_compiler` / `register_provider` 两参数旧分支与 `_callable_compiler`+`setattr` 包装、`ConfigCompiler` 类型别名及 `ModelsDSLCompiler.__call__`，统一收敛为外层 `Plugin(ABC)` + 内层强类型子协议（`Tool` / `RetrieverProtocol` / `ConfigCompilerProtocol` / `LLMProviderProtocol`）；无 `Plugin` 子类的单文件工具脚本仅支持 `@tool` 装饰器一种写法。
  3. 清理 `load_plugins` / `load_plugin_tools` / `load_plugin_retrievers` / `load_plugin_providers` / `load_plugin_compilers` 5 个批量加载薄壳，统一使用唯一加载入口 `PluginManager.from_entries()`。
  4. 清理 `build_dynamic_plugin_tool`、`memory_tools()`、`command_tools()` 及 `ToolsConfig` 上冗余的 `plugin_tools` / `plugin_retrievers` / `plugin_compilers` 快照字段和 `plugin_manager is None` 死分支，统一使用 `build_tool()`、`MemoryToolsetProvider`、`CommandToolsetProvider` 与 `ToolsConfig.plugin_manager` 单一真相源。
  5. 清理 `consolidator.py` 旧格式 `items` 兼容分支与 `MemoryStore.save_memory_entries()` 死接口，唯一保留结构化 `operations`（`add / merge / replace / remove`）与 `apply_memory_operations()`。
- **改动文件（Issue `#34`，代码提交 `9ce3cc0`）**：
  - 核心代码：`personal_assistant/agent/plugins.py`、`personal_assistant/agent/tools.py`、`personal_assistant/agent/__init__.py`、`personal_assistant/assistant/commands.py`、`personal_assistant/assistant/memory.py`、`personal_assistant/assistant/consolidator.py`、`personal_assistant/assistant/cli.py`、`personal_assistant/assistant/__init__.py`。
  - 插件与脚本：`plugins/models_dsl.py`、`scripts/demo.py`。
  - 单测升级与负向守卫：`tests/test_plugins.py`、`tests/test_architecture.py`、`tests/test_fts5_plugin.py`、`tests/test_models_dsl.py`、`tests/test_step3_memory.py`、`tests/test_consolidator.py`、`tests/test_shell_plugin.py`、`tests/test_web_plugin.py`（新增 `test_legacy_compat_shims_removed_and_rejected` 与 `test_consolidator_ignores_legacy_items_format`）。
- **Layer 2 独立只读审查 trace（`9ce3cc0`）**：Verdict **`Pass`**（0 Blocker，3 条非阻塞 Nits：`tests/test_plugins.py` 未使用导入 `LLMProviderProtocol`、可补测 2 参数调用拒收及本条 `dev-log.md` 延后追加提醒）。
- **同口径验证对比（改前 `main` `9b4b7f3` → 改后 `9ce3cc0`）**：
  - 单元测试 (`uv run pytest -v`)：**104 passed → 106 passed**（4.72s，含三层架构守卫及 2 条新增负向守卫单测）。
  - AST 中文 `#` 注释校验：全部 18 个改动 `.py` 文件 **0 缺失**。
  - 离线演示 (`uv run python scripts/demo.py`)：两段离线 ReAct 演示正常跑通。
  - 40 题自建检索 (`fts5_bm25`)：**38/40 (95.0%)** 持平（`mean_source_recall=0.9483`、`mean_mrr=0.9483`、`negative_clean_rate=1.0`），零退步。
  - LoCoMo 200 题离线检索 (`fts5_bm25`)：**105/200 (52.5%)** 持平（`mean_source_recall=0.6851`、`mean_mrr=0.6221`），零退步。
  - **未验证项**：本次未重跑实模多轮/端到端评测（Memory 16 / LongMemEval 35 / LoCoMo 50 / Agent 40），因 `SYSTEM_PROMPT`、`CONSOLIDATION_SYSTEM_PROMPT`、检索算法及模型调用逻辑均零改动。
- **下一步**：待本 PR 合入 `main` 后，基于干净底座按「干净版路线 A」推进主线——为历史对话 `chat_history.sqlite3` 增加零 Schema 侵入、复用同一 FTS5 分词底座的只读按需原文检索工具 `search_chat_history`。


---

### [2026-11-23] 新增只读增量 FTS5 历史对话检索插件工具 `search_chat_history`（Issue `#36`）
- **目标与思路**：按干净版路线 A 为“热记忆只提炼用户事实/偏好、不抄录长段助手科普/推荐”的天然权衡补齐 Tier-3 冷原话按需回溯能力，同时严格保持 `db.py`、`prompts.py`（采纳 Option 1 与 `main` `a46878c` 逐字一致）与 `evals/` 零改动、零 Schema 侵入与零旧库兼容包袱：
  1. **纯 `@tool` 插件落地（`plugins/tools/history_tools.py`）**：通过只读 SQLite URI（`?mode=ro`）读取 `chat_history.sqlite3`，按自增 `id > last_indexed_id` 维护内存增量 FTS5 索引（与 `plugins/fts5_retriever.py` 同构），直接复用 `tokenize_text()` 与 `FTS5Config`（BM25 + 词覆盖率双门控）。
  2. **通用上下文注入与快照隔离**：在 Layer 2 `PluginContext` 新增业务无关的 `extra_bindings` 与 `retrieval_config` 透传；`AssistantApp.build_tools()` 注入 `{"history_db_path": self.state.db_path, "before_msg_id": message_id}`，检索端以 `id < before_msg_id` 严格隔离当前轮消息（`prev_tokens` 仅单向向后引用同会话前一条、`next_rows` 同受 `id < before_msg_id` 约束），同时保持 `data/state` 仍在 `files`/`shell` 拒绝名单内。
  3. **参数引导与三大实测缺陷根治（`c7e2ea2` → `0a6cf70`）**：`@tool search_chat_history` 仅暴露单参数 `query`（契合 `ToolRegistry._validate_values` 非空校验），并在参数描述中明确提示使用“原对话语言”（避免中文系统提示词下对英文原文发中文词搜空）；在工具描述内引导先查 `data/memory/`、记忆未记细节时再调 `search_chat_history`；针对单主题 `IDF <= 0`（SQLite 钳位到 `1e-6`）导致 `round(..., 4)` 坍缩 `0.0` 与 `2/3` 查询误杀，放宽 `high_coverage = coverage >= 0.6` 并以 `base_score = max(bm25_val, cfg.min_bm25_score, 0.1)` 保序；传播 `HistoryHit.prev_matched` 让因 `prev_tokens` 跨轮命中的前一条助手答复同样保留 `MAX_MATCHED_MSG_CHARS`（1800 字符），避免长推荐列表被 `MAX_NEIGHBOR_MSG_CHARS`（600 字符）腰斩（直击 `lme-c7cf7dfd` 店铺列表第 4 项 `Nostalgia` 被截断问题）。
- **改动文件（Issue `#36`，代码提交 `af4d334` → `671ff3b` → `c7e2ea2` → `0a6cf70`）**：
  - `plugins/tools/history_tools.py`（新增）：`HistoryHit`（含 `prev_matched`）、`_HistoryFTSIndex`（只读增量同步、事务提交后推进游标、异常重置、单主题 `IDF` 托底与 `>= 0.6` 覆盖门控）、有界 LRU 租约钉住（`_checkout_history_index` / `_release_history_index`）、`format_history_hits`（配对回复与 `prev_matched` 证据 1800 字符保留）、`execute_search_chat_history` 与 `@tool search_chat_history`。
  - `personal_assistant/agent/plugins.py`：新增 `PluginContext.extra_bindings` 与 `retrieval_config` 透传（核心绑定优先覆盖），以及 `_import_plugin_module()` 在 `_IMPORT_LOCK` 内基于文件指纹 `(st_mtime_ns, st_size, st_ino)` 的线程安全模块缓存（支持热重载同时确保跨轮保留 `_HISTORY_INDICES` 状态）。
  - `personal_assistant/assistant/commands.py`、`app.py`、`config/tools.json`：透传 `tool_context` 并在 `enabled_tools` 启用 `search_chat_history`（`prompts.py`、`db.py` 与 `evals/` 均与 `main` `a46878c` 零 diff）。
  - `tests/test_history_plugin.py`（新增 10 条单测）与 `tests/test_step2_tools.py`：覆盖只读零改表边界、原对话语言提示契约、中英双语/跨轮 `prev_tokens`/上下文窗口/`before_msg_id` 隔离、增量同步/删库重建/负例空返回、`AssistantApp.chat()` 端到端与多轮复用、多线程 LRU 租约淘汰、截断上限与 `retrieval_config` 透传、单主题 `IDF<=0` 2/3 覆盖召回与非零排序/负例门控、以及 `prev_matched` 证据免于 600 字符截断。
- **Layer 2 独立只读审查 trace**：
  - **Round 1（`af4d334`）→ `Blocker`（2 Blocker + 4 Nits）**：① `_get_history_index` 释放 `_CACHE_LOCK` 后并发 LRU 淘汰关闭竞态与锁内阻塞；② `ToolsPlugin.initialize()` 每轮重执行模块清空 `_HISTORY_INDICES` 并泄漏内存连接；另指出 `sync()` 游标更新时机与测试缺口。
  - **Round 2（`671ff3b`）→ `Pass`（0 Blocker）**：核验 `_active_users`/`_retired` 锁外延迟关闭、文件指纹缓存模块、事务后推进游标与 113/113 全绿。
  - **Round 3（`c7e2ea2`）→ `Pass`（0 Blocker）**：核验 `prompts.py` 回退 `main`、`_IMPORT_LOCK` 覆盖 `stat()` 与 `high_coverage` 放行，114/114 全绿。
  - **Round 4（`0a6cf70`）→ `Pass`（0 Blocker，3 条非阻塞 Nits）**：核验 `base_score`、`high_coverage >= 0.6`、低覆盖负例门控、`prev_matched` 传播与 116/116 全绿。
- **同口径验证对比（改前 `main` `a46878c` → 改后 `0a6cf70`）**：
  - 单元测试 (`uv run pytest -v`)：**106 passed → 116 passed**（4.70s，含 `test_architecture.py` 及 10 条新增单测）；AST 中文 `#` 注释校验：**0 缺失**。
  - 离线检索（`fts5_bm25`）：40 题保持 **38/40 (95.0%)**（`MRR=0.9483`、`negative_clean_rate=1.0`）；LoCoMo 200 题保持 **105/200 (52.5%)**（`Recall=0.6851`、`MRR=0.6221`），零退步。
  - **当日同配置对照组 A/B（关闭 vs 开启 `search_chat_history`，同一天 `deepseek-chat`）**：
    - `single-session-assistant`（5 题）：`main` 等价关闭组 **`1/5 (20.0%)`**（仅过 `lme-561fabcd`） → `0a6cf70` 冒烟实测 **`5/5 (100.0%)`**、全量 35 题复测 **`5/5 (100.0%)`**（连续两次全对；4 道在 `main` 上失败的题目 `lme-1d4da289`、`lme-c7cf7dfd`、`lme-cc539528`、`lme-0e5e2d1a` 均先检索 `data/memory/` 再转调 `search_chat_history` 精准作答）。
    - `single-session-preference`（5 题）：`main` 等价关闭组 `3/5 (60.0%)` → `0a6cf70` 全量实测 **`4/5 (80.0%)`**，零回退。
  - **16 场景多轮记忆 Dev（`/tmp/issue_36_v4_memory16.json`，`git=0a6cf70`）**：**16/16 (100.0%)**，服务端 KV 命中 **66.15%**、本地前缀复用 **84.38%**（`completion_tokens=13,228`）；`knowledge-update 4/4`、`abstention 2/2`、`extraction 3/3`、`multi-session 2/2`、`temporal 3/3`、`compaction-cache 2/2` 全类通过；`kv-cache-efficiency` 闲聊全为 `tools=[]`；`temp-fact-expiry` 即使调用过 `search_chat_history` 仍未复活已过期事件，`explicit-forget-remove` 亦通过。
  - **LongMemEval 35 题全量（`/tmp/issue_36_v4_lme35.json`，`git=0a6cf70`）**：
    - 总计：**30/35 (85.71%)**，服务端 KV 命中 **55.14%**、本地前缀复用 **76.35%**（`completion_tokens=36,730`）。
    - 分类：**`single-session-assistant` 5/5 (100%)**、**`abstention` 5/5 (100%)**、**`single-session-user` 5/5 (100%)**、**`single-session-preference` 4/5 (80%)**、`temporal-reasoning` 4/5 (80%)、`knowledge-update` 4/5 (80%)、`multi-session` 3/5 (60%)。
    - 如实汇报 5 例失败（均 **0 次**调用 `search_chat_history`，无冷库污染或防回退退步）：① `lme-9ea5eabc`（`knowledge-update`，仅调 `read_markdown`，因 5 月 Hawaii 与上月 4 月 Paris 沉淀未分离 `event_date` 错判更近行程）；② `lme-gpt4_194be4b3`（`multi-session`，仅调 `read_markdown`，将待售的 Pearl Export 架子鼓排除在现有乐器数外答成 3 件）；③ `lme-5a7937c8`（`multi-session`，`read_markdown`/`search_markdown` 后仍漏算 Dec 10 共 2 天 vs 3 天）；④ `lme-gpt4_2312f94c`（`temporal-reasoning`，仅调 `read_markdown`，双时间轴有序待 PR #33 修复）；⑤ `lme-35a27287`（`single-session-preference`，`tools=[]` 直接反问城市）。
  - **未验证项**：LoCoMo 50 题多轮记忆与 `--mode agent` 40 题未在本 PR 重跑。
- **下一步**：合入 PR `#37` 后，将 PR `#33`（`event_date` 结构化时间轴）rebase 到最新 `main` 验收合并（可同时覆盖 `lme-gpt4_2312f94c` 与 `lme-9ea5eabc` 类发生日乱序），随后按需另开极小 PR 隔离 `memory16` 中 `get_current_time()` 宿主机墙钟泄露。


---

### [2026-11-23] 线程安全化 `last_usage` / `CacheMonitor` 与评测 `--workers` 并发开关（Issue `#38`）
- **目标与思路**：解决多轮实模评测串行耗时（LME 35 + Memory 16 合计 ~16 分钟）及多线程并发共享单个 `OpenAICompatibleLLM` 实例时的状态竞态缺陷：
  1. **线程隔离与互斥锁（`personal_assistant/ai/client.py`、`cache.py`）**：`OpenAICompatibleLLM.last_usage` 与 `TrackingLLMClient.last_usage` 改为 `threading.local()` 存取，并在每次 `respond()` 发起前清零（避免异常路径残留旧统计）；`CacheMonitor` 使用 `threading.Lock` 保护 `snapshot_index()`、`record()` 与 `summary()` 切片快照，仅在内存字典/列表读写期间持锁，不阻塞外部 HTTP 请求与汇总计算（同时根治主对话线程与后台沉淀线程并发写监控的潜在竞态）。
  2. **纯场景级 `--workers` 并发（`evals/memory_eval.py`、`evals/cli.py`）**：新增可选 `workers: int = 1` / `--workers` 参数（严格校验非 `bool` 且 `>= 1`）；Phase 1 先完成所有 `shared_group` 只读沉淀快照后再启动 Phase 2 各独立场景并发（单场景内部仍保持串行以维护 KV Cache），`on_setup_done` / `on_progress` 在主线程顺序回调，最终严格按输入原始索引恢复 `scenarios`、`failures`、`cache` 与 `latencies` 确定性顺序，异常时取消尚未启动的队列任务。
- **改动文件（Issue `#38`，代码提交 `e7ab0a9` → 精简重构 `3fa319d`）**：
  - `personal_assistant/ai/client.py`：`OpenAICompatibleLLM._usage_local`、`@property`/`@last_usage.setter`，并在 `respond()` 先清零与校验提取 `choices[0]["message"]` 后再写入 `last_usage`。
  - `personal_assistant/ai/cache.py`：`CacheMonitor._lock` 互斥保护与 `TrackingLLMClient._usage_local` 线程局部统计。
  - `evals/memory_eval.py` & `evals/cli.py`：抽离统一保序并发助手 `_run_ordered_jobs()` 消除两阶段 `ThreadPoolExecutor` 重复代码，接入 `evaluate_memory(..., workers=1)` 与 `--workers` 参数（代码+单测整体瘦身 133 行）。
  - `tests/test_cache_metrics.py` & `tests/test_memory_eval.py`：新增 8 线程双 Barrier 共享客户端/监控压测单测与紧凑版 4 场景/3 共享 QA Barrier 真并发保序及 CLI `--workers` 单测。
- **Layer 2 独立只读审查 trace**：
  - **Round 1（`e7ab0a9`）→ `Pass`（0 Blocker，3 条非阻塞 Nits）**；
  - **Round 2（`3fa319d`）→ `Pass`（0 Blocker）**：核验 `_run_ordered_jobs` 全边界（`0/1/>1`、异常取消、原序归位）、`respond()` 畸形响应防污染、`118 passed` 与 AST 中文 `#` 注释 0 缺失（附注 Phase 1 多组并发与异常取消未单设额外单测）。
- **同口径验证对比（改前 `main` `52867b0` → 改后 `3fa319d`）**：
  - 单元测试 (`uv run pytest -v`)：**116 passed → 118 passed**（`4.67s~5.01s`，含 `tests/test_architecture.py` 及 2 条并发 Barrier 单测）；AST 中文 `#` 注释校验：**0 缺失**。
  - 离线检索（`fts5_bm25`）：`e7ab0a9` 实测 40 题保持 **38/40 (95.0%)**（`MRR=0.9483`），LoCoMo 200 题保持 **105/200 (52.5%)**（`Recall=0.6851`），重构 `3fa319d` 未触碰检索与记忆模块。
  - **真实 API 5 题并发冒烟（`--mode memory --category single-session-assistant --max-cases 5 --workers 5`，跑于等价并发调度提交 `e7ab0a9`，`/tmp/issue_eval_workers_smoke5.json`）**：**5/5 (100.0%)**、服务端 KV 命中 **71.81%**、本地前缀复用 **65.62%**（`llm_calls=17`, `prompt_tokens=37,251`），总墙钟耗时 **`17.90s`**（5 题单题延迟之和 `68.2s`，逆序完成但输出列表严格保持原 JSONL 题序）。
  - **未验证项**：本次小 PR 未重跑完整 35 题 LME 与 16 场景全集（留作下一步 `rebase` PR `#33` 时直接使用 `--workers` 实测验收）。
- **下一步**：合入本 PR 后，将 PR `#33`（`event_date` 结构化时间轴）rebase 到最新 `main`，用 `--workers 5` 快速完成全量 LME 35 + Memory 16 验收。


---

### [2026-11-24] `main` (`dba72a3`) LoCoMo 100 基线与 PR `#33` 复盘；微内核 SSE 流式管道、`AgentEvent` 与 `ObserverProtocol` 插槽（Issue `#40`）
- **`main` (`dba72a3`) 基线补全与 PR `#33` 关闭记录**：
  - 补跑严格同日 `--workers 5` LoCoMo 全量 100 题对比：`main`（`dba72a3`）取得 **`84/100 (84.00%)`**（前 50 题 **`43/50`**、后 50 题 **`41/50`**、`temporal` **`19/21`**、`WALL_TIME_SEC=300.65s`）；而 PR `#33`（`event_date`，`0290a97`）虽使 LongMemEval 35 由 `30/35 → 32/35`，但在 LoCoMo 两次独立重跑前 50 题均仅 `40/50`、全量 **`81/100`**（`temporal` `16/21` `-3`、QA `completion_tokens +18.3%`，且 `q006` 暴露强推精确日导致误猜 `Saturday, May 20, 2023`）。遵循“未证明跨集净进步不合入”原则关闭 PR `#33`（Issue `#32` 保持 `OPEN`），保持 `main` 简洁稳健。
- **目标与思路（Issue `#40`，微内核机制与观测插件两 PR 拆分之 PR 1）**：按照“**机制在微内核，策略在插件**”与“**绝不重复造轮子**”原则，不在微内核塞入 `deque` / 统计策略，也不在 Layer 2 重包 Layer 1 流式事件类；仅在三层微内核焊好薄流式管道与第 5 个主板插槽 **`ObserverProtocol`**：
  1. **Layer 1 (`personal_assistant/ai/`)**：`types.py` 新增 `TextDelta` / `StreamDone(message, usage, ttft_ms, latency_ms)`；`client.py` 抽离连接前退避重试 `_open_with_retry()`，实现纯标准库 `urllib` SSE `default_post_stream()` 与紧致的 `_consume_openai_sse()`（支持分片 `tool_calls` 拼装、空 `data:` 忽略、流内 `"error"` 帧抛错、`stream_options.include_usage` 回收 `cached_tokens` 并在首包前 HTTP 400 自动降级重试一次），自定义子类默认通过 `LLMClient.stream()` 兼容回退；`cache.py` 补充 `TrackingLLMClient.stream()` 单次去重记账与 `last_usage` 同步。
  2. **Layer 2 (`personal_assistant/agent/`)**：`react.py` 彻底废除裸 3 参数 `on_step`，直接透传 Layer 1 的 `TextDelta` / `StreamDone`，仅补充 3 个极简事件 `ToolStep(duration_ms)`、`TurnEnd(reply, total_ms)` 与 `TaskEvent`（`iter_react()` 与 `run_react(..., on_event=...)` 保持会话无关）；`plugins.py` 通过共享助手 `_register_slot_factory()` 统一 `register_retriever()` 与新增的第 5 插槽 `register_observer()`，定义 `ObserverProtocol`（`name` / `on_event(session_key, event)` / `snapshot`）。
  3. **Layer 3 (`personal_assistant/assistant/`) & 外围入口**：`commands.py` 解析可选 `telemetry` 配置并提供 `ToolsConfig.build_observers()`（支持 `enabled: false` 关闭）；`consolidator.py` 在 `BackgroundConsolidator._run_worker` 单点发送 `TaskEvent`；`app.py` 新增 `stream_chat()`（与 `chat()` 共享单一实现，先校验再开流）并通过 `_emit_event` / `_collect_telemetry` 对故障观测插件做异常隔离；`cli.py`（4 行 `_print_tool_step` + 实时增量打印）、[scripts/web_chat.py](scripts/web_chat.py)（`POST /api/chat/stream` SSE 与 `Connection: close`）、[scripts/web_ui.html](scripts/web_ui.html)（`ReadableStream` 实时渲染与工具耗时展示）、`scripts/demo.py` 与 `evals/agent_eval.py` 同步对齐 `ToolStep`。
- **改动文件（Issue `#40`，精简收口提交 `bb7c650`）**：
  - 核心：`personal_assistant/ai/{types,client,cache,__init__}.py`、`personal_assistant/agent/{react,plugins,__init__}.py`、`personal_assistant/assistant/{commands,consolidator,app,cli}.py`
  - 入口与评测：`scripts/{web_chat,demo}.py`、`scripts/web_ui.html`、`evals/agent_eval.py`
  - 单测：`tests/test_step1_react.py`、`tests/test_plugins.py`、新增 `tests/test_streaming.py`（共享单测模型 `ScriptedStreamLLM`，覆盖 SSE 文本/分片 `tool_calls`、HTTP 400 `stream_options` 降级、空 `data:`/流内错误帧、`ObserverProtocol` 工厂与故障隔离、`POST /api/chat/stream` 400/503/200 端到端）
- **Layer 2 独立只读审查 trace**：
  - 初版（`fa4f698`）→ **`Pass`**（0 Blocker，4 Nits）；
  - 去套娃瘦身精简版（`bb7c650`，微内核净增压至 `+569 / -67` 行、`react.py` 净 diff 减半）→ **`Pass`**（0 Blocker，3 条非阻塞 Nits）。
- **同口径验证对比（改前 `main` `dba72a3` → 改后 `bb7c650`）**：
  - 单元测试 (`uv run pytest -v`)：**118 passed → 124 passed**（`7.37s`，含 `tests/test_architecture.py` 三层依赖守卫与流式/第 5 插槽/故障隔离全测）；AST 中文 `#` 注释校验：475 个定义/类 **0 缺失**。
  - 离线演示与检索：[scripts/demo.py](scripts/demo.py) 离线演示通过；`fts5_bm25` 40 题保持 **38/40 (95.0%)**（`MRR=0.9483`），LoCoMo 200 题保持 **105/200 (52.5%)**，零退步。
  - **真实模型（`deepseek-chat`）SSE 流式冒烟（测于底层流式一致的提交 `fa4f698`）**：Turn 1 直答实时收到 **6** 个 `text_delta` 分片（`ttft_ms=3129.6ms < turn_total_ms=3357.9ms`，`prompt_tokens=1773`, `completion_tokens=6`）；Turn 2 触发增量工具调用 `get_current_time`（`0.141ms`）并流式收到 **16** 个 `text_delta` 分片，服务端 KV 命中 **`91.2%` (`3328/3649`)**、前缀复用 **`96.1%`**（`stream_options.include_usage` 正常生效）。
  - **退步与未验证项**：无退步；因本 PR 不改 `SYSTEM_PROMPT`、`CONSOLIDATION_SYSTEM_PROMPT`、检索算法及记忆写入逻辑，多轮记忆全集（Memory 16 / LongMemEval 35 / LoCoMo 100）未在本 PR 重跑，`bb7c650` 精简后端到端行为由覆盖完整 HTTP SSE 路径的 `test_streaming.py` 与 `124 passed` 守卫验证。
- **下一步**：PR `#41` 已以 Squash Merge 合入 `main`（`5b79f6c`，含后续 `c02d9cd` CLI/请求去重收口），接着开启 **PR 2（纯插件 `plugins/telemetry.py`）** 实现 `ObserverProtocol`，在不改动微内核机制代码的前提下落地有界窗口、分阶段耗时拆解与后台沉淀开销观测。


---

### [2026-11-24] 零微内核改动运行期观测插件 `plugins/telemetry.py`（Issue `#42`）
- **目标与设计落点**：基于合入 `main` (`5b79f6c`) 的第 5 主板插槽 `ObserverProtocol`，以**零 `personal_assistant/` 微内核改动**的独立插件完成运行期分阶段耗时拆解、有界窗口与后台记忆沉淀观测账本：
  1. **`plugins/telemetry.py`（`311` 行，仅依赖 `personal_assistant.ai` / `agent` 公开类型）**：
     - 外层继承 `Plugin(ABC)`（`TelemetryPlugin`），在 `initialize()` 向插槽 5 注册工厂 `create_observer(cfg)`（确保每个 `AssistantApp` 实例拥有独立观测状态）；
     - 内层 `TelemetryObserver` 用 `threading.Lock` 保护全路径，使用 `deque(maxlen=window_size)` 存最近回合（`TurnRecord`）与后台任务（`TaskRecord`）并配合单调递增游标 `_turn_seq` / `_task_seq`；通过 `_touch_bounded(..., max_sessions)` 为 `_in_flight` 与 `_sessions` 提供有界 MRU 淘汰；全局 `totals` 与单会话 `session` 复用共享累计器 `_CumulativeStats`，前后对话与后台 `TaskEvent` 严格隔离；`TelemetryConfig` 对 `bool` / 非正数回退默认值并在 `__post_init__` 拦截非法 `<= 0` 配置，各事件耗时字段统一非负钳位。
  2. **配置、Web UI 与文档**：`config/tools.json` 增加 `"telemetry": {"enabled": true, "window_size": 200}`；[scripts/web_ui.html](scripts/web_ui.html) 在回复气泡下方渲染分阶段耗时（总计 / 首字 / 模型 / 工具）并在 `cachePill.title` 悬停提示后台沉淀次数与最近耗时（经 `!Number.isFinite` 与 `escapeHtml` 安全防护）；[README.md](README.md) 同步补全插件结构说明。
- **改动文件（Commit `f21df2a`）**：
  - `plugins/telemetry.py`、`config/tools.json`、`scripts/web_ui.html`、`README.md`
  - `tests/test_telemetry_plugin.py`（复用 `tests.test_streaming.ScriptedStreamLLM`，覆盖插件发现与开关隔离、分阶段拆解与有界 MRU/负耗时/非法配置校验、真实 `AssistantApp` + `BackgroundConsolidator` 端到端链路、8 线程并发压测）及更新 `tests/test_plugins.py`、`tests/test_history_plugin.py` 插件数量清点断言。
  - 微内核核查：`git diff --exit-code main -- personal_assistant/` 确认 **`0` 行改动**。
- **Layer 2 独立只读审查 trace**：
  - **Round 1（`44be770`）→ `Pass`**（0 Blocker，提出 5 条关于全路径非负钳位、`_pos_int`/`TelemetryConfig` 正数校验、避免 `asdict` 冗余展开、补充 `_in_flight` 淘汰测试与 `Number.isFinite` 校验的 Nits）；
  - **Round 2（`f21df2a`）→ `Pass`**（`0 Blockers / 0 Nits`）。
- **同口径验证对比（改前 `main` `5b79f6c` → 改后 `f21df2a`）**：
  - 单元测试 (`uv run pytest -v`)：**124 passed → 128 passed**（`6.57s`，含 `tests/test_architecture.py` 三层依赖守卫）；AST 中文 `#` 注释校验：**0 缺失**。
  - 离线检索（`fts5_bm25`，跑于等价逻辑提交）：40 题保持 **38/40 (95.0%)**（`MRR=0.9483`），LoCoMo 200 题保持 **105/200 (52.5%)**（`Recall=0.6851, MRR=0.6221`），零退步。
  - **真实模型（`deepseek-chat`）2 轮端到端冒烟（`/tmp/telemetry_real_smoke`，跑于等价逻辑提交，已清理）**：
    - Turn 1（直答）：`total_ms=2245.795`, `ttft_ms=2044.105`, `llm_ms=2245.314`, `tool_ms=0.0`, `cached_tokens=1536/1760 (87.27%)`；
    - Turn 2（触发 `get_current_time`）：`total_ms=4904.028`, `ttft_ms=2539.205`, `llm_ms=4902.172`, `tool_ms=0.089`, `cached_tokens=3200/3589 (89.16%)`；
    - `app.status()`：`observers=["telemetry"]`，累计 2 轮总命中率 `88.54% (4736/5349)`。
- **未验证项**：因本 PR 零微内核改动、未修改任何 Prompt、检索算法与记忆写入逻辑，未重跑真实模型多轮记忆全集（`memory_scenarios` 16 / LongMemEval 35 / LoCoMo 100）。
- **下一步**：合入 Issue `#42` 对应 PR 后，再评估基于实测遥测结果与未解决的检索/长记忆优化项（Issue `#32` / `#13`）。


---

### [2026-11-25] Slot-4 推理模型插件 `plugins/reasoning_provider.py` 与微内核通用传输底座抽离（Issue `#44`）
- **目标与架构演进（方案 A：微内核纯协议复用底座 + 推理插件零重复与去补丁收口）**：补齐第 4 主板插槽 `LLMProviderProtocol`，完成 `DeepSeek Reasoner / R1` 等 `reasoning_content` 协议推理模型接入；在 `2740c57` 抽离通用底座后，进一步于 `cf85539` 清除了流式兼容后门与隐式回调胶水：
  1. **微内核 Layer 1 通用底座（[personal_assistant/ai/client.py](personal_assistant/ai/client.py) & [personal_assistant/ai/__init__.py](personal_assistant/ai/__init__.py)，零推理特判、零测试兼容后门）**：
     - 仅向外新增暴露 `OpenAIChatClient` 一个通用传输底座，将 `_consume_openai_sse` 与 `_open_openai_stream` 收回模块私有，避免过度暴露内部传输细节；
     - `OpenAIChatClient(LLMClient)` 提供四个对称受保护钩子：`_prepare_messages`、`_extra_payload`、`_stream_delta_fields`（声明式聚合扩展文本增量并计入 `ttft_ms`，取代隐式 `on_delta -> bool` / `on_message` 闭包胶水）与 `_normalize_response_message`（同步 `respond()` 与流式 `stream()` 统一复用）；
     - 彻底移除 `OpenAICompatibleLLM.stream()` 中为旧版 `post_json`-only 测试桩保留的兼容分支（`OpenAICompatibleLLM.stream is OpenAIChatClient.stream`），并将 `_build_openai_compatible` 的空白环境变量兜底风格与插件对齐。
  2. **`plugins/reasoning_provider.py`（`234` 行 → `90` 行 → **`69` 行**，零克隆、零闭包胶水）**：
     - 外层 `ReasoningProviderPlugin(Plugin)`（`plugin_id = "openai_reasoning"`）向主板 Slot 4 注册 `OpenAIReasoningProvider`（`LLMProviderProtocol`，`build(env)` 对空白 `LLM_BASE_URL` / `LLM_MODEL` 自动兜底）；
     - 内层 `OpenAIReasoningLLM(OpenAIChatClient)` 与父类构造签名完全对称（`model` 必填，默认模型收敛在 Provider 工厂），仅声明 `_stream_delta_fields() -> ("reasoning_content",)`、跨轮剥离历史 `reasoning_content`（`_prepare_messages`）与同步/流式统一的 `_normalize_response_message`（工具续传保留字符串、无工具空值省去冗余键）。
  3. **`plugins/models_dsl.py` & `README.md`**：`ModelPreset` 新增 `provider` 字段并加入 `deepseek_reasoner`（别名 `ds-r1` / `reasoner` / `r1`，协议 `openai_reasoning`）；`render_models_dsl` / `ModelsDSLCompiler.render` 支持显式 `provider` 参数并在非 `openai_compatible` 时输出 `provider` 字段，统一用水平空白裁剪 + `_MODEL_NAME_RE.fullmatch` 防止模型名换行/非法字符注入；修正文案为 `DeepSeek Reasoner / R1 等 reasoning_content 推理模型`。至此 5 大主板插槽（`Tool`、`RetrieverProtocol`、`ConfigCompilerProtocol`、`LLMProviderProtocol`、`ObserverProtocol`）全部实现实装插件。
- **改动文件（最终代码提交 `cf85539`）**：
  - [personal_assistant/ai/client.py](personal_assistant/ai/client.py)、[personal_assistant/ai/__init__.py](personal_assistant/ai/__init__.py)
  - [plugins/reasoning_provider.py](plugins/reasoning_provider.py)、[plugins/models_dsl.py](plugins/models_dsl.py)、[README.md](README.md)
  - [tests/test_reasoning_provider.py](tests/test_reasoning_provider.py)、[tests/test_streaming.py](tests/test_streaming.py)、[tests/test_memory_eval.py](tests/test_memory_eval.py)、[tests/test_plugins.py](tests/test_plugins.py)、[tests/test_history_plugin.py](tests/test_history_plugin.py)、[tests/test_models_dsl.py](tests/test_models_dsl.py)
- **审查与重构 trace**：
  - **Round 1（`8f5a412`）→ `Request Changes`**（`0 Blocker / 3 Nits`）；
  - **Round 2（`ea550b1`）→ `Pass`**（`0 Blockers / 0 Nits`）；
  - **去补丁清理（`f1d690b`）**：移除 `lines.insert(4)`、同步降级后门与重叠兜底分支；
  - **Round 3（方案 A 去重重构 `2740c57`）→ `Pass`**；
  - **优雅度终审清理（`cf85539`）**：删除 `OpenAICompatibleLLM.stream()` 旧测试桩兼容分支（[tests/test_memory_eval.py](tests/test_memory_eval.py) 改用真 SSE `post_stream` 并发测试）、收回 `_open_openai_stream`/`_consume_openai_sse` 为私有实现、用 `_stream_delta_fields` + 统一 `_normalize_response_message` 取代 `on_delta -> bool`/`on_message` 胶水、统一 `OpenAIReasoningLLM` 构造签名并将底座通用流/线程单测归位至 [tests/test_streaming.py](tests/test_streaming.py)。
- **同口径验证对比（`main` `38dfa45` / 清理前 `7b1c4fe` → 清理后 `cf85539`）**：
  - 单元测试 (`uv run pytest -v`)：**128 passed (`38dfa45`) → 133 passed (`7b1c4fe`, `7.63s`) → 133 passed (`cf85539`, `5.08s`)**（含 `tests/test_architecture.py` 三层依赖守卫与子进程独立运行测试）；CI AST 中文 `#` 注释校验：**0 缺失**。
  - 离线检索（`fts5_bm25` 40 题）：保持 **38/40 (95.0%)**（`mean_mrr=0.9483`, `negative_clean_rate=1.0`，失败题维持 `para-search-fallback`、`multi-honest-eval-discipline`），零退步。
  - **真实模型（`deepseek-reasoner`）2 轮 + 工具调用端到端冒烟（`/tmp/reasoner_slot4_smoke`，测于提交 `8f5a412`，已清理）**：
    - `app.configure_model(preset="reasoner", ...)` 热生效为 `OpenAIReasoningLLM`；
    - **Turn 1**（触发 `get_current_time`）：轮内回传 `reasoning_content` 顺利完成多跳回答（`"现在是北京时间 2026-11-25 12:32:24..."`），`total_ms=7581.498`, `ttft_ms=4881.155`, `llm_ms=7580.936`, `tool_ms=0.04`, `cached_tokens=1664/3643 (45.68%)`；
    - **Turn 2**（跨轮追问，触发 `get_current_time`）：跨轮剥离旧思维链无 HTTP 400，`total_ms=9487.951`, `ttft_ms=3066.208`, `llm_ms=9487.072`, `tool_ms=0.043`, `cached_tokens=3712/4021 (92.32%)`。
- **退步与未验证项**：无退步；`deepseek-reasoner` 真机双轮冒烟跑于早期提交 `8f5a412`，本次去补丁与声明式收口重构（`cf85539`）已重跑全部 `133` 项确定性单测、`fts5_bm25` 40 题与 AST 校验，未重跑真机实模冒烟及多轮长记忆全集（Memory 16 / LongMemEval 35 / LoCoMo 100）。
- **下一步**：待 PR `#45`（`reasoning_provider` + 通用底座重构）与 PR `#47`（网页端零依赖 XSS 安全 Markdown 渲染）经用户确认并合入后，推进混合检索插件 `plugins/hybrid_retriever.py`（向量嵌入 + SQLite BLOB 缓存 + RRF）。


---

### [2026-11-26] 移除 `MAX_TOOL_ROUNDS` 与 `tools=None` 强行总结兜底，回归纯 ReAct 循环（Issue `#50`）
- **背景与设计动机**：
  - 原 `personal_assistant/agent/react.py` 硬编码了 `MAX_TOOL_ROUNDS = 4`，并在跑满轮次后原地改写最后一条 `role: "tool"` 消息追加中文提示词，再传 `tools=None` 发起第 5 次模型调用强行总结——不仅瞬间 100% 击穿整轮服务端 KV Cache 前缀、污染工具观察原文并违背 Layer 2 微内核业务无关原则，还会在多跳检索未完成时诱导模型半途猜测，以及在携带 `tool_calls` 历史的推理模型下制造协议不一致风险。
  - 对标 `pi-agent-core` 标准事件循环，彻底移除 `MAX_TOOL_ROUNDS`、`max_rounds` 参数、`_extract_reply` 冗余 `fallback` 形参及 `tools=None` 强行总结分支：`iter_react()` / `run_react()` 改为纯粹 `while True` 循环，进入循环前一次性固定 `tool_defs`，全程只追加消息不篡改，仅以模型未返回 `tool_calls`（`not tool_calls`）作为唯一自然终止条件。
- **改动文件（代码提交 `b7b14b5` → 精简收口 `ee36ba3`）**：
  - 核心与外围：[personal_assistant/agent/react.py](personal_assistant/agent/react.py)、[personal_assistant/agent/__init__.py](personal_assistant/agent/__init__.py)、[personal_assistant/assistant/app.py](personal_assistant/assistant/app.py)、[evals/agent_eval.py](evals/agent_eval.py)、[docs/mvp-design.md](docs/mvp-design.md)
  - 单测：[tests/test_step1_react.py](tests/test_step1_react.py)（替换为 `test_unbounded_react_loop_stops_only_when_no_tool_calls`，验证移除 `MAX_TOOL_ROUNDS`、5 处入口拒绝 `max_rounds`、连续 6 轮工具调用 + 第 7 轮自然收束且 `tools` 定义恒定、`role="tool"` 消息零篡改）
- **Layer 2 独立只读审查 trace**：
  - **Round 1（`b7b14b5`）→ `Pass`**（`0 Blocker / 3 Nits`）；
  - **收口清理（`ee36ba3`）**：内联移除 `_extract_reply` 残留 `fallback` 形参并对齐 `docs/mvp-design.md` 的 `iter_react()` 表述。
- **同口径验证对比（改前 `main` `16ebd78` → 改后 `b7b14b5` / `ee36ba3`）**：
  - 单元测试 (`uv run pytest -v`)：**133 passed → 133 passed**（`5.5s`，含 `tests/test_architecture.py` 三层依赖漏斗与子进程独立运行测试）；CI AST 中文 `#` 注释校验：**0 缺失**。
  - 离线检索（`fts5_bm25`）：40 题保持 **38/40 (95.0%)**（`MRR=0.9483`，失败题维持 `para-search-fallback`、`multi-honest-eval-discipline`），LoCoMo 200 题保持 **105/200 (52.5%)**（`Recall=0.6851, MRR=0.6221`），零退步。
  - **16 场景真实模型多轮记忆评测（`deepseek-chat`, `workers=4`，测于核心提交 `b7b14b5`）**：**`16/16 (100%)`** 全过（6 大分类全部 `1.0`，零死循环），`llm_calls=129`，服务端 KV Cache 命中率 **`75.16%`**（`191,232 / 254,439`），本地前缀复用率 **`91.37%`**。
- **退步与未验证项**：无退步；未重跑公开集 LongMemEval 35 / LoCoMo 100 长跑。
- **下一步**：待开放 PR `#49`（评测 CLI Slot-4 与独立 Judge）及 PR `#47`（网页端安全 Markdown 渲染）rebase 最新 `main` 合入后，继续评估活跃上下文压缩窗口（Issue `#13` 剩余部分）或主对话/后台沉淀双模型分槽。


---

### [2026-11-26] 网页端零依赖 XSS 安全完整 Markdown 流式渲染（Issue `#46`）
- **目标与思路**：保持 `personal_assistant/` 微内核 `0` 行改动与零第三方前端依赖，将 [scripts/web_ui.html](scripts/web_ui.html) 的轻量渲染器升级为完整结构化 Markdown 流式渲染，同时严格守住 XSS 安全边界。
- **改动文件（代码提交 `d969928`）**：
  - [scripts/web_ui.html](scripts/web_ui.html)：先经 `escapeHtml`（含单引号 `&#39;`）转义再解析标题、表格、有序/无序列表、引用块、分割线、闭合及流式未闭合代码块、行内代码、粗体与严格限制 `^https?:\/\/[^\s)]+$` 的安全外链（`rel="noopener noreferrer"`）。
  - [tests/test_web_chat.py](tests/test_web_chat.py)：新增 `test_web_ui_markdown_renderer_and_xss_safety()`，通过 Node.js 实跑验证标题/表格/列表/流式未闭合代码块与 `<script>` / `<img onerror>` / `javascript:` 注入拦截。
- **Layer 2 独立只读审查 trace**：`Pass`（`0 Blockers / 0 Nits`）。
- **同口径验证对比（改前 `main` `13a9388` → 改后合流提交）**：
  - 单元测试 (`uv run pytest -v`)：**133 passed → 134 passed**（含 Node 前端渲染与 XSS 单测、三层架构守卫）；AST 中文 `#` 注释校验：**0 缺失**。
  - 离线检索（`fts5_bm25` 40 题）：保持 **38/40 (95.0%)**（`MRR=0.9483`），零退步。
- **下一步**：继续合入 PR `#49`（Issue `#48` 统一 `ToolsConfig.build_llm` 与评测 CLI `--model` / `--judge-model`），并启动安全沙箱加固与 `pi-agent-core` 架构优雅化演进。
