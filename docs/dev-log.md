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
| `2ca2580` | LoCoMo 共享长记忆评测模式、紧凑 `shared_setup` JSONL、完整会话导出与实时进度（`#5` / `#6`，纯 `evals/` 单线程，零改动 `personal_assistant/`） | 消除 LoCoMo 多轮评测的重复沉淀浪费与 4 会话硬截断，保持极简单线程确定性且绝不因评测触碰产品核心代码 |
| Issue `#7` | PR A 纯确定性代码小修：`_parse_json_object()` 改用 `raw_decode()` + 日期注入统一补齐星期几 | 先做零 Prompt 语义风险的确定性修复，消除 `q016` 尾部多余文本 JSON 解析崩溃，并根治跨周相对日期心算偏移 |

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
