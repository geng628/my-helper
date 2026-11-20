# 系统架构与记忆设计

## ReAct 推理循环

核心推理循环采用原生 OpenAI 兼容 Function Calling，最多允许 8 轮工具调用（MAX_TOOL_ROUNDS = 8）。
当工具参数校验失败或执行抛出异常时，不中断对话，而是将错误信息作为 Observation 回传给大模型触发自愈重试。

## KV Cache 前缀稳定化

系统提示词严格按静态指令、行为规则、MEMORY.md 目录索引、当前日期的顺序拼接，并在单个会话内冻结快照。
MEMORY.md 索引强制限长在前 100 行和 4000 字符以内，历史对话采用标准追加式 user/assistant 列表，避免每轮重写消息打穿服务端 KV Cache。

## 后台记忆沉淀与上下文压缩

当活跃会话消息数达到 16 条或总字符超过 8000 时，标准库后台线程自动触发 MemoryConsolidator 提炼长期记忆并更新 SQLite 的 session_compactions 滚动摘要与游标。
只有当 Markdown 记忆落盘且摘要 JSON 解析全部成功后，才原子推进 compacted_until_id，保证断电或模型超时不丢上下文。

## 记忆生命周期（冲突消解与过期归档）

沉淀器写入前先读取已有主题快照，支持 add、replace、remove 三种原子操作，新旧事实冲突时原地替换绝不保留双版本。
带截止日期的临时事实记录为 (until: YYYY-MM-DD)，查询时自动屏蔽过期条目，每日首次对话自动将过期条目迁入 data/memory/archive/YYYY-MM.md。
