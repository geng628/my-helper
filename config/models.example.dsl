# 极简模型配置 DSL 示例（复制本文件为 config/models.dsl 后直接填入你的 api_key 即可使用）
# 系统已默认将 config/models.dsl 与编译产物 config/models.json 加入 .gitignore、0600 权限与工具沙箱屏蔽，无需手动折腾环境变量

default deepseek

model deepseek {
  provider openai_compatible
  base_url https://api.deepseek.com/v1
  model    deepseek-chat
  api_key  sk-替换为你的真实密钥
  temp     0.2
}

# 可按需定义多个模型，运行时用 `--model local_qwen` 一键切换
model local_qwen {
  provider openai_compatible
  base_url http://127.0.0.1:11434/v1
  model    qwen2.5:7b
  api_key  ollama
  temp     0.1
}
