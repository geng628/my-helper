# 工具插件与运维约定

## 受控 Shell 插件（plugins/shell_tools.py）

run_command 工具采用第一档白名单策略，仅放行 config/tools.json 中允许的只读命令，禁止管道符、分号和重定向等 shell 元字符。
执行前强制校验路径边界并清理环境变量，仅保留 PATH、HOME、LANG 等最小安全子集，避免泄露 API Key。

## 联网与防 SSRF 插件（plugins/web_tools.py）

web_search 默认零密钥使用 DuckDuckGo HTML 检索，失败或无结果时自动回退到 Bing RSS；配置密钥后也支持 Tavily 与 Brave API。
fetch_webpage 在发请求及每次 HTTP 重定向前校验目标域名解析出的全部 IP，拦截回环地址（127.0.0.0/8）、私网段（10/8、172.16/12、192.168/16）及云元数据地址（169.254.169.254），防范 SSRF 攻击。

## 本地网页端与部署红线

本地网页调试界面由 scripts/web_chat.py 提供，基于 Python 标准库 ThreadingHTTPServer 默认监听 127.0.0.1:8765。
上线部署时绝对禁止覆盖服务器现有的 data/ 目录、.env 密钥文件或线上配置文件。
发送外部邮件、提交外部表单等产生外部副作用的操作，必须经过用户显式确认后才能执行。
