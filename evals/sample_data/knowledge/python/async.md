# Python 异步与并发笔记

## asyncio 事件循环

asyncio 使用单线程协作式多任务，适合高并发网络 I/O（如 HTTP 请求、长轮询）。
在协程里调用阻塞的 time.sleep 会卡住整个事件循环，应改用 await asyncio.sleep。

## 线程池与子进程

CPU 密集型计算受限于 GIL，可使用多进程或外部子进程执行。
调用外部命令必须使用固定 argv 数组并设置 shell=False，防止命令注入。
