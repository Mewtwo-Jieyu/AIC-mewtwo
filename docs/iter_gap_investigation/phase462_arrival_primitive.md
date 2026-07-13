# Phase462 Step2a-3: EngineCore 可见到达原语审计

结论：**Step2a-3 测量门失败，Step2c 继续锁住。** Step2a-2 的 16/16 对齐仍证明首次分歧位于调度器外、EngineCore 可见到达边界；但“`process_inputs` 逐请求串行，因此相邻到达差一个处理耗时”的结构不成立。部署态 vLLM 0.19.0 在 `process_inputs` 之前使用异步 tokenizer 微批：最多 32 条请求、2ms 聚合窗、单线程 executor 每次执行一个批次。现有 raw 又没有 pipeline 起点、tokenizer batch 和 EngineCore 入队的同 request-id 时间戳，无法测得数值原语，也不能运行三签名原型。

| 门 | 结果 | 含义 |
|---|---|---|
| 源码结构门 | pass | 可定义为微批排队，不是逐请求串行排队 |
| 数值测量门 | fail | 现有 4 份 serve log 的完整边界配对为 0 |
| 离线原型门 | blocked | 禁止把 Iteration(1) 的“一拍”反推成处理耗时 |
| runtime | locked | 不改 runtime、PerfDB 或 gate |

## 源码规格

| 位置 | 已确认语义 |
|---|---|
| `vllm/utils/async_utils.py:24-52,90-132` | `AsyncMicrobatchTokenizer(max_batch_size=32, batch_wait_timeout_s=0.002)`；单 executor 串行的是微批调用，不是请求 |
| `vllm/renderers/base.py:141-147,379-390,843-862` | completion 请求先进入共享 async tokenizer，再转成 EngineInput |
| `vllm/v1/engine/async_llm.py:340-420` | tokenized EngineInput 经同步 `process_inputs` 后 `add_request_async`；`Added request` 仅在 `log_requests` 开启时出现 |
| `vllm/v1/engine/core.py:1136-1175` | 每个 engine step 前 drain input queue；空拍只证明下一请求当时尚未进入该 queue |
| cb_sim `simulator.py:152-157` | 当前把初始并发全部设为 `arrival_time_ms=0.0` |

部署源码 SHA256：`utils/async_utils.py`=82953898a30494e4da43236d140153c2c9d05a876b8267efbea27170ac074d7b; `renderers/base.py`=74d716ff06a0e5f480dbab51fd6c36246fc22f51178051810d5d3b0fe54653cb; `v1/engine/async_llm.py`=b101a3cd5ea4b7d0cf1852824d8fc96ec8c6a83c0396aad54ad9c1c8d69442c3; `v1/engine/core.py`=896730e749cbcabb487ce50c703974594d197fc31a1d3b26fe096197d142d2d5。

## 原语边界

正确的候选结构是**微批排队原语**：HTTP 请求进入 tokenizer 队列，按源码固定的 32 条上限和 2ms 聚合窗形成批次；单 executor 依次处理批次；批内请求完成 tokenization 后，再经 `process_inputs` 和 IPC 到 EngineCore。需要实测的数值至少有两段：`tokenizer batch service time(total_prompt_tokens, batch_size)` 与 `tokenize_done -> EngineCore received` 延迟。只按单请求 prompt 长度存一个常数，会漏掉 batch size，仍是隐藏经验项。

现有 Phase458 `bench_records.jsonl` 只有 request index、总 latency 和 token 数，没有提交时刻或 EngineCore 可见时刻；serve log 未开启 `Added request`，Step2b 也只记录抢占决策。因此以下三项均为 `not_run`：

| 原型签名 | 状态 |
|---|---|
| 逐请求可见步对齐 | not_run |
| 前 16 步继续 16/16 对齐 | not_run |
| 自抢占归零、重复 victim 收敛、26 -> 约 10 | not_run |

## 最小补观测规格

下一步只能是 logging-only 短跑，且先经确认：用同一个 request id 记录 `HTTP/render start`、`tokenizer batch id/start/end + batch size + total prompt tokens`、`add_request_async call/end`、`EngineCore receive + visible iteration`，统一 monotonic ns。patch 只加日志，`diagnostic_only=true / perf_database=false`，开销门仍为 2%。拿到这组边界时间后，才能测量原语并运行三签名原型。

本步骤 `report-only / diagnostic_only=true / valid_for_default=false / perf_database=false`。Default AIC 维持 No-Go。
