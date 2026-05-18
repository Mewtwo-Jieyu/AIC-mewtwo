# Phase 58: Scheduler Descriptor Go/No-Go

## 结论

Phase 58 可以作为 experimental descriptor 进入下一步 polish，但不能进入 latency 建模。

| 检查项 | 结论 |
|---|---|
| schema | Go，字段只包含 scheduler/runtime shape 机制输入 |
| validate experimental 输出 | Go，显式 flag 才输出 CSV |
| diagnose experimental 输出 | Go，显式 flag + 输出路径才输出 CSV |
| 默认 `cb_sim` | Go，默认 validate 不触发 scheduler descriptor |
| latency / perf | No-Go，没有 clean timing 数据 |
| 真实 vLLM scheduler runtime | No-Go，当前没有同源 request split + cudagraph mode 输入 |

## 已满足

| 门槛 | 结果 |
|---|---|
| request split | `scheduled_context_reqs` / `scheduled_decode_reqs` 必填 |
| topology | `topology_key` 必须和 `tp/dp/moe_tp/moe_ep` 一致 |
| runtime link | `forward_token_count`、`forward_regime`、`cudagraph_runtime_mode` 已进入 descriptor |
| 边界字段 | 固定 `valid_for_default=false`、`perf_database=false`、`diagnostic_only=true` |
| 禁止字段 | schema 不含 latency、residual、profiler、trace、throughput 字段 |

## 不允许升级

| 方向 | 原因 |
|---|---|
| 默认 latency | 只有 descriptor，没有模块 perf |
| `PerfDatabase` | 没有 clean module timing |
| residual 补偿 | 违反 Phase 40-44 收口结论 |
| profiler/NCCL/debug 数字 | 只能诊断，不能做模型输入 |

## Phase 59 入口

| 可做 | 不做 |
|---|---|
| descriptor 字段命名 polish | 不加 ms 字段 |
| 输出 CSV 位置和文档链接整理 | 不接默认 `cb_sim` |
| 和 Phase 41 compiled-body key 做 descriptor index | 不写 perf table |
| coverage matrix 文档化 | 不做 latency 公式 |

## 口径风险

| 风险 | 收口 |
|---|---|
| `validate_input_shape` 被误读成真实 vLLM runtime | 明确标成 cb_sim/input-shape descriptor |
| cb_sim trace 被误读成线上 scheduler | 明确标成 simulator trace |
| vLLM compare row 被误用为因果判断 | 暂不接 CLI compare；等真实同源 vLLM scheduler row 后再接 |

## 停止条件

| 情况 | 处理 |
|---|---|
| 缺 request split | 停止，不补默认值 |
| 需要 latency 才能解释 | 停止，descriptor 不负责建模 |
| 默认 validate 发生变化 | 回退 Phase 58 代码 |
| 出现 `PerfDatabase` / `run_static` / `IterationLatencyCalculator` 新接入 | 停止并回滚 |
