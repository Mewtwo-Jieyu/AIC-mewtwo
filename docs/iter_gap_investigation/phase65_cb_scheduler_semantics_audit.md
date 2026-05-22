# Phase 65 cb_sim Scheduler Semantics Audit

## Conclusion

Phase 65 当前是 No-Go：Phase64 compare 接口按 `(alignment_key, dp_rank)` strict join 是正确的，但 cb_sim 侧 aligned descriptor 不是 vLLM 同源 DP scheduler row。它把一条 cb_sim 全局 scheduler trace 按用户传入的 `dp_rank` 打上 DP key，不能代表 vLLM EngineCore 每个 DP 的真实调度序列。

| 项 | 结论 |
|---|---|
| compare 接口 | 正确，不能退回 phase ordinal 或交集对齐 |
| cb_sim aligned key | 格式正确，但语义不是 vLLM DP scheduler step |
| cb_sim request/token split | 来自单一 cb_sim scheduler trace，不是 DP-aware split |
| cb_sim forward regime | `AIC_UNSET:<tokens>`，不是 vLLM runtime padded shape / graph mode |
| 默认 AIC | 不改，不写 perf data |

## Source Findings

| 位置 | 语义 |
|---|---|
| `build_cb_scheduler_aligned_descriptors(...)` | 对 `CBIterationTraceRow` 做 `enumerate(rows)`，用 ordinal 生成 `engine_step_id` |
| `dp_rank` 参数 | 只进入 `alignment_key`，不改变 scheduler 状态和 row 内容 |
| `forward_token_count` | 当前是 `row.prefill_tokens + row.decode_batch_size` |
| `cudagraph_runtime_mode` | 固定 `AIC_UNSET` |
| `collect_cb_iteration_trace(...)` | 单 waiting/running/completed 池，不区分 vLLM DP EngineCore 调度序列 |

## Evidence

Phase65 gap 表只含 descriptor shape 字段，不含耗时、残差、profiler、NCCL、sync 或 throughput 字段。

| artifact | 内容 |
|---|---|
| `phase65_scheduler_semantics_gap.csv` | cb_sim vs Phase62 vLLM aligned descriptor shape gap |
| 输入 cb_sim | `10k2k_b32`、`num_requests=32`、`bt=8192`、`tp4dp2ep8`，分别显式输出 `dp_rank=0/1` |
| 输入 vLLM | Phase62 dedup aligned CSV |

| 对齐键 | cb_sim row | vLLM row | 判断 |
|---|---|---|---|
| `engine_dp:0:step:2` | mixed `8191+1`，`AIC_UNSET:8192` | pure_decode `0+16`，`FULL:16` | 不同源 |
| `engine_dp:1:step:2` | mixed `8191+1`，`AIC_UNSET:8192` | mixed `240+1` padded `NONE:248` | 同 phase 但 shape 不同源 |
| `engine_dp:1:step:2001` | pure_decode `31`，`AIC_UNSET:31` | pure_decode `15` padded `FULL:16` | tail 语义不同 |

## Interpretation

这不是 Phase64 compare builder 的问题。当前 cb_sim descriptor builder 按字段构造合法 row，但源 row 本身不是 vLLM scheduler row。继续按 strict key compare 会正确失败；如果改成取交集或 phase ordinal，会隐藏语义差异，制造假对齐。

## Next Boundary

Phase66 如果继续，只能做 `vLLM-like scheduler descriptor generator` 设计，目标是生成与 vLLM DP EngineCore 同源的 request/token split 和 runtime padded shape。不能在 Phase65 直接修改 builder 兜底，也不能接 compare CLI。
