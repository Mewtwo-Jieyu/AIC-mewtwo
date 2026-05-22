# Phase 63 Scheduler Alignment Go / No-Go

## Conclusion

Phase 63 可以收口为 Go：cb_sim 已能输出与 Phase 62 vLLM row 同构的 DP-aware aligned descriptor。它仍然只是一条 experimental descriptor 线，不能接 compare CLI，更不能进入默认 latency。

| 项 | 判断 |
|---|---|
| cb_sim aligned row | Go |
| `engine_core_dp_step` key | Go |
| `iteration == engine_step_id` | Go |
| `dp_rank` 显式 | Go |
| latency/perf 字段 | 未加入 |
| compare CLI | 暂不接，留到 Phase 64 |
| 默认 AIC | 不改 |

## What Changed

| 文件 | 内容 |
|---|---|
| `forward_descriptor.py` | 新增 `VLLMSchedulerAlignedDescriptor` 和 builder |
| `validate_cb_simulator.py` | 新增 `--experimental-scheduler-alignment-descriptor` |
| `diagnose_cb_iter_latency.py` | 新增同名 experimental 输出，`dp>1` 需要显式 `--scheduler-alignment-dp-rank` |
| `test_forward_descriptor_scheduler_alignment.py` | 覆盖 schema、fail-fast、cb_sim ordinal |

## Why Compare Is Still No-Go

| 问题 | 处理 |
|---|---|
| vLLM Phase 62 row 已有 DP step | 可作为右侧真实 row |
| cb_sim Phase 63 row 已有 DP step | 可作为左侧候选 row |
| 两者是否同源 | 还没做字段级 compare 规则 |
| ordinal 是否可直接对齐 | 需要 Phase 64 明确，只能用 `alignment_key`，不能退回 phase ordinal |
| 默认模型 | 不接 |

## Next Gate

| Phase 64 前置 | 标准 |
|---|---|
| CSV 同构 | vLLM Phase62 dedup CSV 和 cb_sim Phase63 CSV header 对齐 |
| key 对齐 | 只允许 `engine_core_dp_step` |
| 缺 key | fail-fast |
| 禁止字段 | 无 `ms/residual/profiler/NCCL/sync/throughput` |
| 输出 | 只能 experimental compare CSV |

## Stop Rules

| 现象 | 动作 |
|---|---|
| 需要 latency 才能解释 | 停止 |
| 需要默认 `PerfDatabase` | 停止 |
| 需要 `run_static` 或 `IterationLatencyCalculator` 接入 | 停止 |
| 需要把 cb_sim row 复制成多 DP | 停止 |
| 需要 phase ordinal 代替 alignment key | 停止 |
