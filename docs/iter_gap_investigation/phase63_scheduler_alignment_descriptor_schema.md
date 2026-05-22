# Phase 63 Scheduler Alignment Descriptor Schema

## Conclusion

Phase 63 给 cb_sim 增加 DP-aware scheduler alignment descriptor。它只补对齐字段，不建 latency，不写 `PerfDatabase`，不接默认 `cb_sim`。

| 项 | 结论 |
|---|---|
| 对齐键 | `engine_core_dp_step` |
| key 格式 | `engine_dp:<dp_rank>:step:<engine_step_id>` |
| cb_sim step | DP 内 scheduler ordinal，从 `0` 开始 |
| `iteration` | 必须等于 `engine_step_id` |
| 输出资格 | `valid_for_default=false`，`perf_database=false`，`diagnostic_only=true` |

## Schema

| 字段组 | 字段 |
|---|---|
| alignment | `alignment_key`、`alignment_key_type`、`engine_step_id`、`dp_rank` |
| identity | `source`、`scenario`、`iteration`、`phase` |
| scheduler tokens | `scheduled_context_tokens`、`scheduled_decode_tokens`、`scheduled_total_tokens` |
| scheduler requests | `scheduled_context_reqs`、`scheduled_decode_reqs`、`scheduled_total_reqs` |
| budget | `max_num_batched_tokens`、`max_num_seqs` |
| runtime link | `forward_token_count`、`forward_regime`、`cudagraph_runtime_mode` |
| topology | `topology_key`、`tp`、`dp`、`moe_tp`、`moe_ep` |
| boundary | `valid_for_default`、`perf_database`、`diagnostic_only` |

## Fail-Fast Rules

| 条件 | 处理 |
|---|---|
| `alignment_key_type` 不是 `engine_core_dp_step` | 拒绝 |
| `dp_rank < 0` 或 `dp_rank >= dp` | 拒绝 |
| `engine_step_id < 0` | 拒绝 |
| `iteration != engine_step_id` | 构造时不允许发生 |
| `topology_key` 和 `tp/dp/moe_tp/moe_ep` 不一致 | 拒绝 |
| `forward_token_count < scheduled_total_tokens` | 拒绝 |

## CLI Boundary

| 入口 | 输出 |
|---|---|
| `validate_cb_simulator.py --experimental-scheduler-alignment-descriptor` | validate input-shape aligned descriptor，`dp=1` |
| `diagnose_cb_iter_latency.py --experimental-scheduler-alignment-descriptor` | cb_sim trace aligned descriptor |

`diagnose_cb_iter_latency.py` 在 `dp>1` 时必须显式传 `--scheduler-alignment-dp-rank`。原因是当前 cb_sim trace 是单 DP scheduler 视图，不能静默复制成多个 DP。

## Forbidden Fields

| 禁止字段 | 原因 |
|---|---|
| `latency_ms`、`duration_ms` | descriptor 不是耗时 |
| `residual_ms` | 不包装经验缺口 |
| `profiled_cuda_time_ms` | profiler 数据不能入默认模型 |
| `nccl_trace_line_count` | trace 行数不是性能 |
| `sync_wait_ms` | sync probe 改变执行边界 |
| `throughput_ratio` | 验收指标不能混入 shape descriptor |
