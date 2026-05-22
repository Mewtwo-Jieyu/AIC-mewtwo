# Phase 64 Scheduler Alignment Compare Schema

## Conclusion

Phase 64 新增 scheduler alignment compare 接口，但只允许按 `(alignment_key, dp_rank)` 精确 join。它不使用 phase ordinal，不输出 latency，不接默认 `cb_sim`。

| 项 | 规则 |
|---|---|
| join key | `(alignment_key, dp_rank)` |
| `alignment_key_type` | 只允许 `engine_core_dp_step` |
| cb 缺 key | fail-fast |
| vLLM 缺 key | fail-fast |
| 重复 key | fail-fast |
| phase ordinal | 禁止 |
| 输出资格 | `valid_for_default=false`，`perf_database=false`，`diagnostic_only=true` |

## Compare Row

| 字段组 | 字段 |
|---|---|
| alignment | `alignment_key`、`alignment_key_type`、`dp_rank` |
| identity | `scenario`、`phase`、`topology_key` |
| cb side | `cb_engine_step_id`、`cb_iter_index`、tokens、reqs、`cb_forward_token_count`、`cb_forward_regime`、`cb_cudagraph_runtime_mode` |
| vLLM side | `vllm_engine_step_id`、`vllm_iter_index`、tokens、reqs、`vllm_forward_token_count`、`vllm_forward_regime`、`vllm_cudagraph_runtime_mode` |
| deltas | context/decode/total token delta、context/decode/total request delta、forward token delta |
| same flags | `same_phase`、`same_topology_key`、`same_scheduled_total_tokens`、`same_scheduled_total_reqs`、`same_forward_token_count`、`same_forward_regime`、`same_cudagraph_runtime_mode` |
| boundary | `valid_for_default`、`perf_database`、`diagnostic_only` |

## CLI Boundary

| 入口 | 说明 |
|---|---|
| `diagnose_cb_iter_latency.py --experimental-scheduler-alignment-descriptor` | 生成 cb_sim aligned descriptor |
| `--scheduler-alignment-vllm-csv` | 读取 Phase62 vLLM aligned descriptor CSV |
| `--scheduler-alignment-compare-out` | 输出 strict key compare CSV |

`--scheduler-alignment-compare-out` 必须和 `--scheduler-alignment-vllm-csv` 同时使用。`dp>1` 仍必须显式传 `--scheduler-alignment-dp-rank`，不允许默认猜 DP。

## Forbidden Fields

| 禁止 | 原因 |
|---|---|
| `latency_ms`、`duration_ms` | compare 只比 shape |
| `residual_ms` | 不包装经验缺口 |
| profiler / NCCL / sync 字段 | 都不是 clean descriptor |
| throughput 字段 | 验收指标不能混入 scheduler compare |
