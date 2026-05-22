# Phase 70 Short-ISL Scheduler Generator

## Conclusion

Phase70 extends the experimental vLLM-like scheduler descriptor generator to cover the Phase69 `3k3k_b128 bt8192 tp4dp2ep8` short-ISL holdout. This is still descriptor-only and does not change default `cb_sim` latency.

| item | result |
|---|---|
| target holdout | `3k3k_b128 bt8192 tp4dp2ep8` |
| generated rows | `6002` |
| strict compare rows | `6002` |
| same flags | all token/request/forward/regime/cudagraph flags are `1` |
| deltas | all token/request/forward deltas are `0` |
| existing `10k2k_b32` | still `4003` strict-match rows |
| default path | unchanged |

## Short-ISL Rule

The new branch is intentionally narrow. It only supports the captured Phase69 first-evidence shape:

| field | value |
|---|---|
| `isl` | `3000` |
| `osl` | `3000` |
| `concurrency` | `128` |
| `max_num_batched_tokens` | `8192` |
| `tp/dp/moe_tp/moe_ep` | `4/2/1/8` |
| `graph_padding_multiple` | `8` |

Unsupported short-ISL shapes still fail fast. This avoids turning the observed DP split and graph padding behavior into an unverified general rule.

## Matched Rows

| row | expected descriptor |
|---|---|
| `engine_dp:0:step:0` | `prefill 3000+0 / NONE:3000` |
| `engine_dp:0:step:1` | `pure_decode 0+1 / NONE:1` |
| `engine_dp:0:step:2` | `mixed 496+1 / PIECEWISE:512` |
| `engine_dp:1:step:0` | `prefill 3512+0 / NONE:3512` |
| `engine_dp:1:step:1` | `pure_decode 0+65 / PIECEWISE:512` |
| `engine_dp:1:step:2` | `pure_decode 0+65 / FULL:72` |
| `engine_dp:0:step:3001` | `pure_decode 0+62 / FULL:64` |
| `engine_dp:1:step:2999` | `pure_decode 0+65 / FULL:72` |

## Artifacts

| artifact | purpose |
|---|---|
| `phase70_3k3k_b128_vllm_like_descriptor.csv` | generated cb_sim vLLM-like descriptor |
| `phase70_3k3k_b128_scheduler_shape_gap.csv` | strict compare against Phase69 vLLM rows |
| `/private/tmp/aic_phase70_10k2k_scheduler_shape_gap.csv` | non-committed regression check for Phase62 shape |

## Boundary

This is not a latency model. It emits no latency, residual, profiler, NCCL, sync, throughput, or perf-table fields. The output remains experimental-only with `valid_for_default=false`, `perf_database=false`, and `diagnostic_only=true`.

## Next Gate

Phase71 can decide whether to abstract broader short-ISL rules. That requires additional real vLLM descriptor holdouts; it cannot use phase ordinal, intersection-only matching, residuals, or timing fields to force generalization.
