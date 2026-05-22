# Phase 73 32k1k Scheduler Branch Closeout

## Conclusion

Phase73 implements the narrow `32k1k_b16 bt8192 tp4dp2ep8` vLLM-like scheduler descriptor branch. It only extends descriptor shape coverage and does not change default `cb_sim` latency.

| item | result |
|---|---|
| target scenario | `32k1k_b16 bt8192 tp4dp2ep8` |
| generated descriptor rows | `2006` |
| DP rows | DP0 `1003`, DP1 `1003` |
| phase split | prefill `8`, pure_decode `1998`, mixed `0` |
| strict compare rows | `2006` |
| same flags | all `2006/2006` |
| deltas | all token/request/forward deltas are `0` |
| default path | unchanged |

## Implemented Rule

| step range | DP0 descriptor | DP1 descriptor |
|---|---|---|
| `0..2` | `prefill 8192+0 / NONE:8192` | `prefill 8192+0 / NONE:8192` |
| `3` | `prefill 7536+0 / NONE:7536` | `prefill 7536+0 / NONE:7536` |
| `4` | `pure_decode 0+8 / FULL:8` | `pure_decode 0+8 / NONE:8` |
| `5..1002` | `pure_decode 0+8 / FULL:8` | `pure_decode 0+8 / FULL:8` |

## Guarded Scope

| gate | value |
|---|---|
| `isl` | `32000` |
| `osl` | `1000` |
| `concurrency` | `16` |
| `max_num_batched_tokens` | `8192` |
| `tp/dp/moe_tp/moe_ep` | `4/2/1/8` |
| graph padding multiple | `8` |

Unsupported `32k1k` variants still fail fast. This avoids turning one holdout into an unverified generic long-ISL scheduler rule.

## Artifacts

| artifact | purpose |
|---|---|
| `phase73_32k1k_b16_vllm_like_descriptor.csv` | generated cb_sim vLLM-like descriptor |
| `phase73_32k1k_b16_scheduler_shape_gap.csv` | strict compare against Phase71 real vLLM rows |

## Boundary

This remains descriptor-only. It emits no latency, residual, profiler, NCCL, sync, throughput, or perf-table fields. It does not write `PerfDatabase` and does not modify `run_static` or `IterationLatencyCalculator`.

## Phase74 Link

Phase74 summarizes this branch as one of three covered scheduler descriptor evidence cases.
