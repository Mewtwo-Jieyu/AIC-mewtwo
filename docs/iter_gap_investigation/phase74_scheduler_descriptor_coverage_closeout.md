# Phase 74 Scheduler Descriptor Coverage Closeout

## Conclusion

Scheduler descriptor coverage is now closed for three evidence cases. This is input-shape semantics coverage only, not latency modeling.

| scenario | status | strict compare | descriptor shape |
|---|---|---:|---|
| `10k2k_b32 bt8192 tp4dp2ep8` | covered | `4003/4003` all match | chunked prefill plus mixed bridge |
| `3k3k_b128 bt8192 tp4dp2ep8` | covered | `6002/6002` all match | short-ISL leader/follower plus mixed bridge |
| `32k1k_b16 bt8192 tp4dp2ep8` | covered | `2006/2006` all match | multi-chunk prefill then pure decode |

## What Is Covered

| field group | covered meaning |
|---|---|
| alignment | `engine_dp:<dp_rank>:step:<engine_step_id>` |
| scheduler split | context/decode token split and request split |
| runtime link | padded forward token count and forward regime |
| graph mode | observed `NONE`, `PIECEWISE`, and `FULL` rows |
| topology | `tp=4`, `dp=2`, `moe_tp=1`, `moe_ep=8` |
| boundary flags | `valid_for_default=false`, `perf_database=false`, `diagnostic_only=true` |

## What Is Not Covered

| item | status |
|---|---|
| other budgets | unknown |
| other topology | unknown |
| other model/backend | unknown |
| generic scheduler rule | not claimed |
| default `cb_sim` latency | No-Go |
| perf table | No-Go |

## Evidence Artifacts

| scenario | generated descriptor | compare artifact |
|---|---|---|
| `10k2k_b32` | `/private/tmp/aic_phase73_10k2k_vllm_like_descriptor.csv` | `/private/tmp/aic_phase73_10k2k_scheduler_shape_gap.csv` |
| `3k3k_b128` | `/private/tmp/aic_phase73_3k3k_vllm_like_descriptor.csv` | `/private/tmp/aic_phase73_3k3k_scheduler_shape_gap.csv` |
| `32k1k_b16` | `phase73_32k1k_b16_vllm_like_descriptor.csv` | `phase73_32k1k_b16_scheduler_shape_gap.csv` |

The `/private/tmp` artifacts are verification outputs, not committed evidence. The committed Phase73 `32k1k` artifacts are kept because Phase71 introduced that holdout in this worktree.

## Boundary

Do not use these descriptor rows as latency data. They contain no latency, residual, profiler, NCCL, sync, throughput, or perf-table fields. They also do not modify `PerfDatabase`, `run_static`, or `IterationLatencyCalculator`.

## Next Gate

Phase75 can decide whether to stage a minimal descriptor delivery set. If a new scenario is needed, first capture descriptor-only vLLM rows, then design a guarded rule. Do not broaden the generator by guessing.
