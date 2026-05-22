# Phase 69 Holdout Capture Plan

## Conclusion

Phase69 only captures a real vLLM scheduler alignment descriptor holdout for `3k3k_b128`. It does not collect latency and does not change default `cb_sim`.

| item | decision |
|---|---|
| holdout | `3k3k_b128` |
| budget | `max_num_batched_tokens=8192` |
| topology | `tp=4, dp=2, moe_tp=1, moe_ep=8` |
| marker | reuse `AIC_SCHEDULER_ALIGNMENT_DESCRIPTOR_ROW` |
| parser | reuse Phase62 parser and dedupe by `(alignment_key, dp_rank)` |
| default path | unchanged |
| remote | `ws-faaf0de74ef9a14d-worker-tc88x` |

## Capture Boundary

| allowed | forbidden |
|---|---|
| scheduler token/request split | latency fields |
| DP-aware alignment key | residual fields |
| padded forward shape | profiler fields |
| cudagraph runtime mode | NCCL trace fields |
| topology fields | sync probe fields |
| `valid_for_default=false` | throughput fields |

## Execution

| step | action | output |
|---|---|---|
| 1 | stop the GPU occupancy script | no compute app conflict |
| 2 | run `collector/vllm/run_phase69_scheduler_alignment_holdout.sh` remotely | descriptor-only marker log |
| 3 | parse with `scripts/analyze_vllm_scheduler_descriptor_phase62.py` | raw and dedup CSV |
| 4 | pull back the minimal artifact directory | local evidence for generator audit |
| 5 | try the Phase66 generator for `3k3k_b128` | either descriptor CSV or fail-fast |
| 6 | write Go/No-Go | decide whether Phase70 can generalize |

## Output Files

| file | meaning |
|---|---|
| `phase69_3k3k_b128_scheduler_alignment/scheduler_alignment_rows.csv` | raw vLLM scheduler descriptor rows |
| `phase69_3k3k_b128_scheduler_alignment/scheduler_alignment_rows_dedup_by_alignment_dp.csv` | deduped rows by `(alignment_key, dp_rank)` |
| `phase69_3k3k_b128_scheduler_alignment/scheduler_alignment_markers.log` | raw marker-only log |
| `phase69_3k3k_b128_vllm_like_descriptor.csv` | generated only if Phase66 generator supports the holdout |
| `phase69_3k3k_b128_scheduler_shape_gap.csv` | generated only if both sides exist |

## Capture Result

| item | result |
|---|---|
| benchmark requests | `128/128` succeeded |
| raw marker rows | `24008` |
| dedup descriptor rows | `6002` |
| dedup phase split | prefill `2`, mixed `1`, pure_decode `5999` |
| DP row split | DP0 `3002`, DP1 `3000` |
| key example | `engine_dp:0:step:2 = 496+1 / PIECEWISE:512` |
| default path | unchanged |
| remote cleanup | vLLM source marker restored to `0`; stress script restarted |

The holdout capture is valid descriptor evidence. It still does not include latency, profiler, NCCL trace, sync, throughput, or residual fields.

## Stop Rules

| condition | action |
|---|---|
| marker is missing required descriptor fields | No-Go |
| parser rejects forbidden fields | No-Go |
| vLLM source restore leaves marker behind | No-Go |
| Phase66 generator rejects holdout shape | stop at generator semantics, do not fake gap rows |
| any result requires latency or residual to explain | stop |
