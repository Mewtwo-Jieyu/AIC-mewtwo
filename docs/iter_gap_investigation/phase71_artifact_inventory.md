# Phase 71 Artifact Inventory

## Conclusion

Phase71 did not find an existing local `32k1k_b16` scheduler alignment artifact, so it captured one descriptor-only vLLM holdout. The artifact is usable for scheduler shape analysis and does not contain a timing marker.

| item | result |
|---|---|
| searched local artifact | no pre-existing `32k1k_b16` scheduler alignment directory |
| remote entry | `ws-faaf0de74ef9a14d-worker-tc88x` |
| workload | `32k1k_b16 bt8192 tp4dp2ep8` |
| benchmark requests | `16/16` succeeded |
| raw marker rows | `8024` |
| deduped `(alignment_key, dp_rank)` rows | `2006` |
| source markers restored | yes, marker grep returned empty |
| GPU occupancy restored | yes, `run_qwen3_8b_grpo.py` restarted |

## Local Artifacts

| artifact | purpose |
|---|---|
| `phase71_32k1k_b16_scheduler_alignment/scheduler_alignment_markers.log` | raw descriptor-only vLLM marker log |
| `phase71_32k1k_b16_scheduler_alignment/scheduler_alignment_rows.csv` | parsed raw descriptor rows |
| `phase71_32k1k_b16_scheduler_alignment/scheduler_alignment_rows_dedup_by_alignment_dp.csv` | deduped descriptor rows by `(alignment_key, dp_rank)` |
| `phase71_32k1k_b16_scheduler_alignment/bench_result.json` | request success metadata only; not used as descriptor timing data |
| `phase71_32k1k_b16_scheduler_alignment/bench_records.jsonl` | request success records only; not used for latency modeling |
| `phase71_32k1k_b16_scheduler_alignment/runner.log` | remote run bookkeeping |
| `phase71_32k1k_b16_generator_fail_fast.txt` | current generator fail-fast record |

## Descriptor Row Summary

| group | count |
|---|---:|
| prefill | `8` |
| pure_decode | `1998` |
| mixed | `0` |
| DP0 rows | `1003` |
| DP1 rows | `1003` |

## Key Rows

| row | descriptor |
|---|---|
| `engine_dp:0:step:0` | `prefill 8192+0 / NONE:8192` |
| `engine_dp:0:step:1` | `prefill 8192+0 / NONE:8192` |
| `engine_dp:0:step:2` | `prefill 8192+0 / NONE:8192` |
| `engine_dp:0:step:3` | `prefill 7536+0 / NONE:7536` |
| `engine_dp:0:step:4` | `pure_decode 0+8 / FULL:8` |
| `engine_dp:1:step:0` | `prefill 8192+0 / NONE:8192` |
| `engine_dp:1:step:1` | `prefill 8192+0 / NONE:8192` |
| `engine_dp:1:step:2` | `prefill 8192+0 / NONE:8192` |
| `engine_dp:1:step:3` | `prefill 7536+0 / NONE:7536` |
| `engine_dp:1:step:4` | `pure_decode 0+8 / NONE:8` |
| `engine_dp:0:step:1002` | `pure_decode 0+8 / FULL:8` |
| `engine_dp:1:step:1002` | `pure_decode 0+8 / FULL:8` |

## Boundary

The parsed descriptor CSV header contains scheduler split, request split, padded forward shape, graph mode, topology, and boundary flags only. It does not add latency, residual, profiler, NCCL, sync, throughput, or perf-table fields.

## Phase72 Link

Phase72 turns this artifact into a rule-design review only. See `phase72_scheduler_regime_taxonomy.md` and `phase72_32k1k_scheduler_rule_design.md`.
