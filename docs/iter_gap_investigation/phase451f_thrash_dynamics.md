# Phase451-F thrash dynamics

结论: `H2_wave_overshoot`。F1-F3 只做动力学判别;fix gate=`blocked_pending_wave_model`。本报告不改 scheduler/runtime/PerfDB/validate gate。

- Default AIC remains `No-Go`.

## Summary

| section | metric | value | target | status | note |
|---|---|---:|---|---|---|
| f1_loop | total_preemptions | 824 |  |  |  |
| f1_loop | repeat_share | 0.90534 | dominant repeat-victim signature |  |  |
| f1_loop | near_mixed_share | 1 | >=0.80 supports H2 |  |  |
| f1_loop | nearest_mixed_distance_median | 1 |  |  |  |
| f1_loop | loop_interval_count | 674 |  |  |  |
| f1_loop | loop_interval_median | 2 |  |  |  |
| f1_loop | loop_interval_cv | 1.899665 | low supports H1; high supports H2 |  |  |
| f1_loop | over_blocks_median | 26.5 |  |  |  |
| f1_loop | over_blocks_p90 | 72 |  |  |  |
| f2_ledger | sim_capacity_blocks | 28633 | 28633 |  |  |
| f2_ledger | real_capacity_blocks | 28633 | 28633 |  |  |
| f2_ledger | capacity_ratio | 1 | 1.0 |  |  |
| f2_ledger | prefill_chunk_blocks | 500 | 8000/16=500 |  |  |
| f2_ledger | decode_growth_blocks | 125 | about 120-125 |  |  |
| f2_ledger | full_request_blocks | 625 | (8000+2000)/16=625 |  |  |
| f2_ledger | capacity_full_request_ceiling | 45.8128 |  |  |  |
| f2_ledger | block_deficit_detected | False | false for H2 |  |  |
| f3_counterfactual | staggered_preemptions | 0 | 0 supports H2 |  |  |
| f3_counterfactual | staggered_preemptions_per_request_slot | 0 | <=0.05 supports H2; >=0.20 supports H1 |  |  |
| decision | hypothesis | H2_wave_overshoot |  | pass | preemptions cluster near mixed waves and disappear when lifecycle is staggered |
| decision | phase451f_fix_gate | blocked_pending_wave_model |  | blocked |  |
| decision | phase451e_runtime_patch | not_applied |  | blocked | report-only F1-F3; H2 requires wave model design before runtime changes |

## Boundary

- Step 1-3 are report-only.
- H2 does not authorize a runtime patch by itself; it requires a non-parametric wave model design and a separate red-green phase.
- Default AIC remains No-Go.
