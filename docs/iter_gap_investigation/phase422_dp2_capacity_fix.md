# Phase422 DP2 Capacity Fix

Verdict: `dp2_bt65536_capacity_truth_is_dirty_reference_not_fixable_by_wiring`.

Default AIC: `No-Go`.

Phase422 fixes the validation harness wiring for `max_num_seqs`, then accounts for the DP2 bt65536 capacity truth. The logged retry5e capacity is only `25,744` KV tokens (`1,609` blocks) with `max_num_seqs=128`; that truth blocks an independent 64k prefill step. There is no log-backed larger capacity to wire in.

## Validation A/B

| Scenario | Phase421 error | Phase422 error | Class | Phase422 sim |
|---|---:|---:|---|---:|
| K2.5-tp4ep8dp2-32k3k | 1.734090 | 1.734090 | unchanged | 92.387868 |
| K2.5-tp4ep8dp2-8k2k | 2.553139 | 2.553139 | unchanged | 351.608074 |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 5.642576 | 5.642576 | unchanged | 27.638441 |
| K2.5-tp8ep8-32k3k | 1.068795 | 1.068795 | unchanged | 56.078752 |
| K2.5-tp8ep8-8k2k | 1.250461 | 1.250461 | unchanged | 166.971525 |
| K2.5-tp8ep8-8k2k-bt65536 | 1.119881 | 1.119881 | unchanged | 155.069882 |

## Capacity Truth

| Scenario | KV tokens | Blocks | max_num_seqs | Full sequence capacity | Source lines |
|---|---:|---:|---:|---:|---|
| K2.5-tp8ep8-8k2k-bt65536 | 343552 | 21472 | 256 | 34.355200 | 195 |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 25744 | 1609 | 128 | 2.574400 | 200;210 |

## Trace

| Scenario | Configured bt | Initial tokens | Initial reqs | Trace status | MoE path | EP path |
|---|---:|---:|---:|---|---|---|
| K2.5-tp8ep8-8k2k-bt65536 | 65536 | 65536 | 9/8 | tp8_reaches_64k_step | phase397v_sol_out_of_coverage | ep8_alltoall_fallback |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 65536 | 24000 | 3/8 | real_capacity_blocks_64k_step | moe_perf_lookup | ep8_alltoall_fallback |

## Phase423 Target

`phase423_recollect_or_remove_dp2_bt65536_dirty_reference_before_large_step_cost`.

The next step is not to invent capacity. Either recollect a clean DP2 bt65536 reference with consistent KV capacity, or remove this dirty cached reference from the acceptance surface before judging large-step cost.

## Guardrails

- `diagnostic_only=true`.
- `valid_for_default=false`.
- `runtime_modified=false`.
- `perf_database=false`.
- `gpu_allowed=false`, `ssh_allowed=false`.
