# Phase421 Wire BT65536

Verdict: `bt65536_wired_tp8_recovers_dp2_dirty_capacity_remains`.

Default AIC: `No-Go`.

## A/B Validation

| Scenario | Before error | After error | Class | Before sim | After sim |
|---|---:|---:|---|---:|---:|
| K2.5-tp4ep8dp2-32k3k | 1.734090 | 1.734090 | unchanged | 92.387868 | 92.387868 |
| K2.5-tp4ep8dp2-8k2k | 2.553139 | 2.553139 | unchanged | 351.608074 | 351.608074 |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 7.524468 | 5.642576 | improved | 20.725982 | 27.638441 |
| K2.5-tp8ep8-32k3k | 1.068795 | 1.068795 | unchanged | 56.078752 | 56.078752 |
| K2.5-tp8ep8-8k2k | 1.250461 | 1.250461 | unchanged | 166.971525 | 166.971525 |
| K2.5-tp8ep8-8k2k-bt65536 | 1.190376 | 1.119881 | improved | 116.324618 | 155.069882 |

## Post-Wiring Trace

`Initial reqs` can exceed `full-fit` by one when the final request is a partial chunk.

| Scenario | Configured bt | Initial tokens | Initial reqs | Capacity | MoE path | EP path | Status |
|---|---:|---:|---:|---:|---|---|---|
| K2.5-tp8ep8-8k2k-bt65536 | 65536 | 65536 | 9/8 | 34.355200 | phase397v_sol_out_of_coverage | ep8_alltoall_fallback | wired_full_budget |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 65536 | 24000 | 3/8 | 2.574400 | moe_perf_lookup | ep8_alltoall_fallback | wired_but_capacity_limited |

## Phase422 Target

`phase422_resolve_dirty_dp2_bt65536_capacity_before_large_step_cost`.

The TP8 bt65536 path now reaches the 64k step. The DP2 bt65536 path is still limited by the dirty capacity row before it can exercise the 64k step.

## Guardrails

- `diagnostic_only=true`.
- `valid_for_default=false`.
- `runtime_modified=false`.
- `perf_database=false`.
- `gpu_allowed=false`, `ssh_allowed=false`.
