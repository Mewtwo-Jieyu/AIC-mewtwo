# Phase199: Holdout Budget Mechanism Interpretation

## Decision

| Item | Result |
|---|---|
| Scope | diagnostic mechanism candidate only |
| Default AIC | No-Go |
| PerfDatabase | No-Go |
| GPU benchmark | No-Go in Phase199 |
| Boundary flags | diagnostic_only=true valid_for_default=false perf_database=false |

## Pair Analysis

| topology_key | shape_key | control_max_bt | holdout_max_bt | clean_high_over_control | clean_delta_from_1 | steady_state_time_ratio |
|---|---|---:|---:|---:|---:|---:|
| tp8_dp1_ep8 | isl4000_osl2000_batch128 | 4000 | 65536 | 1.012212 | 0.012212 | 3.688569 |
| tp4_dp2_ep8 | isl4000_osl2000_batch128 | 4000 | 65536 | 1.133187 | 0.133187 | 6.840699 |
| tp8_dp1_ep8 | isl12000_osl2000_batch128 | 12000 | 65536 | 1.074279 | 0.074279 | 2.460281 |
| tp4_dp2_ep8 | isl12000_osl2000_batch128 | 12000 | 65536 | 0.993809 | -0.006191 | 4.198380 |

## Interpretation

The holdout rows show very large steady-state time ratios while clean high/control throughput stays near 1.0 or above 1.0 in three of four pairs.
That supports the diagnostic hypothesis that cb_sim over-penalizes high-budget steady-state scheduler cost.
The tp4_dp2_ep8 12k2k pair is below 1.0 by about 0.6%, which is too small and too shape-specific to justify a raw budget_gap multiplier.
The 4k2k and 12k2k behavior differs, so any future mechanism item must be keyed by topology_key + shape_key + max_bt; it is not a global constant.
These results remain diagnostic-only and are not valid for Default AIC or PerfDatabase.
