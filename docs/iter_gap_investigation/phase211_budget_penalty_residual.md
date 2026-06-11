# Phase211: Budget Penalty Residual

## Decision

| Item | Result |
|---|---|
| Default AIC | No-Go |
| VLLMBackend.run_agg | Unchanged |
| PerfDatabase | No-Go |
| GPU benchmark | No-Go in Phase211 |
| Allowed use | diagnostic-only analysis |
| Boundary flags | diagnostic_only=true valid_for_default=false perf_database=false |

## Pair Residuals

| topology_key | shape_key | control_bt | holdout_bt | clean_budget_effect | steady_state_time_ratio | penalty_overstatement | default_readiness |
|---|---|---:|---:|---:|---:|---:|---|
| tp8_dp1_ep8 | isl4000_osl2000_batch128 | 4000 | 65536 | 1.012212 | 3.688569 | 3.644068 | No-Go |
| tp4_dp2_ep8 | isl4000_osl2000_batch128 | 4000 | 65536 | 1.133187 | 6.840699 | 6.036690 | No-Go |
| tp8_dp1_ep8 | isl12000_osl2000_batch128 | 12000 | 65536 | 1.074279 | 2.460281 | 2.290169 | No-Go |
| tp4_dp2_ep8 | isl12000_osl2000_batch128 | 12000 | 65536 | 0.993809 | 4.198380 | 4.224534 | No-Go |

## Interpretation

All 4 holdout pairs keep clean_budget_effect near 1 while steady_state_time_ratio is much larger.
That means the scheduler penalty is overstated in cb_sim relative to observed clean GPU throughput.
The pair behavior differs by topology and shape, so this is not a global constant.
The next mechanism candidate must stay exact-keyed by topology_key + shape_key + budget pair.
A raw multiplier or direct default AIC integration remains No-Go.
