# Phase171: Budget Mechanism Candidate

## Decision

| Item | Result |
|---|---|
| Scope | Diagnostic mechanism candidate only |
| Default AIC | No-Go |
| PerfDatabase | No-Go |
| GPU benchmark | No-Go in Phase171-dev |
| Exact gap upper bound | Diagnostic only, not a model multiplier |

## Mechanism Check

| topology_key | clean_budget_effect | sim_budget_effect | steady_state_time_ratio | depenalized_budget_effect | depenalized_error_ratio | exact_gap_upper_bound |
|---|---:|---:|---:|---:|---:|---:|
| tp4_dp2_ep8 | 1.167497 | 0.161350 | 6.377721 | 1.029045 | 0.881412 | 7.235797 |
| tp8_dp1_ep8 | 0.987382 | 0.283937 | 3.624202 | 1.029045 | 1.042196 | 3.477467 |

## Interpretation

The depenalized estimate removes the high-budget steady-state time penalty from the existing cb_sim diagnostic output. If it moves closer to the clean budget effect, the next candidate is a mechanism term around budget scheduler steady-state cost, not a direct budget_gap multiplier.

The exact_gap_upper_bound column is retained only as an exact diagnostic upper bound from Phase165. It is not valid for default AIC, not a PerfDatabase row, and not an interpolation or extrapolation rule.

diagnostic_only=true valid_for_default=false perf_database=false
