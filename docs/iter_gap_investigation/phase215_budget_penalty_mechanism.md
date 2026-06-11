# Phase215: Budget Penalty Mechanism Comparison

## Decision

| Item | Result |
|---|---|
| Default AIC | No-Go |
| VLLMBackend.run_agg | Unchanged |
| PerfDatabase | No-Go |
| GPU benchmark | No-Go in Phase215 |
| Recommended boundary | diagnostic_only_exact_key |
| Boundary flags | diagnostic_only=true valid_for_default=false perf_database=false |

## Mechanism Comparison

| topology_key | shape_key | clean_budget_effect | steady_state_time_ratio | steady_linear_error | no_budget_penalty_error | topology_shape_spread | recommended_model_boundary |
|---|---|---:|---:|---:|---:|---:|---|
| tp8_dp1_ep8 | isl4000_osl2000_batch128 | 1.012212 | 3.688569 | 3.644068 | 1.012212 | 1.061318 | diagnostic_only_exact_key |
| tp4_dp2_ep8 | isl4000_osl2000_batch128 | 1.133187 | 6.840699 | 6.036690 | 1.133187 | 1.140246 | diagnostic_only_exact_key |
| tp8_dp1_ep8 | isl12000_osl2000_batch128 | 1.074279 | 2.460281 | 2.290169 | 1.074279 | 1.061318 | diagnostic_only_exact_key |
| tp4_dp2_ep8 | isl12000_osl2000_batch128 | 0.993809 | 4.198380 | 4.224534 | 1.006230 | 1.140246 | diagnostic_only_exact_key |

## Interpretation

high-budget steady ratio is large while clean effect stays near 1.
global constant is No-Go because topology and shape behavior is not uniform.
The next step can only be a diagnostic_only_exact_key candidate, not Default AIC.
