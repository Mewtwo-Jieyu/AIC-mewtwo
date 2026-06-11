# Phase222: Budget Penalty Evidence Family

## Decision

| Item | Result |
|---|---|
| Default AIC | No-Go |
| VLLMBackend.run_agg | Unchanged |
| PerfDatabase | No-Go |
| GPU benchmark | No-Go in Phase222 |
| Recommended boundary | diagnostic_only_exact_key |
| Boundary flags | diagnostic_only=true valid_for_default=false perf_database=false |

## Evidence Rows

| topology_key | shape_key | control_bt | holdout_bt | clean_budget_effect | no_budget_penalty_error | evidence_source | clean_effect_spread |
|---|---|---:|---:|---:|---:|---|---:|
| tp8_dp1_ep8 | isl4000_osl2000_batch128 | 4000 | 65536 | 1.012212 | 1.012212 | phase215_budget_penalty_mechanism | 1.088007 |
| tp8_dp1_ep8 | isl8000_osl2000_batch128 | 8000 | 65536 | 0.987382 | 1.012779 | phase164_clean_gpu_budget_manifest | 1.088007 |
| tp8_dp1_ep8 | isl12000_osl2000_batch128 | 12000 | 65536 | 1.074279 | 1.074279 | phase215_budget_penalty_mechanism | 1.088007 |
| tp4_dp2_ep8 | isl4000_osl2000_batch128 | 4000 | 65536 | 1.133187 | 1.133187 | phase215_budget_penalty_mechanism | 1.174770 |
| tp4_dp2_ep8 | isl8000_osl2000_batch128 | 8000 | 65536 | 1.167497 | 1.167497 | phase164_clean_gpu_budget_manifest | 1.174770 |
| tp4_dp2_ep8 | isl12000_osl2000_batch128 | 12000 | 65536 | 0.993809 | 1.006230 | phase215_budget_penalty_mechanism | 1.174770 |

## Interpretation

Phase164 clean evidence supplies the 8k2k rows; those rows are not interpolation.
Phase215 supplies the 4k2k and 12k2k diagnostic mechanism evidence.
Across both topologies, high-budget clean effect stays close to 1 compared with the cb_sim steady penalty family.
Shape and topology spread remains visible, so Default AIC remains No-Go.
The only defensible next boundary is a diagnostic_only_exact_key family, not a global constant or a default multiplier.
