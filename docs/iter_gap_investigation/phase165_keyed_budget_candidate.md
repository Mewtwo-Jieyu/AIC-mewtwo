# Phase165: Clean cb_sim Budget Candidate

## Decision

| Item | Result |
|---|---|
| Scope | Diagnostic keyed model item design only |
| Source | phase164_clean_gpu_benchmark |
| Default AIC | No-Go |
| PerfDatabase | No-Go |
| GPU benchmark | No-Go in Phase165 |

## Observed Budget Effects

| topology_key | shape_key | clean_budget_effect | sim_budget_effect | budget_gap |
|---|---|---:|---:|---:|
| tp8_dp1_ep8 | isl8000_osl2000_batch128 | 0.987382 | 0.283937 | 3.477467 |
| tp4_dp2_ep8 | isl8000_osl2000_batch128 | 1.167497 | 0.161350 | 7.235797 |

## Candidate Key

Use `topology_key + shape_key + max_num_batched_tokens` as the diagnostic key.

| Field | Value |
|---|---|
| source | phase164_clean_gpu_benchmark |
| diagnostic_only | true |
| valid_for_default | false |
| perf_database | false |

## Boundary

This candidate is a design artifact. It is not a default cb_sim formula change, not a PerfDatabase row, and not a claim that benchmark metrics are globally portable across other models or hardware.

Next gate: Phase166 may implement a callable diagnostic-only model item. Default AIC integration still requires a separate holdout gate.

diagnostic_only=true valid_for_default=false perf_database=false
