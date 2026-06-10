# Phase177: Budget Mechanism Holdout Plan

## Decision

| Item | Result |
|---|---|
| Scope | Holdout design only |
| Current candidate | Diagnostic-only exact-key mechanism candidate |
| Default AIC | No-Go |
| PerfDatabase | No-Go |
| GPU benchmark | No-Go in Phase177 |
| Model formula change | No-Go |

Phase171 showed that removing the high-budget steady-state time penalty moves cb_sim much closer to the Phase164 clean measurements. Phase177 does not treat that as a model fix. It defines the next clean holdout gate needed before any diagnostic model item expansion.

## Input Evidence

| Input | Required shape |
|---|---|
| `phase164_clean_gpu_budget_manifest.csv` | Exactly 4 clean benchmark rows |
| `phase171_budget_mechanism_candidate.csv` | Exactly 2 mechanism candidate rows |
| `phase172_budget_mechanism_candidate_query.md` | Exact-key, diagnostic-only query boundary |

All inputs remain diagnostic evidence. They are not default cb_sim rows and not PerfDatabase rows.

## Mechanism Hypothesis

cb_sim appears to over-penalize scheduler steady-state time when `max_num_batched_tokens=65536`. The holdout should test whether the same mechanism holds when the input shape changes, not only on the Phase164 `isl8000_osl2000_batch128` points.

| Topology | Phase171 observation |
|---|---|
| `tp8_dp1_ep8` | depenalized effect is close to clean effect, but slightly high |
| `tp4_dp2_ep8` | depenalized effect is closer than raw sim, but still low |

The exact Phase165 gap remains an upper bound only. It must not be multiplied into the default model.

## Holdout Matrix

Run pairs, not isolated rows. Each high-budget row must have a same-topology same-shape control row with non-65536 `max_bt`.

| scenario | topology_key | shape_key | max_bt | role |
|---|---|---|---:|---|
| `tp8ep8-4k2k-bt4000` | `tp8_dp1_ep8` | `isl4000_osl2000_batch128` | 4000 | control |
| `tp8ep8-4k2k-bt65536` | `tp8_dp1_ep8` | `isl4000_osl2000_batch128` | 65536 | holdout |
| `tp4dp2ep8-4k2k-bt4000` | `tp4_dp2_ep8` | `isl4000_osl2000_batch128` | 4000 | control |
| `tp4dp2ep8-4k2k-bt65536` | `tp4_dp2_ep8` | `isl4000_osl2000_batch128` | 65536 | holdout |
| `tp8ep8-12k2k-bt12000` | `tp8_dp1_ep8` | `isl12000_osl2000_batch128` | 12000 | control |
| `tp8ep8-12k2k-bt65536` | `tp8_dp1_ep8` | `isl12000_osl2000_batch128` | 65536 | holdout |
| `tp4dp2ep8-12k2k-bt12000` | `tp4_dp2_ep8` | `isl12000_osl2000_batch128` | 12000 | control |
| `tp4dp2ep8-12k2k-bt65536` | `tp4_dp2_ep8` | `isl12000_osl2000_batch128` | 65536 | holdout |

This matrix keeps the Phase164 topology pair, changes `isl`, and includes non-65536 controls. If runtime budget is limited, run one topology-complete shape pair first, then stop and review before expanding.

## Required Fields

| Field | Use |
|---|---|
| `topology_key` | Join and candidate key |
| `shape_key` | Exact shape boundary |
| `max_num_batched_tokens` | Budget boundary |
| `real_output_tok_s_gpu` | Clean measured output-only throughput |
| `real_total_tok_s_gpu` | Traffic sanity metric, not model input by default |
| `request_success_count` | Traffic health |
| `request_fail_count` | Traffic health |
| `sim_output_tok_s_gpu` | cb_sim output-only comparison |
| `baseline_steady_state_time_ms` | Control scheduler time |
| `budget_steady_state_time_ms` | High-budget scheduler time |
| `steady_state_time_ratio` | Mechanism variable |
| `depenalized_budget_effect` | Diagnostic estimate only |
| `diagnostic_only` | Must be `true` |
| `valid_for_default` | Must be `false` |
| `perf_database` | Must be `false` |

## Holdout Calculations

For each same-topology same-shape pair:

| Metric | Formula |
|---|---|
| `clean_budget_effect` | `clean_high_budget_output / clean_control_output` |
| `sim_budget_effect` | `sim_high_budget_output / sim_control_output` |
| `steady_state_time_ratio` | `budget_steady_state_time_ms / baseline_steady_state_time_ms` |
| `depenalized_budget_effect` | `sim_budget_effect * steady_state_time_ratio` |
| `raw_error_ratio` | `sim_budget_effect / clean_budget_effect` |
| `depenalized_error_ratio` | `depenalized_budget_effect / clean_budget_effect` |

Use output-only throughput for model comparison. Benchmark latency and throughput fields may prove traffic health, but they do not become timing evidence for default cb_sim.

## Go / No-Go

| Gate | Go |
|---|---|
| Clean rows | Every holdout row has successful requests and diagnostic flags |
| Pairing | Every high-budget row has a same-topology same-shape control row |
| Mechanism direction | `depenalized_error_ratio` is closer to 1.0 than `raw_error_ratio` for both topologies |
| Cross-shape check | The improvement appears on more than one `shape_key` |
| Residual shape | Residual error does not require a single naked ms constant |
| Boundary | No default AIC, no PerfDatabase, no interpolation, no extrapolation |

## Blockers

| Condition | Result |
|---|---|
| Depenalized only works on Phase164 `8k2k` | Do not expand candidate |
| One topology improves and the other worsens | Stop and split topology mechanism design |
| Non-65536 control is missing | Reject the holdout artifact |
| Any row has failed requests | Reject the holdout artifact |
| Flags are not diagnostic-only | Reject the holdout artifact |
| A runner or parser needs schema changes | Open a separate phase before benchmark |

## Next Phase

Phase178 may implement runner/preflight for this holdout matrix. It must still be diagnostic-only. Phase177 does not authorize GPU runs, default model changes, PerfDatabase writes, PR creation, or interpolation/extrapolation.

diagnostic_only=true valid_for_default=false perf_database=false

## Phase178 Entry

Phase178 may add only runner/preflight and manifest guards for this matrix. It must not run GPU benchmark, change default AIC, write PerfDatabase rows, or change `VLLMBackend.run_agg`.
