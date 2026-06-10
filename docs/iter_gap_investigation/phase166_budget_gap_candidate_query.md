# Phase166: Budget Gap Candidate Query

## Decision

| Item | Result |
|---|---|
| Scope | Diagnostic-only exact-key query API |
| Input | `phase165_clean_cb_sim_gap.csv` |
| Default AIC | No-Go |
| PerfDatabase | No-Go |
| GPU benchmark | No-Go |
| Interpolation / extrapolation | No-Go |

Phase166 turns the two Phase165 clean budget gap rows into a callable query API. It does not change the cb_sim default model and does not write any performance database row.

## API

| Symbol | Contract |
|---|---|
| `VLLMCleanBudgetGapCandidate` | Frozen dataclass carrying one Phase165 candidate row |
| `clean_budget_gap_candidate_from_row(row)` | Parse and validate one CSV row |
| `load_clean_budget_gap_candidates(path)` | Load exactly the accepted two-row CSV |
| `get_clean_budget_gap_candidate(candidates, topology_key, shape_key, max_num_batched_tokens)` | Return exact-key candidate or raise `KeyError` |

## Accepted Keys

| candidate_key | Source |
|---|---|
| `tp8_dp1_ep8:isl8000_osl2000_batch128:max_bt65536` | Phase164 clean benchmark + Phase165 gap analysis |
| `tp4_dp2_ep8:isl8000_osl2000_batch128:max_bt65536` | Phase164 clean benchmark + Phase165 gap analysis |

The key is `topology_key + shape_key + max_num_batched_tokens`. Missing keys, unknown topology, or `max_bt` values other than the accepted exact key fail fast. There is no nearest match.

## Row Consistency Guard

| Check | Required |
|---|---|
| `candidate_key` | Must match `topology_key:shape_key:max_bt<budget_max_num_batched_tokens>` |
| `tp/dp/ep` | Must match the accepted key metadata |
| `isl/osl/batch_size` | Must be `8000/2000/128` for both accepted keys |
| `baseline_max_num_batched_tokens` | Must be `8000` |
| `budget_max_num_batched_tokens` | Must be `65536` |
| `clean_budget_effect/sim_budget_effect/budget_gap` | Must be positive |
| `budget_gap` | Must match `clean_budget_effect / sim_budget_effect` within CSV rounding tolerance |

The parser does not trust a row just because its key string is accepted. The key and all shape, topology, budget, and ratio fields must be self-consistent.

## Boundary Flags

| Field | Required |
|---|---|
| `diagnostic_only` | `true` |
| `valid_for_default` | `false` |
| `perf_database` | `false` |
| `source` | `phase164_clean_gpu_benchmark` |

Any mismatch rejects the candidate row.

## Next Gate

Phase167 can package the Phase164-166 files for review. A later phase may build a callable diagnostic-only model item around this API. Default cb_sim integration still needs a separate holdout gate.
