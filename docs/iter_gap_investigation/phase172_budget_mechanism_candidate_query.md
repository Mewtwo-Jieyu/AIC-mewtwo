# Phase172: Budget Mechanism Candidate Query

## Decision

| Item | Result |
|---|---|
| Scope | Diagnostic-only exact-key mechanism candidate query |
| Input | `phase171_budget_mechanism_candidate.csv` |
| Default AIC | No-Go |
| PerfDatabase | No-Go |
| GPU benchmark | No-Go |
| Interpolation / extrapolation | No-Go |

Phase172 turns the two Phase171 mechanism rows into a callable diagnostic API. It does not change the cb_sim default model and does not write performance database rows.

## API

| Symbol | Contract |
|---|---|
| `VLLMBudgetMechanismCandidate` | Frozen dataclass carrying one Phase171 mechanism candidate row |
| `load_budget_mechanism_candidates(path)` | Load exactly the accepted two-row CSV |
| `get_budget_mechanism_candidate(candidates, topology_key, shape_key, max_num_batched_tokens)` | Return exact-key candidate or raise `KeyError` |

## Accepted Keys

| candidate_key | Source |
|---|---|
| `tp8_dp1_ep8:isl8000_osl2000_batch128:max_bt65536` | Phase171 budget mechanism analysis |
| `tp4_dp2_ep8:isl8000_osl2000_batch128:max_bt65536` | Phase171 budget mechanism analysis |

Missing keys, unknown topology, unknown shape, or `max_bt` values other than `65536` fail fast. There is no nearest match.

## Row Guard

| Field | Required |
|---|---|
| `source` | `phase171_budget_mechanism_candidate` |
| `diagnostic_only` | `true` |
| `valid_for_default` | `false` |
| `perf_database` | `false` |
| `steady_state_time_ratio` | Must match `budget_steady_state_time_ms / baseline_steady_state_time_ms` |
| `depenalized_budget_effect` | Must match `sim_budget_effect * steady_state_time_ratio` |
| `raw_error_ratio` | Must match `sim_budget_effect / clean_budget_effect` |
| `depenalized_error_ratio` | Must match `depenalized_budget_effect / clean_budget_effect` |
| `depenalized_is_closer` | Must be `true` |

The parser does not trust the row just because its topology is accepted. The ratio fields must stay self-consistent with the mechanism calculation.

## Boundary

This API exposes a diagnostic mechanism candidate only. It is not a default cb_sim formula, not a PerfDatabase source, and not a rule for interpolation or extrapolation. Phase173 may package the Phase171-172 files for review; any default model integration still needs a separate holdout gate.

diagnostic_only=true valid_for_default=false perf_database=false
