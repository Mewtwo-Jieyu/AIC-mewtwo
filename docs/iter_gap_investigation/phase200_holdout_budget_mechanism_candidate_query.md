# Phase200: Holdout Budget Mechanism Candidate Query

## Decision

| Item | Result |
|---|---|
| Scope | diagnostic-only exact-query API |
| Default AIC | No-Go |
| VLLMBackend.run_agg | Unchanged |
| PerfDatabase | No-Go |
| GPU benchmark | No-Go |

## API

```python
from aiconfigurator.sdk.backends.cb_simulator import (
    VLLMHoldoutBudgetMechanismCandidate,
    get_holdout_budget_mechanism_candidate,
    holdout_budget_mechanism_candidate_from_row,
    load_holdout_budget_mechanism_candidates,
)
```

The accepted key is:

```text
{topology_key}:{shape_key}:control_bt{control_max_bt}:holdout_bt{holdout_max_bt}
```

## Accepted Keys

| candidate_key |
|---|
| tp8_dp1_ep8:isl4000_osl2000_batch128:control_bt4000:holdout_bt65536 |
| tp4_dp2_ep8:isl4000_osl2000_batch128:control_bt4000:holdout_bt65536 |
| tp8_dp1_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536 |
| tp4_dp2_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536 |

## Row Guards

| Field | Rule |
|---|---|
| source | Must be `phase199_holdout_budget_mechanism_analysis` |
| row count | Must be exactly 4 rows |
| candidate key | Must be one of the four accepted exact keys |
| duplicate key | Rejected |
| flags | `diagnostic_only=true valid_for_default=false perf_database=false` |
| token budgets | Positive, with `holdout_bt65536` only |
| throughput | Control and holdout output throughput must be positive |
| clean_high_over_control | Must equal holdout output / control output |
| clean_delta_from_1 | Must equal clean_high_over_control - 1 |
| steady_state_time_ratio | Must equal holdout steady-state time / control steady-state time |

## Boundary

This API is a diagnostic lookup wrapper around Phase199 holdout evidence. It does not interpolate, extrapolate, collapse rows into a global constant, write PerfDatabase rows, or affect the default cb_sim model path.
