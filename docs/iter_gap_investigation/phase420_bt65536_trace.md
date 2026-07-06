# Phase420 BT65536 Trace

Verdict: `bt65536_budget_not_wired_plus_phase417_single_step_regression`.

Default AIC: `No-Go`.

Direct finding: both bt65536 validation points still run cb_sim with `configured_max_num_batched_tokens=8000`, so the scheduler admits `1/8` possible full-prefill requests. The 64k budget path has not actually been exercised yet.

## Result

| Scenario | Before error | After error | Capacity | First prefill reqs | Step ms before -> after | Cause |
|---|---:|---:|---:|---:|---:|---|
| K2.5-tp4ep8dp2-8k2k-bt65536 | 6.573003 | 7.524468 | 2.574400 | 1/8 (configured bt 8000) | 209.103539 -> 233.506826 | bt65536_budget_not_wired_plus_dirty_capacity |
| K2.5-tp8ep8-8k2k-bt65536 | 1.026471 | 1.190376 | 34.355200 | 1/8 (configured bt 8000) | 151.060832 -> 164.724770 | phase417_single_prefill_step_cost_regression_after_budget_not_wired |

## Query Path Delta

| Scenario | MoE before -> after | EP before -> after |
|---|---|---|
| K2.5-tp4ep8dp2-8k2k-bt65536 | phase397v_int4_wo_calibrated_sol -> moe_perf_lookup | fallback_tp_dp_collectives -> vllm_module_exact_lookup |
| K2.5-tp8ep8-8k2k-bt65536 | phase397v_int4_wo_calibrated_sol -> moe_perf_lookup | fallback_tp_dp_collectives -> vllm_module_exact_lookup |

## Phase421 Target

`phase421_wire_bt65536_budget_then_retrace_query_cost`.

Phase421 should first wire the scenario `max_num_batched_tokens=65536` into cb_sim validation. Only after that rerun this trace to decide whether large-step MoE/EP query cost still needs a code fix.

## Guardrails

- `diagnostic_only=true`.
- `valid_for_default=false`.
- `runtime_modified=false`.
- `perf_database=false`.
- `gpu_allowed=false`, `ssh_allowed=false`.
