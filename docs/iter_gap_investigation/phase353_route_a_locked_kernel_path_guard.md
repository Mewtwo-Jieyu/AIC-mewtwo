# Phase353 Route A Locked Kernel Path Guard

| Item | Decision |
|---|---|
| Default AIC | No-Go |
| GPU allowed | false |
| Diagnostic only | true |
| Valid for default | false |
| PerfDatabase | false |

Phase353 selects locked kernel path guard first. It does not SSH, does not run GPU, does not change runtime, and does not update PerfDatabase.

PerfDatabase schema update is deferred until the measurement value set exists. Without a source-checked kernel name, adding schema fields would only move ambiguity into the table.

The required next step is a source-check guard that proves exactly one vLLM MoE kernel path. If the guard cannot prove that, fail fast and do not run GPU.

Any later smoke is diagnostic-only until a separate phase updates PerfDatabase loader/query contracts and tests.

| Guard row | Guard contract | Decision | Next required action |
|---|---|---|---|
| route_choice | choose_locked_kernel_path_guard_before_schema_update | locked_kernel_path_guard_first | write_h_source_check_for_single_kernel_path |
| perfdb_schema_update | do_not_change_perfdb_schema_in_phase353 | deferred_until_measurement_value_set_exists | defer_schema_update_until_source_checked_value_exists |
| kernel_source_value_set | source_check_must_name_exact_kernel_path | blocked_pending_source_checked_kernel_name | collect_source_checked_kernel_name_without_gpu |
| source_check_guard | prove_exactly_one_kernel_path_before_any_gpu_smoke | required_before_any_gpu_smoke | fail_if_multiple_kernel_paths_or_missing_anchor |
| guard_failure_policy | fail_fast_without_gpu_when_guard_fails | fail_fast_no_gpu_run | stop_and_reconcile_schema_before_gpu |
| measurement_row_eligibility | locked_path_smoke_is_diagnostic_not_perfdb_row | diagnostic_smoke_only_not_perfdb | keep_measurement_rows_out_of_perfdb |
| gpu_smoke_readiness | gpu_smoke_requires_locked_guard_to_clear_first | blocked_until_locked_guard_clears | run_h_source_check_only_after_guard_spec_is_committed |

Conclusion: GPU smoke remains blocked until the locked kernel path source-check guard clears.
