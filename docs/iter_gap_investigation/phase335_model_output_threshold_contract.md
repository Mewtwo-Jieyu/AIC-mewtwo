# Phase335 Model Output And Threshold Contract

| Item | Decision |
|---|---|
| Candidate | topology_specific_cadence_boundary_candidate |
| Default AIC | No-Go |
| GPU run allowed | false |
| Runtime integration | Not allowed in this phase |
| PerfDatabase | Not written |

The direction output is limited to holdout_faster / holdout_slower / inconclusive.

No posthoc feature is allowed. The feature set is limited to Phase333 registered inputs: actual_scheduled_tokens, phase_mix, boundary_cadence, trace_integrity, and worker_payload_alignment.

The numeric threshold remains blocked_until_numeric_model_form_exists because there is no numeric prediction formula yet. The threshold_type is explicitly `blocking_precondition` — this is not a pass threshold and must not be consumed as validation success by downstream parsers.

| Contract | Model output | Threshold status | Threshold type |
|---|---|---|---|
| direction_prediction_contract | holdout_faster_or_holdout_slower_or_inconclusive | direction_only_contract_registered | active_contract |
| cadence_boundary_feature_contract | cadence_boundary_direction_features | feature_set_registered_no_posthoc_features | active_contract |
| numeric_error_threshold_contract | numeric_error_threshold | blocked_until_numeric_model_form_exists | blocking_precondition |

| Flag | Value |
|---|---|
| diagnostic_only | true |
| valid_for_default | false |
| perf_database | false |
