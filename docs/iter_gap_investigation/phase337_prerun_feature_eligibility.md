# Phase337 Pre-Run Feature Eligibility

| Item | Decision |
|---|---|
| Default AIC | No-Go |
| Runtime integration | Not allowed in this phase |
| PerfDatabase | Not written |

The scope keys only define exact-key coverage. They are not standalone throughput predictors.

post-run trace features cannot be default prediction inputs. They are diagnostic audit evidence only.

| Feature | Availability | Default prediction | Leakage risk |
|---|---|---|---|
| topology_key | pre_run_available | false | false |
| shape_key | pre_run_available | false | false |
| control_bt_holdout_bt | pre_run_available | false | false |
| actual_scheduled_tokens | post_run_trace_derived | false | true |
| phase_mix_boundary_cadence | post_run_trace_derived | false | true |

| Flag | Value |
|---|---|
| diagnostic_only | true |
| valid_for_default | false |
| perf_database | false |
