# Phase451-I DEBUG ramp

结论: DEBUG 短跑确认 `engine_burst_confirmed`; victim 级观测 `not_available`。因此本轮只收报告,不做调度语义修复。

## Summary

| section | metric | value | target | status | note |
|---|---|---:|---|---|---|
| i3_gpu_run | bench_ok_requests | 128 | 128 | pass |  |
| i3_gpu_run | bench_failed_requests | 0 | 0 | pass |  |
| i3_gpu_run | output_tok_s | 1233 |  |  |  |
| i3_gpu_run | gpu_residual_after_empty | True |  | pass |  |
| i3_gpu_run | process_residual_after_empty | True |  | pass |  |
| i3_received_counts | counts_samples | 266 |  |  |  |
| i3_received_counts | max_waiting_per_engine | 63 | >=50 confirms burst | pass |  |
| i3_received_counts | max_waiting_global | 124 | >=100 confirms burst |  |  |
| i3_received_counts | max_running_per_engine | 57 |  |  |  |
| i3_received_counts | max_running_global | 114 |  |  |  |
| i3_received_counts | max_waiting_skew | 63 |  |  | early [[0,1],[63,1]] API view before both engines sync |
| i3_received_counts | first_nonzero_counts | [[0,1],[0,0]] |  |  |  |
| i3_received_counts | first_asymmetric_burst_counts | [[0,1],[63,1]] |  |  |  |
| i3_received_counts | first_symmetric_burst_counts | [[62,2],[62,2]] |  |  |  |
| i3_received_counts | engine_visibility_debug | engine_burst_confirmed | engine_burst_confirmed | pass |  |
| i3_iterations | iteration_rows | 8049 |  |  |  |
| i3_iterations | iteration_rows_dp0 | 4025 |  |  |  |
| i3_iterations | iteration_rows_dp1 | 4024 |  |  |  |
| i3_iterations | ctx_step_count | 132 |  |  |  |
| i3_iterations | mixed_step_count | 130 |  |  |  |
| i3_iterations | decode_step_count | 7913 |  |  |  |
| i3_iterations | mixed_decode_batch_p10 | 6 |  |  |  |
| i3_iterations | mixed_decode_batch_p50 | 32 |  |  |  |
| i3_iterations | mixed_decode_batch_p90 | 50 |  |  |  |
| i3_iterations | mixed_decode_batch_max | 56 |  |  |  |
| i3_iterations | decode_batch_p50 | 20 |  |  |  |
| i3_iterations | decode_batch_max | 56 |  |  |  |
| i3_observability | added_line_count | 0 | >0 would expose request arrival | blocked |  |
| i3_observability | preempt_line_count | 0 | >0 would expose preemption | blocked |  |
| i3_observability | victim_line_count | 0 | >0 would expose victim identity | blocked |  |
| i3_observability | victim_observability | not_available | available | blocked |  |
| i3_metrics | num_preemptions_engine0 | 12 | metric counter available |  |  |
| i3_metrics | num_preemptions_engine1 | 12 | metric counter available |  |  |
| i3_metrics | prompt_tokens_recomputed_engine0 | 0 | 0 means no recompute tokens observed |  |  |
| i3_metrics | prompt_tokens_recomputed_engine1 | 0 | 0 means no recompute tokens observed |  |  |
| i3_metrics | request_success_length_engine0 | 64 |  |  |  |
| i3_metrics | request_success_length_engine1 | 64 |  |  |  |
| i4_accept | runtime_patch | not_applied |  | blocked | DEBUG confirms burst but does not expose victim-level decision state |
| i4_accept | default_aic | No-Go |  | blocked |  |

## Boundary

- `Received counts` proves EngineCore-visible arrival is burst-scale, not client-side drizzle hidden by API/tokenization.
- DEBUG level exposes queue counts and per-step composition, but not victim identity or per-request preemption decision state in this run.
- No runtime, PerfDB, scheduler, or validate gate change is justified by this evidence alone; Default AIC remains No-Go.
