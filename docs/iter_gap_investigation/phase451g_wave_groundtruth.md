# Phase451-G wave ground truth

结论: `scheduler_dynamics_internal`; wave_model_gate=`blocked_pending_scheduler_mechanism`。本报告只做 G1-G3 取证,不改 runtime/PerfDB/gate。

- Default AIC remains `No-Go`.

## Summary

| section | metric | value | target | status | note |
|---|---|---:|---|---|---|
| g1_groundtruth | bench_records | 512 | all ok records |  |  |
| g1_groundtruth | max_concurrency | 128 |  |  |  |
| g1_groundtruth | reconstructed_wall_error | 0.000006 | <=0.02 | pass | reconstructed=754.860s; bench=754.864s |
| g1_groundtruth | wave_count | 4 |  |  |  |
| g1_groundtruth | start_gap_p50_ms | 35.972776 |  |  |  |
| g1_groundtruth | start_gap_p90_ms | 1134 |  |  |  |
| g1_wave | wave_0_start_span_ms | 0 |  |  | count=128 |
| g1_wave | wave_0_finish_span_ms | 164498 |  |  | growth_vs_wave0=1 |
| g1_wave | wave_1_start_span_ms | 157725 |  |  | count=128 |
| g1_wave | wave_1_finish_span_ms | 162721 |  |  | growth_vs_wave0=0.9892 |
| g1_wave | wave_2_start_span_ms | 155997 |  |  | count=128 |
| g1_wave | wave_2_finish_span_ms | 161503 |  |  | growth_vs_wave0=0.981797 |
| g1_wave | wave_3_start_span_ms | 161503 |  |  | count=128 |
| g1_wave | wave_3_finish_span_ms | 138453 |  |  | growth_vs_wave0=0.841673 |
| g2_puzzle | bench_ok_request_count | 512 |  |  |  |
| g2_puzzle | metrics_success_count_delta | 510 |  |  | metrics polling can miss already-finished requests at the first sample |
| g2_puzzle | num_preemptions_total | 108 | explain 54-style counter |  | vllm/v1/core/sched/scheduler.py:957-973 increments request.num_preemptions |
| g2_puzzle | engine_0_num_preemptions_total | 54 | 54-style per-engine counter |  |  |
| g2_puzzle | engine_1_num_preemptions_total | 54 | 54-style per-engine counter |  |  |
| g2_puzzle | preemptions_per_request | 0.210938 |  |  |  |
| g2_puzzle | prompt_tokens_recomputed_total | 0 | not full preemption recompute |  | vllm/v1/metrics/stats.py:249-296 defines recomputed as cached-token accounting |
| g2_puzzle | prompt_tokens_total_per_request | 8000 | ~8000 if no full prompt replay is visible in counters |  |  |
| g2_puzzle | prefill_kv_computed_tokens_per_request | 7969 | ~8000 |  |  |
| g2_puzzle | mixed_steps_per_request | 1.035156 | ~1.04 |  | mixed_steps=530 |
| g2_puzzle | serve_preemption_log_lines | 0 | victim context unavailable if 0 | blocked |  |
| g2_puzzle | puzzle_resolution | partial_counter_semantics_only |  | blocked | preemption counter and recomputed-token counter measure different things; serve.log lacks victim context, so recovery path remains unproven |
| g3_replay | closed_loop_preemptions | 824 |  |  |  |
| g3_replay | closed_loop_mixed_share | 0.06326 |  |  |  |
| g3_replay | real_arrival_replay_preemptions | 824 |  |  |  |
| g3_replay | real_arrival_replay_preemptions_per_request | 1.609375 |  |  |  |
| g3_replay | real_arrival_replay_mixed_share | 0.06326 |  |  |  |
| g3_replay | real_arrival_replay_completed | 512 | 512 | pass | completed_by_replica={0: 256, 1: 256} |
| g3_replay | responsibility | scheduler_dynamics_internal |  | pass | real reconstructed arrivals still preserve most thrash or mixed-share error |
| g4_decision | wave_model_gate | blocked_pending_scheduler_mechanism |  | blocked |  |
| g4_decision | runtime_patch | not_applied |  | blocked | Phase451-G is report-only until a non-parametric mechanism passes declared gates |
| g4_decision | default_aic | No-Go |  | blocked |  |

## Boundary

- `bench_records` has latency only; start times are reconstructed from the benchmark worker queue semantics.
- `prompt_tokens_recomputed_total` is cached-token accounting, not a direct full-prompt preemption recovery counter.
- `serve.log` has no victim-level preemption records in this run, so victim recovery path remains unproven.
- No runtime patch is applied in this phase.
