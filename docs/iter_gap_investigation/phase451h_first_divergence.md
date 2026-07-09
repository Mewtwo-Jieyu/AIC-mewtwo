# Phase451-H first divergence

结论: 首次分歧状态 `found`, 初步语义指向 `admission_or_capacity_gate`。分岔点是 axis=1:真实只 decode,sim 同步 admit 下一条 prefill。但真实日志没有 per-step waiting 队列和 victim request id,所以不能把它硬判成 scheduler 源码错误。

- Default AIC remains `No-Go`.
- Report-only: no scheduler/runtime/PerfDB/validate gate changes.

## Summary

| section | side | metric | value | target | status | note |
|---|---|---|---:|---|---|---|
| h1_real_ramp | real | real_trace_steps | 8192 |  |  |  |
| h1_real_ramp | real | real_mixed_share | 0.039307 | 0.0243±0.0050 |  |  |
| h1_real_ramp | real | metrics_preemptions_total | 108 | overlay only; no exact step timestamps |  |  |
| h1_real_ramp | real | serve_preemption_log_lines | 0 | >0 needed for victim context | blocked |  |
| h1_real_ramp | real | latency_p10_ms | 139377 |  |  |  |
| h1_real_ramp | real | latency_p50_ms | 161248 |  |  |  |
| h1_real_ramp | real | latency_p90_ms | 246114 |  |  |  |
| h1_real_ramp | real | immediate_refill_like_outliers | 31 |  | informational |  |
| h1_real_ramp | real | wait_for_completion_wave_like_outliers | 88 |  | informational |  |
| h1_real_ramp | real | victim_recovery_path_gate | blocked |  | blocked | bench_records has request latency but no victim id; recovery path remains suggestive, not proven |
| h2_sim_ramp | sim | sim_trace_steps | 8192 |  |  |  |
| h2_sim_ramp | sim | sim_preemptions | 824 |  |  |  |
| h2_sim_ramp | sim | sim_completed | 512 |  |  |  |
| h2_sim_ramp | sim | sim_mixed_share | 0.081299 | 0.0243±0.0050 |  |  |
| h3_divergence | compare | first_divergence_status | found |  | pass |  |
| h3_divergence | compare | first_divergence_axis | 1 |  |  |  |
| h3_divergence | compare | first_differing_fields | running,free_blocks,phase,prefill_reqs,prefill_tokens,total_tokens |  |  |  |
| h3_divergence | compare | semantic_hint | admission_or_capacity_gate |  | informational |  |
| h3_divergence | compare | first_divergence_observability_gate | blocked |  | blocked | real event rows have no per-step waiting queue; axis=1 may be engine-arrival visibility, not scheduler admission semantics |
| h3_source | vllm | waiting_admission | scheduler.py:575-700; kv_cache_manager.py:327-334 |  |  | source point to inspect only if first divergence is admission/capacity |
| h3_source | vllm | running_then_waiting_order | scheduler.py:530-700 |  |  | vLLM does attempt WAITING scheduling after RUNNING when token budget remains |
| h3_source | vllm | preemption_reentry | scheduler.py:956-971 |  |  | source point to inspect only if victim recovery is proven |
| h4_fix | decision | runtime_patch | not_applied |  | blocked | report-only H1-H3; current logs do not identify a single source-level fix |
| h4_fix | decision | default_aic | No-Go |  | blocked |  |

## First Divergence Row

| label | axis | replica | phase | running | decode | prefill_reqs | prefill_tokens | free_blocks | waiting | note |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| real_first_divergence | 1 | 0 | decode | 1 | 1 | 0 | 0 | 28132 | nan | real event rows lack request ids; free_blocks is ramp-only cumulative token ledger |
| sim_first_divergence | 1 | 0 | mixed_prefill | 2 | 1 | 1 | 7999 | 27632 | 63 |  |

## Sim Ramp Sample

| axis | replica | phase | running | waiting | decode | prefill_tokens | free_blocks |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 0 | prefill | 1 | 64 | 0 | 8000 | 28133 |
| 0 | 1 | prefill | 1 | 64 | 0 | 8000 | 28133 |
| 1 | 0 | mixed_prefill | 2 | 63 | 1 | 7999 | 27632 |
| 1 | 1 | mixed_prefill | 2 | 63 | 1 | 7999 | 27632 |
| 2 | 0 | mixed_prefill | 3 | 62 | 1 | 7999 | 27132 |
| 2 | 1 | mixed_prefill | 3 | 62 | 1 | 7999 | 27132 |
| 3 | 0 | mixed_prefill | 4 | 61 | 2 | 7998 | 26631 |
| 3 | 1 | mixed_prefill | 4 | 61 | 2 | 7998 | 26631 |
| 4 | 0 | mixed_prefill | 5 | 60 | 3 | 7997 | 26130 |
| 4 | 1 | mixed_prefill | 5 | 60 | 3 | 7997 | 26130 |
| 5 | 0 | mixed_prefill | 6 | 59 | 4 | 7996 | 25629 |
| 5 | 1 | mixed_prefill | 6 | 59 | 4 | 7996 | 25629 |
| 6 | 0 | mixed_prefill | 7 | 58 | 5 | 7995 | 25129 |
| 6 | 1 | mixed_prefill | 7 | 58 | 5 | 7995 | 25129 |
| 7 | 0 | mixed_prefill | 8 | 57 | 6 | 7994 | 24628 |
| 7 | 1 | mixed_prefill | 8 | 57 | 6 | 7994 | 24628 |
| 8 | 0 | mixed_prefill | 9 | 56 | 7 | 7993 | 24127 |
| 8 | 1 | mixed_prefill | 9 | 56 | 7 | 7993 | 24127 |
| 9 | 0 | mixed_prefill | 10 | 55 | 8 | 7992 | 23627 |
| 9 | 1 | mixed_prefill | 10 | 55 | 8 | 7992 | 23627 |

## Boundary

- Real event rows identify step composition, not request identity.
- Metrics expose total preemptions, not exact victim context or exact step timestamp.
- A runtime fix remains blocked until the first divergent decision can be mapped to one source-level semantic with victim recovery evidence.
