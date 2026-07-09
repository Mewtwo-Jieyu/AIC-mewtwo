# Phase453 admission headroom

## Verdict

- `admission_headroom_supported`: 24/24 victim 用 recompute-from-zero 恢复；释放后 chunk-fit 已够，但 p50 场外等待 47.41s，reschedule 与完成计数跳变对齐率 100.0%。
- 修复处方: cb_sim waiting admission 增加 vLLM 同口径 full-sequence/headroom gate；不加系数。
- 修后复验: Phase451D 显示 sim preemption 明显下降，但最终 runtime fix gate 仍未过。
- validate 复验: dp2-8k2k 收到 1.14x；六点最大仍为 tp8-8k2k 1.44x。
- Default AIC: No-Go; this phase only justifies the next simulator fix.

## Key Rows

| section | metric | value | target | status | note |
|---|---|---:|---|---|---|
| wait_pattern | scenario | K2.5-tp4ep8dp2-8k2k |  |  |  |
| wait_pattern | victim_pairs | 24 | 24 |  |  |
| wait_pattern | recompute_from_zero | 24 | 24 | pass |  |
| wait_pattern | chunk_fit_after_free | 24 | 24 | pass | sim chunk-fit gate would consider these victims schedulable |
| wait_pattern | wait_p50_s | 47.413822 |  |  |  |
| wait_pattern | wait_p90_s | 81.264014 |  |  |  |
| wait_pattern | wait_min_s | 0.958761 |  |  |  |
| wait_pattern | wait_max_s | 87.720675 |  |  |  |
| wait_pattern | metrics_poll_interval_s | 2.00962 |  |  |  |
| wait_pattern | completion_aligned_share | 1 | >=0.80 | pass |  |
| wait_pattern | immediate_reentry_within_one_poll | 2 |  |  |  |
| wait_pattern | free_after_free_p50_blocks | 554.5 |  |  |  |
| wait_pattern | free_after_alloc_p50_blocks | 1238.5 |  |  |  |
| wait_pattern | full_headroom_after_alloc_p50_blocks | 684 |  |  |  |
| source_gap | vllm_waiting_gate | scheduler_reserve_full_isl + can_fit_full_sequence | vllm/v1/core/sched/scheduler.py:733-744 | source_located | waiting request breaks before allocate_slots when full sequence cannot fit |
| source_gap | vllm_full_sequence_blocks | full_num_tokens -> get_num_blocks_to_allocate | vllm/v1/core/kv_cache_manager.py:218-254 | source_located | docstring says this prevents over-admitting chunked prefill |
| source_gap | cbsim_current_gate | scheduled chunk only | cb_simulator/scheduler.py:208-216 | gap | cb_sim appends chunk then checks current scheduled blocks only |
| verdict | verdict | admission_headroom_supported | chunk-fit enough, but real reentry waits for completion wave | pass |  |
| post_fix | sim_preemption_events_after_fix | 192 | ~24 real observed victims | partial | Phase451D forensics rerun after full-sequence admission gate |
| post_fix | sim_throughput_tok_s_gpu_after_fix | 155.535661 |  |  |  |
| post_fix | remaining_thrash_repeat_victim | 120 |  |  |  |
| post_fix | remaining_decode_growth_pressure | 66 |  |  |  |
| post_fix | phase451e_runtime_fix_gate | blocked_pending_unique_semantic_fix | declared metrics pass | blocked | headroom gate removes one cause but does not close all declared metrics |
| validate | dp2_8k2k_error_ratio | 1.136279 | <=1.15 | pass | post-headroom default validate multi-config |
| validate | multi_config_max_error_ratio | 1.438569 | <=1.50 default gate; <=1.15 target criterion | pass_default_open_15pct | max_scenario=K2.5-tp8ep8-8k2k |
| validate | multi_config_mean_error_ratio | 1.191294 |  |  |  |

## Source Boundary

- vLLM source has the full-sequence admission gate in the waiting path.
- The local vLLM source tree does not expose the `SchedulerConfig` field definition; serve.log also does not print the field. This report therefore treats the Phase452 wait pattern as the runtime confirmation.
- Diagnostic data is not ingested into PerfDB.
