# Phase451 preemption ledger

结论: 真实 run 的 mixed 步不是 preemption/recompute 主驱动。真实每 engine preemption counter 有增量,但 `prompt_tokens_recomputed_total=0`,且每请求 computed KV 仍是 8000。当前 sim 的 preemption/recompute 反而明显高于真实,451-3 不能按“真实大量重预填”假设直接修。

| section | side | engine | metric | value | target | status | note |
|---|---|---:|---|---:|---|---|---|
| mixed_profile | real | all | mixed_share | 0.029321 | 0.0243±0.0050 | informational | mixed_steps=2760; total_steps=94132 |
| mixed_profile | sim | all | mixed_share | 0.06326 | 0.0243±0.0050 | fail | mixed_steps=1488; total_steps=23522 |
| preemption_ledger | real | 0 | request_success_length | 255 |  |  | delta from metrics counter |
| preemption_ledger | real | 0 | mixed_steps_per_request | 5.411765 |  |  | mixed_steps=1380 |
| preemption_ledger | real | 0 | preemptions_per_request | 0.211765 |  |  | preemptions=54.0 |
| preemption_ledger | real | 0 | prompt_tokens_recomputed_per_request | 0 |  | pass | prompt_tokens_recomputed_total=0.0 |
| preemption_ledger | real | 0 | prefill_kv_tokens_per_request | 8000 | 8000 | pass | prompt_tokens_total=2048000.0; kv_count=255.0 |
| preemption_gate | real | 0 | preemption_drives_mixed_steps | 0.03913 | >=0.80 and recomputed_tokens>0 | fail | mixed steps are not mostly preemption/recompute driven |
| preemption_ledger | real | 1 | request_success_length | 255 |  |  | delta from metrics counter |
| preemption_ledger | real | 1 | mixed_steps_per_request | 5.411765 |  |  | mixed_steps=1380 |
| preemption_ledger | real | 1 | preemptions_per_request | 0.211765 |  |  | preemptions=54.0 |
| preemption_ledger | real | 1 | prompt_tokens_recomputed_per_request | 0 |  | pass | prompt_tokens_recomputed_total=0.0 |
| preemption_ledger | real | 1 | prefill_kv_tokens_per_request | 8000 | 8000 | pass | prompt_tokens_total=2048000.0; kv_count=255.0 |
| preemption_gate | real | 1 | preemption_drives_mixed_steps | 0.03913 | >=0.80 and recomputed_tokens>0 | fail | mixed steps are not mostly preemption/recompute driven |
| preemption_ledger | sim | all | preemptions_per_request | 1.609375 |  |  | preemptions=824; request_count=512 |
| preemption_ledger | sim | all | recompute_tokens_per_request | 14497 |  |  | recompute_tokens=7422568 |
| preemption_ledger | sim | all | preempted_prefill_steps_per_request | 1.609375 |  |  | preempted_prefill_steps=824 |
| preemption_ledger | sim | 0 | mixed_steps_per_request | 2.90625 |  |  | mixed_steps=744; expected_requests_per_engine=256 |
| preemption_ledger | sim | 1 | mixed_steps_per_request | 2.90625 |  |  | mixed_steps=744; expected_requests_per_engine=256 |
| verdict | real | all | real_preemption_explains_1380_mixed_steps | false | true | fail | real_preemptions=108; real_recomputed_tokens=0; mixed_steps=2760 |
| verdict | sim | all | sim_preemption_over_real_preemption | 7.62963 | near 1.0 | fail | sim preemption/recompute is excessive relative to real counters; a runtime fix must target sim over-preemption, not assume real re-prefill |
| decision | all | all | phase451_3_runtime_fix_gate | blocked | only after exact semantic diff is isolated | blocked | report-only ledger falsifies the original real re-prefill premise |
| source_audit | vllm | all | running_preemption_trigger | allocate_slots_none |  |  | vllm/v1/core/sched/scheduler.py:460-508 preempts only when RUNNING allocation fails |
| source_audit | vllm | all | waiting_admission_guard | skip_waiting_if_preempted |  |  | vllm/v1/core/sched/scheduler.py:563-564 schedules WAITING only when no preemption happened |
| source_audit | vllm | all | preempt_recompute_state | num_computed_tokens_reset |  |  | vllm/v1/core/sched/scheduler.py:956-971 frees KV, status=PREEMPTED, num_computed_tokens=0 |
| source_audit | vllm | all | kv_allocate_failure | return_none_if_free_blocks_insufficient |  |  | vllm/v1/core/kv_cache_manager.py:327-334 describes allocation stages; allocate_slots returns None on insufficient free blocks |
| source_audit | sim | all | preempt_recompute_state | prefill_tokens_remaining=isl+generated |  |  | src/.../cb_simulator/scheduler.py:76-93 requeues PREEMPTED at waiting head |
| source_audit | sim | all | waiting_admission_guard | non_preemptive_admission |  |  | src/.../cb_simulator/scheduler.py:197-217 admits only if _fits_block_capacity |

## Boundary

- Report-only: no runtime, PerfDB, validate gate, or reference data changed.
- The real side uses Phase446 B2b 8k2k event/metrics from the same run.
- Default AIC remains No-Go until the full table meets the agreed gate.
