# Phase451-D preemption forensics

结论: 真实侧按 EngineCore iteration 行复核后 mixed/request 约 1.04,且 recompute 仍为 0。当前 sim 的抢占由 repeat-victim thrash 主导;D 门通过,但 E 门仍阻塞,因为还没有唯一且已验证会改善指标的源码语义修复。

- classification gate: `pass`
- runtime fix gate: `blocked` (blocked_pending_unique_semantic_fix)
- Report-only: this analyzer does not change scheduler, PerfDB, validate gates, or reference data.
- Default AIC remains `No-Go`.

## Summary

| section | side | metric | value | target | status | note |
|---|---|---|---:|---|---|---|
| real_tp_dedupe | real | mixed_share | 0.022518 | 0.0243±0.0050 | pass | mixed_steps=530; total_steps=23537; source=serve.log iteration lines |
| real_tp_dedupe | real | engine_0_mixed_steps_per_request | 1.039216 | ~1.0 | pass | mixed_steps=265; requests=255; EngineCore iteration lines |
| real_tp_dedupe | real | engine_0_recomputed_tokens_per_request | 0 | 0 | pass |  |
| real_tp_dedupe | real | engine_1_mixed_steps_per_request | 1.039216 | ~1.0 | pass | mixed_steps=265; requests=255; EngineCore iteration lines |
| real_tp_dedupe | real | engine_1_recomputed_tokens_per_request | 0 | 0 | pass |  |
| sim_forensics | sim | preemption_events | 824 |  |  | throughput_tok_s_gpu=108.743492 |
| sim_forensics | sim | classification_coverage | 1 | >=0.80 | pass | minimal source-backed candidate set covers the target share |
| sim_forensics | sim | minimal_candidate_set_coverage | 0.90534 | >=0.80 | pass | coverage of the minimal candidate set used by the D gate |
| sim_forensics | sim | candidate_semantic_diff_count | 1 | <=3 for D gate; E requires a proven semantic fix | pass | preempt_reentry_thrash |
| decision | sim | phase451d_classification_gate | True | coverage>=0.80 and candidates<=3 | pass |  |
| decision | sim | phase451e_runtime_fix_gate | blocked_pending_unique_semantic_fix | unique source-backed fix that reduces recompute/preemption | blocked | dominant signature isolated, but no unique source-backed runtime change has been proven to reduce the declared metrics |
| sim_category | sim | admission_induced | 6 |  |  | share=0.007282 |
| sim_category | sim | decode_growth_pressure | 66 |  |  | share=0.080097 |
| sim_category | sim | prefill_growth_pressure | 6 |  |  | share=0.007282 |
| sim_category | sim | thrash_repeat_victim | 746 |  |  | share=0.905340 |
| candidate | sim | block_growth_or_watermark | 66 |  |  | source-backed signature candidate; not a fix by itself |
| candidate | sim | preempt_reentry_thrash | 746 |  |  | source-backed signature candidate; not a fix by itself |
| candidate | sim | prefill_chunk_allocation | 6 |  |  | source-backed signature candidate; not a fix by itself |
| candidate | sim | waiting_admission_gate | 6 |  |  | source-backed signature candidate; not a fix by itself |
| source_semantics | vllm | preempt_trigger | running_allocate_slots_none |  |  | vllm/v1/core/sched/scheduler.py:460-508; preemption occurs while scheduling RUNNING requests |
| source_semantics | vllm | waiting_after_preempt | skip_waiting_when_preempted |  |  | vllm/v1/core/sched/scheduler.py:563-564; WAITING scheduling is skipped if preempted_reqs is non-empty |
| source_semantics | vllm | preempt_reentry | free_kv_reset_computed_prepend_waiting |  |  | vllm/v1/core/sched/scheduler.py:956-971 |
| source_semantics | vllm | waiting_admission_allocation | allocate_slots_for_new_chunk_or_skip |  |  | vllm/v1/core/sched/scheduler.py:575-700 and kv_cache_manager.py:327-334 |
| source_semantics | sim | preempt_reentry | prefill_remaining_isl_plus_generated_insert_waiting_head |  |  | src/aiconfigurator/sdk/backends/cb_simulator/scheduler.py:76-93 |
| source_semantics | sim | waiting_admission | non_preemptive_fits_check |  |  | src/aiconfigurator/sdk/backends/cb_simulator/scheduler.py:197-217 |

## Sample Preemption Events

| category | replica | iter | trigger | victim | victim_preemptions_before | blocks_over | running | waiting |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| admission_induced | 0 | 60 | DECODING | 112 | 0 | 1 | 57 | 7 |
| admission_induced | 0 | 2194 | DECODING | 202 | 0 | 1 | 55 | 9 |
| admission_induced | 0 | 3610 | DECODING | 204 | 0 | 1 | 48 | 16 |
| admission_induced | 1 | 60 | DECODING | 113 | 0 | 1 | 57 | 7 |
| admission_induced | 1 | 2194 | DECODING | 203 | 0 | 1 | 55 | 9 |
| admission_induced | 1 | 3610 | DECODING | 205 | 0 | 1 | 48 | 16 |
| decode_growth_pressure | 0 | 61 | DECODING | 114 | 0 | 1 | 57 | 7 |
| decode_growth_pressure | 0 | 203 | DECODING | 110 | 0 | 1 | 56 | 8 |
| decode_growth_pressure | 0 | 352 | DECODING | 108 | 0 | 1 | 55 | 9 |
| decode_growth_pressure | 0 | 505 | DECODING | 106 | 0 | 1 | 54 | 10 |
| decode_growth_pressure | 0 | 665 | DECODING | 104 | 0 | 1 | 53 | 11 |
| decode_growth_pressure | 0 | 831 | DECODING | 102 | 0 | 1 | 52 | 12 |
| decode_growth_pressure | 0 | 1003 | DECODING | 100 | 0 | 1 | 51 | 13 |
| decode_growth_pressure | 0 | 1182 | DECODING | 98 | 0 | 1 | 50 | 14 |
| decode_growth_pressure | 0 | 1369 | DECODING | 96 | 0 | 1 | 49 | 15 |
| decode_growth_pressure | 0 | 1563 | DECODING | 94 | 0 | 1 | 48 | 16 |
| decode_growth_pressure | 0 | 1766 | DECODING | 92 | 0 | 1 | 47 | 17 |
| decode_growth_pressure | 0 | 1977 | DECODING | 90 | 0 | 1 | 46 | 18 |
| decode_growth_pressure | 0 | 4186 | DECODING | 310 | 0 | 1 | 56 | 8 |
| decode_growth_pressure | 0 | 4334 | DECODING | 308 | 0 | 1 | 55 | 9 |
| decode_growth_pressure | 0 | 4487 | DECODING | 306 | 0 | 1 | 54 | 10 |
| decode_growth_pressure | 0 | 4646 | DECODING | 304 | 0 | 1 | 53 | 11 |
| decode_growth_pressure | 0 | 5008 | DECODING | 302 | 0 | 1 | 51 | 13 |
| decode_growth_pressure | 0 | 5187 | DECODING | 300 | 0 | 1 | 50 | 14 |
| decode_growth_pressure | 0 | 5374 | DECODING | 298 | 0 | 1 | 49 | 15 |
| decode_growth_pressure | 0 | 5568 | DECODING | 296 | 0 | 1 | 48 | 16 |
| decode_growth_pressure | 0 | 5990 | DECODING | 294 | 0 | 1 | 46 | 18 |
| decode_growth_pressure | 0 | 6103 | DECODING | 406 | 0 | 1 | 56 | 8 |
| decode_growth_pressure | 0 | 6104 | DECODING | 408 | 0 | 1 | 56 | 8 |
| decode_growth_pressure | 0 | 6400 | DECODING | 402 | 0 | 1 | 54 | 10 |
| decode_growth_pressure | 0 | 7566 | DECODING | 400 | 0 | 1 | 48 | 16 |
| decode_growth_pressure | 0 | 8352 | DECODING | 510 | 0 | 1 | 55 | 0 |
| decode_growth_pressure | 0 | 8506 | DECODING | 508 | 0 | 1 | 54 | 1 |
| decode_growth_pressure | 0 | 8665 | DECODING | 506 | 0 | 1 | 53 | 2 |
| decode_growth_pressure | 0 | 8830 | DECODING | 504 | 0 | 1 | 52 | 3 |
| decode_growth_pressure | 0 | 9002 | DECODING | 502 | 0 | 1 | 51 | 4 |
| decode_growth_pressure | 0 | 9387 | DECODING | 500 | 0 | 1 | 49 | 5 |
| decode_growth_pressure | 0 | 9581 | DECODING | 498 | 0 | 1 | 48 | 6 |
| decode_growth_pressure | 0 | 10002 | DECODING | 496 | 0 | 1 | 46 | 7 |
| decode_growth_pressure | 1 | 61 | DECODING | 115 | 0 | 1 | 57 | 7 |
| decode_growth_pressure | 1 | 203 | DECODING | 111 | 0 | 1 | 56 | 8 |
| decode_growth_pressure | 1 | 352 | DECODING | 109 | 0 | 1 | 55 | 9 |
| decode_growth_pressure | 1 | 505 | DECODING | 107 | 0 | 1 | 54 | 10 |
| decode_growth_pressure | 1 | 665 | DECODING | 105 | 0 | 1 | 53 | 11 |
| decode_growth_pressure | 1 | 831 | DECODING | 103 | 0 | 1 | 52 | 12 |
| decode_growth_pressure | 1 | 1003 | DECODING | 101 | 0 | 1 | 51 | 13 |
| decode_growth_pressure | 1 | 1182 | DECODING | 99 | 0 | 1 | 50 | 14 |
| decode_growth_pressure | 1 | 1369 | DECODING | 97 | 0 | 1 | 49 | 15 |
| decode_growth_pressure | 1 | 1563 | DECODING | 95 | 0 | 1 | 48 | 16 |
| decode_growth_pressure | 1 | 1766 | DECODING | 93 | 0 | 1 | 47 | 17 |
| decode_growth_pressure | 1 | 1977 | DECODING | 91 | 0 | 1 | 46 | 18 |
| decode_growth_pressure | 1 | 4186 | DECODING | 311 | 0 | 1 | 56 | 8 |
| decode_growth_pressure | 1 | 4334 | DECODING | 309 | 0 | 1 | 55 | 9 |
| decode_growth_pressure | 1 | 4487 | DECODING | 307 | 0 | 1 | 54 | 10 |
| decode_growth_pressure | 1 | 4646 | DECODING | 305 | 0 | 1 | 53 | 11 |
| decode_growth_pressure | 1 | 5008 | DECODING | 303 | 0 | 1 | 51 | 13 |
| decode_growth_pressure | 1 | 5187 | DECODING | 301 | 0 | 1 | 50 | 14 |
| decode_growth_pressure | 1 | 5374 | DECODING | 299 | 0 | 1 | 49 | 15 |
| decode_growth_pressure | 1 | 5568 | DECODING | 297 | 0 | 1 | 48 | 16 |
| decode_growth_pressure | 1 | 5990 | DECODING | 295 | 0 | 1 | 46 | 18 |
| decode_growth_pressure | 1 | 6103 | DECODING | 407 | 0 | 1 | 56 | 8 |
| decode_growth_pressure | 1 | 6104 | DECODING | 409 | 0 | 1 | 56 | 8 |
| decode_growth_pressure | 1 | 6400 | DECODING | 403 | 0 | 1 | 54 | 10 |
| decode_growth_pressure | 1 | 7566 | DECODING | 401 | 0 | 1 | 48 | 16 |
| decode_growth_pressure | 1 | 8352 | DECODING | 511 | 0 | 1 | 55 | 0 |
| decode_growth_pressure | 1 | 8506 | DECODING | 509 | 0 | 1 | 54 | 1 |
| decode_growth_pressure | 1 | 8665 | DECODING | 507 | 0 | 1 | 53 | 2 |
| decode_growth_pressure | 1 | 8830 | DECODING | 505 | 0 | 1 | 52 | 3 |
| decode_growth_pressure | 1 | 9002 | DECODING | 503 | 0 | 1 | 51 | 4 |
| decode_growth_pressure | 1 | 9387 | DECODING | 501 | 0 | 1 | 49 | 5 |
| decode_growth_pressure | 1 | 9581 | DECODING | 499 | 0 | 1 | 48 | 6 |
| decode_growth_pressure | 1 | 10002 | DECODING | 497 | 0 | 1 | 46 | 7 |
| prefill_growth_pressure | 0 | 2058 | PREFILLING | 200 | 0 | 169 | 56 | 8 |
| prefill_growth_pressure | 0 | 4069 | PREFILLING | 312 | 0 | 95 | 57 | 7 |
| prefill_growth_pressure | 0 | 6071 | PREFILLING | 404 | 0 | 16 | 56 | 8 |
| prefill_growth_pressure | 1 | 2058 | PREFILLING | 201 | 0 | 169 | 56 | 8 |
| prefill_growth_pressure | 1 | 4069 | PREFILLING | 313 | 0 | 95 | 57 | 7 |
| prefill_growth_pressure | 1 | 6071 | PREFILLING | 405 | 0 | 16 | 56 | 8 |
| thrash_repeat_victim | 0 | 206 | DECODING | 114 | 1 | 1 | 56 | 8 |
| thrash_repeat_victim | 0 | 207 | DECODING | 110 | 1 | 1 | 56 | 8 |
