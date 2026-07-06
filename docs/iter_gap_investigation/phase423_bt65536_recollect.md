# Phase423 DP2 bt65536 Clean Recollect

## Verdict

Phase423 没有产出可替换的 benchmark reference。清洁协议在 vLLM KV cache 初始化阶段失败，API server 没有进入 ready，benchmark 没有开始。

| Item | Value |
|---|---:|
| scenario | K2.5-tp4ep8dp2-8k2k-bt65536 |
| max_model_len | 262144 |
| max_num_batched_tokens | 65536 |
| max_num_seqs | 256 |
| prefix_caching | false |
| prompt_variant_mode | rotating |
| needed_kv_cache_gib | 17.160000 |
| available_kv_cache_gib | 8.620000 |
| estimated_max_model_len | 131648 |
| collection_status | service_init_failed |

## Decision

| Decision | Value |
|---|---|
| capacity_arbitration | clean_protocol_cannot_start_at_max_model_len_262144 |
| mechanism_verdict | clean_recollect_blocked_by_kv_init_capacity |
| reference_action | no_reference_replacement |
| ab_action | not_run_no_bench_result |
| large_step_trace_action | not_run_service_not_ready |
| next_phase_target | phase424_reference_protocol_decision |

这次结果不能把旧 `retry5e` reference 直接替换掉，也不能执行 A/B 全表。它只证明：在 `max_model_len=262144`、`max_num_batched_tokens=65536`、`gpu_memory_utilization=0.8`、prefix off 的清洁协议下，DP2 bt65536 自然 profile 出来的可用 KV 内存不足以启动服务。

## Cleanup

| Check | Value |
|---|---|
| gpu_after_empty | true |
| process_after_empty | true |

## Boundary

本阶段只记录 diagnostic evidence：不写 PerfDatabase，不改 runtime 计费，不替换 validate reference，不打开 Default AIC。
