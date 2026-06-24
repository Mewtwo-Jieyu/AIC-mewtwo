# Phase344 Route A Source Schema Audit

| Item | Decision |
|---|---|
| Default AIC | No-Go |
| GPU allowed | false |
| Diagnostic only | true |
| Valid for default | false |
| PerfDatabase | false |

vLLM 0.19.0 source/schema is not cleared. Phase344 only audits the local source and schema boundary for Route A; it does not run GPU, benchmark, SSH, default AIC, or PerfDatabase writes.

PerfDatabase vLLM MoE query does not use kernel_source as a key. The local loader reads kernel_source, but ordinary MoE data is still split only into default versus low-latency buckets; the vLLM query path keys by quant, distribution, topk, experts, hidden, inter, moe_tp, moe_ep, and num_tokens.

Phase344 simultaneously anchors the PerfDatabase MoE query schema and the operations.py vLLM MoEDispatch comm branch.

H200 vLLM local perf table version | 0.12.0 only. The existing local attention, GEMM, MoE, and custom allreduce tables therefore remain a legacy assumption for any vLLM 0.19.0 Route A path.

EP8 communication remains schema-pending: current local comm logic is custom_allreduce and NCCL all_gather/alltoall/reduce_scatter modeling, not a cleared vLLM EP8 alltoall measured schema.

full trace must stay validation-only. It must not feed module latency back into PerfDatabase rows.

| Gate | Decision | Blocking condition | Next allowed phase |
|---|---|---|---|
| vllm_0190_moe_kernel_api | blocked_pending_vllm_0190_source_check | vllm_0190_moe_kernel_api_not_locally_verified | phase345_blocked_until_source_schema_clears |
| perfdb_moe_kernel_source_schema | blocked_perfdb_schema_missing_kernel_source_key | perfdb_vllm_moe_query_does_not_key_kernel_source | phase345_blocked_until_perfdb_schema_update_or_source_check |
| vllm_ep8_comm_scope | blocked_ep8_comm_schema_pending_source_check | current_comm_model_is_custom_allreduce_nccl_not_vllm_ep8_alltoall_schema | phase345_blocked_until_ep8_comm_schema_clears |
| legacy_attention_gemm_reuse_assumption | assumption_pending_validation | h200_vllm_local_perf_table_is_0120_only | phase345_blocked_until_0190_attention_gemm_source_check |
| cb_sim_token_shape_fidelity_input | retained_as_prerun_input_dependency | token_shape_fidelity_must_be_verified_before_full_validation | phase345_source_check_only_before_gpu_smoke |
| full_trace_feedback_guard | rejected_data_leakage | full_trace_must_not_generate_perfdb_rows | not_allowed |

Conclusion: Route A is retained as the next modeling route, but Phase345 GPU smoke is blocked until the vLLM 0.19.0 MoE kernel API, EP8 comm schema, and PerfDatabase query key contract are source-cleared.
