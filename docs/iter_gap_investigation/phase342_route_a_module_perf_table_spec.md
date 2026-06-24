# Phase342 Route A Module Perf Table Spec

| Item | Decision |
|---|---|
| Route A | vLLM 0.19.0 module-level perf table |
| Route B | absorbed as token-shape input layer |
| Full trace feedback | rejected_data_leakage |
| Default AIC | No-Go |
| GPU allowed | false |
| PerfDatabase | false |

Route A becomes the primary diagnostic model candidate: _compute_3pass -> run_static -> PerfDatabase query -> compose. This phase only specifies the route and gates; it is not a default model.

Pre-run inputs: model_arch_dims, hardware, topology, scheduled_tokens, phase, quant, distribution, moe_tp, moe_ep. Output: iteration_latency_ms.

MoE key: num_tokens, hidden, inter, topk, experts, moe_tp, moe_ep, quant, distribution, is_context.

Initial validation gate: mean error <=20%, max error <=30%, direction must be correct. Parameters must be fixed on tp8; tp4dp2 is holdout.

No-Go gates: unstable vLLM 0.19.0 kernel source schema, missing standalone MoE microbench, unclear EP8 comm scope, insufficient cb_sim token-shape fidelity, or any need to backfill module latency from full trace.

| Candidate | Role | Decision | Blocking condition | Next phase |
|---|---|---|---|---|
| route_a_module_perf_table_candidate | primary_model_candidate | retained_as_primary_candidate | module_perf_table_assumptions_pending_source_check | phase343_vllm_0190_moe_source_api_check |
| route_b_runtime_shape_input_layer | token_shape_source | absorbed_as_token_shape_source | cb_sim_token_shape_fidelity_must_be_sufficient | phase343_source_check_before_any_gpu |
| legacy_attention_gemm_reuse | legacy_assumption | assumption_pending_validation | vllm_0190_kernel_source_must_match_legacy_table_scope | phase343_kernel_source_schema_check |
| vllm_ep8_comm_modeling | comm_schema_candidate | schema_pending_source_check | vllm_ep8_comm_scope_must_be_defined_before_gpu | phase343_ep8_comm_source_schema_check |
| full_trace_feedback_to_perf_table | forbidden_feedback_path | rejected_data_leakage | data_leakage_from_full_trace_feedback | not_allowed |

| Flag | Value |
|---|---|
| diagnostic_only | true |
| valid_for_default | false |
| perf_database | false |
