# Phase347 Route A Schema Source Reconciliation

| Item | Decision |
|---|---|
| Default AIC | No-Go |
| GPU allowed | false |
| Diagnostic only | true |
| Valid for default | false |
| PerfDatabase | false |

Phase346 cleared only the remote vLLM baseline: H worker vLLM is 0.19.0 and the source root was locatable. That is not enough for a single-point GPU smoke.

This is not a GPU smoke spec. The next step is contract/spec work: MoE measurement API contract, kernel_source schema contract, and vLLM EP8 comm schema.

MoE measurement API contract must decide whether Route A measures FusedMoE or bare fused_experts, and must define tensor shapes, weights, router logits, dtype, quant, and kernel path before any GPU.

kernel_source schema contract is still blocked because one logical MoE query may map to multiple vLLM kernel paths. PerfDatabase cannot accept that ambiguity as a default model key.

vLLM EP8 comm schema is still blocked because vLLM exposes multiple all2all backends; WideEP schema must not be reused.

| Gate | Phase346 fact | Decision | Next required action |
|---|---|---|---|
| remote_vllm_version_source_root | remote_vllm_version_0_19_0_source_root_cleared | cleared_baseline_only | use_as_source_check_baseline_only |
| moe_measurement_api_contract | kimi_moe_calls_fusedmoe_default_runner_and_torch_ops_moe_forward | blocked_define_fusedmoe_or_fused_experts_api | define_measurement_api_tensor_shapes_weights_router_quant |
| kernel_source_key_contract | multiple_moe_kernel_paths_possible_under_same_logical_query | blocked_add_or_lock_kernel_source | add_kernel_source_to_perfdb_key_or_lock_single_kernel_path |
| vllm_ep8_comm_schema_contract | vllm_ep8_comm_has_multiple_all2all_backends_not_wideep_schema | blocked_define_vllm_ep8_comm_scope | define_vllm_ep8_comm_backend_and_measured_schema |
| attention_gemm_reuse_policy | local_h200_vllm_perf_tables_are_0_12_0_only | blocked_assumption_pending_0190_validation | validate_0190_attention_gemm_kernel_drift_before_reuse |
| token_shape_fidelity_input | token_shape_remains_prerun_input_only | retained_prerun_only_no_trace_feedback | keep_cb_sim_token_shape_as_gate_without_trace_backfill |
| gpu_smoke_design | single_point_gpu_smoke_not_ready | blocked_until_contracts_clear | write_schema_source_reconciliation_before_any_gpu_smoke_spec |

Conclusion: GPU smoke design remains blocked until the MoE measurement API, kernel_source key contract, and vLLM EP8 comm schema clear first.
