# Phase351 Route A kernel_source Key Contract

| Item | Decision |
|---|---|
| Default AIC | No-Go |
| GPU allowed | false |
| Diagnostic only | true |
| Valid for default | false |
| PerfDatabase | false |

Phase351 only defines the Route A kernel_source key contract. It does not update PerfDatabase loader or query code, and it does not run GPU.

There are two legal routes before any measurement row can become eligible: include kernel_source in the key, or lock exactly one kernel path with a source-check guard.

The ambiguous old vLLM MoE key is rejected. A logical key made only from quant, distribution, topk, experts, hidden, inter, moe_tp, moe_ep, and num_tokens is not enough when multiple kernel paths can produce different latency.

The schema update route is only a candidate. A separate phase must change loader/query tests before any PerfDatabase row can be written.

| Contract row | Key contract | Decision | Next required action |
|---|---|---|---|
| kernel_source_identity | include_kernel_source_in_route_a_moe_measurement_identity | required_key_dimension | define_kernel_source_value_set_and_capture_point |
| logical_moe_key_without_kernel_source | reject_quant_distribution_topk_experts_hidden_inter_moe_tp_moe_ep_num_tokens_without_kernel_source | rejected_ambiguous_identity | do_not_reuse_old_vllm_moe_key_without_kernel_source |
| lock_single_kernel_path_option | lock_exactly_one_kernel_source_with_source_check_guard | allowed_only_with_source_check_guard | write_source_check_guard_for_locked_kernel_path |
| perfdb_schema_update_option | add_kernel_source_to_loader_query_and_measurement_row_identity | candidate_requires_loader_query_contract_change | open_separate_phase_for_perfdb_loader_query_schema |
| measurement_row_eligibility | measurement_row_requires_kernel_source_identity_before_perfdb | blocked_until_kernel_source_identity_defined | choose_kernel_source_key_or_locked_path_before_row_is_eligible |
| gpu_smoke_readiness | gpu_smoke_requires_kernel_source_contract_before_run | blocked_until_kernel_source_contract_clears | choose_schema_update_or_locked_kernel_path_before_gpu_smoke |

Conclusion: GPU smoke and PerfDatabase writes remain blocked until the kernel_source identity is either added to the key contract or locked to a single source-checked path.
