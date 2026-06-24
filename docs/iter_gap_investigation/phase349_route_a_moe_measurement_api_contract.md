# Phase349 Route A MoE Measurement API Contract

| Item | Decision |
|---|---|
| Default AIC | No-Go |
| GPU allowed | false |
| Diagnostic only | true |
| Valid for default | false |
| PerfDatabase | false |

Phase349 only defines the Route A MoE measurement API contract. It does not run GPU, does not change runtime behavior, and does not write PerfDatabase rows.

The primary candidate boundary is the vLLM FusedMoE default runner boundary. A bare fused_experts is only an auxiliary kernel probe, not a PerfDatabase row, because it does not prove the default Route A module boundary used by vLLM.

The measurement output is latency_ms only. It is not a throughput prediction, not a global correction, and not default evidence.

GPU smoke remains blocked until tensor inputs, quant fields, and kernel_source identity are defined first.

| Contract row | Boundary | Decision | Required contract |
|---|---|---|---|
| primary_measurement_boundary | fusedmoe_default_runner | candidate_fusedmoe_default_runner_boundary | measure_the_default_vllm_fusedmoe_runner_boundary_before_perfdb_row |
| bare_fused_experts_probe | bare_fused_experts | auxiliary_kernel_probe_not_perfdb_row | only_use_bare_fused_experts_to_explain_kernel_behavior_not_default_model |
| required_input_tensors | fusedmoe_default_runner | blocked_until_hidden_states_router_weights_expert_weights_defined | hidden_states_router_weights_expert_weights_and_token_shapes |
| required_quant_fields | fusedmoe_default_runner | blocked_until_dtype_quant_method_kernel_source_defined | dtype_quant_method_kernel_source_and_distribution |
| kernel_source_capture | all_moe_measurement_rows | required_for_any_measurement_row | kernel_source_must_be_part_of_any_route_a_moe_measurement_identity |
| measurement_output | route_a_moe_measurement | latency_ms_only_no_default_prediction | output_is_latency_ms_not_throughput_prediction_or_default_correction |
| gpu_smoke_readiness | route_a_moe_smoke | blocked_until_api_contract_and_kernel_source_clear | gpu_smoke_requires_moe_api_contract_and_kernel_source_contract_first |

Conclusion: Route A MoE measurement is still diagnostic-only. The next step is the kernel_source key contract; EP8 comm schema is out of scope for this phase.
