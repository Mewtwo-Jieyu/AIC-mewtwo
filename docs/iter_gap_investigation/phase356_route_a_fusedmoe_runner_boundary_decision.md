# Phase356 Route A FusedMoE Runner Boundary Decision

| Item | Decision |
|---|---|
| Default AIC | No-Go |
| GPU allowed | false |
| Diagnostic only | true |
| Valid for default | false |
| PerfDatabase | false |

Phase356 changes the Route A MoE measurement boundary to the FusedMoE.forward() runner level. It does not SSH, does not run GPU, does not change runtime, and does not write PerfDatabase rows.

kernel_source stays measurement metadata. It is not a PerfDatabase lookup key for this runner-level measurement decision.

Runtime dispatch is deterministic only inside a fixed model config, hardware, and vLLM version tuple. The next GPU step remains blocked until FusedMoE.forward() tensor shapes are defined.

## Decision Evidence (Phase355 Source Check Evidence)

| Field | Value |
|---|---|
| worker | worker-892rz |
| vllm_version | 0.19.0 |
| source_root | /usr/local/lib/python3.12/dist-packages/vllm |
| blocking_reasons | quant_method_runtime_selection;unquantized_backend_runtime_selection;shape_specific_fallback;multiple_expert_kernel_classes |
| candidate_kernel_paths | Triton;Cutlass;DeepGemm;FlashInfer;Marlin;TRTLLM |

These facts show that a source-level single kernel guard is not applicable. They do not block runner-level measurement because the measurement boundary is FusedMoE.forward() runner level.

The locked kernel path guard is not applicable for this boundary and is superseded by the runner-level contract.

| Field | Value |
|---|---|
| source | phase356_route_a_fusedmoe_runner_boundary_decision |
| worker | worker-892rz |
| vllm_version | 0.19.0 |
| source_root | /usr/local/lib/python3.12/dist-packages/vllm |
| measurement_boundary | fusedmoe_forward_runner_level |
| runner_boundary_api | FusedMoE.forward() |
| locked_kernel_path_guard_required | false |
| locked_kernel_path_guard_decision | superseded_not_applicable |
| kernel_source_lookup_key | false |
| kernel_source_metadata | true |
| runtime_dispatch_deterministic | true |
| runtime_dispatch_scope | fixed_model_config_hw_vllm_version_tuple |
| source_check_evidence | KimiMoE;FusedMoE;DefaultMoERunner;torch.ops.vllm.moe_forward |
| source_check_interpretation | single_kernel_guard_not_applicable_runner_measurement_retained |
| blocking_reasons | quant_method_runtime_selection;unquantized_backend_runtime_selection;shape_specific_fallback;multiple_expert_kernel_classes |
| candidate_kernel_paths | Triton;Cutlass;DeepGemm;FlashInfer;Marlin;TRTLLM |
| gpu_smoke_readiness | blocked_until_fusedmoe_forward_tensor_shapes_defined |
| next_allowed_phase | phase358_moe_tensor_shape_smoke_spec |
| gpu_allowed | false |
| default_readiness | No-Go |
| diagnostic_only | true |
| valid_for_default | false |
| perf_database | false |

Conclusion: Route A proceeds only as a diagnostic FusedMoE.forward() runner-level measurement spec. GPU smoke is still blocked until tensor shapes are defined.
