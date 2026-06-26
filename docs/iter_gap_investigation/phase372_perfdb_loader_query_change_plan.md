# Phase372 PerfDatabase Loader Query Change Plan

| Item | Result |
|---|---|
| Verdict | plan-only loader/query change design |
| Default AIC | No-Go |
| PerfDatabase | not written |
| Loader/query implementation | not implemented |

## Plan Boundary

- Phase372 depends on the Phase371 module-level schema design spec.
- This is plan-only; it does not implement loader/query behavior and does not change runtime behavior.
- The design is to add vLLM module-level lookup design and it does not replace the existing TRT-LLM path.
- Allowed module boundaries are `fusedmoe_runner_compute;ep8_comm_dispatch_combine`.
- Key dimensions remain `model;hardware;vllm_version;topology;bucket_tokens;module_boundary;quant_runtime`.
- The kernel source remains metadata only and is not a lookup key.
- Allowed buckets are `1/15/16/241/1808/2048/8192`; bucket `128` remains excluded.
- This phase does not write PerfDatabase rows, fit curves, interpolate, extrapolate, or enable default AIC.
- Phase373 may design the loader/query implementation spec or minimal implementation plan.

## Rows

| row_type | decision |
|---|---|
| phase371_schema_spec_prerequisite | require_phase371_module_schema_spec |
| change_scope_plan_only | plan_only_no_loader_query_implementation |
| vllm_module_lookup_boundary | add_vllm_module_level_lookup_design_not_trtllm_replacement |
| allowed_module_boundaries | lock_phase371_module_boundaries |
| key_dimensions_contract | reuse_phase371_module_key_dimensions |
| kernel_source_metadata_policy | kernel_source_metadata_only_not_lookup_key |
| bucket_whitelist | allow_only_phase124_real_buckets |
| prohibited_actions | no_write_no_fit_no_default_in_phase372 |
| next_phase | phase373_may_design_loader_query_implementation_spec |
