# Phase373 Loader Query Implementation Spec

| Item | Result |
|---|---|
| Verdict | implementation spec only |
| Default AIC | No-Go |
| PerfDatabase | not written |
| src change | not changed |

## Implementation Boundary

- Phase373 depends on the Phase372 loader/query change plan.
- The new data file design is `vllm_module_perf.txt`; it is not placed into `moe_perf.txt`.
- Future code entry points are `src/aiconfigurator/sdk/common.py` and `src/aiconfigurator/sdk/perf_database.py`.
- The new loader name is `load_vllm_module_data`, but Phase373 does not implement it.
- The new query name is `query_vllm_module`, but Phase373 does not implement it.
- The design does not replace `query_moe(...)` and does not change existing TRT-LLM/SGLang query paths.
- Key dimensions remain `model;hardware;vllm_version;topology;bucket_tokens;module_boundary;quant_runtime`.
- Allowed module boundaries are `fusedmoe_runner_compute;ep8_comm_dispatch_combine`.
- The kernel source remains metadata only and is not a lookup key.
- Allowed buckets are `1/15/16/241/1808/2048/8192`; bucket `128` remains excluded.
- This phase does not change `src/`, write PerfDatabase rows, fit curves, interpolate, extrapolate, or enable default AIC.
- Phase374 may do the minimal loader/query implementation.

## Rows

| row_type | decision |
|---|---|
| phase372_plan_prerequisite | require_phase372_loader_query_change_plan |
| data_file_contract | design_new_vllm_module_perf_file_not_moe_perf |
| future_code_entry_common | future_common_entry_only_no_src_change |
| future_code_entry_perf_database | future_perf_database_entry_only_no_src_change |
| loader_contract | design_load_vllm_module_data_without_implementation |
| query_contract | design_query_vllm_module_without_implementation |
| existing_path_guard | do_not_replace_query_moe_or_change_trtllm_sglang_paths |
| key_dimensions_contract | reuse_phase371_phase372_key_dimensions |
| module_boundary_contract | allow_only_phase371_module_boundaries |
| kernel_source_metadata_policy | kernel_source_metadata_only_not_lookup_key |
| bucket_whitelist | allow_only_phase124_real_buckets |
| prohibited_actions | no_src_change_no_perfdb_rows_no_fit_no_default |
| next_phase | phase374_may_do_minimal_loader_query_implementation |
