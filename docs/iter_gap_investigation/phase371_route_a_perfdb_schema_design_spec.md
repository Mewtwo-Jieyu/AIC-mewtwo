# Phase371 Route A PerfDatabase Schema Design Spec

| Item | Result |
|---|---|
| Verdict | module-level row only schema design spec |
| Default AIC | No-Go |
| PerfDatabase | not written |
| Loader/query change | not changed |

## Schema Boundary

- The row boundary is module-level row only, not an end-to-end row.
- Required module rows are `fusedmoe_runner_compute` and `ep8_comm_dispatch_combine`.
- Key dimensions are `model;hardware;vllm_version;topology;bucket_tokens;module_boundary;quant_runtime`.
- The kernel source remains metadata only and is not a lookup key.
- Allowed buckets are `1/15/16/241/1808/2048/8192`; bucket `128` remains excluded.
- This phase does not write PerfDatabase rows, fit curves, interpolate, extrapolate, or enable default AIC.
- Phase372 may design the loader/query change plan, but Phase371 does not change loader/query behavior.

## Rows

| row_type | decision |
|---|---|
| schema_scope | module_level_row_only |
| module_row_fusedmoe_runner_compute | define_fusedmoe_runner_compute_module_row |
| module_row_ep8_comm_dispatch_combine | define_ep8_comm_dispatch_combine_module_row |
| key_dimensions | lock_minimal_module_perf_key_dimensions |
| kernel_source_policy | kernel_source_metadata_only_not_lookup_key |
| bucket_policy | allow_only_phase124_real_buckets |
| prohibited_actions | no_write_no_fit_no_default_in_phase371 |
| next_phase | phase372_may_design_loader_query_change_plan |
