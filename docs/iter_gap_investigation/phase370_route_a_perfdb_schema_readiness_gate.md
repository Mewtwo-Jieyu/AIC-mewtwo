# Phase370 Route A PerfDatabase Schema Readiness Gate

| Item | Result |
|---|---|
| Verdict | ready for Phase371 PerfDatabase schema design spec |
| PerfDatabase write | not ready for PerfDatabase write |
| Default AIC | No-Go |
| PerfDatabase | not written |

## Gate Result

- Phase366 EP8 comm 7-bucket result is accepted as a schema design input.
- Phase369 FusedMoE runner 7-bucket result is accepted as a schema design input.
- Shared buckets are exactly `1/15/16/241/1808/2048/8192`.
- Bucket `128` remains smoke-only and is excluded from schema readiness.
- Kernel source `CompressedTensorsWNA16MarlinMoEMethod:Marlin` remains metadata only, not a lookup key.
- Phase371 may write a PerfDatabase schema design spec.
- This phase still cannot write PerfDatabase rows, fit curves, interpolate, extrapolate, or enable default AIC.

## Rows

| row_type | verdict |
|---|---|
| phase366_ep8_comm_shape_sweep_passed | schema_design_input_accepted |
| phase369_fusedmoe_runner_shape_sweep_passed | schema_design_input_accepted |
| shared_bucket_alignment_passed | shared_bucket_set_aligned |
| smoke_bucket_128_excluded | smoke_bucket_excluded_from_schema_readiness |
| kernel_source_metadata_only | metadata_only_not_lookup_key |
| perfdb_schema_design_ready | ready_for_phase371_schema_design_spec |
| perfdb_write_blocked | blocked_no_perfdb_row_or_curve |
| default_aic_blocked | blocked_default_aic_no_go |
