# Phase369 FusedMoE Runner Shape Sweep Result

| Item | Result |
|---|---|
| Verdict | Phase368 FusedMoE runner minimal shape sweep passed |
| Default AIC | No-Go |
| PerfDatabase | not written |
| Evidence status | diagnostic-only FusedMoE runner shape sweep |

## Result

- Phase368 artifact: `/mnt/shared-storage-user/ailab-sys/zhaojieyu/aic/phase368_fusedmoe_runner_minimal_shape_sweep_6e45508`
- Measurement boundary is `FusedMoE.forward()` runner level.
- Buckets are `1`, `15`, `16`, `241`, `1808`, `2048`, `8192`.
- `128` is not included; it remains a smoke-only point.
- Runtime quant method was `CompressedTensorsWNA16MarlinMoEMethod` for every row.
- Kernel source metadata was `CompressedTensorsWNA16MarlinMoEMethod:Marlin`; it is not a lookup key.
- Output shape matched `<bucket>x7168` and all outputs were finite.
- Cleanup was `true` and GPU/process residue was `false`.
- These latencies are diagnostic only and cannot be interpolated or extrapolated into a PerfDatabase curve.
- They are not default AIC evidence.

## Bucket Results

| bucket_tokens | latency_ms_median | output_shape | output_all_finite |
|---:|---:|---|---|
| 1 | 0.187888 | 1x7168 | true |
| 15 | 0.242813 | 15x7168 | true |
| 16 | 0.239764 | 16x7168 | true |
| 241 | 1.889204 | 241x7168 | true |
| 1808 | 4.729947 | 1808x7168 | true |
| 2048 | 5.344776 | 2048x7168 | true |
| 8192 | 20.612520 | 8192x7168 | true |
