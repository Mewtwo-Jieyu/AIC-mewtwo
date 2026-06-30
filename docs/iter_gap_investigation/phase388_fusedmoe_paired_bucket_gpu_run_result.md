# Phase388 FusedMoE Paired Bucket GPU Run Result

| Item | Result |
|---|---|
| Verdict | Phase387 FusedMoE paired bucket GPU run passed |
| Default AIC | No-Go |
| PerfDatabase | not written |
| vllm_module_perf.txt | not written |
| Evidence status | diagnostic-only FusedMoE.forward paired bucket GPU run |

## Result

- Phase387 artifact: `/mnt/shared-storage-user/ailab-sys/zhaojieyu/aic/phase387_fusedmoe_paired_bucket_gpu_run_2ee80b7`
- Measurement boundary is `FusedMoE.forward()` runner level.
- Buckets are `2`, `30`, `32`, `482`, `3616`, `4096`, `16384`.
- `128` is not included.
- Runtime quant method was `CompressedTensorsWNA16MarlinMoEMethod` for every row.
- Kernel source metadata was `CompressedTensorsWNA16MarlinMoEMethod:Marlin`; it is not a lookup key.
- Output shape matched `<bucket>x7168` and all outputs were finite.
- Cleanup was `true` and GPU/process residue was `false`.
- Phase388 does not write `vllm_module_perf.txt`, write PerfDatabase rows, or open Default AIC.

## Bucket Results

| bucket_tokens | latency_ms_median | output_shape | output_all_finite |
|---:|---:|---|---|
| 2 | 0.188482 | 2x7168 | true |
| 30 | 0.355168 | 30x7168 | true |
| 32 | 0.353423 | 32x7168 | true |
| 482 | 2.040215 | 482x7168 | true |
| 3616 | 9.283537 | 3616x7168 | true |
| 4096 | 10.359648 | 4096x7168 | true |
| 16384 | 41.146568 | 16384x7168 | true |
