# Phase385 FusedMoE Paired Bucket GPU Data Spec

Phase385 is a spec-only phase. It defines future FusedMoE paired bucket GPU data collection, but does not SSH, run GPU, write `vllm_module_perf.txt`, or open Default AIC.

| Gate | Decision |
|---|---|
| target buckets | 2/30/32/482/3616/4096/16384 |
| measurement boundary | FusedMoE.forward() runner level |
| hardware | h200_sxm |
| vLLM version | 0.19.0 |
| topology | tp4dp2ep8 |
| quant runtime | CompressedTensorsWNA16MarlinMoEMethod |
| EP8 | Do not collect EP8 again |
| existing rows | preserve current 14 rows |
| bucket 128 | blocked |
| lookup policy | exact-only; no interpolation, extrapolation, or nearest lookup |
| Default AIC | No-Go |

Do not run GPU in Phase385. The next allowed phase is `phase386_fusedmoe_paired_bucket_gpu_run_decision`.
