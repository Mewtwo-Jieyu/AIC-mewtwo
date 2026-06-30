# Phase386 FusedMoE Paired Bucket GPU Run Decision Spec

Phase386 is a GPU run decision spec. This is not GPU result evidence. It does not SSH, run GPU, write real data, or open Default AIC.

| Gate | Decision |
|---|---|
| decision type | GPU run decision spec |
| target buckets | 2/30/32/482/3616/4096/16384 |
| measurement boundary | FusedMoE.forward() runner level |
| hardware | h200_sxm |
| vLLM version | 0.19.0 |
| topology | tp4dp2ep8 |
| quant runtime | CompressedTensorsWNA16MarlinMoEMethod |
| stop rules | ptx_or_compat_error;forward_context_error;quant_runtime_mismatch;shape_or_output_non_finite;gpu_or_process_residue;missing_bucket |
| Phase386 GPU/SSH | blocked |
| Default AIC | No-Go |

The next allowed phase is `phase387_fusedmoe_paired_bucket_gpu_run`.
