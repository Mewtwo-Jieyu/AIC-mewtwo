# Phase384 Paired Bucket Expansion Spec

Phase384 is a paired bucket expansion spec. It does not run GPU, change runtime code, write real PerfDatabase data, or open Default AIC.

| Gate | Decision |
|---|---|
| existing vLLM module rows | preserve 14 current rows |
| EP8 buckets | 1/15/16/241/1808/2048/8192 |
| future FusedMoE paired buckets | 2/30/32/482/3616/4096/16384 |
| planned new rows | 7 |
| future row count after later materialization | 21 |
| bucket 128 | blocked |
| lookup policy | exact-only; no interpolation, extrapolation, or nearest lookup |
| Default AIC | No-Go |

Do not write vllm_module_perf.txt in Phase384. The next allowed phase is `phase385_fusedmoe_paired_bucket_gpu_data_spec`.
