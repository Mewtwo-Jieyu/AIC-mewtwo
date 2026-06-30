# Phase383 Runtime Bucket Semantics Decision Spec

Phase383 is a decision spec. It does not change runtime code, write PerfDatabase data, run GPU, or open Default AIC.

| Gate | Decision |
|---|---|
| current EP8 formula | max(1, raw_tokens//4) |
| current FusedMoE formula | max(1, raw_tokens//4)*2 |
| runtime semantics | Do not change runtime bucket semantics |
| selected route | paired_bucket_data_expansion_first |
| preserve existing rows | 14 current vLLM module rows stay in place |
| future FusedMoE paired buckets | 2/30/32/482/3616/4096/16384 |
| existing EP8 buckets | 1/15/16/241/1808/2048/8192 |
| bucket 128 | blocked |
| Default AIC | No-Go |

Changing the runtime formula would make FusedMoE lookup diverge from the real module shape, so the first legal fix is paired bucket data expansion.
The next allowed phase is `phase384_paired_bucket_expansion_spec`.
