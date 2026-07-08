# Phase449 KV watermark profile

结论: `validate 8k2k DP2 capacity matches real log`。real KV=458128 tokens, validate KV=458128 tokens, ratio=1.000x。

| section | metric | engine | value | p10 | p50 | p90 | min | max | n | note |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| real_capacity | kv_cache_tokens | 0 | 458128 |  |  |  |  |  | 1 | num_gpu_blocks=28633; block_size=16 |
| real_capacity | kv_cache_tokens | 1 | 458128 |  |  |  |  |  | 1 | num_gpu_blocks=28633; block_size=16 |
| capacity_diff | validate_tokens_over_real_tokens | all | 1 |  |  |  | 458128 | 458128 |  | validate_blocks=28633; real_blocks=28633 |
| mixed_aligned | mixed_context_tokens | 0 | 7698 | 7948 | 7957 | 7974 | 45 | 7999 | 265 | nearest metrics poll within 3.0s |
| mixed_aligned | mixed_context_tokens | 1 | 7698 | 7948 | 7957 | 7974 | 45 | 7999 | 265 | nearest metrics poll within 3.0s |
| mixed_aligned | mixed_decode_batch | 0 | 40.6528 | 26.4000 | 42 | 51 | 1 | 56 | 265 | nearest metrics poll within 3.0s |
| mixed_aligned | mixed_decode_batch | 1 | 40.6528 | 26.4000 | 42 | 51 | 1 | 56 | 265 | nearest metrics poll within 3.0s |
| mixed_aligned | mixed_nearest_running | 0 | 42.6491 | 28.8000 | 44 | 53 | 3 | 56 | 265 | nearest metrics poll within 3.0s |
| mixed_aligned | mixed_nearest_running | 1 | 42.6491 | 28.8000 | 44 | 53 | 3 | 56 | 265 | nearest metrics poll within 3.0s |
| mixed_aligned | mixed_nearest_waiting | 0 | 21.0792 | 10 | 18 | 41.2000 | 0 | 67 | 265 | nearest metrics poll within 3.0s |
| mixed_aligned | mixed_nearest_waiting | 1 | 18.4981 | 9 | 18 | 29.2000 | 0 | 55 | 265 | nearest metrics poll within 3.0s |
| mixed_aligned | mixed_nearest_kv_usage | 0 | 0.7989 | 0.5034 | 0.8455 | 0.9623 | 0.0524 | 0.9977 | 265 | nearest metrics poll within 3.0s |
| mixed_aligned | mixed_nearest_kv_usage | 1 | 0.7989 | 0.5034 | 0.8455 | 0.9623 | 0.0524 | 0.9977 | 265 | nearest metrics poll within 3.0s |
| mixed_aligned | mixed_nearest_preemptions | 0 | 21.4302 | 0 | 21 | 43 | 0 | 43 | 265 | nearest metrics poll within 3.0s |
| mixed_aligned | mixed_nearest_preemptions | 1 | 21.4302 | 0 | 21 | 43 | 0 | 43 | 265 | nearest metrics poll within 3.0s |
| mixed_aligned | mixed_metric_delta_s | 0 | 0.5141 | 0.1048 | 0.5324 | 0.9139 | 0.0006 | 1.0006 | 265 | nearest metrics poll within 3.0s |
| mixed_aligned | mixed_metric_delta_s | 1 | 0.5156 | 0.1048 | 0.5391 | 0.9139 | 0.0006 | 1.0006 | 265 | nearest metrics poll within 3.0s |
| derived | kv_ceiling_full_8k2k_requests | all | 45.8128 |  |  |  |  |  |  | rough ceiling for full 8000+2000 token requests per engine |

## Scope

- `valid_for_default=false`; this is a waterline audit, not a readiness upgrade.
- `num_preemptions_total` is reported from the same metrics stream; no GPU collection was run.
