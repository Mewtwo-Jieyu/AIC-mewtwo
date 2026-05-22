# Phase 68 Available Descriptor Artifacts

## Conclusion

Phase68 found only one real vLLM scheduler/alignment descriptor artifact suitable for strict compare: the Phase61/62 `10k2k_b32 bt8192 tp4dp2ep8` capture. There are no existing real vLLM scheduler descriptor artifacts for `3k3k_b128`, `32k1k_b16`, different budget, or different topology holdouts.

| artifact | rows | scenario | usable for holdout |
|---|---:|---|---|
| `phase61_scheduler_descriptor_10k2k_b32/scheduler_descriptor_rows.csv` | `16012` | `10k2k_b32_bt8192` | no, raw TP-rank rows without alignment key |
| `phase61_scheduler_descriptor_10k2k_b32/scheduler_descriptor_rows_dedup_by_dp.csv` | `4003` | `10k2k_b32_bt8192` | no, pre-alignment descriptor only |
| `phase62_scheduler_alignment_10k2k_b32/scheduler_alignment_rows.csv` | `16012` | `10k2k_b32_bt8192` | no, raw TP-rank rows |
| `phase62_scheduler_alignment_10k2k_b32/scheduler_alignment_rows_dedup_by_alignment_dp.csv` | `4003` | `10k2k_b32_bt8192` | yes, but already used by Phase66/67 |
| `phase66_vllm_like_scheduler_descriptor.csv` | `4003` | `10000x2000_b32` | no, generated cb_sim-side descriptor |
| `phase67_vllm_like_scheduler_strict_compare.csv` | `4003` | `10000x2000_b32` | no, compare result |

## Non-usable Historical Data

Older artifacts mention `3k3k_b128`, `32k1k_b16`, and budget sweeps, but they are residual, latency, runtime-shape, or profiler/debug artifacts. They are not scheduler alignment descriptors and cannot be used to validate scheduler shape generalization.

| data type | reason rejected |
|---|---|
| residual candidate CSV | not scheduler descriptor; contains latency/residual semantics |
| runtime shape rank rows | lack scheduler request split and DP EngineCore step key |
| profiler/debug trace | not descriptor-only and not scheduler semantics |
| cb_sim generated rows | not real vLLM holdout evidence |

## Boundary

Phase68 does not run remote capture. The current GPU entry is only recorded for future use:

`ssh -CAXY ws-faaf0de74ef9a14d-worker-kw5fs.zhaojieyu+root.ailab-sys.pod@h.pjlab.org.cn`

If future work needs holdouts, it must collect descriptor-only vLLM scheduler alignment rows first. It must not substitute latency, residual, profiler, NCCL, sync, or throughput data.
