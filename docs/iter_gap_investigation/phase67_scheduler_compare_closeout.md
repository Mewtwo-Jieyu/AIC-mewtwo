# Phase 67 Scheduler Strict Compare Closeout

## Conclusion

Phase67 completes the descriptor-only strict compare for the first-evidence scheduler shape. The experimental vLLM-like scheduler descriptor generated from cb_sim inputs matches the Phase62 real vLLM scheduler descriptor exactly under `(alignment_key, dp_rank)`.

This is not a latency model. It only proves that AIC can express the scheduler/runtime shape input sequence for this scenario.

| Item | Result |
|---|---|
| join key | `(alignment_key, dp_rank)` |
| key type | `engine_core_dp_step` |
| output rows | `4003` |
| missing cb / vLLM keys | none |
| token/request/forward deltas | all `0` |
| same flags | all `1` |
| default AIC | unchanged |

## Artifacts

| File | Content |
|---|---|
| `phase67_vllm_like_scheduler_descriptor.csv` | cb_sim-side experimental vLLM-like descriptor |
| `phase67_vllm_like_scheduler_strict_compare.csv` | strict compare against Phase62 vLLM descriptor |

The compare CSV contains no latency, residual, profiler, NCCL, sync, or throughput fields.

## Key Rows

| key | result |
|---|---|
| `engine_dp:1:step:2` | `240+1`, `NONE:248`, all deltas `0` |
| `engine_dp:1:step:2001` | `15`, `FULL:16`, all deltas `0` |

## CLI Used

```bash
conda run -n aic env PYTHONPATH=src python scripts/diagnose_cb_iter_latency.py \
  --isl 10000 \
  --osl 2000 \
  --concurrency 32 \
  --tp 4 \
  --dp 2 \
  --moe-tp 1 \
  --moe-ep 8 \
  --max-num-batched-tokens 8192 \
  --experimental-vllm-like-scheduler-descriptor \
  --vllm-like-scheduler-descriptor-out docs/iter_gap_investigation/phase67_vllm_like_scheduler_descriptor.csv \
  --scheduler-alignment-vllm-csv docs/iter_gap_investigation/phase62_scheduler_alignment_10k2k_b32/scheduler_alignment_rows_dedup_by_alignment_dp.csv \
  --scheduler-alignment-compare-out docs/iter_gap_investigation/phase67_vllm_like_scheduler_strict_compare.csv
```

## Boundary

The compare path reuses `compare_scheduler_aligned_descriptors(...)`. It does not relax key matching, does not use phase ordinal, does not take intersections, and does not infer missing rows.

The result should be treated as a scheduler shape input alignment, not a performance result.
