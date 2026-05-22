# Phase 66 vLLM-like Scheduler Descriptor Design

## Implementation

Phase66 adds `vllm_like_scheduler_aligned_descriptors(...)` in `forward_descriptor.py` and an experimental diagnose CLI flag:

`--experimental-vllm-like-scheduler-descriptor`

The output is a descriptor CSV only. It does not run cb_sim latency simulation and does not change the default validate or diagnose paths.

## Supported Shape Class

The first implementation intentionally supports the Phase62 first-evidence shape class only:

| Requirement | Rule |
|---|---|
| uniform batch | `concurrency % dp == 0` |
| per-DP requests | `concurrency / dp`, must be at least `2` |
| chunked leader prefill | `isl > max_num_batched_tokens` |
| follower suffix | `(per_dp_reqs - 1) * block_size` |
| token budget | leader remainder + follower suffix must fit one step |
| graph padding | graph path pads to multiple of `8` |

Unsupported shapes fail fast instead of guessing vLLM internals.

## Generator Semantics

| DP path | Sequence |
|---|---|
| `dp_rank=0` | leader `8192`, leader remainder + follower suffixes, then pure decode |
| `dp_rank>0` | leader `8192`, leader remainder, mixed follower suffixes + one leader decode, then pure decode tail |

For `10k2k_b32 bt8192 dp=2`, this yields exactly:

| key | generated row |
|---|---|
| `engine_dp:1:step:2` | `240+1`, `NONE:248` |
| `engine_dp:1:step:2001` | `15`, `FULL:16` |

## CLI

Example:

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
  --vllm-like-scheduler-descriptor-out docs/iter_gap_investigation/phase66_vllm_like_scheduler_descriptor.csv
```

## Output Artifacts

| artifact | result |
|---|---|
| `phase66_vllm_like_scheduler_descriptor.csv` | `4003` descriptor rows |
| `phase66_vllm_like_scheduler_gap.csv` | `4003` `shape_match` rows against Phase62 |

## Forbidden Fields

The generator does not emit timing, residual, profiler, NCCL, sync, or throughput fields. It also never sets `valid_for_default=true` or `perf_database=true`.
