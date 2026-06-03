# Phase126: Exact-Shape Capture Design

## Decision

| Item | Result |
|---|---|
| Goal | Reproduce exact MoE activation buckets from Phase124 |
| Target buckets | `1 / 15 / 16 / 241 / 1808 / 2048 / 8192` |
| Output dir | `phase126_moe_activation_10k2k_b32_bt8192_p18000_capture1` |
| Traffic generation | Go |
| Benchmark metric use | No-Go |
| Capture1 result | Blocked by exact coverage mismatch |
| Default AIC | No-Go |
| PerfDatabase | No-Go |

## Fixed Run Parameters

| Parameter | Value |
|---|---|
| `PORT` | `18000` |
| `BENCH_NUM_PROMPTS` | `32` |
| `BENCH_MAX_CONCURRENCY` | `32` |
| `BENCH_INPUT_LEN` | `10000` |
| `BENCH_OUTPUT_LEN` | `2000` |
| `OUT_DIR` | `docs/iter_gap_investigation/phase126_moe_activation_10k2k_b32_bt8192_p18000_capture1` |

## Evidence Contract

| File | Allowed use |
|---|---|
| `bench_result.json` | Traffic health only: `ok_requests=32`, `failed_requests=0` |
| `moe_activation_rows.csv` | Marker schema evidence |
| `moe_token_bucket_coverage.csv` | Exact bucket/count coverage evidence |
| `serve.log` | Raw marker source and debugging only |

`bench_result.json` must not be used as model, default AIC, or PerfDatabase evidence.

## Expected Coverage

| tokens_actual | occurrence_count | phase117_exact_key_covered |
|---:|---:|---|
| 1 | 6960 | `False` |
| 15 | 240 | `False` |
| 16 | 959280 | `False` |
| 241 | 240 | `False` |
| 1808 | 240 | `False` |
| 2048 | 240 | `False` |
| 8192 | 480 | `False` |

Total marker rows must be `967680`.

## Stop Rules

| Condition | Action |
|---|---|
| Any schema rejection | Stop and write blocker |
| Any bucket/count mismatch | Stop and write blocker |
| Any covered bucket | Stop and write blocker |
| Traffic failure | Stop and write blocker |
| Cleanup failure | Stop and write blocker |
| Default AIC request | No-Go |
| PerfDatabase request | No-Go |
| Interpolation or extrapolation request | No-Go |

## Cleanup Checks

| Check | Required result |
|---|---|
| Original vLLM source marker | Absent |
| vLLM serve process | Absent |
| GPU compute residual | Absent |

## Capture1 Result

| Item | Result |
|---|---|
| Traffic health | PASS: `ok_requests=32`, `failed_requests=0` |
| Parser | PASS |
| Coverage acceptance | FAIL: bucket `1` expected `6960`, actual `2858` |
| Cleanup | PASS |
| Next full capture | No-Go until blocker review |

## Capture Command

```bash
cd /mnt/nvme1n1/ml_research/jieyu/aic
PORT=18000 \
BENCH_NUM_PROMPTS=32 \
BENCH_MAX_CONCURRENCY=32 \
BENCH_INPUT_LEN=10000 \
BENCH_OUTPUT_LEN=2000 \
OUT_DIR=docs/iter_gap_investigation/phase126_moe_activation_10k2k_b32_bt8192_p18000_capture1 \
bash collector/vllm/run_phase126_moe_activation_marker.sh
```
