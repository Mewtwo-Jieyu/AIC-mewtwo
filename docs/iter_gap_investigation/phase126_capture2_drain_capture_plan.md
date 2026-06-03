# Phase126: Capture2 Drain-Instrumented Plan

## Decision

| Item | Result |
|---|---|
| Capture attempt | `capture2` |
| Full capture allowance | Once only |
| Drain boundary | Required |
| Benchmark metric use | No-Go |
| Default AIC | No-Go |
| PerfDatabase | No-Go |
| Interpolation / extrapolation | No-Go |

## Fixed Output

| Item | Value |
|---|---|
| Output dir | `docs/iter_gap_investigation/phase126_moe_activation_10k2k_b32_bt8192_p18000_capture2_drain` |
| Remote workdir | `/mnt/nvme1n1/ml_research/jieyu/aic` |

## Fixed Parameters

| Parameter | Value |
|---|---:|
| `PORT` | 18000 |
| `BENCH_NUM_PROMPTS` | 32 |
| `BENCH_MAX_CONCURRENCY` | 32 |
| `BENCH_INPUT_LEN` | 10000 |
| `BENCH_OUTPUT_LEN` | 2000 |
| `DRAIN_POLL_SECONDS` | 5 |
| `DRAIN_STABLE_POLLS` | 3 |
| `DRAIN_TIMEOUT_SECONDS` | 900 |

## Acceptance Criteria

| Artifact | Required result |
|---|---|
| `bench_result.json` | `ok_requests=32`, `failed_requests=0` |
| `bench_records.jsonl` | 32 records, token shape `10000/2000/12000` |
| `drain_boundary.log` | Last row `close_reason=stable`; marker rows stable for 3 polls |
| `moe_activation_rows.csv` | Parser accepted schema; `timing=false` |
| `moe_token_bucket_coverage.csv` | 7 buckets, total `967680`, all `phase117_exact_key_covered=False` |
| cleanup | source marker absent, vLLM serve absent, GPU residual absent |

## Expected Coverage

| tokens_actual | occurrence_count |
|---:|---:|
| 1 | 6960 |
| 15 | 240 |
| 16 | 959280 |
| 241 | 240 |
| 1808 | 240 |
| 2048 | 240 |
| 8192 | 480 |

## Capture2 Result

| Check | Result |
|---|---|
| Capture status | Accepted |
| Remote output dir | `/mnt/nvme1n1/ml_research/jieyu/aic/docs/iter_gap_investigation/phase126_moe_activation_10k2k_b32_bt8192_p18000_capture2_drain` |
| `bench_result.json` | `ok_requests=32`, `failed_requests=0` |
| `bench_records.jsonl` | 32 records; token shape `10000/2000/12000` |
| `drain_boundary.log` | `close_reason=stable`, final marker rows `967680` |
| Stable tail marker rows | `967680 / 967680 / 967680` |
| `moe_activation_rows.csv` | parser accepted; rows `967680`; `timing=false` |
| `moe_token_bucket_coverage.csv` | matches expected coverage; total `967680`; all `phase117_exact_key_covered=False` |
| cleanup | source marker absent; vLLM serve absent; GPU residual absent |

## Failure Handling

| Failure | Action |
|---|---|
| Traffic failure | Write blocker; do not accept artifact |
| Drain timeout | Write blocker; do not parse artifact as accepted |
| Drain count decrease | Write blocker; do not parse artifact as accepted |
| Coverage mismatch | Write blocker; do not relax Phase125 manifest |
| Cleanup failure | Write blocker; do not accept artifact |

## Capture Command

```bash
cd /mnt/nvme1n1/ml_research/jieyu/aic
PORT=18000 \
BENCH_NUM_PROMPTS=32 \
BENCH_MAX_CONCURRENCY=32 \
BENCH_INPUT_LEN=10000 \
BENCH_OUTPUT_LEN=2000 \
DRAIN_POLL_SECONDS=5 \
DRAIN_STABLE_POLLS=3 \
DRAIN_TIMEOUT_SECONDS=900 \
OUT_DIR=docs/iter_gap_investigation/phase126_moe_activation_10k2k_b32_bt8192_p18000_capture2_drain \
bash collector/vllm/run_phase126_moe_activation_marker.sh
```

## Boundary

`bench_result.json` is traffic-health evidence only. Capture2 remains diagnostic exact-shape evidence only and must not enter default AIC or PerfDatabase.
