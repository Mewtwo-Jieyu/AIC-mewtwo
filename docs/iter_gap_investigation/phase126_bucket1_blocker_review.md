# Phase126: Bucket1 Blocker Review

## Decision

| Item | Result |
|---|---|
| Review scope | Read-only blocker diagnosis |
| Second full capture | Not run |
| Request shape difference | Not found |
| Parser issue | Not found |
| Cleanup issue | Not found |
| Root cause class | `rank=0 / tokens_actual=1` tail marker boundary |
| Next action | Fix runner drain boundary, then allow one drain-instrumented capture design |
| Exact count rule | Keep blocked until drain boundary is defined |

## Request Consistency

| Field | Phase124 | Capture1 | Result |
|---|---:|---:|---|
| `ok_requests` | 32 | 32 | Same |
| `failed_requests` | 0 | 0 | Same |
| `bench_records.jsonl` rows | 32 | 32 | Same |
| request indexes | `0..31` | `0..31` | Same |
| prompt tokens | `10000` | `10000` | Same |
| completion tokens | `2000` | `2000` | Same |
| total tokens | `12000` | `12000` | Same |
| status codes | `200` | `200` | Same |

Conclusion: client request shape does not explain the missing markers. Bench result remains traffic-health evidence only.

## Token Diff

| tokens_actual | Phase124 | Capture1 | Delta |
|---:|---:|---:|---:|
| 1 | 6960 | 2858 | -4102 |
| 15 | 240 | 240 | 0 |
| 16 | 959280 | 959280 | 0 |
| 241 | 240 | 240 | 0 |
| 1808 | 240 | 240 | 0 |
| 2048 | 240 | 240 | 0 |
| 8192 | 480 | 480 | 0 |

All non-`1` buckets match exactly.

## Rank0 Check

| tokens_actual | Phase124 rank0 | Capture1 rank0 | Delta |
|---:|---:|---:|---:|
| 1 | 6960 | 2858 | -4102 |
| 15 | 240 | 240 | 0 |
| 16 | 959280 | 959280 | 0 |
| 241 | 240 | 240 | 0 |
| 1808 | 240 | 240 | 0 |
| 2048 | 240 | 240 | 0 |
| 8192 | 480 | 480 | 0 |

All `tokens_actual=1` rows are on `rank=0` in both runs.

## Tail Segment

| Field | Phase124 | Capture1 |
|---|---:|---:|
| total marker rows | 967680 | 963578 |
| first marker line | 375 | 370 |
| last marker line | 972116 | 967986 |
| last non-1 line | 965336 | 965331 |
| final token1 segment start | 965346 | 965336 |
| final token1 segment end | 972116 | 967986 |
| final token1 segment count | 6720 | 2618 |
| non-tail token1 count | 240 | 240 |

The missing `4102` rows are exactly the difference in the final `rank=0 / tokens_actual=1` tail segment: `6720 - 2618 = 4102`.

## Marker Boundary Finding

Capture1 removes the enable file after the benchmark client returns and before stopping the service. Phase124 captured a longer final `rank=0 / tokens_actual=1` drain segment after the last non-1 marker. Capture1 captured the same core request buckets, but closed the marker boundary before the tail segment reached the Phase124 length.

This is a marker-boundary mismatch, not a request-shape mismatch.

## Classification

| Candidate | Result | Reason |
|---|---|---|
| Request difference | Rejected | 32 records, token usage, status, and request indexes match |
| Parser issue | Rejected | Parser produced all 7 buckets and all non-tail counts match |
| Cleanup issue | Rejected | source marker absent, serve process absent, GPU residual absent |
| rank0 tail boundary | Supported | only final `rank=0/token=1` tail segment differs |

## Decision

Do not relax Phase125 manifest and do not accept capture1. Keep full capture No-Go.

Next work should fix or instrument the runner drain boundary, then allow one drain-instrumented capture design. The design must make the marker close point explicit before any exact total-count acceptance is used.

## Read-Only Remote Commands

| Command class | Purpose |
|---|---|
| `bench_result.json` / `bench_records.jsonl` read | Compare request shape |
| `moe_activation_rows.csv` streaming read | Aggregate token/rank/tail windows |
| `moe_token_bucket_coverage.csv` read | Confirm bucket/count mismatch |
| source grep | Confirm original vLLM source marker absent |
| process/GPU checks | Confirm cleanup state |

No large remote CSV was copied into the local worktree.
