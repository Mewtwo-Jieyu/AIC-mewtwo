# Phase 72 Go/No-Go

## Conclusion

Phase72 is Go for Phase73 implementation, but only as a narrow `32k1k_b16` descriptor branch. It is not Go for generic scheduler rules, default compare, or latency modeling.

| gate | decision |
|---|---|
| Phase71 rows explainable by a deterministic rule | Go |
| current generator already generalizes | No |
| Phase73 may add a guarded branch | Go |
| Phase73 may abstract generic long-ISL rules | No |
| Phase73 may touch default latency | No |

## Why This Is Go

| fact | meaning |
|---|---|
| Phase71 row set is complete | no missing descriptor fields |
| key is stable | `engine_dp:<dp_rank>:step:<engine_step_id>` is present |
| row count is deterministic | `1003` rows per DP |
| prefill chunks are explicit | `8192,8192,8192,7536` |
| decode tail is stable | `0+8 / FULL:8` after the first decode row |

## Why This Is Still Narrow

| limit | consequence |
|---|---|
| only one `32k1k` holdout | no generic long-ISL claim |
| only one budget | no budget sweep claim |
| only one topology | no topology claim |
| only descriptor rows | no latency claim |

## Phase73 Entry Conditions

| condition | required action |
|---|---|
| implement code | add only the guarded `32k1k_b16` branch |
| add tests | cover first rows, tail rows, row count, and strict compare |
| preserve regressions | `10k2k_b32` and `3k3k_b128` strict compare must remain all-match |
| unsupported inputs | fail-fast without fallback |
| output fields | keep descriptor-only fields |

## Stop Conditions

| condition | action |
|---|---|
| implementation needs heuristics | stop |
| implementation needs phase ordinal join | stop |
| implementation creates mixed row for `32k1k` | stop |
| output adds latency, residual, profiler, NCCL, sync, or throughput fields | stop |
| default `cb_sim` path changes | stop |

## Next Step

Phase73 can implement the narrow branch and strict tests. It should not run remote capture, change `PerfDatabase`, or modify `run_static` / `IterationLatencyCalculator`.
