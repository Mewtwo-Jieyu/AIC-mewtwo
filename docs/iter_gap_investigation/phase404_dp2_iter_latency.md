# Phase404 DP2 decode-iteration attribution

Phase404 finds only a minor active decode iteration gap. The larger residual is aggregate-implied latency, so Phase405 should target decode duty cycle or queueing first, not a single generation op.

## Rows

| scenario | row_type | real active ms/iter | aggregate implied ms/iter | sim ms/iter | residual active | residual aggregate | dispatch ms | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| K2.5-tp4ep8dp2-8k2k | dp2_attribution | 41.015245 | 92.583825 | 31.365256 | 9.649990 | 61.218569 | 3.389192 | aggregate_gap_dominates_minor_active_decode_gap |
| K2.5-tp4ep8dp2-32k3k | dp2_attribution | 26.443245 | 50.953952 | 21.056851 | 5.386393 | 29.897100 | 2.445050 | aggregate_gap_dominates_minor_active_decode_gap |
| K2.5-tp8ep8-32k3k | tp8_crosscheck |  | 34.930725 | 34.190681 |  | 0.740044 | 0.633463 | tp8_control_no_dp_only_iter_gap |

## Interpretation

- real_active_decode_ms_per_iter comes from Phase403 /metrics generation-token counters and running batch.
- real_aggregate_implied_ms_per_iter comes from total output throughput and running max; it includes queueing, prefill, and idle/duty-cycle effects.
- gen_dispatch_ms is already charged serially in the pure decode path, and its magnitude is too small to explain a large aggregate-implied residual when active decode matches.
- TP8-32k is retained only as a control row; Phase403 did not recollect active decode counters for TP8.

## Boundary

- gpu_allowed=false, ssh_allowed=false.
- runtime_modified=false, perf_database=false, valid_for_default=false, diagnostic_only=true, default_readiness=No-Go.
- Phase405 should target the aggregate decode duty/queueing gap unless a later trace contradicts the active decode counter result.
