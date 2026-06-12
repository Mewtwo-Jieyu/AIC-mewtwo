# Phase247: 4k2k Actual Scheduled Token Family

## Decision

| Item | Result |
|---|---|
| Mechanism hypothesis | actual_scheduled_tokens_not_configured_budget |
| Default AIC | No-Go |
| PerfDatabase | No-Go |
| GPU benchmark | No-Go in Phase247 |
| Boundary flags | diagnostic_only=true valid_for_default=false perf_database=false |

## Family Summary

| Topology | Role | Scenario | max_bt | output tok/s | output ratio | rank min p50/p99/max | rank max p50/p99/max | rank sum p50/p99/max | rank sum mean fill | rank sum max fill |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tp8_dp1_ep8 | control | tp8ep8-4k2k-bt4000 | 4000 | 4071.651448 | 1.000000 | 128.000000/128.000000/4000 | 128.000000/128.000000/4000 | 128.000000/128.000000/4000 | 0.032705 | 1.000000 |
| tp8_dp1_ep8 | holdout | tp8ep8-4k2k-bt65536 | 65536 | 4027.060291 | 0.989048 | 128.000000/128.000000/4000 | 128.000000/128.000000/4000 | 128.000000/128.000000/4000 | 0.001995 | 0.061035 |
| tp4_dp2_ep8 | control | tp4dp2ep8-4k2k-bt4000 | 4000 | 3143.891608 | 1.000000 | 60.000000/60.000000/4000 | 68.000000/68.000000/4000 | 128.000000/128.000000/4000 | 0.032703 | 1.000000 |
| tp4_dp2_ep8 | holdout | tp4dp2ep8-4k2k-bt65536 | 65536 | 3445.355421 | 1.095889 | 63.000000/63.000000/4000 | 65.000000/65.000000/4240 | 128.000000/128.000000/8240 | 0.002027 | 0.125732 |

## Interpretation

Both 4k2k topology pairs support actual_scheduled_tokens_not_configured_budget.
The high-budget rows configure max_bt=65536, but actual scheduled rank-sum tokens stay far below that ceiling.
For tp8_dp1_ep8, worker payloads are expected to be identical within an iteration and are deduplicated by iteration.
For tp4_dp2_ep8, DP=2 rows are aggregated as rank min/max/sum because different DP payloads can appear in the same iteration.
This keeps the mechanism diagnostic-only: model scheduler cost from actual scheduled tokens, phase mix, and decode batch tokens, not from a linear penalty on max_num_batched_tokens.
Do not wire this into VLLMBackend.run_agg, default AIC, or PerfDatabase.
