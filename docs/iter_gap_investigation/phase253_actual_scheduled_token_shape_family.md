# Phase253: Actual Scheduled Token Shape Family

## Decision

| Item | Result |
|---|---|
| Mechanism hypothesis | actual_scheduled_tokens_not_configured_budget |
| Default AIC | No-Go |
| PerfDatabase | No-Go |
| GPU benchmark | No-Go in Phase253 |
| Boundary flags | diagnostic_only=true valid_for_default=false perf_database=false |

## Family Summary

| Topology / shape | Role | Scenario | max_bt | output tok/s | output ratio | rank sum p50/p95/p99/max | rank sum mean/p99/max fill |
|---|---|---|---:|---:|---:|---:|---:|
| tp8_dp1_ep8 / 4k2k | control | tp8ep8-4k2k-bt4000 | 4000 | 4071.651448 | 1.000000 | 128.000000/128.000000/128.000000/4000 | 0.032705/0.032000/1.000000 |
| tp8_dp1_ep8 / 4k2k | holdout | tp8ep8-4k2k-bt65536 | 65536 | 4027.060291 | 0.989048 | 128.000000/128.000000/128.000000/4000 | 0.001995/0.001953/0.061035 |
| tp4_dp2_ep8 / 4k2k | control | tp4dp2ep8-4k2k-bt4000 | 4000 | 3143.891608 | 1.000000 | 128.000000/128.000000/128.000000/4000 | 0.032703/0.032000/1.000000 |
| tp4_dp2_ep8 / 4k2k | holdout | tp4dp2ep8-4k2k-bt65536 | 65536 | 3445.355421 | 1.095889 | 128.000000/128.000000/128.000000/8240 | 0.002027/0.001953/0.125732 |
| tp8_dp1_ep8 / 12k2k | control | tp8ep8-12k2k-bt12000 | 12000 | 2777.513790 | 1.000000 | 128.000000/128.000000/128.000000/12000 | 0.011201/0.010667/1.000000 |
| tp8_dp1_ep8 / 12k2k | holdout | tp8ep8-12k2k-bt65536 | 65536 | 2687.912821 | 0.967741 | 128.000000/128.000000/128.000000/12000 | 0.002055/0.001953/0.183105 |

## Interpretation

The Phase253 family adds the tp8_dp1_ep8 / 12k2k pair to the Phase247 4k2k trace evidence.
The high-budget rows configure max_bt=65536, but actual scheduled rank-sum tokens remain far below that ceiling.
For tp8_dp1_ep8, worker payloads are deduplicated by iteration because TP workers should report the same scheduler payload.
For tp4_dp2_ep8, DP=2 rows are aggregated as rank min/max/sum because different DP payloads can appear in the same iteration.
This supports modeling scheduler cost from actual scheduled tokens, phase mix, and decode batch tokens instead of a linear cost on max_num_batched_tokens.
This remains diagnostic-only evidence. Do not wire it into VLLMBackend.run_agg, default AIC, or PerfDatabase.
