# Phase258: Actual Scheduled Token Full Family

## Decision

| Item | Result |
|---|---|
| Mechanism hypothesis | actual_scheduled_tokens_not_configured_budget |
| Default AIC | No-Go |
| PerfDatabase | No-Go |
| GPU benchmark | No-Go in Phase258 |
| Boundary flags | diagnostic_only=true valid_for_default=false perf_database=false |
| Budget denominator | configured_budget_aggregate=max_num_batched_tokens * dp |

## Family Summary

| Topology / shape | Role | Scenario | max_bt | aggregate budget | output tok/s | output ratio | rank sum p50/p95/p99/max | rank sum mean/p99/max fill |
|---|---|---|---:|---:|---:|---:|---:|---:|
| tp8_dp1_ep8 / 4k2k | control | tp8ep8-4k2k-bt4000 | 4000 | 4000 | 4071.651448 | 1.000000 | 128.000000/128.000000/128.000000/4000 | 0.032705/0.032000/1.000000 |
| tp8_dp1_ep8 / 4k2k | holdout | tp8ep8-4k2k-bt65536 | 65536 | 65536 | 4027.060291 | 0.989048 | 128.000000/128.000000/128.000000/4000 | 0.001995/0.001953/0.061035 |
| tp4_dp2_ep8 / 4k2k | control | tp4dp2ep8-4k2k-bt4000 | 4000 | 8000 | 3143.891608 | 1.000000 | 128.000000/128.000000/128.000000/8000 | 0.016601/0.016000/1.000000 |
| tp4_dp2_ep8 / 4k2k | holdout | tp4dp2ep8-4k2k-bt65536 | 65536 | 131072 | 3445.355421 | 1.095889 | 128.000000/128.000000/128.000000/8240 | 0.001013/0.000977/0.062866 |
| tp8_dp1_ep8 / 12k2k | control | tp8ep8-12k2k-bt12000 | 12000 | 12000 | 2777.513790 | 1.000000 | 128.000000/128.000000/128.000000/12000 | 0.011201/0.010667/1.000000 |
| tp8_dp1_ep8 / 12k2k | holdout | tp8ep8-12k2k-bt65536 | 65536 | 65536 | 2687.912821 | 0.967741 | 128.000000/128.000000/128.000000/12000 | 0.002055/0.001953/0.183105 |
| tp4_dp2_ep8 / 12k2k | control | tp4dp2ep8-12k2k-bt12000 | 12000 | 24000 | 2091.977129 | 1.000000 | 128.000000/128.000000/128.000000/24000 | 0.005867/0.005333/1.000000 |
| tp4_dp2_ep8 / 12k2k | holdout | tp4dp2ep8-12k2k-bt65536 | 65536 | 131072 | 2802.211032 | 1.339504 | 128.000000/128.000000/128.000000/24000 | 0.001074/0.000977/0.183105 |

## Interpretation

Phase258 extends Phase253 from six rows to the full eight-row trace family: 4k2k and 12k2k across tp8_dp1_ep8 and tp4_dp2_ep8, each with control and holdout.
For tp4_dp2_ep8, rank-sum scheduled tokens are compared against the aggregate configured budget, max_num_batched_tokens * dp, so control rows are not misread as overfilled.
The high-budget rows configure max_bt=65536, but their rank-sum p99 and max scheduled tokens remain far below the aggregate budget.
This supports modeling scheduler cost from actual scheduled tokens, phase mix, and decode batch tokens instead of a linear cost on max_num_batched_tokens.
This remains diagnostic-only evidence. Do not wire it into VLLMBackend.run_agg, default AIC, or PerfDatabase.
