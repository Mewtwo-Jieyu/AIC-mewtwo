# Phase242: Actual Scheduled Token Mechanism

## Decision

| Item | Result |
|---|---|
| Mechanism hypothesis | actual_scheduled_tokens_not_configured_budget |
| Default AIC | No-Go |
| PerfDatabase | No-Go |
| GPU benchmark | No-Go in Phase242 |
| Boundary flags | diagnostic_only=true valid_for_default=false perf_database=false |

## Pair Summary

| Role | Scenario | max_num_batched_tokens | max scheduled | p50 scheduled | p99 scheduled | mean fill | max fill | output tok/s | output ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | tp8ep8-4k2k-bt4000 | 4000 | 4000 | 128.000000 | 128.000000 | 0.032705 | 1.000000 | 4071.651448 | 1.000000 |
| holdout | tp8ep8-4k2k-bt65536 | 65536 | 4000 | 128.000000 | 128.000000 | 0.001995 | 0.061035 | 4027.060291 | 0.989048 |

## Interpretation

max_num_batched_tokens is a ceiling, not the actual per-iteration scheduler cost.
The holdout config sets max_num_batched_tokens=65536, but max scheduled total tokens stays at 4000.
The steady decode distribution stays near p50=128.000000 and p99=128.000000 scheduled tokens.
This supports modeling scheduler cost from scheduled_total_tokens, phase mix, and decode batch tokens instead of applying a linear penalty to the configured budget ceiling.
The evidence is still diagnostic-only and must not be wired into VLLMBackend.run_agg, default AIC, or PerfDatabase.

## Guard

Control and holdout share topology_key=tp8_dp1_ep8 and shape_key=isl4000_osl2000_batch128.
The pair keeps diagnostic_only=true, valid_for_default=false, and perf_database=false.
