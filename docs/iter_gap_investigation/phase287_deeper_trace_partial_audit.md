# Phase287 Deeper Trace Partial Audit

This diagnostic audit compares the tp8 12k2k deeper trace control/holdout pair.

| Item | Value |
|---|---|
| Pair | tp8_dp1_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536 |
| Output ratio | 0.969300 |
| Unique iteration delta | 6 |
| Mixed iteration delta | 3 |
| Control mixed iterations | 2,3 |
| Holdout mixed iterations | 2,3,4,8,9 |
| Control tail decode tokens | 128,128,127,127,12 |
| Holdout tail decode tokens | 30,30,30,30,9 |
| Holdout max fill | 0.183105 |
| Verdict | partial_only |
| Mechanism | boundary_mixed_overhead_partial_only |
| Default AIC | No-Go |

The conclusion remains diagnostic-only: boundary, mixed iteration count, and rank-max timing overhead explain only part of the observed throughput delta. The holdout still schedules at most 12000 tokens against a configured max budget of 65536, so the configured budget is not a linear runtime cost.

Flags: diagnostic_only=true, valid_for_default=false, perf_database=false.
