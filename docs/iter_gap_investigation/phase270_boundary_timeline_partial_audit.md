# Phase270: Boundary / Timeline Partial Audit

## Decision

| Item | Result |
|---|---|
| Conclusion | boundary_timeline_partial_only |
| Boundary / timeline | Partial-only, not a model |
| tp8 12k2k needs deeper trace | true |
| Default AIC | No-Go |
| PerfDatabase | No-Go |
| Boundary flags | diagnostic_only=true valid_for_default=false perf_database=false |

## Pair Audit

| Pair | control boundary | holdout boundary | total iteration delta | verdict | next |
|---|---|---|---:|---|---|
| tp8_dp1_ep8:isl4000_osl2000_batch128:control_bt4000:holdout_bt65536 | mixed=2;first_pure=1;tail=128,128,128,127,127;iters=2002 | mixed=2,3;first_pure=1;tail=128,128,127,127,30;iters=2003 | 1 | boundary_explains | boundary_timeline_partial_only |
| tp4_dp2_ep8:isl4000_osl2000_batch128:control_bt4000:holdout_bt65536 | mixed=1,2;first_pure=3;tail=128,128,128,126,111;iters=2002 | mixed=2;first_pure=1;tail=128,128,128,111,111;iters=2002 | 0 | boundary_explains | boundary_timeline_partial_only |
| tp8_dp1_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536 | mixed=2,4,7,8;first_pure=1;tail=62,30,30,30,20;iters=2008 | mixed=2,3,4;first_pure=1;tail=128,127,127,48,30;iters=2004 | -4 | needs_deeper_trace | deeper_trace_for_tp8_12k2k |
| tp4_dp2_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536 | mixed=1,2;first_pure=3;tail=128,128,128,126,72;iters=2002 | mixed=2,3;first_pure=1;tail=128,128,126,126,15;iters=2003 | 1 | boundary_partial | boundary_timeline_partial_only |

## Interpretation

Phase270 makes the Phase269 boundary/timeline readout machine-checkable.
Boundary timing explains both 4k2k pairs, but tp8 12k2k moves in the wrong direction and cannot be explained by this layer.
tp4dp2 12k2k has useful boundary signal, but the signal is not strong enough to explain the full throughput ratio.
The next diagnostic step is deeper trace instrumentation for tp8 12k2k before any default model discussion.
This remains diagnostic-only evidence. Do not wire it into VLLMBackend.run_agg, default AIC, or PerfDatabase.
