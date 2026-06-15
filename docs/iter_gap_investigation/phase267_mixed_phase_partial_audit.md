# Phase267: Mixed Phase Partial-Only Audit

## Decision

| Item | Result |
|---|---|
| Mixed phase | Mixed phase is partial-only |
| Pure decode | Invariant at p99/max 128/128 |
| Budget ceiling | Still not a linear configured-budget cost |
| Next mechanism | boundary_timeline_diagnostic |
| Default AIC | No-Go |
| PerfDatabase | No-Go |
| Boundary flags | diagnostic_only=true valid_for_default=false perf_database=false |

## Pair Audit

| Pair | output ratio | mixed count delta | mixed p99 delta | mixed max delta | verdict |
|---|---:|---:|---:|---:|---|
| tp8_dp1_ep8:isl4000_osl2000_batch128:control_bt4000:holdout_bt65536 | 0.989048 | 1 | -480.000000 | -480 | partial |
| tp4_dp2_ep8:isl4000_osl2000_batch128:control_bt4000:holdout_bt65536 | 1.095889 | -1 | 0.000000 | 0 | explains |
| tp8_dp1_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536 | 0.967741 | -1 | 224.000000 | 224 | partial |
| tp4_dp2_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536 | 1.339504 | 0 | 570.000000 | 570 | does_not_explain |

## Interpretation

Phase267 makes the Phase266 readout machine-checkable.
Pure decode batch size is not the source of the spread because every pair keeps p99/max at 128/128.
The high-budget rows still schedule far below the aggregate configured budget, so max_num_batched_tokens remains a ceiling rather than a direct cost.
Mixed phase changes explain only one pair cleanly and are partial or insufficient for the others.
The next diagnostic layer should inspect boundary/timeline behavior instead of turning mixed phase into a default model.
This remains diagnostic-only evidence. Do not wire it into VLLMBackend.run_agg, default AIC, or PerfDatabase.
