# Phase301 Boundary Timeline Cadence Diagnostic

This diagnostic compares the tp4dp2 12k2k deeper trace control/holdout pair.

| Item | Value |
|---|---|
| Pair | tp4_dp2_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536 |
| Source SHA comparison | startup_diagnostics_only |
| Output ratio | 1.321996 |
| Mixed sequence | 1,2 -> 2 |
| First pure decode | 3 -> 1 |
| Trace wall-span | 125.456810s -> 94.883212s |
| Iteration cadence p50 | 59.192128ms -> 43.536153ms |
| Scheduled max fill | 1.000000 -> 0.188721 |
| Verdict | boundary_timeline_explains_direction |
| Mechanism | wall_span_iteration_cadence_diagnostic |
| Default AIC | No-Go |

The main signal is shorter trace wall-span and faster iteration-start cadence. Mixed phase and first-pure-decode movement explain the direction, while overhead p50/p95 is not promoted into a standalone model.

The result stays diagnostic-only. It must not be used for default AIC, interpolation, extrapolation, or PerfDatabase writes.

Flags: diagnostic_only=true, valid_for_default=false, perf_database=false.
