# Phase456 TP8 dynamics

Verdict: `tp8_capacity_wiring_improves_but_residual_remains`.

## Real Profile

| metric | value | note |
|---|---:|---|
| kv_cache_tokens | 546160 |  |
| num_gpu_blocks | 34135 |  |
| running_p50 | 56.0 | p10=52.000000; p90=64.300000; n=88 |
| waiting_p50 | 69.0 | p10=31.000000; p90=74.000000; n=88 |
| kv_usage_p50 | 98.5 | p10=79.180000; p90=99.630000; n=88 |

## Real Fingerprint

| metric | value | note |
|---|---:|---|
| mixed_share | 0.028950276243093924 | count=4192; mean=0.028950; p10=; p90= |
| mixed_decode_batch | 51.0 | count=4192; mean=50.213740; p10=50.000000; p90=60.000000 |
| mixed_ctx_tokens | 7949.0 | count=4192; mean=7801.526718; p10=7938.000000; p90=7950.000000 |
| mixed_forward_busy_ms | 608.102294921875 | count=4192; mean=598.138319; p10=603.096851; p90=611.393494 |
| prefill_forward_busy_ms | 608.0960083007812 | count=4200; mean=598.110471; p10=602.951434; p90=611.393494 |
| decode_batch | 59.0 | count=140600; mean=56.744865; p10=32.000000; p90=65.000000 |
| decode_forward_busy_ms | 27.37742328643799 | count=140600; mean=31.305285; p10=20.654298; p90=31.736589 |

## Capacity Counterfactual

| metric | value | status | note |
|---|---:|---|---|
| old_capacity_error_ratio | 1.4385687849872588 | fail | sim=192.089; real=133.528; peak_decode=94; avg_decode=76.107 |
| current_capacity_error_ratio | 1.2440613566389125 | fail | sim=166.117; real=133.528; peak_decode=67; avg_decode=54.576 |
| error_ratio_delta | 0.19450742834834633 | pass | capacity wiring fixes most TP8-8k2k overprediction but does not close the 15% gate |

The TP8 8k2k capacity wiring update moves sim throughput toward the Phase454 real profile, but the remaining error is still above the 15% target.
The withdrawn Phase455 TP8 B2b rows stay out of PerfDB; the residual is a dynamics issue, not a per-step serving-state row issue.

## Validate A/B

| scenario | error | class | note |
|---|---:|---|---|
| K2.5-tp8ep8-8k2k | 1.244x | improved | baseline=1.439; current_sim=166.117; delta=-0.195 |
| K2.5-tp4ep8dp2-8k2k | 1.136x | unchanged | baseline=1.136; current_sim=156.484; delta=+0.000 |
| K2.5-tp8ep8-8k2k-bt65536 | 1.120x | unchanged | baseline=1.120; current_sim=155.070; delta=+0.000 |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 1.122x | improved | baseline=1.234; current_sim=127.764; delta=-0.112 |
| K2.5-tp8ep8-32k3k | 1.163x | unchanged | baseline=1.163; current_sim=61.024; delta=+0.000 |
| K2.5-tp4ep8dp2-32k3k | 1.056x | unchanged | baseline=1.056; current_sim=56.281; delta=+0.000 |

`diagnostic_only=true valid_for_default=false perf_database=false`; Default AIC stays No-Go.
