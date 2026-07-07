# Phase441 steady wall-clock primitive audit

- verdict: `b1_failed_turn_to_b2_or_redefine_with_new_evidence`.
- life_gate: `failed`.
- explained_share: `0.066814`.
- high_batch_gate: `failed`.
- high_batch_max_error_pct: ``.
- No PerfDB rows are ingested by this report.
- GPU pure-log collection remains blocked unless this audit passes.
- Default AIC remains No-Go.

## Audit Points

| bucket | batch | steps | category ms | wall non-attn ms | error % | explained % | unexplained ms | band |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8000 | 1 | 4 | 709.554924 | 1072.601028 | 51.165328 | 6.681355 | 338.789706 | ramp_or_low_batch |

## High Batch

| bucket | batch | steps | error % |
|---:|---:|---:|---:|

## Missing Category Coverage

| bucket | batch | steps | wall non-attn ms | band |
|---:|---:|---:|---:|---|
| 89 | 44 | 1 | 666.077306 | ramp_or_low_batch |
| 94 | 47 | 1 | 41.284341 | ramp_or_low_batch |
| 108 | 54 | 3 | 257.871060 | ramp_or_low_batch |
| 110 | 55 | 2 | 53.338687 | ramp_or_low_batch |
| 163 | 55 | 1 | 73.768505 | ramp_or_low_batch |
| 259 | 43 | 1 | 675.498462 | ramp_or_low_batch |
| 260 | 55 | 1 | 83.025531 | ramp_or_low_batch |
| 836 | 54 | 1 | 712.510013 | ramp_or_low_batch |
| 1129 | 47 | 1 | 847.664569 | ramp_or_low_batch |
| 1597 | 56 | 1 | 301.303742 | ramp_or_low_batch |
| 2401 | 55 | 1 | 248.011028 | ramp_or_low_batch |
| 3301 | 54 | 1 | 676.909494 | ramp_or_low_batch |
| 3543 | 55 | 1 | 329.876739 | ramp_or_low_batch |
| 3969 | 54 | 1 | 879.934376 | ramp_or_low_batch |
| 4497 | 54 | 1 | 905.920869 | ramp_or_low_batch |
| 5210 | 55 | 1 | 443.691406 | ramp_or_low_batch |
| 5674 | 55 | 1 | 666.883779 | ramp_or_low_batch |
| 8000 | 2 | 2 | 1068.983942 | ramp_or_low_batch |
| 8000 | 3 | 2 | 1069.514959 | ramp_or_low_batch |
| 8000 | 4 | 2 | 1069.975977 | ramp_or_low_batch |
| 8000 | 5 | 2 | 1070.239397 | ramp_or_low_batch |
| 8000 | 6 | 2 | 1071.162818 | ramp_or_low_batch |
| 8000 | 7 | 2 | 1074.311238 | ramp_or_low_batch |
| 8000 | 8 | 2 | 1074.504659 | ramp_or_low_batch |
| 8000 | 9 | 2 | 1075.265392 | ramp_or_low_batch |
| 8000 | 10 | 2 | 1072.781126 | ramp_or_low_batch |
| 8000 | 11 | 2 | 1072.631860 | ramp_or_low_batch |
| 8000 | 12 | 2 | 1077.522594 | ramp_or_low_batch |
| 8000 | 13 | 2 | 1076.953328 | ramp_or_low_batch |
| 8000 | 14 | 2 | 1076.734061 | ramp_or_low_batch |
| 8000 | 15 | 2 | 1076.669795 | ramp_or_low_batch |
| 8000 | 16 | 2 | 1080.420529 | ramp_or_low_batch |
| 8000 | 17 | 2 | 1078.873050 | ramp_or_low_batch |
| 8000 | 18 | 2 | 1081.240572 | ramp_or_low_batch |
| 8000 | 19 | 2 | 1075.843093 | ramp_or_low_batch |
| 8000 | 20 | 2 | 1082.090615 | ramp_or_low_batch |
| 8000 | 21 | 2 | 1081.668136 | ramp_or_low_batch |
| 8000 | 22 | 2 | 1084.020657 | ramp_or_low_batch |
| 8000 | 23 | 2 | 1086.833179 | ramp_or_low_batch |
| 8000 | 24 | 2 | 1084.950700 | ramp_or_low_batch |
| 8000 | 25 | 2 | 1086.358222 | ramp_or_low_batch |
| 8000 | 26 | 2 | 1081.620743 | ramp_or_low_batch |
| 8000 | 27 | 2 | 1084.923264 | ramp_or_low_batch |
| 8000 | 28 | 2 | 1083.515786 | ramp_or_low_batch |
| 8000 | 29 | 2 | 1083.308307 | ramp_or_low_batch |
| 8000 | 30 | 2 | 1093.665829 | ramp_or_low_batch |
| 8000 | 31 | 2 | 1095.278350 | ramp_or_low_batch |
| 8000 | 32 | 2 | 1095.875872 | ramp_or_low_batch |
| 8000 | 33 | 2 | 1085.323474 | ramp_or_low_batch |
| 8000 | 34 | 2 | 1087.446076 | ramp_or_low_batch |
| 8000 | 35 | 2 | 1084.938677 | ramp_or_low_batch |
| 8000 | 36 | 2 | 1086.561279 | ramp_or_low_batch |
| 8000 | 37 | 2 | 1085.023881 | ramp_or_low_batch |
| 8000 | 38 | 2 | 1085.016483 | ramp_or_low_batch |
| 8000 | 39 | 2 | 1088.124085 | ramp_or_low_batch |
| 8000 | 40 | 2 | 1083.731687 | ramp_or_low_batch |
| 8000 | 41 | 102 | 1088.463603 | ramp_or_low_batch |
| 8000 | 42 | 142 | 1085.511348 | ramp_or_low_batch |
| 8000 | 43 | 74 | 1085.421264 | ramp_or_low_batch |
| 8000 | 44 | 12 | 1083.515429 | ramp_or_low_batch |
| 8000 | 45 | 11 | 1084.810152 | high_batch |
| 8000 | 46 | 10 | 1087.799299 | high_batch |
| 8000 | 47 | 10 | 1015.410901 | high_batch |
| 8000 | 48 | 9 | 1016.089170 | high_batch |
| 8000 | 49 | 9 | 987.375105 | high_batch |
| 8000 | 50 | 9 | 963.069930 | high_batch |
| 8000 | 51 | 9 | 938.670309 | high_batch |
| 8000 | 52 | 9 | 938.857356 | high_batch |
| 8000 | 53 | 9 | 938.636625 | high_batch |
| 8000 | 54 | 10 | 685.966115 | high_batch |
| 8000 | 55 | 4 | 636.441506 | high_batch |

## B2 Trigger

- B1 does not pass the offline life gate.
- The overlap delta is not explained by log-visible peer wait, Phase433 host-gap bound, and attention-error bound.
- The high-batch segment has iteration-log candidates but lacks complete serving-state category coverage, so the steady convergence gate cannot be proven.
- A pure-log GPU run may add coverage, but it cannot fix the failed overlap explanation gate by itself.
- B2 must time graph outer boundaries only; graph-internal CUDA events are invalid for replay timing and must not be used as the acceptance signal.
