# Phase440 non-attention total rows

- verdict: `calibration_gate_failed_do_not_ingest`.
- existing log max decode_batch: 55.
- derived curve rows: 72.
- target decode_batch minimum: 56.
- target bucket_tokens minimum: 7000.
- GPU pure-log collection is skipped while the calibration hard gate is failed.
- Default AIC remains No-Go until A/B validation is clean.

## Calibration

| bucket_tokens | decode_batch | category sum ms | non_attn_total ms | error % | gate |
|---:|---:|---:|---:|---:|---|
| 8000 | 1 | 709.554924 | 1072.601028 | 51.165328 | failed |

## Coverage

| decode_batch | bucket_tokens | steps | latency ms | coverage |
|---:|---:|---:|---:|---|
| 44 | 89 | 1 | 666.077306 | below_target_working_point |
| 47 | 94 | 1 | 41.284341 | below_target_working_point |
| 54 | 108 | 3 | 257.871060 | below_target_working_point |
| 55 | 110 | 2 | 53.338687 | below_target_working_point |
| 55 | 163 | 1 | 73.768505 | below_target_working_point |
| 43 | 259 | 1 | 675.498462 | below_target_working_point |
| 55 | 260 | 1 | 83.025531 | below_target_working_point |
| 54 | 836 | 1 | 712.510013 | below_target_working_point |
| 47 | 1129 | 1 | 847.664569 | below_target_working_point |
| 56 | 1597 | 1 | 301.303742 | below_target_working_point |
| 55 | 2401 | 1 | 248.011028 | below_target_working_point |
| 54 | 3301 | 1 | 676.909494 | below_target_working_point |
| 55 | 3543 | 1 | 329.876739 | below_target_working_point |
| 54 | 3969 | 1 | 879.934376 | below_target_working_point |
| 54 | 4497 | 1 | 905.920869 | below_target_working_point |
| 55 | 5210 | 1 | 443.691406 | below_target_working_point |
| 55 | 5674 | 1 | 666.883779 | below_target_working_point |
| 1 | 8000 | 4 | 1072.601028 | below_target_working_point |
| 2 | 8000 | 2 | 1068.983942 | below_target_working_point |
| 3 | 8000 | 2 | 1069.514959 | below_target_working_point |
| 4 | 8000 | 2 | 1069.975977 | below_target_working_point |
| 5 | 8000 | 2 | 1070.239397 | below_target_working_point |
| 6 | 8000 | 2 | 1071.162818 | below_target_working_point |
| 7 | 8000 | 2 | 1074.311238 | below_target_working_point |
| 8 | 8000 | 2 | 1074.504659 | below_target_working_point |
| 9 | 8000 | 2 | 1075.265392 | below_target_working_point |
| 10 | 8000 | 2 | 1072.781126 | below_target_working_point |
| 11 | 8000 | 2 | 1072.631860 | below_target_working_point |
| 12 | 8000 | 2 | 1077.522594 | below_target_working_point |
| 13 | 8000 | 2 | 1076.953328 | below_target_working_point |
| 14 | 8000 | 2 | 1076.734061 | below_target_working_point |
| 15 | 8000 | 2 | 1076.669795 | below_target_working_point |
| 16 | 8000 | 2 | 1080.420529 | below_target_working_point |
| 17 | 8000 | 2 | 1078.873050 | below_target_working_point |
| 18 | 8000 | 2 | 1081.240572 | below_target_working_point |
| 19 | 8000 | 2 | 1075.843093 | below_target_working_point |
| 20 | 8000 | 2 | 1082.090615 | below_target_working_point |
| 21 | 8000 | 2 | 1081.668136 | below_target_working_point |
| 22 | 8000 | 2 | 1084.020657 | below_target_working_point |
| 23 | 8000 | 2 | 1086.833179 | below_target_working_point |
| 24 | 8000 | 2 | 1084.950700 | below_target_working_point |
| 25 | 8000 | 2 | 1086.358222 | below_target_working_point |
| 26 | 8000 | 2 | 1081.620743 | below_target_working_point |
| 27 | 8000 | 2 | 1084.923264 | below_target_working_point |
| 28 | 8000 | 2 | 1083.515786 | below_target_working_point |
| 29 | 8000 | 2 | 1083.308307 | below_target_working_point |
| 30 | 8000 | 2 | 1093.665829 | below_target_working_point |
| 31 | 8000 | 2 | 1095.278350 | below_target_working_point |
| 32 | 8000 | 2 | 1095.875872 | below_target_working_point |
| 33 | 8000 | 2 | 1085.323474 | below_target_working_point |
| 34 | 8000 | 2 | 1087.446076 | below_target_working_point |
| 35 | 8000 | 2 | 1084.938677 | below_target_working_point |
| 36 | 8000 | 2 | 1086.561279 | below_target_working_point |
| 37 | 8000 | 2 | 1085.023881 | below_target_working_point |
| 38 | 8000 | 2 | 1085.016483 | below_target_working_point |
| 39 | 8000 | 2 | 1088.124085 | below_target_working_point |
| 40 | 8000 | 2 | 1083.731687 | below_target_working_point |
| 41 | 8000 | 102 | 1088.463603 | below_target_working_point |
| 42 | 8000 | 142 | 1085.511348 | below_target_working_point |
| 43 | 8000 | 74 | 1085.421264 | below_target_working_point |
| 44 | 8000 | 12 | 1083.515429 | below_target_working_point |
| 45 | 8000 | 11 | 1084.810152 | below_target_working_point |
| 46 | 8000 | 10 | 1087.799299 | below_target_working_point |
| 47 | 8000 | 10 | 1015.410901 | below_target_working_point |
| 48 | 8000 | 9 | 1016.089170 | below_target_working_point |
| 49 | 8000 | 9 | 987.375105 | below_target_working_point |
| 50 | 8000 | 9 | 963.069930 | below_target_working_point |
| 51 | 8000 | 9 | 938.670309 | below_target_working_point |
| 52 | 8000 | 9 | 938.857356 | below_target_working_point |
| 53 | 8000 | 9 | 938.636625 | below_target_working_point |
| 54 | 8000 | 10 | 685.966115 | below_target_working_point |
| 55 | 8000 | 4 | 636.441506 | below_target_working_point |
