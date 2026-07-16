# Phase462 Step 2c-9 二次修正门重跑

结论：`pass`。本门只决定六点 A/B 是否可运行；默认路径和 `Default AIC=No-Go` 不变。

| 条款 | 结果 | 目标 |
|---|---:|---:|
| oracle 全步语义 | 12031 步精确 | 12031 步精确 |
| oracle 稳态自抢占 | 0 | 0 |
| 短跑抢占 | 10 | 10+/-2 |
| 短跑重复 victim | 0 | 0 |
| 预测自抢占机械归因 | 2 | 全部通过 |

| step | request | 路径 | 判据 | 结果 |
|---:|---:|---|---|---|
| 2167 | 27 | strict | attributed_to_workload_exposure | PASS |
| 10616 | 124 | compound exception | verified_compound_preemption_exception | PASS |

Step2c-8 证据 SHA256：`524d44ca4df36835e3d0a041b6591ba113d329bac8ca61f9b037dfc3401e8af9`。

三修禁止条款已生效：同一门不得第三次逐案修正；再出现缺口时，直接进入长 real 窗口止损核验或按实现缺口处理。

`six_point_ab_allowed=True`，`default_enable_allowed=False`。边界：`diagnostic_only=true`、`valid_for_default=false`、`perf_database=false`、`Default AIC=No-Go`。
