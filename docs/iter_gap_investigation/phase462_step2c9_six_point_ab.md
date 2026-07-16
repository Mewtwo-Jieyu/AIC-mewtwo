# Phase462 Step 2c-9 六点 A/B

结论：`completed_regression_requires_separate_case`；当前计分板 `2/6`。engine-loop 候选发生回归，不启用；本轮只评估，不改参数、runtime 或 PerfDB，`Default AIC=No-Go`。

预注册：动态变化应集中在 TP 引擎环；Phase461 已过三点不得倒退；TP8-32k3k 不预期大幅移动。DP 引擎环仍由 runtime 显式拒绝，三个 DP 点保持默认路径，留待 Step 3。

| 场景 | candidate 路径 | baseline error | current error | delta | A/B | 15% 门 |
|---|---|---:|---:|---:|---|---|
| K2.5-tp8ep8-8k2k | engine_loop | 1.132524 | 1.168856 | +0.036332 | regressed | FAIL |
| K2.5-tp8ep8-32k3k | engine_loop | 1.103595 | 1.005709 | -0.097886 | improved | PASS |
| K2.5-tp4ep8dp2-8k2k | deferred_dp_control | 1.041312 | 1.057775 | +0.016463 | regressed | PASS |
| K2.5-tp4ep8dp2-32k3k | deferred_dp_control | 1.173318 | 1.195756 | +0.022438 | regressed | FAIL |
| K2.5-tp8ep8-8k2k-bt65536 | engine_loop | 1.234521 | 1.318934 | +0.084412 | regressed | FAIL |
| K2.5-tp4ep8dp2-8k2k-bt65536 | deferred_dp_control | 1.184544 | 1.200038 | +0.015494 | regressed | FAIL |

baseline：`docs/iter_gap_investigation/phase461_step4b_final_ab.csv` 的 Phase461 最终 `current_*` 计分板。A/B 变化阈值：`0.005`。

已过场景倒退：`["K2.5-tp8ep8-8k2k", "K2.5-tp4ep8dp2-8k2k"]`。当前过门：`["K2.5-tp8ep8-32k3k", "K2.5-tp4ep8dp2-8k2k"]`。

边界：`diagnostic_only=true`、`valid_for_default=false`、`perf_database=false`、`Default AIC=No-Go`。
