# Phase462 Step 2c-5a release-chain backtrace

结论：两条释放链落入预注册分支 **(b) `workload_exposure_boundary`**。它们是不同最早祖先，但都回溯到不依赖前序释放的初始 workload 集合；最早可见集合分叉在 schedule `1`。因此不能把残差修成 future 提前回填，也不能用 Step 2a-3f 的聚合稳态结论替代事件级硬门。

| 目标请求 | 释放链 | 最早祖先 | 滞后步数 | 转折类型 |
|---:|---|---:|---:|---|
| 27 | 27 <- 13 <- 0 | 0 | 1 | ungated_lagged_admission |
| 124 | 124 <- 110 <- 96 <- 82 <- 69 | 69 | 1 | ungated_lagged_admission |

| 目标 | 被 admit 请求 | gating 释放请求 | 释放输入 real/sim | admit real/sim | 释放滞后 | 继承滞后 | 生命周期增量 | 向前携带滞后的 admit |
|---:|---:|---:|---|---|---:|---:|---:|---|
| 27 | 27 | 13 | 1476/1477 | 1476/1477 | 1 | 1 | +0 | resume 1202/1203 |
| 27 | 13 | 0 | 1202/1203 | 1202/1203 | 1 | 1 | +0 | new 1/2 |
| 124 | 124 | 110 | 10324/10326 | 10324/10326 | 2 | 2 | +0 | new 8893/8895 |
| 124 | 110 | 96 | 8893/8895 | 8893/8895 | 2 | 2 | +0 | new 7445/7447 |
| 124 | 96 | 82 | 7445/7447 | 7445/7447 | 2 | 1 | +1 | resume 7150/7151 |
| 124 | 82 | 69 | 7150/7151 | 7150/7151 | 1 | 1 | +0 | new 5756/5757 |

回溯规则只接受唯一释放依赖：目标 admit 步的 real/sim 必须各有且只有一个完成释放，并且 request ID 相同；每一跳的 release lag 必须等于下游 admit lag。前序请求如经历 resume，优先沿与 release lag 相同的 admission 回溯；若没有完全相同的 lag，只允许唯一一种正 lag 继续向前，并把差值记为生命周期增量。没有正 lag admission 才是释放转折点。任何多候选或不一致都会让分析器直接失败。

request 82 在 resume 时继承 1 步滞后，release 时变成 2 步，因此给 request 124 链额外放大 1 步；这不是最早起点。N128/C128 下 request 0 与 69 都属于初始 128 请求，二者首次 admit 均不依赖完成释放，最早可见集合差仍是 schedule 1 的 workload 暴露。

## 与 Step 2a-3f 对账

| 口径 | 结论 |
|---|---|
| 聚合稳态指标 | t=0 与 oracle 暴露近似，不否定吞吐级建模边界 |
| 当前事件级门 | 单步平移会改变 trigger/victim 归因，不能由聚合近似推出 `自抢占=0` |
| 是否矛盾 | 不矛盾，但两种口径不能互相替代 |
| 事件流覆盖 | real `12032` / sim `12031` schedule；real 末尾仅多 1 个空调度步 |
| 当前状态 | `blocked_event_gate_reaches_declared_workload_boundary` |

## 下一确认点

分支 (b) 只能单独评审二选一：最小化建模 workload 暴露过程，或用更长 real 窗口取得事件级证据后重议门口径。当前不选择、不改门。

边界：`diagnostic_only=true`、`valid_for_default=false`、`perf_database=false`、未改 runtime/gate/PerfDB、未跑 `--ab`、`Default AIC=No-Go`。红测 2167/10616 保留。
