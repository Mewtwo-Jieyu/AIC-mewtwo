# Phase462 DP Step 3c 嵌套 oracle 归因

结论：官方 `1.20` 路径没有 DP route 链，不能机械拆成 route/receive/admit。以下 O1-O3 只在双副本 O0 母体内逐层注入；官方→O0 单列为架构口径差。裁决为 `admit_oracle_not_closed_no_mechanism_decision`，`Default AIC=No-Go`。

| 层 | error | rank 构成全等率 | 同 cell spread | real admit step 命中率 |
|---|---:|---:|---:|---:|
| O0 | 1.089548 | 100.00% | 1.000x | 2.60% |
| O1 | 1.089548 | 99.19% | 1.000x | 2.34% |
| O2 | 1.173375 | 99.31% | 1.000x | 0.26% |
| O3 | 1.186323 | 1.82% | 1.000x | 67.97% |
| real | - | 0.00% | 8.540x (`(6486, 15)`) | 100.00% |

| 归属段 | 对照 | error 增量 |
|---|---|---:|
| official_to_multi_replica_architecture | official→O0 | -0.110490 |
| route_rank_sequence | O0→O1 | +0.000000 |
| route_to_receive_delay | O1→O2 | +0.083827 |
| receive_to_admit_step | O2→O3 | +0.012948 |

`official_to_multi_replica_architecture` 不属于路由链，禁止计入 route 收益。O2 注入的是逐请求实测 `route→receive` 延迟；O3 仅按真实 per-rank `schedule_seq` 放行请求，没有强改 scheduler 输出。若 O3 admit 命中率不足 100%，本步不得宣称链外无残余。

| O3 admit 闭合审计 | 数值 |
|---|---:|
| exact | 261 |
| early | 0 |
| late | 123 |
| median lag steps | 0.0 |
| max lag steps | 557 |

当前 trace 只有首次 admit step，没有该步每请求 scheduled token/chunk 和完整 scheduler 入参。强行把剩余请求塞进目标步会同时改 token budget、容量和队列语义，不再是单段 oracle；因此 O3 不闭合时只能停线，不能现场补齐。

本步全离线，不改 runtime、PerfDB 或 gate；O0 与内置双副本路径逐字数值一致。
