# Phase462 TP8-bt65536 非 recompute 残差分解

结论：null-only N384 下，sim-only mixed wall 中 recompute 关联占 63.55%，剩余 36.45%（占全部 mixed wall 18.70%）。以下三维是同一残差的并列签名视图，不能彼此相加。`Default AIC=No-Go`。

| 口径 | wall ms | 占比 |
|---|---:|---:|
| 全部 mixed | 126677.573097 | 100.00% |
| sim-only mixed | 64994.187169 | 51.31% |
| sim-only 且含 recompute | 41302.572511 | 63.55% of sim-only |
| sim-only 且无 recompute | 23691.614658 | 36.45% of sim-only |

## Chunk 构成

| 构成 | wall ms | 残差占比 |
|---|---:|---:|
| continued_complete | 180.009987 | 0.76% |
| continued_complete+fresh_complete | 7389.678865 | 31.19% |
| continued_complete+fresh_complete+fresh_partial | 13806.847424 | 58.28% |
| fresh_complete | 587.617603 | 2.48% |
| fresh_complete+fresh_partial | 1727.460780 | 7.29% |

## Serving-state 覆盖

| 覆盖 | wall ms | 残差占比 |
|---|---:|---:|
| all_miss | 23691.614658 | 100.00% |

| miss reason | wall ms | 残差占比 |
|---|---:|---:|
| table_missing | 23691.614658 | 100.00% |

## 调度节奏

| 单步新 admit 数 | wall ms | 残差占比 |
|---|---:|---:|
| 0 | 180.009987 | 0.76% |
| 1 | 1493.116243 | 6.30% |
| 2 | 587.617603 | 2.48% |
| 3 | 904.154320 | 3.82% |
| 5 | 2255.626376 | 9.52% |
| 6 | 2736.781926 | 11.55% |
| 8 | 13806.847424 | 58.28% |
| 9 | 1727.460780 | 7.29% |

| 连续残差步长度 | wall ms | 残差占比 |
|---|---:|---:|
| 1 | 3304.146592 | 13.95% |
| 2 | 13841.933342 | 58.43% |
| 4 | 6545.534724 | 27.63% |

本步仅离线分解，不改 runtime、PerfDB 或 gate；抢占数为 149，用于和 null-only 基线核对。
