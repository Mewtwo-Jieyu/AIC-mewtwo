# Phase466 v3.3 stock probe gate closeout

结论：overhead gate 为 `INCONCLUSIVE`，stock iteration probe 路线关闭。
正式 N512 场景未运行，三类建模候选保持 unresolved，不选择 Phase467。

## Evidence identity

| 项 | 值 |
|---|---|
| remote artifact | `/mnt/shared-storage-user/zhaojieyu/backup/aic/phase466_v33_full_rank_timing_0765bad7_20260720T133525Z` |
| `overhead_gate.json` SHA256 | `e2fe782bfee3e9965187047c36cae08fc005f5885fc99d75eac911fd5b30b950` |
| `phase466_result.json` SHA256 | `bdd1b94d576f46eb6bc18fd6060560bad7c470b3cf0cbbe03e7b83515d723122` |
| execution manifest SHA256 | `bf562c5a649106acb12f1f981cb8d60a637ee079d02c7f3666507b77ba6c7609` |
| formal directory | absent |
| terminal status | `STOPPED_BEFORE_FORMAL` |
| readiness | `diagnostic_only=true`; `valid_for_default=false`; `perf_database=false`; `Default AIC=No-Go` |

## Paired results

| Pair | Order | OFF tok/s | ON tok/s | ON/OFF | Second/first |
|---|---|---:|---:|---:|---:|
| pair-01 | OFF/ON | 841.706552 | 858.560818 | 1.020024 | 1.020024 |
| pair-02 | ON/OFF | 891.517223 | 815.501999 | 0.914735 | 1.093213 |
| pair-03 | OFF/ON | 770.002289 | 887.009200 | 1.151957 | 1.151957 |
| pair-04 | ON/OFF | 874.299897 | 827.842114 | 0.946863 | 1.056119 |
| pair-05 | OFF/ON | 834.951382 | 860.888261 | 1.031064 | 1.031064 |
| pair-06 | ON/OFF | 845.904458 | 856.954282 | 1.013063 | 0.987106 |

| 统计量 | 结果 |
|---|---:|
| OFF mean | 843.063633 |
| OFF sample CV | 4.958% |
| ON mean | 851.126113 |
| ON sample CV | 3.011% |
| geometric mean ON/OFF | 1.010241 |
| 90% CI | [0.946082, 1.078752] |
| equivalence bounds | [0.980000, 1.020000] |

5/6 pair 的第二次运行更快，主要噪声来自服务重启或运行顺序。均值接近 1 不代表等价，**不能解释为 probe 开销约 1%**。

## Closed route and remaining boundary

当前 ON rank logs 只保留为失败测量方法的诊断记录，不可用于模型、PerfDatabase 或 readiness。stock probe 不再重跑；只有单独评审通过的新低扰动测量设计才能重启建模取证。

未解决候选：

- `schedule_merged_batch_composition`
- `iteration_cost_serving_state_coverage`
- `dp_rank_synchronization_asymmetry`

三个候选均保持 `unresolved`；本 closeout 不选择、实现或预占 Phase467。
