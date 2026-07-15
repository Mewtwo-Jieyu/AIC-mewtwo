# Phase462 DP Step 3 null-only 分诊

结论：当前 `dp2-bt65536` 默认路径的“单副本模拟后乘 DP”结构不成立；现成双副本锁步虽把分数推入 15%，但 per-rank 相位签名失败，仍不得进入默认路径。下一步必须先补 request 到 rank 的 receive/admit 可见性，不能按分数倒推参数。`Default AIC=No-Go`。

## DP2-bt65536 重基线

| 路径 | sim tok/s/GPU | error | 15% 分数门 | 相位门 |
|---|---:|---:|---|---|
| 当前默认：单副本乘 DP | 131.404059008 | 1.200038049 | FAIL | 不具备 per-rank 构成 |
| 离线双副本锁步候选 | 119.907410606 | 1.095045741 | PASS | FAIL |

## 相位与成本签名

| 指标 | real | 锁步候选 |
|---|---:|---:|
| 含 prefill 的共同局部步 | 85 | 123 |
| rank 构成逐步全等率 | 0.000000000 | 1.000000000 |
| phase 类型全等率 | 0.200000000 | 1.000000000 |
| 首次 prefill rank 比 | 8.192000000 | 1.000000000 |
| mixed 步 rank0/rank1 | 53/47 | 122/122 |
| real 同 cell 最大 rank spread | 8.540214162 | 无法表达 |

`1.095` 只来自现有锁步计算方式，候选两个 rank 始终同构，不能解释 real 的相位分裂与 `8.54x` 同 cell spread，所以不得进入默认路径。

## TP8 大 bucket 复查

| 指标 | null-only 新基线 | Phase461 |
|---|---:|---:|
| mixed 步 | 173 | 185 queries |
| reference 未观察 wall 占比 | 0.513067827 | 0.4215 |
| recompute 关联占比 | 0.635481022 | 0.8541 |
| 原共根门 | 未过 0.70 | 已过 0.70 |

null block 改变了轨迹，但没有消掉 sim-only 大 bucket；当前超过一半 mixed wall 仍在 Phase458 reference 未观察 cell 中。recompute 关联已降到原 70% 硬门以下，不能继续把剩余问题只归为抢占过发。


## 下一门

| 项目 | 裁决 |
|---|---|
| 下一动作 | `logging_only_dp_route_visibility_required` |
| 需要的新证据 | logging-only：request id、目标 DP rank、router 可见 waiting/running、EngineCore receive、首次 admit 的同 ID 时间链 |
| 现有数据是否够定结构缺口 | 够；已证明默认路径与锁步候选都不能复现 rank 相位 |
| 是否现在改 runtime | 否 |
| 是否现在改 PerfDB | 否；`8.54x` 行继续 diagnostic-only |
| 是否现在跑 GPU | 否；采集设计与成本需单独审批 |
| Default AIC | No-Go |
