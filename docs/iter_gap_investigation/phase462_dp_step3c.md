# Phase462 DP Step 3c 收口

结论：DP 嵌套 oracle 未闭合，不能从现有三点链批准 route/drain 建模；TP8 的 8宽 mixed 波次在 real 也存在，sim-only cell 的首因是初始 admission 波次相位偏移，不是 chunk 规则错误。`Default AIC=No-Go`。

## DP 嵌套 oracle

官方 `1.200038` 使用单副本结果乘 DP，没有 route/receive/admit 链。它到 O0 的变化只能记为架构口径差，不能归给 route。

| 层 | error | rank 构成全等率 | 同 cell spread | real admit step 命中率 |
|---|---:|---:|---:|---:|
| O0：双副本现状 | 1.089548 | 100.00% | 1.000x | 2.60% |
| O1：+ real rank 序列 | 1.089548 | 99.19% | 1.000x | 2.34% |
| O2：+ route→receive 延迟 | 1.173375 | 99.31% | 1.000x | 0.26% |
| O3：+ admit step 放行 | 1.186323 | 1.82% | 1.000x | 67.97% |
| real | - | 0.00% | 8.540x | 100.00% |

| 增量 | error 变化 | 裁决 |
|---|---:|---|
| official→O0 | -0.110490 | 架构口径差，不属于路由链 |
| O0→O1 | +0.000000 | rank 分配序列单独无解释力 |
| O1→O2 | +0.083827 | 分数移动但 rank 相位仍近全等，不能判 receive 主导 |
| O2→O3 | +0.012948 | 相位明显靠近 real，但 O3 未闭合 |

O3 的 384 个请求中 261 个按目标步 admit，123 个更晚，最大晚 557 个 per-rank schedule step。现有 trace 没有目标步内每请求的 scheduled token/chunk 与完整 scheduler 入参；强行补进目标步会同时改 token budget、容量和队列语义，不再是单段 oracle。因此本步裁决为 `admit_oracle_not_closed_no_mechanism_decision`。

## TP8 8宽 mixed

| 首波 | context requests | context tokens | decode requests |
|---|---:|---:|---:|
| real iteration 0 | 1 | 8000 | 0 |
| sim iteration 1 | 9 | 65536 | 0 |

sim 从 iteration 2 起连续 4 步形成 8-new-request mixed 波次。单步机械链为“剩余 token budget 形成 fresh partial → partial-prefill stop → 下一步 1 continued + 8 fresh”。real 随后同样出现 9 个 context 且 decode 每步 `+8`，说明这套 chunk/budget 机制两侧共有。

真正首分歧是初始波次：real 首步只 admit 1 个 context，sim 首步 admit 9 个；随后 decode 相位相差 7，才产生 real 从未命中的精确 mixed cell。裁决为 `initial_admission_wave_phase_offset`。到达/暴露段仍属于既定部署边界，本步不授权修改 token budget、chunk 规则或默认暴露模型。

## 边界

| 项目 | 状态 |
|---|---|
| runtime | 未改 |
| PerfDB | 未改 |
| gate | 未改 |
| GPU | 未使用 |
| SSH hostkey | 下次 GPU 会话前仍须带外核验 |
| Step 4 | 顺延 |

细表见 `phase462_dp_step3c/phase462_dp_nested_oracle.csv` 与 `phase462_dp_step3c/phase462_tp8_mixed_first_divergence.csv`。
