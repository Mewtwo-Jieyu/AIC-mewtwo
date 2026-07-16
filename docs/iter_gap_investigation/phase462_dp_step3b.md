# Phase462 DP Step 3b 路由取证与 TP8 残差分解

结论：DP 路由链判为 `H-mixed`。不对称既在 route 时间序列中出现，per-rank admit cadence 也不全等；现有证据不能只选 H-route 或 H-drain。TP8-bt65536 的非 recompute 残差占 sim-only mixed wall 的 36.45%，主签名是 8-request fresh/continued 混合 chunk、serving-state 行全 miss 和连续 2–4 步聚簇。

## DP 路由链

协议为 dp2-bt65536、N=512、C=128、ISL=8000、OSL=2000、单次 logging-only real serving。

| 指标 | rank 0 | rank 1 |
|---|---:|---:|
| route 请求数 | 257 | 255 |
| route→receive 中位延迟 | 3082.740 ms | 2275.193 ms |
| receive→admit 中位延迟 | 188745.749 ms | 189004.433 ms |

| 冻结判据 | 结果 |
|---|---|
| route 总量严格 256/256 | FAIL |
| route 逐请求严格交替 | FAIL |
| 最大连续同 rank 路由 | 14 |
| 511 个相邻决策中的 rank 切换 | 367 |
| score 并列路由 | 36 |
| 选中 rank 的 score 始终为最小值 | PASS |
| 两 rank admit gap 分布逐项全等 | FAIL |
| 机械裁决 | `H-mixed` |

route 总量只差 2 个请求，但时间序列存在明显聚簇；route 后两侧又形成不同 admit gap 分布。因此不能把 8.54x 的跨 rank spread 归结为单一锁步成本，也不能在本步直接选择“结构建模”或“scoped 测量”。score 陈旧、并发客户端决策和 per-rank drain 的相对贡献仍需单独机制评审。

## TP8 非 recompute 残差

| 口径 | wall ms | 占比 |
|---|---:|---:|
| 全部 mixed | 126677.573 | 100.00% |
| sim-only mixed | 64994.187 | 全部 mixed 的 51.31% |
| sim-only recompute | 41302.573 | sim-only 的 63.55% |
| sim-only 非 recompute | 23691.615 | sim-only 的 36.45% |

非 recompute 残差中，`continued_complete+fresh_complete+fresh_partial` 占 58.28%，单步新 admit=8 同样占 58.28%；serving-state 覆盖 100% 为 `table_missing`。这三项是同一批残差的并列视图，不能相加。现阶段应把它作为“大批 chunk 构成 + scoped 行缺失 + 调度聚簇”的复合残差，不应继续塞回 recompute 共根。

## 卫生

| 检查 | 结果 |
|---|---|
| 请求完整性 | 512/512，失败 0 |
| logging 开销 | 1.692%，≤2% PASS |
| trace footer / event count | PASS |
| vLLM 源码哈希恢复 | PASS |
| 残留进程 | 0 |
| runtime / PerfDB / gate 改动 | 0 |
| Default AIC | No-Go |

本步停在机制证据，不实施路由模型、scoped 测量行或任何参数补偿。后续方案需单独确认。
