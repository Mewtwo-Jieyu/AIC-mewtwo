# Phase462 Step2a: 抢占首次分歧离线审计

结论：离线数据不足以裁定唯一抢占语义差，Step2c 继续锁住。sim 首次抢占可精确定位到 decode 增长使块账本超容量 1 块；真实侧只能把首次计数增长夹在约 2 秒 metrics 区间，缺 free blocks、申请块数、trigger/victim id 和 victim 队列位置。因此触发 Step2b logging-only 短跑，不允许从 KV usage 百分比反推决策点。Default AIC 维持 No-Go。

## 首次分歧

| 侧 | 可定位结果 | 精度 |
|---|---|---|
| sim | local_iter=944，decode growth，超容量 1 块 | 决策点精确 hook |
| real | preemption counter 首次跃迁区间 2.005s | scrape 区间，非决策时刻 |
| 对比结论 | 只能确认两边都发生抢占，不能判断为何 real 少抢 | blocked |

Phase458 已能排除 prefix reuse：该 session 明确 `enable_prefix_caching=False`；容量真值与 sim 同为 461,200 tokens / 28,825 blocks。剩余候选仍包括块增长账本、申请块口径、victim 选择和 waiting/running 时序，现有日志无法互相区分。

## 符号预测

抢占减少会减少 recompute 工作，静态一阶符号是 sim 吞吐上升。这里只判方向，不预测幅度。

| 场景 | 当前 ratio | 方向 | 15% 风险 |
|---|---:|---|---|
| K2.5-tp8ep8-8k2k | 1.133 | worsen | current_pass_at_risk |
| K2.5-tp4ep8dp2-8k2k | 1.041 | improve_until_crossing | current_pass_not_immediately_at_risk |
| K2.5-tp8ep8-8k2k-bt65536 | 1.235 | worsen | current_fail_worsens |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 1.185 | worsen | current_fail_worsens |
| K2.5-tp4ep8dp2-32k3k | 1.173 | worsen | current_fail_worsens |
| K2.5-tp8ep8-32k3k | 1.104 | improve_until_crossing | current_pass_not_immediately_at_risk |

关键反证：当前三个 fail 点的 sim 吞吐都已高于 real，单独减少抢占会让它们继续变差；抢占修复是语义正确性修复和步构成修复，不是这三个点的直接数值补丁。`tp8-8k2k` 目前虽达标，但 sim 只剩约 1.54% 上升空间就会越过 1.15，Step2c 必须按动态 `--ab` 判卷。

## Step2b 最小观测

选择 `K2.5-tp8ep8-32k3k`：它的 sim/real 抢占比最高。只在 allocate 失败/抢占决策点记录以下字段：decision timestamp、free blocks before、requested blocks、trigger request id、victim request id、victim queue position。logging-only、diagnostic-only、运行行为不变，开销门仍为 2%。

本步未使用 GPU，未改 runtime、PerfDB 或 gate。
