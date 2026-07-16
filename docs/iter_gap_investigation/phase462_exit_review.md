# Phase462 Step 2c-5b 出口评审

结论：推荐出口 B：现有 12k 窗口内滞后不是单调增长，且两个预测自抢占都可确定性回溯到已声明的 workload 暴露边界；无需先补长 real 窗口。

## 滞后放大量化

| 项目 | 结果 |
|---|---|
| admit 事件数 | 138 |
| lag 分布 | lag=0: 126, lag=1: 9, lag=2: 3 |
| 首个 / 末个 lag | 1 / 0 |
| 最小 / 最大 lag | 0 / 2 |
| 上升 / 下降 / 不变 | 9 / 10 / 118 |
| 生命周期 lag 变化 | request 82: 1 -> 2 (+1), request 124: 2 -> 1 (-1) |
| 12k 窗口分类 | `bounded_oscillation` |
| 是否需更长 real 窗口 | 不需要 |

分类规则没有拟合阈值：全程只升不降且至少升一次记 `monotonic_growth`；全程不反弹、持续下降并回到 0 记 `convergent_to_zero`；其余记 `bounded_oscillation`。这是当前 N128/C128/ISL32k/OSL1200 的观测结论，不外推到无限时长。

完整轨迹见 `phase462_admit_lag_trajectory.csv`。

| 请求 | 继承 lag -> release lag | 放大量 | 触发前步 real/sim | free blocks real/sim | running real/sim | waiting real/sim | running/waiting 顺序相同 | computed real/sim | placeholder real/sim |
|---:|---|---:|---|---|---|---|---|---|---|
| 82 | 1 -> 2 | +1 | 7444/7446 | 567/566 | 14/14 | 32/32 | True/True | 33199/33199 | 1/1 |

完整放大清单见 `phase462_lag_amplification.csv`。唯一放大点两侧均有 566 个以上 free blocks，running/waiting 数量与 ID 顺序相同，目标 computed/placeholder/block 状态也相同；没有容量耗尽或队列重排证据。这里把“触发条件”限定为可观测事实，不把伴随状态硬写成因果规则。

## 双出口评审

| 标准 | 出口 A：最小暴露建模 | 出口 B：两段式门 |
|---|---|---|
| 可迁移性 | 差。client/HTTP/api-server 调度是部署特定输入，2a-3c/3e 已证明现有原语不能预测首批暴露 | 好。不拟合部署时序，只把它保留为显式建模边界 |
| 捕虫能力 | 取决于暴露模型是否准确，错误拟合会掩盖 scheduler 问题 | 保留严格 oracle 语义门；预测路径只接受完整链归因 |
| 有界性 | 仅在滞后增长时才有必要 | 当前分类 `bounded_oscillation` |
| GPU 成本 | 需要跨部署重采和验证 | 当前证据足够时为 0；增长才补长窗口 |
| 本轮建议 | 不推荐 | `b_two_part_gate` |

现有证据：oracle schedule 精确对齐=`True`，oracle 自抢占=`0`，预测自抢占请求=`[27, 124]`，admit 对齐后的 computed 语义全等=`True`，两事件链归因=`完整`。

## 门修正案草案

本草案只替换“预测变体稳态自抢占必须等于 0”这一条；抢占总数 `10+/-2`、重复 victim `0`、oracle 回归锚和其他既有门保持不变。

1. **语义门（严格）**：oracle 暴露变体在共同的全部非空 schedule 上必须逐步对齐，稳态自抢占必须为 `0`。任何 scheduler、容量、队列顺序、sampled/computed/placeholder 或 victim 语义差异都直接失败。
2. **归因门（严格）**：预测变体出现自抢占时，每个事件必须同时满足：
   - real/sim request ID 和 admission 类型可一一配对；
   - 从目标事件沿唯一的 `release -> admit` 依赖逐跳回溯，real/sim 的 release lag 必须等于下游 admit lag；
   - admit 对齐后，该请求的 computed/placeholder/block 相位逐步全等；
   - 最早分叉早于首个 completion release，根请求属于初始 workload，分类必须是 `workload_exposure_boundary`；
   - 任一多候选、缺事件、计数不等、链中断或其他根因都直接失败。
3. **禁止事项**：不按当前自抢占数量、request ID、step 编号或最大 lag 裁剪门；不把 oracle 时间戳喂入交付路径；不新增启发式补偿。
4. **证据失效条件**：协议、arrival 原语、queue depth、scheduler/容量语义或 trace schema 任一变化，当前边界归因失效，必须重新取证。

草案状态：`gate_amendment_ready_for_separate_approval`。它尚未生效，必须单独确认后才能改门。

边界：`diagnostic_only=true`、`valid_for_default=false`、`perf_database=false`、未改 runtime/gate/PerfDB、未跑 GPU、未跑 `--ab`、`Default AIC=No-Go`。
