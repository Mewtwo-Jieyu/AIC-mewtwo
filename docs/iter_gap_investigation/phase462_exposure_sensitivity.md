# Phase462 Step 2a-3f 暴露时序止损评估

稳态对暴露爬坡不敏感；2c 可进入‘引擎环 + t=0 workload 暴露’方案评审。

| 场景 | 相位 TV | 抢占率 | 重算量 | 巨 bucket | 稳态吞吐 | 判决 |
|---|---:|---:|---:|---:|---:|---|
| tp8_32k3k_short | 3.578e-05 / 0.01474 | 0 / 0 | 0.05469 / 0.5105 | 不适用 | 0.888 / 43.14 | A≈B |
| tp8_bt65536 | 0.00046 / 0.01383 | 0 / 0 | 0.2734 / 3.587 | 0 / 0 | 0.3753 / 3.243 | A≈B |

表中每格为 `A/B 差值 / 预注册噪声带`。稳态窗口取第一次请求完成与最后一次首次 prefill 完成之间的状态区间：容量充足时是全 admission 平台，KV 受限时是 waiting 非空的替换平台；没有固定毫秒或固定步数。原先只接受前一种顺序，首次执行在产生 A/B 结果前 fail closed，本报告保留该纠错。相位使用有限样本带；抢占和重算使用同 request 配对差的 95% 单元波动带；巨 bucket 使用墙钟加权有效样本数；吞吐按源码 16-token KV 周期配对分块后计算 95% 块间波动。这里比较的是实际波动尺度，不是随样本数缩小的均值置信区间。采集器 on/off 差只作包含 ramp 的全程吞吐诊断，不冒充重复性噪声。
巨 bucket 的 `0/0` 只证明两种暴露输入之间不敏感；本原型没有命中 Phase458 的目标 cell，不能替代 N=512 动态验收。bt65536 的全程吞吐差超出 on/off 包络，但该读数包含 ramp，只作为已声明边界记录。

## 两个暴露变体

| 场景 | A: t=0 workload + 源码 tokenizer | B: oracle 暴露 | 自抢占 A/B |
|---|---|---|---:|
| tp8_32k3k_short | `[32, 32, 32, 32]` | `[1, 2, 32, 32, 32, 29]` | 2/2（{'ramp': 1, 'steady': 1} / {'ramp': 1, 'steady': 1}） |
| tp8_bt65536 | `[32, 32, 32, 32]` | `[1, 1, 32, 32, 32, 30]` | 1/0（{'steady': 1} / {}） |

Oracle 时间戳只用于本敏感度实验，不得进入 runtime、PerfDB 或交付路径。暴露过程正式归入 workload 规格；本步不再继续拟合 API 客户端到 tokenizer queue 的过程。

## 六点符号预测

| 场景 | 预测方向 | 状态 |
|---|---|---|
| K2.5-tp8ep8-8k2k | `worsen` | `pending_dynamic_ab` |
| K2.5-tp4ep8dp2-8k2k | `improve_until_crossing` | `pending_dynamic_ab` |
| K2.5-tp8ep8-8k2k-bt65536 | `worsen` | `pending_dynamic_ab` |
| K2.5-tp4ep8dp2-8k2k-bt65536 | `worsen` | `pending_dynamic_ab` |
| K2.5-tp4ep8dp2-32k3k | `worsen` | `pending_dynamic_ab` |
| K2.5-tp8ep8-32k3k | `improve_until_crossing` | `pending_dynamic_ab` |

方向只来自已登记的‘引擎环减少抢占、sim 吞吐上移’假设，不是验收结果；必须由 2c 动态 `--ab` 逐点证伪。

## 2c 范围

| 项目 | 结论 |
|---|---|
| Step 2c | `candidate` |
| 范围 | `engine_loop_with_t0_workload_exposure` |
| ramp 边界 | `first_schedule_trajectory_not_modeled` |
| 自抢占验收 | 2c 仍须按稳态分区判到 0；本步只裁决 workload 暴露敏感度 |
| runtime / PerfDB / validate / gate | 本步均未改 |
| Default AIC | No-Go |

边界：`diagnostic_only=true`、`valid_for_default=false`、`perf_database=false`。
