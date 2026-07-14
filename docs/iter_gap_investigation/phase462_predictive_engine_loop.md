# Phase462 Step 2a-3e 全预测引擎环

全预测门失败：引擎环结构正确，但仅靠合规输入无法预测目标时序；2c 保持锁定。

| 判卷项 | 全预测 | 门 | 结果 |
|---|---:|---:|---|
| 首调度步精确吻合 | 0.78% | >=85% | 失败 |
| 前 16 步形状吻合 | 6.25% | >=14/16 | 失败 |
| 抢占 / 自抢占 / 重复 victim | 10 / 2 / 0 | 10±2 / 0 / 0 | 失败 |
| EngineCore drain 步吻合 | 0.78% | 诊断读数 | 不入门 |

## 输入边界

| 类别 | 内容 | 用法 |
|---|---|---|
| 预测白名单 | bench 协议、部署配置、PerfDB iteration cost、tokenizer measured primitive、源码引擎环规则 | 预测输入 |
| 目标运行时间戳、首调度步、iteration trace、抢占签名 | Phase462 target run | 预测完成后判卷，禁止回灌 |
| tokenizer 初始 batch | 预测 `[32, 32, 32, 32]`；目标 `[1, 2, 32, 32, 32, 29]` | 目标值只用于解释偏差 |

预测按 bench 的 128 并发初始 cohort 在 `t=0` 提交，并按运行时 `32 requests / 2ms / 1 worker` 与已过门 tokenizer 原语生成输入。若 batch 形态与目标不同，缺的是 bench 提交到 tokenizer queue 的 API 侧暴露时序；该量不在白名单内，不能拿目标时间戳补齐。

## 止损与后续

| 项目 | 结论 |
|---|---|
| Step 2c | `locked`；`structure_correct_prediction_limited` |
| 三个 fail 点可收敛度 | 当前不可识别，不声称符号或收益 |
| 六点符号预测 | 六点均不可识别；不使用失败原型预测方向 |
| Phase461 DP 存量 | ranks=[0, 1]，row_counts={0: 151684, 1: 151644}，timestamped_rows={}；有构成/耗时行但无 per-rank 时序，不能继承 |
| dp2 logging-only 短跑 | 不触发 |
| runtime / PerfDB / validate / gate | 均不改 |
| Default AIC | No-Go |

边界：`diagnostic_only=true`、`valid_for_default=false`、`perf_database=false`。
