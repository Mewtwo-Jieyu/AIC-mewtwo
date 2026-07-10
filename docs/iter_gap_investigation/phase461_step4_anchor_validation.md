# Phase461 Step4a-3 TP8 32k 离线锚验证

结论：离线锚门 `pass`，排除漂移行后的 Step4b 解锁门 `pass`。锚点只按三份数据的最小样本数确定，不按延迟接近程度挑选。未改 runtime、PerfDB 或 gate；Default AIC 维持 No-Go。

| 检查 | 值 | 门 | 状态 |
|---|---:|---:|---|
| 三方完全同 cell | 31 | >=5 | pass |
| 选定锚点 | 5 | 5 | pass |
| 每锚最大漂移 | 见下表 | <=10% | pass |
| 漂移候选行 | 1 | 排除并报告 | pass |
| 扣除受漂移行影响后的墙钟覆盖 | 99.27% | >=95% | pass |
| Step4b 解锁 | pass | pass | pass |

| cell | event / 同 session wall | 同 session / Phase458 wall | event / Phase458 wall | 最大漂移 | 样本 event/same/ref | 状态 |
|---|---:|---:|---:|---:|---|---|
| 26/13 | 1.0198 | 1.0078 | 1.0277 | 2.77% | 11/11/11 | pass |
| 147/13 | 1.0223 | 0.9988 | 1.0211 | 2.23% | 26/26/26 | pass |
| 32000/10 | 0.9996 | 1.0000 | 0.9996 | 0.04% | 372/372/372 | pass |
| 32000/11 | 0.9996 | 0.9997 | 0.9993 | 0.07% | 76/65/65 | pass |
| 32000/12 | 0.9984 | 0.9995 | 0.9979 | 0.21% | 49/38/38 | pass |

## 排除行

| cell | event / Phase458 wall | 最大漂移 | 查询影响次数 | 决策 |
|---|---:|---:|---:|---|
| 92/13 | 0.3198 | 69.18% | 18 | 不入库；受其影响的查询不计合格覆盖 |

## Step4b 解锁范围

| 项目 | 决定 |
|---|---|
| MLA batch8/KV32768 | 允许用 Phase461 microbench 真值替换 |
| TP8 32k mixed | 允许 scoped 入库，但明确排除 `92/13` |
| TP8 bt65536 | 继续排除，维持解析链，等待 Phase462 dynamics |
| 六点 A/B | 入库后必须全表运行；8k2k 两点应零移动 |

| 口径 | 说明 |
|---|---|
| event busy / same wall | 检查 graph-outer event 是否遗漏同 session 的 iteration wall |
| same wall / reference wall | 检查 Step3 patched session 与 Phase458 vanilla session 是否漂移 |
| event busy / reference wall | 最终验证拟入库 forward_total 是否可代表验收参考 |
| 选择纪律 | 只按共同样本支持度排序；漂移只用于过门，不参与选点 |
