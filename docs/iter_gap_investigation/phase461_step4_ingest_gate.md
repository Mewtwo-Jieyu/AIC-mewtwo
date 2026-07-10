# Phase461 Step4 入库前硬门

结论：两个 TP8 新表都未覆盖当前查询包络，且与 Phase454 参考 session 没有任何完全相同的 `(bucket_tokens, decode_batch)` cell。跨 session 可比性无法成立，Step4 在入库前停止；MLA 坏行和 TP8 rows 均不写入 PerfDB，不运行带病 `--ab`。Default AIC 维持 No-Go。

| 场景 | 查询覆盖 | 未覆盖唯一 cell | 新表 cell | 旧 session 精确重叠 | 入库门 |
|---|---:|---:|---:|---:|---|
| K2.5-tp8ep8-32k3k | 82/131 (62.6%) | 46 | 55 | 0 | blocked |
| K2.5-tp8ep8-8k2k-bt65536 | 31/150 (20.7%) | 119 | 82 | 0 | blocked |

| 硬门 | 预先声明 | 实际结果 |
|---|---|---|
| 二维覆盖 | 每个实际 mixed 查询都有完整 token 与 batch 括号 | 两场景均失败 |
| session 可比性 | 至少 5 个完全同 cell，最大相对漂移不超过 15% | 两场景均为 0 个重叠 cell |
| 数据处置 | 任一硬门失败即停止 | 未改 runtime / PerfDB / gate |

下一步不能靠扩大插值、clamp 或邻点替代。需要补一个带校准锚点的 TP8 workload window：同一 session 先重复至少 5 个 Phase454 cell，再覆盖 32k/65k 当前查询包络；或者由用户明确批准改用同 run event busy / iteration wall 作为新的可比性门。
