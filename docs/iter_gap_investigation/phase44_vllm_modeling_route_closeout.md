# Phase 44: vLLM Modeling Route Closeout

## 结论

Phase 44 收口为：vLLM compiled body 这条线当前只能保留 experimental descriptor 和 Go/No-Go gate，不能进入 latency 模型。Phase 44 不跑实验、不写 `PerfDatabase`、不改默认 `cb_sim`。

| 项 | 决策 |
|---|---|
| 默认 AIC | 保持 Phase 4 baseline |
| Phase 41 key | 保留 experimental descriptor |
| Phase 42 perf 口径 | 保留，作为未来数据资格 |
| Phase 43 Go/No-Go | 生效，Phase 44 timing 暂停 |
| latency 模型 | 当前不建 |

## 决策链

| Phase | 决策 | 对 Phase 44 的影响 |
|---|---|---|
| 40 | compiled body 只能进 experimental descriptor，不能进 latency | closeout 的基线 |
| 41 | `VLLMCompiledBodyRuntimeKey` 只表达 shape/topology/comm/MoE key | 保留 key，不加 ms |
| 42 | 定义 clean perf 数据资格和拒绝规则 | 作为后续重开门槛 |
| 43 | MoE WNA16 和 compiled comm 都 No-Go | 不进入 timing smoke |
| 44 | 固化停止原因和重开条件 | 防止 diagnostic 数据误接模型 |

## 当前能保留什么

| 内容 | 状态 |
|---|---|
| `VLLMCompiledBodyRuntimeKey` | 保留，descriptor-only |
| `tp_comm_candidate` | 保留，candidate-only |
| `ep_or_global_comm_candidate` | 保留，candidate-only |
| `unknown_comm_present` | 保留，candidate-only |
| MoE WNA16 runtime shape | 保留，diagnostic-only |
| TRT-LLM 方法论 | 保留，只借模块拆分和 key 纪律 |

## 当前不能做什么

| 禁止项 | 原因 |
|---|---|
| 接默认 latency | 没有 clean perf 数据 |
| 写 `PerfDatabase` | profiler / NCCL trace / sync-probe 都不合格 |
| 给 runtime key 加 ms | 会把 descriptor 变成经验补偿 |
| 用 residual 常数 | 不是物理模型 |
| 继续 Phase 44 timing smoke | Phase 43 两条线都 No-Go |

## 一句话收口

vLLM compiled body 缺口已经被定位到 comm / MoE aggregate 方向，但还没有达到 AIC 建模数据门槛。当前正确状态是：descriptor 保留，perf 线暂停，默认路径不动。
