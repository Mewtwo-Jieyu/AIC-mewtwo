# Phase 40: Next Modeling Gate

## 结论

下一阶段不能直接建 latency。只有先定义干净 perf 数据口径，才允许进入 experimental 模型。

| 方向 | 当前状态 | 下一步条件 |
|---|---|---|
| runtime descriptor | 可以继续 | 只接字段，不接 ms |
| compiled comm model | 不能开始 | 需要稳定通信 perf 采集口径 |
| MoE WNA16 model | 不能开始 | 需要真实权重、真实 tuning config、非 profiler 数据 |
| forward envelope model | 不能开始 | 需要不改变执行语义的测量边界 |
| 默认 AIC | 不动 | 需要跨 shape/topology/model 验证 |

## 最低门槛

| 门槛 | 标准 |
|---|---|
| 源码语义 | 能指出 vLLM 实际执行路径和模块边界 |
| runtime key | 字段能完整表达 phase、shape、topology、compiled path、comm、MoE |
| perf 数据口径 | 不依赖 profiler ms、NCCL debug line count、sync-probe host wait |
| 可复现 | 同一 key 多次采集稳定 |
| 可泛化 | 至少跨 shape 或 topology holdout 不崩 |
| 默认安全 | `valid_for_default=false` 在升级前必须保持 |

## 可选后续路线

| 路线 | 做什么 | 不做什么 |
|---|---|---|
| Phase 41 experimental key interface | 把 `VLLMRuntimeKey` 接到 diagnose/validate experimental 输出 | 不改 latency |
| Phase 42 perf 口径设计 | 设计 compiled comm / MoE aggregate 的干净采集方法 | 不用 profiler/NCCL debug 数字 |
| 停在 Phase 40 | 归档证据链，保持默认 AIC 不动 | 不继续采样调参 |

## 数据资格判定

| 数据 | 能否建模 | 原因 |
|---|---|---|
| Phase 36 profiler family ms | 不能 | coverage 不完整，时序被 profiler 改变 |
| Phase 39 NCCL trace | 不能 | 只能归因 group，不是耗时 |
| Phase 33 sync-probe | 不能 | 显式 sync 改变异步执行边界 |
| Phase 25 MoE timing | 不能 | random weight + fallback |
| 未来干净 module benchmark | 待定 | 需要先证明边界干净 |

## 升级规则

| 从 | 到 | 条件 |
|---|---|---|
| diagnostic descriptor | experimental latency | 有干净 perf 口径和单 key 复现 |
| experimental latency | default AIC | 跨 shape/topology/model 验证通过 |
| candidate comm key | communication model | 能区分 TP / EP / global 且有稳定 timing |
| candidate MoE key | MoE model | 真实 tuning config + loaded weight 或等价稳定模块数据 |

## 当前决策

| 问题 | 答案 |
|---|---|
| 是否继续 Phase 40 后直接建模型 | 否 |
| 是否可以做 Phase 41 key interface | 可以，但只输出 descriptor |
| 是否可以做 Phase 42 perf 设计 | 可以，但先写口径，不跑大矩阵 |
| 是否修改默认 `cb_sim` | 否 |
