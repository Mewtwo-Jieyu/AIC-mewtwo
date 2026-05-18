# Phase 45: Modeling Decision Handoff

## 交付结论

vLLM mixed gap 已经定位到 compiled body 里的 comm / MoE aggregate 方向，但没有达到 AIC latency 建模门槛。当前交付状态是：descriptor 可以保留，perf 线暂停，默认 `cb_sim` 不动。

| 项 | 交付判断 |
|---|---|
| 默认 AIC | 保持 Phase 4 baseline |
| Phase 41 key | 保留 experimental descriptor |
| latency 模型 | 不建 |
| perf table | 不写 |
| 后续实验 | 只有满足 Phase 44 条件后才重开 |

## 可以保留

| 内容 | 保留理由 |
|---|---|
| `VLLMForwardDescriptor` | AIC 需要表达 vLLM forward shape |
| `VLLMRuntimeShapeKey` 和子 key | 能表达 attention / KV / wrapper 变量缺口 |
| `VLLMCompiledBodyRuntimeKey` | 能表达 compiled body / comm / MoE diagnostic key |
| `tp_comm_candidate` | Phase 39 trace 可见 `nranks=4` 候选 |
| `ep_or_global_comm_candidate` | Phase 39 trace 可见 `nranks=8` 候选 |
| `unknown_comm_present` | Phase 39 仍有 `nranks=2` unknown |
| MoE WNA16 runtime shape | Phase 36-38 指向 vLLM fused MoE aggregate |
| stopped path 文档 | 防止已排除小模块回流 |

## 不能回流成模型

| 内容 | 决策 |
|---|---|
| slot mapping latency | kernel 小，host 大头是队列等待 |
| logits latency | CUDA event 小量级 |
| KV shape prep | A/B 都是小量级 |
| attention metadata build | 小量级，只是 metadata build，不是 attention execute |
| MLA `forward_mqa` | 小量级，不能解释 mixed gap |
| random-weight MoE timing | fallback + 非真实权重 |
| profiler CUDA 数字 | coverage 不完整，且 profiler 改变时序 |
| NCCL trace `line_count` | 只能归因，不能计时 |
| sync-probe host wait | 显式 sync 改变执行边界 |
| `residual_ms` bucket | empirical evidence，不是物理模型 |

## 重开条件

| 路线 | 必须先满足 |
|---|---|
| MoE WNA16 perf | `tuning_config_loaded=true`，`fallback=false`，权重口径明确 |
| compiled comm perf | 找到 comm-only CUDA event boundary，不靠 profiler / NCCL debug / sync |
| experimental latency | clean timing、单 key 复现、至少一个 shape/topology holdout |
| default AIC | 多配置验证通过，再讨论默认接入 |

## 接手规则

| 如果后续要做 | 先做什么 |
|---|---|
| 继续 vLLM 建模 | 从 Phase 44 reopen conditions 开始，不从 residual 开始 |
| 加 latency 字段 | 停止；descriptor 不能带 ms / residual |
| 重跑 benchmark | 先写 clean perf 口径和 Go/No-Go，不直接跑大矩阵 |
| 对照 TRT-LLM | 只比较建模对象、输入参数、数据来源、查询接口和校准边界 |
| 接默认路径 | 先证明不是 diagnostic-only，再跑默认 validate 和多配置验证 |

## 当前验收口径

| 检查 | 当前记录 |
|---|---|
| 默认 validate | PASS |
| throughput | `1.50x` |
| multi-config | `1.47x` |
| TTFT | `1.79x` |
| Phase 45 范围 | 文档交付，不跑 GPU，不连远端 |

## 一句话给接手人

不要把 Phase 5-44 的诊断数字改造成 latency 常数。当前真正可交付的是 vLLM runtime descriptor 和重开门槛；如果没有 clean perf 数据，就停在 descriptor。
