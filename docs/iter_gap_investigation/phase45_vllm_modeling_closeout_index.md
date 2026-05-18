# Phase 45: vLLM Modeling Closeout Index

## 结论

Phase 45 是交付包入口，不是新实验。默认 AIC 继续保持 Phase 4 baseline；Phase 41 的 runtime key 只保留为 experimental descriptor；Phase 43/44 的 No-Go 和重开门槛继续生效。

| 项 | 当前状态 |
|---|---|
| 默认 `cb_sim` | 不动 |
| latency 模型 | 不建 |
| `PerfDatabase` | 不写 |
| runtime key | descriptor-only |
| 后续实验 | 暂停，除非满足 Phase 44 重开条件 |

## 交付包入口

| 文档 | 用途 |
|---|---|
| `phase45_vllm_modeling_closeout_index.md` | Phase 5-44 总入口和阅读顺序 |
| `phase45_artifact_inventory.md` | docs / scripts / tests / CSV / log 产物清单 |
| `phase45_modeling_decision_handoff.md` | 接手决策、禁止回流项、重开条件 |
| `phase40_compiled_body_modeling_decision.md` | Phase 27-39 证据链和 compiled body 建模决策 |
| `phase41_compiled_body_runtime_key_interface.md` | `VLLMCompiledBodyRuntimeKey` experimental 接口 |
| `phase42_perf_data_rejection_rules.md` | 哪些数据没有资格进入后续 latency |
| `phase43_perf_feasibility_go_nogo.md` | MoE WNA16 / compiled comm perf No-Go |
| `phase44_reopen_conditions.md` | 未来重开实验和模型的最低门槛 |

## Phase 5-44 证据链

| 阶段 | 关键结论 | 最终处理 |
|---|---|---|
| Phase 5-6 | mixed residual 和 runtime shape/topology 有关；bucket 有解释力但不是物理模型 | 归档为 evidence，不接默认 |
| Phase 7-11 | vLLM forward descriptor、runtime shape key、子 key 能表达缺失变量 | 只保留 descriptor，不建 latency |
| Phase 12-18 | KV shape prep、attention metadata、MLA kernel 都是小量级 | 停止小模块路线 |
| Phase 19-25 | MoE 单层边界可测，但 random-weight + fallback 只能诊断 | 不写 perf 表，不外推 60 层 |
| Phase 26-35 | `cuda_forward` 主体被定位到 Kimi compiled body | 保留 forward envelope 诊断 |
| Phase 36-39 | compiled body 主方向是 NCCL/comm + MoE WNA16 aggregate，但证据不足以计时建模 | 只保留 comm candidate 和 MoE diagnostic key |
| Phase 40-44 | runtime key 可以留；clean perf 数据门槛未满足 | 路线收口，默认路径不动 |

## 当前能保留什么

| 内容 | 用法 |
|---|---|
| `VLLMForwardDescriptor` | 表达 vLLM forward execution shape |
| runtime shape subkeys | 区分 attention / KV / wrapper 机制变量 |
| `VLLMCompiledBodyRuntimeKey` | 表达 compiled body / comm / MoE diagnostic key |
| `tp_comm_candidate` | 只表示 NCCL trace 可见 TP 候选 |
| `ep_or_global_comm_candidate` | 只表示跨 DP/EP/global 候选 |
| MoE WNA16 shape key | 只表示 vLLM fused MoE compiled aggregate shape |
| stopped paths | 防止后续重复挖已排除模块 |

## 当前不能入模什么

| 内容 | 原因 |
|---|---|
| `residual_ms` bucket | 不是物理模型 |
| profiler CUDA 数字 | profiler 改变时序，coverage 不完整 |
| NCCL trace `line_count` | 只能做归因，不是耗时 |
| sync-probe host wait | 显式 sync 改变执行边界 |
| random-weight MoE timing | fallback + 非真实权重 |
| slot mapping host wall | 已证明主要是队列等待，不是 kernel 成本 |
| logits / metadata / KV prep / MLA 小模块 | 已证实小量级 |

## 接手顺序

| 顺序 | 动作 |
|---|---|
| 1 | 先读 `phase45_modeling_decision_handoff.md`，确认当前不能建 latency |
| 2 | 再读 `phase45_artifact_inventory.md`，定位对应脚本、测试和原始结果 |
| 3 | 如要重开 perf，先逐条检查 `phase44_reopen_conditions.md` |
| 4 | 如条件不满足，停在 descriptor / diagnostic，不进入 timing smoke |
| 5 | 如条件满足，另起新 Phase，先定义 clean perf 口径，再谈模型 |
