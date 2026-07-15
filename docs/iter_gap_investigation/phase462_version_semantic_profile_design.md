# Phase462 版本语义 profile 设计附录

结论：版本 profile 只承载 EngineCore 可由源码核验的结构语义；tokenizer、HTTP、client 暴露节奏不得进入默认 profile。本步只设计，不实施。

## 字段

| 字段 | 类型 | 来源 | 未知版本行为 |
|---|---|---|---|
| `backend` / `backend_version` | exact key | PerfDB deployment metadata | fail closed |
| `applicability.pipeline_parallel_sizes` | list[int] | source path audit | reject |
| `batch_queue_depth` | int | `step_with_batch_queue` source/config | reject |
| `async_scheduling` | bool | source/config | reject |
| `null_blocks_per_pool` | int | `BlockPool` source | reject |
| `request_progress_semantics` | enum | sampled/computed/placeholder source order | reject |
| `completion_release_order` | enum | future callback source order | reject |
| `preemption_victim_policy` | enum | scheduler source | reject |
| `admission.queue_order` | enum | scheduler source | reject |
| `admission.stop_after_preemption` | bool | scheduler source | reject |
| `chunking.stop_after_partial_prefill` | bool | scheduler source | reject |
| `router_policy` | enum | Step 3 source/runtime evidence | unresolved; DP reject |

## 分发

| 位置 | 当前问题 | 设计动作 |
|---|---|---|
| `datatypes.py` | null block 常数硬编码 | 由不可变 profile 注入 `CBSimConfig` |
| `arrival.py` / tokenizer resource | queue depth 与部署测量混放 | queue depth 移入 profile；arrival row 保持 scoped diagnostic |
| `engine_loop.py` | 生命周期与 admission 语义散落在类实现 | resolver 选择明确的语义枚举，运行时不猜 |
| `simulator.py` | engine loop 同时绑定 arrival 与 core | core input 只接 workload 可见请求；arrival adapter 独立可选 |
| `vllm_backend.py` | TP/DP 分发规则分散 | `backend_version -> profile` 一次解析；DP 在 router profile 未完成前显式拒绝 |

建议资源形态：`resources/backend_semantic_profiles.json` + 单一 `resolve_backend_semantic_profile(backend, version, topology)`。profile 只做精确分发，不含默认值、模糊匹配、版本回退或运行时启发式。

边界：`router_policy` 必须等 DP Step 3；本附录不改 runtime、gate 或 PerfDB。
