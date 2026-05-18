# Phase 43: Compiled Comm Event Boundary Check

## 结论

compiled comm perf 线当前 No-Go。现有证据能把 NCCL trace 分成 TP candidate、EP/global candidate 和少量 unknown，但找不到不改变执行语义、只包通信调用的 CUDA event 边界。

| 项 | 判断 |
|---|---|
| Python comm marker | Phase 38 未命中 |
| NCCL trace | 只能分 group / op，不是 timing |
| profiler | coverage 不完整且改变时序 |
| sync-probe | 改变异步边界 |
| CUDAGraphWrapper event | 只能包 compiled body / language model，不是 comm-only |
| Phase 44 comm timing | No-Go |

## 已知证据

| Phase | 结论 | 对 Phase 43 的影响 |
|---|---|---|
| Phase 36 | profiler 显示 NCCL/comm 是主方向之一，但 coverage 约 69% | 只能做方向证据 |
| Phase 38 | `communication_op.py` marker 未命中，runtime summary 只有 `comm_api=none` | Python 层无法提供干净 comm event 边界 |
| Phase 39 | NCCL trace 可按 `nranks=4/8/2` 分组 | 能归因 candidate，不能给耗时 |
| Phase 42 | 要求 timing 不依赖 profiler、NCCL debug、sync-probe | 当前不满足 |

## 边界审查

| 候选边界 | 是否可用 | 原因 |
|---|---|---|
| `tensor_model_parallel_all_reduce` Python wrapper | 不可用 | compiled body 下 marker 未命中 |
| `tensor_model_parallel_all_gather` Python wrapper | 不可用 | compiled body 下 marker 未命中 |
| `CUDAGraphWrapper.__call__` | 不可用 | 包住 whole compiled body，不是 comm-only |
| `Kimi language_model(...)` | 不可用 | 包含 MoE、attention、comm、graph envelope |
| NCCL debug trace | 不可用 | 无稳定 elapsed time，trace 只做归因 |
| profiler kernel family | 不可用 | 改变执行时序且 coverage 不完整 |

## Go/No-Go

| 条件 | 结果 |
|---|---|
| 能定位 comm-only runtime call | 未满足 |
| 能在不加全局 sync 的情况下打 CUDA event | 未满足 |
| 能区分 TP / EP-global / unknown 并获得稳定 elapsed time | 未满足 |
| 能进入 Phase 44 single-key timing smoke | No-Go |

## 后续条件

| 如果以后要重开 compiled comm perf | 必须先满足 |
|---|---|
| 找到 compiled body 内可拦截的 comm-only API | 不是 whole model 或 whole graph replay |
| event 不改变执行语义 | 不加全局 sync，不用 profiler |
| key 能标清来源 | `tp_comm_candidate / ep_or_global / unknown` |
| 单 key 可复现 | 多次采集稳定 |

当前决策：compiled comm 只保留 runtime candidate key，不做 Phase 44 timing。
