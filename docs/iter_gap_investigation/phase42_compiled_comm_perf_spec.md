# Phase 42: Compiled Comm Perf Spec

## 结论

compiled comm 现在只能设计 future experimental latency 的数据资格，不能建模型。合格数据必须能区分通信来源，并使用不依赖 profiler、NCCL debug 行数、sync-probe host wait 的计时口径。

| 项 | 决策 |
|---|---|
| 当前是否写 `PerfDatabase` | 否 |
| 当前是否改 `run_static` | 否 |
| 当前是否用 Phase 39 NCCL trace 数字 | 否 |
| 当前是否可保留 runtime key | 可以，仍是 diagnostic-only |
| 下一步最低门槛 | 干净 comm timing 口径 + 来源归因 |

## 数据资格

| 要求 | 标准 |
|---|---|
| 通信来源 | 至少区分 `tp_comm_candidate`、`ep_or_global_comm_candidate`、`unknown_comm_present` |
| 拓扑字段 | 必须包含 `tp / dp / ep / world_size / rank / group_size` |
| shape 字段 | 必须包含通信 tensor shape、dtype、tokens、phase、forward_regime |
| 计时字段 | 必须来自 CUDA event 或等价的不改变执行语义的模块 timing |
| 边界字段 | 必须固定 `valid_for_default=false`、`perf_database=false` |
| 缺字段 | fail-fast，不补默认值、不插值、不外推 |

## 允许的数据源

| 数据源 | 是否可用 | 条件 |
|---|---|---|
| future clean comm benchmark | 待定 | 能证明不改变执行语义，并能标清 TP/EP/global 来源 |
| future runtime comm event marker | 待定 | event 只包通信调用，不引入全局 sync，不污染服务语义 |
| Phase 41 runtime key | 可用 | 只能做 descriptor，不能输出 latency |

## 禁止的数据源

| 数据源 | 处理 |
|---|---|
| Phase 36 profiler ms | 拒绝，coverage 不完整且 profiler 改变时序 |
| Phase 39 NCCL trace line count | 拒绝，line count 不是耗时 |
| Phase 33 sync-probe host wait | 拒绝，显式 sync 改变异步边界 |
| residual bucket | 拒绝，不是物理模块数据 |
| kernel 名字符串 | 只能做候选分类，不能做归因或计时 |

## Future Key 草案

| 字段组 | 字段 |
|---|---|
| identity | `backend=vllm`, `vllm_version`, `module=compiled_body_comm` |
| workload | `phase`, `forward_regime`, `tokens_padded`, `tokens_actual` |
| topology | `rank`, `dp`, `tp`, `ep`, `world_size`, `group_size` |
| comm | `comm_api`, `comm_candidate`, `op_type`, `tensor_shape`, `dtype` |
| compiled path | `cudagraph_runtime_mode`, `compiled_body=true` |
| data boundary | `timing_source`, `valid_for_default=false`, `perf_database=false`, `diagnostic_only=true` |

## 进入 experimental latency 的门槛

| 门槛 | 通过标准 |
|---|---|
| 来源归因 | 能把 comm 归到 TP、EP/global 或明确 unknown |
| 计时干净 | 计时不依赖 profiler、debug trace、sync-probe |
| 单 key 复现 | 同一 key 多次采集稳定 |
| 不写默认表 | 数据仍保存在 experimental 路径 |
| 跨 key 验证 | 至少有一个 shape 或 topology holdout |

## 停止条件

| 情况 | 处理 |
|---|---|
| 无法区分 TP 和 EP/global | 只保留 descriptor，不设计 latency |
| 只能通过 profiler 得到 ms | 停止，不入表 |
| 只能通过 NCCL debug 行数估计 | 停止，不入表 |
| 需要 full `_model_forward` 残差相减 | 停止，避免回到 residual 模型 |
| 想接默认 AIC | 拒绝，直到通过多 key 和跨配置验证 |

## Phase 43 入口

| 文档 | 作用 |
|---|---|
| `phase43_compiled_comm_event_boundary_check.md` | 判断 compiled comm 是否有干净 event boundary |
| `phase43_perf_feasibility_go_nogo.md` | 汇总是否进入 Phase 44 timing smoke |
