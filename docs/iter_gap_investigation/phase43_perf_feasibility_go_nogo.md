# Phase 43: Clean Perf Feasibility Go/No-Go

## 结论

Phase 43 收口为 No-Go。compiled comm 和 MoE WNA16 都没有达到 Phase 44 single-key timing smoke 的最低资格。默认 AIC 不动，Phase 41 runtime key 继续只做 experimental descriptor。

| 方向 | 结论 | 原因 |
|---|---|---|
| MoE WNA16 perf | No-Go | tuning config 仍 fallback，历史 timing 是 random-weight diagnostic |
| compiled comm perf | No-Go | 没有 comm-only CUDA event 边界 |
| TRT-LLM 方法论 | Pass | 只能作为拆分和 key 纪律 |
| Phase 44 timing smoke | 不进入 | 两条 perf 线都没过 Gate |
| 默认 `cb_sim` | 不动 | 没有合格 latency 数据 |

## 输入文档

| 文档 | 作用 |
|---|---|
| `phase43_moe_wna16_tuning_config_check.md` | MoE WNA16 tuning config Go/No-Go |
| `phase43_compiled_comm_event_boundary_check.md` | compiled comm event boundary Go/No-Go |
| `phase43_trtllm_methodology_gate.md` | TRT-LLM 方法论 Gate |
| `phase42_perf_data_rejection_rules.md` | 无资格数据拒绝规则 |

## Go/No-Go 明细

| 检查项 | 结果 | 决策 |
|---|---|---|
| `E=48,N=2048,NVIDIA_H200.json` 是否存在 | 否 | MoE No-Go |
| `dtype=bfloat16` tuning config 是否存在 | 否 | MoE No-Go |
| 是否 loaded-weight perf | 否 | MoE No-Go |
| Python comm marker 是否命中 | 否 | comm No-Go |
| NCCL trace 是否能给 elapsed time | 否 | comm No-Go |
| profiler ms 是否可用 | 否 | comm / MoE 都 No-Go |
| sync-probe host wait 是否可用 | 否 | comm / MoE 都 No-Go |

## 当前允许保留

| 项 | 状态 |
|---|---|
| `VLLMCompiledBodyRuntimeKey` | 保留 |
| `tp_comm_candidate` | 保留 descriptor |
| `ep_or_global_comm_candidate` | 保留 descriptor |
| `unknown_comm_present` | 保留 descriptor |
| MoE WNA16 key | 保留 diagnostic key |
| `valid_for_default=false` | 必须保留 |

## 当前禁止

| 禁止项 | 原因 |
|---|---|
| Phase 44 single-key timing smoke | 没有 Go 条件 |
| 写 `PerfDatabase` | 没有合格 perf 数据 |
| 改 `run_static` / `IterationLatencyCalculator` | 默认路径不能被诊断数据污染 |
| 用 profiler/NCCL trace/sync 数字 | 不符合 Phase 42 数据资格 |
| 用 residual 常数 | 不是物理模型 |

## 下一步选择

| 方向 | 条件 |
|---|---|
| 停在 Phase 43 | 当前最合理，归档 No-Go |
| 重开 MoE perf | 先解决 non-fallback tuning config 和权重口径 |
| 重开 compiled comm perf | 先找到 comm-only event boundary |
| 继续 AIC 建模 | 只能做 descriptor / gate，不做 latency |

当前结论：Phase 43 已阻止不合格 timing 进入下一阶段。Phase 44 暂停。

## Phase 44 入口

| 文档 | 作用 |
|---|---|
| `phase44_vllm_modeling_route_closeout.md` | 把 No-Go 结论收成路线决策 |
| `phase44_stopped_perf_lines.md` | 防止停止路线回流 |
| `phase44_reopen_conditions.md` | 定义未来重开门槛 |
| `phase44_default_path_safety_check.md` | 确认默认 AIC 不受影响 |
