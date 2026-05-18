# Phase 40: vLLM Compiled Body Modeling Decision

## 结论

| 项 | 决策 |
|---|---|
| 默认 `cb_sim` | 不动，继续保留 Phase 4 baseline |
| vLLM compiled body | 保留为 diagnostic forward envelope |
| runtime key | 可以进入 experimental descriptor |
| latency 模型 | 现在不建 |
| `PerfDatabase` | 不写 |

Phase 27-39 的证据链已经说明：vLLM mixed 几百毫秒 gap 不在 slot mapping kernel、logits、attention metadata、KV prep、MLA 单 kernel、或单层 random-weight MoE。当前可落到 AIC 的不是 latency 数字，而是 vLLM runtime shape / compiled body / comm / MoE 的 experimental descriptor。

## Phase 27-39 证据链

| Phase | 结论 | 建模决策 |
|---|---|---|
| 27 | mixed `execute_model_wall≈750ms`，`model_forward≈284ms`，还有 post/hidden envelope | 不直接跳 layer-loop |
| 28 | hidden envelope 拆干净，mixed 主项是 pre-batch、model forward、logits | 继续拆连续源码段 |
| 29 | `_prepare_inputs≈310ms`，`compute_logits≈144ms` | 不接默认模型 |
| 30 | `_prepare_inputs` 主项落在 `compute_slot_mapping` host wall；logits 内部拆分 No-Go | 不用无效 `lm_head/tp=0` 字段 |
| 31 | `get_slot_mappings≈0.03ms`；logits owner 是 `CUDAGraphWrapper` | 不回挖 KV B 段，不 patch `LogitsProcessor` 外层 |
| 32 | slot host `309ms`，CUDA event `0.14ms` | 不能把 slot host 当 kernel 成本 |
| 33 | `pre_slot_sync≈309ms`，sync 后 slot 约 `1.2ms` | slot mapping latency 路线停止 |
| 34 | `model_forward_event≈431ms`，`compute_logits_event≈2ms` | logits 降级，转 model forward |
| 35 | Kimi forward 覆盖 runner model forward 约 `99.997%` | wrapper 外壳不是主因 |
| 36 | compiled body profiler 主项是 NCCL/comm + MoE，coverage 约 `69%` | 只做方向证据，不用 ms 建模 |
| 37 | MoE marlin 可归 WNA16 aggregate，NCCL 仅 kernel 名不能归 TP/EP | 不靠 kernel 名硬判 |
| 38 | Python comm marker 未命中；WNA16 key 可诊断 | 停止 Python comm patch |
| 39 | NCCL trace 可按 `nranks` 分组 | 只标 candidate，不写 perf table |

## 模块决策

| 模块/边界 | 当前状态 | 决策 |
|---|---|---|
| slot mapping | kernel 小，host 大头是队列等待 | 不建 slot latency |
| logits | CUDA event 小量级 | 不建 logits latency |
| Kimi compiled body | 主项进入 compiled body | 保留 forward envelope |
| MoE WNA16 | 有 shape key，但 `fallback=true` | 只做 diagnostic key |
| NCCL TP candidate | `nranks=4` | 可保留 runtime comm key |
| NCCL EP/global candidate | `nranks=8` | 只能叫 EP/global candidate |
| NCCL unknown | `nranks=2`，8 行 | 保留 unknown |
| default AIC | Phase 4 validate 继续 PASS | 不动 |

## 可以进入 experimental descriptor 的信息

| 字段组 | 用途 |
|---|---|
| workload shape | 表达 vLLM 实际 forward regime |
| topology | 区分 TP/DP/EP/world size |
| compiled path | 标记 `CUDAGraphWrapper` / compiled body / graph mode |
| comm candidate | 记录 TP candidate、EP/global candidate、unknown |
| MoE WNA16 key | 记录 vLLM fused MoE aggregate 的 runtime shape |
| boundary flags | 固定 `valid_for_default=false`、`perf_database=false`、`diagnostic_only=true` |

## 不能进入 latency 的数据

| 数据 | 原因 |
|---|---|
| profiler CUDA ms | profiler 改变时序，coverage 不完整 |
| NCCL trace line count | 只是日志记录数，不是性能 |
| sync-probe host wait | 显式 sync 改变执行语义 |
| residual bucket | 不是物理模型 |
| Phase 25 single-layer MoE timing | random weight + fallback，不能代表 compiled body aggregate |

## 最终判断

| 问题 | 当前答案 |
|---|---|
| 能不能把 compiled body gap 接默认 AIC | 不能 |
| 能不能保留 runtime key | 能，但只限 experimental descriptor |
| 下一阶段要先做什么 | 定义干净 perf 数据口径 |
| 现在是否继续补 op 表 | 不继续 |

## Phase 44 入口

| 文档 | 作用 |
|---|---|
| `phase44_vllm_modeling_route_closeout.md` | 汇总 Phase 40-43 决策链 |
| `phase44_stopped_perf_lines.md` | 固化停止路线 |
| `phase44_reopen_conditions.md` | 定义重开条件 |
| `phase44_default_path_safety_check.md` | 复核默认路径安全 |
