# Phase 40: vLLM Runtime Key Schema

## 结论

第一版 runtime key 只描述 vLLM compiled body 的执行形态，不包含 latency，不包含 profiler ms，不包含 residual。

| 属性 | 决策 |
|---|---|
| schema 状态 | experimental descriptor |
| latency 字段 | 不包含 |
| 默认路径 | 不接 |
| 缺字段处理 | fail-fast，不插值、不外推 |
| 适用范围 | H200 / vLLM 0.19 / Kimi-K2.5 / `tp4dp2ep8` 当前证据域 |

## Schema

| key 组 | 字段 | 说明 |
|---|---|---|
| workload | `phase` | `prefill / mixed / pure_decode` |
| workload | `forward_regime` | 例如 `NONE:248`、`PIECEWISE:248`、`FULL:16` |
| workload | `tokens_padded` | vLLM forward padded token count |
| workload | `tokens_actual` | attention / effective token count |
| topology | `tp` | tensor parallel size |
| topology | `dp` | data parallel size |
| topology | `ep` | expert parallel size |
| topology | `world_size` | distributed world size |
| compiled path | `cudagraph_runtime_mode` | `NONE / FULL / PIECEWISE / AIC_UNSET` |
| compiled path | `compiled_body` | `true` if execution is through compiled Kimi language model body |
| comm | `tp_comm_candidate` | true when NCCL trace has `nranks=tp` |
| comm | `ep_or_global_comm_candidate` | true when NCCL trace has `nranks=world_size` |
| comm | `unknown_comm_present` | true when trace has comm groups outside known topology mapping |
| MoE | `module` | e.g. `DeepseekV2MoE` |
| MoE | `kernel` | e.g. `wna16` |
| MoE | `hidden` | model hidden size |
| MoE | `intermediate` | MoE intermediate size |
| MoE | `experts` | routed expert count |
| MoE | `topk` | routing top-k |
| MoE | `dtype` | runtime dtype |
| MoE | `fallback` | true when vLLM MoE tuning config fallback is used |
| boundary | `valid_for_default` | fixed `false` |
| boundary | `perf_database` | fixed `false` |
| boundary | `diagnostic_only` | fixed `true` |

## 当前样例

| 字段 | 值 |
|---|---|
| `phase` | `mixed` |
| `forward_regime` | `NONE:248` |
| `tokens_padded` | `248` |
| `tokens_actual` | `241` |
| `tp/dp/ep/world_size` | `4 / 2 / 8 / 8` |
| `cudagraph_runtime_mode` | `NONE` |
| `compiled_body` | `true` |
| `tp_comm_candidate` | `true` |
| `ep_or_global_comm_candidate` | `true` |
| `unknown_comm_present` | `true` |
| `module` | `DeepseekV2MoE` |
| `kernel` | `wna16` |
| `hidden/intermediate/experts/topk` | `7168 / 2048 / 384 / 8` |
| `dtype` | `bfloat16` |
| `fallback` | `true` |
| `valid_for_default` | `false` |

## 不进入 Schema 的内容

| 内容 | 原因 |
|---|---|
| `profiled_cuda_time_ms` | profiler 不是稳定采集口径 |
| `nccl_trace_line_count` | line count 不是 latency |
| `pre_slot_sync_ms` | sync probe 改变异步边界 |
| `execute_residual_ms` | residual 不是机制输入 |
| bucket residual | Phase 6 已归档为 empirical diagnostic |

## 使用规则

| 规则 | 说明 |
|---|---|
| 只做 descriptor | 只能帮助 AIC 表达 vLLM runtime shape |
| 不做 latency | 不能返回预测 ms |
| 不做 fallback | 缺字段直接拒绝，不构造默认值 |
| 不做外推 | 新模型、新 topology、新 backend 必须重新取证 |
