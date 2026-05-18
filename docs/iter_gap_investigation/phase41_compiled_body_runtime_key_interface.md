# Phase 41: Compiled-Body Runtime Key Interface

## 结论

Phase 41 只把 Phase 36-39 的 compiled body / comm / MoE 证据接成 experimental descriptor。它不建 latency，不写 `PerfDatabase`，不改 `run_static` / `IterationLatencyCalculator`。

| 项 | 状态 |
|---|---|
| `VLLMCompiledBodyRuntimeKey` | 已新增 |
| validate experimental 输出 | 已新增 `--experimental-compiled-body-key` |
| diagnose experimental 输出 | 已新增 `--experimental-compiled-body-key` |
| 默认 `cb_sim` | 不变 |
| latency / residual | 不进入 key |

## Schema

| 字段组 | 字段 |
|---|---|
| shape | `phase`, `forward_regime`, `tokens_padded`, `tokens_actual`, `cudagraph_runtime_mode` |
| topology | `topology_key`, `tp`, `dp`, `ep`, `world_size` |
| compiled path | `compiled_body` |
| comm | `tp_comm_candidate`, `ep_or_global_comm_candidate`, `unknown_comm_present` |
| MoE | `moe_module`, `moe_kernel`, `moe_hidden`, `moe_intermediate`, `moe_experts`, `moe_topk`, `moe_dtype`, `tuning_config_loaded`, `fallback` |
| boundary | `valid_for_default=false`, `perf_database=false`, `diagnostic_only=true` |

## 禁止字段

| 字段类型 | 处理 |
|---|---|
| profiler ms | 不进入 schema |
| NCCL trace line count | 不进入 schema |
| sync wait | 不进入 schema |
| residual bucket | 不进入 schema |
| Phase 25 timing | 不进入 schema |

## Builder 规则

| 规则 | 说明 |
|---|---|
| 输入 | Phase 39 `nccl_trace_phase39_summary.csv` |
| 用途 | 只读取 `comm_candidate`、`valid_for_default`、`perf_database` |
| `valid_for_default` | 必须为 false |
| `perf_database` | 必须为 false |
| 缺 `comm_candidate` | fail-fast |
| 未知 `comm_candidate` | fail-fast |

## 当前输出

| 命令 | 输出 |
|---|---|
| `validate_cb_simulator.py --experimental-compiled-body-key` | `phase41_compiled_body_runtime_key/validate_compiled_body_runtime_key.csv` |
| `diagnose_cb_iter_latency.py --experimental-compiled-body-key ...` | `phase41_compiled_body_runtime_key/diagnose_compiled_body_runtime_key.csv` |

当前 key 的通信候选：

| 字段 | 值 | 来源 |
|---|---|---|
| `tp_comm_candidate` | true | Phase 39 `nranks=4` |
| `ep_or_global_comm_candidate` | true | Phase 39 `nranks=8` |
| `unknown_comm_present` | true | Phase 39 `nranks=2` |

## 边界

| 不做 | 原因 |
|---|---|
| 不改默认 validate | 这是 diagnostic descriptor |
| 不写 `PerfDatabase` | 没有干净 perf 数据 |
| 不建 latency | key 只表达机制输入 |
| 不把 EP/global 拆成 EP | Phase 39 只能证明 `nranks=8` |
| 不用 line count 当强度 | line count 不是 elapsed time |

## 后续

Phase 41 完成后，如果继续推进，只能进入 Phase 42：设计 compiled comm / MoE aggregate 的干净 perf 数据口径。不能直接把这个 key 乘一个常数接进默认模型。

Phase 42 入口：

| 文档 | 作用 |
|---|---|
| `phase42_trtllm_perf_methodology_review.md` | 对照 TRT-LLM WideEP / MoE perf 方法论 |
| `phase42_compiled_comm_perf_spec.md` | 定义 compiled comm future perf 数据资格 |
| `phase42_moe_wna16_perf_spec.md` | 定义 MoE WNA16 aggregate future perf 数据资格 |
| `phase42_perf_data_rejection_rules.md` | 固化无资格数据和停止条件 |
