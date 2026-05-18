# Phase 59: Descriptor Coverage Matrix

## 结论

Phase 59 只整理 descriptor 交付边界。当前可以交付四类 experimental descriptor，但都不能进入默认 latency。

| descriptor | 覆盖什么 | 当前缺口 | 默认路径 |
|---|---|---|---|
| `VLLMForwardDescriptor` | scheduled token、forward token、cudagraph mode、topology | 真实 vLLM metadata/KV/compiled body 细节不在这里 | 不接默认 latency |
| `VLLMRuntimeShapeKey` | attention tokens、query len、slot mapping、block table、forward context | AIC scheduled view 里的 attention/KV 字段是 coverage gap，不是 prediction error | 不接默认 latency |
| `VLLMCompiledBodyRuntimeKey` | compiled body、comm candidate、MoE WNA16 shape、fallback 状态 | 没有 clean perf；profiler/NCCL trace 不能当 ms | 不接默认 latency |
| `VLLMSchedulerRuntimeDescriptor` | scheduler token/request split、budget、forward regime、topology | 真实 vLLM scheduler row 尚缺同源 request split + cudagraph mode | 不接默认 latency |

## 输出口径

| 输出来源 | 含义 | 不能解释成 |
|---|---|---|
| `validate_input_shape` | validate 输入形状的 cb_sim/scheduled view | 真实 vLLM scheduler runtime |
| `cb_sim` | `diagnose_cb_iter_latency.py` 生成的 simulator trace | 线上 vLLM scheduler 行为 |
| `phase39_nccl_trace` | 只给 compiled-body comm candidate 布尔值 | NCCL 耗时或性能数据 |
| `vllm_runtime_shape` | Phase 9/10 rank row 机制 shape | 默认 latency 模型输入 |

## 禁止回流

| 数据 | 处理 |
|---|---|
| latency / duration 字段 | 不进入任何 descriptor |
| residual bucket | 不进入 descriptor，不进入模型 |
| profiler CUDA ms | 只做诊断，不进入 key |
| NCCL trace line count | 只做归因，不进入 key |
| sync-probe host wait | 只做诊断，不进入 key |
| throughput ratio | 只做验收指标，不进入 key |

## Phase 59 最小交付

| 类别 | 文件 |
|---|---|
| schema/export | `forward_descriptor.py`、`cb_simulator/__init__.py` |
| experimental CLI | `validate_cb_simulator.py`、`diagnose_cb_iter_latency.py` |
| tests | `tests/unit/test_forward_descriptor_scheduler.py` |
| docs | Phase58 schema、Phase58 Go/No-Go、Phase59 coverage matrix |

## 后续门槛

| 方向 | 必须先满足 |
|---|---|
| 接 vLLM scheduler compare | 有同源 vLLM row，且含 request split、token split、forward regime、cudagraph mode |
| 接 experimental latency | 有 clean module timing，不来自 profiler/NCCL trace/sync wait/residual |
| 接默认 `cb_sim` | 多配置验证通过，并且明确不退化 Phase 4 baseline |
