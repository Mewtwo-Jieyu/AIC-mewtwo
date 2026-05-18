# Phase 58: Scheduler Runtime Descriptor Schema

## 结论

Phase 58 只补 vLLM scheduler/runtime shape 的表达能力，不建 latency，不写 perf table，不改默认 `cb_sim`。

| 项 | 决策 |
|---|---|
| descriptor | 新增 experimental `VLLMSchedulerRuntimeDescriptor` |
| 默认路径 | 不接 `PerfDatabase`、不改 `run_static`、不改 `IterationLatencyCalculator` |
| 数据口径 | scheduler token/request split + runtime shape link + topology |
| 缺字段 | fail-fast，不填默认值 |
| 输出资格 | `valid_for_default=false`、`perf_database=false`、`diagnostic_only=true` |

## 输出口径

| 输出 | 口径 |
|---|---|
| `source=validate_input_shape` | 只表示 validate 输入形状的 cb_sim/scheduled view，不是真实 vLLM scheduler runtime row |
| `source=cb_sim` | 只表示 `diagnose_cb_iter_latency.py` 本地 simulator trace，不是真实 vLLM scheduler runtime row |
| 真实 vLLM scheduler row | 需要同源 runtime log 同时提供 request split、token split、forward regime、cudagraph mode 后才能接入 |
| compare row | 目前只在 descriptor 层提供 ordinal diagnostic compare，不在 CLI 硬造 vLLM compare |

## 字段

| 字段组 | 字段 | 说明 |
|---|---|---|
| identity | `source`、`scenario`、`iteration`、`phase` | 数据来源和迭代坐标 |
| scheduler tokens | `scheduled_context_tokens`、`scheduled_decode_tokens`、`scheduled_total_tokens` | scheduler 看到的 token split |
| scheduler requests | `scheduled_context_reqs`、`scheduled_decode_reqs`、`scheduled_total_reqs` | scheduler 看到的 request split |
| budget | `max_num_batched_tokens`、`max_num_seqs` | scheduler 预算 |
| runtime shape link | `forward_token_count`、`forward_regime`、`cudagraph_runtime_mode` | 与 forward descriptor 对齐的 runtime shape 坐标 |
| topology | `tp`、`dp`、`moe_tp`、`moe_ep`、`topology_key` | vLLM/AIC topology |
| boundary | `valid_for_default`、`perf_database`、`diagnostic_only` | 固定 experimental-only |

## 禁止字段

| 字段 | 原因 |
|---|---|
| `latency_ms` / `duration_ms` | descriptor 不能变成模型数据 |
| `residual_ms` | 不能包装经验缺口 |
| profiler / NCCL trace / sync wait | 不是 clean runtime descriptor |
| throughput / ratio | 这是验收指标，不是机制输入 |
| 缺字段默认值 | 会把 coverage gap 伪装成真实机制 |

## fail-fast 规则

| 条件 | 处理 |
|---|---|
| `phase` 不是 `prefill/mixed/pure_decode` | 抛错 |
| token/request split 为负 | 抛错 |
| `forward_token_count < scheduled_total_tokens` | 抛错 |
| topology size 非正数 | 抛错 |
| `topology_key` 和 topology size 不一致 | 抛错 |
| request split 缺失 | Python keyword 缺失直接失败，不提供默认值 |

## 输出入口

| 入口 | 行为 |
|---|---|
| `validate_cb_simulator.py --experimental-scheduler-descriptor` | 输出 validate 输入形状的 scheduler descriptor CSV |
| `diagnose_cb_iter_latency.py --experimental-scheduler-descriptor --scheduler-descriptor-out <csv>` | 输出 cb_sim trace 的 scheduler descriptor CSV |
| 默认 `validate_cb_simulator.py` | 不输出 descriptor，不改 latency |
| 默认 `diagnose_cb_iter_latency.py` | 不输出 descriptor，不改 latency |

## TRT-LLM 方法论对照

| 问题 | Phase 58 处理 |
|---|---|
| TRT-LLM 方法能否复用 | 只复用“先定义 key，再决定是否采 perf”的纪律 |
| 数据能否复用 | 不能，vLLM scheduler/runtime shape 和 TRT-LLM 静态表 key 不同 |
| 接口能否复用 | 不能直接复用 perf query，只能复用 descriptor 边界 |
| vLLM 还缺什么 | 后续若建模，需要 clean module timing；Phase 58 不提供 timing |
