# Phase 42: TRT-LLM Perf Methodology Review

## 结论

Phase 42 只复用 TRT-LLM 的方法论：先把通信、MoE compute、dispatch/combine 拆成干净模块，再给每类数据定义 key 和数据资格。TRT-LLM 的数值、公式、GB200 数据不能迁移到 H200/vLLM compiled body。

| 项 | 决策 |
|---|---|
| 是否复用 TRT-LLM perf 数值 | 否 |
| 是否复用 TRT-LLM 公式 | 否 |
| 是否复用采集纪律 | 是 |
| 是否复用 key 设计思想 | 是 |
| 是否改默认 `cb_sim` | 否 |

## AIC 现有 TRT-LLM 方法

| 对象 | 入口 | 方法论 |
|---|---|---|
| WideEP All2All | `collector/slurm_comm_collector/collect_trtllm_alltoall.py` | 拆 `prepare / dispatch / combine / low_precision_combine`，不和 MoE compute 混在一起 |
| WideEP MoE compute | `collector/trtllm/collect_wideep_moe_compute.py` | compute-only 采集，明确不包含 All2All 通信 |
| Perf table loader | `src/aiconfigurator/sdk/perf_database.py` | 表 key 显式包含 kernel、quant、token、hidden、expert、TP/EP 等字段 |
| Runtime op | `src/aiconfigurator/sdk/operations.py` | `TrtLLMWideEPMoEDispatch` 和 `TrtLLMWideEPMoE` 分开查询 |
| Model wiring | `src/aiconfigurator/sdk/models.py` | context/generation 都按 pre-dispatch、MoE compute、post-dispatch 三段接入 |

## 可复用的方法

| 方法 | vLLM Phase 42 用法 |
|---|---|
| 模块先拆开 | compiled comm 和 MoE WNA16 aggregate 必须分别定义数据口径 |
| key 必须完整 | key 至少包含 backend、kernel、dtype、tokens、topology、tuning/fallback 状态 |
| 缺数据 fail-fast | 不能用 profiler ms、trace 行数、residual 常数兜底 |
| 采集口径先验收 | 先证明数据不改变执行语义，再考虑 experimental latency |
| 默认路径隔离 | 所有 Phase 42 数据资格都必须保持 `valid_for_default=false` |

## 不能复用的假设

| TRT-LLM 假设 | vLLM 不成立的原因 |
|---|---|
| All2All op 有明确 `prepare / dispatch / combine` Python 边界 | Phase 38 Python 通信 marker 未命中 vLLM compiled comm 主项 |
| kernel source 能按 TRT-LLM selector 选出 | Phase 36/39 只看到 NCCL trace candidate，不等于 TRT WideEP kernel |
| MoE compute 表能代表模块成本 | Phase 36 的 MoE 是 vLLM compiled WNA16 aggregate，不是 TRT-LLM MoEOp |
| perf 数据来自同一后端和硬件 | 当前证据域是 H200 SXM + vLLM 0.19 + Kimi-K2.5 |
| profiler family ms 可校准表 | Phase 36 profiler coverage 不完整，不能入表 |

## 对 vLLM Phase 42 的直接约束

| 方向 | 允许推进 | 禁止推进 |
|---|---|---|
| compiled comm | 设计干净 timing 资格和 comm key | 用 NCCL trace ms 或 line count 当 perf |
| MoE WNA16 | 设计真实 shape、真实 tuning config、权重口径 | 用 random-weight fallback timing 当 perf |
| runtime key | 保留 experimental descriptor | 接默认 latency |
| TRT-LLM 对照 | 借拆分方法和数据格式纪律 | 迁移 TRT/GB200 数值 |

## Phase 42 产物关系

| 文档 | 作用 |
|---|---|
| `phase42_compiled_comm_perf_spec.md` | 定义 compiled comm 数据资格 |
| `phase42_moe_wna16_perf_spec.md` | 定义 MoE WNA16 aggregate 数据资格 |
| `phase42_perf_data_rejection_rules.md` | 定义无资格数据和停止条件 |
