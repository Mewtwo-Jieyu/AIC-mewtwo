# Phase464 Kimi vLLM 0.19 cb_sim baseline closeout

结论：当前分支可以作为 Kimi-K2.5 / H200 SXM / vLLM 0.19.0 的阶段性专用 baseline 交付，但不能表述为通用 vLLM 仿真模型。Phase463 严格 15% 吞吐门仍为 `3/6`，延迟只有对比证据而没有准入门；代码审计还发现模型作用域、EP8 非精确 bucket 和 DP 大 bucket 路径需要收紧或补充结构建模。因此 `Default AIC=No-Go`。

## Baseline identity

| 项 | 当前边界 |
|---|---|
| 评审对象 | `feature/kimi-vllm019-cb-sim-baseline` at `7702f27470580a7e91ce958e662ee883fb3f3c83` |
| 模型 | `moonshotai/Kimi-K2.5` |
| 硬件 | 8 x NVIDIA H200 SXM |
| backend | vLLM `0.19.0` |
| 已验证拓扑 | `tp8ep8`、`tp4dp2ep8` |
| 正式协议 | N512、C128；C64 只作诊断 |
| GPU artifact | `/mnt/shared-storage-user/zhaojieyu/backup/aic/phase463_six_point_latency_recollect_263a969` |
| 证据属性 | `diagnostic_only=true`、`valid_for_default=false`、`perf_database=false` |

这份 baseline 的正确定位是“一个部署组合的结构化仿真基线”。当前证据不能外推到其他模型、硬件、vLLM 版本、并行拓扑、并发、请求长度分布或尾延迟目标。

## Delivered work

| 能力 | 已完成内容 | 当前定位 |
|---|---|---|
| 模型与算子 | Kimi-K2.5 配置、W4A16/Marlin MoE、MLA、CustomAllReduce 和 EP8 通信路径 | 只在已采集部署组合内有证据 |
| module PerfDB | `vllm_module_perf.txt` 提供 Kimi、H200、vLLM 0.19、`tp4dp2ep8` 的 module-level exact rows | exact-key 数据，不是通用曲线 |
| serving-state PerfDB | 按 topology、max batched tokens、phase、bucket tokens、decode batch 等查询实测 serving state | 作用域和状态维度仍需收紧 |
| 调度器 | vLLM 风格 running-then-waiting、chunked prefill、KV block、preemption 和 closed-loop 请求推进 | 简化状态机，不等于完整 EngineCore |
| backend 语义 | vLLM 0.19 profile 固化每个 BlockPool 的 null block、队列和抢占相关语义 | engine loop 默认关闭 |
| DP | 有单副本路径和一段已验证条件下的 multi-replica lockstep 路径 | 大 batched-token 路径仍不完整 |
| 测量工具 | 六场景顺序运行、失败跳过、Prometheus/client 双源延迟、GPU 遥测和残留检查 | Phase463 已形成可复核 artifact |
| 失败路线收口 | engine-loop A/B 回归；composition logging v2/v3/v4 开销均超过 2% | 不进入默认路径，不得复用其数据 |

## Phase463 scoreboard

吞吐正式门为 simulator/real 绝对误差不超过 15%。

| 场景 | 实测 output tok/s/GPU | 仿真 output tok/s/GPU | 绝对误差 | 15% 门 |
|---|---:|---:|---:|---|
| K2.5-tp8ep8-8k2k | 146.1113 | 165.9845 | 13.60% | pass |
| K2.5-tp8ep8-32k3k | 50.0801 | 49.5027 | 1.15% | pass |
| K2.5-tp4ep8dp2-8k2k | 164.3446 | 163.0730 | 0.77% | pass |
| K2.5-tp4ep8dp2-32k3k | 52.6355 | 64.0492 | 21.68% | fail |
| K2.5-tp8ep8-8k2k-bt65536 | 124.2006 | 155.8699 | 25.50% | fail |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 110.1960 | 127.4641 | 15.67% | fail |

C64 诊断点 `K2.5-tp4ep8dp2-32k3k-c64-diagnostic` 为实测 `53.4844`、仿真 `64.0492`、误差 `19.75%`。它不计入六点正式门，也不能替代 C128 结论。

## Mean latency

下表只比较均值。ratio 为 sim/real；小于 1 表示仿真低估延迟，大于 1 表示仿真高估延迟。

| 场景 | TTFT real / sim / ratio (ms) | TPOT real / sim / ratio (ms) | E2E real / sim / ratio (ms) |
|---|---:|---:|---:|
| K2.5-tp8ep8-8k2k | 105573.42 / 95635.50 / 0.906x | 48.72 / 43.41 / 0.891x | 202966.76 / 182406.85 / 0.899x |
| K2.5-tp8ep8-32k3k | 756839.83 / 877331.74 / 1.159x | 31.27 / 33.43 / 1.069x | 850633.17 / 977576.67 / 1.149x |
| K2.5-tp4ep8dp2-8k2k | 42886.67 / 39622.88 / 0.924x | 69.95 / 72.81 / 1.041x | 182721.52 / 185164.15 / 1.013x |
| K2.5-tp4ep8dp2-32k3k | 689454.70 / 561265.59 / 0.814x | 40.62 / 34.92 / 0.859x | 811288.87 / 665981.52 / 0.821x |
| K2.5-tp8ep8-8k2k-bt65536 | 165474.44 / 138575.02 / 0.837x | 34.24 / 29.97 / 0.875x | 233926.87 / 198476.24 / 0.848x |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 200142.37 / 171586.55 / 0.857x | 28.20 / 28.15 / 0.998x | 256523.58 / 227865.82 / 0.888x |

延迟证据目前只能说明均值偏差方向。仿真器没有对应真实请求分布的 p50/p90/p99 生成机制，因此不能从 mean TTFT/TPOT 外推尾延迟 readiness。

## Measurement consistency

| 检查 | 结果 | 含义 |
|---|---:|---|
| streaming canary throughput delta | 0.0122% | 低于 2% 测量扰动门 |
| client vs Prometheus mean 最大差异 | 0.048561% | 两套均值测量一致 |
| Phase463 实测 vs frozen real 最大漂移 | 1.2050% | 六点实测基线稳定 |
| real vs sim decode requests/iteration 最大差异 | 4.9779% | decode 聚合数量较接近，但不能单独解释时延 |
| real vs sim prefill requests/iteration 差异范围 | 18.74%-53.86% | 只能作为 composition 线索，不能直接认定根因 |
| 正式点 GPU util mean | 97.85%-99.80% | 实测运行处于持续高负载 |

Phase462 composition logging v2/v3/v4 的 off/on 扰动分别为 `13.8486%`、`8.7915%`、`8.2043%`。这些数据未通过 2% 门，不得用来证明 prefill composition、sim undercharge 或 PerfDB row 候选。

## Gate interpretation

| Gate | 当前结果 | 可得结论 |
|---|---|---|
| 旧 `validate_cb_simulator.py` multi-config 门 | 上限 `1.50x`；当前 max `1.26x`、mean `1.13x`，PASS | 只证明没有超过旧宽门，不代表 Default readiness |
| validator 旧 0.17 throughput/TTFT 面 | 当前最大分别为 `3.72x`、`15.64x`，均明确 SKIP | 不参与当前准入，也不能拿来证明延迟准确 |
| Phase463 严格吞吐门 | `3/6` | 三个动态场景仍未收敛 |
| TTFT/TPOT/E2E | `baseline_only` | 有均值对比，没有准入门 |
| Default AIC | `No-Go` | 不改默认、不扩大支持声明 |

## Generalization and correctness review

| 检查项 | 源码事实 | 判断 | 后续动作 |
|---|---|---|---|
| module exact binding | `operations.py` 对 model、hardware、version、`tp4dp2ep8` 和 exact bucket 做显式检查 | 专用作用域清楚，可以保留 | 继续 fail closed，不扩成 nearest lookup |
| serving-state model scope | `_serving_state_scope_enabled()` 检查 backend、H200、0.19 和 EP8 结构，但不检查 `model_path`；查询时固定传 `model="kimi-k2.5"` | 相同结构的非 Kimi 模型可能静默读取 Kimi 数据 | Phase465 增加 runtime model identity guard 和负向测试 |
| int4 MoE calibrated SOL | `use_phase397v_int4_wo_calibrated_sol()` 只检查 H200、vLLM 0.19 和 `int4_wo`；缺测量覆盖时使用单个 profiler anchor scale | 校准可能外溢到其他 int4 模型和未验证形状 | Phase465 把模型身份和适用形状带入查询；只有命中匹配的通用 PerfDB 数据时才使用通用路径，否则明确失败 |
| EP8 non-exact bucket | `_query_vllm_ep8_alltoall_fallback()` 捕获 `AttributeError`、数据缺失和范围错误，再按理想双向带宽计算 | 数据缺失被隐藏，当前结果来源不够显式 | Phase465 拆成明确 measured/structural source；结构公式必须带 provenance 和有效域，未过门前只作诊断；缺必要元数据直接失败 |
| DP route selection | `use_dp_lockstep = dp > 1 and ctx_tokens == isl`；其他 DP 场景用单副本结果乘 DP | 等式开关是已验证场景选择，不是通用 DP 语义 | Phase466 先定位大 bucket 的 rank 同步、负载和 wall-time，再决定模型 |
| serving-state dimensions | key 包含 topology、max batched tokens、phase、bucket tokens、decode batch 等，但没有 prefill request count、average KV、preemption/recompute 或 rank state | 同一 key 可能覆盖不同执行组成 | Phase466 证明缺失维度与误差有因果关系后才扩 schema |
| forward-total observability | 命中 `forward_total` 后总耗时可替换，但 breakdown 将 attention/non-attention 记为 0 | 总值可用，算子归因丢失 | 后续保留 row provenance 和不可分解标志，不能伪装成零成本算子 |
| backend semantic profile | profile 只按 backend/version 解析 | 当前只证明 vLLM 0.19/H200/Kimi 测试组合，不能声称所有 vLLM 0.19 部署一致 | Phase464 限定支持域；跨硬件/配置需独立 oracle |
| `overlap_factor` | 参数仍被校验、缓存和输出，但当前 mixed/prefill/decode 都按串行相加，结果与它无关 | 对外接口含义与实现不一致，但不是当前三点误差根因 | 后续删除或显式废弃，不用它拟合 gap |
| holdout | 当前六点既参与历史建模又承担验收 | 只能证明同域回归，不能证明 workload 外扩 | 修完结构问题后增加独立 holdout |

上述问题分为两类：

1. **作用域漏洞**：serving-state 和 int4 校准可能读取不属于当前模型的数据，必须先修。
2. **缺失建模**：DP 大 bucket、serving-state 状态维度和尾延迟尚无充分因果证据，不能直接加系数或补 row。

## Supported and unsupported claims

| 可以声明 | 不能声明 |
|---|---|
| 当前分支已经具备 Kimi-K2.5/H200/vLLM 0.19 的结构化 cb_sim baseline | 已经完成通用 vLLM continuous batching 模型 |
| 六个正式场景都有新鲜实测吞吐和 mean latency 对比 | 三个失败场景已经找到根因 |
| TP8-8k2k、TP8-32k3k、DP2-8k2k 通过严格 15% 吞吐门 | 其他模型、硬件、版本、拓扑或 workload 可以复用相同精度 |
| Phase462 已排除当前 engine-loop 和高扰动 composition logger | 可以根据 aggregate prefill mismatch 直接修改 scheduler 或 PerfDB |
| 这份 baseline 可以作为短期阶段性交付 | 可以打开 Default AIC |

## Next phases

| Phase | 目标 | 必须通过的门 |
|---|---|---|
| Phase465 | runtime applicability fail-closed hardening | 非 Kimi 不读取 Kimi serving-state；非 Kimi int4 不使用 Phase397v anchor；EP8 数据源不靠异常 fallback；六点 Kimi 输出不回归 |
| Phase466 | 三个失败场景的无扰动归因 | probe off/on 吞吐扰动 `<=2%`；覆盖 DP2-32k3k、TP8-bt65536、DP2-bt65536；能区分调度组成、单轮成本和 rank 不均衡 |
| Phase467 | 只实现 Phase466 证实的结构模型 | scheduler、serving-state 或 DP 三条路线只选证据支持的一条；禁止经验补偿和 nearest patch |
| Phase468 | 独立 holdout 与延迟门 | 使用未参与建模的 workload；正式吞吐门达到 `6/6 <=15%` 且已通过点不回归；另行定义 mean 和 tail latency 验收 |

Phase466 的采集只允许低频、聚合、rank-local 的 iteration 级指标：prefill/decode tokens 和 requests、preemption/recompute、rank wall time、有效 forward source。不得恢复 Phase462 的逐请求高频 composition logging。

根据 Phase466 结果，Phase467 只能选择以下一种主路径：

- schedule composition 与真实不一致：补 scheduler/merged-batch 粒度模型；
- schedule composition 一致但单轮耗时偏低：补 iteration cost 或 serving-state 状态维度；
- rank-local 成本接近但全局 wall time 不一致：补 DP 同步和 rank 非对称模型。

在严格吞吐门达到 6/6、作用域漏洞关闭、独立 holdout 和延迟门建立前，保持 `diagnostic_only=true`、`valid_for_default=false`、`perf_database=false`、`Default AIC=No-Go`。

## Evidence links

- [Phase462 baseline handoff](phase462_baseline_handoff.md)
- [Phase462 engine-loop arc closure](phase462_engine_loop_arc_closure.md)
- [Phase463 six-point latency recollect](phase463_six_point_latency_recollect.md)
- [Phase463 six-point CSV](phase463_six_point_latency_recollect.csv)
