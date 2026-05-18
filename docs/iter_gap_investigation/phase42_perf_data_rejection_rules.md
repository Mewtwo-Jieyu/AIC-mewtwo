# Phase 42: Perf Data Rejection Rules

## 结论

Phase 42 的核心是先拒绝不合格数据。只要数据会改变执行语义、不能标清模块边界、或来自 residual / profiler / debug trace，就不能进入 future experimental latency。

| 数据类型 | 决策 |
|---|---|
| profiler ms | 拒绝 |
| NCCL trace line count | 拒绝 |
| sync-probe host wait | 拒绝 |
| residual bucket | 拒绝 |
| random-weight fallback timing | 拒绝 |
| clean benchmark | 待定，先过数据资格 |

## 硬拒绝规则

| 规则 | 原因 |
|---|---|
| 不用 profiler ms 建表 | profiler 改变时序，Phase 36 coverage 也不完整 |
| 不用 NCCL trace 行数建表 | 行数只代表日志记录，不代表 elapsed time |
| 不用 sync-probe host wait 建表 | 显式 sync 改变异步边界 |
| 不用 residual bucket 建表 | residual 不是物理模块 |
| 不用 random-weight fallback MoE timing 建表 | 不能代表真实 Kimi/vLLM WNA16 性能 |
| 不用 full forward 差分建表 | 差分会把 wrapper、comm、kernel、sync 混在一起 |
| 不复用 TRT-LLM/GB200 数值 | 后端、硬件、kernel、topology 不一致 |

## 具体数据判定

| 数据来源 | 是否有资格 | 处理 |
|---|---|---|
| Phase 33 `pre_slot_sync_ms` | 否 | 只说明 queue wait，不建 slot latency |
| Phase 36 profiler family ms | 否 | 只做方向证据 |
| Phase 39 NCCL trace | 否 | 只做 group/candidate 归因 |
| Phase 25 MoE timing | 否 | 只说明 random-weight fallback 单层不是大项 |
| Phase 41 runtime key | 是 descriptor | 不能返回 latency |
| future compiled comm benchmark | 待定 | 必须先证明边界和 timing 口径干净 |
| future MoE WNA16 benchmark | 待定 | 必须 non-fallback tuning，并明确权重口径 |

## 合格数据最低标准

| 标准 | 说明 |
|---|---|
| 源码边界明确 | 能指出数据对应哪段源码或 kernel API |
| key 字段完整 | shape、topology、dtype、kernel、backend、boundary flag 全部显式 |
| timing 不改变语义 | 不依赖全局 sync/profiler/debug trace |
| 不写默认路径 | `valid_for_default=false` 必须保留 |
| 可复现 | 单 key 多次采集稳定 |
| 可扩展验证 | 至少能设计跨 shape 或 topology holdout |

## 升级门槛

| 阶段 | 允许 | 不允许 |
|---|---|---|
| descriptor | 输出 runtime key | 输出 ms |
| experimental latency | 使用 clean benchmark 数据 | 使用 profiler/debug/residual 数据 |
| default AIC | 需要跨 shape/topology/model 验证 | 单场景直接接默认 |

## 当前停止路线

| 路线 | 停止原因 |
|---|---|
| slot mapping latency | kernel 小，host 大头是 queue wait |
| logits latency | CUDA event 小量级 |
| attention metadata / MLA kernel | 小量级，不解释 mixed gap |
| Python comm marker | Phase 38 未命中 compiled comm 主项 |
| random-weight MoE timing | fallback + synthetic，只能 diagnostic |

## Phase 43 入口

| 文档 | 作用 |
|---|---|
| `phase43_trtllm_methodology_gate.md` | 复核 TRT-LLM 方法论 Gate |
| `phase43_perf_feasibility_go_nogo.md` | 汇总 clean perf 是否 Go |
