# Phase 43: TRT-LLM Methodology Gate

## 结论

TRT-LLM 方法论继续成立，但只作为 Gate。它要求 vLLM compiled comm / MoE WNA16 在进入 perf 采集前先有干净模块边界、完整 key、非 fallback 数据和 fail-fast 规则。当前两条线都没过 Gate。

| 对照项 | TRT-LLM 做法 | vLLM 当前状态 | Gate |
|---|---|---|---|
| 通信拆分 | All2All 拆 `prepare / dispatch / combine` | compiled comm 没有 comm-only event 边界 | No-Go |
| MoE compute | compute-only，不含通信 | WNA16 仍 fallback，且历史 timing 是 random-weight | No-Go |
| Perf key | kernel、dtype、tokens、hidden、expert、TP/EP 显式 | Phase 41 key 已表达 descriptor，但没有合格 timing | Descriptor only |
| 数据来源 | 专门 collector，明确 warmup/measure/timing | 当前只有 profiler/NCCL trace/sync diagnostic | No-Go |
| 缺数据处理 | 缺表拒绝或显式非默认 | Phase 42 已写拒绝规则 | Pass |

## 能继续复用什么

| 方法 | Phase 43 处理 |
|---|---|
| 模块拆分纪律 | 保留，继续要求 comm 和 MoE 分开 |
| key 纪律 | 保留，Phase 41/42 key 不含 ms/residual |
| no fallback | 保留，fallback=true 不入表 |
| 采集前审查 | 保留，Phase 43 就是 Go/No-Go gate |

## 不能复用什么

| 内容 | 原因 |
|---|---|
| TRT-LLM All2All 数据 | vLLM compiled comm 不是已确认 WideEP alltoallv |
| TRT-LLM MoE compute 表 | vLLM WNA16 compiled aggregate 路径不同 |
| GB200 tuning / perf | 当前证据域是 H200/vLLM/Kimi |
| TRT 静态 op 公式 | vLLM runtime compiled body / graph mode 不同 |

## Gate 结果

| 方向 | Gate 结论 | 原因 |
|---|---|---|
| compiled comm | No-Go | 没有 comm-only event boundary |
| MoE WNA16 | No-Go | tuning config fallback，权重口径不合格 |
| runtime key | Go | 只能作为 descriptor |
| experimental latency | No-Go | 还没有合格 perf 数据 |

当前决策：Phase 43 不进入 Phase 44 timing smoke。
