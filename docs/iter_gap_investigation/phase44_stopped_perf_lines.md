# Phase 44: Stopped Perf Lines

## 结论

下面这些 perf 路线已经停止。后续不能把它们改名后接成 latency 模型，也不能用它们的诊断数字写 perf table。

| 路线 | 状态 | 最终处理 |
|---|---|---|
| MoE WNA16 timing | No-Go | 保留 diagnostic key |
| compiled comm timing | No-Go | 保留 comm candidate key |
| profiler family ms | 停止 | 只做方向证据 |
| NCCL debug trace | 停止 | 只做 group 归因 |
| sync-probe host wait | 停止 | 只做 queue wait 证据 |
| residual bucket | 停止 | 不接模型 |

## MoE WNA16 停止原因

| 原因 | 说明 |
|---|---|
| tuning config fallback | `E=48,N=2048,device_name=NVIDIA_H200.json` 未命中 |
| dtype config fallback | `dtype=bfloat16` 版本也未命中 |
| 权重口径不合格 | 历史 timing 是 random-weight diagnostic |
| 不代表 compiled aggregate | 单层 synthetic timing 不能外推完整 compiled body |
| 不能写表 | `fallback=true` 不满足 Phase 42 数据资格 |

## compiled comm 停止原因

| 原因 | 说明 |
|---|---|
| Python comm marker 未命中 | Phase 38 `communication_op.py` marker 没抓到 compiled path |
| NCCL trace 不是 timing | Phase 39 只能按 `nranks` 分组 |
| profiler 不合格 | coverage 不完整，且改变时序 |
| sync-probe 不合格 | 显式 sync 改变异步边界 |
| 没有 comm-only event boundary | 不能只包通信调用 |

## 已经停止的小模块路线

| 路线 | 停止原因 |
|---|---|
| slot mapping kernel | CUDA event 小，host 大头是 queue wait |
| logits | CUDA event 小量级 |
| KV shape prep | A/B 都是小量级 |
| attention metadata | metadata build 小量级 |
| MLA kernel | `forward_mqa` 小量级 |
| Python communication marker | compiled body 下未命中 |

## 禁止回流

| 数据 | 禁止用途 |
|---|---|
| profiler family ms | 禁止写 latency |
| NCCL trace line count | 禁止当通信强度或耗时 |
| sync-probe host wait | 禁止当模块成本 |
| Phase 25 MoE timing | 禁止代表 Kimi real perf |
| residual bucket | 禁止默认入模 |
