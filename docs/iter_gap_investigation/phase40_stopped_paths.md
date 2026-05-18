# Phase 40: Stopped Paths

## 结论

下面这些路线已经有足够证据停止。后续不要重复排查，也不要把它们包装成默认 latency 模型。

| 路线 | 停止原因 | 最终处理 |
|---|---|---|
| slot mapping kernel | CUDA event 小，host 大头是队列等待 | 不建 slot latency |
| `_get_slot_mappings` | Phase 15f/31 都是小量级 | 不回挖 |
| KV shape prep | A/B 都是亚毫秒级 | 非主因 |
| attention metadata build | `FlashAttentionMetadataBuilder.build` 约 `0.025ms` | 非主因 |
| MLA `forward_mqa` | 单 key 约 `0.07ms` | 非主因 |
| logits / `LogitsProcessor` 外层 | logits event 约 `2ms`，且 owner 是 `CUDAGraphWrapper` | 不继续 patch 外层 |
| Python communication marker | compiled body 下未命中 | 停止 patch Python comm 入口 |
| 单层 random-weight MoE timing | fallback + random weight + 小毫秒级 | 不外推 60 层建模 |
| residual bucket | 只能解释现象，不能解释机制 | 不接默认模型 |

## 关键禁止项

| 禁止项 | 原因 |
|---|---|
| 用 slot host wall 建 slot perf | Phase 33 证明大头在 pre-slot sync |
| 用 profiler ms 校准 AIC | Phase 36 coverage 约 `69%`，且 profiler 改变时序 |
| 用 NCCL trace line count 建通信成本 | line count 不是 elapsed time |
| 把 `nranks=8` 直接叫 EP | 只能说 EP/global candidate |
| 用 Phase 25 MoE timing 当 Kimi 真实性能 | 没加载真实权重，且 tuning config fallback |
| 继续 patch `LogitsProcessor.forward/_get_logits` | 当前 runtime owner 是 `CUDAGraphWrapper` |

## 保留项

| 项 | 用途 |
|---|---|
| slot / logits / metadata / KV 结论 | 作为非主因证据 |
| MoE WNA16 runtime shape | 作为 diagnostic key |
| NCCL `nranks=4/8/2` 分类 | 作为 comm candidate key |
| compiled body envelope | 作为下一阶段 perf 口径设计对象 |

## 不再重复的验证

| 方向 | 状态 |
|---|---|
| `gpu_model_runner.py` slot mapping host wall 复测 | 停止 |
| `LogitsProcessor` Python patch | 停止 |
| attention metadata / MLA kernel smoke | 停止 |
| KV shape prep benchmark | 停止 |
| Python comm marker | 停止 |
| residual bucket 拟合 | 停止 |
