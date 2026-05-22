# Phase 64 Scheduler Alignment Compare Go / No-Go

## Conclusion

Phase 64 当前是 No-Go：compare 接口能严格按 key 工作，但 Phase62 vLLM artifact 和 Phase63 cb_sim artifact 不能完整对齐。不能退回 phase ordinal，也不能只取交集生成表。

| 项 | 判断 |
|---|---|
| compare schema | Go |
| strict key join | Go |
| `engine_dp:1:step:2` mixed key | 能命中 |
| 完整 CSV key 集 | No-Go |
| compare CSV | 不生成 |
| 默认 AIC | 不改 |
| 远端 | 未使用；如需重采，入口是 `ws-faaf0de74ef9a14d-worker-kw5fs` |

## Artifact Check

| 检查 | 结果 |
|---|---|
| vLLM Phase62 dedup rows | `4003` data rows |
| cb_sim Phase63 dp1 trial rows | `2019` data rows |
| key example | `engine_dp:1:step:2` 两边都有 |
| mixed row 语义 | vLLM 是 `NONE:248`，cb_sim 是 `AIC_UNSET:8192` |
| fail-fast | `missing vllm alignment keys: engine_dp:1:step:2002 ...` |

## Why No-Go

| 原因 | 说明 |
|---|---|
| cb_sim scheduler 语义不同 | cb_sim step2 仍在大块 prefill+decode 混合，vLLM step2 已是 `240+1` padded to `248` |
| key 数量不同 | 严格 join 要求两边 key 集完全一致 |
| 不能用交集 | 会隐藏缺失 key，制造假对齐 |
| 不能用 phase ordinal | Phase62 已经证明 worker-local ordinal 不够干净 |

## Next Gate

| 下一步 | 前置 |
|---|---|
| Phase65 scheduler semantics audit | 查 cb_sim 是否能产生 vLLM 同源 DP scheduler row |
| 接 compare CLI | 只有两边 key 集完整一致后才允许 |
| 继续 descriptor-only | 不加 latency、不接默认模型 |

## Stop Rules

| 现象 | 动作 |
|---|---|
| 需要只取交集 | 停止 |
| 需要 phase ordinal | 停止 |
| 需要补 latency 或 residual | 停止 |
| 需要修改默认 `PerfDatabase` / `run_static` / `IterationLatencyCalculator` | 停止 |
