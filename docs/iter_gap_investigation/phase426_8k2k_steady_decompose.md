# Phase426 8k2k Steady Decompose

Phase426 复用 Phase425 N=512 trace，只做离线 steady-window 分解；不改 runtime、PerfDatabase 或 gate。

## Verdict

- verdict: `prefill_dominant_with_decode_gap_secondary`
- Phase427 target: `audit_prefill_occupancy_then_decode_gap`
- trace steady penalty: `1.516977`
- validation steady penalty: `1.961257`
- consistency gate: `passed`

## Decomposition

| block | ms | share |
|---|---:|---:|
| prefill wall | 244454.830000 | 0.595460 |
| DP peer-stall extra | 19314.870000 | 0.047048 |
| clean decode latency gap | 146761.520489 | 0.357492 |

## Decode Audit

| shape | gap mean ms | gap min | gap max | slope ms/request | real mean | sim mean | real/sim |
|---|---:|---:|---:|---:|---:|---:|---:|
| batch_scaled_gap | 16.494223 | 7.404402 | 909.992915 | -1.603929 | 42.849884 | 26.355661 | 1.630877 |

## EP Latency Floor Estimate

| layers | directions | us/call range | ms/iter range | covers mean decode gap | source |
|---:|---:|---:|---:|---|---|
| 60 | 2 | 50.000000 - 70.000000 | 6.000000 - 8.400000 | false | vllm_0.19_allgather_reducescatter_AgRsAll2AllManager |

## Expected Residual Direction

| hypothetical fix | remaining attribution share |
|---|---:|
| fix decode gap only | 0.642508 |
| fix prefill wall only | 0.404540 |
| fix prefill + decode | 0.047048 |

## Batch Curve

| decode batch | steps | real ms | sim ms | gap ms |
|---:|---:|---:|---:|---:|
| 43 | 5 | 934.648000 | 24.655085 | 909.992915 |
| 44 | 31 | 649.728065 | 24.907509 | 624.820556 |
| 45 | 160 | 70.268563 | 25.144386 | 45.124177 |
| 46 | 1679 | 43.182644 | 25.381263 | 17.801382 |
| 47 | 1662 | 39.581486 | 25.618140 | 13.963346 |
| 48 | 2818 | 38.953314 | 25.870563 | 13.082751 |
| 49 | 1743 | 44.016816 | 26.092939 | 17.923877 |
| 50 | 1736 | 44.830127 | 26.315314 | 18.514813 |
| 51 | 1613 | 41.723063 | 26.537690 | 15.185373 |
| 52 | 1625 | 42.493243 | 26.775612 | 15.717631 |
| 53 | 1557 | 41.482441 | 26.997987 | 14.484453 |
| 54 | 1376 | 39.624767 | 27.220363 | 12.404405 |
| 55 | 1202 | 38.524351 | 27.442738 | 11.081613 |
| 56 | 401 | 35.085062 | 27.680660 | 7.404402 |

## 32k3k Reference

| source | steady penalty | prefill share | peer-stall share | decode gap share | verdict |
|---|---:|---:|---:|---:|---|
| Phase413 32k3k | 1.357625 | 0.647177 | 0.000299 | 0.352524 | prefill_occupancy_dominates |

## Interpretation

- trace steady penalty 用 Phase425 N=512 raw metrics 重新计算，并与 Phase425 CSV 交叉核对。
- validation steady penalty 只作为现行验收口径参考；它和 trace 口径不同，不混入模型分解。
- EP 延迟底估算按 vLLM 0.19.0 默认 allgather/reducescatter EP a2a 路径，60 层、dispatch/combine 两方向、每次 50-70us。
- 本报告只做 attribution；若离线估算不能覆盖 decode gap，Phase427 需要先 profiler 或继续拆 fixed gap。

## Boundary

- GPU/SSH: not used in Phase426.
- Runtime/PerfDatabase/gate: not modified.
- Phase405 penalty: not read.
- Default AIC: No-Go.
