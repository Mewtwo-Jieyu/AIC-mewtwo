# Phase412 Arrival Sweep

Phase412 对比 N=128 突发 baseline 与持续到达样本，只做 GPU 测量结果归因；不改 runtime、PerfDatabase 或 gate。

## Verdict

- verdict: `arrival_sweep_partial_collapse`
- Phase413 target: `separate_tail_or_drain_from_remaining_dp_cost`
- request imbalance monotonic down: `true`
- penalty monotonic down: `true`
- last request ratio: `1.000000`
- last overall penalty: `1.740990`

## Samples

| N | artifact | request ratio | overall penalty | steady penalty | bench tok/s/gpu | trace fidelity |
|---:|---|---:|---:|---:|---:|---|
| 128 | docs/iter_gap_investigation/phase409_iter_trace/K2.5-tp4ep8dp2-32k3k | 1.415094 | 1.887133 | 1.616155 | 45.741879 | passed |
| 512 | docs/iter_gap_investigation/phase412_arrival_sweep/K2.5-tp4ep8dp2-32k3k | 1.000000 | 1.740990 | 1.357625 | 53.182455 | passed |

## Interpretation

- N=512 持续到达把 per-engine request ratio 从 `1.415094` 拉平到 `1.000000`，说明 Phase411 的 request-count imbalance 是闭集突发放大的。
- overall penalty 只从 `1.887133` 降到 `1.740990`，没有塌到 `<=1.25`；所以不能说只是 benchmark 突发伪影。
- steady-window penalty 是 `1.357625`，明显低于 overall penalty；剩余问题更像 tail/drain 或持续到达下的 DP cost，需要 Phase413 单独拆。
- N=256 暂不必补：N=512 已经把请求分配拉平，趋势问题不在中间点，而在 steady 与 overall 的差。

## Boundary

- GPU/SSH: used only for Phase412 measurement artifacts.
- Large raw logs are stored compressed locally (`serve.log.gz`, `metrics.jsonl.gz`); remote raw originals remain unchanged.
- Runtime/PerfDatabase/gate: not modified.
- Phase405 penalty: not read.
- Default AIC: No-Go.
