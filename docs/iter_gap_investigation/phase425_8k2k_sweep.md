# Phase425 8k2k Arrival Sweep

Phase425 镜像 Phase412，只对 DP2 8k2k 做 N=128 突发与 N=512 持续到达对照；不改 runtime、PerfDatabase 或 gate。

## Verdict

- verdict: `8k2k_burst_artifact_significant`
- Phase426 target: `decide_validation_arrival_mode_or_steady_gap_modeling`
- current validation baseline penalty: `2.553139`
- current validation steady penalty: `1.961257`
- current validation burst artifact multiplier: `1.301787`
- trace baseline penalty: `1.955651`

## Samples

| N | artifact | request ratio | overall penalty | steady penalty | bench tok/s/gpu | trace fidelity |
|---:|---|---:|---:|---:|---:|---|
| 128 | docs/iter_gap_investigation/phase403_dp2_stats/K2.5-tp4ep8dp2-8k2k | 1.723404 | 1.955651 | 1.711253 | 139.063168 | passed |
| 512 | docs/iter_gap_investigation/phase425_8k2k_sweep/K2.5-tp4ep8dp2-8k2k | 1.007843 | 1.647975 | 1.516977 | 165.026189 | passed |

## 32k3k Reference

| source | request ratio | overall penalty | steady penalty | verdict |
|---|---:|---:|---:|---|
| Phase412 32k3k | 1.000000 | 1.740990 | 1.357625 | arrival_sweep_partial_collapse |

## Validation Baseline

| source | real tok/s/gpu | sim tok/s/gpu | baseline penalty | N=512 overall penalty | N=512 steady penalty | artifact multiplier |
|---|---:|---:|---:|---:|---:|---:|
| Phase424 current validation | 137.716000 | 351.608074 | 2.553139 | 2.130620 | 1.961257 | 1.301787 |
| Phase403/407 trace baseline |  |  | 1.955651 | 1.647975 | 1.516977 | 1.289176 |

## Interpretation

- 当前 validate 的 8k2k `2.553139x` 口径可拆成 artifact multiplier `1.301787` 和 steady gap `1.961257`。
- Phase403/407 trace-baseline 口径是 `1.955651`，用于和 Phase412 方法保持可比；它不是当前 validate 的最大残差口径。
- 若 N=512 把 request ratio 拉平但 steady penalty 仍高，Phase426 需要在验收到达模式和稳态模型残差之间做口径决策。
- 本报告只提供测量证据；不替换验收参考，不打开 Default AIC。

## Boundary

- GPU/SSH: used only for Phase425 measurement artifacts.
- Runtime/PerfDatabase/gate: not modified.
- Phase405 penalty: not read.
- Default AIC: No-Go.
