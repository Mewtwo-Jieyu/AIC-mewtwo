# Phase408 DP Replica Asymmetry Attribution

Phase408 is diagnostic-only. It reads Phase403 per-engine metrics and Phase407 uncoupled/real outputs; it does not read Phase405 penalty rows and does not modify runtime, PerfDatabase, or gates.

## Verdict

Replica asymmetry has enough magnitude to explain the missing DP2 penalty: a phase offset over the observed per-engine prefill-token streams can move the Phase407 uncoupled prediction back to the Phase403 real throughput for both DP2 scenarios.

## DP2 Results

| scenario | corr_all | corr_active | needed_penalty | phi0_penalty | target_phi | target_penalty | adjusted_ratio | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| K2.5-tp4ep8dp2-8k2k | 0.585182 | -0.267058 | 1.955651 | 1.281668 | 79 | 1.953176 | 1.001267 | replica_asymmetry_sufficient_missing_variable |
| K2.5-tp4ep8dp2-32k3k | 0.846859 | 0.334434 | 1.887133 | 1.140526 | 294 | 1.890607 | 0.998162 | replica_asymmetry_sufficient_missing_variable |

## Interpretation

- The target penalty is `phase407_uncoupled_joint_output_tok_s_gpu / real_output_tok_s_gpu`, not a Phase405 reconstructed penalty.
- The diagnostic knob is the observed independent per-engine completion stream plus circular phase offset; no shipped simulator class is changed.
- `phi0_penalty` is still far below the needed penalty; shifting one replica's prefill activity exposes the missing max-coupling cost.
- TP8 remains covered by Phase407's dp=1 early-exit control; Phase408 only tests DP2 replica asymmetry.
- Load imbalance remains a secondary item where mean prefill rate or active fraction differs by at least 20 percent; Phase409 should model replica phase first and keep load imbalance explicit.

## Boundary

- GPU/SSH: not used.
- Runtime/PerfDatabase/gate: not modified.
- Default AIC: No-Go.
