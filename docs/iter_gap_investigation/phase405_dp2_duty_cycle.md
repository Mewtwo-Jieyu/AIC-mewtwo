# Phase405 DP2 duty-cycle attribution

Phase405 explains the DP2 throughput gap as duty-cycle loss plus a smaller active-iteration term. TP8 is retained as the control and does not show the same DP duty gap.

## Decomposition

| scenario | row_type | observed ratio | duty ratio | active iter ratio | reconstructed | error pct | raw batch ratio | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| K2.5-tp4ep8dp2-8k2k | dp2_duty_attribution | 1.955651 | 1.363822 | 1.307665 | 1.783422 | 8.806692 | 1.534967 | duty_cycle_plus_active_iter_reconstructs_dp2_gap |
| K2.5-tp4ep8dp2-32k3k | dp2_duty_attribution | 1.887133 | 1.496021 | 1.255802 | 1.878707 | 0.446503 | 1.149428 | duty_cycle_plus_active_iter_reconstructs_dp2_gap |
| K2.5-tp8ep8-32k3k | tp8_control | 0.780201 |  |  | 0.921785 | 18.147015 | 1.047349 | tp8_control_no_dp_duty_gap |

## Interpretation

- raw_batch_occupancy_ratio is the direct sim avg decode batch over real running mean check; it is diagnostic but not sufficient alone because running includes non-output active time.
- duty_cycle_ratio uses real running mean, real output throughput, and Phase404 active decode iter latency to capture the wall-clock duty loss.
- reconstructed_tput_ratio = duty_cycle_ratio * active_iter_ratio for DP2; residual under 10% means the two-factor model explains the observed gap.
- TP8-32k is the control: peak batch matches and no DP-specific duty verdict is emitted.

## Boundary

- gpu_allowed=false, ssh_allowed=false.
- runtime_modified=false, perf_database=false, valid_for_default=false, diagnostic_only=true, default_readiness=No-Go.
- Phase406 should target DP prefill occupancy / lockstep duty modeling, not PerfDatabase values.
