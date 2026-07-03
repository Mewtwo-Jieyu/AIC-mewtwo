# Phase407 DP lockstep joint simulation

Phase407 supersedes Phase406's circular penalty check, and the genuine joint loop does not pass the DP2 gates. The uncoupled harness matches Phase401, but the symmetric max-token coupling produces about 1.0x penalty instead of the required 1.78-1.88x, so pad-to-max alone is not a sufficient runtime fix.

## Rule

- phase405_penalty_read=false: the model does not read Phase405 duty, active-iter, or reconstructed penalty fields.
- uncoupled mode uses the existing CBSimulator path as the harness gate.
- coupled mode runs two replicas with independent schedulers and shared wall time; each step charges padded non-attention at `max(tokens_A, tokens_B)` plus `max(attn_A, attn_B)`.

## Gates

| scenario | row_type | uncoupled gate | coupled ratio | coupled gate | penalty | penalty gate | occupancy gate | verdict |
|---|---|---|---:|---|---:|---|---|---|
| K2.5-tp4ep8dp2-8k2k | dp2_joint_sim | passed | 1.931591 | failed | 1.012456 | failed | failed | independent_joint_sim_incomplete |
| K2.5-tp4ep8dp2-32k3k | dp2_joint_sim | passed | 1.908546 | failed | 0.988780 | failed | passed | independent_joint_sim_incomplete |
| K2.5-tp8ep8-32k3k | tp8_early_exit_control | passed | 0.780201 | not_applicable | 1.000000 | not_applicable | not_applicable | tp8_dp1_joint_sim_no_change |

## Interpretation

- Harness fidelity passed: uncoupled joint output matches Phase401 for both DP2 scenarios and the TP8 control.
- The coupled loop stays near the uncoupled DP2 output, so the model does not independently reproduce the real duty loss.
- This means Phase408 should not directly implement this symmetric lockstep rule. The next target is replica asymmetry / prefill occupancy behavior that creates the real running mean gap.

## Boundary

- report-only; runtime_modified=false, perf_database=false.
- gpu_allowed=false, ssh_allowed=false.
- valid_for_default=false, diagnostic_only=true, default_readiness=No-Go.
- Phase408 should only touch runtime if this genuine joint simulation is accepted.
