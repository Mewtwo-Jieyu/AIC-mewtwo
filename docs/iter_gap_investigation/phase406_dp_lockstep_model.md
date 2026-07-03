# Phase406 DP lockstep structural model

Phase406 validates DP pad-to-max lockstep as the structural model for the DP2 duty loss. Coupling the two replicas by max step tokens brings both DP2 scenarios back to the clean real baseline within 10%.

## Mechanism

- pad_rule_source: `vllm/v1/worker/dp_utils.py:_post_process_dp_padding:L78-L90;vllm/v1/worker/dp_utils.py:_synchronize_dp_ranks:L153;vllm/v1/worker/gpu_model_runner.py:L3615-L3640`.
- serve prerequisites checked from Phase403 logs: data_parallel_size=2, enforce_eager=false, chunked prefill enabled, FULL_AND_PIECEWISE CUDA graph mode, and both DP workers present.
- paper model: consume the Phase405 scheduler/metrics decomposition and charge the uncoupled DP2 output by the deterministic `step_tokens = max(tokens_dp0, tokens_dp1)` lockstep penalty. This is not a runtime implementation.

## Result

| scenario | row_type | uncoupled ratio | penalty | coupled ratio | error pct | verdict |
|---|---|---:|---:|---:|---:|---|
| K2.5-tp4ep8dp2-8k2k | dp2_lockstep_paper_model | 1.955651 | 1.783422 | 1.096572 | 9.657196 | dp_pad_to_max_lockstep_model_matches_real |
| K2.5-tp4ep8dp2-32k3k | dp2_lockstep_paper_model | 1.887133 | 1.878707 | 1.004485 | 0.448489 | dp_pad_to_max_lockstep_model_matches_real |
| K2.5-tp8ep8-32k3k | tp8_early_exit_control | 0.780201 | 1.000000 | 0.780201 | 21.979865 | tp8_dp1_no_lockstep_change |

## Boundary

- report-only; runtime_modified=false, perf_database=false.
- gpu_allowed=false, ssh_allowed=false.
- valid_for_default=false, diagnostic_only=true, default_readiness=No-Go.
- Phase407 target: implement DP lockstep coupling in runtime only if this report is accepted.
