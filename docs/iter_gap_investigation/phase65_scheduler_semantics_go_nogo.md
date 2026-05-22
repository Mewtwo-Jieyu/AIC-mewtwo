# Phase 65 Scheduler Semantics Go / No-Go

## Conclusion

Phase65 is No-Go for compare CLI. The strict compare interface is valid, but cb_sim cannot yet produce vLLM-source-equivalent DP scheduler rows. The current output is useful as a semantics gap artifact only.

| Gate | Result |
|---|---|
| `(alignment_key, dp_rank)` strict join | Go |
| vLLM Phase62 DP row source | Go |
| cb_sim DP-aware scheduler semantics | No-Go |
| compare CSV for modeling input | No-Go |
| default latency path | unchanged |

## Gap Artifact

| artifact | result |
|---|---|
| `phase65_scheduler_semantics_gap.csv` rows | `4078` |
| `shape_mismatch` | `4003` |
| `missing_vllm` | `75` |
| `shape_match` | `0` |

The mismatch count includes graph-mode mismatch because cb_sim currently emits `AIC_UNSET` and vLLM emits concrete `NONE` / `FULL`. That is intentional: graph regime is part of the scheduler/runtime descriptor boundary.

## Decision

| Question | Answer |
|---|---|
| Is cb_sim step equal to vLLM DP step? | No |
| Is cb_sim mixed cut the same as vLLM chunked prefill? | No |
| Is `forward_token_count` from true vLLM padded runtime shape? | No |
| Is this only a descriptor builder bug? | No; source scheduler semantics are missing |
| Can Phase64 compare be enabled? | No |

## Stop Rules Preserved

| Stop rule | Status |
|---|---|
| No intersection-only compare | Preserved |
| No phase ordinal compare | Preserved |
| No latency or residual field | Preserved |
| No profiler / NCCL / sync / throughput field | Preserved |
| No `PerfDatabase`, `run_static`, or `IterationLatencyCalculator` integration | Preserved |

## Phase66 Entry Criteria

Only enter Phase66 if the goal is to design a vLLM-like scheduler descriptor generator, not to patch compare output.

| Requirement | Reason |
|---|---|
| DP-local scheduler sequence | Needed to match `engine_core_dp_step` |
| request split from the same scheduling semantics | Needed to compare `scheduled_*_reqs` |
| token split from the same scheduling semantics | Needed to compare `scheduled_*_tokens` |
| runtime padded forward shape | Needed to compare `forward_token_count` / `forward_regime` |
| concrete cudagraph mode | Needed to compare runtime graph path |

If these fields cannot be generated without guessing vLLM internals, keep the descriptor line diagnostic-only and do not enable compare CLI.
