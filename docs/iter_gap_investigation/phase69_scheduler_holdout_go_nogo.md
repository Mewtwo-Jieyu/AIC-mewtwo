# Phase 69 Scheduler Holdout Go/No-Go

## Status

Phase69 captured the `3k3k_b128 bt8192 tp4dp2ep8` holdout on the updated `tc88x` pod. The remote marker run succeeded, but the current Phase66 generator does not support this short-ISL scheduler shape.

| gate | current status |
|---|---|
| remote entry | `ws-faaf0de74ef9a14d-worker-tc88x` |
| real vLLM holdout rows | captured: `24008` raw marker rows |
| deduped `(alignment_key, dp_rank)` rows | generated: `6002` rows |
| Phase66 generator attempt | fail-fast: `vLLM-like scheduler descriptor requires a chunked prefill prompt` |
| strict gap audit | not generated |
| default `cb_sim` | unchanged |

## Holdout Shape Summary

| item | observed |
|---|---|
| benchmark | `128/128` requests succeeded |
| phase split | prefill `2`, mixed `1`, pure_decode `5999` |
| DP split | DP0 `3002`, DP1 `3000` |
| DP0 first rows | `step0 prefill 3000+0 / NONE:3000`; `step1 decode 0+1 / NONE:1`; `step2 mixed 496+1 / PIECEWISE:512` |
| DP1 first rows | `step0 prefill 3512+0 / NONE:3512`; `step1 decode 0+65 / PIECEWISE:512`; `step2 decode 0+65 / FULL:72` |
| tail | DP0 ends with `FULL:64`; DP1 remains `FULL:72` |

## Decision Rule

| result | decision |
|---|---|
| holdout rows captured and Phase66 generator matches | Go to Phase70 generalization |
| holdout rows captured but generator fails | No-Go for generalization; fix generator semantics first |
| holdout rows cannot be captured | No-Go; do not use residual, profiler, or old runtime-shape artifacts |

## Current Decision

Phase69 is No-Go for strict holdout comparison. Two facts are separate:

| fact | meaning |
|---|---|
| real vLLM `3k3k_b128` descriptor artifact exists | the remote capture path works for descriptor-only rows |
| local generator rejects `3k3k_b128` | the current first-evidence generator only supports chunked-prefill shapes where `isl > max_num_batched_tokens` |

No fallback CSV is generated. The next valid action is to design a new generator rule for short-ISL shapes using the captured vLLM rows as the reference. It is still not valid to use phase ordinal, intersections, latency, residual, profiler, NCCL trace, sync wait, or throughput to force a match.
