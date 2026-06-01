# Phase122: Default AIC Gate Design for MoE WNA16

## Conclusion

Phase117/118 的 exact-key table 只能作为 experimental query table。它还不能进入默认 AIC，因为它只覆盖 4 个 synthetic token shape，没有真实 activation 分布，也没有端到端误差闭环。

| Item | Decision |
|---|---|
| Current artifact | Experimental exact-key table for 128/248/512/1024 tokens |
| Default AIC status | No-Go |
| Phase123 allowed work | Evidence acquisition plan only |
| Forbidden work | No PerfDatabase write, no run_static change, no IterationLatencyCalculator change |

## Gate Rules

| Gate | Minimum requirement before default AIC |
|---|---|
| Shape coverage | Cover MoE token buckets that really appear from vLLM scheduler/runtime rows, not only 128/248/512/1024 synthetic shapes |
| Topology coverage | Validate the target default topology. Current evidence only speaks for tp4dp2ep8 |
| Input distribution | Separate synthetic_random_hidden_states from real loaded-model activations |
| Exact key | Key must include backend, version, device, model, kernel, dtype, config SHA, topology, hidden, intermediate, experts, topk, and token shape |
| Query policy | Missing key must fail. No interpolation, extrapolation, or empirical fallback for this table |
| End-to-end check | Prove default integration improves end-to-end error without residual, profiler, NCCL trace, or sync-wait constants |
| Rollback | Default path must be disableable and must return to the Phase4 baseline |

## Current Gaps

| Area | Current state | Required before default |
|---|---|---|
| Token shapes | 128/248/512/1024 only | Runtime bucket distribution and enough measured buckets |
| Inputs | synthetic_random_hidden_states | Real activation evidence or a written proof that synthetic input is acceptable |
| Model scope | Direct MoE aggregate only | Integration proof inside full AIC iteration accounting |
| Query | Experimental CSV query | Default runtime key mapping and fail-fast behavior |
| Validation | Diagnostic module timing | End-to-end error closure and holdout |

## Stop Rules

| Stop condition | Action |
|---|---|
| Any request to use the Phase116 linear fit as a default formula | Stop |
| Any request to write Phase117 table into default PerfDatabase now | Stop |
| Any integration requiring run_static or IterationLatencyCalculator change before evidence | Stop |
| Any gap filled by residual, profiler, NCCL trace, sync wait, fallback timing, or random-weight timing | Stop |
| Any missing exact key handled by interpolation or fallback | Stop |

## Phase123 Entry

Phase123 may design evidence acquisition for real activation and token-bucket coverage. It must not implement default AIC integration.
