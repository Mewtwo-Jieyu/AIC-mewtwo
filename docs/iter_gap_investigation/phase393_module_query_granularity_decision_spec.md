# Phase393 Module Query-Granularity Decision Spec (offline, verdict-only)

| Item | Result |
|---|---|
| Workload | 10k2k_b32 / bt8192 / tp4dp2ep8 / Kimi-K2.5 / vLLM 0.19.0 |
| Root cause | 3-pass queries MoE/EP8 at the prefill chunk token count, not the merged batch |
| Verdict | **adopt merged-batch query granularity for MoE/EP8** |
| Mixed reachability | split 0/118 (0.0%) -> merged 110/118 (93.2%) |
| Default AIC | No-Go |
| Runtime / PerfDatabase / GPU | not touched |

## Root cause

- `scheduler.py` reserves 1 token per decode request, so the mixed prefill chunk = `max_num_batched_tokens - decode_bs` (8191, 8190, ...).
- `iteration_latency._compute_3pass` Pass 1 queries the MoE/EP8 module with `isl=prefill_tokens` (the split chunk). Correct for attention, wrong for the token-parallel fused MoE.
- `phase124_moe_activation_rows.csv` shows the real vLLM fused MoE `tokens_actual = 8192` (merged batch), not 8191. So the materialized buckets are merged-batch token counts.

## Confirmation probe (offline, exact-only, no GPU)

| Phase | n | split granularity exact-hit | merged granularity exact-hit |
|---|---|---|---|
| prefill | 2 | 2/2 | 2/2 |
| mixed | 118 | 0/118 (0.0%) | 110/118 (93.2%) |

- Merged batch token == 8192 for 112/118 mixed iters; the 8 misses are non-saturated tail iterations (merged token != 8192).

## Candidates

1. **Merged-batch granularity (recommended primary)**: query MoE/EP8 at `prefill_tokens + decode_bs`; attention stays split. This is a query-granularity correction, not a lookup relaxation and not a scheduler rewrite. Lifts mixed reachability 0% -> 93%.
2. **Bucketization contract (fallback)**: round non-saturated tail iterations up to the nearest materialized bucket via an explicit documented quantization. Only if merged granularity leaves gaps.
3. **Scheduler alignment (last resort)**: rewrite cb_sim scheduler token accounting. Highest risk; defer unless needed.

## Verdict and next step

- Adopt **merged-batch query granularity** for the MoE/EP8 module lookup. It resolves the Phase392 reachability blocker for 93% of mixed iterations without relaxing exact-only or using GPU.
- Phase394: make the runtime change (`phase394_merged_granularity_runtime_change_and_bare_error_remeasure`) and re-measure the bare error against `compare_10k2k_b32_dp0.csv` for prefill/mixed.
- Residual: the 8 non-saturated tail iters and the decode 2.01x (small-bucket MoE + EP8 all2all comm) are handled separately; decode goes to a later GPU measurement phase.

## No-Go discipline held

- exact-only model contract unchanged; merged granularity is querying the correct token count, not nearest/interpolation/extrapolation.
- no fudge tuning; runtime (`operations.py` / `iteration_latency.py` / `vllm_backend.py`) and PerfDatabase not modified this phase.
- no GPU/SSH used; Default AIC remains No-Go.
