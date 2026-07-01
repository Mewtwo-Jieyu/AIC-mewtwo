# Phase397c tp16 Decode Accumulation/Composition Attribution (Route delta)

| Item | Result |
|---|---|
| Question | after 397b cleared the decode MLA-vs-KV table, WHERE does the standing tp16 ~1.5x live in the decode accumulation/composition? |
| Answer | the **skip-trapezoid** in `simulator.py _estimate_decode_skip_latency`, seeded with the TRIGGERING mixed/prefill iteration latency, net of a compensating per-iteration decode under-count |
| Gate (10k3k_b128) | over_count=1.517x; canonical steady wall is 94.7% decode-skip-trapezoid; decode wall alone (754175 ms) > real-implied TOTAL wall (524812 ms) |
| Runtime / table / Default AIC | not modified (verdict-only, offline, no GPU/SSH) |

## Ruled out up front (tp16 pure decode)

- `per_iteration_overhead_ms = 0` for tp16 (`_CB_SIM_DEFAULT_DECODE_OVERHEAD_MS = 0.0`; the `90.0` ms overhead applies only to 8-GPU EP8).
- `overlap_factor = 0.0` and `_combine_with_overlap(a,b) = max(a,b) + factor*min(a,b)` -> pure-decode `iter_lat = max(gen_non_attn, gen_attn)`; decode MoE is max'd, NOT serial-summed. "Missing overlap serial sum" is not the bug.
- MTP: the in-repo Kimi-K2.5 config has **no** `num_nextn_predict_layers` (`nextn=0`, `mtp_scale_factor=1.0`); the only in-repo real Kimi vLLM trace shows `speculative_config=None`, 1 token/step. Phase397b's `num_nextn_predict_layers=1` note (from the collector's DeepSeek-V3 fake config) is **RETRACTED**.

## Canonical run() steady-wall decomposition

| Scenario | over_count | steady_wall | decode(trapezoid) | mixed | trapezoid/expand-periter | decode_wall/real_implied |
|---|---|---|---|---|---|---|
| 10k3k_b128 | 1.517x | 796040 ms | 754175 ms (94.7%) | 5.3% | 3.19x | 1.437x |
| 10k2k_b32 | 1.458x | 496487 ms | 458200 ms (92.3%) | 7.7% | 2.80x | 1.345x |
| 16k2k_b32 | 1.341x | 545446 ms | 495461 ms (90.8%) | 9.2% | 2.08x | 1.218x |

## Root cause -- skip trapezoid seeded with the triggering mixed iteration

`simulator.py::_estimate_decode_skip_latency` computes the skipped pure-decode wall as `(first_iter_latency_ms + end_iter_latency_ms) * skip_iters / 2`. The caller passes `first_iter_latency_ms = iter_lat`, but `iter_lat` is the latency of the iteration that TRIGGERS the skip -- the one that just finished the last prefill chunk. That is a MIXED/prefill iteration carrying the ~isl-token prefill compute, not a pure-decode-at-start-KV latency.

| Scenario | skip first_lat (mixed) | skip end_lat (pure decode) | trapezoid eff/iter | true pure-decode/iter |
|---|---|---|---|---|
| 10k3k_b128 | 231.5 ms | 30.2 ms | 79.2 ms | 27.1 ms |
| 10k2k_b32 | 99.0 ms | 13.1 ms | 40.3 ms | 12.3 ms |
| 16k2k_b32 | 99.0 ms | 22.8 ms | 43.5 ms | 18.8 ms |

At 10k3k_b128 two skip events (bs=128, skip~2869, first_lat~232 ms) carry 99.99% of the decode wall; averaging 232 ms down to 30 ms across ~2869 *pure-decode* iters yields an effective ~79 ms/iter versus the true ~27 ms/iter -> ~3.2x decode-wall inflation.

## Compensating per-iteration decode under-count

- The trapezoid over-count (~3.2x) is partially compensated by a per-iteration decode UNDER-count: `gen_non_attn` (decode MoE) is divided by `tp_size` in `_scale_generation_non_attention` and then hidden under attention by `overlap_factor=0` `max()`, so the per-iter decode (~27 ms) is ~3x below the real-implied decode iteration (~85 ms).
- The two opposing errors nearly cancel and leave the standing ~1.5x. **Neither is fixable alone**: correcting only the trapezoid seed swings the prediction ~2.6x UNDER real; correcting only the per-iter composition swings it OVER.

## Verdict -- Route delta

- The standing tp16 ~1.5x is a **runtime accumulation/composition defect** (skip-trapezoid seeded with the triggering mixed-iteration latency) net of a compensating per-iteration decode under-count. It is **NOT** the decode table (cleared by 397b), **NOT** decode MoE as a standalone term (masked by `overlap_factor=0` max), **NOT** MTP, **NOT** per-iteration overhead.
- Next: **phase397d_decode_skip_trapezoid_seed_and_periter_composition_fix** -- a GATED runtime fix that corrects BOTH the trapezoid seed AND the per-iteration decode composition TOGETHER, then re-validates offline. No table write and no GPU required for the fix.

## No-Go discipline held

- Verdict-only, offline: runtime / operations / PerfDatabase not modified; no fudge tuning; no scope gating; no GPU/SSH; Default AIC remains No-Go.
- Raw canonical wall-split + skip-event evidence at `docs/iter_gap_investigation/phase397c_canonical_wall_split_raw.csv`.
