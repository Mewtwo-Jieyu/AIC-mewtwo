# Phase397f decode Per-Iteration Composition Runtime Fix (Route delta step 3)

| Item | Result |
|---|---|
| Change | pure-decode-local runtime fix in `iteration_latency.py`: drop the `/tp` double-count on `generation_moe*`, sum decode attention + non-attention serially, remove dead `_scale_generation_non_attention` |
| Throughput | max 4.13x -> 1.44x (gate 1.499): PASS |
| Multi-config | max 3.26x -> 2.17x (gate 1.470): FAIL |
| TTFT | max 1.79x (gate 1.790): PASS (prefill-side, unchanged) |
| Default AIC | **No-Go** |

## Runtime change (only file: iteration_latency.py, pure-decode branch)

1. `_split_generation_non_attention`: dropped `/ tp_size` on `generation_moe*` -- the perf-database values are per-rank (TP is in the `moe_tp_size` lookup key), so dividing again was a double-count (phase397e).
2. pure-decode branch of `_compute_3pass`: `total = generation_non_attn + gen_attn` (serial sum) instead of `_combine_with_overlap(...) = max(...)`. `overlap_factor` still governs the mixed / pure-prefill branches.
3. removed dead `_scale_generation_non_attention`.

## Per-iteration reconciliation after fix (tp16 gates)

| Scenario | per-iter after (serial sum) | real decode iter | ratio |
|---|---|---|---|
| 10k3k_b128 | 77.7 ms | 89.4 ms | 0.87x |
| 10k2k_b32 | 44.6 ms | 40.2 ms | 1.11x |
| 16k2k_b32 | 50.3 ms | 48.0 ms | 1.05x |

## Full-table before/after (offline validate_cb_simulator)

| Metric | before (397e HEAD) | after (397f) | gate | result |
|---|---|---|---|---|
| Throughput max | 4.13x | 1.44x | 1.499 | PASS |
| Throughput mean | 2.43x | 1.18x | - | - |
| Multi-config max | 3.26x | 2.17x | 1.470 | FAIL |
| Multi-config mean | 1.75x | 1.47x | - | - |
| TTFT max | 1.79x | 1.79x | 1.790 | PASS |

## Per-config (multi-config) before/after

| Config | ratio before | ratio after | abs-err before | abs-err after | note |
|---|---|---|---|---|---|
| K2.5-tp8ep8-8k2k | 1.02x | 0.62x | 1.02x | 1.61x | REGRESSED |
| K2.5-tp8ep8-32k3k | 1.79x | 1.24x | 1.79x | 1.24x | improved |
| K2.5-tp4ep8dp2-8k2k | 1.82x | 1.09x | 1.82x | 1.09x | improved |
| K2.5-tp4ep8dp2-32k3k | 3.26x | 2.17x | 3.26x | 2.17x | improved |
| K2.5-tp8ep8-8k2k-bt65536 | 0.98x | 0.60x | 1.02x | 1.67x | REGRESSED |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 1.61x | 0.96x | 1.61x | 1.04x | improved |

## Verdict -- Route delta step 3

- The tp16 decode throughput path (the original ~1.5x over-count target) is now structurally consistent and passes its gate (max 4.13x -> 1.44x). Every tp16 gate improved.
- The expert-parallel (ep8 / dp2) multi-config path improved (3.26x -> 2.17x max, mean 1.75x -> 1.47x) but still fails, and two `tp8ep8-8k2k` cases over-corrected to ~0.6x. Root: `_get_tp_size()` returns the ATTENTION tp, so the removed `/tp` never matched ep8 MoE sharding, and ep8 decode has different per-rank semantics (`moe_ep` vs `moe_tp`).
- Default AIC remains **No-Go** (multi-config gate fails).
- Next: **phase397g_ep8_decode_periter_composition** -- forensics + fix for the ep8/dp2 decode composition; re-validate the multi-config table offline. Tightening the acceptance gates (frozen at the pre-397d ~1.5x behaviour) is a candidate but is NOT applied here.

## Discipline

- Runtime change limited to the pure-decode composition in `iteration_latency.py`; `overlap_factor`, mixed/prefill branches, `simulator.py`, and the PerfDatabase are untouched.
- No fudge tuning, no scope gating, no GPU/SSH, no acceptance-gate changes.
- Raw evidence: `docs/iter_gap_investigation/phase397f_fulltable_before_after_raw.csv`, `docs/iter_gap_investigation/phase397f_periter_after_raw.csv`.
