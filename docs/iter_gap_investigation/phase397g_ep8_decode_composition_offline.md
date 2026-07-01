# Phase397g ep8/dp2 decode Composition Residual Attribution (Route delta step 4)

| Item | Result |
|---|---|
| Question | after 397f, why does the ep8 multi-config table still FAIL (2.17x) with a bidirectional residual (tp8ep8-8k2k 0.62x under, tp4ep8dp2-32k3k 2.17x over)? |
| Answer | three structural causes: (1) the flat 90ms ep8 overhead is a miscalibrated fudge that over-corrects decode-heavy configs; (2) dp2 throughput is normalized by num_gpus=tp instead of tp*dp; (3) the 32k3k long-context configs over-predict in the mixed/prefill path. |
| Runtime / table / Default AIC | not modified (verdict-only, offline, no GPU/SSH, no overhead tuning) |

## 1. ep8 overhead is a miscalibrated blunt fudge

`validate_cb_simulator.py` feeds a flat **90ms/iter** overhead only to the ep8 multi-config path (`ep8_per_iteration_overhead_ms` default 90.0, line 1096; `_make_cb_config` -> `CBSimConfig`; charged in `iteration_latency` when `decode_bs>0`). The tp16 main table uses 0.

| Config | ratio @ovh0 | ratio @ovh90 | abs-err @0 | abs-err @90 | note |
|---|---|---|---|---|---|
| tp8ep8-8k2k | 1.16x | 0.62x | 1.16x | 1.61x | REGRESSED |
| tp8ep8-32k3k | 1.95x | 1.24x | 1.95x | 1.24x | helps |
| tp4ep8dp2-8k2k | 1.88x | 1.09x | 1.88x | 1.09x | helps |
| tp4ep8dp2-32k3k | 3.22x | 2.17x | 3.22x | 2.17x | helps |

At `overhead=0` EVERY ep8 config over-predicts (1.12x-3.22x), so the 90ms was calibrated to the pre-397f under-count. Post-397f it over-corrects decode-heavy `tp8ep8-8k2k` (1.16x -> 0.62x). The physically owed per-iter EP comm there is `real 119.8ms - compose 100.2ms ~= 20ms`, not 90ms.

## 2. dp2 per-GPU normalization bug

`_run_agg_cb_sim` normalizes with `num_gpus=tp`, but attention-dp configs run on `tp*dp` physical GPUs.

| Config | compose (no ovh) | real_iter(ng=tp) | real_iter(ng=tp*dp) | compose/real(ng=tp*dp) |
|---|---|---|---|---|
| tp8ep8-8k2k | 100.2 ms | 119.8 ms | 119.8 ms | 0.84x |
| tp8ep8-32k3k | 149.8 ms | 304.9 ms | 304.9 ms | 0.49x |
| tp4ep8dp2-8k2k | 116.4 ms | 232.4 ms | 116.2 ms | 1.00x |
| tp4ep8dp2-32k3k | 166.6 ms | 600.6 ms | 300.3 ms | 0.55x |

For `tp4ep8dp2-8k2k` the pure-decode compose (116.39ms) matches the real decode iter at `num_gpus=tp*dp=8` (116.18ms, **1.00x**), not at `num_gpus=tp=4` (0.50x). Dividing dp2 sim throughput by `dp` lands `tp4ep8dp2-8k2k` at 0.94x.

## 3. Per-config attribution

| Config | tp | dp | shape | dominant cause |
|---|---|---|---|---|
| tp8ep8-8k2k | 8 | 1 | 8k2k | ep8_overhead_90ms_overcorrects |
| tp8ep8-32k3k | 8 | 1 | 32k3k | longcontext_mixed_prefill |
| tp4ep8dp2-8k2k | 4 | 2 | 8k2k | dp2_pergpu_normalization |
| tp4ep8dp2-32k3k | 4 | 2 | 32k3k | dp2_normalization_plus_longcontext |

## Verdict -- Route delta step 4

- The ep8 residual is NOT a single decode scaling error.
- (i) The flat 90ms ep8 overhead is a miscalibrated fudge (owed comm ~20ms, not 90ms). (ii) dp2 throughput is normalized by `num_gpus=tp` instead of `tp*dp`. (iii) The 32k3k long-context configs over-predict in the mixed/prefill path.
- Default AIC remains **No-Go**.
- Next: **phase397h_ep8_decode_composition_fix** -- replace the flat 90ms ep8 overhead with a structural per-rank EP dispatch/combine comm term and fix the dp per-GPU normalization (`num_gpus=tp*dp`); then a mixed-path phase for the 32k3k long-context residual. Re-validate the multi-config table offline.

## Discipline

- Verdict-only, offline: runtime / PerfDatabase not modified; the ep8 overhead was only SWEPT ({0,90}) as a diagnostic via the existing CLI switch, not tuned or persisted; no scope gating; no GPU/SSH; Default AIC No-Go.
- Raw evidence: `docs/iter_gap_investigation/phase397g_overhead_sensitivity_raw.csv`, `docs/iter_gap_investigation/phase397g_residual_attribution_raw.csv`.
