# Phase397g ep8/dp2 decode Composition Residual Attribution (Route delta step 4)

> CORRECTION (supersedes commit c935c9e5): the first cut claimed a "dp2 per-GPU normalization bug". That was WRONG -- dp cancels exactly in the per-GPU normalization. This corrected cut re-attributes the residual to under-modeled EP all-to-all communication.

| Item | Result |
|---|---|
| Question | after 397f, why does the ep8 multi-config table still FAIL (2.17x) with a bidirectional residual (tp8ep8-8k2k 0.62x under, tp4ep8dp2-32k3k 2.17x over)? |
| Answer | three causes: (1) the flat 90ms ep8 overhead is a miscalibrated fudge that over-corrects decode-heavy configs; (2) EP all-to-all comm is under-modeled (owed ~20ms tp8dp1 to ~116ms tp4dp2); (3) the 32k3k long-context configs over-predict in the mixed/prefill path. dp normalization is CORRECT. |
| Runtime / table / Default AIC | not modified (verdict-only, offline, no GPU/SSH, no overhead tuning) |

## 1. ep8 overhead is a miscalibrated blunt fudge

`validate_cb_simulator.py` feeds a flat **90ms/iter** overhead only to the ep8 multi-config path (`ep8_per_iteration_overhead_ms` default 90.0, line 1096; `_make_cb_config` -> `CBSimConfig`; charged in `iteration_latency` when `decode_bs>0`). The tp16 main table uses 0.

| Config | ratio @ovh0 | ratio @ovh90 | abs-err @0 | abs-err @90 | note |
|---|---|---|---|---|---|
| tp8ep8-8k2k | 1.16x | 0.62x | 1.16x | 1.61x | REGRESSED |
| tp8ep8-32k3k | 1.95x | 1.24x | 1.95x | 1.24x | helps |
| tp4ep8dp2-8k2k | 1.88x | 1.09x | 1.88x | 1.09x | helps |
| tp4ep8dp2-32k3k | 3.22x | 2.17x | 3.22x | 2.17x | helps |

At `overhead=0` EVERY ep8 config over-predicts (1.12x-3.22x), so the 90ms was calibrated to the pre-397f under-count. A single constant cannot cover a cost that scales with the topology: `tp8ep8-8k2k` owes ~20ms, `tp4ep8dp2-8k2k` owes ~116ms.

## 2. dp normalization is correct; the gap is un-modeled EP comm

The per-GPU throughput normalizes as `tokens_s_gpu = throughput_tok_s*(pp*dp)/(tp*pp*dp) = throughput_tok_s/tp` (`vllm_backend.py`:699-723; `simulator.py`:303), so **dp cancels** -- there is no normalization bug. Yet the real per-replica decode iter roughly doubles from tp8dp1 to tp4dp2 while the sim barely moves:

| op (8k2k, kv=9000, bs=128) | tp8 ms | tp4 ms | tp4/tp8 | reading |
|---|---|---|---|---|
| generation_moe | 71.776 | 84.764 | 1.18x | dominant ~72%; flat (EP fixed, tokens 128->256) |
| generation_attention | 21.736 | 21.523 | 0.99x | flat -- MLA latent KV, plausibly correct |
| generation_moe_pre_dispatch | 1.021 | 2.455 | 2.40x | comm proxy, tiny |
| generation_moe_post_dispatch | 1.021 | 2.334 | 2.28x | comm proxy, tiny |
| dense_gemm_sum | 4.159 | 4.863 | 1.17x | tiny |
| SIM_TOTAL | 99.714 | 115.939 | 1.16x | sim barely moves (1.16x) |
| REAL_DECODE_ITER | 119.830 | 232.360 | 1.94x | real ~2x -> owed cost is EP comm |

The only comm proxy (`generation_moe_pre/post_dispatch`) totals ~2-5ms -- far below the owed 20ms (tp8dp1) / 116ms (tp4dp2). The missing per-iter cost is un-modeled EP dispatch/combine all-to-all communication, which grows with `attention_dp * decode_bs` tokens and with cross-dp-group hops.

## 3. Per-config attribution

| Config | tp | dp | shape | owed comm | dominant cause |
|---|---|---|---|---|---|
| tp8ep8-8k2k | 8 | 1 | 8k2k | ~20ms | ep_comm_undermodeled_owes~20ms |
| tp8ep8-32k3k | 8 | 1 | 32k3k | ~155ms | longcontext_mixed_prefill |
| tp4ep8dp2-8k2k | 4 | 2 | 8k2k | ~116ms | ep_comm_undermodeled_owes~116ms |
| tp4ep8dp2-32k3k | 4 | 2 | 32k3k | ~434ms | ep_comm_plus_longcontext |

## Verdict -- Route delta step 4 (corrected)

- The ep8 residual is NOT a single decode scaling error and NOT a normalization bug (dp cancels).
- (i) The flat 90ms ep8 overhead is a miscalibrated fudge standing in for topology-dependent EP dispatch/combine communication. (ii) That EP all-to-all comm is under-modeled (modeled ~2-5ms vs owed 20-116ms). (iii) The 32k3k long-context configs over-predict in the mixed/prefill path.
- Default AIC remains **No-Go**.
- Next: **phase397h_ep8_decode_composition_fix** -- model EP dispatch/combine all-to-all comm structurally (per-rank, scaling with `attention_dp*decode_bs` and cross-group hops) in place of the flat 90ms; verify the `generation_moe` token count under `attention_dp`; then a mixed-path phase for the 32k3k residual. Re-validate the multi-config table offline.

## Discipline

- Verdict-only, offline: runtime / PerfDatabase not modified; the ep8 overhead was only SWEPT ({0,90}) as a diagnostic via the existing CLI switch, not tuned or persisted; no scope gating; no GPU/SSH; Default AIC No-Go.
- Raw evidence: `docs/iter_gap_investigation/phase397g_overhead_sensitivity_raw.csv`, `docs/iter_gap_investigation/phase397g_residual_attribution_raw.csv`, `docs/iter_gap_investigation/phase397g_tp_scaling_probe_raw.csv`.
