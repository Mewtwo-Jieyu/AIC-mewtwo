# Phase397h ep8 dispatch-comm Mechanism Verification (Route delta step 5)

> REFUTATION: this phase was chartered to CONFIRM the corrected 397g root cause ("EP all-to-all communication is under-modeled"). Read-only forensics REFUTED it. The vLLM dispatch path is additive (allreduce + cross-dp comm), the measured ep8 comm module is ~13ms all-layer, the sim MoE matches the measured module, and the sim decode composition (~124ms) matches the measured modules -- so 397g's 232ms "real decode iter" was a prefill-inflated aggregate, not a decode iter. The residual reattributes to PREFILL / MIXED-ITERATION accounting.

| Item | Result |
|---|---|
| Question | is the ep8 residual really under-modeled EP all-to-all comm (corrected 397g), and can it be fixed structurally in 397i? |
| Answer | NO. Comm is small and correctly structured; the over-prediction grows with isl (prefill signature). 397i pivots to prefill/mixed accounting; the equal-isl tp4 residual needs real per-iter TPOT. |
| Runtime / table / Default AIC | not modified (verdict-only, offline, no GPU/SSH, no overhead tuning) |

## 1. Dispatch branch: vLLM path is ADDITIVE, not if/elif

The configs use the vLLM backend, whose `MoEDispatch.query` (`operations.py`:776-788) charges `if attention_tp>1: += allreduce` THEN `if attention_dp>1: += dp_comm` -- additive, not the trtllm `if/elif`. So tp4ep8dp2 already pays both the allreduce and the cross-dp all_gather/reduce_scatter term. The 397h premise ("never reaches all-to-all") is false for this backend.

| config | attention_tp | attention_dp | decode_tokens | branch | sim dispatch |
|---|---|---|---|---|---|
| tp8ep8dp1 | 8 | 1 | 128 | vllm_additive_allreduce_only(dp=1) | 2.042ms |
| tp4ep8dp2 | 4 | 2 | 256 | vllm_additive_allreduce+dp_allgather/reducescatter | 4.789ms |

## 2. Measured ep8 comm module is small

v0.19.0 ships a measured module `ep8_comm_dispatch_combine` (topology tp4dp2ep8): 0.2890ms/call at bucket 241. Calibrating the per-call -> all-layer multiplier from the sim's own MoE (generation_moe 84.76ms / measured fusedmoe 1.889ms/call = 44.9 calls) gives measured all-layer comm ~= 13.0ms. The sim synthetic dispatch is 4.789ms, so the sim under-counts comm by only ~8ms -- far below the 397g "owed ~116ms". **EP comm is not the gap.**

## 3. MLA attention is correctly TP-independent

`num_heads//tp` doubles 8 -> 16 from tp8 to tp4 (num_heads=64), yet `generation_attention` is flat (21.74 vs 21.52ms at bs128; 40.05 vs 40.30ms at bs256) and scales with bs, not tp. MLA attention is KV/latency-bound here and must NOT be scaled by 1/tp.

## 4. Owed magnitude: 232ms was prefill-inflated

The budget-breakdown steady-state decode iter matches the measured composition; the aggregate-derived `bs/(real_out*tp)` figure is much larger because it folds prefill wall-time into a fake "decode iter".

| config | tp | dp | shape | sim steady iter | aggregate-derived iter | @ovh0 |
|---|---|---|---|---|---|---|
| tp8ep8-8k2k | 8 | 1 | 8k2k | 103.0ms | 119.8ms | 1.16x |
| tp8ep8-32k3k | 8 | 1 | 32k3k | 156.7ms | 304.9ms | 1.95x |
| tp4ep8dp2-8k2k | 4 | 2 | 8k2k | 123.6ms | 232.4ms | 1.88x |
| tp4ep8dp2-32k3k | 4 | 2 | 32k3k | 186.3ms | 600.6ms | 3.22x |

For tp4ep8dp2-8k2k the measured composition (MoE 84.8 + measured comm ~13 + attention 21.5 + dense ~5 ~= 124ms) matches the sim steady iter 123.6ms, so the 232.4ms aggregate figure is prefill-inflated, not a decode iter.

## 5. The over-prediction grows with isl (prefill signature)

- tp8ep8: 1.16x @8k2k -> 1.95x @32k3k
- tp4ep8dp2: 1.88x @8k2k -> 3.22x @32k3k

Over-prediction growing with isl at fixed tp is a prefill/mixed-iteration accounting fingerprint, not a fixed per-decode-iter comm gap.

## Verdict -- Route delta step 5

- The corrected-397g root cause (under-modeled EP all-to-all comm) is **REFUTED**: the dispatch is additive, measured comm ~13ms all-layer, sim MoE matches the measured module, and sim decode composition matches measured.
- The ep8 residual is dominated by **prefill / mixed-iteration accounting** (grows with isl). MLA attention and EP comm are confirmed NOT the cause.
- The residual tp4-vs-tp8 over-prediction at equal isl (1.88x vs 1.16x) is **not separable offline** -> NEEDS real per-iter TPOT for tp8dp1 and tp4dp2.
- Default AIC remains **No-Go**.
- Next: **phase397i_prefill_mixed_accounting** -- investigate the prefill / mixed-iteration accounting path (context cost per mixed iter, chunked-prefill scheduling), do NOT scale MLA attention, and take a real per-iter TPOT measurement to settle the equal-isl tp4 residual. The EP-comm fix is dropped.

## Discipline

- Verdict-only, offline: runtime / PerfDatabase not modified; no overhead tuning; no scope gating; no GPU/SSH; Default AIC No-Go. The measured ep8_comm module (v0.19.0) was only READ for an offline magnitude comparison; it was not bound into the validation DB.
- Raw evidence: `docs/iter_gap_investigation/phase397h_dispatch_branch_raw.csv`, `docs/iter_gap_investigation/phase397h_mla_tp_probe_raw.csv`.
