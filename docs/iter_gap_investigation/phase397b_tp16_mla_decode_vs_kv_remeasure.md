# Phase397b tp16 num_heads=4 Decode MLA-vs-KV Single-GPU Remeasure (Route beta)

| Item | Result |
|---|---|
| Question | are the 0.12.0 decode MLA-vs-KV values (num_heads=4) ~1.5x too high? |
| Answer | **No -- REFUTED.** fresh current-kernel is >= stored at the worst-over-count point |
| Gate scenario (10k3k_b128) | fresh/stored = **1.393** at b128, KV~11500 (fresh HIGHER) |
| Shipped table / runtime / Default AIC | not modified (Route A: measure+verdict only) |

## Measurement

- Single GPU (H200, driver 570.133.20), vLLM **0.19.0** (only version on node), `num_heads=4`, float16 KV, `vllm_flash_attn_mla`.
- Faithful path: collector `run_attention_torch` setup + `log_perf`, decode call adapted to 0.19.0 `FlashAttnMLAImpl.forward_mqa` (monolithic `.forward` removed). Provenance identical to stored rows (global num_heads=128, tp_size=32 -> local 4).
- CAVEAT: `forward_mqa` measures decode MQA attention over the paged latent cache; it EXCLUDES the small q-absorption GEMM + kv-write that the old monolithic 0.12.0 `forward` folded in. So fresh is a LOWER bound on the true current-kernel cost -- which only strengthens a refutation.
- Why 0.19.0 is the right reference: the real 89.5 tok/s/gpu benchmark was run on a modern vLLM; the sim currently READS a stale 0.12.0 table. Decode MLA is memory-bandwidth bound at the operating KV, so version drift is not ~1.5x.

## Point-wise decode-band comparison (num_heads=4, float16)

| batch | KV(step) | fresh 0.19.0 | stored 0.12.0 | fresh/stored |
|---|---|---|---|---|
| 128 | 8191 | 0.44054 | 0.32821 | 1.342 |
| 128 | 16383 | 0.85527 | 0.59606 | 1.435 |
| 128 | 32767 | 1.68142 | 1.14990 | 1.462 |
| 32 | 8191 | 0.12732 | 0.16859 | 0.755 |
| 32 | 16383 | 0.23046 | 0.25930 | 0.889 |

## Operating points (interpolated as the sim reads the table)

| Scenario | over-count | b/KV | fresh | stored | fresh/stored |
|---|---|---|---|---|---|
| 10k3k_b128 | 1.517x | b128/KV~11500 | 0.60807 | 0.43640 | 1.393 |
| 10k2k_b32 | 1.458x | b32/KV~11000 | 0.16269 | 0.19970 | 0.815 |
| 16k2k_b32 | 1.341x | b32/KV~17000 | 0.23829 | 0.29358 | 0.812 |

- Route beta would need fresh/stored ~= 0.667 (stored 1.5x too high) across the decode band, especially at the gate scenario. Observed: the gate scenario is **1.393** (fresh HIGHER, opposite sign), and the b32 scenarios are ~0.81 (partly stored high-KV noise) -- nowhere near 0.66.

## Verdict -- REFUTE Route beta

- The decode MLA-vs-KV per-point table values are **NOT** the source of the standing tp16 ~1.5x over-count. If anything the shipped 0.12.0 values are slightly LOW versus the current kernel.
- Phase397a already localized the over-count to the decode component; this measurement clears the per-point values, so the ~1.5x lives in the decode **accumulation / composition** layer:
  1. the KV-growth trapezoid decode-iteration count / effective KV trajectory;
  2. missing decode overlap (real vLLM overlaps memory-bound MLA attention under compute-bound MoE/dense GEMMs; the sim serial-sums with `overlap_factor=0`);
  3. missing MTP / speculative multi-token acceptance (real Kimi K2.5 config carries `num_nextn_predict_layers=1`, but the sim runs `nextn=0` / `mtp_scale=1.0`, so the real system emits >1 token per decode step).
- Next: Phase397c (`phase397c_route_delta_decode_accumulation_composition`) -- Route delta investigates the accumulation/composition, NOT the table. Route gamma (freeze legacy 0.12.0 as diagnostic) remains valid since the table is not the bug.

## No-Go discipline held

- Route A measure+verdict only: shipped `generation_mla_perf.txt` NOT written; runtime / operations / PerfDatabase not modified; no fudge tuning; no scope gating; Default AIC remains No-Go.
- Raw single-GPU measurement retained at `docs/iter_gap_investigation/phase397b_mla_num_heads4_raw_measurement.csv`.
