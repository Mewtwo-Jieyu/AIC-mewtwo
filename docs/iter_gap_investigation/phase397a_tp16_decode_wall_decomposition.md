# Phase397a tp16 Decode Wall Decomposition (offline, verdict-only)

| Item | Result |
|---|---|
| Question | which single locus holds the standing tp16 ~1.5x over-count? |
| Answer | **the 0.12.0 decode MLA-latency-vs-KV table values** (num_heads=4) |
| Runtime / PerfDatabase / GPU | not touched (verdict-only) |
| Default AIC | No-Go |

## Model constants (measured, kimi-k2.5 tp16 dp1)

- `num_layers = 61`, `nextn = 0` -> `mtp_scale_factor = 1.0`.
- `num_heads` global = 64; local heads at tp16 = 64/16 = **4** (Route-beta target MLA row).
- Decode attention is charged as `mla_per_layer * 61 * 1.0`. Layer count and MTP are the REAL Kimi values and cannot inflate by 1.5x -> RULED OUT.

## Canonical run_agg over-count + wall decomposition

| Scenario | steady wall ms | real-implied ms | over-count | tok/s/gpu | avg_prefill/iter | mixed % | decode(+skip) % | decode/1.5 vs real |
|---|---|---|---|---|---|---|---|---|
| 10k3k_b128 | 796040 | 524812 | 1.517x | 59.01 | 0.084 | 10.0% | 90.0% | 1.062x |
| 10k2k_b32 | 496487 | 340553 | 1.458x | 34.16 | 0.028 | 8.0% | 92.0% | 1.011x |
| 16k2k_b32 | 545446 | 406704 | 1.341x | 31.09 | 0.028 | 9.5% | 90.5% | 0.936x |

- Steady state is ~pure decode; the wall is 90..92% pure-decode (scheduled + KV-growth skip-trapezoid), 8..10% mixed, 0% prefill.
- Decode iteration latency is attention-bound: with `overlap_factor=0` the pure-decode total = `max(gen_attn, gen_non_attn)` = `gen_attn`; `gen_non_attn` (MoE/GEMM) is masked.

## Locate experiment

- Dividing ONLY the decode(=attention) component by 1.5 converges the canonical wall to the real-implied wall within ~6% for all three scenarios (1.062x / 1.011x / 0.936x). A uniform decode scaling closes the gap -> the 1.5x is CONCENTRATED in the decode/attention component, not broadly spread.

## Verdict

- By elimination, the standing tp16 ~1.5x over-count is the **0.12.0 decode MLA-latency-vs-KV table values** (`num_heads=4` for tp16), integrated over the decode KV trajectory via the trapezoid. It is NOT num_layers, NOT mtp_scale, NOT prefill, NOT mixed, and NOT a broad multi-component spread. The trapezoid is the (correct) accumulation vehicle, not the error.
- Phase397b (`phase397b_route_beta_single_gpu_mla_decode_vs_kv_remeasure`): Route beta -- a fresh **single-GPU** MLA-decode-vs-KV microbench (`num_heads=4`, collector shards heads on `WORLD_SIZE=1`) compared point-wise to the 0.12.0 table. **Cross-node tp16 is NOT required.** Route gamma (freeze legacy 0.12.0 as diagnostic) remains the fallback.

## No-Go discipline held

- verdict-only: runtime / operations / PerfDatabase not modified; no fudge tuning; no scope gating; no GPU/SSH; Default AIC remains No-Go.
