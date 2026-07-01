# Phase396 tp16 Standing Over-count Attribution (offline, verdict-only)

| Item | Result |
|---|---|
| Question | where (absolute ms) is the standing tp16 ~1.5x over-count? |
| Answer | **decode-side and attention-bound** (0.12.0 gen_attn vs KV) |
| Runtime / PerfDatabase / GPU | not touched (verdict-only) |
| Default AIC | No-Go |

## Step 0: metric reconciliation (3.4x gap closed)

- Canonical basis: `tokens/s/gpu = steady_output_tokens / (steady_time_ms/1000) / num_gpus`, num_gpus = tp*pp*dp = 16.
- run_agg real steady run (10k-3k b=128): steady_iters=5874, steady_wall=796040 ms, steady_output_tokens~=751531 -> 59.01 tok/s/gpu (matches validate).
- Root cause of the Phase395 ~201 vs 59 gap: the diagnose `--cb-trace-out --expand-skips` harness under-counts the decode wall ~6.6x (expands pure-decode skips at near-constant latency), whereas the real sim charges the KV-growth trapezoid (`_estimate_decode_skip_latency`). `--no-expand-skips` reproduces a same-order steady decode wall.
- Phase395's RELATIVE merged-vs-split conclusions still hold; only its ABSOLUTE trace throughput was on the wrong basis (corrected here).

## Attribution (canonical run_agg basis)

- Steady state is ~pure decode: `avg_prefill_reqs_per_iter = 0.08` -> the over-count is in the DECODE path, not prefill. No real-TTFT split needed (steady throughput gap == decode wall gap).
- Each decode iteration is 100% attention-bound:

| Scenario | decode_bs | gen_attn ms | gen_non_attn ms | attn share |
|---|---|---|---|---|
| 10k3k_b128 | 126 | 27.17 | 6.82 | 100% |
| 10k2k_b32 | 28 | 11.48 | 5.15 | 100% |
| 16k2k_b32 | 28 | 17.25 | 5.15 | 100% |

  With `overlap_factor=0` the pure-decode total = `max(...)` = gen_attn; MoE/GEMM (gen_non_attn) is masked.
- Over-count magnitude: sim steady wall 796040 ms vs real-implied 524810 ms (= 751531 tok / (89.5 * 16)) = **1.517x** -- exactly the throughput gate.

## Verdict

- The standing tp16 ~1.5x over-count is **decode-side and attention-bound**: the 0.12.0 decode attention latency vs KV (`gen_attn`), accumulated over ~3000 decode steps via the KV-growth trapezoid. NOT MoE/GEMM, NOT prefill, NOT created by merged.
- Opposite-sign backends (0.12.0 over-predicts, measured 0.19.0 tp4dp2ep8 under-predicts) => pure data magnitude, not a shared composition bug.
- Phase397 (`phase397_route_beta_measured_decode_attention_vs_kv_table`): Route beta -- measure a decode-attention-vs-KV table for Kimi tp16 on the module-boundary (0.19.0) schema and unify the two perf backends. Route gamma (freeze legacy 0.12.0 as diagnostic) is the fallback.

## No-Go discipline held

- verdict-only: runtime / operations / PerfDatabase not modified; no fudge tuning; no scope gating; no GPU/SSH; Default AIC remains No-Go.
