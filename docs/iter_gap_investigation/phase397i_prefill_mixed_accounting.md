# Phase397i ep8 Baseline Realignment + Real per-iter TPOT (Route delta step 6)

> REVERSAL: 397c-397h chased an ep8 'over-prediction' (1.16x-3.22x) that was measured against a **stale vLLM 0.17 real baseline**. Realigning to the in-repo Phase164 **0.19.0** real (same shapes, same H200 box) flips the sign: the sim **under-predicts**. An authorized read-only per-iteration TPOT capture confirms the sim's decode iteration is **~3x too slow** vs 0.19.0.

| Item | Result |
|---|---|
| Question | is the ep8 over-prediction real, or an artifact of the stale 0.17 baseline? |
| Answer | ARTIFACT. vs 0.19.0 the sim under-predicts (tp8 0.36x, tp4dp2 0.74x); its decode iter is ~3x too slow. |
| Runtime / table / Default AIC | not modified (verdict-only). GPU/SSH used ONLY for a read-only 0.19.0 per-iter TPOT capture; nothing written to the validation real source or PerfDatabase. |

## 1. Baseline realignment: 0.17 -> 0.19.0 flips the sign

In-repo Phase164 (vLLM 0.19.0, H200, identical shapes) measured much higher real throughput than the 0.17 gate. This phase reproduced the two 8k2k reruns (tp8ep8 434.96 / tp4dp2ep8 348.50 tok/s/gpu), matching Phase164.

| config | tp | dp | real 0.17 | real 0.19.0 | sim @ovh0 | sim/0.17 | sim/0.19.0 |
|---|---|---|---|---|---|---|---|
| tp8ep8-8k2k | 8 | 1 | 133.53 | 437.89 | 155.21 | 1.16x | 0.354x |
| tp8ep8-8k2k-bt65536 | 8 | 1 | 138.47 | 432.36 | 154.95 | 1.12x | 0.358x |
| tp4ep8dp2-8k2k | 4 | 2 | 137.72 | 343.64 | 258.69 | 1.88x | 0.753x |
| tp4ep8dp2-8k2k-bt65536 | 4 | 2 | 155.95 | 401.20 | 256.70 | 1.65x | 0.640x |

Every config flips: apparent over-prediction vs 0.17, actual under-prediction vs 0.19.0. The 0.19.0 ordering (tp8 437 > tp4dp2 343 per-gpu) is also the opposite of the sim (tp4dp2 259 > tp8 155).

## 2. Real per-iter TPOT: sim decode iter is ~3x too slow

`--enable-logging-iteration-details` emits, per scheduler step: `Iteration(N): C context requests, C context tokens, G generation requests, G generation tokens, iteration elapsed time: X ms`. Pure-decode iters (0 context, full batch) are the TPOT ground truth (tp4dp2 split by DP rank).

| config | real decode iter (0.19.0) | p99 | sim steady iter | sim/real | sim out/real |
|---|---|---|---|---|---|
| tp8ep8-8k2k | 34.29ms (bs~128/replica) | 36.90ms | 103.0ms | 3.00x | 0.36x |
| tp4ep8dp2-8k2k | 44.12ms (bs~64/replica) | 46.46ms | 123.6ms | 2.80x | 0.74x |

tp8ep8 pure-decode = 34.3ms at bs128 (p99 36.9); tp4dp2ep8 = ~44ms/replica (DP0 44.09 @bs72, DP1 44.15 @bs56). The sim's steady decode iter (103ms / 123.6ms) is ~3x larger.

## 3. Root cause: over-sized decode iteration

- **Decode-MoE magnitude (primary).** At tp8dp1 the decode token-count is correct (128 = real bs), yet the sim `generation_moe` (~71.8ms, 0.12.0 DB) ALONE exceeds the entire real 0.19.0 decode iter (34.3ms). The 0.12.0-DB MoE latency over-states 0.19.0 decode MoE by ~2-3x.
- **dp batch-split / token-count (secondary).** tp4dp2 charges decode/MoE at bs*attention_dp=256 tokens on one bs=128 replica, while real splits concurrency across 2 DP replicas (~64 tokens/replica).
- This **refutes 397h**'s reconciliation that the sim decode composition (~124ms) matched the measured modules: that used a per-call fusedmoe module x ~45 layers, which over-counts. The measured decode iter is ~3x below the sim.

## 4. Mechanisms A/B are minor (and not the fix target vs 0.19.0)

- **Mechanism A** (`max()` drops context attention at overlap_factor=0): real and isl-growing, but only ~4-17% of steady wall (tp8 0.042@8k -> 0.109@32k; tp4 0.048@8k -> 0.170@32k). It makes the sim FASTER, so vs 0.19.0 it REDUCES the under-prediction gap -- fixing it would worsen the error.
- **Mechanism B** (prefill wall diluted by batch-skip): the raw skip/steady fraction is confounded (skip wall spans warmup); steady throughput is dominated by pure-decode skip wall, so prefill under-charge has limited leverage.

## Verdict -- Route delta step 6

- The ep8 'over-prediction' of 397c-397h was a **stale-0.17 baseline artifact**. Against in-repo 0.19.0 real the sim **under-predicts** (tp8 0.36x, tp4dp2 0.74x).
- The sim's **decode iteration is ~3x too slow** vs 0.19.0 (real 34ms/44ms vs sim 103ms/124ms), dominated by decode-MoE magnitude (0.12.0 DB) with a secondary dp token-count error.
- Mechanisms A/B are minor and are NOT fix targets vs 0.19.0.
- Default AIC remains **No-Go**.
- Next: **phase397j_decode_iter_baseline_realign** -- migrate the multi-config real baseline to 0.19.0 (drop 0.17), fix the over-sized decode iteration (decode-MoE magnitude + dp token-count), and decide 0.12.0-DB vs 0.19.0 version alignment.

## Discipline

- Verdict-only: simulator runtime / PerfDatabase NOT modified; no overhead/fudge tuning; no scope gating; Default AIC No-Go.
- The ONLY runtime-adjacent change is the collector benchmark harness adding the `--enable-logging-iteration-details` measurement flag (v0.19.0 serve). It changes logging only, not vLLM compute or the simulator.
- GPU/SSH authorized this round for a READ-ONLY per-iter TPOT capture; no real data written into the validation source or PerfDatabase.
- Note: the 0.19.0 weights (`models--moonshotai--Kimi-K2.5`) load as compressed-tensors WNA16-Marlin MoE; the runner's recorded `quantization=fp8` is its hardcoded default. Aggregate throughput still matches Phase164.
- Raw evidence: `phase397i_maxdrop_attn_raw.csv`, `phase397i_prefill_wallshare_raw.csv`, `phase397i_real_tpot_raw.csv`, `phase397i_tpot/*/decode_iter_extract.txt`.
