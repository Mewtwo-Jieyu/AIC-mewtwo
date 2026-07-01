# Phase397e decode Per-Iteration Composition Offline Verdict (Route delta step 2)

| Item | Result |
|---|---|
| Question | after 397d unmasked a decode non-attention under-count, is the `generation_moe` table aggregate or per-rank, and is `iteration_latency.py`'s `/tp_size` a double-count? |
| Answer | table is **per-rank** (TP in the lookup key); the `/tp_size` is a **double-count**; and `overlap_factor=0` `max()` wrongly drops the serial non-attention term. Both are coupled decode-composition errors. |
| Runtime / table / Default AIC | not modified (verdict-only, offline, no GPU/SSH) |

## A. `generation_moe` table semantics = per-rank

- `perf_database.py` `load_moe_data` / `query_moe` / `query_vllm_module`: no `/tp` at load or query; TP is encoded in the `moe_tp_size` (and topology) key.
- `collector/vllm/collect_moe.py`: one GPU, weights sharded by `(moe_tp_size, moe_ep_size)`, FULL `num_tokens` fed to that rank; `log_perf` records single-GPU latency (empirical: 32 tokens, moe_tp=1 ~0.688ms vs moe_tp=16 ~0.094ms -> TP shrinks per-rank latency via the KEY).
- `operations.py` `MoE.query`: generation MoE passes `moe_tp_size` as a key and scales only by `num_layers*mtp`; no `/tp` divisor.

## B. Double-count

`iteration_latency.py::_split_generation_non_attention` (live; pure-decode branch) divides every op whose name contains `generation_moe` by `tp_size` AGAIN. `_scale_generation_non_attention` (lines 92-109) is dead code.

| Scenario | gen_non_attn raw | gen_non_attn /tp | raw/div |
|---|---|---|---|
| 10k3k_b128 | 51.10 ms | 6.89 ms | 7.4x |
| 10k2k_b32 | 32.42 ms | 5.28 ms | 6.1x |
| 16k2k_b32 | 32.42 ms | 5.28 ms | 6.1x |

## C. Per-iteration reconciliation (vs real decode-iter budget)

`real_decode_iter_ms = batch / (real_out_tok_s_gpu * num_gpus) * 1000`.

| Scenario | current (/tp,max) | raw+max | raw+SUM | real decode iter |
|---|---|---|---|---|
| 10k3k_b128 | 26.6 ms (0.30x) | 51.1 ms (0.57x) | 77.7 ms (0.87x) | 89.4 ms |
| 10k2k_b32 | 12.2 ms (0.30x) | 32.4 ms (0.81x) | 44.6 ms (1.11x) | 40.2 ms |
| 16k2k_b32 | 17.9 ms (0.37x) | 32.4 ms (0.68x) | 50.3 ms (1.05x) | 48.0 ms |

Current (`/tp` then `max`) under-counts ~3x (non-attention is both crushed by `/tp` AND masked by `max()`). Raw non-attention + `max(attn)` is still under. Raw non-attention **serial-summed** with attention lands within ~13% of real at all three gates -- decode attention and MoE/FFN run serially within a layer, so they should SUM.

## Two coupled errors (both owed to 397f)

1. **`/tp` double-count** on `generation_moe*` (per-rank value divided by TP again).
2. **`overlap_factor=0` `max()`** for pure decode drops the smaller of attention / non-attention; the two are serial and should sum.

397c's "overlap ruled out" was narrowly true only because the `/tp` under-count had already masked the non-attention term, so max-vs-sum did not matter at the time.

## Verdict -- Route delta step 2

- Owed decode per-iteration term = `gen_non_attn_raw + gen_attn` (serial), which reconciles to real +/-13% across the gates.
- Next: **phase397f_decode_periter_composition_fix** -- runtime fix that (i) drops the `generation_moe*` `/tp` double-count, (ii) makes pure-decode composition a serial sum, and (iii) removes the dead `_scale_generation_non_attention`; then re-validate 397d+397e+397f offline on the 3 gates and the validate throughput table.

## Discipline

- Verdict-only, offline: runtime / PerfDatabase not modified; no fudge tuning; no scope gating; no GPU/SSH; Default AIC remains No-Go.
- Raw probe evidence at `docs/iter_gap_investigation/phase397e_periter_reconciliation_raw.csv`.
