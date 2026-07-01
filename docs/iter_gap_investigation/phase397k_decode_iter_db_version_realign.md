# Phase397k 0.19 DB Version Realignment -- E2E Absolute-Magnitude Convergence

> REPORT ONLY. Phase397i/j deferred the ~0.6x absolute under-prediction (sim 82.8 vs "real" 133.5 for tp8ep8-8k2k) to "a separate DB-version issue", hypothesizing that the sim reading the stale **0.12.0** perf DB (instead of the freshly-collected **0.19.0** DB) was the cause. This phase freshly measures vLLM **0.19** serving throughput across all six `MULTI_CONFIG_DATA` points and A/Bs cb_sim on 0.12.0 vs 0.19.0 against that ground truth. **The stale-DB hypothesis is refuted.** The DB swap moves the sim by only 0.94x-1.32x (mean +9%), nowhere near the 2.5x-3.8x needed. The real driver of the absolute gap is twofold: (1) the frozen gate baseline is **vLLM 0.17** data, and vLLM **0.19 is ~3.2x faster** on Kimi-K2.5; (2) the `ep8_per_iteration_overhead_ms=90.0` constant (calibrated to 0.17) is ~2.4x the actual measured 0.19 per-iteration time (~37 ms). No gate constants changed.

| Item | Result |
|---|---|
| Question | Does swapping the sim perf DB 0.12.0 -> 0.19.0 close the absolute-magnitude gap vs freshly-measured vLLM 0.19? |
| Answer | NO (in any material sense). Max symmetric error 5.22x -> 4.87x, mean 3.88x -> 3.54x -- both DBs under-predict fresh 0.19 by ~3.5-5x. The DB swap is a ~9% mean sim change; the gap is dominated by baseline version + overhead constant, not the DB. |
| Runtime / table / Default AIC | Measured fresh 0.19 (128/128 OK all points); added offline A/B script + measured manifest; PerfDatabase files, `MULTI_CONFIG_DATA`, and all gate constants NOT changed. |

## 1. Config matrix (single 8-GPU node, ep8)

Mirrors `MULTI_CONFIG_DATA` in `scripts/validate_cb_simulator.py` (source of truth). All points are 8 total GPUs, model `moonshotai/Kimi-K2.5` served as `kimi-k2.5`.

| point | topology | tp | dp | moe_ep | isl | osl | batch | max_num_batched_tokens |
|---|---|---|---|---|---|---|---|---|
| K2.5-tp8ep8-8k2k | tp8ep8 | 8 | 1 | 8 | 8000 | 2000 | 128 | 8000 |
| K2.5-tp8ep8-32k3k | tp8ep8 | 8 | 1 | 8 | 32000 | 3000 | 128 | 32000 |
| K2.5-tp4ep8dp2-8k2k | tp4dp2ep8 | 4 | 2 | 8 | 8000 | 2000 | 128 | 8000 |
| K2.5-tp4ep8dp2-32k3k | tp4dp2ep8 | 4 | 2 | 8 | 32000 | 3000 | 128 | 32000 |
| K2.5-tp8ep8-8k2k-bt65536 | tp8ep8 | 8 | 1 | 8 | 8000 | 2000 | 128 | 65536 |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | 4 | 2 | 8 | 8000 | 2000 | 128 | 65536 |

tp16 (16-GPU cross-node) throughput points are out of scope (the single-node 0.19 DB collection has no 16-way comms).

## 2. Fresh vLLM 0.19 measurement (ground truth)

Serve: `vLLM 0.19.0`, `--enable-expert-parallel`, `--enable-chunked-prefill`, `--enable-prefix-caching`, `--data-parallel-size 2` for the dp2 variants. Standard points used `--gpu-memory-utilization 0.90 --max-model-len 262144 --max-num-seqs 256`. Client: `run_openai_fixed_shape_benchmark.py --num-prompts 128 --max-concurrency 128`, matching ISL/OSL. All points completed 128/128 requests, 0 failures.

`output_tok_s_gpu = output_tok_s / world_size` (world = tp*dp = 8); this equals `total_tok_s_gpu * osl/(isl+osl)`, the same derivation the frozen baseline uses.

| point | wall_s | output_tok_s | total_tok_s | measured output tok/s/gpu | ~per-iter ms (out_tok_s/conc) |
|---|---|---|---|---|---|
| K2.5-tp8ep8-8k2k | 74.1 | 3453.4 | 17267.2 | **431.7** | 37.1 |
| K2.5-tp8ep8-8k2k-bt65536 | 74.0 | 3457.5 | 17287.7 | **432.2** | 37.0 |
| K2.5-tp8ep8-32k3k | 241.7 | 1589.1 | 18539.1 | **198.6** | 80.6 |
| K2.5-tp4ep8dp2-8k2k | 92.6 | 2763.3 | 13816.4 | **345.4** | 46.3 |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 147.7 | 1733.1 | 8665.6 | **216.6** | 73.9 |
| K2.5-tp4ep8dp2-32k3k | 279.9 | 1371.9 | 16005.2 | **171.5** | 93.3 |

Manifest: `docs/iter_gap_investigation/phase397k_measured_0190_manifest.csv`. Raw per-point trees (meta.json + bench_result.json + serve.log): `docs/iter_gap_investigation/phase397k_measured_0190/`.

Two measurement notes:
- **Point `tp4ep8dp2-8k2k-bt65536` was measured under constrained serve params** (`--gpu-memory-utilization 0.75 --max-model-len 12288 --max-num-seqs 128`) because a co-tenant left a 22 GiB orphan on GPU4 that pushed the standard 0.90/262144/256 config into CUDA-graph-capture OOM. For a 10 000-token (8k+2k) workload at concurrency 128 these caps do not bind on steady-state decode, so the number is a valid throughput point, but it is not byte-identical serve config to the other five.
- **The large token budget hurts the dp2 case**: `tp4dp2ep8-8k2k` drops 345.4 -> 216.6 tok/s/gpu going bt8000 -> bt65536, whereas `tp8ep8-8k2k` is flat (431.7 -> 432.2). The oversized 65536 budget with dp2 changes scheduling/batch composition; the tp8 (dp1) path is unaffected at this concurrency.

## 3. A/B convergence table (measured 0.19 vs sim@0.12 vs sim@0.19)

cb_sim aggregate predictor (`method="cb_sim"`, `database_mode=HYBRID`, `overlap_factor=0.0`, ep8 auto `per_iteration_overhead_ms=90.0`), run twice per point on `PerfDatabase(version="0.12.0")` and `version="0.19.0")`. Symmetric error `max(sim/real, real/sim)` (same `_abs_error` as the gate). `frozen` = the current `MULTI_CONFIG_DATA` baseline (**0.17** data).

| point | frozen(0.17) | measured(0.19) | sim@0.12 | sim@0.19 | err012 | err019 | 19/12 |
|---|---|---|---|---|---|---|---|
| K2.5-tp8ep8-8k2k | 133.5 | 431.7 | 82.8 | 88.9 | 5.21x | 4.85x | 1.07 |
| K2.5-tp8ep8-32k3k | 52.5 | 198.6 | 64.8 | 60.6 | 3.06x | 3.28x | 0.94 |
| K2.5-tp4ep8dp2-8k2k | 137.7 | 345.4 | 82.7 | 94.8 | 4.17x | 3.64x | 1.15 |
| K2.5-tp4ep8dp2-32k3k | 53.3 | 171.5 | 57.1 | 75.1 | 3.01x | 2.28x | 1.32 |
| K2.5-tp8ep8-8k2k-bt65536 | 138.5 | 432.2 | 82.8 | 88.7 | 5.22x | 4.87x | 1.07 |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 156.0 | 216.6 | 82.6 | 94.6 | 2.62x | 2.29x | 1.15 |
| **MAX symmetric error** | | | | | **5.22x** | **4.87x** | |
| **MEAN symmetric error** | | | | | **3.88x** | **3.54x** | |

CSV: `docs/iter_gap_investigation/phase397k_ab_convergence.csv`.

The script's literal verdict is "CLOSES" only because `max(err019) 4.87 < max(err012) 5.22`. That is a ~7% shave on a 5x error -- not a real convergence.

## 4. Root cause: why the DB swap does not close the gap

**(a) The frozen baseline is 0.17; vLLM 0.19 is ~3.2x faster.** `scripts/validate_cb_simulator.py` header states the data source: `Kimi-K2.5, vLLM 0.17`. The frozen `real_total_tok_s_gpu` values (667.64, 612.14, ...) are 0.17 measurements. Fresh 0.19 is 2.5x-3.8x higher per point (mean ~3.1x):

| point | frozen 0.17 | fresh 0.19 | 0.19/0.17 |
|---|---|---|---|
| tp8ep8-8k2k | 133.5 | 431.7 | 3.23x |
| tp8ep8-32k3k | 52.5 | 198.6 | 3.79x |
| tp4dp2ep8-8k2k | 137.7 | 345.4 | 2.51x |
| tp4dp2ep8-32k3k | 53.3 | 171.5 | 3.22x |
| tp8ep8-8k2k-bt65536 | 138.5 | 432.2 | 3.12x |
| tp4dp2ep8-8k2k-bt65536 | 156.0 | 216.6 | 1.39x |

So the sim (calibrated near the 0.17 era at ~83 tok/s/gpu) was already ~0.6x the 0.17 baseline; against the ~3.2x-faster 0.19 target it is now ~5x low. The DB version is not the lever -- the *baseline version* moved.

**(b) The `ep8_per_iteration_overhead_ms=90.0` constant dominates.** Measured 0.19 per-iteration time for tp8ep8-8k2k is ~37 ms (3453 out_tok_s / 128 concurrency). The sim's fixed 90 ms ep8 overhead alone is **2.4x that entire measured iteration**. The sim's reported tpot is ~180 ms (0.12.0) / ~181 ms (0.19.0) vs the measured ~37 ms -- a 4.9x gap that the DB swap only nudges (194 -> 181 ms, -7%). The 90 ms floor, calibrated against slower 0.17 iterations, caps sim throughput far below 0.19.

**(c) The DB swap effect is real but small and mixed.** Swapping 0.12.0 -> 0.19.0 changes the kernel-table compute composition by 0.94x-1.32x (it even *slows* the sim at tp8ep8-32k3k, 19/12 = 0.94). This is a second-order kernel-latency refresh, not the 3x-5x correction the absolute magnitude needs.

> Modeling note: both DB versions were forced onto the standard kernel-table path (`model.model_name` override in the A/B script) so the comparison is like-for-like. Without it, `operations._is_vllm_module_scope()` auto-routes only the (0.19.0, tp4dp2ep8) combination to an experimental Kimi module-runtime perf table that enforces discrete token buckets and cannot run the aggregate arbitrary chunk sizes -- an orthogonal phase342-style path, not the kernel DB re-collected here.

## 5. Verdict and follow-up (report only)

- **Verdict: the stale-DB hypothesis is refuted.** Realigning the sim perf DB 0.12.0 -> 0.19.0 does not close the absolute-magnitude gap against measured vLLM 0.19 (max error 5.22x -> 4.87x; still ~3.5-5x under). The gap is driven by the gate baseline being 0.17 data (0.19 is ~3.2x faster) and by the 90 ms ep8 per-iteration overhead constant being ~2.4x the measured 0.19 iteration, not by the perf DB version.
- **A true 0.19 realignment is a two-part recalibration, deferred to a follow-up phase:** (1) re-freeze `MULTI_CONFIG_DATA` `real_total_tok_s_gpu` to the fresh 0.19 manifest in this phase (and update the header data-source line 0.17 -> 0.19); (2) recalibrate `ep8_per_iteration_overhead_ms` downward (90 ms is ~2.4x the measured ~37 ms floor; a value in the ~15-35 ms range is where the arithmetic lands) before re-freezing `THROUGHPUT_MAX_ACCEPTANCE` / `MULTI_CONFIG_MAX_ACCEPTANCE` / `TTFT_MAX_ACCEPTANCE`. Doing (1) without (2) would just invert the error sign.
- **No constants touched this phase**, per plan: `MULTI_CONFIG_DATA`, `THROUGHPUT_MAX_ACCEPTANCE`, `MULTI_CONFIG_MAX_ACCEPTANCE`, `TTFT_MAX_ACCEPTANCE`, and the PerfDatabase files are unchanged.

## 6. Reproduction

Measurement (on an 8-GPU H200 node, `vLLM 0.19.0`, weights at `.../models--moonshotai--Kimi-K2.5`):

```bash
# from WORKDIR=/mnt/shared-storage-user/ailab-sys/zhaojieyu/aic
bash collector/vllm/run_phase397k_agg_bench.sh
# constrained variant used for the one orphan-affected point:
GPU_MEMORY_UTILIZATION=0.75 MAX_MODEL_LEN=12288 MAX_NUM_SEQS=128 \
  bash collector/vllm/run_phase397k_agg_bench.sh
```

Offline A/B (CPU):

```bash
python scripts/analyze_phase397k_db_version_ab.py build-measured \
  --results-dir docs/iter_gap_investigation/phase397k_measured_0190 \
  --out docs/iter_gap_investigation/phase397k_measured_0190_manifest.csv
python scripts/analyze_phase397k_db_version_ab.py ab \
  --measured docs/iter_gap_investigation/phase397k_measured_0190_manifest.csv \
  --out docs/iter_gap_investigation/phase397k_ab_convergence.csv
```

## 7. Artifacts produced

- `collector/vllm/run_phase397k_agg_bench.sh` -- 6-point vLLM 0.19 serving driver (graceful DP teardown, orphan-PID filtering, skip-if-done).
- `scripts/analyze_phase397k_db_version_ab.py` -- offline `build-measured` + `ab` A/B harness.
- `docs/iter_gap_investigation/phase397k_measured_0190/` -- raw per-point trees (128/128 OK each).
- `docs/iter_gap_investigation/phase397k_measured_0190_manifest.csv` -- flat measured manifest.
- `docs/iter_gap_investigation/phase397k_ab_convergence.csv` -- A/B convergence table.
