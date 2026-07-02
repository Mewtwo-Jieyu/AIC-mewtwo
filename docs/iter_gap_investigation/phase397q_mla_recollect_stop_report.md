# Phase397q MLA Recollect Stop Report

Phase397q did not satisfy the stated b128 acceptance gate, so it stops before any perf table write.

## Result

| Check | Result |
|---|---|
| Planned target | heads8 / batch128 / s9001 / block64 / float16 |
| Serve anchor | 0.260256 ms per call |
| b128 graph p50 | 0.475328 ms |
| b128 ratio vs serve | 1.826x |
| Verdict | stop, no `generation_mla_perf.txt` update |

## What Changed In Collector

- Added Phase397q-only profiler output for `forward_mqa`.
- Added Phase397q-only full-cudagraph scheduler metadata mode.
- Added `--max-num-splits` sweep support.
- Did not change runtime, models, gate, or perf tables.

## Evidence

| Probe | graph p50 ms | scheduler metadata | max splits | Interpretation |
|---|---:|---|---:|---|
| b128, no full-cg metadata | 0.475168 | absent | 0 | old microbenchmark path remains slow |
| b128, full-cg metadata | 0.475328 | present, shape 257 | 32 | metadata alone does not reach serve |
| b128, best split sweep | 0.472192 | present | 1 | split count is not the root fix |
| b64, full-cg metadata | 0.243456 | present, shape 129 | 32 | within 8% of 0.260256 ms serve anchor |

The b64 result is useful only as diagnosis. It does not authorize writing a b128 row, because the original Phase397q gate explicitly required b128 to hit the serve anchor.

## Stop Reason

The evidence points to a batch-semantics mismatch: the serve trace anchor that was treated as heads8/b128 behaves like the b64 microbenchmark shape. Until the serve profiler row is mapped to the exact effective batch shape, replacing `generation_mla_perf.txt` would encode the wrong key.

## Artifacts

- Raw directory: `docs/iter_gap_investigation/phase397q_mla_recollect/`
- b128 full-cg run: `phase397q_mla_split_instrument_fullcg.csv`
- split sweep: `phase397q_mla_split_sweep_*.csv`
- batch check: `phase397q_mla_batch_check_*.csv`
- b64 profiler: `phase397q_mla_b64_acceptance_profiler.txt`
- residue proof: `gpu_compute_apps_after.txt`, `process_residual_after.txt`

## Next Step

Do not run the full generation_mla grid yet. First write a follow-up batch-shape attribution spec that maps the Phase397l serve profiler calls to effective per-rank batch size, then rerun the acceptance gate with the corrected shape.

Default AIC remains No-Go.
