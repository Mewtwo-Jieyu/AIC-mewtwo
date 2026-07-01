# Phase397d decode-skip Trapezoid Seed Structural Fix (Route delta, step 1)

| Item | Result |
|---|---|
| Fix | `simulator.py::_estimate_decode_skip_latency` now computes BOTH trapezoid endpoints as pure-decode iterations (`start_avg_kv_len` and `start_avg_kv_len + skip_iters`); the `first_iter_latency_ms=iter_lat` seed (the triggering mixed/prefill iteration) is removed. |
| Nature | structural, deterministic, no tuning; runtime modified but gated (`valid_for_default=false`). |
| Gate (10k3k_b128) | sim/real throughput 0.66x (before, too slow) -> 2.65x (after, too fast). |
| Scope | claims skip accounting is structurally correct; does NOT claim throughput accuracy. Default AIC stays No-Go until 397e. |

## Before/after gate throughput (canonical run_agg cb_sim, offline)

| Scenario | real out | sim/real before | sim/real after | crosses 1.0 |
|---|---|---|---|---|
| 10k3k_b128 | 89.5 | 0.66x | 2.65x | yes |
| 10k2k_b32 | 49.8 | 0.69x | 2.22x | yes |
| 16k2k_b32 | 41.7 | 0.75x | 1.87x | yes |

## Why the prediction flips direction (expected)

397c showed the standing miss was two compensating errors:

1. the skip-trapezoid seed inflated the decode wall ~3.2x (sim too slow), and
2. the per-iteration decode was under-counted ~3x (`generation_moe /tp_size` hidden under attention by `overlap_factor=0` max), which partially cancelled (1).

397d removes (1) only. With nothing left to compensate the still-present under-count (2), sim wall drops too far and throughput crosses from ~0.66x real (too slow) to ~1.9-2.7x real (too fast) -- exactly the ~2.6x reversal 397c predicted for a trapezoid-only fix. This is expected and gated.

## Verdict -- Route delta step 1

- The decode-skip trapezoid is now structurally correct (pure-decode ramp), with no empirical tuning.
- Next: **phase397e_decode_periter_composition_offline** -- offline determination of the `generation_moe` table semantics (aggregate vs per-rank) to decide whether the `/tp_size` scaling in `iteration_latency.py` double-counts and whether `overlap_factor=0` max masks a real decode non-attention term; then re-validate 397d+397e together before re-opening the accuracy gate.

## Discipline

- `runtime_modified=true` (the structural fix) but `valid_for_default=false`, `fudge_factor_tuning_used=false`, `scope_gating_used=false`, no PerfDatabase write, no real-data write, no GPU/SSH; Default AIC remains No-Go.
- Raw before/after evidence at `docs/iter_gap_investigation/phase397d_gate_before_after_raw.csv`.
