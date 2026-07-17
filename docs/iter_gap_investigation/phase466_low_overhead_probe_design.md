# Phase466 low-overhead probe design

结论：本地设计与解析器通过，GPU 开销门尚未执行。该 probe 默认关闭，只使用 vLLM 0.19.0
自带的 `--enable-logging-iteration-details` 和 Prometheus preemption counter，不修改 vLLM 源码，
不采 per-request 高频事件。当前状态是 `GPU_GATE_PENDING`，Default AIC 继续 `No-Go`。

## Scope

| 项 | 值 |
|---|---|
| base commit | `9ba5ad93ca8805a1eb427593ced8cee01aea6804` |
| tooling result commit | `7480099f3033befd0f05ed1f17950cf4fe3df183` |
| branch | `experiment/phase466-low-overhead-probe` |
| hardware/runtime | H200 SXM / vLLM 0.19.0 / Kimi-K2.5 |
| implementation | stock vLLM aggregate iteration details; no source patch |
| result | local design PASS; GPU gate pending |
| flags | `diagnostic_only=true`; `valid_for_default=false`; `perf_database=false` |

Changed files at the tooling commit:

- `scripts/analyze_phase466_low_overhead_probe.py`
- `tests/unit/scripts/test_analyze_phase466_low_overhead_probe.py`

The analyzer emits the exact serve and benchmark argv, parses one aggregate row per engine iteration,
joins rank-local preemption deltas, validates source hashes and process cleanup, and refuses formal collection
when the off/on throughput delta exceeds `2%`.

The parser writes both `iteration_rows.csv` and `rank_summary.csv`. Iteration rows use the same field names as
the residual-attribution contract: `rank_id`, `iteration_seq`, `iteration_start_offset_ms`,
`iteration_end_offset_ms`, scheduled prefill/decode tokens, workload digest and cumulative-token progress window.

## Measurement contract

| Stage | Scenario | Protocol | Decision |
|---|---|---|---|
| overhead off/on | `K2.5-tp4ep8dp2-8k2k-bt65536` | N128/C128 | absolute output-throughput delta must be `<=2%` |
| formal 1 | `K2.5-tp4ep8dp2-32k3k` | N512/C128 | run only after overhead PASS |
| formal 2 | `K2.5-tp8ep8-8k2k-bt65536` | N512/C128 | topology control, not a pure DP control |
| formal 3 | `K2.5-tp4ep8dp2-8k2k-bt65536` | N512/C128 | run only after overhead PASS |

Each iteration row carries `run_id`, `source`, `workload_cohort_digest`, rank-local elapsed offsets,
scheduled-token progress bounds and a 65,536-token progress-window id. Real/simulator comparison must use
the same workload digest and cumulative-token progress window. Absolute clocks and naked iteration indices
are not alignment keys.

The off/on artifact validator requires exact request/token counts, vLLM 0.19.0 source hashes, empty GPU and
process residue, two DP rank summaries, non-resetting preemption counters and matching workload digests.
Any mismatch fails immediately.

This stock probe does not expose prefill chunk histograms, fresh/recompute/resume state, queue occupancy,
decode KV sums or simulator component costs. Therefore a passing overhead gate and formal collection can screen
rank/timing hypotheses, but cannot by itself close all Track C disproof conditions or select a Phase467 model.
If those missing fields remain necessary, the result stays `INCONCLUSIVE`; a separate low-overhead field design
must pass its own off/on gate before use.

## Prior evidence boundary

| Evidence | Hash | Reuse decision |
|---|---|---|
| `phase462_bt65536_metric_and_composition_audit.md` | `820a3ddbeb07244d5e249f92ee25d23c85e5b6c1007d9a4c6d713d663d65861e` | v2/v3/v4 deltas `13.8486% / 8.7915% / 8.2043%`; inadmissible |
| `phase463_six_point_latency_recollect.md` | `826149442cb93a43194cdf476a8677c9d4fb824d8c19a349fa73819cb0ccd5e8` | formal measurements used the stock flag, but had no logging-off control |

Phase463 proves the stock field exists and can be parsed. It does not prove the flag is below the `2%`
overhead limit. Phase462 custom composition logging cannot be reused because all three variants failed their
own gate.

## Verification

```text
PYTHONPYCACHEPREFIX=/tmp/phase466_b_pyc python3 -m py_compile \
  scripts/analyze_phase466_low_overhead_probe.py \
  tests/unit/scripts/test_analyze_phase466_low_overhead_probe.py

PYTHONPYCACHEPREFIX=/tmp/phase466_b_pyc \
  /Users/mewtwo/2026/work/codebase/AIC-mewtwo/.worktrees/feature-pr403/.venv312/bin/python \
  -m pytest -q tests/unit/scripts/test_analyze_phase466_low_overhead_probe.py

PYTHONPYCACHEPREFIX=/tmp/phase466_b_pyc python3 \
  scripts/analyze_phase466_low_overhead_probe.py plan \
  --output-json /tmp/phase466_b_plan.json
```

Result: `py_compile` PASS, analyzer plan generation PASS, `7 passed`, `git diff --check` PASS.
No SSH or GPU command was run.

## Gate and next action

The first allowed remote action is only the serialized N128/C128 DP2-bt65536 off/on gate. If its absolute
throughput delta is above `2%`, stop and record FAIL; do not start the N512 runs. If it passes, collect only
the three preregistered formal scenarios after Track A is reviewed and integrated.

Forbidden reuse: this design is not GPU evidence, not a PerfDatabase row, not a latency model, and not
evidence for enabling Default AIC.
