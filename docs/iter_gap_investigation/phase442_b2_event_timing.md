# Phase442 B2 Event Timing Gate

## Verdict

B2 graph-outer event timing is measurable, but not admissible for PerfDB ingestion in this form.

The overhead gate failed: patched run output throughput dropped from `1356.8445` to `1279.8277` tok/s, a `5.676%` slowdown against the `2.0%` limit. Stop at report-only; do not ingest `non_attn_total` rows.

## Run

| item | value |
|---|---:|
| scenario | `K2.5-tp4ep8dp2-8k2k` |
| protocol | N=512, C=128, ISL=8000, OSL=2000, prefix off |
| raw directory | `docs/iter_gap_investigation/phase442_b2_event_timing/overhead_gate_20260708_164017/` |
| off output tok/s | `1356.8445453862983` |
| on output tok/s | `1279.827655143875` |
| overhead | `5.6761764274547915%` |
| allowed overhead | `2.0%` |
| event rows | `94212` |

Large raw files (`event_timing.jsonl`, `metrics.jsonl`, and `serve.log`) are stored as `.gz` in the raw directory.

## Checks

| check | result |
|---|---|
| event JSONL emitted | pass |
| benchmark off/on request failures | `0 / 0` |
| overhead gate | fail |
| installed vLLM patch restored | pass |
| current GPU compute apps after run | empty |

`process_residual_after.txt` contains the runner/monitor command itself due to the grep pattern. A live post-run check showed no vLLM/benchmark process and no GPU compute app.

## Decision

Do not run Phase442 Step 4. The event timing patch perturbs the benchmark too much, so using its rows would change the measured system.

Next viable path is not to relax the gate. The collector needs a lower-overhead timing design, or the line should stay at the Phase438 floor until a non-perturbing measurement primitive is available.
