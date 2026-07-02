# Phase397w Exact Module Fallback Gate

Phase397w rejects the GPU bucket-widening route and freezes the module exact table as an anchor-only fast path.

## Decision

| Item | Result |
|---|---|
| Required bucket enumeration | `fd6861a9` proved tp4dp2ep8 emits a broad bucket family, including bucket `128` |
| Bucket widening | rejected |
| Exact table role | anchor-only fast path |
| Exact miss behavior | fall back to structural modeling |
| FusedMoE miss fallback | `query_moe(...)`, including Phase397v int4_wo calibrated-SOL |
| EP8 comm miss fallback | existing structured vLLM communication path |
| Exact lookup semantics | unchanged when an anchor key exists |
| Nearest/interpolation/extrapolation | not allowed |

This keeps the strict collector/table contract intact. The runtime no longer treats every scheduler-shaped bucket as something that must exist in `vllm_module_perf.txt`.

## Validate Gate

`scripts/validate_cb_simulator.py` now uses `vllm 0.19.0`, default EP8 overhead `0.0 ms`, and a 0.19-real 8-card MULTI_CONFIG acceptance surface. The legacy tp16 0.17 throughput and TTFT surfaces are still printed, but they are not acceptance gates.

| Section | Gate | Result |
|---|---|---|
| Throughput tp16 legacy 0.17 | skipped | `max=2.62x mean=2.29x` |
| TTFT tp16 legacy 0.17 | skipped | threshold path `max=6.30x mean=5.00x` |
| 0.19-real MULTI_CONFIG x6 | active | `max=3.57x mean=2.58x`, threshold `1.50x`, FAIL |

## Verdict

The exact bucket crash is fixed, but Phase397w does not pass the active gate. Default AIC remains No-Go.

The next blocker is the 0.19 MULTI_CONFIG over-prediction, not missing exact buckets. Do not restart GPU bucket collection, do not widen the exact table, and do not relax exact lookup to nearest or interpolation.
