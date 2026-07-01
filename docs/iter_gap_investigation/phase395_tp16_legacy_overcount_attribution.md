# Phase395 tp16 Legacy Over-count Attribution (offline, verdict-only)

| Item | Result |
|---|---|
| Question | is the tp16 1.50->1.52 move a masked over-count exposed by merged? |
| Answer | **largely NO** -- merged is negligible on tp16; the ~1.5x error is standing and broad |
| Runtime / PerfDatabase / GPU | not touched (verdict-only) |
| Default AIC | No-Go |

## Harness

`scripts/diagnose_cb_iter_latency.py --cb-trace-out`, tp16 dp1 moe_tp16 moe_ep1, `overlap_factor=0` `per_iteration_overhead_ms=0`, `max_num_batched_tokens=isl`, num_requests/warmup matching `validate_cb_simulator`. Three over-predicting throughput scenarios, each run merged (HEAD) and split (parent `d8827534`), aggregating steady-state per-phase time and per-mixed-iter component means.

## Measured findings

| Scenario | steady tok/s/gpu split->merged | delta | validate Sim/Out split->merged |
|---|---|---|---|
| 10k2k_b32 | 117.99 -> 117.91 | -0.07% | 0.70x -> 0.69x |
| 10k3k_b128 | 201.05 -> 200.55 | -0.25% | 0.67x -> 0.66x |
| 16k2k_b32 | 80.92 -> 80.88 | -0.05% | 0.76x -> 0.75x |

- **Pure prefill and pure decode paths are byte-identical** merged vs split (identical pure_decode time; identical mixed ctx_attn / gen_attn).
- Merged's ONLY per-mixed-iter effect on tp16 is `+decode_bs` tokens on ctx_non_attn (+0.2%..+0.7%). gen_non_attn folds 5-7 ms -> 0 but is INVISIBLE: with `overlap_factor=0` the mixed total is `max(...)`-dominated by ctx_non_attn, and gen_non_attn was never the max.
- Trace-implied steady throughput moves only -0.05%..-0.25%, far smaller than the validate gate move (metric/steady-state sensitivity).
- The ~1.5x tp16 error is **standing**: split already scores 0.67x (=1.49x) on 10k-3k b=128; merged 0.66x (=1.52x). Merged did not create it.
- Steady time is **65%..75% pure_decode** (unchanged by merged), so the tp16 error is a broad legacy-path calibration gap, not localized to mixed non-attention.

## Verdict

- The Phase394 'masked over-count exposed by merged' hypothesis is **largely refuted**. tp16's ~1.5x error predates merged and spans the decode-heavy steady state; merged is negligible on tp16.
- **Do NOT scope-gate merged** (it barely matters on tp16) and do not chase a merged artifact.
- Phase396 (`phase396_unify_perf_backends_onto_module_boundary_schema`): target the standing legacy-path calibration error and unify the two perf backends onto the measured module-boundary schema.

## No-Go discipline held

- verdict-only: runtime / operations / PerfDatabase not modified; no fudge tuning; no scope gating; no GPU/SSH; Default AIC remains No-Go.
