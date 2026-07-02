# Phase397z Effective Batch Attribution

Phase397z is offline and report-only. It does not change scheduler code, runtime, DB tables, thresholds, or Default AIC gates.

## Summary

| Item | Result |
|---|---|
| Active surface | 0.19-real 8-card MULTI_CONFIG x6 |
| Worst error | K2.5-tp4ep8dp2-32k3k = 3.28x |
| Mean error | 2.47x |
| Mean effective batch ratio | 4.00x |
| Dominant driver rows | 6/6 effective batch overestimate |
| Scheduler localization | decode reserved before prefill + closed-loop replacement keeps sim decode occupancy high |
| Default AIC | Default AIC remains No-Go |

## Scenario Decomposition

| Scenario | Tier | Sim/real throughput | Effective batch ratio | Iter latency factor | Sim eff batch | Real eff batch | Driver |
|---|---:|---:|---:|---:|---:|---:|---|
| K2.5-tp4ep8dp2-32k3k | B | 3.28x | 6.05x | 0.54x | 63.7 | 10.5 | effective_batch_overestimate |
| K2.5-tp4ep8dp2-8k2k | A | 2.79x | 2.34x | 1.20x | 63.6 | 27.2 | effective_batch_overestimate |
| K2.5-tp4ep8dp2-8k2k-bt65536 | B | 2.48x | 2.07x | 1.20x | 63.9 | 30.8 | effective_batch_overestimate |
| K2.5-tp8ep8-8k2k | A | 2.13x | 2.98x | 0.71x | 125.2 | 42.1 | effective_batch_overestimate |
| K2.5-tp8ep8-32k3k | B | 2.07x | 7.63x | 0.27x | 126.1 | 16.5 | effective_batch_overestimate |
| K2.5-tp8ep8-8k2k-bt65536 | B | 2.07x | 2.93x | 0.71x | 127.6 | 43.6 | effective_batch_overestimate |

## Iteration Structure

| Scenario | Sim iter count / real | Sim wall / real | Sim occupancy | Real occupancy | Steady sim iters | Steady sim ms |
|---|---:|---:|---:|---:|---:|---:|
| K2.5-tp4ep8dp2-32k3k | 0.17x | 0.30x | 1.00 | 0.16 | 11939 | 1092529.4 |
| K2.5-tp4ep8dp2-8k2k | 0.43x | 0.36x | 0.99 | 0.43 | 7939 | 329778.3 |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 0.48x | 0.40x | 1.00 | 0.48 | 7992 | 330674.6 |
| K2.5-tp8ep8-8k2k | 0.34x | 0.47x | 0.98 | 0.33 | 3874 | 218117.4 |
| K2.5-tp8ep8-32k3k | 0.13x | 0.48x | 0.99 | 0.13 | 5873 | 866271.1 |
| K2.5-tp8ep8-8k2k-bt65536 | 0.34x | 0.48x | 1.00 | 0.34 | 3984 | 222701.5 |

## Interpretation

The throughput gap is dominated by effective batch / decode occupancy, not by mixed-step latency. In the Tier A `tp8ep8-8k2k` row, sim holds about 125 decode requests per iteration while the real aggregate throughput implies about 42 effective decode requests. The sim decode iteration is slower than the real pure-decode anchor, so latency is not the source of the over-predicted throughput.

Tier A rows use the Phase397l real decode op-sum directly. Tier B rows reuse the matching topology anchor and are aggregate estimates only.

## Scheduler Localization

- `CBScheduler.schedule()` reserves one token for every decoding request before admitting prefill.
- `CBSimulator.run()` replaces completed requests immediately in a closed loop.
- The simulator therefore keeps decode occupancy near the configured per-replica batch, while the real aggregate surface includes prefill blocking, KV/prefix capacity effects, and lower sustained decode occupancy.

## Boundaries

- No GPU or SSH was used.
- No runtime, scheduler, DB row, threshold, or gate was changed.
- Default AIC remains No-Go.
