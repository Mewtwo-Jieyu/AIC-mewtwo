# Phase398 KV Capacity Fix

## Verdict

| Item | Result |
|---|---|
| Scope | Offline validate-layer probe |
| GPU/SSH | Not used |
| Change | Feed Phase397k per-engine KV capacity into MULTI_CONFIG cb_sim config |
| Before | Phase397z MULTI_CONFIG max 3.28x, mean 2.47x, mostly over-predict |
| After | Phase398 MULTI_CONFIG max 60.62x, mean 16.92x, severe under-predict |
| Gate | No-Go |

Phase398 proves the raw KV capacity hypothesis is not sufficient. The scheduler capacity path is active, but directly using `GPU KV cache size / 16` over-restricts long-context and bt65536 cases.

## Capacity Source

| Scenario | KV tokens | num_gpu_blocks | Serve log lines | Override |
|---|---:|---:|---|---:|
| K2.5-tp8ep8-8k2k | 760160 | 47510 | 202 | 512 |
| K2.5-tp8ep8-32k3k | 675216 | 42201 | 195 | 512 |
| K2.5-tp4ep8dp2-8k2k | 672128 | 42008 | 210, 213 | 512 |
| K2.5-tp4ep8dp2-32k3k | 320800 | 20050 | 204, 212 | 512 |
| K2.5-tp8ep8-8k2k-bt65536 | 343552 | 21472 | 195 | 512 |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 25744 | 1609 | 200, 210 | 256 |

`run_agg` splits global batch by `attention_dp_size` before calling the simulator, then scales throughput back by DP. The capacity is therefore per engine, not DP-aggregated.

## Validation Result

| Scenario | Before error | After validation error | Avg decode before | Avg decode after | Interpretation |
|---|---:|---:|---:|---:|---|
| K2.5-tp8ep8-8k2k | 2.13x | 1.57x | 125.18 | 76.11 | Capacity active, but now under-predicts |
| K2.5-tp8ep8-32k3k | 2.07x | 11.69x | 126.14 | 18.27 | Long context over-restricted |
| K2.5-tp4ep8dp2-8k2k | 2.79x | 2.79x | 63.60 | 63.60 | Capacity not binding |
| K2.5-tp4ep8dp2-32k3k | 3.28x | 21.12x | 63.73 | 8.00 | Long context over-restricted |
| K2.5-tp8ep8-8k2k-bt65536 | 2.07x | 3.71x | 127.62 | 34.81 | bt65536 under-predicts |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 2.48x | 60.62x | 63.93 | 1.62 | bt65536 capacity is unusably tight |

The original failure was real: cb_sim did not feed `num_gpu_blocks`, so KV preemption was disabled. But the simple fix is too literal. The serve log capacity token count is not enough to reproduce vLLM effective batch under these benchmarks.

## Code Notes

| File | Change |
|---|---|
| `scripts/validate_cb_simulator.py` | Adds Phase397k KV capacity table and passes `num_gpu_blocks` for MULTI_CONFIG validation, diagnostics, and budget breakdown only |
| `src/aiconfigurator/sdk/backends/cb_simulator/scheduler.py` | Replaces repeated list membership in block accounting with per-request scheduled deltas; scheduling semantics unchanged |
| `tests/unit/scripts/test_phase398_kv_capacity_validate.py` | Locks capacity table, scheduler capacity behavior, and validate wiring |

## Next

Do not promote this to backend default. Phase399 should first identify the missing vLLM capacity semantics: operative KV capacity versus override, prefix/cache sharing, and preemption/recompute accounting. Default AIC remains No-Go.
