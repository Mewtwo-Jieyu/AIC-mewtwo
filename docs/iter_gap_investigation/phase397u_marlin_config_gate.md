# Phase397u Marlin Config Gate

Phase397u tested the last two structural no-k levers. Both missed the Phase397l serve anchor band.

| Shot | Latency ms/layer | Ratio vs anchor | Gate |
|---|---:|---:|---|
| local48_direct | 0.693781 | 5.190314x | shot_gate_failed |
| force_block64 | 0.742794 | 5.556992x | shot_gate_failed |

Decision: `structure_path_exhausted_switch_to_option_a`.

Remote vLLM 0.19.0 source exposes `fused_marlin_moe(...)` without an explicit thread/stage config argument; the force-config shot therefore pins the only Python-visible selector, `block_size_m=64`.

Do not append `moe_perf.txt`, do not retire `k=0.312`, and do not persist validate 0.19.0 repoint.

Next allowed phase: `option_a_profiler_anchored_calibration_spec`.

Default AIC remains No-Go.
