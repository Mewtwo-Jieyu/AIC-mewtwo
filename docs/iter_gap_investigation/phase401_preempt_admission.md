# Phase401 preempt admission

Phase401 changes only the CB scheduler waiting-admission rule: a waiting request may use free capacity, but it must not preempt an already-running request. Capacity formulas, PerfDatabase rows, and acceptance gates are unchanged.

| scenario | clean tok/s/gpu | sim tok/s/gpu | error | direction | running max | sim peak decode |
|---|---:|---:|---:|---|---:|---:|
| K2.5-tp8ep8-8k2k | 137.386464 | 169.141051 | 1.231133 | sim_over_predicts_throughput | 67 | 67.000000 |
| K2.5-tp8ep8-32k3k | 50.099161 | 39.087433 | 1.281720 | sim_under_predicts_throughput | 14 | 14.000000 |
| K2.5-tp4ep8dp2-8k2k | 151.681886 | 271.958961 | 1.792956 | sim_over_predicts_throughput |  | 56.000000 |
| K2.5-tp4ep8dp2-32k3k | 50.064056 | 92.590110 | 1.849433 | sim_over_predicts_throughput |  | 10.000000 |

Source check:

- vLLM v0.19.0 scheduler source: `https://raw.githubusercontent.com/vllm-project/vllm/v0.19.0/vllm/v1/core/sched/scheduler.py`.
- Lines 385-516 schedule running requests and preempt only when running KV allocation fails.
- Lines 564-791 schedule waiting requests; if `allocate_slots` returns `None`, waiting admission stops instead of preempting running requests.

Boundary:

- `runtime_modified=true`, but only for scheduler admission semantics.
- `capacity_formula_modified=false`, `perf_database=false`, `valid_for_default=false`, `default_readiness=No-Go`.

Next: Phase402 should decide how to generalize capacity formulas from model/system memory, not from serve logs.
