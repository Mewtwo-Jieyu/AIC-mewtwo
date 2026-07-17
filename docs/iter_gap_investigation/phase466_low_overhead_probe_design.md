# Phase466 low-overhead probe design

结论：本地执行链已经补齐，GPU 开销门尚未执行。probe 默认关闭，只使用 vLLM 0.19.0 自带的
`--enable-logging-iteration-details` 和 Prometheus preemption counter，不修改 vLLM 源码，不采
per-request 高频 composition 事件。当前状态是 `GPU_GATE_PENDING`，Default AIC 继续 `No-Go`。

## Scope

| 项 | 值 |
|---|---|
| integration base | `822ae421a0b79eb0a69e8f59a2c4d09cba327eaa` |
| execution branch | `experiment/phase466-probe-execution-hardening` |
| hardware/runtime | H200 SXM / vLLM 0.19.0 / Kimi-K2.5 |
| implementation | stock vLLM aggregate iteration details; no source patch |
| local result | analyzer, supervisor, simulator exporter and exit gate PASS |
| remote result | GPU gate pending |
| flags | `diagnostic_only=true`; `valid_for_default=false`; `perf_database=false` |

## Measurement contract

| Stage | Scenario | Protocol | Decision |
|---|---|---|---|
| overhead | `K2.5-tp4ep8dp2-8k2k-bt65536` | 6 paired OFF/ON runs; each run first N128/C128 warmup, then N128/C128 measurement | 90% paired log-ratio CI must be fully inside `[0.98, 1.02]` |
| formal 1 | `K2.5-tp4ep8dp2-32k3k` | warmup N128/C128; measurement N512/C128 | only after overhead `PASS` |
| formal 2 | `K2.5-tp8ep8-8k2k-bt65536` | warmup N128/C128; measurement N512/C128 | topology control, not pure DP control |
| formal 3 | `K2.5-tp4ep8dp2-8k2k-bt65536` | warmup N128/C128; measurement N512/C128 | only after overhead `PASS` |

The six pair orders are fixed as `OFF/ON`, `ON/OFF`, repeated three times. `PASS`, `FAIL`, and
`INCONCLUSIVE` are distinct results. Only `PASS` permits formal collection. Missing runs, invalid artifacts or a
confidence interval crossing either equivalence bound produce no formal evidence.

Each measured row carries `run_id`, `source`, `rank_id`, `rank_scope`, `workload_cohort_digest`, rank-local elapsed
offsets, scheduled-token progress bounds and a 65,536-token progress-window id. Warmup iteration ids are captured
per rank and excluded from measured rows. Real/simulator comparison requires the same workload digest and joined
progress windows; absolute clocks and naked iteration ids are not join keys.

## Execution contract

| Item | Behavior |
|---|---|
| supervisor | serialized runs, atomic status updates and a 30-second heartbeat |
| preflight | clean GPU/process state, exact vLLM version, git HEAD, GPU identity and exact stock-source hashes |
| cleanup | terminate service process group, `ray stop --force`, then require empty GPU/process residue |
| failure handling | normal benchmark failure may continue to the next preregistered run; cleanup or integrity failure stops all runs |
| gate validation | validates all 6 complete pairs; the old single-pair gate entry no longer exists |
| formal validation | exact scenario, request/token counts, rank set, source hash, warmup cutoff and passed v2 gate |
| simulator export | exact H200/vLLM 0.19.0 database; simulator rows use `rank_scope=global_simulator` |
| exit review | required fields are checked separately on real and simulator sources; blank fields count as missing |

The supervisor never reruns a failed measurement automatically. An interrupted or invalid artifact remains evidence
of that attempt and requires review before another artifact root is started.

## Evidence boundary

The stock probe exposes identity, rank, iteration elapsed time, aggregate prefill/decode request and token counts,
preemption deltas and progress windows. It does not expose prefill chunk histograms, fresh/recompute/resume state,
decode KV sums, cudagraph mode or a simulator serving-state key.

Therefore a passing overhead gate and three valid formal runs can make the DP-rank candidate evaluable. They cannot
by themselves make schedule composition or serving-state cost coverage evaluable. Missing fields remain
`INCONCLUSIVE_MISSING_FIELDS`; fields from one source cannot satisfy requirements on the other source.

| Prior evidence | Reuse decision |
|---|---|
| Phase462 custom logging v2/v3/v4 | overhead `13.8486% / 8.7915% / 8.2043%`; inadmissible |
| Phase463 stock-flag runs | prove the field is available, but have no OFF control and cannot pass this gate |

## Local verification

| Check | Result |
|---|---|
| Phase466 analyzer/supervisor/exporter/exit tests | PASS |
| `py_compile` with `/tmp` bytecode cache | PASS |
| preregistered plan generation | PASS |
| exact vLLM 0.19.0 simulator export | PASS; 3 scenarios, 28,161 rows |
| `git diff --check` | PASS |
| SSH/GPU | not run |

The first allowed remote action is the serialized 6-pair overhead gate. If the result is not `PASS`, stop before the
N512 formal runs and record the gate result. This design is not GPU evidence, not a PerfDatabase row, not a latency
model and not evidence for enabling Default AIC.
