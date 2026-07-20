# Phase466 low-overhead probe design

结论：Phase466 v3 的本地执行契约已闭合。probe 默认关闭，只使用 vLLM 0.19.0 自带的
`--enable-logging-iteration-details` 和 Prometheus preemption counter，不修改 vLLM 源码，不采
per-request 高频 composition 事件。本轮只形成 rank-timing 诊断，Default AIC 继续 `No-Go`。

## Scope

| 项 | 值 |
|---|---|
| integration base | `822ae421a0b79eb0a69e8f59a2c4d09cba327eaa` |
| execution branch | `feature/kimi-vllm019-cb-sim-post-baseline` |
| hardware/runtime | H200 SXM / vLLM 0.19.0 / Kimi-K2.5 |
| implementation | stock vLLM aggregate iteration details; no source patch |
| local result | 99 Phase466/benchmark tests passed; default simulator validation passed |
| remote result | fresh v3 run pending |
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
| supervisor | fixed node lock, serialized runs, atomic status updates and a 30-second heartbeat |
| identity preflight | no artifact and no vLLM service; validate uploaded bytes, vLLM/GPU/prompt identity, and snapshot or flat-mirror fingerprint |
| execution preflight | only accepts `phase466_execution_manifest_v3`; then creates the fresh artifact root and rechecks clean GPU/process state |
| cleanup | terminate the service process group, then require empty GPU/process residue |
| failure handling | normal benchmark failure may continue to the next preregistered run; cleanup or integrity failure stops all runs |
| gate validation | validates all 6 complete pairs; the old single-pair gate entry no longer exists |
| formal validation | exact scenario, request/token counts, prompt-token digest, rank set, source/tool/model hashes, gate digest and CSV identity |
| simulator export | exact H200/vLLM 0.19.0 database; only valid per-rank paths use `rank_scope=dp_rank` |
| rank timing | TP8 is a single-rank control; DP2 emits rank0/rank1 progress windows, elapsed/token, token/request mix and preemptions |
| exit review | separate `phase466_exit_review_v2`; one scenario selects only with one `PASS` and two `DISPROVED` |

The supervisor never reruns a failed measurement automatically. An interrupted or invalid artifact remains evidence
of that attempt and requires review before another artifact root is started.

## Evidence boundary

The stock probe exposes identity, rank, iteration elapsed time, aggregate prefill/decode request and token counts,
preemption deltas and progress windows. It does not expose prefill chunk histograms, fresh/recompute/resume state,
decode KV sums, cudagraph mode or a simulator serving-state key.

Therefore a passing overhead gate and three valid formal runs only produce `DIAGNOSTIC_COMPLETE`.
They do not produce a model attribution `PASS`, do not execute route selection and do not synthesize DP2-bt65536 simulator rows.

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
| exact vLLM 0.19.0 simulator export | expected fail-closed at `tp4dp2-8k2k-bt65536`; no output directory written |
| `git diff --check` | PASS |
| SSH/GPU | invalid partial attempt stopped; no residue; not gate evidence |

The next action is a fresh serialized six-pair gate on the approved worker. A non-`PASS` gate stops before N512;
`PASS` runs exactly the three real scenarios. This design is not GPU evidence, not a PerfDatabase row,
not a latency model and not evidence for enabling Default AIC.
