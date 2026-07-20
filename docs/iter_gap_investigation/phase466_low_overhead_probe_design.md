# Phase466 low-overhead probe design

结论：Phase466 v3.3 的 idle-safe 本地执行契约已闭合。probe 默认关闭，只使用 vLLM 0.19.0 自带的
`--enable-logging-iteration-details` 和 Prometheus preemption counter，不修改 vLLM 源码，不采
per-request 高频 composition 事件。本轮只形成 rank-timing 诊断，Default AIC 继续 `No-Go`。

## Scope

| 项 | 值 |
|---|---|
| integration base | `822ae421a0b79eb0a69e8f59a2c4d09cba327eaa` |
| execution branch | `feature/kimi-vllm019-cb-sim-post-baseline` |
| hardware/runtime | H200 SXM / vLLM 0.19.0 / Kimi-K2.5 |
| implementation | stock vLLM iteration details + rank-local logging handler; no source patch |
| local result | v3.3 verification PASS；第二次 canary 待单独批准 |
| remote result | v3.2 canary `FAILED`；rank-local transport 已验证，idle/measurement 契约失败 |
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

The benchmark return records each rank's measurement cutoff before prompt verification, metrics scraping or cleanup.
Zero-token, zero-request idle rows may have `0.00 ms`; they remain in raw CSV and total elapsed audit fields but do not
enter work latency percentiles, progress coverage or rank-timing windows. Positive-token rows still require finite,
strictly positive elapsed time. A canary cutoff may precede the final rank-log row, but every trailing row must be idle
and every rank must contain measured work.

## Execution contract

| Item | Behavior |
|---|---|
| supervisor | fixed node lock, serialized runs, atomic status updates and a 30-second heartbeat |
| identity preflight | no artifact and no vLLM service; validate uploaded bytes, vLLM/GPU/prompt identity, and snapshot or flat-mirror fingerprint |
| execution preflight | only accepts `phase466_execution_manifest_v4`; then creates the fresh artifact root and rechecks clean GPU/process state |
| log transport | custom logging config filters iteration records from stdout and writes strict per-rank JSONL; no stdout rank fallback |
| canary | a separately approved second DP2 N16/C16 run must pass the v2 idle-safe contract before the full gate is allowed |
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
| SSH/GPU | v3.2 artifact remains `FAILED`: `/mnt/shared-storage-user/zhaojieyu/backup/aic/phase466_v32_rank_local_canary_2221f30e_20260720T110252Z` |

The next action, only after separate approval, is the second rank-local canary on the approved worker. Only a canary `PASS` permits a fresh serialized
six-pair gate. A non-`PASS` gate stops before N512; `PASS` runs exactly the three real scenarios. This design is not GPU evidence, not a PerfDatabase row,
not a latency model and not evidence for enabling Default AIC.
