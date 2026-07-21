# Phase466 low-overhead probe design

结论：Phase466 v3.3 overhead gate 最终为 `INCONCLUSIVE`，stock iteration probe 路线关闭。正式 N512
场景未运行，三类建模候选保持 unresolved，不选择 Phase467。Default AIC 继续 `No-Go`。

## Scope

| 项 | 值 |
|---|---|
| integration base | `822ae421a0b79eb0a69e8f59a2c4d09cba327eaa` |
| execution branch | `feature/kimi-vllm019-cb-sim-post-baseline` |
| hardware/runtime | H200 SXM / vLLM 0.19.0 / Kimi-K2.5 |
| implementation | stock vLLM iteration details + rank-local logging handler; no source patch |
| local result | v3.3 idle-safe 契约和 closeout analyzer 验证通过 |
| remote result | v3.3 canary `PASS`；overhead gate `INCONCLUSIVE`；`STOPPED_BEFORE_FORMAL` |
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
| canary | second DP2 N16/C16 run passed the v2 idle-safe contract before the full gate |
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

The gate did not pass, so no formal collection, route selection or simulator-row synthesis occurred. Current ON rank logs
belong to an inconclusive measurement method and are inadmissible for model changes, PerfDatabase entries or readiness.

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
| final overhead gate | `INCONCLUSIVE`: geometric mean `1.010241`, 90% CI `[0.946082, 1.078752]` |
| terminal result | `STOPPED_BEFORE_FORMAL`; `formal_scenarios=[]` |

冻结证据和逐 pair 重算见
[Phase466 v3.3 stock probe gate closeout](phase466_stock_probe_gate_closeout/phase466_stock_probe_gate_closeout.md)。
5/6 pair 的第二次运行更快，均值接近 1 不能解释为 probe 开销约 1%。三类候选
`schedule_merged_batch_composition`、`iteration_cost_serving_state_coverage` 和
`dp_rank_synchronization_asymmetry` 均保持 unresolved。stock probe 不再重跑；只有单独评审通过的新低扰动测量设计
才能重新启动建模取证。
