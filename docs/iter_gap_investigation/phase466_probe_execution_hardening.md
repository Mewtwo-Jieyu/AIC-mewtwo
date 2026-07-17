# Phase466 probe execution hardening

结论：Phase466 的本地采集工具已从“可生成命令”补齐到“可 fail-closed 执行”，但没有产生 GPU 证据。
下一步只能在指定 H200 worker 上执行 6 对 overhead gate；gate 不是 `PASS` 时禁止 N512 formal collection。

| 项 | 结果 |
|---|---|
| base | `822ae421a0b79eb0a69e8f59a2c4d09cba327eaa` |
| branch | `experiment/phase466-probe-execution-hardening` |
| runtime/model/PerfDB change | none |
| analyzer schema | `phase466_stock_probe_v2` |
| overhead design | 6 counterbalanced OFF/ON pairs; paired log-ratio TOST |
| supervisor | serialized, heartbeat, exact preflight, strict cleanup, no automatic rerun |
| simulator export | exact H200 / vLLM 0.19.0, global simulator rank scope |
| exit gate | source-specific required fields; at most one human-reviewed route may be selected |
| current gate | `GPU_GATE_PENDING` |
| readiness | `diagnostic_only=true`; `valid_for_default=false`; `perf_database=false`; `No-Go` |

## Changed files

- `scripts/analyze_phase466_low_overhead_probe.py`
- `scripts/run_phase466_low_overhead_probe.py`
- `scripts/export_phase466_cb_sim_iterations.py`
- `scripts/analyze_phase466_exit_review.py`
- four matching unit-test files
- `docs/iter_gap_investigation/phase466_low_overhead_probe_design.md`
- `docs/iter_gap_investigation/phase466_residual_attribution.md`
- this report

## Review fixes

| Finding | Fix |
|---|---|
| old CLI could pass a single OFF/ON pair | removed; root validator requires all six preregistered pairs |
| pair validation could raise before writing a gate result | writes `INVALID` gate and stops formal collection |
| integrity errors used inconsistent prefix/substring checks | one explicit integrity classifier is used everywhere |
| long nohup run had only transition updates | atomic 30-second heartbeat added |
| residue commands were executed repeatedly while writing snapshots | each postflight snapshot is captured once |
| simulator exporter inherited stale vLLM 0.12.0 | exporter explicitly locks exact vLLM 0.19.0 |
| exit review used a union of real/sim fields | required fields are validated per source; blank values are missing |
| plan claimed a 2-second GPU sampler that did not exist | contract now states only the implemented before/after residue snapshots |
| environment contract omitted git/GPU data in artifacts | preflight now records git HEAD and GPU identity |
| remote code or benchmark could be locally modified | preflight requires a clean git worktree and records analyzer/supervisor/benchmark hashes |
| non-PASS gate returned without a common final result | writes `STOPPED_BEFORE_FORMAL`, compresses logs and keeps Default AIC `No-Go` |

## Local outputs

The simulator exporter produced three temporary local CSVs under `/tmp/phase466-sim-export-codex-v2-822ae421`:

| Scenario | Rows |
|---|---:|
| `K2.5-tp4ep8dp2-32k3k` | 12,131 |
| `K2.5-tp8ep8-8k2k-bt65536` | 8,015 |
| `K2.5-tp4ep8dp2-8k2k-bt65536` | 8,015 |

These rows are reproducibility evidence for the local exporter only and are not committed as model data.

## Next gate

1. Verify the worker is clean and the exact source hashes still match.
2. Launch one supervisor artifact root; do not manually start individual pairs.
3. Wait for all 6 pairs and inspect `overhead/overhead_gate.json`.
4. Only a v2 `PASS` with `pair_count=6` permits the three formal runs.
5. Run simulator export and exit review against valid formal rows.
6. Select at most one Phase467 route. Zero or multiple passing routes stay `INCONCLUSIVE`.

No result from this phase may be written to PerfDatabase or used to enable Default AIC.
