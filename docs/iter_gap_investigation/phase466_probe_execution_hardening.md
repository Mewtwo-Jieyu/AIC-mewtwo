# Phase466 probe execution hardening

结论：Phase466 的执行链已补成 fail-closed，但第一次远端试跑在独立 review 发现 contract blocker 后受控终止，
没有形成有效 overhead gate。当前不能直接重跑：stock probe 缺少预注册的 queue/state 字段，且
`tp4dp2-8k2k-bt65536` 的 simulator 仍是代表性单-replica 路径，无法生成可对齐的 DP-rank trace。

| 项 | 结果 |
|---|---|
| base | `822ae421a0b79eb0a69e8f59a2c4d09cba327eaa` |
| branch | `experiment/phase466-probe-execution-hardening` |
| runtime/model/PerfDB change | none |
| analyzer schema | `phase466_stock_probe_v2` |
| overhead design | 6 counterbalanced OFF/ON pairs; paired log-ratio TOST |
| supervisor | node lock, heartbeat, immutable execution manifest, signal cleanup, no automatic rerun |
| simulator export | exact H200 / vLLM 0.19.0; DP rank scope only; unsupported legacy DP path fails closed |
| exit gate | exact three scenarios; rank-local join; incomplete fields cannot be human-overridden to `PASS` |
| invalid remote attempt | `ABORTED_BY_REVIEW`; `gate_status=NOT_EVALUATED`; no formal runs |
| current gate | `BLOCKED_BEFORE_RERUN` |
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
| environment contract omitted source/GPU identity | preflight requires a 40-character source commit and records GPU identity |
| remote directory is not a Git checkout | analyzer/supervisor/benchmark hashes are mandatory execution identity; no optional Git fallback |
| worker image has no Ray package or CLI | cleanup uses the service process group and fails on any remaining GPU/process residue |
| non-PASS gate returned without a common final result | writes `STOPPED_BEFORE_FORMAL`, compresses logs and keeps Default AIC `No-Go` |
| all capture commands accepted exit code 1 | only `pgrep` accepts 0/1; `nvidia-smi` and identity commands require 0 and preserve stderr |
| serialization existed only inside one Python process | fixed node-level `flock` is held through final cleanup and result write |
| source hash was cached from preflight | source files are rehashed before and after every run; actual imported module paths are exact-checked |
| 40-character commit was not bound to uploaded tools | coordinator emits a content-addressed manifest binding commit, contract, supervisor, benchmark and vLLM source hashes |
| TERM/HUP could orphan the service process group | supervisor tracks active service/benchmark and converts signals into cleanup plus terminal `ABORTED` |
| process leader exit was treated as process-group exit | service and benchmark PGIDs are checked directly; lingering groups receive TERM then KILL and block continuation |
| signal could race the final PASS/FAILED write | terminal result uses a single commit point; pre-commit signals rewrite the final artifact to `ABORTED` |
| startup or validation failure could be treated like a skippable benchmark error | only an explicit benchmark command failure may continue; every other exception stops all runs |
| uploaded tooling was checked only at preflight | contract, supervisor and benchmark hashes are revalidated before and after every run |
| coordinator commit could be paired with an arbitrary contract path | production contract override is removed; all three tool paths must be the exact clean checkout paths for `source_commit` |
| formal execution and artifact validation shared one catch | artifact validation failure always stops remaining formal runs |
| PASS result could precede compression/final audit | `phase466_result.json` is the final atomic write |
| zero-token iterations disappeared from elapsed time | zero-token rows retain wall time and unchanged progress |
| exit review accepted a non-empty scenario subset | exact three formal scenarios, passed gate, manifest digest and per-run artifacts are required |
| real DP-rank and global simulator windows were joined by bare id | both sides must use `dp_rank` and join `(rank_id, progress_window_id)` |
| one joined rank could make a DP scenario look evaluable | every expected rank must have at least one joined progress window |
| iteration CSVs were checked only for existence | real result, probe summary and simulator manifest bind CSV SHA256, row identity and rank bounds |
| old exporter over-batched DP2 on one scheduler | 32k3k uses the formal lockstep path; unsupported bt65536 legacy DP trace fails before writing output |

## Invalidated outputs

| Artifact | Decision |
|---|---|
| `/mnt/shared-storage-user/zhaojieyu/backup/aic/phase466_stock_probe_4a9cb6b_20260717` | `ABORTED_BY_REVIEW`; partial overhead data is not gate evidence |
| `/tmp/phase466-sim-export-codex-v2-822ae421` | invalidated; old exporter used global single-scheduler DP semantics |

The remote service and benchmark process groups were stopped manually after freezing the supervisor. GPU and matching process residue were empty.

## Next gate

1. Decide whether the next run only collects rank timing, or must select one Phase467 route.
2. If route selection is required, first close both known evidence gaps: stock real rows need the preregistered queue/state fields, and
   the bt65536 DP simulator needs a validated per-rank execution semantic.
3. Only after that amendment passes review may a fresh artifact root run the six counterbalanced pairs.
4. Gate is not `PASS` means stop before N512. Gate `PASS` still does not waive the exact-three-scenario exit contract.

No result from this phase may be written to PerfDatabase or used to enable Default AIC.
