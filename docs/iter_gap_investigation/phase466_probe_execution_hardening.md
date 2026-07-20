# Phase466 probe execution hardening

结论：hardening 已以 fast-forward 方式吸收到 `feature/kimi-vllm019-cb-sim-post-baseline`。
Phase466 v3 只采真实 rank-local timing，不再要求本轮生成 simulator DP-rank trace 或执行路线选择。

| 项 | 结果 |
|---|---|
| base | `822ae421a0b79eb0a69e8f59a2c4d09cba327eaa` |
| integration branch | `feature/kimi-vllm019-cb-sim-post-baseline` |
| runtime/model/PerfDB change | none |
| analyzer schema | `phase466_rank_timing_v3` |
| overhead design | 6 counterbalanced OFF/ON pairs; paired log-ratio TOST |
| supervisor | node lock, heartbeat, immutable execution manifest, signal cleanup, no automatic rerun |
| simulator export | exact H200 / vLLM 0.19.0; DP rank scope only; unsupported legacy DP path fails closed |
| exit review | `phase466_exit_review_v2`; per-scenario judgement; not executed by this diagnostic run |
| invalid remote attempt | `ABORTED_BY_REVIEW`; `gate_status=NOT_EVALUATED`; no formal runs |
| current gate | local contract verified; fresh v3 remote gate pending |
| readiness | `diagnostic_only=true`; `valid_for_default=false`; `perf_database=false`; `No-Go` |

## Changed files

- `scripts/analyze_phase466_low_overhead_probe.py`
- `scripts/run_phase466_low_overhead_probe.py`
- `scripts/analyze_phase466_rank_timing.py`
- `scripts/export_phase466_cb_sim_iterations.py`
- `scripts/analyze_phase466_exit_review.py`
- five matching unit-test files
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
| model/tokenizer identity was not immutable | manifest binds the snapshot revision and SHA256 of `config.json`, `tokenizer_config.json`, `tiktoken.model`, and `tokenization_kimi.py`; every run rechecks them before and after |
| workload identity stopped at prompt length | benchmark records a digest of the exact token-id cohort; missing or changed digests stop execution |
| coordinator commit could be paired with an arbitrary contract path | production contract override is removed; all four tool paths must be the exact clean checkout paths for `source_commit` |
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

1. Use a fresh `phase466_v3_rank_timing_<postbaseline_sha>_<timestamp>` artifact root.
2. Run the six counterbalanced pairs; any non-`PASS` result stops before N512.
3. After `PASS`, collect exactly three real scenarios and emit `DIAGNOSTIC_COMPLETE`.
4. Phase467 remains reserved for a later model change supported by this evidence.

No result from this phase may be written to PerfDatabase or used to enable Default AIC.
