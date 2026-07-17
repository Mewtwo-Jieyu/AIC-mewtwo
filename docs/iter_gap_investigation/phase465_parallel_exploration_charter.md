# Phase465 post-baseline parallel exploration charter

结论：Phase464 baseline 固定在 `9e31b2d0746a6f983745bbd440debbe11651a602`。后续工作只进入
`feature/kimi-vllm019-cb-sim-post-baseline` 集成分支，不再修改 baseline。第一轮采用“一修两查”：
Track A 修 correctness 作用域，Track B 设计低扰动探针，Track C 用现有证据做残差分型。任何 track
都不能直接打开 Default AIC。

## Frozen boundary

| 项 | 固定值 |
|---|---|
| baseline branch | `feature/kimi-vllm019-cb-sim-baseline` |
| baseline commit | `9e31b2d0746a6f983745bbd440debbe11651a602` |
| integration branch | `feature/kimi-vllm019-cb-sim-post-baseline` |
| model scope | `moonshotai/Kimi-K2.5` |
| hardware/backend | H200 SXM / vLLM 0.19.0 |
| formal protocol | N512/C128; C64 diagnostic only |
| current throughput gate | `3/6` within 15% |
| readiness | `diagnostic_only=true`; `valid_for_default=false`; `perf_database=false`; `Default AIC=No-Go` |

Agent branches start from this charter commit and use separate worktrees. Workers may commit only to their assigned
branch. They must not merge, cherry-pick, rebase, force-push, modify the frozen baseline, or delete another worktree.
The coordinator owns all integration operations.

## Parallel tracks

| Track | Branch | Responsibility | Production behavior |
|---|---|---|---|
| A | `experiment/phase465-applicability-hardening` | Make Kimi serving-state and calibrated int4 paths fail closed outside their measured scope; make the EP8 source explicit | May change runtime only to close scope/provenance holes; current Kimi six-point outputs must not change |
| B | `experiment/phase466-low-overhead-probe` | Design a default-off, iteration/rank aggregate probe with parser, tests, collection command and overhead preregistration | No model or PerfDB change; no GPU before coordinator review |
| C | `experiment/phase466-residual-attribution` | Re-analyze admissible Phase462/463 evidence for the three failed cells and preregister discriminating hypotheses | Report-only; no runtime, schema, PerfDB or GPU change |

Track B and C are Phase466 preparation. They may run in parallel with Phase465 correctness hardening, but formal GPU
evidence is collected only after Track A is reviewed and integrated.

## Track gates

### Track A

- A non-Kimi runtime must not read Kimi serving-state rows.
- The Phase397v calibrated int4 anchor must not apply to a different model or unsupported measured shape.
- EP8 measured and structural sources must be distinguishable without exception-driven provenance.
- Missing required scope metadata fails explicitly; no nearest lookup, silent fallback, clamp or heuristic repair.
- Add failing tests before implementation and retain exact current Kimi outputs across the six formal scenarios.

### Track B

- Record only low-frequency aggregate iteration/rank fields needed to distinguish schedule composition, iteration cost,
  and DP rank asymmetry.
- The probe is disabled by default and must not emit per-request high-frequency composition logs.
- Local parser and lifecycle tests must pass before any SSH or GPU execution.
- The first GPU gate is an off/on comparison on DP2-bt65536. Absolute throughput delta must be `<=2%`.
- If the overhead gate fails, stop before N512 collection. If it passes, collect only DP2-32k3k, TP8-bt65536 and
  DP2-bt65536 under N512/C128. GPU runs are serialized.

### Track C

- Use only evidence that already passed its measurement gate. Phase462 composition logging v2/v3/v4 is inadmissible.
- Do not call aggregate prefill mismatch, preemption count or PerfDB miss rate a root cause by itself.
- For each candidate, state the expected observation, disproof condition and missing field.
- The result may be inconclusive; it must not select a model without discriminating evidence.

## Result contract

Each track ends with an independent Markdown report and, when there are tabular results, a lightweight CSV. The report
must record the base and result commits, changed files, input artifact paths and hashes, exact commands, verification,
gate outcome, remaining uncertainty and forbidden reuse. Large traces stay in remote artifacts and are referenced by
path and checksum.

| Outcome | Integration action |
|---|---|
| PASS | Coordinator reviews and uses `--no-ff` merge for the complete branch |
| FAIL | Do not merge experimental code; cherry-pick only the independent report/data commit |
| INCONCLUSIVE | Preserve the independent report and missing evidence; do not choose a modeling route |

Reviewer agents are read-only and do not need branches. Their findings are claims until the coordinator inspects the
diff and reruns the relevant verification.

## Integration and exit

1. Review and integrate Track A first.
2. Integrate the admissible Track C report.
3. Review Track B locally, then run its GPU gate and conditionally collect the three failed cells.
4. Produce a Phase466 exit review that chooses at most one evidence-supported Phase467 implementation route:
   scheduler/merged-batch granularity, iteration-cost/serving-state dimensions, or DP synchronization/rank asymmetry.
5. If evidence is insufficient, continue measurement design instead of implementing multiple compensating models.

Every integration step reruns targeted tests, `scripts/validate_cb_simulator.py`, `git diff --check`, and worktree
boundary checks. Until the six-point throughput gate reaches `6/6 <=15%`, passed cells do not regress, an independent
holdout exists, and latency gates are defined, Default AIC remains No-Go.
