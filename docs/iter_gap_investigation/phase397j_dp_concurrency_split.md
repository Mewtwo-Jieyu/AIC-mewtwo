# Phase397j DP Concurrency / MoE Token-count Modeling Patch (cb_sim agg)

> RUNTIME MODIFIED. The cb_sim agg path (`_run_agg_cb_sim`) ran the deployment's full concurrency `b` on ONE dp replica and charged the MoE op at `b*attention_dp` tokens, so a `dp>1` config modeled 2x the real global concurrency and 2x the MoE token-count -- inverting the tp8-vs-tp4dp2 ranking. This phase interprets `b` as GLOBAL concurrency and splits it per replica (`per_replica = ceil(b/dp)`). Version-independent structural fix; the absolute under-prediction residual is a separate DB-version issue.

| Item | Result |
|---|---|
| Question | does the dp concurrency split fix the tp8-vs-tp4dp2 ranking? |
| Answer | YES. Ranking now matches real (both ~equal); dp=1 bit-for-bit unchanged; multi-config max error improves 2.17x -> 1.89x. |
| Runtime / table / Default AIC | runtime modified (dp split only); PerfDatabase and the 0.17 real baseline NOT changed; no GPU/SSH; Default remains No-Go. |

## 1. The DP token-count over-count (before -> after)

A read-only probe on the agg path (concurrency passed to `CBSimulator.run`; MoE tokens after the op's `x*attention_dp` gather; reported `global_bs`):

| config | tp | dp | per-replica bs | MoE tokens | global_bs | real gathered |
|---|---|---|---|---|---|---|
| tp8ep8-8k2k | 8 | 1 | 128 -> 128 | 128 -> 128 | 128 -> 128 | 128 |
| tp4ep8dp2-8k2k | 4 | 2 | 128 -> 64 | 256 -> 128 | 256 -> 128 | 128 |

tp8 (dp=1) is a no-op (`ceil(b/1)=b`). tp4dp2 now decodes 64/replica and gathers 128 MoE tokens -- matching the measured 0.19.0 decode gather (DP0 72 + DP1 56).

## 2. Multi-config throughput (baseline stays 0.17)

tok/s/GPU output-only; overlap_factor=0.0, ep8 overhead=90ms, 0.12.0 DB.

| config | tp | dp | real | sim before | err | sim after | err |
|---|---|---|---|---|---|---|---|
| K2.5-tp8ep8-8k2k | 8 | 1 | 133.5 | 82.8 | 1.61x | 82.8 | 1.61x |
| K2.5-tp8ep8-32k3k | 8 | 1 | 52.5 | 64.8 | 1.24x | 64.8 | 1.24x |
| K2.5-tp4ep8dp2-8k2k | 4 | 2 | 137.7 | 149.7 | 1.09x | 82.7 | 1.66x |
| K2.5-tp4ep8dp2-32k3k | 4 | 2 | 53.3 | 115.8 | 2.17x | 57.1 | 1.07x |
| K2.5-tp8ep8-8k2k-bt65536 | 8 | 1 | 138.5 | 82.8 | 1.67x | 82.8 | 1.67x |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 4 | 2 | 156.0 | 149.1 | 1.05x | 82.6 | 1.89x |

Max error 2.17x -> 1.89x (worst config improves); mean 1.47x -> 1.52x. dp=1 rows are byte-identical before/after.

## 3. The ranking fix (version-independent win)

tp8 (dp1) vs tp4dp2 per-GPU throughput ratio, sim vs real:

| shape | real ratio | sim before | sim after |
|---|---|---|---|
| 8k2k | 0.970 | 0.553 | 1.001 |
| 32k3k | 0.985 | 0.560 | 1.136 |
| bt65536 | 0.888 | 0.555 | 1.003 |

At 8k2k the sim went from tp4dp2 looking ~1.8x faster than tp8 (ratio 0.553) to tp8 ~= tp4dp2 (ratio 1.001), matching real (0.970). Every shape moves toward real.

## 4. Residual is a DB-version issue, not a dp bug

- tp8 (dp=1) is 0.62x vs 0.17 and UNCHANGED by this fix. After the fix tp4dp2 shares the same ~0.6x. The absolute under-prediction is therefore not a dp bug.
- The pre-fix tp4dp2 'accuracy' (1.09x / 0.96x) was two errors cancelling: the dp over-count (over-prediction) masked the tp8-shared version under-prediction.
- Closing the absolute gap needs a vLLM 0.19.0 full DB re-collection (the 0.12.0 DB models older/slower kernels) -- a separate data phase, not a sim edit.

## 5. Cross-backend follow-up (not changed here)

`batch_sync` agg (`scale_factor = pp*dp`, `global_bs = b*dp`) and `trtllm_backend` carry the identical per-replica assumption. phase397j scopes the fix to the cb_sim agg path (the validated path); their alignment is a recorded follow-up.

## Verdict

- The dp concurrency / MoE token-count over-count is **FIXED**: tp4dp2 models `b/dp` per replica and MoE at the real gathered token-count; the tp8-vs-tp4dp2 per-GPU ranking matches real; dp=1 is bit-for-bit unchanged.
- Multi-config max error improves 2.17x -> 1.89x.
- The remaining absolute under-prediction (~0.6x, shared by tp8 dp=1) is a 0.12.0-DB-vs-0.19.0-engine **version** gap, out of scope.
- `MULTI_CONFIG_MAX_ACCEPTANCE` left unchanged; Default AIC remains **No-Go**.
- Next: **phase397k_decode_iter_db_version_realign** -- re-collect a vLLM 0.19.0 full perf DB so the absolute decode-iter magnitude matches the 0.19.0 engine.

## Discipline

- Runtime change is scoped to `_run_agg_cb_sim`'s dp concurrency split; no MoE op or PerfDatabase change; no overhead/fudge tuning; no scope gating.
- The 0.17 multi-config real baseline is unchanged; the acceptance gate is not loosened; Default AIC stays No-Go.
- dp=1 configs verified bit-for-bit identical before/after.
- Raw evidence: `phase397j_multi_config_before_after_raw.csv`, `phase397j_dp_tokencount_raw.csv`.
