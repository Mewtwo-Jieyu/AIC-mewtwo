# Phase452 dual observation

logging-only raw; no runtime or PerfDB change.

## verdict

- Route: stats overwrite does interleave with adds, but this run only shows a small number of lost local increments; model only if a stable cadence is proven, otherwise keep it as deployment-timing boundary.
- Victim: observed recovery is recompute-from-zero, matching vLLM source and current cb_sim PREEMPTED semantics; do not change victim recovery to chase thrash.
- Raw gates: route_rows=128, stats_overwrite_rows=274, preempt_decision_rows=24.

## self_check

| metric | value | target | status | note |
|---|---:|---:|---|---|
| route_rows | 128 | >=120 |  |  |
| stats_overwrite_rows | 274 | >0 |  |  |
| preempt_decision_rows | 24 | >0 |  |  |

## route

| metric | value | target | status | note |
|---|---:|---:|---|---|
| route_counts | {"0": 69, "1": 59} |  |  | chosen_engine_index histogram |
| per_client_route_counts | {"pid=9407:client=0": {"0": 36, "1": 30}, "pid=9408:client=1": {"0": 33, "1": 29}} |  |  |  |
| adjacent_route_pairs | 126 |  |  |  |
| stats_between_pairs | 3 |  |  |  |
| local_increment_lost_pairs | 3 |  |  | post_counts of one route differs from pre_counts of the next route |
| lost_pair_examples | [{"client": "pid=9407:client=0", "dt_ms": 322.767271, "left_engine": 0, "left_post": [[2, 0], [0, 0]], "pair_index": 0, "right_engine": 1, "right_pre": [[0, 1], [0, 0]], "stats_between": 1}, {"client": "pid=9407:client=0", "dt_ms": 244.843643, "left_engine": 0, "left_post": [[32, 1], [32, 0]], "pair_index": 1, "right_engine": 0, "right_pre": [[0, 1], [8, 1]], "stats_between": 1}, {"client": "pid=9408:client=1", "dt_ms": 205.67689, "left_engine": 0, "left_post": [[32, 1], [32, 0]], "pair_index": 2, "right_engine": 0, "right_pre": [[0, 1], [8, 1]], "stats_between": 1}] |  |  | first examples only |
| verdict | stats_overwrite_interleaves_add |  | pass |  |

## victim

| metric | value | target | status | note |
|---|---:|---:|---|---|
| unique_victims | 24 |  |  |  |
| victim_reschedules | 24 |  |  |  |
| victim_position_hist | {"45": 2, "46": 2, "47": 2, "48": 2, "49": 2, "50": 2, "51": 2, "52": 2, "53": 2, "54": 2, "55": 2, "56": 2} |  |  |  |
| free_blocks_before_range | 0..0 |  |  |  |
| free_blocks_after_range | 500..621 |  |  |  |
| reschedule_new_tokens | [299, 1118, 2110, 3283, 4643, 6198, 6594, 6780, 7116, 7606, 7956, 7957] |  |  | unique values |
| recompute_from_zero | 24 |  |  |  |
| resume_with_cached_tokens | 0 |  |  |  |
| missing_reschedule | 0 |  |  |  |
| verdict | victim_recompute_from_zero |  | pass |  |

## decision

| metric | value | target | status | note |
|---|---:|---:|---|---|
| route_line_next | model_or_boundary |  | needs_decision | overwrite exists; only stable cadence should be modeled |
| victim_line_next | victim_recovery_not_root_cause |  | excluded | real recovery recomputes from zero; cb_sim already uses PREEMPTED recompute semantics |
