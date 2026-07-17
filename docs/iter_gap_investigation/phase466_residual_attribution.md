# Phase466 residual attribution

结论：`INCONCLUSIVE`。现有合格证据只能把三个失败点保留为三类可证伪候选，不能选择建模路线。
`aggregate prefill mismatch`、preemption count 和 table miss 都不是独立根因。
Phase462 composition logging v2/v3/v4 的开销为 `13.8486% / 8.7915% / 8.2043%`，未过 `<=2%` 门，全部排除。
候选解释属于 human-reviewed preregistration；analyzer 只机械核对输入数值、结构化状态、字段契约和 MD/CSV
一致性，不把文字判断伪装成自动因果证明。

| Item | Value |
|---|---|
| Base commit | `9ba5ad93ca8805a1eb427593ced8cee01aea6804` |
| Result commit | `4ab587ee` |
| Remote Phase463 artifact | `/mnt/shared-storage-user/zhaojieyu/backup/aic/phase463_six_point_latency_recollect_263a969` |
| Protocol | N512/C128 formal; C64 diagnostic only |
| Outcome | `INCONCLUSIVE` |
| Boundary | `diagnostic_only=true`; `valid_for_default=false`; `perf_database=false`; `Default AIC=No-Go` |
| Execution | no SSH; no GPU; no runtime, PerfDB, validator, or historical-evidence change |

## Failed-cell baseline

| Scenario | Throughput error | TTFT sim/real | TPOT sim/real | E2E sim/real | Prefill req delta | Decode req delta |
|---|---|---|---|---|---|---|
| K2.5-tp4ep8dp2-32k3k | 21.685% | 0.814 | 0.859 | 0.821 | 20.394% | 0.031% |
| K2.5-tp8ep8-8k2k-bt65536 | 25.498% | 0.837 | 0.875 | 0.848 | 28.248% | -4.482% |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 15.670% | 0.857 | 0.998 | 0.888 | 53.864% | 4.978% |

上表的请求数差异只是 Phase463 聚合观测。它缺少 chunk、fresh/recompute/resume、decode KV 和 rank 对齐，
不能单独支持 schedule、cost 或 DP 归因。三个点的 decode 差异方向不同；DP2-bt65536 的 TPOT 又接近 1，
因此不存在一个已被现有数据证明的共同根因。

## Falsifiable candidates

### `schedule_merged_batch_composition`

| Scenario | Status | Existing support | Existing counterevidence |
|---|---|---|---|
| K2.5-tp4ep8dp2-32k3k | open_unseparated | Phase463 shows sim prefill requests/iteration above real alongside a positive throughput residual; see the numeric evidence columns. | Decode requests/iteration are nearly aligned; aggregate request counts contain no chunk or request-state composition. |
| K2.5-tp8ep8-8k2k-bt65536 | open_unseparated | Phase463 shows sim prefill requests/iteration above real and decode requests/iteration below real; see the numeric evidence columns. | The prior engine-loop candidate worsened this cell by +0.084412 error; no admissible exact-composition sample exists. |
| K2.5-tp4ep8dp2-8k2k-bt65536 | open_unseparated | Phase463 shows sim prefill and decode requests/iteration above real; a coarse Phase462 cell showed hidden 8.525597x elapsed spread. | The coarse cell lacks chunk/state/KV fields and Phase462 composition logging failed its measurement gate, so neither number identifies a cause. |

| Required Phase466B fields | Expected discriminator | Disproof rule |
|---|---|---|
| run_id;source;rank_id;iteration_start_offset_ms;iteration_end_offset_ms;workload_cohort_digest;cumulative_scheduled_tokens;progress_window_id;iteration_elapsed_ms;prefill_request_count;decode_request_count;prefill_chunk_token_histogram;fresh_prefill_tokens;recompute_prefill_tokens;resume_prefill_tokens;decode_kv_token_sum;running_count;waiting_count;cudagraph_mode | See per-scenario expected_signal in the CSV; the signal must repeat under the formal protocol. | See per-scenario disproof_condition in the CSV; any disproof keeps this route closed. |
### `iteration_cost_serving_state_coverage`

| Scenario | Status | Existing support | Existing counterevidence |
|---|---|---|---|
| K2.5-tp4ep8dp2-32k3k | open_unseparated | Phase463 shows TTFT, TPOT, and E2E all underpredicted while simulated throughput is high; see the numeric evidence columns. | There is no composition-matched real iteration joined to a simulator cost row, so latency underprediction may still originate in scheduling. |
| K2.5-tp8ep8-8k2k-bt65536 | open_unseparated | Phase463 shows TTFT, TPOT, and E2E all underpredicted while simulated throughput is high; see the numeric evidence columns. | Aggregate latency ratios do not identify the undercharged operator or serving-state dimension. |
| K2.5-tp4ep8dp2-8k2k-bt65536 | open_with_decode_counterevidence | Phase463 shows TTFT and E2E underpredicted while simulated throughput is high; see the numeric evidence columns. | TPOT is nearly aligned, which argues against a uniform decode-cost undercharge as the complete explanation. |

| Required Phase466B fields | Expected discriminator | Disproof rule |
|---|---|---|
| run_id;source;rank_id;iteration_start_offset_ms;iteration_end_offset_ms;workload_cohort_digest;cumulative_scheduled_tokens;progress_window_id;iteration_elapsed_ms;scheduled_prefill_tokens;scheduled_decode_tokens;prefill_chunk_token_histogram;fresh_prefill_tokens;recompute_prefill_tokens;resume_prefill_tokens;decode_kv_token_sum;cudagraph_mode;sim_serving_state_key;sim_predicted_iteration_ms;sim_component_cost_ms | See per-scenario expected_signal in the CSV; the signal must repeat under the formal protocol. | See per-scenario disproof_condition in the CSV; any disproof keeps this route closed. |
### `dp_rank_synchronization_asymmetry`

| Scenario | Status | Existing support | Existing counterevidence |
|---|---|---|---|
| K2.5-tp4ep8dp2-32k3k | open_topology_only | The DP2 32k3k cell fails while the TP8 32k3k control passes under the same formal protocol, leaving a topology-specific factor plausible. | Phase462 proved DP throughput multiplication/division is conserved; Phase463 has no rank-local timing and DP2 8k2k passes. |
| K2.5-tp8ep8-8k2k-bt65536 | topology_negative_control | This dp=1 cell is the same-workload topology control for the DP2-bt65536 measurement. | It fails without DP ranks, proving DP rank asymmetry cannot be a common cause of all three failed cells. |
| K2.5-tp4ep8dp2-8k2k-bt65536 | open_topology_only | Phase462 observed rank0/rank1 counts of 38021/37256 and an 8.525597x elapsed spread inside one coarse mixed cell. | The cell lacks exact composition, DP arithmetic is conserved, and TPOT ratio 0.998 does not support a uniform generation-rank penalty. |

| Required Phase466B fields | Expected discriminator | Disproof rule |
|---|---|---|
| run_id;source;rank_id;iteration_start_offset_ms;iteration_end_offset_ms;workload_cohort_digest;cumulative_scheduled_tokens;progress_window_id;iteration_elapsed_ms;scheduled_prefill_tokens;scheduled_decode_tokens;running_count;waiting_count;completed_request_count | See per-scenario expected_signal in the CSV; the signal must repeat under the formal protocol. | See per-scenario disproof_condition in the CSV; any disproof keeps this route closed. |

Real 与 sim 不共享绝对时钟。正式 join 使用相同的 `workload_cohort_digest`，再按
`cumulative_scheduled_tokens` 切分 `progress_window_id`；start/end offset 只在各自 run 内计算窗口 wall，
禁止用裸 iteration index 或跨进程绝对时间直接配对。

Track B 的 stock-vLLM 第一层 probe 只能直接提供 identity、rank、iteration elapsed、prefill/decode
request/token totals 和 progress window。它不提供本表要求的 chunk/state/queue/KV/component-cost 字段，
因此即使第一层 GPU gate 通过，也不能单独把本报告升级为根因结论或选择 Phase467 模型。
Phase466 v2 exit review 按 real/simulator 来源分别检查字段；仿真侧已有字段不能替代真实侧缺失字段，空值也按缺失处理。

## Machine-checked matrix

| Scenario | Candidate | Status |
|---|---|---|
| K2.5-tp4ep8dp2-32k3k | schedule_merged_batch_composition | open_unseparated |
| K2.5-tp4ep8dp2-32k3k | iteration_cost_serving_state_coverage | open_unseparated |
| K2.5-tp4ep8dp2-32k3k | dp_rank_synchronization_asymmetry | open_topology_only |
| K2.5-tp8ep8-8k2k-bt65536 | schedule_merged_batch_composition | open_unseparated |
| K2.5-tp8ep8-8k2k-bt65536 | iteration_cost_serving_state_coverage | open_unseparated |
| K2.5-tp8ep8-8k2k-bt65536 | dp_rank_synchronization_asymmetry | topology_negative_control |
| K2.5-tp4ep8dp2-8k2k-bt65536 | schedule_merged_batch_composition | open_unseparated |
| K2.5-tp4ep8dp2-8k2k-bt65536 | iteration_cost_serving_state_coverage | open_with_decode_counterevidence |
| K2.5-tp4ep8dp2-8k2k-bt65536 | dp_rank_synchronization_asymmetry | open_topology_only |

完整的每场景缺失观测、Phase466B 字段、预期信号和证伪条件见
`phase466_residual_attribution.csv`。任何单项计数、单个 table miss 或单轮异常都不得覆盖证伪条件。

## Recommended measurement order

| Order | Measurement | Reason | Stop rule |
|---:|---|---|---|
| 1 | DP2-bt65536 off/on (6 pairs) | 同时覆盖构成、cost、rank 三类候选，并承接 Phase462 coarse-cell spread | 90% paired log-ratio CI 未完全落入 `[0.98, 1.02]` 时停止 formal collection |
| 2 | DP2-bt65536 N512/C128 | 在同一 cell 先形成 composition-cost-rank 联合样本 | incomplete rank or iteration fields => reject artifact |
| 3 | TP8-bt65536 N512/C128 | 同 workload 的 dp=1 topology control，只排除跨拓扑共同根因 | protocol/hash mismatch => reject comparison |
| 4 | DP2-32k3k N512/C128 | 用 TP8-32k3k pass cell 作长度/拓扑控制，检查 topology-specific 信号是否复现 | non-repeatable or non-discriminating signal => remain INCONCLUSIVE |

Phase466 exit review 最多只能选择一个通过证伪门的 Phase467 路线。若三类信号仍纠缠，继续测量设计，
不能并行实现补偿模型。

## Evidence limits and forbidden reuse

- Phase462 v2/v3/v4 composition artifacts 不可用于数值归因、PerfDB 行或 serving-state 候选。
- Phase462 engine-loop A/B 是有效的负结果，只能证明该实现回归，不能否定所有 schedule 模型。
- Phase462 的 `8.525597x` 是 coarse-cell anomaly；缺失 exact composition，不能称为 rank 或 cost 根因。
- Phase463 latency 为 baseline-only；没有 readiness gate，也没有 simulator tail-latency distribution。
- 禁止 nearest lookup、常数拟合、启发式修正、runtime/PerfDB/default 路径变更。

## Inputs

| Input | SHA256 |
|---|---|
| docs/iter_gap_investigation/phase465_parallel_exploration_charter.md | 4eb14cb3eef59c0ff70b67ad80c6e49c0278178232fa56e244933065cdea6a22 |
| docs/iter_gap_investigation/phase462_baseline_handoff.md | 4bc678ba37e0eb8146fb5250ffd18b047fbb8ec2ef3e3145e1647419543bc03c |
| docs/iter_gap_investigation/phase462_bt65536_metric_and_composition_audit/phase462_bt65536_metric_and_composition_audit.md | 820a3ddbeb07244d5e249f92ee25d23c85e5b6c1007d9a4c6d713d663d65861e |
| docs/iter_gap_investigation/phase462_exit_review.md | f0dd26192fe16c758614466db1bf6772efcc2364ee5ccfd2b543c4b1f100088c |
| docs/iter_gap_investigation/phase462_step3f/phase462_step3f.md | 92f40b1a60d7897adef7f679a47f99b28add7c131d4237d1c385af824557300d |
| docs/iter_gap_investigation/phase462_dual_replica_metric_consistency/phase462_dual_replica_metric_consistency.md | 64b4ac9d0068c560f0f42a67dec9bfeb1dd8b9820589ddac84b7ee62315465d2 |
| docs/iter_gap_investigation/phase462_engine_loop_arc_closure.md | 21fa5d59e13808adce5349bc5e9fe73780ec6f3ec2993a36553b6861aee4496f |
| docs/iter_gap_investigation/phase462_step2c9_six_point_ab.md | 6ffbb3ba9dc7deaf5ae9ead252b2105546ab7affe89d2cea611ad899c34afcee |
| docs/iter_gap_investigation/phase463_six_point_latency_recollect.md | 826149442cb93a43194cdf476a8677c9d4fb824d8c19a349fa73819cb0ccd5e8 |
| docs/iter_gap_investigation/phase463_six_point_latency_recollect.csv | 41007d3601e97289a809064788e8459a63d278e0d5ae3d5ee391e4f674b6f57b |

## Reproduction and verification

```bash
python3 scripts/analyze_phase466_residual_attribution.py
python3 -m py_compile scripts/analyze_phase466_residual_attribution.py tests/unit/scripts/test_analyze_phase466_residual_attribution.py
pytest -q -c /dev/null -o cache_dir=/tmp/phase466_pytest_cache --confcutdir=tests/unit/scripts tests/unit/scripts/test_analyze_phase466_residual_attribution.py
git diff --check
```

Changed files:

- `scripts/analyze_phase466_residual_attribution.py`
- `tests/unit/scripts/test_analyze_phase466_residual_attribution.py`
- `docs/iter_gap_investigation/phase466_residual_attribution.csv`
- `docs/iter_gap_investigation/phase466_residual_attribution.md`
