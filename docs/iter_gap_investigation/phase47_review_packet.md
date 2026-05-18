# Phase 47: Review Packet

## 结论

Reviewer 只需要检查三件事：默认路径是否没变、descriptor 是否不含 latency、研究产物是否没有被误放进最小交付包。

| 检查项 | 期望 |
|---|---|
| 默认 validate | PASS |
| Phase 41 单测 | PASS |
| descriptor key | 无 ms / residual / profiler / line-count |
| research artifact | 不接默认路径 |
| delivery manifest | 最小交付范围明确 |

## Review 顺序

| 顺序 | 看什么 | 文件 |
|---|---|---|
| 1 | 交付边界 | `phase47_minimal_delivery_manifest.md` |
| 2 | 研究归档边界 | `phase47_research_artifact_manifest.md` |
| 3 | staging 范围 | `phase48_minimal_delivery_staging_plan.md` |
| 4 | review-ready 检查 | `phase48_review_ready_checklist.md` |
| 5 | 变更冻结决策 | `phase46_delivery_freeze_decision.md` |
| 6 | 接手结论 | `phase45_modeling_decision_handoff.md` |
| 7 | compiled body 决策 | `phase40_compiled_body_modeling_decision.md` |
| 8 | descriptor 实现 | `forward_descriptor.py`、`validate_cb_simulator.py`、`diagnose_cb_iter_latency.py` |
| 9 | 单测 | `tests/unit/test_forward_descriptor_compiled_body.py` |

## 必须确认

| 问题 | 通过标准 |
|---|---|
| 默认 `cb_sim` 是否变了 | 不带 experimental flag 的 validate 仍 PASS |
| 是否写 `PerfDatabase` | 没有新增写入 |
| 是否改 `run_static` / `IterationLatencyCalculator` | 没有新增 diff 接入 |
| key 是否带 latency | 不包含 ms、residual、profiler、trace count |
| MoE / comm 是否建模 | 没有，只保留 candidate / diagnostic key |
| collector / runner 是否默认调用 | 没有，只是 research artifact |

## 命令包

| 检查 | 命令 |
|---|---|
| 当前变更 | `git status --short` |
| 最小交付文件存在 | `test -f <path>` 逐项检查 |
| Phase 41 单测 | `conda run -n aic env PYTHONPATH=src python -m pytest tests/unit/test_forward_descriptor_compiled_body.py -q` |
| 默认 validate | `conda run -n aic python scripts/validate_cb_simulator.py` |
| 全量源码污染扫描 | `rg -n "query_.*compiled|wna16.*perf|compiled.*perf|PerfDatabase|IterationLatencyCalculator|run_static" src/aiconfigurator/sdk scripts` |
| diff 级新增污染扫描 | `git diff -U0 -- src/aiconfigurator/sdk scripts | rg -n "^\\+.*(query_.*compiled|wna16.*perf|compiled.*perf|PerfDatabase|IterationLatencyCalculator|run_static)"` |
| 文档空白 | `perl -ne 'print "$ARGV:$.:$_" if /[ \\t]$/' docs/iter_gap_investigation/phase4[0-7]_*.md` |
| diff | `git diff --check` |

## 污染扫描解释

| 命中来源 | 处理 |
|---|---|
| 既有 `PerfDatabase` / `run_static` 基础路径 | 允许，这是项目原有模型路径 |
| 新增 experimental flag | 允许，只要显式触发且不改默认 latency |
| 新增 `query_.*compiled` 或 compiled perf 查询 | 不允许 |
| 新增 `wna16.*perf` 默认接口 | 不允许 |
| 新增 `IterationLatencyCalculator` 接入 compiled body key | 不允许 |

## Reviewer 结论模板

| 结论 | 含义 |
|---|---|
| Accept minimal delivery | 只接 descriptor / docs / tests |
| Archive research artifacts | collector、runner、parser、raw logs 不进默认交付 |
| Reject latency model | 没有 clean perf 数据，不允许建模 |
| Reopen later | 必须先满足 Phase 44 reopen conditions |
