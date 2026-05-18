# Phase 48: Review-Ready Checklist

## 结论

Review 前只检查最小交付包，不检查研究归档全量。collector、runner、parser、raw logs 如果出现在 staging 里，就是错误。

## 文件存在性检查

```bash
for p in \
  src/aiconfigurator/sdk/backends/cb_simulator/forward_descriptor.py \
  src/aiconfigurator/sdk/backends/cb_simulator/__init__.py \
  scripts/validate_cb_simulator.py \
  scripts/diagnose_cb_iter_latency.py \
  tests/unit/test_forward_descriptor_compiled_body.py \
  docs/iter_gap_investigation/phase48_minimal_delivery_staging_plan.md \
  docs/iter_gap_investigation/phase48_review_ready_checklist.md; do
  test -f "$p" || exit 1
done
```

## 必跑验证

| 检查 | 命令 | 通过标准 |
|---|---|---|
| Phase 41 单测 | `conda run -n aic env PYTHONPATH=src python -m pytest tests/unit/test_forward_descriptor_compiled_body.py -q` | `4 passed` |
| 默认 validate | `conda run -n aic python scripts/validate_cb_simulator.py` | throughput / multi-config / TTFT 均 PASS |
| diff 级污染 | `git diff -U0 -- src/aiconfigurator/sdk scripts \| rg -n "^\\+.*(query_.*compiled|wna16.*perf|compiled.*perf|PerfDatabase|IterationLatencyCalculator|run_static)"` | 无输出 |
| 文档空白 | `perl -ne 'print "$ARGV:$.:$_" if /[ \\t]$/' docs/iter_gap_investigation/phase4[0-8]_*.md` | 无输出 |
| diff | `git diff --check` | PASS |

## Staging dry-run 检查

```bash
git add -n \
  src/aiconfigurator/sdk/backends/cb_simulator/forward_descriptor.py \
  src/aiconfigurator/sdk/backends/cb_simulator/__init__.py \
  scripts/validate_cb_simulator.py \
  scripts/diagnose_cb_iter_latency.py \
  tests/unit/test_forward_descriptor_compiled_body.py \
  docs/iter_gap_investigation/phase4[0-8]_*.md
```

dry-run 输出必须只包含：

| 允许出现 | 不允许出现 |
|---|---|
| descriptor schema | collector |
| descriptor export | remote runner |
| validate / diagnose experimental flag | analyzer |
| Phase 41 单测 | raw logs |
| Phase 40-48 docs | CSV / JSONL / tgz |

## Reviewer 必问问题

| 问题 | 期望答案 |
|---|---|
| 是否改默认 latency | 否 |
| 是否写 `PerfDatabase` | 否 |
| 是否改 `run_static` / `IterationLatencyCalculator` | 否 |
| key 是否含 ms / residual / profiler 数字 | 否 |
| MoE WNA16 是否建模 | 否，只保留 diagnostic key |
| compiled comm 是否建模 | 否，只保留 candidate |
| raw evidence 是否进最小交付 | 否 |

## 最终 Go/No-Go

| 条件 | 决策 |
|---|---|
| 所有必跑验证通过，dry-run 范围干净 | 可以 staging |
| 任一验证失败 | 不 staging |
| dry-run 包含研究归档 | 不 staging |
| 出现新增 default latency 接入 | 不 staging |
