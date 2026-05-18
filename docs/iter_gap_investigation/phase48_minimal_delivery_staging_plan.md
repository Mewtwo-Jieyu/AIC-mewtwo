# Phase 48: Minimal Delivery Staging Plan

## 结论

Phase 48 只定义 staging 范围，不执行 `git add`。最小交付只包含 descriptor、显式 experimental flag、单测和 Phase 40-48 决策文档；研究产物继续留在 worktree，不进入 staging。

| 类别 | Staging 决策 |
|---|---|
| descriptor schema | staging |
| validate / diagnose experimental flag | staging |
| compiled-body descriptor 单测 | staging |
| Phase 40-48 决策文档 | staging |
| collector / runner / parser | 不 staging |
| raw logs / CSV / bench records | 不 staging |
| residual / profiler / NCCL trace 数据 | 不 staging，不入模 |

## 最小 staging 命令

先 dry-run 看集合，不改变 index：

```bash
git add -n \
  src/aiconfigurator/sdk/backends/cb_simulator/forward_descriptor.py \
  src/aiconfigurator/sdk/backends/cb_simulator/__init__.py \
  scripts/validate_cb_simulator.py \
  scripts/diagnose_cb_iter_latency.py \
  tests/unit/test_forward_descriptor_compiled_body.py \
  docs/iter_gap_investigation/phase4[0-8]_*.md
```

确认 dry-run 输出只包含上面范围后，再执行：

```bash
git add \
  src/aiconfigurator/sdk/backends/cb_simulator/forward_descriptor.py \
  src/aiconfigurator/sdk/backends/cb_simulator/__init__.py \
  scripts/validate_cb_simulator.py \
  scripts/diagnose_cb_iter_latency.py \
  tests/unit/test_forward_descriptor_compiled_body.py \
  docs/iter_gap_investigation/phase4[0-8]_*.md
```

## Staging 文件解释

| 文件 | 为什么 staging |
|---|---|
| `forward_descriptor.py` | Phase 41 最小接口，表达 vLLM runtime descriptor |
| `cb_simulator/__init__.py` | 导出 descriptor 类型 |
| `validate_cb_simulator.py` | 默认 validate 不变，显式 flag 才输出 descriptor |
| `diagnose_cb_iter_latency.py` | 诊断侧显式 flag 输出 descriptor |
| `test_forward_descriptor_compiled_body.py` | 验证 key 不含 latency / residual / profiler 数据 |
| `phase40_*.md` 到 `phase48_*.md` | 决策链、No-Go、交付边界、review/staging 依据 |

## 明确排除集合

| 排除项 | 原因 |
|---|---|
| `collector/vllm/collect_*.py` | research artifact，不是默认功能 |
| `collector/vllm/run_*.sh` | 远端临时 runner / source-check |
| `scripts/analyze_vllm_*.py` | 历史实验 parser |
| `scripts/fit_vllm_execute_residual.py` | residual candidate，不能入模 |
| `scripts/diagnose_iter_gap_residual.py` | residual diagnostic，不能入模 |
| `scripts/summarize_vllm_budget_sweep.py` | sweep 归档工具，不是交付核心 |
| `tests/unit/collector/` | collector 研究测试 |
| `tests/unit/scripts/` | analyzer 研究测试 |
| `docs/ep_moe_falsification/` | raw evidence，不进最小交付 |
| `docs/iter_gap_investigation/phase*_*/` | raw result 目录，不进最小交付 |
| `docs/iter_gap_investigation/ab_*` | graph/eager A/B raw evidence |

## Review 前必须检查

| 检查 | 命令 |
|---|---|
| 当前状态 | `git status --short` |
| dry-run staging | `git add -n <最小 staging 命令中的文件集合>` |
| Phase 41 单测 | `conda run -n aic env PYTHONPATH=src python -m pytest tests/unit/test_forward_descriptor_compiled_body.py -q` |
| 默认 validate | `conda run -n aic python scripts/validate_cb_simulator.py` |
| diff 级污染 | `git diff -U0 -- src/aiconfigurator/sdk scripts \| rg -n "^\\+.*(query_.*compiled|wna16.*perf|compiled.*perf|PerfDatabase|IterationLatencyCalculator|run_static)"` |
| 文档空白 | `perl -ne 'print "$ARGV:$.:$_" if /[ \\t]$/' docs/iter_gap_investigation/phase4[0-8]_*.md` |
| diff | `git diff --check` |

## 停止条件

| 发现 | 处理 |
|---|---|
| dry-run 出现 collector / runner / parser | 停止，修 staging 命令 |
| dry-run 出现 raw logs / CSV 目录 | 停止，修 staging 命令 |
| diff 级污染出现 `PerfDatabase` / `run_static` 新增接入 | 停止，不 staging |
| Phase 41 单测失败 | 停止 |
| 默认 validate 失败 | 停止 |
