# Phase 47: Minimal Delivery Manifest

## 结论

最小交付包只包含 descriptor 接口、显式 experimental 开关、对应单测和 Phase 40-47 决策文档。它不包含 collector、远端 runner、parser、raw logs，也不包含任何 latency / perf 数据入口。

| 项 | 决策 |
|---|---|
| 默认 AIC | 不动 |
| 交付目标 | 让 AIC 表达 vLLM compiled body runtime key |
| latency | 不交付 |
| `PerfDatabase` | 不写 |
| research artifact | 默认不进最小交付包 |

## 必须交付的代码文件

| 文件 | 原因 |
|---|---|
| `src/aiconfigurator/sdk/backends/cb_simulator/forward_descriptor.py` | 定义 vLLM forward descriptor、runtime shape key、compiled body key |
| `src/aiconfigurator/sdk/backends/cb_simulator/__init__.py` | 导出 descriptor 类型 |
| `scripts/validate_cb_simulator.py` | 提供 `--experimental-forward-descriptor`、`--experimental-runtime-shape-key`、`--experimental-compiled-body-key` |
| `scripts/diagnose_cb_iter_latency.py` | 诊断侧输出 descriptor/key，不改 latency |
| `tests/unit/test_forward_descriptor_compiled_body.py` | 验证 compiled body key 不含 latency / profiler / residual 数据 |

## 必须交付的文档

| 文件组 | 用途 |
|---|---|
| `docs/iter_gap_investigation/phase40_*.md` | compiled body 建模决策和 stopped paths |
| `docs/iter_gap_investigation/phase41_compiled_body_runtime_key_interface.md` | runtime key 接口说明 |
| `docs/iter_gap_investigation/phase42_*.md` | clean perf 数据资格和拒绝规则 |
| `docs/iter_gap_investigation/phase43_*.md` | clean perf Go/No-Go |
| `docs/iter_gap_investigation/phase44_*.md` | 路线收口和重开条件 |
| `docs/iter_gap_investigation/phase45_*.md` | closeout index、artifact inventory、handoff |
| `docs/iter_gap_investigation/phase46_*.md` | changeset inventory 和 delivery freeze |
| `docs/iter_gap_investigation/phase47_*.md` | 最小交付 manifest、研究归档 manifest、review packet |

## 必须满足的边界

| 边界 | 要求 |
|---|---|
| descriptor | 只能包含 shape / topology / compiled path / comm candidate / MoE key |
| experimental flag | 必须显式触发 |
| default validate | 不加 experimental flag 时输出不变 |
| latency 字段 | 禁止 |
| residual / profiler / NCCL debug 数字 | 禁止进入 key |
| fallback MoE timing | 禁止进入 perf table |

## 交付包最小检查命令

| 检查 | 命令 |
|---|---|
| Phase 41 单测 | `conda run -n aic env PYTHONPATH=src python -m pytest tests/unit/test_forward_descriptor_compiled_body.py -q` |
| 默认 validate | `conda run -n aic python scripts/validate_cb_simulator.py` |
| diff | `git diff --check` |

## 不属于最小交付包

| 文件组 | 原因 |
|---|---|
| `collector/vllm/collect_*.py` | diagnostic collector，不是默认功能 |
| `collector/vllm/run_*.sh` | 远端临时 runner / source-check |
| `scripts/analyze_vllm_*.py` | 解析历史实验日志 |
| `tests/unit/collector/` | collector 研究测试 |
| `tests/unit/scripts/` | analyzer 研究测试 |
| `docs/iter_gap_investigation/phase*_*/` | raw evidence 目录，不应整包进入最小交付 |
