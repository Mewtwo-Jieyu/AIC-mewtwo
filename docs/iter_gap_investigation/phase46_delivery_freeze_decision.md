# Phase 46: Delivery Freeze Decision

## 结论

Phase 46 冻结为：当前交付只包含 vLLM runtime descriptor、compiled body diagnostic key、决策文档和验证测试。所有 collector、runner、parser、CSV、log 都按 research artifact 处理，不能进入默认 latency 模型。

| 项 | 冻结决策 |
|---|---|
| 默认 AIC | 保持 Phase 4 baseline |
| Phase 41 descriptor | 保留，experimental-only |
| Phase 40-46 docs | 保留，作为交付证据 |
| diagnostic scripts | 保留为研究归档 |
| raw CSV / log | 保留为证据，不作为模型输入 |
| perf / latency | 当前不交付 |

## 推荐交付范围

| 范围 | 文件 |
|---|---|
| descriptor schema | `src/aiconfigurator/sdk/backends/cb_simulator/forward_descriptor.py` |
| descriptor export | `src/aiconfigurator/sdk/backends/cb_simulator/__init__.py` |
| validate experimental flag | `scripts/validate_cb_simulator.py` |
| diagnose experimental flag | `scripts/diagnose_cb_iter_latency.py` |
| descriptor test | `tests/unit/test_forward_descriptor_compiled_body.py` |
| modeling decision docs | `docs/iter_gap_investigation/phase40_*.md` 到 `phase46_*.md` |

## 不进入默认模型的范围

| 范围 | 处理 |
|---|---|
| `collector/vllm/collect_*.py` | research artifact |
| `collector/vllm/run_*.sh` | research artifact |
| `scripts/analyze_vllm_*.py` | research artifact |
| `tests/unit/collector/` | research artifact tests |
| `tests/unit/scripts/` | research artifact tests |
| `docs/iter_gap_investigation/phase*_*/serve.log` | raw evidence |
| `docs/iter_gap_investigation/phase*_*/bench_records.jsonl` | raw evidence |
| profiler / NCCL / sync-probe CSV | diagnostic evidence |

## 默认路径冻结规则

| 规则 | 状态 |
|---|---|
| 不写 `PerfDatabase` | 冻结 |
| 不改 `run_static` latency 逻辑 | 冻结 |
| 不改 `IterationLatencyCalculator` | 冻结 |
| 不把 descriptor key 变成 ms key | 冻结 |
| 不把 residual bucket 接默认路径 | 冻结 |
| 不把 profiler / NCCL trace / sync-probe 数字当 perf 数据 | 冻结 |

## 后续如果要合 PR

| 问题 | 决策 |
|---|---|
| 是否要把所有 raw logs 都带上 | 不建议；优先带 canonical docs 和必要 CSV |
| 是否要带 remote runner | 只在 research artifact 包里带，不作为默认产品路径 |
| 是否要带 collector timing smoke | 只作为诊断工具，不作为 perf 数据入口 |
| 是否要带 residual fitting 脚本 | 可归档，但必须标 experimental / failed physical candidate |
| 是否要带 Phase 41 key | 可以；它是本轮最小可交付接口 |

## 重开门槛

| 路线 | 先满足 |
|---|---|
| MoE WNA16 perf | non-fallback tuning config + 权重口径 + 单 key 复现 |
| compiled comm perf | comm-only CUDA event boundary + 来源归因 |
| experimental latency | clean timing + holdout + `valid_for_default=false` |
| default AIC | 多配置验证 + 机制闭环 + 默认 validate 不退化 |

## 最终交付口径

这批 worktree 变更不是“vLLM latency 模型已完成”，而是“vLLM mixed gap 的诊断链路已收口，AIC 现在只能安全接 descriptor，不能接 latency”。后续要重开，必须先满足 Phase 44 的重开条件。
