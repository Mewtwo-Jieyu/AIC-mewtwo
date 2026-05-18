# Phase 47: Research Artifact Manifest

## 结论

研究归档可以保留，但不能默认进入交付包。它的价值是复盘证据链和重开实验，不是给 AIC 提供默认 latency 模型。

| 类别 | 处理 |
|---|---|
| collector | research artifact |
| remote runner | research artifact |
| analyzer | research artifact |
| raw logs / CSV | research evidence |
| residual / profiler / sync 数字 | 禁止作为模型输入 |

## Collector 归档

| 文件 | 归档理由 |
|---|---|
| `collector/vllm/collect_kv_shape_prep.py` | KV shape prep state / timing smoke |
| `collector/vllm/collect_attention_metadata.py` | attention metadata builder smoke |
| `collector/vllm/collect_mla.py` | MLA smoke 复用现有 collector 路径 |
| `collector/vllm/collect_mla_state_smoke.py` | MLA state / timing wrapper |
| `collector/vllm/collect_moe_state_smoke.py` | MoE construction / forward / timing diagnostic |
| `collector/vllm/collect_ep_moe_block.py` | EP MoE falsification evidence |

## Remote runner / source-check 归档

| 文件组 | 归档理由 |
|---|---|
| `collector/vllm/run_attention_*_check.sh` | attention backend / MLA source check |
| `collector/vllm/run_moe_*check.sh`、`collector/vllm/run_moe_preflight.sh` | MoE distributed / config / construction preflight |
| `collector/vllm/run_cuda_forward_envelope_phase*.sh` | Phase 27-28 forward envelope 打点 |
| `collector/vllm/run_cuda_forward_phase*.sh` | Phase 29-38 临时远端打点 |
| `collector/vllm/run_phase*_source_check.sh` | 只读源码边界检查 |
| `collector/vllm/run_nccl_trace_phase39.sh` | NCCL trace 归因 |

## Analyzer 归档

| 文件组 | 归档理由 |
|---|---|
| `scripts/analyze_vllm_runtime_shape_segments.py` | runtime shape / mechanism summary |
| `scripts/analyze_vllm_cuda_forward_envelope.py` | Phase 27 parser |
| `scripts/analyze_vllm_cuda_forward_envelope_v2.py` | Phase 28 parser |
| `scripts/analyze_vllm_cuda_forward_phase29.py` 到 `scripts/analyze_vllm_cuda_forward_phase38.py` | forward envelope 分段 parser |
| `scripts/analyze_vllm_nccl_trace_phase39.py` | NCCL trace parser |
| `scripts/diagnose_iter_gap_residual.py`、`scripts/fit_vllm_execute_residual.py` | residual candidate 证据，不能入模 |
| `scripts/summarize_vllm_budget_sweep.py` | budget sweep 归档工具 |

## Tests 归档

| 文件组 | 归档理由 |
|---|---|
| `tests/unit/collector/` | collector dry-run / state-smoke / timing-smoke 测试 |
| `tests/unit/scripts/` | analyzer schema / safety 测试 |

## Raw evidence 归档

| 文件组 | 归档理由 |
|---|---|
| `docs/ep_moe_falsification/` | EP MoE falsification 原始证据 |
| `docs/iter_gap_investigation/ab_*` | graph/eager A/B 原始证据 |
| `docs/iter_gap_investigation/phase*_*/serve.log` | 单次远端服务日志 |
| `docs/iter_gap_investigation/phase*_*/bench_records.jsonl` | 单次压测请求记录 |
| `docs/iter_gap_investigation/phase*_*/cuda_forward_*rows.csv` | parser 明细行 |
| `docs/iter_gap_investigation/phase*_*/cuda_forward_*summary.csv` | parser 聚合结果 |
| `docs/iter_gap_investigation/phase39_nccl_trace_10k2k_b32_bt8192/nccl_logs/` | NCCL debug 原始日志 |

## 禁止升级为模型输入

| 数据 | 原因 |
|---|---|
| residual bucket | empirical correction，不是物理模型 |
| profiler CUDA ms | coverage 不完整，且 profiler 改变时序 |
| NCCL line count | 只能说明 trace 记录，不是耗时 |
| sync-probe host wait | 显式 sync 改变执行边界 |
| random-weight MoE timing | fallback + 非真实权重 |
| collector single-key timing | diagnostic-only，没有泛化和 clean perf gate |

## 重开时怎么用

| 目标 | 使用方式 |
|---|---|
| 复盘证据链 | 从 Phase 40-47 文档入口读，不从 raw log 开始 |
| 重开 MoE WNA16 | 先满足 Phase 44 non-fallback tuning config 和权重口径 |
| 重开 compiled comm | 先找到 comm-only CUDA event boundary |
| 准备 PR | 默认不整包带 raw logs，只带必要 canonical docs / schema / tests |
