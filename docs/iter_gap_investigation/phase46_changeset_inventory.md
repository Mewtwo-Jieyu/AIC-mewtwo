# Phase 46: Changeset Inventory

## 结论

当前 worktree 变更应按“交付核心”和“研究归档”分开。默认 AIC latency 路径没有资格接入任何 Phase 5-45 的诊断数字；Phase 46 只冻结交付边界，不删除历史产物。

| 类别 | 处理 |
|---|---|
| Phase 40-46 决策文档 | 交付核心 |
| Phase 41 descriptor / experimental flag / 单测 | 交付核心，但只限 experimental |
| collector / runner / analyzer / raw log | 研究归档 |
| residual bucket / profiler / NCCL trace 数字 | 禁止成为模型输入 |
| 默认 latency 路径 | 不改 |

## Tracked 变更

| 文件 | 归属 | 交付判断 |
|---|---|---|
| `src/aiconfigurator/sdk/backends/cb_simulator/__init__.py` | descriptor export | 可保留，导出 descriptor 类型 |
| `scripts/validate_cb_simulator.py` | experimental CLI | 可保留，显式 flag 才输出 descriptor |
| `scripts/diagnose_cb_iter_latency.py` | experimental CLI | 可保留，显式 flag 才输出 descriptor |
| `collector/vllm/collect_mla.py` | diagnostic collector | 研究归档，不接默认路径 |
| `collector/vllm/utils.py` | diagnostic collector support | 研究归档，不接默认路径 |

## Untracked 交付核心

| 文件 / 目录 | 归属 | 交付判断 |
|---|---|---|
| `src/aiconfigurator/sdk/backends/cb_simulator/forward_descriptor.py` | Phase 8-41 descriptor schema | 必须保留，descriptor-only |
| `tests/unit/test_forward_descriptor_compiled_body.py` | Phase 41 schema test | 必须保留 |
| `docs/iter_gap_investigation/phase40_*.md` | compiled body 建模决策 | 必须保留 |
| `docs/iter_gap_investigation/phase41_compiled_body_runtime_key_interface.md` | compiled body key 接口 | 必须保留 |
| `docs/iter_gap_investigation/phase42_*.md` | clean perf 数据资格 | 必须保留 |
| `docs/iter_gap_investigation/phase43_*.md` | clean perf Go/No-Go | 必须保留 |
| `docs/iter_gap_investigation/phase44_*.md` | 路线收口和重开条件 | 必须保留 |
| `docs/iter_gap_investigation/phase45_*.md` | 交付包入口和 handoff | 必须保留 |
| `docs/iter_gap_investigation/phase46_*.md` | 变更集边界和冻结结论 | 必须保留 |

## Untracked 研究归档

| 文件组 | 代表文件 | 处理 |
|---|---|---|
| runtime shape analyzer | `scripts/analyze_vllm_runtime_shape_segments.py` | 研究归档 |
| cuda forward analyzer | `scripts/analyze_vllm_cuda_forward_phase29.py` 到 `scripts/analyze_vllm_cuda_forward_phase38.py` | 研究归档 |
| NCCL trace analyzer | `scripts/analyze_vllm_nccl_trace_phase39.py` | 研究归档 |
| residual candidate | `scripts/fit_vllm_execute_residual.py`、`scripts/diagnose_iter_gap_residual.py` | 研究归档，不能入模 |
| budget sweep summary | `scripts/summarize_vllm_budget_sweep.py` | 研究归档 |
| KV / metadata / MLA / MoE collector | `collector/vllm/collect_kv_shape_prep.py`、`collector/vllm/collect_attention_metadata.py`、`collector/vllm/collect_mla_state_smoke.py`、`collector/vllm/collect_moe_state_smoke.py` | 研究归档 |
| remote source-check / runner | `collector/vllm/run_phase*_source_check.sh`、`collector/vllm/run_cuda_forward_phase*.sh` | 研究归档 |
| collector tests | `tests/unit/collector/` | 跟 collector 一起归档 |
| analyzer tests | `tests/unit/scripts/` | 跟 analyzer 一起归档 |
| raw evidence | `docs/iter_gap_investigation/phase*_*/`、`docs/ep_moe_falsification/` | 研究归档，只引用 canonical summary |

## Canonical 结果入口

| 证据链 | Canonical 入口 |
|---|---|
| bucket / residual 收口 | `phase5_closeout.md`、`phase6_closeout.md` |
| runtime descriptor | `phase8_forward_descriptor_modeling_plan.md`、`vllm_forward_descriptor_schema.md` |
| runtime shape key | `phase10_runtime_shape_key_design.md`、`phase11_runtime_shape_subkeys_design.md` |
| 小模块排除 | `phase19_excluded_modules_closeout.md` |
| forward envelope | `phase27_cuda_forward_envelope_instrumentation.md` 到 `phase35_model_forward_layer_loop_probe.md` |
| compiled body | `phase36_compiled_body_profiler.md`、`phase40_compiled_body_modeling_decision.md` |
| comm / MoE candidate | `phase37_comm_moe_boundary.md`、`phase38_comm_runtime_attribution.md`、`phase39_compiled_comm_closeout.md` |
| clean perf gate | `phase42_perf_data_rejection_rules.md`、`phase43_perf_feasibility_go_nogo.md` |
| route closeout | `phase44_vllm_modeling_route_closeout.md`、`phase45_modeling_decision_handoff.md` |

## 禁止回流项

| 文件 / 数据 | 禁止原因 |
|---|---|
| `mixed_ctx_bucket` residual 数据 | empirical correction，不是物理模型 |
| profiler CUDA aggregate | coverage 不完整，且 profiler 改变时序 |
| NCCL trace log count | 只能归因，不是耗时 |
| sync-probe host wait | 显式 sync 改变执行边界 |
| random-weight MoE timing | fallback + 非真实权重 |
| collector timing CSV | diagnostic-only，不能直接写 perf table |
