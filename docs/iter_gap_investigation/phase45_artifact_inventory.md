# Phase 45: Artifact Inventory

## 结论

这个清单只说明 Phase 5-44 产物在哪里，以及哪些代码属于 experimental / diagnostic。它不是发布清单，不要求清理未跟踪文件，也不改变默认 AIC 路径。

## 默认路径边界

| 类别 | 文件 | 状态 |
|---|---|---|
| experimental descriptor | `src/aiconfigurator/sdk/backends/cb_simulator/forward_descriptor.py` | 新增 descriptor / key dataclass 和 builder |
| validate 开关 | `scripts/validate_cb_simulator.py` | 只在显式 experimental flag 下输出 key |
| diagnose 开关 | `scripts/diagnose_cb_iter_latency.py` | 只在显式 experimental flag 下输出 key |
| schema 测试 | `tests/unit/test_forward_descriptor_compiled_body.py` | 验证 key 不含 latency 字段 |
| 默认 latency | `PerfDatabase` / `run_static` / `IterationLatencyCalculator` | 未作为 Phase 45 修改对象 |

## Experimental / diagnostic collector

| 文件 | 用途 | 默认路径 |
|---|---|---|
| `collector/vllm/collect_kv_shape_prep.py` | KV shape prep state / timing smoke | 不接 |
| `collector/vllm/collect_attention_metadata.py` | attention metadata builder smoke | 不接 |
| `collector/vllm/collect_mla.py` | 现有 vLLM MLA collector 路径 | collector-only |
| `collector/vllm/collect_mla_state_smoke.py` | MLA state / timing smoke wrapper | 不接 |
| `collector/vllm/collect_moe_state_smoke.py` | MoE construction / forward / timing diagnostic | 不接 |
| `collector/vllm/collect_ep_moe_block.py` | 早期 EP MoE falsification evidence | 不接 |

## Source-check / remote runner

| 文件组 | 代表文件 | 作用 |
|---|---|---|
| forward envelope | `collector/vllm/run_cuda_forward_envelope_phase27.sh`、`collector/vllm/run_cuda_forward_envelope_phase28.sh` | 临时远端打点 wrapper |
| prepare / logits | `collector/vllm/run_cuda_forward_phase29.sh` 到 `collector/vllm/run_cuda_forward_phase34.sh` | 拆 prepare、logits、sync queue |
| model forward / compiled body | `collector/vllm/run_cuda_forward_phase35.sh`、`collector/vllm/run_cuda_forward_phase36.sh` | 拆 Kimi compiled body |
| comm / NCCL | `collector/vllm/run_cuda_forward_phase38.sh`、`collector/vllm/run_nccl_trace_phase39.sh` | runtime comm marker 和 NCCL trace |
| source check | `collector/vllm/run_phase*_source_check.sh` | 只 inspect，不创建 CUDA tensor |
| MoE preflight | `collector/vllm/run_moe_preflight.sh` | distributed / construction / forward diagnostic |

## Analyzer 和测试

| 文件组 | 用途 |
|---|---|
| `scripts/analyze_vllm_runtime_shape_segments.py` | Phase 7-11 runtime shape / mechanism row 解析 |
| `scripts/analyze_vllm_cuda_forward_envelope.py` | Phase 27 envelope parser |
| `scripts/analyze_vllm_cuda_forward_envelope_v2.py` | Phase 28 envelope v2 parser |
| `scripts/analyze_vllm_cuda_forward_phase29.py` 到 `scripts/analyze_vllm_cuda_forward_phase38.py` | Phase 29-38 分段 parser |
| `scripts/analyze_vllm_nccl_trace_phase39.py` | Phase 39 NCCL trace parser |
| `tests/unit/scripts/test_analyze_vllm_cuda_forward_*.py` | analyzer schema / safety tests |
| `tests/unit/scripts/test_analyze_vllm_nccl_trace_phase39.py` | NCCL trace parser tests |
| `tests/unit/collector/test_collect_*.py` | collector dry-run / state-smoke / timing-smoke tests |

## 关键结果目录

| 目录 / 文件 | 代表内容 |
|---|---|
| `phase9_forward_mechanism_10k2k_b32_b65536_20260507_r2/` | runtime shape / mechanism summary |
| `phase10_runtime_shape_key/` | runtime shape key 对照 |
| `phase11_runtime_shape_subkeys/` | vLLM-only subkey distribution |
| `kv_shape_prep_timing_smoke_phase15f.csv`、`kv_shape_prep_compute_slot_timing_smoke_phase15h.csv` | KV A/B 小量级证据 |
| `phase16_attention_metadata_backend_builder_timing_smoke.csv` | metadata builder 小量级证据 |
| `phase18_mla_forward_mqa_timing_smoke.csv` | MLA kernel 小量级证据 |
| `phase27_cuda_forward_envelope_10k2k_b32_bt8192/` | execute_model envelope |
| `phase28_cuda_forward_envelope_v2_10k2k_b32_bt8192/` | hidden envelope 清理 |
| `phase29_cuda_forward_prepare_logits_10k2k_b32_bt8192/` | prepare / logits split |
| `phase30_prepare_inputs_compute_logits_10k2k_b32_bt8192/` | prepare 内部和 logits No-Go |
| `phase31_slot_mapping_logits_runtime_10k2k_b32_bt8192/` | slot mapping / logits runtime type |
| `phase32_slot_mapping_device_vs_host_10k2k_b32_bt8192_rerun4/` | slot host vs CUDA event |
| `phase33_slot_mapping_sync_probe_10k2k_b32_bt8192/` | pre-slot sync probe |
| `phase34_forward_logits_queue_probe_10k2k_b32_bt8192/` | model forward / logits queue probe |
| `phase35_model_forward_layer_loop_probe_10k2k_b32_bt8192/` | Kimi forward coverage |
| `phase36_compiled_body_profiler_10k2k_b32_bt8192/` | compiled body profiler aggregate |
| `phase38_comm_runtime_attribution_10k2k_b32_bt8192/` | Python comm marker No-Go |
| `phase39_nccl_trace_10k2k_b32_bt8192/` | NCCL trace classification |
| `phase41_compiled_body_runtime_key/` | compiled body runtime key CSV |

## 决策文档

| 文档 | 内容 |
|---|---|
| `phase40_compiled_body_modeling_decision.md` | Phase 27-39 证据链和建模决策 |
| `phase40_vllm_runtime_key_schema.md` | runtime key schema |
| `phase40_stopped_paths.md` | 已停止路线 |
| `phase41_compiled_body_runtime_key_interface.md` | compiled body key 接口 |
| `phase42_compiled_comm_perf_spec.md` | compiled comm perf 数据资格 |
| `phase42_moe_wna16_perf_spec.md` | MoE WNA16 perf 数据资格 |
| `phase42_perf_data_rejection_rules.md` | 无资格数据拒绝规则 |
| `phase43_perf_feasibility_go_nogo.md` | clean perf No-Go |
| `phase44_vllm_modeling_route_closeout.md` | 路线收口 |
| `phase44_reopen_conditions.md` | 重开条件 |
| `phase45_modeling_decision_handoff.md` | 接手页 |

## 处理规则

| 场景 | 处理 |
|---|---|
| 看到未跟踪 Phase 产物 | 不删除、不重命名，先按本清单定位 |
| 需要默认模型变更 | 先检查 Phase 44 重开条件 |
| 需要 perf 数据 | 先证明 clean timing 口径，不用 profiler / NCCL trace / sync-probe |
| 需要引用 TRT-LLM | 只借方法论，不迁移数据和公式 |
