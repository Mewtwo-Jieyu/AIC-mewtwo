# Phase 83 MoE Loaded-Weight Source Descriptor

## Conclusion

Phase83 turns the Phase82 loaded-weight MoE evidence into an experimental source descriptor. It does not model latency, does not write perf data, and does not change default `cb_sim`.

## What Changed

| Area | Change |
|---|---|
| Schema | Added `VLLMMoESourceRuntimeKey` |
| Builder | Added `moe_source_runtime_key_from_loaded_weight_boundary_row(...)` |
| Export | Exposed the descriptor and builder from `cb_simulator.__init__` |
| Validate CLI | Added `--experimental-moe-source-key` |
| Diagnose CLI | Added `--experimental-moe-source-key --moe-source-key-out` |
| Tests | Added schema, fail-fast, forbidden-field, topology, and export checks |

## Descriptor Boundary

| Field group | Fields |
|---|---|
| Backend | `runtime_backend`, `vllm_version`, `model_family`, `module_class`, `experts_class` |
| MoE shape | `hidden_size`, `moe_intermediate_size`, `n_routed_experts`, `local_experts`, `global_experts`, `topk`, `n_shared_experts` |
| Kernel | `moe_method`, `kernel_backend`, `group_size`, `num_bits`, `dtype` |
| Topology | `tp_size`, `dp_size`, `ep_size`, `world_size`, `rank`, `device` |
| Config gate | `tuning_config_loaded`, `moe_config_fallback`, `moe_tuning_config_file` |
| Safety | `loaded_weight`, `random_weight`, `timing`, `valid_for_default`, `perf_database`, `diagnostic_only` |

## Phase82 Evidence Encoded

| Item | Value |
|---|---|
| vLLM version | `0.19.0` |
| model family | `kimi_k25` |
| module class | `DeepseekV2MoE` |
| experts class | `SharedFusedMoE` |
| hidden | `7168` |
| MoE intermediate | `2048` |
| routed experts | `384` |
| local experts | `48` |
| topk | `8` |
| kernel | `wna16_marlin` |
| weight status | `loaded_weight=true`, `random_weight=false` |
| config status | `tuning_config_loaded=false`, `moe_config_fallback=true` |

## Fail-Fast Rules

| Bad input | Result |
|---|---|
| `loaded_weight=false` | reject |
| `random_weight=true` | reject |
| `timing=true` | reject |
| `valid_for_default=true` | reject |
| `perf_database=true` | reject |
| missing module class or shape | reject |
| topology mismatch | reject |
| forbidden fields such as latency, residual, profiler, trace, sync, throughput | reject |

## Decision

| Gate | Result |
|---|---|
| Source descriptor | Go |
| MoE timing | No-Go until exact non-fallback H200 `E=48,N=2048` config exists |
| Default latency | No-Go |
| PerfDatabase | No-Go |
