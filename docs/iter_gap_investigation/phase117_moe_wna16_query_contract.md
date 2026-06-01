# Phase117 MoE WNA16 Experimental Query Contract

## Decision

Phase117 defines an experimental exact-key table contract. It is a diagnostic prototype and cannot be used by default AIC.

| Item | Rule |
|---|---|
| Table | `phase117_moe_wna16_experimental_table.csv` |
| Query policy | `exact_match_only` |
| Measured token keys | `128`, `248`, `512`, `1024` |
| Missing token key | Fail-fast |
| Interpolation | Not allowed |
| Extrapolation | Not allowed |
| Default AIC | Not allowed |

## Required Query Key

| Field | Required value |
|---|---|
| `backend` | `vllm` |
| `vllm_version` | `0.19.0` |
| `device` | `NVIDIA_H200` |
| `model_family` | `KimiK25` |
| `module` | `DeepseekV2MoE` |
| `experts_impl` | `SharedFusedMoE` |
| `kernel` | `WNA16` |
| `dtype` | `int4_w4a16` |
| `activation_dtype` | `bfloat16` |
| `hidden` | `7168` |
| `intermediate` | `2048` |
| `global_experts` | `384` |
| `local_experts` | `48` |
| `topk` | `8` |
| `topology` | `tp4dp2ep8` |
| `config_sha256` | `15c797eed1dd441e1d6d50fcf7584b2264d248d5ebc1144e1c95e6c4e885d853` |
| `input_kind` | `synthetic_random_hidden_states` |
| `tokens_actual` | One of `128`, `248`, `512`, `1024` |

## Output Fields

| Field | Meaning |
|---|---|
| `cuda_event_ms_mean` | Diagnostic MoE WNA16 CUDA event mean |
| `wall_ms_mean` | Sanity timing only |
| `run_count` | Number of diagnostic shape runs represented |
| `model_use` | Must be `experimental_table_prototype_only` |
| `diagnostic_only` | Must be `true` |
| `valid_for_default` | Must be `false` |
| `perf_database` | Must be `false` |

## Reject Rules

| Query | Result |
|---|---|
| `tokens_actual=128` | Hit |
| `tokens_actual=248` | Hit |
| `tokens_actual=512` | Hit |
| `tokens_actual=1024` | Hit |
| `tokens_actual=256` | Fail-fast |
| `tokens_actual=768` | Fail-fast |
| Any changed model/config/topology field | Fail-fast |
