# Phase 43: MoE WNA16 Tuning Config Check

## 结论

MoE WNA16 perf 线当前 No-Go。`E=48,N=2048,device_name=NVIDIA_H200.json` 没有 non-fallback tuning config 命中；继续 timing 只能得到 fallback diagnostic，不能进入 future experimental latency。

| 项 | 判断 |
|---|---|
| lookup key | `E=48,N=2048,device_name=NVIDIA_H200.json` |
| dtype lookup key | `E=48,N=2048,device_name=NVIDIA_H200,dtype=bfloat16.json` |
| package config | 不存在 |
| user config | 不存在 |
| `tuning_config_loaded` | `false` |
| `fallback` | `true` |
| Phase 44 MoE timing | No-Go |

## 依据

| 来源 | 结论 |
|---|---|
| `phase24_moe_config_source_check.txt` | `default_lookup_exists=False`、`dtype_lookup_exists=False`、`user_lookup_exists=False` |
| `phase24_moe_config_fallback_audit.md` | fallback 不影响 state-smoke 边界，但影响 timing/perf 解释 |
| `phase42_moe_wna16_perf_spec.md` | fallback=true 不允许进表 |

## 字段语义

| 字段 | 含义 |
|---|---|
| `E=48` | rank-local expert 数，来自 `384 / ep_size=8` |
| `N=2048` | Kimi MoE intermediate size |
| `NVIDIA_H200` | vLLM 对 H200 系列的 device name 归一化 |
| `lookup_dtype=None` | 当前 random-weight path 先查无 dtype suffix config |
| `dtype=bfloat16` | dtype suffix 版本也已查过，但不存在 |

## Go/No-Go

| 条件 | 结果 |
|---|---|
| tuning config non-fallback 命中 | 未满足 |
| 能证明真实 Kimi WNA16 kernel config | 未满足 |
| 权重口径能代表真实性能 | 未满足，当前历史 timing 是 random-weight diagnostic |
| 能进入 Phase 44 single-key timing smoke | No-Go |

## 后续条件

| 如果以后要重开 MoE WNA16 perf | 必须先满足 |
|---|---|
| 找到或生成真实 H200 WNA16 tuning config | `tuning_config_loaded=true` |
| 明确权重口径 | `loaded_weight=true` 或证明 random-weight 等价 |
| 只包 MoE aggregate 边界 | 不用 full `_model_forward` 差分 |
| 仍保持 experimental-only | `valid_for_default=false`、`perf_database=false` |

当前决策：MoE WNA16 只保留 diagnostic runtime shape key，不做 Phase 44 timing。
