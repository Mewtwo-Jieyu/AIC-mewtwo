# Phase399 KV Capacity Semantics

## Verdict

| Item | Result |
|---|---|
| Scope | Offline forensics only |
| GPU/SSH | Not used |
| vLLM source | v0.19.0 `kv_cache_utils.py` official tag |
| Main finding | `GPU KV cache size` is real logged token capacity after override, but cb_sim cannot use it as full per-request KV ownership under prefix sharing |
| Phase398 failure | Naive capacity binding caused severe under-prediction, worst 60.62x on K2.5-tp4ep8dp2-8k2k-bt65536 |
| Gate recommendation | gate shrink to four clean scenarios; demote bt65536 to reference until recollected |
| Default | Default AIC remains No-Go |

## Source Semantics

| Fact | Lines | Meaning |
|---|---:|---|
| override | 823-836 | `num_gpu_blocks_override` replaces computed `num_blocks` |
| GPU KV size | 1303-1319 | logged token capacity is derived from `kv_cache_config.num_blocks`, group count, and block size |

## Scenario Classification

| Scenario | Little in-system | Decode batch global | Full-seq capacity global | Phase398 error | Class |
|---|---:|---:|---:|---:|---|
| K2.5-tp4ep8dp2-32k3k | 127.97 | 67.80 | 18.33 | 21.12x | preempt_thrash_artifact |
| K2.5-tp4ep8dp2-8k2k | 127.96 | 136.57 | 134.43 | 2.79x | kv_capacity_semantics_incomplete |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 91.51 | 85.65 | 5.15 | 60.62x | dirty_bt65536_capacity_conflict |
| K2.5-tp8ep8-32k3k | 127.86 | 62.59 | 19.29 | 11.69x | preempt_thrash_artifact |
| K2.5-tp8ep8-8k2k | 127.97 | 136.02 | 76.02 | 1.57x | kv_capacity_semantics_incomplete |
| K2.5-tp8ep8-8k2k-bt65536 | 127.96 | 136.18 | 34.36 | 3.71x | dirty_bt65536_capacity_conflict |

## Clean Gate

Clean candidates: K2.5-tp4ep8dp2-32k3k, K2.5-tp4ep8dp2-8k2k, K2.5-tp8ep8-32k3k, K2.5-tp8ep8-8k2k.
Reference-only: K2.5-tp4ep8dp2-8k2k-bt65536, K2.5-tp8ep8-8k2k-bt65536.

The two bt65536 rows are not acceptable gate rows here: one has a different max model length / memory utilization / max seqs, and both are retry-path budget variants. They can stay as diagnostic references only.

## Next

Phase400 should not promote the Phase398 wiring directly. It should first model prefix-shared KV ownership and vLLM preemption/recompute behavior, then re-run only the four clean scenarios.
