# Phase437 Serving-State Grid Collection

## Verdict

`grid_collection_incomplete_do_not_ingest`.

Phase437 采到了新的 serving-state profiler raw, 但没有形成可入库网格: 32k mixed 窗口没有有效 trace, 低 ISL decode sweep 触发 CUDA launch failure, 8k2k 工作点仍大量 miss。候选表只做取证, 已从 `vllm_serving_state_perf.txt` 撤回。Default AIC 继续 No-Go。

Raw profiler artifacts are kept locally under `docs/iter_gap_investigation/phase437_serving_state_collect/`. They are not committed because the trace `.gz` files have no LFS filter and total about 1.0G; the committed manifest is `phase437_serving_state_raw_manifest.csv`.

## Window Outcome

| window | result | useful coverage | failure note |
|---|---|---|---|
| mixed_isl2k_c128 | pass | ctx steps 48, decode batch max 65 | - |
| mixed_isl4k_c128 | pass | ctx steps 72, decode batch max 32 | - |
| mixed_isl8k_c128_ramp | pass on retry | ctx steps 96, decode batch max 16 | first attempt trace EOF |
| mixed_isl16k_c64 | pass after protocol change | ctx steps 184, decode batch max 8 | earlier attempts trace EOF |
| mixed_isl32k_c64 | fail | none | no profile files after retries |
| mixed_isl64k_c128 | pass | ctx steps 96, decode batch max 2 | - |
| decode_isl1k_c16 | fail | none | CUDA launch failure / EngineCore died |
| cross_bt8000_mixed_8k | pass after 90s flush | ctx steps 568, decode batch max 1 | early attempts missing/EOF |
| cross_kv_decode_8k_c128 | skipped | none | decode protocol already failed |

## Extracted Candidate Coverage

These rows are candidate measurements only. They were not ingested because the coverage is not rectangular enough for the current inner-only query policy.

| phase | category | rows | bucket min | bucket max | batch min | batch max | samples |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | collective_other | 4 | 2 | 64 | 2 | 64 | 32 |
| decode | ep_a2a | 4 | 2 | 64 | 2 | 64 | 32 |
| decode | moe_gemm_or_aux | 4 | 2 | 64 | 2 | 64 | 32 |
| decode | other_cuda | 4 | 2 | 64 | 2 | 64 | 32 |
| mixed_prefill | ep_a2a | 7 | 4016 | 64001 | 1 | 34 | 40 |
| mixed_prefill | moe_gemm_or_aux | 7 | 4016 | 64001 | 1 | 34 | 40 |
| mixed_prefill | other_cuda | 7 | 4016 | 64001 | 1 | 34 | 40 |

## Hit Audit With Candidate Table

The audit was run against the Phase437 candidate table using the Phase436 audit harness.

| scenario | hit | interpolation_gap | table_missing | bucket_below_range | decode_batch_below_range | decode_batch_above_range |
|---|---:|---:|---:|---:|---:|---:|
| K2.5-tp4ep8dp2-8k2k | 22 | 236 | 70 | 13 | 96 | 87 |
| K2.5-tp4ep8dp2-32k3k | 4 | 43 | 45 | 100 | 12 | 0 |
| total | 26 | 279 | 115 | 113 | 108 | 87 |

This confirms the table still misses the in-scope 8k2k grid and also loses 32k coverage.

## Validate A/B

| scenario | real tok/s/gpu | phase436 sim | phase436 ratio | phase437 candidate sim | phase437 candidate ratio | classification |
|---|---:|---:|---:|---:|---:|---|
| K2.5-tp8ep8-8k2k | 133.528 | 166.972 | 1.250 | 166.972 | 1.250 | unchanged out of scope |
| K2.5-tp8ep8-32k3k | 52.469 | 56.079 | 1.069 | 56.079 | 1.069 | unchanged out of scope |
| K2.5-tp4ep8dp2-8k2k | 137.716 | 358.061 | 2.600 | 291.700 | 2.118 | improved, still miss-heavy |
| K2.5-tp4ep8dp2-32k3k | 53.277 | 66.840 | 1.255 | 92.400 | 1.734 | regressed: missing 32k rows |
| K2.5-tp8ep8-8k2k-bt65536 | 138.470 | 155.070 | 1.120 | 155.100 | 1.120 | unchanged out of scope |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 113.910 | 142.301 | 1.249 | 142.800 | 1.254 | unchanged |

The candidate table improves `tp4dp2ep8-8k2k`, but it regresses `tp4dp2ep8-32k3k`. That violates the acceptance rule, so the table must not be ingested.

## Phase438 Target

The next collection must use a protocol that guarantees a dense in-scope grid before writing PerfDB:

| need | concrete target |
|---|---|
| mixed 8k2k coverage | include 8k prefill rows at decode batch around 56-64 |
| mixed 32k coverage | restore 32k prefill rows, not just 64k/16k endpoints |
| decode coverage | collect stable decode rows without low-ISL CUDA launch failure |
| query policy | keep inner-only; no clamp, extrapolation, or silent fallback |

The raw traces remain useful evidence, but Phase437 is a failed ingestion attempt, not a model fix.
