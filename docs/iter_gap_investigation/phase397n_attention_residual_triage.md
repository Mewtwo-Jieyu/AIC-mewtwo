# Phase397n Attention Residual Triage

| Item | Result |
|---|---|
| Scope | report-only offline triage for vLLM generation MLA decode |
| Workload | Kimi-K2.5 / h200_sxm / vLLM 0.19.0 / decode 8k2k |
| Primary residual | tp8 attention cb_sim 31.891ms vs measured 16.898ms = 1.887x |
| Verdict | primary hypothesis is generation_mla microbench anchor/workload mismatch, not dtype or layer count |
| Phase397o | recollect generation_mla with CUDA graph replay and paged/varlen-like decode layout |
| Default AIC | No-Go |

## Query reproduction

| Config | KV dtype | heads/GPU | batch | seq | DB ms/layer | x61 ms | measured ms | sim/measured |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tp8ep8-8k2k | float16 | 8 | 128 | 9001 | 0.523 | 31.891 | 16.898 | 1.887x |
| tp8ep8-8k2k | fp8 | 8 | 128 | 9001 | 0.442 | 26.981 | 16.898 | 1.597x |
| tp4ep8dp2-8k2k | float16 | 16 | 69 | 9001 | 0.290 | 17.670 | 13.649 | 1.295x |
| tp4ep8dp2-8k2k | fp8 | 16 | 69 | 9001 | 0.248 | 15.138 | 13.649 | 1.109x |

## Candidate triage

| Candidate | Offline verdict | Evidence |
|---|---|---|
| layer_mtp_count | eliminated_offline | Kimi-K2.5 nextn=0 gives _mtp_scale_factor=1; GenerationMLA is charged as 61 layers, matching Phase397l. |
| dtype_key | not_primary | Runtime kernel is bf16/FlashAttnFwdSm90, mapping to the float16 row. fp8 still predicts 26.981ms vs measured 16.898ms (1.597x). |
| interp_overshoot | not_primary | Interpolation to s=9001 stays between 8192 and 16384 anchors. The 8192 float16 anchor is already 0.486ms/layer = 1.755x measured per-layer. |
| anchor_microbench_high | primary_hypothesis | Current query gives 0.523ms/layer and 31.891ms total vs 16.898ms measured (1.887x). The nearest low anchor itself is high. |
| collector_workload_method | needs_phase397o | collector/vllm/collect_mla.py times eager forward_mqa with CUDA events and no graph replay (lines 303-317); paged/varlen layout and real serve kernel path need GPU recollection. |

## Phase397o recollection spec

| Item | Evidence target |
|---|---|
| cuda_graph_vs_eager_float16_heads8_b128_s9001 | num_heads=8,batch=128,kv_dtype=float16,seq_len around 8192/9001/16384; compare eager DB row vs CUDA-graph replay vs Phase397l serve. |
| cuda_graph_vs_eager_fp8_heads8_b128_s9001 | num_heads=8,batch=128,kv_dtype=fp8,seq_len around 8192/9001/16384; dtype control only, not expected primary fix. |
| cuda_graph_vs_eager_float16_heads16_b128_s9001 | num_heads=16,batch=128,kv_dtype=float16,seq_len around 8192/9001/16384; tp4 head-count control. |
| cuda_graph_vs_eager_fp8_heads16_b128_s9001 | num_heads=16,batch=128,kv_dtype=fp8,seq_len around 8192/9001/16384; tp4 head-count dtype control. |
| paged_varlen_layout_match | capture attn_metadata/kv-cache layout path close to real decode; classify timing-method vs kernel/workload mismatch. |

## Guardrails

- Phase397n does not modify `models.py`, `operations.py`, `perf_database.py`, perf tables, or gates.
- No new scale factor is introduced; MoE `k=0.312` is not applied to attention.
- CSV contains the interpolation anchors; `generation_mla_perf.txt` is not changed.
- Phase397o may use GPU for recollection, but this phase does not use SSH/GPU.
- Default AIC remains No-Go.
