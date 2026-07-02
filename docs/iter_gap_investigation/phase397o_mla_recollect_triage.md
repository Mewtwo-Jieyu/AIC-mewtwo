# Phase397o MLA Recollect Triage

| Item | Result |
|---|---|
| Scope | generation_mla recollection triage only; no runtime/DB/gate change |
| Raw artifact | docs/iter_gap_investigation/phase397o_mla_recollect/phase397o_mla_recollect_raw.csv |
| Sweep | heads 8/16 x dtype float16/fp8 x batch 64/128 x seq 8192/9001/16384 x randomize true/false = 48 rows |
| Primary verdict | B2_kernel_workload_or_serve_seed_difference |
| Default AIC | No-Go |

## Key points

| heads | dtype | batch | seq | backend | eager6 ms | eager200 p50 ms | graph p50 ms | serve/layer ms | graph/serve | graph/eager6 | randomize false/true |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | float16 | 128 | 8192 | FlashAttnMLAImpl | 0.445781 | 0.464192 | 0.439584 | 0.277018 | 1.586843x | 0.986099x | 0.996579x |
| 8 | float16 | 128 | 9001 | FlashAttnMLAImpl | 0.488960 | 0.505952 | 0.481504 | 0.277018 | 1.738168x | 0.984751x | 0.997275x |
| 16 | float16 | 64 | 9001 | FlashAttnMLAImpl | 0.253963 | 0.270432 | 0.245824 | 0.223752 | 1.098643x | 0.967952x | 0.998047x |

## Decision

| Candidate | Verdict | Evidence |
|---|---|---|
| cuda_graph_launch_overhead | timing_method_not_sufficient | graph_p50 stays above 1.7x serve per-layer on the heads=8 batch=128 seq=9001 point |
| randomized_block_layout | randomized_block_layout_not_primary | max graph_p50 randomize false/true delta is 0.0130; layout toggle does not explain the residual |
| fp8_dtype_control | fp8_backend_not_comparable_to_db_flashmla | Phase397o fp8 recollection uses TritonMLAImpl/vllm_triton_mla because block-size 16 excludes FLASHMLA; it is a side control, not a DB-key replacement |
| phase397q_trigger | B2_kernel_workload_or_serve_seed_difference | CUDA graph and block randomization controls do not collapse heads=8 batch=128 float16 to serve timing; next evidence must reproduce serve seed/layout or collect exact table rows with a matching harness |

## Phase397q trigger

- Do not introduce a scalar multiplier; heads=8 and heads=16 need different corrections.
- Do not update `generation_mla_perf.txt` from this evidence alone; fp8 recollection used TritonMLAImpl, not DB `vllm_flashmla`.
- Next evidence must either reproduce serve seed/layout directly or recollect exact rows with a harness proven to match serve.
- Default AIC remains No-Go.

## Cleanup

- `gpu_compute_apps_after.txt` shows 0MiB on all 8 H200 GPUs after clearing stale VLLM workers.
- `process_residual_after.txt` is empty.
