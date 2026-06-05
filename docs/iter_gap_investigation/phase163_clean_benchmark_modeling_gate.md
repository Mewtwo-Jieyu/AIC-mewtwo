# Phase163: Clean GPU Benchmark + Theoretical Modeling Gate

## Decision

| Item | Result |
|---|---|
| Scope | Clean benchmark design and modeling gate |
| Default AIC | No-Go |
| PerfDatabase | No-Go |
| Remote full benchmark | No-Go in Phase163 |
| Model formula change | No-Go |
| Bare residual constant | No-Go |

Phase163 changes the next modeling step from "fit old measurements" to "define a mechanism first, then validate it with clean GPU data from our own vLLM service."

## Data Boundary

| Data type | Allowed use | Forbidden use |
|---|---|---|
| Phase154 Top-K rows | choose shapes and reproduce the rank issue | model calibration |
| Phase159 breakdown rows | identify scheduler / budget variables | accepted timing evidence |
| Phase127 MoE buckets | exact-shape descriptor reference | clean timing evidence |
| New Phase164 GPU benchmark | model validation and holdout candidate input | default AIC without later gate |

Old rows are diagnostic entry points only. They can explain why we should measure `bt65536`, but they cannot justify a cb_sim default model term.

## Clean Benchmark Matrix

| Scenario | tp | dp | ep | isl | osl | batch_size | max_num_batched_tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tp8ep8-8k2k-bt8000` | 8 | 1 | 8 | 8000 | 2000 | 128 | 8000 |
| `tp8ep8-8k2k-bt65536` | 8 | 1 | 8 | 8000 | 2000 | 128 | 65536 |
| `tp4ep8dp2-8k2k-bt8000` | 4 | 2 | 8 | 8000 | 2000 | 128 | 8000 |
| `tp4ep8dp2-8k2k-bt65536` | 4 | 2 | 8 | 8000 | 2000 | 128 | 65536 |

The matrix is intentionally small: one base shape, two topology choices, two budget settings. Phase164 must not add extra shapes unless this gate is revised.

## Required Clean Evidence Fields

| Group | Required fields |
|---|---|
| Result | `real_output_tok_s_gpu`, `real_total_tok_s_gpu`, `request_success_count`, `request_fail_count` |
| Shape | `isl`, `osl`, `batch_size`, `max_num_batched_tokens`, `max_num_seqs` |
| Runtime | `vllm_version`, `model_path`, `dtype`, `quantization`, `serve_command`, `benchmark_command` |
| Parallelism | `tp`, `dp`, `ep`, `world_size`, `gpu_count` |
| Hardware | `gpu_model`, `gpu_memory_gb`, `driver_version`, `cuda_version` |
| Boundary | `diagnostic_only=true`, `valid_for_default=false`, `perf_database=false` |

Any row missing these fields is not clean model evidence. `bench_result.json` can prove traffic health only if it also records request success and failure counts.

## Phase164 Runner / Preflight Contract

| Check | Stop condition |
|---|---|
| GPU state | unknown compute process on target GPUs |
| vLLM config | requested `tp/dp/ep/max_bt` not visible in serve command or config |
| model | model path or quantization differs across the 4 rows |
| traffic | any request failure or missing request count |
| output | missing throughput fields or missing serve metadata |
| schema | missing `diagnostic_only`, `valid_for_default`, or `perf_database` flags |
| cleanup | vLLM / Ray residue remains after run |

Phase164 may build a runner or wrapper, but it must first print the exact serve command and benchmark command. The actual benchmark only starts after the preflight output is reviewable.

## Theoretical Candidate Variables

| Variable | Meaning | Source before Phase164 |
|---|---|---|
| `budget_ratio = max_num_batched_tokens / isl` | how much token budget can be packed per iteration | configured shape |
| `peak_token_ratio = peak_tokens_per_iter / isl` | observed scheduler peak relative to prompt length | cb_sim diagnostic, later runner telemetry |
| `steady_state_iter_ms = steady_state_time_ms / steady_state_iterations` | per-iteration scheduler cost surface | cb_sim diagnostic, later runner telemetry if available |
| `avg_decode_reqs_per_iter` | decode pressure under the budget | cb_sim diagnostic |
| `avg_prefill_reqs_per_iter` | prefill packing under the budget | cb_sim diagnostic |
| `topology_key = tp/dp/ep` | parallel communication and scheduling domain | configured shape |

The candidate model item must be keyed by shape and topology. A valid Phase165 candidate can use these variables to explain why `bt65536` changes rank; it cannot add one constant that only forces `bt65536` to look better.

## Candidate Formula Shape

The only allowed Phase165 direction is a mechanism-shaped correction:

```text
effective_iteration_cost =
    base_compute_cost(topology_key, isl, osl, batch_size)
  + scheduler_budget_cost(
        topology_key,
        budget_ratio,
        peak_token_ratio,
        avg_prefill_reqs_per_iter,
        avg_decode_reqs_per_iter,
        steady_state_iter_ms
    )
```

The output remains candidate metadata:

| Field | Required value |
|---|---|
| `source` | `phase164_clean_gpu_benchmark` |
| `diagnostic_only` | `true` |
| `valid_for_default` | `false` |
| `perf_database` | `false` |

Default cb_sim can only be discussed after a later holdout gate proves the candidate is stable outside the 4-row design matrix.

## Go / No-Go

| Gate | Result |
|---|---|
| Clean benchmark design | Go |
| Four-shape matrix | Go |
| Theoretical mechanism variables | Go |
| Phase164 full GPU benchmark | No-Go until separate phase |
| Phase165 model implementation | No-Go until clean data exists |
| Default AIC | No-Go |
| PerfDatabase | No-Go |

## Next Phases

| Phase | Action |
|---|---|
| Phase164 | run the clean GPU benchmark on the 4 fixed rows |
| Phase165 | compare clean data against the mechanism variables and design a candidate model item |
| Phase166 | only after holdout, decide whether default cb_sim can use the candidate |
