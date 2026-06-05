# Phase164: Clean GPU Budget Benchmark Runner Preflight

## Decision

| Item | Result |
|---|---|
| Scope | Runner, preflight, and clean manifest builder |
| Default AIC | No-Go |
| PerfDatabase | No-Go |
| Model formula change | No-Go |
| Remote full run | No-Go until preflight is accepted |

Phase164 prepares the clean GPU benchmark path for the Phase163 modeling gate. It does not use old Phase154 / Phase159 rows as model evidence.

## Remote Entry

| Item | Value |
|---|---|
| Route | `h200_ais -> h382 -> docker exec jieyu_aic_vllm019` |
| Workdir | `/mnt/nvme1n1/ml_research/jieyu/aic` |
| Model | `/mnt/cfs/models/kimi-2.5-fp8` |
| Port base | `18000` |
| Output root | `docs/iter_gap_investigation/phase164_clean_gpu_budget_benchmark` |

The runner exports CUDA compatibility environment:

```bash
export PATH=/usr/local/nvidia/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export VLLM_ENABLE_CUDA_COMPATIBILITY=1
```

Phase164e proved service init can reach `/v1/models` after container cleanup when
the CUDA library path includes `/usr/local/cuda/lib64`. The runner uses that
stable CUDA symlink instead of a versioned CUDA directory.

## Runner Modes

| Mode | Command | Behavior |
|---|---|---|
| preflight | `bash collector/vllm/run_phase164_clean_budget_benchmark.sh preflight` | prints all serve and benchmark commands; checks GPU state; does not start benchmark |
| run-one | `bash collector/vllm/run_phase164_clean_budget_benchmark.sh run-one <scenario>` | runs one shape, writes clean JSON, then cleans vLLM / Ray |
| cleanup | `bash collector/vllm/run_phase164_clean_budget_benchmark.sh cleanup` | stops vLLM and Ray residue |

Preflight is the next allowed remote action. Full four-shape execution is blocked until preflight output is reviewed.

## Benchmark Matrix

| Scenario | Serve parallelism | max_bt | Benchmark |
|---|---|---:|---|
| `tp8ep8-bt8000` | `--tensor-parallel-size 8 --enable-expert-parallel` | 8000 | input 8000 / output 2000 / concurrency 128 |
| `tp8ep8-bt65536` | same | 65536 | same |
| `tp4dp2ep8-bt8000` | `--tensor-parallel-size 4 --data-parallel-size 2 --enable-expert-parallel` | 8000 | same |
| `tp4dp2ep8-bt65536` | same | 65536 | same |

Default benchmark request settings:

| Field | Value |
|---|---:|
| `num_prompts` | 128 |
| `max_concurrency` | 128 |
| `input_len` | 8000 |
| `output_len` | 2000 |
| `warmup_requests` | 0 |

## Output Files

Each `run-one` scenario writes:

| File | Content |
|---|---|
| `serve.log` | raw vLLM serve log |
| `run_one_<scenario>.log` | runner stdout/stderr for that one scenario |
| `ready_probe_<port>.log` | `/v1/models` readiness probe log |
| `cleanup_<scenario>.log` | service stop and cleanup log |
| `bench_result.json` | raw `run_openai_fixed_shape_benchmark.py` summary |
| `bench_records.jsonl` | per-request records |
| `gpu_compute_apps_before.txt` | GPU process state before run |
| `gpu_compute_apps_after.txt` | GPU process state after cleanup |
| `phase164_result.json` | clean benchmark metadata consumed by the manifest builder |

The manifest builder reads four `phase164_result.json` files:

```bash
python3 scripts/build_phase164_clean_benchmark_manifest.py \
  --input-root docs/iter_gap_investigation/phase164_clean_gpu_budget_benchmark \
  --out docs/iter_gap_investigation/phase164_clean_gpu_budget_manifest.csv
```

The manifest is accepted only when it has exactly 4 data rows and each fixed scenario appears exactly once.

## Clean Schema

| Group | Required fields |
|---|---|
| Result | `ok_requests`, `failed_requests`, `output_tok_s`, `total_tok_s` |
| Shape | `isl`, `osl`, `batch_size`, `max_num_batched_tokens`, `max_num_seqs` |
| Runtime | `vllm_version`, `model_path`, `dtype`, `quantization`, `serve_command`, `benchmark_command` |
| Parallelism | `tp`, `dp`, `ep`, `world_size`, `gpu_count` |
| Hardware | `gpu_model`, `gpu_memory_gb`, `driver_version`, `cuda_version` |
| Boundary | `diagnostic_only=true`, `valid_for_default=false`, `perf_database=false` |

The manifest computes:

| Manifest field | Definition |
|---|---|
| `real_output_tok_s_gpu` | `output_tok_s / gpu_count` |
| `real_total_tok_s_gpu` | `total_tok_s / gpu_count` |
| `request_success_count` | `ok_requests` |
| `request_fail_count` | `failed_requests` |

Any request failure is a blocker. Any missing schema field is a blocker.

## Stop Conditions

| Check | Stop condition |
|---|---|
| GPU state | any compute process exists unless explicitly allowed for debugging |
| Command print | missing model, port, topology, `max_num_batched_tokens`, input/output shape, or concurrency |
| Service readiness | `/v1/models` does not become ready |
| Traffic health | `failed_requests != 0` or `ok_requests != 128` |
| Schema | missing runtime, hardware, shape, or diagnostic flags |
| Cleanup | GPU process remains after service stop |

## No-Go Boundary

| Path | Decision |
|---|---|
| Default AIC | No-Go |
| PerfDatabase | No-Go |
| Model formula change | No-Go |
| Bare constant patch | No-Go |
| Phase165 candidate model item | No-Go until clean data exists |

Phase164 data can feed Phase165 candidate design only after the four-row manifest is complete and accepted.

## Verification

| Check | Command |
|---|---|
| runner syntax | `bash -n collector/vllm/run_phase164_clean_budget_benchmark.sh` |
| manifest compile | `python3 -m py_compile scripts/build_phase164_clean_benchmark_manifest.py` |
| manifest tests | `PYTHONPATH=src conda run -n aic python -m pytest tests/unit/scripts/test_build_phase164_clean_benchmark_manifest.py -q` |
| preflight | remote only: `bash collector/vllm/run_phase164_clean_budget_benchmark.sh preflight` |

Phase164 closeout must keep runner, manifest builder, tests, and Phase163/164 docs in a fixed whitelist.
