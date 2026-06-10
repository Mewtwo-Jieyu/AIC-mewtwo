# Phase178: Budget Mechanism Holdout Runner Preflight

## Decision

| Item | Result |
|---|---|
| Scope | Runner and manifest preflight package |
| GPU benchmark | No-Go |
| Default AIC | No-Go |
| PerfDatabase | No-Go |
| Model formula change | No-Go |
| Push / PR | No-Go |

Phase178 turns the Phase177 holdout matrix into a reviewable runner and manifest contract. It does not authorize a benchmark run.

## Files

| File | Role |
|---|---|
| `collector/vllm/run_phase178_budget_mechanism_holdout.sh` | Print or later run one holdout scenario |
| `scripts/build_phase178_budget_mechanism_holdout_manifest.py` | Build exactly 8-row diagnostic manifest |
| `tests/unit/scripts/test_build_phase178_budget_mechanism_holdout_manifest.py` | Manifest guard tests |
| `phase178_budget_mechanism_runner_preflight.md` | This runner contract |

## Scenario Matrix

| scenario | topology_key | shape_key | max_bt | role |
|---|---|---|---:|---|
| `tp8ep8-4k2k-bt4000` | `tp8_dp1_ep8` | `isl4000_osl2000_batch128` | 4000 | control |
| `tp8ep8-4k2k-bt65536` | `tp8_dp1_ep8` | `isl4000_osl2000_batch128` | 65536 | holdout |
| `tp4dp2ep8-4k2k-bt4000` | `tp4_dp2_ep8` | `isl4000_osl2000_batch128` | 4000 | control |
| `tp4dp2ep8-4k2k-bt65536` | `tp4_dp2_ep8` | `isl4000_osl2000_batch128` | 65536 | holdout |
| `tp8ep8-12k2k-bt12000` | `tp8_dp1_ep8` | `isl12000_osl2000_batch128` | 12000 | control |
| `tp8ep8-12k2k-bt65536` | `tp8_dp1_ep8` | `isl12000_osl2000_batch128` | 65536 | holdout |
| `tp4dp2ep8-12k2k-bt12000` | `tp4_dp2_ep8` | `isl12000_osl2000_batch128` | 12000 | control |
| `tp4dp2ep8-12k2k-bt65536` | `tp4_dp2_ep8` | `isl12000_osl2000_batch128` | 65536 | holdout |

## Runner Contract

| Contract | Requirement |
|---|---|
| Modes | `preflight`, `run-one <scenario>`, `cleanup` |
| Help | Lists only those modes and the 8 scenarios |
| Preflight | Prints 8 serve commands and 8 benchmark commands |
| Preflight boundary | Must print `preflight_benchmark_started=false`; does not call cleanup and does not kill vLLM/Ray |
| Shape | batch 128, output 2000, input 4000 or 12000 |
| Control max_bt | 4000 or 12000 |
| Holdout max_bt | 65536 |
| Logs | `serve.log`, `run_one_<scenario>.log`, `ready_probe_<port>.log`, `cleanup_<scenario>.log` |
| Cleanup | Creates `cleanup_<scenario>.log`, writes `stop_service_start/done` and `cleanup_trap_start/done`, and reuses GPU drain idea: stable empty polls before final after file |
| Flags | `diagnostic_only=true valid_for_default=false perf_database=false` |

`run-one` remains present for the next phase, but Phase178 verification only uses `preflight` and does not start service or benchmark.

## Manifest Contract

| Gate | Requirement |
|---|---|
| Row count | Exactly 8 rows |
| Scenario set | Exactly the Phase177 matrix |
| Duplicate scenario | Fail-fast |
| Pairing | Every topology and shape has one control and one high-budget row |
| Requests | `request_success_count=128`, `request_fail_count=0` expected for accepted rows |
| Flags | `diagnostic_only=true`, `valid_for_default=false`, `perf_database=false` |
| Schema | Includes clean throughput, topology, shape, scheduler, runtime, GPU, commands |

## No-Go

| Item | Decision |
|---|---|
| Default AIC | No-Go |
| PerfDatabase | No-Go |
| `VLLMBackend.run_agg` | No-Go |
| Interpolation / extrapolation | No-Go |
| GPU benchmark | No-Go in Phase178 |
| PR / push | No-Go |

Phase179 may package these files with a dry-run whitelist. Target-container preflight or GPU run must be a separate phase.

diagnostic_only=true valid_for_default=false perf_database=false
