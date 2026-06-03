# Phase126: Runner Preflight

## Decision

| Item | Result |
|---|---|
| Phase126 step | Runner hardening only |
| Remote run | Not done |
| Exact-shape capture | Not done |
| Default AIC | No-Go |
| PerfDatabase | No-Go |

## Runner Contract

| Requirement | Runner value |
|---|---|
| Script | `collector/vllm/run_phase126_moe_activation_marker.sh` |
| Port | `PORT=${PORT:-18000}` |
| Readiness URL | `http://127.0.0.1:${PORT}/v1/models` |
| Benchmark port | Same `PORT` |
| Serve mode | `--enforce-eager` |
| Marker enable | `ENABLE_FILE`, touched only after service readiness |
| Marker close point | after drain stable |
| Drain log | `drain_boundary.log` |
| Drain stable condition | `DRAIN_STABLE_POLLS=${DRAIN_STABLE_POLLS:-3}` |
| Drain poll interval | `DRAIN_POLL_SECONDS=${DRAIN_POLL_SECONDS:-5}` |
| Drain timeout | `DRAIN_TIMEOUT_SECONDS=${DRAIN_TIMEOUT_SECONDS:-900}` |
| CUDA compatibility | `VLLM_ENABLE_CUDA_COMPATIBILITY=${VLLM_ENABLE_CUDA_COMPATIBILITY:-1}` |
| Tuned config | `VLLM_TUNED_CONFIG_FOLDER=${TUNED_CONFIG_DIR}` |

## Marker Boundary

| Field | Value |
|---|---|
| `input_kind` | `real_forward_hidden_states` |
| `activation_source` | `loaded_model_forward` |
| `called_model_forward` | `true` |
| `timing` | `false` |
| `moe_config_fallback` | `false` |
| `loaded_weight/random_weight` | `true / false` |
| `valid_for_default/perf_database` | `false / false` |

The `timing=false` field remains because the Phase123 parser schema requires it. The runner does not add any time measurement fields.

## Local Validation

| Check | Result |
|---|---|
| `bash -n collector/vllm/run_phase126_moe_activation_marker.sh` | PASS |
| `python3 -m py_compile scripts/analyze_vllm_moe_activation_phase123.py` | PASS |
| `python3 -m py_compile scripts/build_moe_wna16_shape_coverage_phase125.py` | PASS |
| `python3 -m py_compile tests/unit/scripts/test_build_moe_wna16_shape_coverage_phase125.py` | PASS |
| Phase125 manifest review diff | PASS |
| `git diff --check` | PASS |
| Staged area | Empty |

## Dry-Run Status

| Check | Result |
|---|---|
| Local shell syntax | PASS |
| Target container patch dry-run | PASS after drain update |
| Original vLLM source marker check | PASS: absent |
| vLLM serve process check | PASS: absent |

Target dry-run output:

```text
351:        aic_phase126_enable_file = os.environ.get("AIC_PHASE126_MOE_ACTIVATION_ENABLE_FILE")
387:            print("AIC_MOE_ACTIVATION_EVIDENCE_ROW " + json.dumps(aic_payload, sort_keys=True), flush=True)
patch_dry_run=PASS
```

## Next Gate

Drain-boundary design and runner preflight are accepted. A full capture run remains blocked until a separate capture2 plan is accepted. The target buckets are `1 / 15 / 16 / 241 / 1808 / 2048 / 8192`.
