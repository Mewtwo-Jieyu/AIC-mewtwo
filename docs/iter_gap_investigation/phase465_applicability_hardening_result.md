# Phase465 applicability hardening result

结论：Track A 通过。Kimi 专用 serving-state、Phase397v int4 校准和 EP8 来源选择已收紧；
六个正式场景的 cb_sim 输出逐项不变。本阶段只修 correctness，不提升精度，不改变 PerfDatabase 数据，
Default AIC 继续 `No-Go`。

## Boundary

| 项 | 值 |
|---|---|
| base commit | `9ba5ad93ca8805a1eb427593ced8cee01aea6804` |
| result commit | `32e356473129725dc05b30bba225d1cf1095e7ce` |
| branch | `experiment/phase465-applicability-hardening` |
| outcome | PASS |
| runtime scope | `moonshotai/Kimi-K2.5` / H200 SXM / vLLM 0.19.0 |
| data/schema change | none |
| GPU/SSH | not used |
| readiness | `diagnostic_only=true`; `valid_for_default=false`; `perf_database=false`; `Default AIC=No-Go` |

Changed files:

- `src/aiconfigurator/sdk/backends/cb_simulator/iteration_latency.py`
- `src/aiconfigurator/sdk/operations.py`
- `src/aiconfigurator/sdk/perf_database.py`
- `tests/unit/sdk/backends/test_cb_simulator.py`
- `tests/unit/sdk/database/test_phase397v_int4_wo_sol_calibration.py`

## Correctness changes

| Area | Previous risk | Current rule |
|---|---|---|
| serving-state | another model could read Kimi rows | exact runtime model is required; missing model metadata raises |
| Phase397v int4 | calibrated roofline was selected without model identity | exact Kimi model, K2.5 MoE structure and documented `(tp1,ep8)/(tp16,ep1)` topologies are required |
| module binding | Kimi on the measured runtime could silently omit topology | missing `vllm_module_topology` raises; explicit non-Kimi/other-topology/old-version paths are unchanged |
| EP8 source | exceptions selected measured versus structural behavior | measured coverage is queried first; caller passes explicit `measured` or `structural` source |
| EP8 structural model | missing or non-positive bandwidth could fail indirectly | required `node.intra_node_bw` is validated explicitly |

Phase397v remains a calibrated roofline over the documented K2.5 model family, not an exact 128-token table.
An intermediate implementation incorrectly reduced it to one token/topology point and changed four formal outputs;
review rejected that implementation before commit. The final change preserves the original continuous token scaling and
both topologies already validated by the Phase397v spec.

## Input provenance

| Input | SHA-256 |
|---|---|
| `phase397v_int4_wo_sol_calibration_spec.md` | `5ce599c5500b8c56bb45f2a32b7610c5c9488785da2fbb5652d8415938d6982c` |
| `vllm_ep8_a2a_decode_perf.txt` | `fb2a6a3596262601bfc93a31eeb0c9b4e51290bc4b4328073330d6b0fb57c658` |
| `vllm_serving_state_perf.txt` | `58415dc84b7994b337ec358f2bfe4dd950f1ce2ae48d71976756a88ae8397889` |

No input data row was added or modified.

## Verification

```text
PYTHONPYCACHEPREFIX=/tmp/phase465_a_pyc \
PYTHONPATH=<phase465-worktree>/src \
<feature-pr403>/.venv312/bin/python -m pytest -q \
  tests/unit/sdk/database/test_vllm_module_perf.py \
  tests/unit/sdk/database/test_phase397v_int4_wo_sol_calibration.py \
  tests/unit/sdk/backends/test_cb_simulator.py

PYTHONPYCACHEPREFIX=/tmp/phase465_a_pyc \
PYTHONPATH=<phase465-worktree>/src \
<feature-pr403>/.venv312/bin/python scripts/validate_cb_simulator.py
```

| Check | Result |
|---|---|
| targeted tests | `116 passed` |
| py_compile | PASS |
| six-point raw cb_sim output | `166.0 / 49.5 / 163.1 / 64.0 / 155.9 / 127.5` |
| multi-config gate | `max=1.26x`, `mean=1.13x`, PASS |
| legacy 0.17 throughput/TTFT | unchanged and still SKIP |
| diff/whitespace | PASS |

## Remaining boundary

This result does not validate another model, hardware, vLLM version or topology. It does not prove the Phase397v
calibration physically generalizes beyond the documented K2.5 family. It does not resolve the three failed Phase463
cells or define TTFT/TPOT acceptance gates. The next measurement step remains the Phase466 probe overhead gate after
this branch is integrated.
