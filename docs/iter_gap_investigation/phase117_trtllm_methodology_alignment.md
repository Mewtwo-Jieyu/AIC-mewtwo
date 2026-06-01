# Phase117 TRT-LLM Methodology Alignment

## Decision

Phase117 aligns with TRT-LLM methodology only at the discipline level. It does not reuse TRT-LLM data, formulas, constants, or GB200 assumptions.

| TRT-LLM method discipline | vLLM WNA16 Phase117 alignment |
|---|---|
| Module-level evidence | Table describes only MoE WNA16 aggregate timing |
| Exact key | Table key fixes backend, version, model, device, config, topology, and token shape |
| Perf table discipline | Missing key fails instead of falling back |
| Query interface discipline | Query must be explicit and experimental-only |
| Validation discipline | Phase116 holdout errors stay as risk evidence |

## Non-Reuse Boundaries

| Not reused | Reason |
|---|---|
| TRT-LLM perf data | Different backend and kernel path |
| GB200 constants | Different hardware |
| TRT-LLM formula | Phase116 showed formula risk; Phase117 uses table lookup |
| Full-forward gap | Not a MoE module model |
| Residual bucket | Not physical module timing |

## AIC Gap Still Open

| Gap | Status |
|---|---|
| Real activation distribution | Missing |
| More token shapes | Missing |
| Other topology coverage | Missing |
| End-to-end validation | Missing |
| Default AIC integration | No-Go |
