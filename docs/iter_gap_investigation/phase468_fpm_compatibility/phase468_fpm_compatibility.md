# Phase468 FPM Input and Runtime Compatibility

Status: `READY_FOR_RUNTIME_PREFLIGHT`.

| Check | Result |
|---|---|
| Source contract | PASS |
| Scenarios | 6/6 |
| Descriptor rows | 347984 |
| Numeric max delta | 1.4210854715202004e-14 |
| Worker/Dynamo method check | Deferred to Phase469 preflight |
| Default AIC | No-Go |

## Scenario Coverage

| Scenario | Execution mode | Rows | Steps | Ranks | Rank 1 |
|---|---|---:|---:|---|---|
| K2.5-tp8ep8-8k2k | tp_single_replica | 12118 | 12118 | 0 | not_applicable |
| K2.5-tp8ep8-32k3k | tp_single_replica | 84063 | 84063 | 0 | not_applicable |
| K2.5-tp4ep8dp2-8k2k | dp_lockstep | 23600 | 11800 | 0,1 | present |
| K2.5-tp4ep8dp2-32k3k | dp_lockstep | 174064 | 87032 | 0,1 | present |
| K2.5-tp8ep8-8k2k-bt65536 | tp_single_replica | 20014 | 20014 | 0 | not_applicable |
| K2.5-tp4ep8dp2-8k2k-bt65536 | dp_legacy_representative | 34125 | 34125 | 0 | not_executed_by_official_legacy_path |

## Exact Support Evidence

- Collector entry: `build_collection_plan(has_model_cases=True)`
- Attention source: `mla_module`
- Actual operations: `ContextMLA`, `GenerationMLA`
- Model config SHA256: `58944cf9cf456619768616f38acd75b3d7ccc26d8b6a64d9e312aba4c8f0b927`

The Phase466 execution-manifest SHA is retained only as historical evidence. Phase469 must attest the flat model identity separately before model load.

`diagnostic_only=true`, `valid_for_default=false`, `perf_database=false`.
