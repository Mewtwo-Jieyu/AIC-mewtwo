# Phase122: AIC Integration Point Audit

## Scope

This is a read-only audit. Phase122 records future integration points and risks. It does not change code.

## Source Audit

| Source | Current behavior | Default-integration risk | Phase122 decision |
|---|---|---|---|
| `src/aiconfigurator/sdk/perf_database.py` `load_moe_data(...)` | Loads MoE CSV into the existing performance database shape | Existing format uses generic MoE timing rows and is not the Phase117 exact-key contract | Do not write Phase117 table into this loader |
| `src/aiconfigurator/sdk/perf_database.py` `query_moe(...)` | Queries MoE table and can interpolate by token count | Phase117 query policy is exact-match only, so silent interpolation would break the evidence boundary | Do not attach the experimental table here before a new exact-key gate |
| `src/aiconfigurator/sdk/backends/vllm_backend.py` `VLLMBackend.run_agg(...)` | Routes aggregate validation through existing backend methods | A MoE-only table could be misread as full aggregate latency evidence | Keep Phase117/118 outside aggregate validation |
| `src/aiconfigurator/sdk/backends/vllm_backend.py` cb_sim aggregate path | Produces aggregate output via simulator/backend wiring | A module table could affect aggregate claims without end-to-end closure | Require e2e holdout before any default route |
| `src/aiconfigurator/sdk/backends/cb_simulator/iteration_latency.py` `IterationLatencyCalculator` | Calls backend static latency queries inside iteration accounting | Changing this path would alter default mixed prefill/decode behavior | No changes until all default gates pass |

## Required Future Design

| Topic | Requirement |
|---|---|
| Exact-key mapping | Runtime MoE key must map one-to-one to the table row |
| Missing key behavior | Missing key must fail fast |
| Interpolation | Not allowed unless a later gate proves it with holdout data |
| Fallback | No empirical fallback for this table |
| Rollback | A default switch must return to the Phase4 baseline |

## No-Code Confirmation

Phase122 does not modify `PerfDatabase`, `run_static`, `IterationLatencyCalculator`, backend query code, or default validation code.
