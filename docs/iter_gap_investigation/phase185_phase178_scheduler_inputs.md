# Phase185: Phase178 Scheduler Inputs

## Decision

| Item | Result |
|---|---|
| Scope | Local cb_sim scheduler input table for Phase178 holdout run-one |
| Data source | `cb_sim_scheduling.steady_state_time_ms` from cb_sim per-ops data |
| GPU benchmark | No-Go in Phase185 |
| Default AIC | No-Go |
| PerfDatabase | No-Go |

## Usage

Use the row for the scenario being run and export the exact env value before `run-one`:

```bash
PHASE178_STEADY_STATE_TIME_MS=<csv steady_state_time_ms> \
bash collector/vllm/run_phase178_budget_mechanism_holdout.sh run-one <scenario>
```

## Scheduler Inputs

| scenario | topology_key | shape_key | max_bt | steady_state_time_ms | env |
|---|---|---|---:|---:|---|
| tp8ep8-4k2k-bt4000 | tp8_dp1_ep8 | isl4000_osl2000_batch128 | 4000 | 522072.148306 | `PHASE178_STEADY_STATE_TIME_MS=522072.148306` |
| tp8ep8-4k2k-bt65536 | tp8_dp1_ep8 | isl4000_osl2000_batch128 | 65536 | 1925699.150926 | `PHASE178_STEADY_STATE_TIME_MS=1925699.150926` |
| tp4dp2ep8-4k2k-bt4000 | tp4_dp2_ep8 | isl4000_osl2000_batch128 | 4000 | 616788.494005 | `PHASE178_STEADY_STATE_TIME_MS=616788.494005` |
| tp4dp2ep8-4k2k-bt65536 | tp4_dp2_ep8 | isl4000_osl2000_batch128 | 65536 | 4219264.491761 | `PHASE178_STEADY_STATE_TIME_MS=4219264.491761` |
| tp8ep8-12k2k-bt12000 | tp8_dp1_ep8 | isl12000_osl2000_batch128 | 12000 | 699735.503859 | `PHASE178_STEADY_STATE_TIME_MS=699735.503859` |
| tp8ep8-12k2k-bt65536 | tp8_dp1_ep8 | isl12000_osl2000_batch128 | 65536 | 1721546.001970 | `PHASE178_STEADY_STATE_TIME_MS=1721546.001970` |
| tp4dp2ep8-12k2k-bt12000 | tp4_dp2_ep8 | isl12000_osl2000_batch128 | 12000 | 856064.657069 | `PHASE178_STEADY_STATE_TIME_MS=856064.657069` |
| tp4dp2ep8-12k2k-bt65536 | tp4_dp2_ep8 | isl12000_osl2000_batch128 | 65536 | 3594084.520777 | `PHASE178_STEADY_STATE_TIME_MS=3594084.520777` |

## Boundary

These rows are scheduler diagnostic inputs only. They are not clean GPU timing,
not default cb_sim data, and not PerfDatabase rows.

diagnostic_only=true valid_for_default=false perf_database=false
