# Phase159: Budget / Scheduler Breakdown Diagnostic

## Decision

| Item | Result |
|---|---|
| Scope | Local root-cause diagnostic tooling |
| Default AIC | No-Go |
| PerfDatabase | No-Go |
| Remote benchmark | No-Go |
| Model formula change | No-Go |

Phase159 adds a budget/scheduler breakdown view for the multi-config rows. It explains the `bt65536` ranking miss without changing the cb_sim model.

## Problem

| Row | Real rank | cb_sim rank | Diagnostic target |
|---|---:|---:|---|
| `K2.5-tp4ep8dp2-8k2k-bt65536` | `1` | `6` | compare scheduler shape against max_bt=`isl` baseline |
| `K2.5-tp8ep8-8k2k-bt65536` | `2` | `5` | compare scheduler shape against max_bt=`isl` baseline |

The likely root-cause direction is budget / scheduler semantics under `max_num_batched_tokens=65536`. Phase159 does not treat Phase127 MoE buckets or Phase148 EP8 overhead as clean timing evidence.

## Command

```bash
PYTHONPATH=src python scripts/validate_cb_simulator.py \
  --diagnostic-multi-config-budget-breakdown
```

Optional CSV output:

```bash
PYTHONPATH=src python scripts/validate_cb_simulator.py \
  --diagnostic-multi-config-budget-breakdown \
  --diagnostic-multi-config-budget-breakdown-out /private/tmp/phase159_budget_breakdown.csv
```

## Output Contract

| Field group | Fields |
|---|---|
| Scenario | `name`, `tp`, `dp`, `ep`, `max_bt` |
| Throughput ranking | `real_output_tok_s_gpu`, `sim_output_tok_s_gpu`, `error_ratio`, `rank`, `real_rank` |
| Scheduler averages | `avg_prefill_reqs_per_iter`, `avg_decode_reqs_per_iter`, `avg_tokens_per_iter` |
| Scheduler peaks | `peak_prefill_reqs_per_iter`, `peak_decode_reqs_per_iter`, `peak_tokens_per_iter` |
| Steady state | `steady_state_iterations`, `steady_state_time_ms` |
| Paired baseline | `paired_baseline_name`, `paired_baseline_max_bt`, `sim_vs_paired_baseline_ratio`, `avg_tokens_vs_paired_baseline_ratio` |
| Boundary flags | `diagnostic_only=true`, `valid_for_default=false`, `perf_database=false` |

The paired baseline is the same `isl` / `osl` / `batch_size` / topology row with `max_bt=isl`. If a row cannot find that baseline, the diagnostic fails.

## Boundary

| Forbidden use | Reason |
|---|---|
| Default AIC integration | The output is diagnostic metadata only |
| PerfDatabase write | No accepted timing source is produced |
| Bare constant patch | The breakdown must point to a keyed mechanism first |
| Phase127 bucket reuse | Those buckets are descriptor evidence, not timing data |
| Remote benchmark claim | Phase159 is local-only |

If the breakdown shows that `bt65536` changes scheduler shape in a repeatable way, Phase160 can design a keyed model item. That item still needs a separate acceptance gate before default use.
