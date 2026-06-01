# Phase119 Review Checklist

## Required Checks

| Check | Command | Expected |
|---|---|---|
| Phase116-118 tests | `conda run -n aic env PYTHONPATH=src python -m pytest tests/unit/scripts/test_fit_moe_wna16_diagnostic_phase116.py tests/unit/scripts/test_build_moe_wna16_experimental_table_phase117.py tests/unit/scripts/test_query_moe_wna16_experimental_table_phase118.py -q` | All pass |
| Script syntax | `conda run -n aic env PYTHONPATH=src python -m py_compile scripts/fit_moe_wna16_diagnostic_phase116.py scripts/build_moe_wna16_experimental_table_phase117.py scripts/query_moe_wna16_experimental_table_phase118.py` | Exit `0` |
| Query hit | `conda run -n aic env PYTHONPATH=src python scripts/query_moe_wna16_experimental_table_phase118.py --key-json docs/iter_gap_investigation/phase118_query_key_tokens128.json --out /private/tmp/phase119_query_tokens128.csv` | Exit `0` |
| Forbidden CSV/header scan | `rg -n "latency_ms|residual|profiled|profiler|nccl|sync|throughput" docs/iter_gap_investigation/phase116_moe_wna16_fit_audit.csv docs/iter_gap_investigation/phase116_moe_wna16_holdout_errors.csv docs/iter_gap_investigation/phase117_moe_wna16_experimental_table.csv docs/iter_gap_investigation/phase118_query_result_tokens128.csv` | No output |
| Default validate | `conda run -n aic python scripts/validate_cb_simulator.py` | PASS |
| Default path scan | `rg -n "PerfDatabase|run_static|IterationLatencyCalculator" scripts/fit_moe_wna16_diagnostic_phase116.py scripts/build_moe_wna16_experimental_table_phase117.py scripts/query_moe_wna16_experimental_table_phase118.py tests/unit/scripts/test_fit_moe_wna16_diagnostic_phase116.py tests/unit/scripts/test_build_moe_wna16_experimental_table_phase117.py tests/unit/scripts/test_query_moe_wna16_experimental_table_phase118.py` | No output |
| Diff check | `git diff --check` | PASS |
| Staged check | `git diff --cached --name-status` | Empty |

## Reviewer Bug Checks

| Risk | What to inspect |
|---|---|
| Hidden interpolation | Query script must only match full exact key |
| Hidden default key | Query JSON must require every key field |
| Unsafe table row | `valid_for_default=false` and `perf_database=false` must be enforced |
| Ambiguous model use | Output must keep `model_use=experimental_table_prototype_only` |
| Formula misuse | Phase116 fit values must not be used by Phase117/118 query path |
| Default AIC pollution | No imports or calls into `PerfDatabase`, `run_static`, or `IterationLatencyCalculator` |

## Stop Conditions

| Condition | Action |
|---|---|
| Any unmeasured token returns a row | Stop and fix query API |
| Any key mismatch returns a row | Stop and fix query API |
| Any default path keyword is introduced into executable code | Stop |
| Any research artifact enters Phase120 dry-run unexpectedly | Stop and fix manifest |
