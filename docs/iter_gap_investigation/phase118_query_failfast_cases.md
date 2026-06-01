# Phase118 Query Fail-Fast Cases

## Decision

Phase118 query is full exact-key only. It does not interpolate, extrapolate, or supply default key values.

| Case | Input change | Result |
|---|---|---|
| Measured token | `tokens_actual=128` with full key | Query succeeds |
| Missing token | `tokens_actual=256` with otherwise matching full key | Fails with `no exact row for full key` |
| Mismatched config | `config_sha256=bad` with `tokens_actual=128` | Fails with `mismatched fields: config_sha256` |

## Smoke Commands

| Check | Command | Expected |
|---|---|---|
| Hit | `conda run -n aic env PYTHONPATH=src python scripts/query_moe_wna16_experimental_table_phase118.py --key-json docs/iter_gap_investigation/phase118_query_key_tokens128.json --out docs/iter_gap_investigation/phase118_query_result_tokens128.csv` | Exit `0` |
| Missing token | `conda run -n aic env PYTHONPATH=src python scripts/query_moe_wna16_experimental_table_phase118.py --key-json /private/tmp/phase118_query_key_tokens256.json` | Exit non-zero |
| Mismatched config | `conda run -n aic env PYTHONPATH=src python scripts/query_moe_wna16_experimental_table_phase118.py --key-json /private/tmp/phase118_query_key_bad_sha.json` | Exit non-zero |

## Boundary

| Rule | Status |
|---|---|
| Interpolation | Forbidden |
| Extrapolation | Forbidden |
| Default key fill | Forbidden |
| `PerfDatabase` | Not used |
| `run_static` / `IterationLatencyCalculator` | Not used |
