# Phase 74 Scheduler Descriptor Review Checklist

## Review Scope

Phase74 is a review packet for scheduler descriptor coverage. It is not a latency model review.

| item | required reviewer check |
|---|---|
| code behavior | only descriptor generator/docstring changed in this phase |
| default path | no new `PerfDatabase`, `run_static`, or `IterationLatencyCalculator` integration |
| output fields | no latency, residual, profiler, NCCL, sync, throughput, or perf-table columns |
| join key | strict `(alignment_key, dp_rank)` only |
| unsupported shapes | fail-fast, no fallback |
| evidence cases | `10k2k`, `3k3k`, `32k1k` all strict-match |

## Commands

| check | command |
|---|---|
| descriptor tests | `conda run -n aic env PYTHONPATH=src python -m pytest tests/unit/test_forward_descriptor_vllm_like_scheduler.py tests/unit/test_forward_descriptor_scheduler_alignment.py tests/unit/scripts/test_analyze_vllm_scheduler_descriptor_phase62.py -q` |
| syntax | `conda run -n aic env PYTHONPATH=src python -m py_compile src/aiconfigurator/sdk/backends/cb_simulator/forward_descriptor.py scripts/diagnose_cb_iter_latency.py` |
| default validate | `conda run -n aic python scripts/validate_cb_simulator.py` |
| diff check | `git diff --check` |
| staged check | `git diff --cached --name-status` |

## Expected Results

| check | expected |
|---|---|
| tests | `27 passed` |
| default validate | throughput `1.50x`, multi-config `1.47x`, TTFT `1.79x` |
| `10k2k` compare | `4003` rows, all same flags `1`, all deltas `0` |
| `3k3k` compare | `6002` rows, all same flags `1`, all deltas `0` |
| `32k1k` compare | `2006` rows, all same flags `1`, all deltas `0` |
| staged area | empty unless a later staging phase explicitly selects files |

## Stop Conditions

| finding | action |
|---|---|
| default latency path changed | stop |
| compare uses phase ordinal or intersection-only rows | stop |
| descriptor schema adds timing/perf fields | stop |
| new scenario lacks real vLLM descriptor artifact | stop |
| generator guesses DP split | stop |
