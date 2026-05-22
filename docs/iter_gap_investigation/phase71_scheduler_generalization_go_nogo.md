# Phase 71 Scheduler Generalization Go/No-Go

## Conclusion

Phase71 is No-Go for generalizing the current vLLM-like scheduler generator. The new `32k1k_b16` holdout was captured cleanly, but the Phase70 generator rejects it before compare. This is the right failure: the current generator only covers the first-evidence long-ISL mixed shape and the Phase69 short-ISL holdout.

| gate | result |
|---|---|
| real vLLM holdout captured | yes |
| descriptor rows complete | yes |
| current generator directly matches | no |
| strict compare CSV generated | no |
| fallback row generated | no |
| default `cb_sim` changed | no |

## What Changed In The Shape

`32k1k_b16` is not the same scheduler pattern as the existing two evidence cases.

| scenario | observed scheduler shape |
|---|---|
| `10k2k_b32` | chunked prefill plus mixed row, including `240+1 / NONE:248` |
| `3k3k_b128` | short-ISL leader/follower split, including `496+1 / PIECEWISE:512` |
| `32k1k_b16` | four prefill chunks per DP, then pure decode; no mixed row |

The key Phase71 rows are:

| DP | steps | descriptor |
|---:|---|---|
| 0 | `0..2` | `prefill 8192+0 / NONE:8192` |
| 0 | `3` | `prefill 7536+0 / NONE:7536` |
| 0 | `4..1002` | `pure_decode 0+8 / FULL:8` |
| 1 | `0..2` | `prefill 8192+0 / NONE:8192` |
| 1 | `3` | `prefill 7536+0 / NONE:7536` |
| 1 | `4` | `pure_decode 0+8 / NONE:8` |
| 1 | `5..1002` | `pure_decode 0+8 / FULL:8` |

## Generator Attempt

| item | result |
|---|---|
| command | `diagnose_cb_iter_latency.py --isl 32000 --osl 1000 --concurrency 16 --tp 4 --dp 2 --moe-tp 1 --moe-ep 8 --max-num-batched-tokens 8192 --experimental-vllm-like-scheduler-descriptor` |
| result | fail-fast |
| error | `ValueError: leader remainder plus follower suffix chunks exceed token budget` |
| output CSV | not generated |

This confirms Phase72 should design a third scheduler rule from the captured rows before writing code. It should not patch around the failure by using phase ordinal, intersection-only matching, or manual row edits.

## Decision

| condition | decision |
|---|---|
| third holdout artifact exists | yes, keep it as reference evidence |
| current generator supports it | no |
| general scheduler rule proven | no |
| Phase72 allowed action | write the `32k1k_b16` rule design first |
| latency modeling | still forbidden |

## Boundary

Phase71 only validates scheduler input semantics. It does not create a latency model, does not write a perf table, and does not change `PerfDatabase`, `run_static`, or `IterationLatencyCalculator`.

## Phase72 Link

Phase72 is the design checkpoint for the third scheduler regime. It may define a narrow `32k1k_b16` descriptor rule, but it must not change code or default latency.
