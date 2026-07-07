# Phase438 Serving-State Merge Dry Run

## Verdict

`offline_union_merge_requires_gpu_window_confirmation`.

Step 0 only: no GPU was used and `vllm_serving_state_perf.txt` was not modified.

## Axis Decisions

| phase | category | decode-batch spread % | bucket spread % | decision |
|---|---|---:|---:|---|
| decode | collective_other | nan | nan | keep_2d |
| decode | ep_a2a | nan | nan | keep_2d |
| decode | moe_gemm_or_aux | nan | nan | keep_2d |
| decode | other_cuda | nan | nan | keep_2d |
| mixed_prefill | collective_other | 0.00 | nan | reduce_decode_batch_axis |
| mixed_prefill | ep_a2a | 70.54 | 188.10 | keep_2d |
| mixed_prefill | moe_gemm_or_aux | 56.42 | 166.67 | keep_2d |
| mixed_prefill | other_cuda | 56.54 | 183.64 | keep_2d |

## Merge Conflicts

| count | action |
|---:|---|
| 5 | exact-key candidate conflicts are reported; existing accepted row is kept |

## Hit Audit Dry Run

| metric | count |
|---|---:|
| bucket_below_range | 148 |
| decode_batch_above_range | 169 |
| hit | 219 |
| interpolation_gap | 300 |
| table_missing | 8 |

## Validate Dry Run

| scenario | baseline ratio | candidate ratio | classification |
|---|---:|---:|---|
| K2.5-tp8ep8-8k2k | 1.250 | 1.250 | unchanged |
| K2.5-tp8ep8-32k3k | 1.069 | 1.069 | unchanged |
| K2.5-tp4ep8dp2-8k2k | 2.600 | 2.118 | improved |
| K2.5-tp4ep8dp2-32k3k | 1.255 | 1.255 | unchanged |
| K2.5-tp8ep8-8k2k-bt65536 | 1.120 | 1.120 | unchanged |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 1.249 | 1.252 | unchanged |

## GPU Window Confirmation List

| window | scenario | reason | count | note |
|---|---|---|---:|---|
| W-A | K2.5-tp4ep8dp2-8k2k | 8k mixed prefill decode_batch above range | 165 | steady trigger required; target decode_batch 56-64 |

## Boundary

- Inner-only query semantics stay unchanged.
- No clamp, extrapolation, or silent overwrite is introduced.
- GPU collection must wait for explicit window confirmation.
- Default AIC remains No-Go.
