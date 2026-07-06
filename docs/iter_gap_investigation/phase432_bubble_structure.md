# Phase432 Bubble Structure

Phase432 is report-only. It reads Phase429 traces and does not change runtime, PerfDatabase, or gate.

## Verdict

- verdict: `execution_overhead_structure_unresolved`
- selected hypothesis: `none`
- Phase433 target: `gpu_nsys_cpu_side_trace_before_modeling`

## Gap Timeline

| window | steps | prefill | decode | batch mean | internal gap ms/step | wall bubble ms/step | a2a-adjacent gap ms/step | a2a gap segments/step | ep kernels/step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w0_prefill | 35 | 35 | 0 | 17.028571 | 435.380138 | 435.600149 | 0.698560 | 354.775000 | 177.457143 |
| w1_decode_c16 | 35 | 2 | 33 | 7.942857 | 1.665563 | 1.676258 | 0.742163 | 353.632143 | 176.842857 |
| w2_decode_c64 | 35 | 0 | 35 | 36.000000 | 1.510698 | 1.528922 | 0.828128 | 360.507143 | 180.275000 |
| w3_decode_c128 | 35 | 0 | 35 | 52.000000 | 1.556309 | 1.586515 | 0.845829 | 360.696429 | 180.353571 |

## Hypothesis Consistency

| hypothesis | parameter ms | prefill err | decode const err | ep excess err | gate |
|---|---:|---:|---:|---:|---|
| H1_per_a2a_call | 0.002179 | 0.998158 | 0.938842 | 1.000124 | failed |
| H2_per_layer | 1.833803 | 0.732974 | 7.563552 | 1.000000 | failed |
| H3_per_decode_request | 0.000280 | 0.999989 | 0.999292 | 0.995119 | failed |

## 32k3k Cross-Check

| hypothesis | target ms/step | predicted ms/step | error ratio |
|---|---:|---:|---:|
| H1_per_a2a_call | 2900.316853 | 0.758876 | 0.999738 |
| H2_per_layer | 2900.316853 | 110.028177 | 0.962063 |
| H3_per_decode_request | 2900.316853 | 0.001808 | 0.999999 |

## Boundary

- A hypothesis only qualifies if the same parameter explains prefill gap, decode constant gap, and ep_a2a serving excess within 20%.
- No hypothesis passed all three checks, so Phase432 does not authorize adding a runtime overhead term.
- GPU/SSH: not used in Phase432.
- Runtime/PerfDatabase/gate: not modified.
- Default AIC: No-Go.
