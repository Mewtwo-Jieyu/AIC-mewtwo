# Phase400 clean recollect

Phase400 recollected four 0.19-real 8-card scenarios with prefix caching disabled. All four benchmark runs completed 128/128 requests and the cleanup evidence is empty.

| scenario | clean output tok/s/gpu | running max | GPU KV tokens | prefix hit max % | no-capacity sim ratio | naive-capacity sim ratio |
|---|---:|---:|---:|---:|---:|---:|
| K2.5-tp8ep8-8k2k | 137.386464 | 67 | 546160 | 0.000000 | 2.066574 | 1.617287 |
| K2.5-tp8ep8-32k3k | 50.099161 | 14 | 461200 | 0.000000 | 2.164443 | 11.163967 |
| K2.5-tp4ep8dp2-8k2k | 151.681886 | missing | 458096 |  | 2.537459 | 2.537459 |
| K2.5-tp4ep8dp2-32k3k | 50.064056 | missing | 320800 |  | 3.490670 | 19.846480 |

Key findings:

- `prefix hit is 0.0%` for TP8 logs; DP2 logs do not emit periodic `Running:` stats in this collection, so their running batch is recorded as missing instead of inferred.
- The command did not pass `--num-gpu-blocks-override`, but vLLM still logged worker-level `num_gpu_blocks_override=512` in every scenario. Phase401 must treat this as an observed vLLM semantic, not as a CLI knob used by the recollect.
- Compared with the clean prefix-off real baseline, the no-capacity simulator path still over-predicts throughput, while the Phase398 naive capacity path under-predicts long-context throughput. Capacity alone is not the final fix; preemption and capacity semantics need a targeted Phase401 change.
- No simulator runtime, PerfDatabase data, or acceptance gate changed in Phase400. Default AIC remains No-Go.

Next: Phase401 should use these clean rows to fix capacity/preemption semantics, then rerun the 4-scenario gate.
