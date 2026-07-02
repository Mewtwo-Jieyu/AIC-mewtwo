# Phase397r Prefix Control

Phase397r did not close the attention line.

| Check | Result |
|---|---|
| Prefix cache | disabled, max hit 0.000000% |
| KV usage | max 99.900000% |
| Profile decode batch | 67 |
| Attention | 16.141000 ms/iter, 0.264607 ms/layer |
| Phase397q b64 graph | 0.243456 ms, ratio 1.069005x |
| Phase397q b128 graph | 0.475328 ms, ratio 1.826386x |
| Verdict | prefix_off_control_inconclusive_due_kv_capacity_batch_drop |

The control did turn prefix caching off: `serve.log` reports 0.0% prefix hits.
But the no-prefix 8k workload filled KV cache and the profiled decode batch fell to 67.
That means this run tested an effective b64-like independent-prefix shape, not the planned b128 shape.

So the current evidence says: do not rewrite `generation_mla_perf.txt`, do not close the attention line, and do not change runtime or gates from this control.

Default AIC remains No-Go.
