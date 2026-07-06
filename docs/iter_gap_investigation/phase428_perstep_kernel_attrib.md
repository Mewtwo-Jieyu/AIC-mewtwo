# Phase428 Per-Step Kernel Attribution

Phase428 只离线榨干 Phase427 trace；不改 runtime、PerfDatabase 或 gate。

## Verdict

- mechanism: `phase427_trace_decode_only_prefill_recollect_required`
- prefill: `prefill_steps_not_captured`
- decode: `decode_batch_span_too_narrow_for_slope`
- reconstruction gate: `blocked_prefill_not_captured`
- Phase429 runtime target: `recollect_true_mixed_prefill_window_before_runtime_change`
- Phase429 PerfDB target: `recollect_decode_batch_sweep_then_reprofile_prefill`

## Step Coverage

| prefill steps | decode steps | steady penalty | prefill share | decode gap share |
|---:|---:|---:|---:|---:|
| 0 | 70 | 1.516977 | 0.595460 | 0.357492 |

## Decode Slope

| category | real mean ms/rank | real slope ms/request |
|---|---:|---:|
| moe_gemm_or_aux | 9.705091 | -0.932917 |
| mla_attention | 11.794939 | -0.151564 |
| ep_a2a | 7.418812 | -0.023147 |
| dense_gemm | 4.986231 | 0.003089 |
| memcpy_memset | 0.164780 | -0.001737 |
| other_cuda | 2.305595 | -0.001029 |
| tp_or_dp_allreduce | 0.824731 | 0.000611 |

## Boundary

- Phase427 的两个 profiler window 均未捕获 `ctx_tokens > 0` 的 prefill step；prefill 欠账不能用这批 trace 定案。
- Phase427 decode batch span 很窄时，只能看 kernel mix，不能给 +16.5ms/batch 斜率定案。
- GPU/SSH: not used in Phase428.
- Runtime/PerfDatabase/gate: not modified.
- Phase405 penalty: not read.
- Default AIC: No-Go.
