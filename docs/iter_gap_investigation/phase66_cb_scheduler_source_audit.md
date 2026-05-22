# Phase 66 cb_sim Scheduler Source Audit

## Conclusion

Phase65 的 mismatch 来自源语义，不是 compare bug。当前 cb_sim aligned descriptor 直接包装 `CBIterationTraceRow`，而 `CBIterationTraceRow` 是单个 cb_sim scheduler 的全局 trace，不是 vLLM EngineCore 的 DP-local scheduler row。

| 项 | 当前 cb_sim 行为 | Phase62 vLLM 行为 |
|---|---|---|
| step id | `enumerate(rows)` 生成 | EngineCore DP 内 step |
| DP split | `dp_rank` 只写入 key，不改变 row | 每个 DP 有独立 row 序列 |
| prompt split | 单 scheduler 继续大块 chunked prefill | prefix/cache 语义下出现 leader + follower suffix |
| forward shape | `prefill_tokens + decode_batch_size` | runtime padded forward tokens |
| graph mode | `AIC_UNSET` | concrete `NONE` / `FULL` |

## Why `8191+1 / AIC_UNSET:8192` Appears

当前 `collect_cb_iteration_trace(...)` 调用 `CBScheduler.schedule(...)` 后，把 `schedule.total_prefill_tokens`、`len(schedule.decode_reqs)` 和 `row.prefill_tokens + row.decode_batch_size` 写入 descriptor。对于 `10k2k_b32 bt8192`，cb_sim 的全局 scheduler 在 step2 仍有大块 prefill，同时已有一个 decode request，因此输出 `8191+1 / AIC_UNSET:8192`。

vLLM Phase62 的 `engine_dp:1:step:2` 不是这个语义。它是 DP1 的 EngineCore row：`240` context tokens、`1` decode token、`15+1` requests，runtime forward padded to `248`，graph mode `NONE`。

## Boundary

The existing cb_sim scheduler remains valid for latency simulation. Phase66 does not modify it. The new generator is experimental-only and only exists to produce a vLLM-like descriptor sequence for shape semantics comparison.

## No Default Path Change

No `PerfDatabase`, `run_static`, or `IterationLatencyCalculator` integration is added. The generator emits descriptors only and is not a latency model.
