# Phase409 Iter Trace

Phase409 是测量-only。它只跑一个 DP2 32k3k GPU 点，开启 vLLM iteration-detail logging 与 cudagraph_metrics，不改 runtime、PerfDatabase 或 gate。

## Verdict

- verdict: `metrics_lower_bound_below_needed_per_step_padded_absent`
- direct penalty source: `metrics_prompt_generation_delta_lower_bound`
- measured token penalty: `1.186159`
- needed penalty from Phase407 uncoupled/real: `1.887133`
- penalty gate: `failed`
- trace fidelity gate: `passed`

## Trace Summary

| scenario | active intervals | trace tok/s/gpu | bench tok/s/gpu | fidelity error | co-prefill share | cudagraph rows | Phase410 target |
|---|---:|---:|---:|---:|---:|---:|---|
| K2.5-tp4ep8dp2-32k3k | 435 | 45.898404 | 45.741879 | 0.342192% | 0.124138 | 0 | collect_per_step_padded_tokens_or_recheck_missing_dp_cost |

## Interpretation

- metrics 的 output tok/s/gpu 与 bench 只差上表的 fidelity error，说明这次采集主窗口可信。
- `cudagraph_rows_found=0`，本次 vLLM 没有落出可解析的 per-step padded token 行；因此 measured token penalty 是 metrics 2 秒采样 delta 的真实相位下界，不是逐 step padded 真值。
- 这个下界是 `1.186159`，明显低于 Phase407 所需 `1.887133`；所以当前证据不能支持“asymmetry + max 已精确解释 1.887”。
- Phase410 不能直接落 runtime 耦合；先要拿到 per-step padded token，或转查缺失 DP cost。

## Boundary

- GPU/SSH: used for measurement only.
- Runtime/PerfDatabase/gate: not modified.
- Phase405 penalty: not read.
- Default AIC: No-Go.
