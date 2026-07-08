# Phase447 scheduler divergence audit

结论: 真实 8k2k 指纹的主构成差异来自 running 调度顺序。vLLM v1 是按 running 队列顺序消费 token budget,当前 cb_sim 旧逻辑先调完所有 decode 再给 prefill,因此 sim 查到 batch 56-64,而真实 mixed 主体是 bucket=8000 且 decode_batch 41-44。

| 审计项 | vLLM 0.19.0 证据 | cb_sim 现状 | 处理 |
|---|---|---|---|
| DP 请求路由 | `DPLBAsyncMPClient.get_core_engine_for_request`: stats 每 100ms 刷新,按 `waiting * 4 + running` 最小 score 选 engine,并本地增加 waiting 计数。源码: `/Users/mewtwo/2026/work/codebase/vllm-0.19.0/vllm/v1/engine/core_client.py:1263-1274`, `:1337-1360` | `VLLMBackend._run_agg_cb_sim` 只跑一个 replica,把 global batch 直接 `ceil(b/dp)`。源码: `src/aiconfigurator/sdk/backends/vllm_backend.py:708-727` | 本 phase 不先造经验相位。先用构成指纹门判定 budget 顺序修复能否让真实构成自然命中。 |
| 闭环补位 | vLLM 请求由 client 持续 `add_request_async`,每次按当前 score 路由。源码: `core_client.py:1283-1298` | `CBSimulator.run` 是单 replica 闭环,完成后把 replacement 直接 append 到同一 waiting 队列。源码: `src/aiconfigurator/sdk/backends/cb_simulator/simulator.py:213-221` | 若指纹门仍缺 phase asymmetry,下一步必须建多 engine global admission,不能手工 offset。 |
| running 打包顺序 | vLLM scheduler 明说没有独立 decode/prefill phase,每个 request 只追 `num_computed_tokens` 到 `num_tokens_with_spec`;先遍历 running,再 waiting。源码: `vllm/v1/core/sched/scheduler.py:348-358`, `:383-518`, `:563-806` | 旧 cb_sim 先把所有 DECODING 请求排完,再继续 PREFILLING,破坏 running 队列顺序。源码修前: `src/aiconfigurator/sdk/backends/cb_simulator/scheduler.py:164-199` | 已修: running 单 pass 队列顺序调度。红测 `test_running_order_interleaves_prefill_before_later_decodes` 覆盖。 |
| DP 每步同步 | `dp_utils._post_process_dp_padding` 在 should_dp_pad 时 pad 到 DP 最大 token 数;`_synchronize_dp_ranks` 在 cudagraph/ubatch 下触发。源码: `/Users/mewtwo/2026/work/codebase/vllm-0.19.0/vllm/v1/worker/dp_utils.py:78-90`, `:148-160` | Phase446 已证明实测序列用 max 耦合可重建 wall,但 cb_sim 当前没有多 replica phase 序列可耦合。 | Step3 指纹不过则不合入 runtime lockstep。 |

## Repair boundary

本轮已落的结构修复只改 running 调度顺序。DP route/closed-loop/lockstep 必须等 Step3 指纹门确认;如果 sim 序列仍不能复现真实 `mixed_prefill+decode` / `mixed_prefill+mixed_prefill` 相位分布,继续改 runtime 就是把等待写成系数,不合入。
