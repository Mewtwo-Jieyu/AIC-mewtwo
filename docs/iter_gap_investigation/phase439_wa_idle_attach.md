# Phase439 W-A idle attach result

## 结论

W-A 仍不能用 torch profiler 收到可入库的稳态窗口。空载 attach 解决了低负载路径：Gate1 低负载冒烟成功产出 8 rank trace；但正式 W-A 在 running 达到 56/engine 后 5 秒左右 EngineCore fatal，`stop_profile` 时端口已拒连。

因此本轮按护栏停在 report-only：不提取、不合并、不写 `vllm_serving_state_perf.txt`、不跑 A/B。Phase438 的 GPU profiler 方案在 vLLM serving 高负载集成层仍不可用，下一步应转 B 方案：CUDA event 计时器，不走 CUPTI。

## Gate 结果

| 门 | 结果 | 关键事实 | 判定 |
|---|---:|---|---|
| Gate 0: endpoint | 过 | service ready `2026-07-07T14:15:52Z`，profiler config 被 vLLM 接收 | profiler 端点路径可用 |
| Gate 1: smoke | 过 | idle attach + C=4 短请求，`trace_files=8`，`window_check_passed` | 原语与低负载 vLLM attach 可用 |
| Gate 2: W-A | 失败 | running 达到 `{"0":56,"1":56}` 连续 3 次；`14:18:37` EngineCore DP0/DP1 fatal；`14:19:32` stop_profile 端口拒连 | 正式 W-A 无可用 profile |
| 提取/入库 | 跳过 | 本地仅有 Gate1 smoke trace，没有 `prof_wa_idle_attach/` | 不能把 smoke 数据写进 serving-state 表 |
| GPU 收尾 | 过 | `nvidia-smi --query-compute-apps` 为空，目标进程为空 | 无残留 |

## 崩溃签名

关键行均来自 `docs/iter_gap_investigation/phase439_wa_idle_attach/K2.5-tp4ep8dp2-8k2k-wa-idle-attach/` 下的小型 raw：

| 文件 | 行/字段 | 内容 |
|---|---|---|
| `runner.nohup.log` | Gate2 | `steady_record_seconds=60` 后执行 `profile_stop gate=PHASE439_GATE2_WA` |
| `runner.nohup.log` | stop | `curl: (7) Failed to connect to 127.0.0.1 port 20939` |
| `serve.log` | EngineCore | `EngineCore encountered a fatal error` |
| `serve.log` | exception | `RuntimeError: cancelled` |
| `serve.log` | profiler | `Requested callback is not found` |
| `tmp_gate2/bench_result.json` | result | `ok_requests=0`, `failed_requests=192` |

## 解释

Phase439 证伪了“只要空载 attach 就能稳定采 W-A”的假设。低负载 smoke 通过，说明节点、CUPTI、端点和基础 profiler 链路可用；正式负载失败，说明问题仍在 vLLM 高负载 serving 集成层，且失败发生在 profiler 录制期间，不是窗口自检或后处理问题。

继续重试同一路径只会重复触发 EngineCore fatal。下一阶段应按既定 B 方案改成 CUDA event 分类计时器：在 vLLM MoE/EP 关键路径内做轻量计时，不使用 CUPTI attach；再用 Phase429 既有 profiler 分类数据做交叉校准门。

## 本轮不做

| 项 | 原因 |
|---|---|
| 逐 step 提取 | 没有正式 W-A trace |
| serving-state 入库 | 只有 Gate1 smoke trace，口径不匹配 |
| validate A/B | 无候选表，跑 A/B 没有意义 |
| Default AIC 放行 | Gate2 未过，继续 No-Go |
