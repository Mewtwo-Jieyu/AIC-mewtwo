# Phase466 v3 rank-timing contract

结论：本阶段只采三个真实场景的 rank-local iteration timing。成功状态是
`DIAGNOSTIC_COMPLETE`，不是模型归因 `PASS`。

| 项 | 冻结口径 |
|---|---|
| branch | `feature/kimi-vllm019-cb-sim-post-baseline` |
| analyzer schema | `phase466_rank_timing_v3` |
| exit review schema | `phase466_exit_review_v2`; old input fails closed |
| overhead gate | 6 组 counterbalanced OFF/ON；90% paired log-ratio CI 完全位于 `[0.98, 1.02]` |
| gate failure | 立即写 `STOPPED_BEFORE_FORMAL`；不运行 N512 |
| formal scenarios | `DP2-32k3k`、`TP8-bt65536`、`DP2-bt65536`，各一次真实采集 |
| identity | source commit、四个工具、vLLM source、model revision、config/tokenizer hashes、prompt token IDs digest |
| output | 每 rank progress window、elapsed/token、prefill/decode tokens、request counts、preemptions |
| forbidden | simulator runtime/PerfDatabase 改动、伪 DP2-bt65536 rank rows、route selection、自动重跑 |
| readiness | `diagnostic_only=true`; `valid_for_default=false`; `perf_database=false`; `No-Go` |

## Scenario judgement

输入是 `scenario -> candidate -> judgement`。一个场景只有在恰好一个候选为 `PASS`、其余候选全部为
`DISPROVED` 时才选择路线；任何 `INCONCLUSIVE` 都保留该场景为 `INCONCLUSIVE`。只有三个场景选择同一路线时
才写 `shared_route`。

本轮 supervisor 不调用该路线选择器。rank timing 只为后续机制评审提供证据。

## Stop rules

| 错误 | 行为 |
|---|---|
| benchmark 命令失败且 cleanup 完整 | 记录失败；按预注册顺序继续 |
| identity、artifact、cleanup、residue 错误 | 停止全部运行 |
| model/tokenizer/vLLM/tool hash 前后漂移 | 停止全部运行 |
| overhead 非 `PASS` | 停止于 N128 gate |

远端必须使用全新 `phase466_v3_rank_timing_<postbaseline_sha>_<timestamp>` artifact root，后台 supervisor
保留 heartbeat，结束后 GPU/process residue 必须为空。
