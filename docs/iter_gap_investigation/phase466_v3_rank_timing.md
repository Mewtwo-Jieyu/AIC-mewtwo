# Phase466 v3.2 rank-local timing contract

结论：本阶段只采三个真实场景的 rank-local iteration timing。成功状态是
`DIAGNOSTIC_COMPLETE`，不是模型归因 `PASS`。

| 项 | 冻结口径 |
|---|---|
| branch | `feature/kimi-vllm019-cb-sim-post-baseline` |
| analyzer schema | `phase466_rank_timing_v4` |
| exit review schema | `phase466_exit_review_v2`; old input fails closed |
| overhead gate | 6 组 counterbalanced OFF/ON；90% paired log-ratio CI 完全位于 `[0.98, 1.02]` |
| gate failure | 立即写 `STOPPED_BEFORE_FORMAL`；不运行 N512 |
| formal scenarios | `DP2-32k3k`、`TP8-bt65536`、`DP2-bt65536`，各一次真实采集 |
| identity | 本地 clean HEAD + 六个执行文件字节哈希；worker vLLM/GPU/prompt/日志传输身份；snapshot revision 或平铺镜像实例指纹 |
| output | 每 rank progress window、elapsed/token、prefill/decode tokens、request counts、preemptions |
| forbidden | simulator runtime/PerfDatabase 改动、伪 DP2-bt65536 rank rows、route selection、自动重跑 |
| readiness | `diagnostic_only=true`; `valid_for_default=false`; `perf_database=false`; `No-Go` |

## Scenario judgement

输入是 `scenario -> candidate -> judgement`。一个场景只有在恰好一个候选为 `PASS`、其余候选全部为
`DISPROVED` 时才选择路线；任何 `INCONCLUSIVE` 都保留该场景为 `INCONCLUSIVE`。只有三个场景选择同一路线时
才写 `shared_route`。

本轮 supervisor 不调用该路线选择器。rank timing 只为后续机制评审提供证据。

## v3.1 flat-mirror identity

当前 Kimi-K2.5 是 64-shard 平铺镜像，不是标准 Hugging Face snapshot。身份分两阶段冻结：本地 coordinator
manifest 证明 clean HEAD 与六个上传文件；worker attestation 绑定实际工具字节、vLLM source/import、GPU、prompt
token cohort、canary prompt cohort 和模型指纹；两者再合成 `phase466_execution_manifest_v4`。

平铺指纹 `phase466_flat_model_fingerprint_v1` 哈希根目录运行小文件和 `.msc/.mv` 原始字节，并按
`model.safetensors.index.json` 精确核对 shard 集合与每个 shard header 的 tensor 集。每个 shard 记录路径、大小、
`mtime_ns` 和 header SHA256，不读取权重数据区。该身份只证明“同一平铺镜像实例”，不能声明为官方 immutable revision。

## v3.2 rank-local transport

`vllm.v1.engine.core` 的 `Iteration(...)` record 由标准库 logging handler 按
`LogRecord.processName` 写入 `rank_logs/rank-<id>.jsonl`。DP2 只接受
`EngineCore_DP0/DP1`，单 DP 只接受 `EngineCore`。stdout 只保留调试日志，不再参与 rank 判定。

每份 rank 文件严格绑定文件名、record rank、process name、稳定 PID、连续 iteration、SHA256、行数和首尾
iteration。OFF 与 ON 使用同一 logging config；OFF 不产生 rank 文件，ON 才启用 vLLM iteration details。
handler、config、`PYTHONPATH` 和 `VLLM_LOGGING_CONFIG_PATH` 一并进入
`phase466_execution_manifest_v4`。

完整 gate 前只允许一次 `N16/C16, ISL128/OSL16, TP4/DP2/EP8` canary。canary 必须得到两份合法 rank 文件并清空
GPU/process residue；失败立即停止，不运行六组 overhead。

## Stop rules

| 错误 | 行为 |
|---|---|
| benchmark 命令失败且 cleanup 完整 | 记录失败；按预注册顺序继续 |
| identity、artifact、cleanup、residue 错误 | 停止全部运行 |
| model/tokenizer/vLLM/tool/prompt 身份前后漂移 | 停止全部运行 |
| overhead 非 `PASS` | 停止于 N128 gate |

远端必须使用全新 `phase466_v3_rank_timing_<postbaseline_sha>_<timestamp>` artifact root，后台 supervisor
保留 heartbeat，结束后 GPU/process residue 必须为空。
