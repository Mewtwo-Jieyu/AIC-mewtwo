# Phase466 v3.3 idle-safe rank-local timing contract

结论：本阶段只采三个真实场景的 rank-local iteration timing。成功状态是
`DIAGNOSTIC_COMPLETE`，不是模型归因 `PASS`。

| 项 | 冻结口径 |
|---|---|
| branch | `feature/kimi-vllm019-cb-sim-post-baseline` |
| analyzer schema | `phase466_rank_timing_v5` |
| canary schema | `phase466_rank_local_canary_v2` |
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

v3.2 canary 已运行并保持 `FAILED`，artifact 为
`/mnt/shared-storage-user/zhaojieyu/backup/aic/phase466_v32_rank_local_canary_2221f30e_20260720T110252Z`。
两份 rank-local 文件、PID 和连续 iteration 均已验证，失败来自旧契约把合法的零 token、零请求、`0.00 ms`
idle iteration 判成非法，并在 benchmark 返回后延迟 2 秒才记录测量终点。该 artifact 缺少完整成功产物，禁止事后改写为
`PASS`。

## v3.3 idle-safe measurement

| 项 | 冻结口径 |
|---|---|
| work iteration | scheduled tokens 大于 0，`elapsed_ms` 必须有限且大于 0 |
| idle iteration | scheduled tokens 和请求数均为 0，`elapsed_ms` 必须有限且大于等于 0 |
| measurement cutoff | benchmark 命令返回后立即按 rank 记录；metrics 和 cleanup 在 cutoff 后执行 |
| raw evidence | iteration CSV 保留 idle 行、连续序列、offset、文件哈希和总 elapsed 审计字段 |
| work statistics | `work_iteration_count`、`work_elapsed_ms_sum`、工作时延分位数和 coverage 只使用正 token 行 |
| canary tail | cutoff 可以早于文件末行；cutoff 后只允许 idle，每个 rank 的测量窗至少一条 work iteration |

rank-log wire format、`phase466_rank_log_identity_v1` 和 `phase466_execution_manifest_v4` 保持不变。
v3.3 已完成本地实现与测试，第二次 canary 待单独批准；本轮不自动重跑，也不进入完整 overhead gate。

## Stop rules

| 错误 | 行为 |
|---|---|
| benchmark 命令失败且 cleanup 完整 | 记录失败；按预注册顺序继续 |
| identity、artifact、cleanup、residue 错误 | 停止全部运行 |
| model/tokenizer/vLLM/tool/prompt 身份前后漂移 | 停止全部运行 |
| overhead 非 `PASS` | 停止于 N128 gate |

后续获批的远端运行必须使用全新 `phase466_v3_rank_timing_<postbaseline_sha>_<timestamp>` artifact root，后台 supervisor
保留 heartbeat，结束后 GPU/process residue 必须为空。
