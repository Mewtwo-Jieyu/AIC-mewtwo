# Phase462 DP Step 3d harness input root audit

结论：预注册的“到达合成主导 0.11”被证伪。官方与 O0 都是 `t=0` 合成闭环；原始差值由每副本观测窗和执行拓扑共同产生。TP8 首拍 9-context 确由 `65536 = 8 x 8000 + 1536` 机械生成；把首拍可见输入改为 1 后，decode cell 差从 7 降到 0，但误差只从 1.256843 到 1.251833，仍未过 1.15。`Default AIC=No-Go`。

## DP 0.11 控制矩阵

| 控制 | 拓扑 | N/W | tok/s/GPU | error | serving-state hit/query |
|---|---|---:|---:|---:|---:|
| single_N384_W128 | single | 384/128 | 131.404059 | 1.200038 | 37/918 |
| single_N192_W64 | single | 192/64 | 123.538026 | 1.128202 | 19/555 |
| multi_N384_W128_O0 | multi | 384/128 | 119.305385 | 1.089548 | 19/537 |
| multi_N768_W128 | multi | 768/128 | 130.120645 | 1.188317 | 33/925 |
| multi_N768_W256_matched | multi | 768/256 | 134.079825 | 1.224474 | 33/925 |

| official -> O0 固定切换路径 | delta error | 路径内占比 |
|---|---:|---:|
| 每副本观测窗 384/128 -> 192/64 | -0.071836 | 65.02% |
| 单副本 -> 双副本全局组装 | -0.038654 | 34.98% |
| 合计 | -0.110490 | 100.00% |

反向核验：O0 只把 N 从 384 归一到 768，已追回原始差值的 89.39%；N/W 都按双副本归一后 error=1.224474，相对官方残差 +0.024436。该非线性说明 O0 的 1.0895 是观测窗口径产物，不能拿来证明 route/drain 改善。上表 65.02%/34.98% 只对当前切换顺序成立，不是顺序无关的唯一因果比例。

到达、闭环、seed、KV 和 PerfDB 输入审计：

| 维度 | 官方 | O0 | 判定 |
|---|---|---|---|
| 到达 | t=0 注入 C64，完成即补位 | t=0 全局注入 C128，完成即补位 | 同为合成闭环，O0 未读取实测到达 |
| seed | 无随机分支 | 无随机分支 | 无可切换 seed |
| KV | physical=8229，每 pool 保留 null block | 相同 | 输入相同 |
| serving-state 数据 | 同一 backend/version/DB | 相同 | hit/query 差是批构成输出，不是输入切换 |
| 观测窗 | 单副本 N384/W128 | 双副本全局 N384/W128 | 每副本仅约 N192/W64，是实际混入项 |

## TP8 首波控制

| 控制 | 首 8 步逐位吻合 real | decode cell 差 | tok/s/GPU | error |
|---|---:|---:|---:|---:|
| baseline | 0.00% | 7 | 155.869852 | 1.256843 |
| first_context_only | 12.50% | 0 | 155.248500 | 1.251833 |
| real_initial_ramp | 100.00% | 0 | 155.248500 | 1.251833 |

`first_context_only` 只在第一次 schedule 暂时暴露 1 个请求；`real_initial_ramp` 再复现下一拍无新 context。两者都不改 chunk、token budget、KV、成本模型或默认 runtime。前 8 步构成可恢复，但吞吐几乎不动，因此首波错位只解释 sim-only 精确 cell，不解释 1.257 的主体。

## 输入契约评审

| artifact | 显式 start timestamp | 重建首拍 t=0 请求 | wall 重建误差 |
|---|---:|---:|---:|
| dp2_bt65536 | False | 128 | 0.000402% |
| tp8_bt65536 | False | 128 | 0.001661% |

裁决：`engine_visibility_boundary_not_observed_keep_official_protocol`。`bench_records` 只有 latency；按冻结的 128-worker 闭环语义可重建客户端提交时刻，且 wall 误差很小，但首批仍是 128 个 t=0。TP8 的 EngineCore 首拍 1-context 发生在 client 提交之后，现有 bt65536 artifact 没有逐请求 receive/visible timestamp；拿 scheduler 聚合输出反灌会形成 target leakage。

因此本步不改计分板协议：继续并列保存官方合成口径与诊断输入控制，暂不把 `bench_records` 宣称为 EngineCore 到达 oracle。生产合成器保真度仍是独立工程项。DP real 8.54x spread 挂起，Step 4 顺延。

本步全离线、report-only；未改 runtime、PerfDB、gate，未使用 GPU。
