# Phase462 bt65536 度量与构成取证合并判卷

结论：A 确认 `measurement_fidelity_bug`，官方 `1.200038` 中约 `0.036` 来自验收度量不一致；B 的 logging-only 协议连续三次未过 `<=2%` 开销门，按预注册规则在 N512 前停线。当前没有可采信的构成级 GPU 数据，不能判定 `8.54x` spread、sim undercharge 或 serving-state 行候选。

| 项 | 结果 | 后续约束 |
|---|---|---|
| 双副本度量 | real 使用 N512 完整客户端墙钟；官方 sim 使用单副本 N384/W128 稳态窗后乘 DP2 | 预注册 `full_closed_loop` 验收口径；本步不实施 |
| DP 算术 | TP-group 吞吐乘 DP2、再除 8 GPU，算术守恒 | 没有额外 2x bug |
| 可解释误差 | `1.200038 -> 1.164056`，约 `0.036` 来自窗口、请求数和 token 定义差异 | 其余保留为真实模型误差，不再外推归因 |
| 构成取证 | 三次有效 off/on 对照均未过开销门 | N512 TP8/DP2 均未运行，禁止构成判卷 |
| 默认路径 | runtime、PerfDB、gate 均未改 | Default AIC 保持 No-Go；Step 4 顺延 |

## 开销硬门

| 版本 | 观测位置 | off tok/s | on tok/s | 绝对差 | 判定 |
|---|---|---:|---:|---:|---|
| v2 | schedule 路径完整请求构成 | 716.027911 | 815.188094 | 13.8486% | fail |
| v3 | schedule 路径压缩为首次请求 ID | 812.202193 | 883.606778 | 8.7915% | fail |
| v4 | submit 只记序号/时间，future 完成后解析 `SchedulerOutput` | 802.973991 | 737.095652 | 8.2043% | fail |

v2/v3 的 on 高于 off，v4 的 on 低于 off，方向发生反转。因此不能把 `8.2043%` 解释成纯 logging 开销；单次 DP2 路由轨迹波动和观测扰动已经大于 `2%` 门，当前协议无法证明自身非侵入性。TP8 不可替代该门，因为 B 的目标包含 DP2 路由与 per-rank 构成，换成 TP8 会隐藏待测扰动。

首次远端尝试因 hook 锚到普通 `future`、没有覆盖实际 queue 路径而 fail-fast；修正为 `exec_future` 后才得到上表三组有效对照，首次尝试不计入门值。

## 判卷边界

| 预注册问题 | 本步判定 |
|---|---|
| `8.54x` spread 是否由 scheduled token/chunk 构成解释 | 未判定；N512 未运行 |
| real 实访构成相对 sim 是否 undercharge | 未判定；无 admissible 构成样本 |
| 是否生成构成 scoped serving-state 行候选 | 否；禁止从未过门数据生成候选 |
| 是否继续 Step 4 | 否；等待单独批准新的非侵入式测量设计 |

v4 退出后，vLLM 两个源码文件恢复到原始 SHA256，进程残留与 GPU compute-app 残留均为空。远端固定入口的 ED25519 指纹已核对为 `SHA256:IVjkRgeMAWHa+gA/SVMZBy2yZBg+m04S1mBpzb2Ufrk`；SSH 的 RSA global-hostkeys proof 告警不改变本次已核对的 ED25519 主机身份。

度量细节见 `../phase462_dual_replica_metric_consistency/phase462_dual_replica_metric_consistency.md`；三次门值与恢复证据保存在 `evidence/`。

永久脚注：Step 3d 的 `65.02%/34.98%` 依赖控制变量切换顺序，反向切换不可交换，只能标为 `path_dependent_not_unique_causal_fraction`，不能作为唯一因果比例。
