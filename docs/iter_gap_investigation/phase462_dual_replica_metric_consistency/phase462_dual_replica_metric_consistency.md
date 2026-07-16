# Phase462 双副本度量一致性审计

结论：`measurement_fidelity_bug`。real 是 N512 完整客户端墙钟；官方 sim 是单副本 N384/W128 稳态窗，再把同一 TP-group 吞吐乘 DP2。DP 乘除本身守恒，但窗口、请求数和 output token 定义均不一致，不能把官方 1.200038 全部记为模型误差。

| 口径 | 请求数 | warmup | token 分子 | 时间分母 | tok/s/GPU | error |
|---|---:|---:|---|---|---:|---:|
| real bench | 512 global | 0 | 1024000，含首 token | 完整 wall 1168.950727s | 109.499911 | 1.000000 |
| official sim | 384 per replica | 128 | steady decode steps，不含 prefill sample | replacement plateau | 131.404059 | 1.200038 |
| official trajectory/full wall | 384 per replica | 0 effective | 完成请求 OSL，含首 token | 0 到最后完成 | 128.552759 | 1.173999 |
| real-size symmetric replica/full wall | 256 per replica | 0 | 完成请求 OSL，含首 token | 0 到最后完成 | 127.464056 | 1.164056 |

双副本组装恒等式：TP-group `525.616236` tok/s x DP2 = global `1051.232472` tok/s；再除 8 GPU 得 `131.404059` tok/s/GPU。这里没有额外 2x bug，问题在组装前的单副本测量定义。

首 token 口径只造成 0.0500%，不是主体。窗口从稳态改为同轨迹完整墙钟后的变化才是主项；N256 是 real N512 在“两个完全对称副本”假设下的控制值，不代表真实 DP 路由构成。

## 修复预注册

| 项 | 冻结方案 | 红绿门 |
|---|---|---|
| 验收度量 | 六场景统一增加显式 `full_closed_loop` 计分口径：总完成 OSL token / 首次提交到最后完成的墙钟 | 当前六点先复现旧值；新口径逐点与 bench 的 N/C/warmup 对等 |
| 生产默认 | 保留 steady-state 作为无有限请求窗的容量预测，不拿它直接和 full-wall bench 判误差 | 默认 AIC 输出在本修复前后逐字节不变 |
| DP 组装 | full-wall 验收必须按全局 N/C 跑真实多副本组装；bt65536 未过构成门前不得用单副本复制冒充实测 DP | DP 全局完成数、每 rank 完成数、全局 wall 三项闭合 |
| token 分子 | 验收统一计 OSL，包含 prefill 产生的首 token | OSL=1 与 OSL>1 定向测试 |

本步只预注册，不实施；A/B 合并判卷前不改 runtime、PerfDB 或 gate。

永久脚注：Step 3d 的 65.02%/34.98% 来自固定切换顺序。反向切换不交换，故只能写作 `path_dependent_not_unique_causal_fraction`，不能当成唯一因果比例。
