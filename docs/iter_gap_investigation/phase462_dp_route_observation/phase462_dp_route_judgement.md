# Phase462 DP Step 3b 路由取证

结论：机械判卷为 `H-mixed`。判卷顺序固定为 route 后 drain；本报告只给机制证据，不修改 runtime、PerfDB 或 gate。`Default AIC=No-Go`。

| 指标 | rank 0 | rank 1 |
|---|---:|---:|
| route 请求数 | 257 | 255 |
| admit step gap 分布 | 0:203/256;1:31/256;11:1/256;32:1/256;94:1/256;139:1/256;1860:1/256;1906:1/256;1967:1/256;1989:1/256;1999:5/128;2000:1/64 | 0:215/254;1:9/127;2:1/254;10:1/254;32:1/254;134:1/254;1866:1/254;1968:1/254;1990:1/254;2000:7/127 |
| route→receive 中位延迟 (ms) | 3082.740253 | 2275.193014 |
| receive→admit 中位延迟 (ms) | 188745.749190 | 189004.432574 |

| 判据 | 结果 |
|---|---|
| route 严格均衡 | false |
| route 逐请求严格交替 | false |
| route 不对称（总量或聚簇） | true |
| drain cadence 逐项同分布 | false |
| 最大连续同 rank 路由 | 14 |
| route rank 切换次数 | 367 |
| score 并列路由次数 | 36 |
| diagnostic_only | true |
| valid_for_default | false |
