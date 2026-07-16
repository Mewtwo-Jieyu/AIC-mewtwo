# Phase462 Step 2c-8 二次门修正案评审包

结论：`amendment_review_ready_for_approval`。本文件只是审批候选，未修改冻结门。

## 修正案全文

仅替换归因门中的 `admit 对齐逐步全等` 条款；oracle 语义门、抢占总数 `10±2`、重复 victim `=0` 均保持不变。

链环默认仍须 admit 对齐逐步全等。唯一例外是同一链环同时满足以下五条：

1. real 与 sim 均有同 trigger、同 victim 的匹配抢占，real 事件不可缺席；
2. 首个 admit 对齐不等区间跨过同一个绝对抢占边界，且区间宽度等于当时累计根滞后；
3. 抢占时 token 差为正，且不大于当时累计根滞后；
4. decode 一步一 token，释放滞后减去 resume 滞后必须严格等于 token 差；
5. 非零对齐偏移只可由已通过前四条的同一匹配抢占事件引起，其他位置仍严格全等。

五条全部通过才豁免该请求从首差开始的剩余链环；下一请求恢复严格比较，除非它独立通过同一五条。任一条失败，归因门失败。判据不含 request ID、step 白名单或可调常数。根仍必须落在 schedule 1 的 N128 初始 workload。

同一门禁止第三次修正。若再次申请修改，直接触发止损：补更长 real 窗口核验门值，或按实现缺口处理。

## 正例干跑

| 链 | 原严格门 | 候选门 | 路径 | 例外请求 |
|---|---:|---:|---|---|
| request_13_control | PASS | PASS | strict | 无 |
| request_27 | PASS | PASS | strict | 无 |
| request_124 | FAIL | PASS | compound_exception | 69,82,96,110 |

## 五条款逐项结果

| request | 抢占步 | trigger/victim | 首差 real/sim | token差/累计滞后 | lag路径 new→resume→release | i | ii | iii | iv | v |
|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 69 | 5825 | 57/69 | 5825/5826 | 1/1 | 1→0→1 | PASS | PASS | PASS | PASS | PASS |
| 82 | 6955 | 78/82 | 6955/6956 | 1/1 | 1→1→2 | PASS | PASS | PASS | PASS | PASS |
| 96 | 8176 | 94/96 | 8175/8177 | 2/2 | 2→0→2 | PASS | PASS | PASS | PASS | PASS |
| 110 | 9396 | 109/110 | 9395/9397 | 2/2 | 2→0→2 | PASS | PASS | PASS | PASS | PASS |

## 负例捕虫

| 注入负例 | 候选门结果 | 命中检查 |
|---|---:|---|
| sim_only_preemption | FAIL | i,ii,v |
| token_delta_above_cumulative_lag | FAIL | iii |
| offset_jump_without_verified_event | FAIL | v |
| root_not_schedule_one_boundary | FAIL | root_not_schedule_one_initial_workload |

## 过拟合风险

这是同一归因门的第二次修正，确有向当前 sim 输出雕刻的风险。护栏只有三项：real 匹配抢占是强制前提；五条均由事件流测量且无自由参数；oracle 自抢占 `=0` 的语义门绝对不动。第三次修正被禁止。

## 边界

- 本步未修改 gate、runtime 或 PerfDB；未运行 `--ab`。
- `six_point_ab_allowed=false`，Default AIC 维持 No-Go。
- 首个可见集合分叉：schedule `1`。
- 证据 SHA256：`524d44ca4df36835e3d0a041b6591ba113d329bac8ca61f9b037dfc3401e8af9`。
- 下一动作：等待审批；批准后才允许新冻结、门原样重跑、六点 `--ab` 与固定级联。
