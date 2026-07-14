# Phase462 Step 2c-2 runtime gate

结论：oracle 回归锚通过，短跑稳态硬门失败；停在六点 `--ab` 前，新引擎环不进入默认路径。

| 门 | 结果 | 目标 | 判定 |
|---|---:|---:|---|
| oracle drain | 100.00% | 100% | PASS |
| oracle 首调度步 | 100.00% | >=93.75% | PASS |
| oracle 前 16 步 | 100.00% | 100% | PASS |
| 短跑抢占 | 10 | 10+/-2 | PASS |
| 短跑稳态自抢占 | 2 | 0 | FAIL |
| 短跑重复 victim | 0 | 0 | PASS |
| sim 实际运行时长 | 1.528s | 记录 | measured |

稳态窗口为 `[1201, 10831)`，模式 `replacement_plateau_while_waiting_nonempty`。自抢占事件：

```json
[
  {
    "step": 2167,
    "trigger_request_id": 27,
    "victim_request_id": 27,
    "victim_preemptions_before": 0,
    "victim_sampled_output_tokens": 688,
    "victim_computed_output_tokens": 688,
    "victim_output_placeholders": 1,
    "recompute_tokens": 32688,
    "metrics_steady_state": true,
    "lifecycle_steady_state": true
  },
  {
    "step": 10616,
    "trigger_request_id": 124,
    "victim_request_id": 124,
    "victim_preemptions_before": 0,
    "victim_sampled_output_tokens": 288,
    "victim_computed_output_tokens": 288,
    "victim_output_placeholders": 1,
    "recompute_tokens": 32288,
    "metrics_steady_state": true,
    "lifecycle_steady_state": true
  }
]
```

2a-3f 已登记的 32k 证据同样包含稳态自抢占：`value=2/2; t0_regions={'ramp': 1, 'steady': 1}; oracle_regions={'ramp': 1, 'steady': 1}`。因此本次失败不是 arrival 接线回归，也不能通过改窗口、经验阈值或随机去同步修补。

`step2c2=blocked`，原因 `steady_state_signature_failed`。`engine_loop_enabled` 默认仍为 false；
DP 新路径在 Step 3 前显式拒绝，不静默切旧环。未运行六点 `--ab`，未进入归档行、DP 或 dp2-32k3k 级联。

边界：`diagnostic_only=true`、`valid_for_default=false`、`perf_database=false`、`Default AIC=No-Go`。
