# Phase 44: Default Path Safety Check

## 结论

Phase 44 没有新增代码路径。默认 `cb_sim` 继续保持 Phase 4 baseline；Phase 41 key 仍只在 experimental flag 下输出 descriptor。

| 检查项 | 结果 |
|---|---|
| `PerfDatabase` | 未新增接入 |
| `run_static` | 未修改 |
| `IterationLatencyCalculator` | 未修改 |
| runtime key | 未增加 latency 字段 |
| GPU benchmark | 未运行 |
| remote service | 未连接 |

## 默认路径要求

| 要求 | 状态 |
|---|---|
| 不写 perf table | 保持 |
| 不接 residual | 保持 |
| 不使用 profiler/NCCL/sync 数字 | 保持 |
| 不改默认 validate 输入 | 保持 |
| 不改 Phase 4 baseline | 保持 |

## 本轮 validate 结果

| 检查 | 结果 |
|---|---|
| throughput max error | PASS，`1.50x` |
| multi-config max error | PASS，`1.47x` |
| TTFT max error | PASS，`1.79x` |

## 安全检查命令

| 检查 | 命令 |
|---|---|
| 默认 validate | `conda run -n aic python scripts/validate_cb_simulator.py` |
| 源码污染 | `rg -n "phase44|query_.*compiled|wna16.*perf|compiled.*perf" src/aiconfigurator/sdk scripts` |
| 禁止字段 | `rg -n "residual_ms|profiled_cuda|line_count|throughput_ratio" docs/iter_gap_investigation/phase44_*.md` |
| whitespace | `perl -ne 'print "$ARGV:$.:$_" if /[ \t]$/' docs/iter_gap_investigation/phase44_*.md` |
| diff | `git diff --check` |

## 判定

| 如果发现 | 处理 |
|---|---|
| `PerfDatabase` 新入口 | 立即回退 |
| `run_static` / `IterationLatencyCalculator` 修改 | 立即回退 |
| Phase 44 文档出现可入模 ms 字段 | 改成禁止说明 |
| 默认 validate 退化 | 停止，不收口 |

当前 Phase 44 只允许文档收口，不允许任何默认路径改动。
