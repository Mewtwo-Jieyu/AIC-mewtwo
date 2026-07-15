#!/usr/bin/env python3
"""Phase462 Step2c-11 engine-loop arc closure."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCOREBOARD = REPO_ROOT / (
    "docs/iter_gap_investigation/phase462_null_only_scoreboard.csv"
)
DEFAULT_REPORT = REPO_ROOT / (
    "docs/iter_gap_investigation/phase462_engine_loop_arc_closure.md"
)
PASSING_SCENARIOS = frozenset(
    {
        "K2.5-tp8ep8-8k2k",
        "K2.5-tp4ep8dp2-8k2k",
        "K2.5-tp8ep8-32k3k",
    }
)


@dataclass(frozen=True)
class ScoreRow:
    scenario: str
    real_output_tok_s_gpu: float
    sim_output_tok_s_gpu: float
    error_ratio: float
    gate_pass: bool


def load_scoreboard(path: Path) -> list[ScoreRow]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = [
            ScoreRow(
                scenario=row["scenario"],
                real_output_tok_s_gpu=float(row["real_output_tok_s_gpu"]),
                sim_output_tok_s_gpu=float(row["sim_output_tok_s_gpu"]),
                error_ratio=float(row["error_ratio"]),
                gate_pass=row["gate_pass"].lower() == "true",
            )
            for row in csv.DictReader(source)
        ]
    scenarios = [row.scenario for row in rows]
    if len(rows) != 6 or len(set(scenarios)) != 6:
        raise AssertionError("null-only scoreboard must contain six unique scenarios")
    passing = {row.scenario for row in rows if row.gate_pass}
    if passing != PASSING_SCENARIOS:
        raise AssertionError(
            f"unexpected null-only passing set: {sorted(passing)}"
        )
    for row in rows:
        if row.gate_pass != (row.error_ratio <= 1.15):
            raise AssertionError(f"{row.scenario}: gate flag disagrees with error")
    return rows


def write_report(path: Path, rows: Sequence[ScoreRow]) -> None:
    scoreboard = "\n".join(
        f"| {row.scenario} | {row.real_output_tok_s_gpu:.6f} | "
        f"{row.sim_output_tok_s_gpu:.6f} | {row.error_ratio:.6f} | "
        f"{'PASS' if row.gate_pass else 'FAIL'} |"
        for row in rows
    )
    path.write_text(
        f"""# Phase462 Step 2c-11 引擎环弧线收口

结论：默认路径只保留 vLLM 0.19 每个 BlockPool 预留 1 个 null block 的容量语义；引擎环核心状态机 parked 且默认关闭，到达/暴露建模剔出默认路径。`null-only` 六点结果定为新的官方基线，计分板保持 3/6，`Default AIC=No-Go`。

## 官方基线

| 场景 | real tok/s/GPU | sim tok/s/GPU | error | 15% gate |
|---|---:|---:|---:|---|
{scoreboard}

通过集合固定为 TP8-8k2k、DP2-8k2k、TP8-32k3k；动态主线目标固定为 DP2-32k3k、TP8-bt65536、DP2-bt65536。

## 产物定位

| 产物 | 收口定位 | 默认路径 |
|---|---|---|
| null block 容量修复 | 源码正确的 vLLM 0.19 容量语义；其暴露的残差移交动态主线 | 保留 |
| EngineCore 引擎环状态机 | oracle 取证成立，但未在六场景证明可迁移收益 | parked / off |
| tokenizer 到达与 workload 暴露 | client、HTTP、api-server、harness 特定 | 剔出 |
| 三点边界日志 | scheduler 入参、结果、future 完成的 `diagnostic-only` 资产 | 不参与预测 |
| first-divergence 分析器族 | 容量、时序、抢占首分叉定位的 `diagnostic-only` 资产 | 不参与预测 |
| 释放链回溯与门判卷器 | 事件级因果审计的 `diagnostic-only` 资产 | 不参与默认验收 |

本弧线的交付收益是 null block 修复；事件级工具链保留为诊断资产，不再以默认建模能力计分。

## 移交

1. 版本语义 profile v1 只收纳源码可核的 backend-version 语义，必须保持默认六点输出逐字节不变。
2. DP Step 3 在本表的 null-only 基线上重做 dp2-bt65536 相位/成本分解，并复查 Phase461 sim-only 大 bucket。
3. Step 4 先解释 dp2-32k3k 从 1.173318 到 1.195756 的暴露面，再做六点验收。

边界：未改 gate 或 PerfDB；未跑 GPU；6/6 前不收紧门限、不翻转 Default AIC。
""",
        encoding="utf-8",
    )


def run_closure(
    *,
    scoreboard: Path = DEFAULT_SCOREBOARD,
    report: Path = DEFAULT_REPORT,
) -> dict[str, object]:
    rows = load_scoreboard(scoreboard)
    write_report(report, rows)
    return {
        "status": "completed_report_only",
        "scoreboard": f"{sum(row.gate_pass for row in rows)}/6",
        "official_baseline": "null_only",
        "engine_loop_default": "off",
        "default_aic": "No-Go",
        "runtime_changed": False,
        "gate_changed": False,
        "perfdb_changed": False,
    }


def main() -> int:
    print(json.dumps(run_closure(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
