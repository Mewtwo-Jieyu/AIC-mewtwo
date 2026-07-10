#!/usr/bin/env python3
"""Phase461 Step4a-2: triage uncovered TP8 serving-state query cells."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import scripts.analyze_phase459_residual_triage as phase459
import scripts.analyze_phase461_step4_ingest_gate as ingest_gate
import validate_cb_simulator as validate
from aiconfigurator.sdk.backends.cb_simulator import CBSimulator
from aiconfigurator.sdk.backends.cb_simulator.iteration_latency import IterationLatencyCalculator


TARGET_WALL_COVERAGE = 0.95
RAW_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase461_cost_recollect"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase461_step4_cell_triage.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase461_step4_cell_triage.md"

SCENARIOS = (
    (
        "K2.5-tp8ep8-32k3k",
        RAW_ROOT / "tp8_32k3k_mixed/overhead_on/event_timing.jsonl.gz",
        REPO_ROOT
        / "docs/iter_gap_investigation/phase458_n512_unify/recollect_tp8_32k3k"
        / "K2.5-tp8ep8-32k3k/serve.log",
    ),
    (
        "K2.5-tp8ep8-8k2k-bt65536",
        RAW_ROOT / "tp8_bt65536_mixed/overhead_on/event_timing.jsonl.gz",
        REPO_ROOT
        / "docs/iter_gap_investigation/phase458_n512_unify/recollect_tp8_8k2k_bt65536"
        / "K2.5-tp8ep8-8k2k-bt65536/serve.log",
    ),
)

CSV_FIELDS = [
    "section",
    "scenario",
    "bucket_tokens",
    "decode_batch",
    "query_count",
    "sim_wall_ms",
    "wall_weight",
    "real_samples",
    "real_wall_median_ms",
    "classification",
    "decision",
    "projected_coverage",
    "metric",
    "value",
    "target",
    "status",
    "note",
]


@dataclass(frozen=True)
class QueryObservation:
    bucket_tokens: int
    decode_batch: int
    wall_ms: float


def summarize_weighted_coverage(
    observations: Iterable[QueryObservation],
    table: dict[int, dict[int, float]],
) -> dict[str, object]:
    observations = list(observations)
    if not observations:
        raise ValueError("no mixed query observations")
    uncovered: dict[tuple[int, int], dict[str, float | int]] = defaultdict(
        lambda: {"query_count": 0, "sim_wall_ms": 0.0}
    )
    covered_queries = 0
    covered_wall = 0.0
    total_wall = sum(item.wall_ms for item in observations)
    for item in observations:
        covered = ingest_gate.query_is_covered(
            table,
            bucket_tokens=item.bucket_tokens,
            decode_batch=item.decode_batch,
        )
        if covered:
            covered_queries += 1
            covered_wall += item.wall_ms
            continue
        cell = uncovered[(item.bucket_tokens, item.decode_batch)]
        cell["query_count"] = int(cell["query_count"]) + 1
        cell["sim_wall_ms"] = float(cell["sim_wall_ms"]) + item.wall_ms
    return {
        "query_count": len(observations),
        "covered_query_count": covered_queries,
        "query_coverage_ratio": covered_queries / len(observations),
        "total_wall_ms": total_wall,
        "covered_wall_ms": covered_wall,
        "uncovered_wall_ms": total_wall - covered_wall,
        "wall_coverage_ratio": covered_wall / total_wall,
        "uncovered_cells": dict(uncovered),
    }


def classify_uncovered_cells(
    cells: dict[tuple[int, int], dict[str, float | int]],
    real_cells: dict[tuple[int, int], list[float]],
) -> dict[tuple[int, int], dict[str, object]]:
    rows: dict[tuple[int, int], dict[str, object]] = {}
    for key, values in cells.items():
        real_values = real_cells.get(key, [])
        rows[key] = {
            **values,
            "real_samples": len(real_values),
            "real_wall_median_ms": statistics.median(real_values) if real_values else None,
            "classification": (
                "real_observed" if real_values else "sim_only_for_phase458_reference"
            ),
        }
    return rows


def attainable_coverage(
    rows: Iterable[dict[str, object]],
    *,
    current_coverage: float,
) -> float:
    additional = sum(
        float(row["wall_weight"])
        for row in rows
        if row["classification"] == "real_observed"
    )
    return round(min(1.0, current_coverage + additional), 12)


def assign_decisions(
    rows: Iterable[dict[str, object]],
    *,
    current_coverage: float,
    target_coverage: float,
) -> list[dict[str, object]]:
    projected = current_coverage
    result: list[dict[str, object]] = []
    for source in rows:
        row = dict(source)
        if row["classification"] != "real_observed":
            row["decision"] = "phase462_dynamics"
        elif projected < target_coverage:
            row["decision"] = "collect_anchor_target"
            projected += float(row["wall_weight"])
        else:
            row["decision"] = "analytic_fallback_tail"
        row["projected_coverage"] = round(min(projected, 1.0), 12)
        result.append(row)
    return result


def next_action(*, current_coverage: float, attainable: float, target: float) -> str:
    if current_coverage >= target:
        return "anchor_calibration_only"
    if attainable < target:
        return "phase462_dynamics_first"
    return "collect_real_observed_then_anchor"


def _capture_sim_mixed_queries(name: str) -> list[QueryObservation]:
    point = next(item for item in validate.MULTI_CONFIG_DATA if item.name == name)
    model, database, backend = validate._load_model_and_db(
        tp=point.tp,
        dp=point.dp,
        moe_tp=point.moe_tp,
        moe_ep=point.moe_ep,
    )
    config = validate._make_multi_config_cb_config(
        point,
        overlap_factor=0.0,
        ep8_per_iteration_overhead_ms=validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
    )
    sim = CBSimulator(backend, model, database, config)
    observations: list[QueryObservation] = []
    original = IterationLatencyCalculator.compute

    def capture(
        calculator: IterationLatencyCalculator,
        prefill_tokens: int,
        prefill_batch_size: int,
        prefill_seq_len: int,
        decode_batch_size: int,
        decode_avg_kv_len: int,
    ) -> float:
        wall_ms = original(
            calculator,
            prefill_tokens,
            prefill_batch_size,
            prefill_seq_len,
            decode_batch_size,
            decode_avg_kv_len,
        )
        if prefill_tokens > 0 and decode_batch_size > 0:
            observations.append(
                QueryObservation(
                    bucket_tokens=prefill_tokens + decode_batch_size,
                    decode_batch=decode_batch_size,
                    wall_ms=wall_ms,
                )
            )
        return wall_ms

    with patch.object(IterationLatencyCalculator, "compute", capture):
        sim.run(
            isl=point.isl,
            osl=point.osl,
            concurrency=math.ceil(point.batch_size / point.dp),
            num_gpus=point.tp,
        )
    return observations


def _real_mixed_cells(path: Path) -> dict[tuple[int, int], list[float]]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for step in phase459.parse_iteration_steps(path):
        if step.is_mixed:
            grouped[(step.ctx_tokens + step.generation_requests, step.generation_requests)].append(
                step.elapsed_ms
            )
    return dict(grouped)


def _candidate_table(path: Path) -> dict[int, dict[int, float]]:
    return ingest_gate._table(ingest_gate.mixed_cell_values(path))


def analyze_scenario(name: str, event_path: Path, real_log: Path) -> dict[str, object]:
    coverage = summarize_weighted_coverage(
        _capture_sim_mixed_queries(name),
        _candidate_table(event_path),
    )
    classified = classify_uncovered_cells(
        coverage["uncovered_cells"],
        _real_mixed_cells(real_log),
    )
    total_wall = float(coverage["total_wall_ms"])
    ranked = []
    for (bucket, batch), row in classified.items():
        ranked.append(
            {
                "bucket_tokens": bucket,
                "decode_batch": batch,
                **row,
                "wall_weight": float(row["sim_wall_ms"]) / total_wall,
            }
        )
    ranked.sort(key=lambda row: float(row["wall_weight"]), reverse=True)
    decisions = assign_decisions(
        ranked,
        current_coverage=float(coverage["wall_coverage_ratio"]),
        target_coverage=TARGET_WALL_COVERAGE,
    )
    return {
        "coverage": coverage,
        "cells": decisions,
        "attainable_coverage": attainable_coverage(
            decisions,
            current_coverage=float(coverage["wall_coverage_ratio"]),
        ),
    }


def build_report_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, event_path, real_log in SCENARIOS:
        result = analyze_scenario(name, event_path, real_log)
        coverage = result["coverage"]
        cells = result["cells"]
        summary = {
            "query_count": coverage["query_count"],
            "uncovered_query_count": (
                int(coverage["query_count"]) - int(coverage["covered_query_count"])
            ),
            "query_coverage_ratio": coverage["query_coverage_ratio"],
            "wall_coverage_ratio": coverage["wall_coverage_ratio"],
            "attainable_coverage": result["attainable_coverage"],
            "real_observed_uncovered_cells": sum(
                row["classification"] == "real_observed" for row in cells
            ),
            "sim_only_reference_cells": sum(
                row["classification"] == "sim_only_for_phase458_reference" for row in cells
            ),
            "collect_anchor_targets": sum(
                row["decision"] == "collect_anchor_target" for row in cells
            ),
            "sim_only_wall_weight": sum(
                float(row["wall_weight"])
                for row in cells
                if row["classification"] == "sim_only_for_phase458_reference"
            ),
            "next_action": next_action(
                current_coverage=float(coverage["wall_coverage_ratio"]),
                attainable=float(result["attainable_coverage"]),
                target=TARGET_WALL_COVERAGE,
            ),
        }
        for metric, value in summary.items():
            rows.append(
                {
                    "section": "summary",
                    "scenario": name,
                    "metric": metric,
                    "value": value,
                    "target": f">={TARGET_WALL_COVERAGE}" if metric == "wall_coverage_ratio" else "",
                    "status": (
                        "pass"
                        if metric != "wall_coverage_ratio" or float(value) >= TARGET_WALL_COVERAGE
                        else "fail"
                    ),
                    "note": "weight=query frequency x current sim iteration wall",
                }
            )
        for cell in cells:
            rows.append(
                {
                    "section": "cell",
                    "scenario": name,
                    **cell,
                    "status": "triaged",
                    "note": "exact Phase458 reference-cell match only",
                }
            )
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    summary_rows = [row for row in rows if row["section"] == "summary"]
    by_scenario: dict[str, dict[str, object]] = defaultdict(dict)
    for row in summary_rows:
        by_scenario[str(row["scenario"])][str(row["metric"])] = row["value"]
    cell_rows = [row for row in rows if row["section"] == "cell"]
    lines = [
        "# Phase461 Step4a-2 未覆盖 cell 分诊",
        "",
        "结论：Step4a 的 62.6%/20.7% 是 latency cache 首次查询口径；本步截取每次真实 sim 调度调用后，按 `查询频次 × 当前 sim 迭代墙钟` 重新加权。`sim_only_for_phase458_reference` 只表示当前 Phase458 参考日志没有出现该精确 cell，不宣称真实系统永远不会产生。全程未改 runtime、PerfDB 或 gate，Default AIC 维持 No-Go。",
        "",
        "| 场景 | 调度查询 | 未覆盖查询 | 查询覆盖 | 墙钟覆盖 | 可达覆盖 | real-observed 缺口 | reference 未观察缺口 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario, values in by_scenario.items():
        lines.append(
            f"| {scenario} | {int(values['query_count'])} | "
            f"{int(values['uncovered_query_count'])} | "
            f"{float(values['query_coverage_ratio']):.1%} | "
            f"{float(values['wall_coverage_ratio']):.1%} | "
            f"{float(values['attainable_coverage']):.1%} | "
            f"{int(values['real_observed_uncovered_cells'])} | "
            f"{int(values['sim_only_reference_cells'])} |"
        )
    lines.extend(
        [
            "",
            "## 下一步分流",
            "",
            "| 场景 | 动作 | 原因 |",
            "|---|---|---|",
        ]
    )
    for scenario, values in by_scenario.items():
        action = str(values["next_action"])
        reason = (
            "加权覆盖已过 95%，不补覆盖点；只设计跨 session 校准锚点"
            if action == "anchor_calibration_only"
            else "当前 reference 未观察到全部缺口，补采无法达到 95%；先移交 Phase462 修 dynamics"
        )
        lines.append(f"| {scenario} | {action} | {reason} |")
    lines.extend(
        [
            "",
            "## 高权重处置",
            "",
            "| 场景 | cell | 权重 | real 样本 | 分类 | 决策 |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for row in sorted(cell_rows, key=lambda item: float(item["wall_weight"]), reverse=True)[:20]:
        lines.append(
            f"| {row['scenario']} | {row['bucket_tokens']}/{row['decode_batch']} | "
            f"{float(row['wall_weight']):.2%} | {row['real_samples']} | "
            f"{row['classification']} | {row['decision']} |"
        )
    lines.extend(
        [
            "",
            "## 口径",
            "",
            "| 项目 | 定义 |",
            "|---|---|",
            "| 墙钟权重 | 当前 sim 每次 mixed 调度实际返回的 iteration latency；通过离线 monkeypatch 采集，不改源码行为 |",
            "| real-observed | Phase458 N=512 serve.log 中存在完全相同的 bucket/decode cell |",
            "| sim-only for reference | 当前 Phase458 reference 未观察到；移交 Phase462 验证 dynamics 修复后是否消失 |",
            "| 95% 门 | 本报告提出的下一轮采集预算门，不修改正式 gate |",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    rows = build_report_rows()
    write_csv(args.csv, rows)
    write_markdown(args.md, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
