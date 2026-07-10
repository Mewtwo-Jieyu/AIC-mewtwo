#!/usr/bin/env python3
"""Phase461 Step4 pre-ingest coverage and session-comparability gates."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import scripts.analyze_phase461_cost_recollect as collect
import scripts.analyze_phase461_max_bt_scope as scope


RAW_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase461_cost_recollect"
REFERENCE_EVENTS = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase454_gpu_batch_scope"
    / "b2b_tp8_8k2k/overhead_on/event_timing.jsonl"
)
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase461_step4_ingest_gate.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase461_step4_ingest_gate.md"
MIN_OVERLAP_CELLS = 5
MAX_RELATIVE_DRIFT = 0.15

SCENARIOS = (
    (
        "K2.5-tp8ep8-32k3k",
        RAW_ROOT / "tp8_32k3k_mixed/overhead_on/event_timing.jsonl.gz",
    ),
    (
        "K2.5-tp8ep8-8k2k-bt65536",
        RAW_ROOT / "tp8_bt65536_mixed/overhead_on/event_timing.jsonl.gz",
    ),
)

CSV_FIELDS = [
    "section",
    "scenario",
    "metric",
    "bucket_tokens",
    "decode_batch",
    "value",
    "target",
    "status",
    "note",
]


def _bracket(value: int, values: Iterable[int]) -> tuple[int, int] | None:
    ordered = sorted(int(item) for item in values)
    if value in ordered:
        return value, value
    left = [item for item in ordered if item < value]
    right = [item for item in ordered if item > value]
    if not left or not right:
        return None
    return max(left), min(right)


def query_is_covered(
    table: dict[int, dict[int, float]],
    *,
    bucket_tokens: int,
    decode_batch: int,
) -> bool:
    token_bracket = _bracket(bucket_tokens, table)
    if token_bracket is None:
        return False
    return all(_bracket(decode_batch, table[token]) is not None for token in token_bracket)


def summarize_query_coverage(
    records: Iterable[dict[str, object]],
    table: dict[int, dict[int, float]],
) -> dict[str, object]:
    cells = Counter(
        (int(record["bucket_tokens"]), int(record["decode_batch"]))
        for record in records
    )
    uncovered = [
        (bucket, batch, count)
        for (bucket, batch), count in sorted(cells.items())
        if not query_is_covered(
            table,
            bucket_tokens=bucket,
            decode_batch=batch,
        )
    ]
    covered_count = sum(cells.values()) - sum(item[2] for item in uncovered)
    query_count = sum(cells.values())
    return {
        "query_count": query_count,
        "unique_query_cells": len(cells),
        "covered_query_count": covered_count,
        "uncovered_query_count": query_count - covered_count,
        "uncovered_unique_cells": len(uncovered),
        "coverage_ratio": covered_count / query_count if query_count else 0.0,
        "uncovered_cells": uncovered,
    }


def summarize_session_comparability(
    reference: dict[tuple[int, int], float],
    candidate: dict[tuple[int, int], float],
    *,
    min_overlap_cells: int,
    max_relative_drift: float,
) -> dict[str, object]:
    overlap = sorted(set(reference) & set(candidate))
    drifts = [
        abs(candidate[key] - reference[key]) / reference[key]
        for key in overlap
        if reference[key] > 0
    ]
    observed_max = round(max(drifts), 12) if drifts else None
    passed = (
        len(overlap) >= min_overlap_cells
        and observed_max is not None
        and observed_max <= max_relative_drift
    )
    return {
        "overlap_cells": len(overlap),
        "max_relative_drift": observed_max,
        "median_relative_drift": round(statistics.median(drifts), 12) if drifts else None,
        "passed": passed,
        "cells": overlap,
    }


def ingest_gate(*, coverage_ratio: float, comparability_passed: bool) -> str:
    return "pass" if coverage_ratio == 1.0 and comparability_passed else "blocked"


def mixed_cell_values(path: Path) -> dict[tuple[int, int], float]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    steps = collect.group_tp_steps(collect.read_event_rows(path), tp_width=8)
    for step in steps:
        if step.phase == "mixed_prefill":
            grouped[(step.bucket_tokens, step.generation_requests)].append(step.forward_busy_ms)
    if not grouped:
        raise ValueError(f"no mixed TP8 cells: {path}")
    return {key: statistics.median(values) for key, values in grouped.items()}


def _table(cells: dict[tuple[int, int], float]) -> dict[int, dict[int, float]]:
    table: dict[int, dict[int, float]] = defaultdict(dict)
    for (bucket, batch), latency in cells.items():
        table[bucket][batch] = latency
    return dict(table)


def _query_records(name: str) -> list[dict[str, object]]:
    records, _ = scope.run_scenario_query_audit(name)
    return [
        record
        for record in records
        if record.get("phase") == "mixed_prefill"
        and record.get("row_kind") == "forward_total"
        and record.get("category") == "forward_total"
    ]


def build_report_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    reference = mixed_cell_values(REFERENCE_EVENTS)
    for scenario, path in SCENARIOS:
        cells = mixed_cell_values(path)
        coverage = summarize_query_coverage(_query_records(scenario), _table(cells))
        comparability = summarize_session_comparability(
            reference,
            cells,
            min_overlap_cells=MIN_OVERLAP_CELLS,
            max_relative_drift=MAX_RELATIVE_DRIFT,
        )
        gate = ingest_gate(
            coverage_ratio=float(coverage["coverage_ratio"]),
            comparability_passed=bool(comparability["passed"]),
        )
        for metric in (
            "query_count",
            "unique_query_cells",
            "covered_query_count",
            "uncovered_query_count",
            "uncovered_unique_cells",
            "coverage_ratio",
        ):
            rows.append(
                {
                    "section": "coverage_summary",
                    "scenario": scenario,
                    "metric": metric,
                    "value": coverage[metric],
                    "target": "1.0" if metric == "coverage_ratio" else "",
                    "status": "pass" if metric != "coverage_ratio" or coverage[metric] == 1.0 else "fail",
                    "note": f"candidate_cells={len(cells)}",
                }
            )
        for bucket, batch, count in coverage["uncovered_cells"]:
            rows.append(
                {
                    "section": "uncovered_query_cell",
                    "scenario": scenario,
                    "metric": "query_count",
                    "bucket_tokens": bucket,
                    "decode_batch": batch,
                    "value": count,
                    "target": "0",
                    "status": "fail",
                    "note": "inner_only has no complete token+batch bracket",
                }
            )
        rows.append(
            {
                "section": "session_comparability",
                "scenario": scenario,
                "metric": "exact_overlap_cells",
                "value": comparability["overlap_cells"],
                "target": f">={MIN_OVERLAP_CELLS}",
                "status": "pass" if comparability["passed"] else "fail",
                "note": (
                    f"max_relative_drift={comparability['max_relative_drift']};"
                    f"reference=phase454_tp8_8k2k"
                ),
            }
        )
        rows.append(
            {
                "section": "ingest_gate",
                "scenario": scenario,
                "metric": "verdict",
                "value": gate,
                "target": "pass",
                "status": gate,
                "note": "both full coverage and session comparability are required",
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
    summaries = [row for row in rows if row["section"] == "coverage_summary"]
    comparability = [row for row in rows if row["section"] == "session_comparability"]
    gates = [row for row in rows if row["section"] == "ingest_gate"]
    by_scenario: dict[str, dict[str, object]] = defaultdict(dict)
    for row in summaries:
        by_scenario[str(row["scenario"])][str(row["metric"])] = row["value"]
    comparisons = {str(row["scenario"]): row for row in comparability}
    verdicts = {str(row["scenario"]): row for row in gates}
    lines = [
        "# Phase461 Step4 入库前硬门",
        "",
        "结论：两个 TP8 新表都未覆盖当前查询包络，且与 Phase454 参考 session 没有任何完全相同的 `(bucket_tokens, decode_batch)` cell。跨 session 可比性无法成立，Step4 在入库前停止；MLA 坏行和 TP8 rows 均不写入 PerfDB，不运行带病 `--ab`。Default AIC 维持 No-Go。",
        "",
        "| 场景 | 查询覆盖 | 未覆盖唯一 cell | 新表 cell | 旧 session 精确重叠 | 入库门 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for scenario, values in by_scenario.items():
        comparison = comparisons[scenario]
        verdict = verdicts[scenario]
        lines.append(
            f"| {scenario} | {int(values['covered_query_count'])}/{int(values['query_count'])} "
            f"({float(values['coverage_ratio']):.1%}) | {int(values['uncovered_unique_cells'])} | "
            f"{str(next(row['note'] for row in summaries if row['scenario'] == scenario)).split('=', 1)[-1]} | "
            f"{comparison['value']} | {verdict['value']} |"
        )
    lines.extend(
        [
            "",
            "| 硬门 | 预先声明 | 实际结果 |",
            "|---|---|---|",
            "| 二维覆盖 | 每个实际 mixed 查询都有完整 token 与 batch 括号 | 两场景均失败 |",
            "| session 可比性 | 至少 5 个完全同 cell，最大相对漂移不超过 15% | 两场景均为 0 个重叠 cell |",
            "| 数据处置 | 任一硬门失败即停止 | 未改 runtime / PerfDB / gate |",
            "",
            "下一步不能靠扩大插值、clamp 或邻点替代。需要补一个带校准锚点的 TP8 workload window：同一 session 先重复至少 5 个 Phase454 cell，再覆盖 32k/65k 当前查询包络；或者由用户明确批准改用同 run event busy / iteration wall 作为新的可比性门。",
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
    return 1 if any(row["section"] == "ingest_gate" and row["status"] != "pass" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
