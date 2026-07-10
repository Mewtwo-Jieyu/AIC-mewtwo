#!/usr/bin/env python3
"""Phase461 Step4a-3: validate TP8 32k cross-session anchors offline."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import scripts.analyze_phase459_residual_triage as phase459
import scripts.analyze_phase461_cost_recollect as collect
import scripts.analyze_phase461_step4_cell_triage as triage


SCENARIO = "K2.5-tp8ep8-32k3k"
RAW_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase461_cost_recollect/tp8_32k3k_mixed"
EVENT_PATH = RAW_ROOT / "overhead_on/event_timing.jsonl.gz"
SAME_SESSION_LOG = RAW_ROOT / "overhead_on" / SCENARIO / "serve.log.gz"
REFERENCE_LOG = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase458_n512_unify/recollect_tp8_32k3k"
    / SCENARIO
    / "serve.log"
)
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase461_step4_anchor_validation.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase461_step4_anchor_validation.md"
MIN_ANCHORS = 5
MAX_DRIFT = 0.10
QueryObservation = triage.QueryObservation

CSV_FIELDS = [
    "section",
    "scenario",
    "bucket_tokens",
    "decode_batch",
    "selected",
    "event_samples",
    "same_wall_samples",
    "reference_wall_samples",
    "event_busy_ms",
    "same_wall_ms",
    "reference_wall_ms",
    "event_over_same_wall",
    "same_wall_over_reference_wall",
    "event_over_reference_wall",
    "max_relative_drift",
    "influence_hit_count",
    "metric",
    "value",
    "target",
    "status",
    "note",
]


@dataclass(frozen=True)
class CellStats:
    median_ms: float
    samples: int


def select_anchor_cells(
    event: dict[tuple[int, int], CellStats],
    same_wall: dict[tuple[int, int], CellStats],
    reference_wall: dict[tuple[int, int], CellStats],
    *,
    count: int,
) -> list[tuple[int, int]]:
    common = set(event) & set(same_wall) & set(reference_wall)
    ranked = sorted(
        common,
        key=lambda key: (
            -min(event[key].samples, same_wall[key].samples, reference_wall[key].samples),
            key[0],
            key[1],
        ),
    )
    return ranked[:count]


def compare_anchor(
    key: tuple[int, int],
    event: CellStats,
    same_wall: CellStats,
    reference_wall: CellStats,
) -> dict[str, object]:
    ratios = {
        "event_over_same_wall": event.median_ms / same_wall.median_ms,
        "same_wall_over_reference_wall": same_wall.median_ms / reference_wall.median_ms,
        "event_over_reference_wall": event.median_ms / reference_wall.median_ms,
    }
    return {
        "bucket_tokens": key[0],
        "decode_batch": key[1],
        "event_samples": event.samples,
        "same_wall_samples": same_wall.samples,
        "reference_wall_samples": reference_wall.samples,
        "event_busy_ms": event.median_ms,
        "same_wall_ms": same_wall.median_ms,
        "reference_wall_ms": reference_wall.median_ms,
        **ratios,
        "max_relative_drift": max(abs(value - 1.0) for value in ratios.values()),
    }


def anchor_gate(
    anchors: Iterable[dict[str, object]],
    *,
    min_cells: int,
    max_drift: float,
) -> str:
    anchors = list(anchors)
    if len(anchors) < min_cells:
        return "blocked"
    if any(float(anchor["max_relative_drift"]) > max_drift for anchor in anchors):
        return "blocked"
    return "pass"


def _bracket(value: int, values: Iterable[int]) -> tuple[int, int] | None:
    ordered = sorted(int(item) for item in values)
    if value in ordered:
        return value, value
    left = [item for item in ordered if item < value]
    right = [item for item in ordered if item > value]
    return (max(left), min(right)) if left and right else None


def query_uses_cell(
    table: dict[int, dict[int, float]],
    *,
    bucket_tokens: int,
    decode_batch: int,
    cell: tuple[int, int],
) -> bool:
    token_bracket = _bracket(bucket_tokens, table)
    if token_bracket is None or cell[0] not in token_bracket:
        return False
    batch_bracket = _bracket(decode_batch, table[cell[0]])
    return batch_bracket is not None and cell[1] in batch_bracket


def ingest_unlock(*, anchor_verdict: str, adjusted_wall_coverage: float) -> str:
    if anchor_verdict == "pass" and adjusted_wall_coverage >= 0.95:
        return "pass"
    return "blocked"


def qualified_wall_coverage(
    observations: Iterable[QueryObservation],
    table: dict[int, dict[int, float]],
    *,
    blocked_cells: Iterable[tuple[int, int]],
) -> dict[str, float | int]:
    observations = list(observations)
    blocked_cells = list(blocked_cells)
    total_wall = sum(item.wall_ms for item in observations)
    qualified_wall = 0.0
    influence_wall = 0.0
    influence_count = 0
    for item in observations:
        covered = triage.ingest_gate.query_is_covered(
            table,
            bucket_tokens=item.bucket_tokens,
            decode_batch=item.decode_batch,
        )
        if not covered:
            continue
        influenced = any(
            query_uses_cell(
                table,
                bucket_tokens=item.bucket_tokens,
                decode_batch=item.decode_batch,
                cell=cell,
            )
            for cell in blocked_cells
        )
        if influenced:
            influence_count += 1
            influence_wall += item.wall_ms
        else:
            qualified_wall += item.wall_ms
    return {
        "qualified_wall_coverage": qualified_wall / total_wall,
        "blocked_influence_count": influence_count,
        "blocked_influence_wall_weight": influence_wall / total_wall,
    }


def _event_cells(path: Path) -> dict[tuple[int, int], CellStats]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    steps = collect.group_tp_steps(collect.read_event_rows(path), tp_width=8)
    for step in steps:
        if step.phase == "mixed_prefill":
            grouped[(step.bucket_tokens, step.generation_requests)].append(step.forward_busy_ms)
    return {
        key: CellStats(statistics.median(values), len(values))
        for key, values in grouped.items()
    }


def _wall_cells(path: Path) -> dict[tuple[int, int], CellStats]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for step in phase459.parse_iteration_steps(path):
        if step.is_mixed:
            grouped[(step.ctx_tokens + step.generation_requests, step.generation_requests)].append(
                step.elapsed_ms
            )
    return {
        key: CellStats(statistics.median(values), len(values))
        for key, values in grouped.items()
    }


def build_report_rows() -> list[dict[str, object]]:
    event = _event_cells(EVENT_PATH)
    same_wall = _wall_cells(SAME_SESSION_LOG)
    reference_wall = _wall_cells(REFERENCE_LOG)
    common = sorted(set(event) & set(same_wall) & set(reference_wall))
    selected = select_anchor_cells(
        event,
        same_wall,
        reference_wall,
        count=MIN_ANCHORS,
    )
    comparisons = {
        key: compare_anchor(key, event[key], same_wall[key], reference_wall[key])
        for key in common
    }
    verdict = anchor_gate(
        [comparisons[key] for key in selected],
        min_cells=MIN_ANCHORS,
        max_drift=MAX_DRIFT,
    )
    blocked_cells = [
        key for key in common if float(comparisons[key]["max_relative_drift"]) > MAX_DRIFT
    ]
    candidate_table = triage._candidate_table(EVENT_PATH)
    observations = triage._capture_sim_mixed_queries(SCENARIO)
    influence_counts = {
        key: sum(
            triage.ingest_gate.query_is_covered(
                candidate_table,
                bucket_tokens=item.bucket_tokens,
                decode_batch=item.decode_batch,
            )
            and query_uses_cell(
                candidate_table,
                bucket_tokens=item.bucket_tokens,
                decode_batch=item.decode_batch,
                cell=key,
            )
            for item in observations
        )
        for key in blocked_cells
    }
    adjusted_coverage = qualified_wall_coverage(
        observations,
        candidate_table,
        blocked_cells=blocked_cells,
    )
    unlock = ingest_unlock(
        anchor_verdict=verdict,
        adjusted_wall_coverage=float(adjusted_coverage["qualified_wall_coverage"]),
    )
    rows: list[dict[str, object]] = [
        {
            "section": "summary",
            "scenario": SCENARIO,
            "metric": "common_cells",
            "value": len(common),
            "target": f">={MIN_ANCHORS}",
            "status": "pass" if len(common) >= MIN_ANCHORS else "fail",
            "note": "exact cell intersection across event/same-wall/reference-wall",
        },
        {
            "section": "summary",
            "scenario": SCENARIO,
            "metric": "selected_anchors",
            "value": len(selected),
            "target": str(MIN_ANCHORS),
            "status": "pass" if len(selected) >= MIN_ANCHORS else "fail",
            "note": "ranked only by minimum sample support; never by latency drift",
        },
        {
            "section": "summary",
            "scenario": SCENARIO,
            "metric": "anchor_gate",
            "value": verdict,
            "target": "pass",
            "status": verdict,
            "note": f"every selected anchor max drift <= {MAX_DRIFT:.0%}",
        },
        {
            "section": "summary",
            "scenario": SCENARIO,
            "metric": "blocked_candidate_cells",
            "value": len(blocked_cells),
            "target": "reported and excluded",
            "status": "pass",
            "note": "all exact overlaps are audited, not only selected anchors",
        },
        {
            "section": "summary",
            "scenario": SCENARIO,
            "metric": "qualified_wall_coverage",
            "value": adjusted_coverage["qualified_wall_coverage"],
            "target": ">=0.95",
            "status": (
                "pass"
                if float(adjusted_coverage["qualified_wall_coverage"]) >= 0.95
                else "fail"
            ),
            "note": (
                "queries influenced by drifted cells are unqualified;"
                f"influence_count={adjusted_coverage['blocked_influence_count']};"
                f"influence_wall_weight={adjusted_coverage['blocked_influence_wall_weight']}"
            ),
        },
        {
            "section": "summary",
            "scenario": SCENARIO,
            "metric": "step4b_unlock",
            "value": unlock,
            "target": "pass",
            "status": unlock,
            "note": "requires anchor gate and >=95% qualified wall coverage",
        },
    ]
    for key in common:
        comparison = comparisons[key]
        rows.append(
            {
                "section": "anchor_cell",
                "scenario": SCENARIO,
                "selected": key in selected,
                **comparison,
                "influence_hit_count": influence_counts.get(key, 0),
                "status": (
                    "pass" if float(comparison["max_relative_drift"]) <= MAX_DRIFT else "fail"
                ),
                "note": "selected by support" if key in selected else "reported, not gate-bearing",
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
    summary = {row["metric"]: row for row in rows if row["section"] == "summary"}
    selected = [
        row for row in rows if row["section"] == "anchor_cell" and bool(row["selected"])
    ]
    blocked = [
        row
        for row in rows
        if row["section"] == "anchor_cell" and float(row["max_relative_drift"]) > MAX_DRIFT
    ]
    lines = [
        "# Phase461 Step4a-3 TP8 32k 离线锚验证",
        "",
        f"结论：离线锚门 `{summary['anchor_gate']['value']}`，排除漂移行后的 Step4b 解锁门 `{summary['step4b_unlock']['value']}`。锚点只按三份数据的最小样本数确定，不按延迟接近程度挑选。未改 runtime、PerfDB 或 gate；Default AIC 维持 No-Go。",
        "",
        "| 检查 | 值 | 门 | 状态 |",
        "|---|---:|---:|---|",
        f"| 三方完全同 cell | {summary['common_cells']['value']} | >={MIN_ANCHORS} | {summary['common_cells']['status']} |",
        f"| 选定锚点 | {summary['selected_anchors']['value']} | {MIN_ANCHORS} | {summary['selected_anchors']['status']} |",
        f"| 每锚最大漂移 | 见下表 | <={MAX_DRIFT:.0%} | {summary['anchor_gate']['status']} |",
        f"| 漂移候选行 | {summary['blocked_candidate_cells']['value']} | 排除并报告 | {summary['blocked_candidate_cells']['status']} |",
        f"| 扣除受漂移行影响后的墙钟覆盖 | {float(summary['qualified_wall_coverage']['value']):.2%} | >=95% | {summary['qualified_wall_coverage']['status']} |",
        f"| Step4b 解锁 | {summary['step4b_unlock']['value']} | pass | {summary['step4b_unlock']['status']} |",
        "",
        "| cell | event / 同 session wall | 同 session / Phase458 wall | event / Phase458 wall | 最大漂移 | 样本 event/same/ref | 状态 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in selected:
        lines.append(
            f"| {row['bucket_tokens']}/{row['decode_batch']} | "
            f"{float(row['event_over_same_wall']):.4f} | "
            f"{float(row['same_wall_over_reference_wall']):.4f} | "
            f"{float(row['event_over_reference_wall']):.4f} | "
            f"{float(row['max_relative_drift']):.2%} | "
            f"{row['event_samples']}/{row['same_wall_samples']}/{row['reference_wall_samples']} | "
            f"{row['status']} |"
        )
    lines.extend(
        [
            "",
            "## 排除行",
            "",
            "| cell | event / Phase458 wall | 最大漂移 | 查询影响次数 | 决策 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in blocked:
        lines.append(
            f"| {row['bucket_tokens']}/{row['decode_batch']} | "
            f"{float(row['event_over_reference_wall']):.4f} | "
            f"{float(row['max_relative_drift']):.2%} | {row['influence_hit_count']} | "
            "不入库；受其影响的查询不计合格覆盖 |"
        )
    lines.extend(
        [
            "",
            "## Step4b 解锁范围",
            "",
            "| 项目 | 决定 |",
            "|---|---|",
            "| MLA batch8/KV32768 | 允许用 Phase461 microbench 真值替换 |",
            "| TP8 32k mixed | 允许 scoped 入库，但明确排除 `92/13` |",
            "| TP8 bt65536 | 继续排除，维持解析链，等待 Phase462 dynamics |",
            "| 六点 A/B | 入库后必须全表运行；8k2k 两点应零移动 |",
        ]
    )
    lines.extend(
        [
            "",
            "| 口径 | 说明 |",
            "|---|---|",
            "| event busy / same wall | 检查 graph-outer event 是否遗漏同 session 的 iteration wall |",
            "| same wall / reference wall | 检查 Step3 patched session 与 Phase458 vanilla session 是否漂移 |",
            "| event busy / reference wall | 最终验证拟入库 forward_total 是否可代表验收参考 |",
            "| 选择纪律 | 只按共同样本支持度排序；漂移只用于过门，不参与选点 |",
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
    verdict = next(row for row in rows if row.get("metric") == "step4b_unlock")
    return 0 if verdict["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
