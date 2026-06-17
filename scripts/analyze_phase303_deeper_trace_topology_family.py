#!/usr/bin/env python3
"""Combine deeper trace topology diagnostics into a family-level audit."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import NamedTuple


SOURCE = "phase303_deeper_trace_topology_family"
DEFAULT_READINESS = "No-Go"
DEFAULT_PHASE287 = Path("docs/iter_gap_investigation/phase287_deeper_trace_partial_audit.csv")
DEFAULT_PHASE301 = Path("docs/iter_gap_investigation/phase301_boundary_timeline_cadence_diagnostic.csv")
DEFAULT_CSV_OUT = Path("docs/iter_gap_investigation/phase303_deeper_trace_topology_family.csv")
DEFAULT_DOC_OUT = Path("docs/iter_gap_investigation/phase303_deeper_trace_topology_family.md")

FIELDNAMES = [
    "source",
    "family_key",
    "topology_key",
    "shape_key",
    "control_scenario",
    "holdout_scenario",
    "control_bt",
    "holdout_bt",
    "output_ratio",
    "throughput_direction",
    "holdout_scheduled_p99",
    "holdout_scheduled_max",
    "holdout_max_fill",
    "budget_ceiling_rejected",
    "verdict",
    "mechanism_conclusion",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]


class SourceSpec(NamedTuple):
    source: str
    topology_key: str
    expected_ratio: str
    direction: str
    verdict: str
    fill_field: str


PHASE287_SPEC = SourceSpec(
    "phase287_deeper_trace_partial_audit",
    "tp8_dp1_ep8",
    "0.969300",
    "holdout_slower",
    "partial_only",
    "holdout_max_fill",
)
PHASE301_SPEC = SourceSpec(
    "phase301_boundary_timeline_cadence_diagnostic",
    "tp4_dp2_ep8",
    "1.321996",
    "holdout_faster",
    "boundary_timeline_explains_direction",
    "holdout_scheduled_max_fill",
)


def _read_single_row(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected exactly one input row: {path}")
    return rows[0]


def _require(row: dict[str, str], key: str, source: str) -> str:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"missing field for {source}: {key}")
    return value


def _validate_flags(row: dict[str, str], source: str) -> None:
    if (
        row.get("diagnostic_only") != "true"
        or row.get("valid_for_default") != "false"
        or row.get("perf_database") != "false"
    ):
        raise ValueError(f"flag mismatch: {source}")
    if row.get("default_readiness") != DEFAULT_READINESS:
        raise ValueError(f"default readiness mismatch: {source}")


def _direction(ratio: float) -> str:
    if ratio < 1.0:
        return "holdout_slower"
    if ratio > 1.0:
        return "holdout_faster"
    return "flat"


def _family_row(row: dict[str, str], spec: SourceSpec) -> dict[str, str]:
    if row.get("source") != spec.source:
        raise ValueError(f"source mismatch: {spec.source}")
    if row.get("topology_key") != spec.topology_key:
        raise ValueError(f"topology mismatch: {spec.source}")
    if row.get("shape_key") != "isl12000_osl2000_batch128":
        raise ValueError(f"shape mismatch: {spec.source}")
    if row.get("control_bt") != "12000" or row.get("holdout_bt") != "65536":
        raise ValueError(f"budget mismatch: {spec.source}")
    if row.get("verdict") != spec.verdict:
        raise ValueError(f"verdict mismatch: {spec.source}")
    _validate_flags(row, spec.source)

    ratio_text = _require(row, "output_ratio", spec.source)
    ratio = float(ratio_text)
    direction = _direction(ratio)
    if direction != spec.direction:
        raise ValueError(f"ratio direction mismatch: {spec.source}")
    if ratio_text != spec.expected_ratio:
        raise ValueError(f"ratio mismatch: {spec.source}")

    holdout_fill = _require(row, spec.fill_field, spec.source)
    if float(holdout_fill) >= 0.5:
        raise ValueError(f"budget ceiling guard mismatch: {spec.source}")

    return {
        "source": SOURCE,
        "family_key": "deeper_trace_12k2k_topology_family",
        "topology_key": row["topology_key"],
        "shape_key": row["shape_key"],
        "control_scenario": row["control_scenario"],
        "holdout_scenario": row["holdout_scenario"],
        "control_bt": row["control_bt"],
        "holdout_bt": row["holdout_bt"],
        "output_ratio": ratio_text,
        "throughput_direction": direction,
        "holdout_scheduled_p99": _require(row, "holdout_scheduled_p99", spec.source),
        "holdout_scheduled_max": _require(row, "holdout_scheduled_max", spec.source),
        "holdout_max_fill": holdout_fill,
        "budget_ceiling_rejected": "true",
        "verdict": row["verdict"],
        "mechanism_conclusion": _require(row, "mechanism_conclusion", spec.source),
        "default_readiness": DEFAULT_READINESS,
        "diagnostic_only": "true",
        "valid_for_default": "false",
        "perf_database": "false",
    }


def analyze_deeper_trace_topology_family(phase287_csv: Path, phase301_csv: Path) -> list[dict[str, str]]:
    """Return the two-row topology family audit from Phase287 and Phase301 CSVs."""
    rows = [
        _family_row(_read_single_row(phase287_csv), PHASE287_SPEC),
        _family_row(_read_single_row(phase301_csv), PHASE301_SPEC),
    ]
    directions = {row["throughput_direction"] for row in rows}
    if directions != {"holdout_slower", "holdout_faster"}:
        raise ValueError("topology family direction mismatch")
    return rows


def write_deeper_trace_topology_family_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_deeper_trace_topology_family_doc(path: Path, rows: list[dict[str, str]]) -> None:
    if len(rows) != 2:
        raise ValueError("Phase303 document expects exactly two rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    by_topology = {row["topology_key"]: row for row in rows}
    tp8 = by_topology["tp8_dp1_ep8"]
    tp4 = by_topology["tp4_dp2_ep8"]
    content = f"""# Phase303 Deeper Trace Topology Family

This family audit combines the Phase287 tp8 diagnostic and the Phase301 tp4dp2 diagnostic.

| Item | Value |
|---|---|
| Family | deeper_trace_12k2k_topology_family |
| tp8 ratio | {tp8["output_ratio"]} ({tp8["throughput_direction"]}) |
| tp4dp2 ratio | {tp4["output_ratio"]} ({tp4["throughput_direction"]}) |
| Budget ceiling rejected | true |
| Default AIC | No-Go |
| Diagnostic only | true |

Both topology traces reject configured max_num_batched_tokens as a linear runtime cost: the high-budget holdouts do not fill the aggregate budget. The throughput direction is topology-dependent: tp8 is slightly slower while tp4dp2 is clearly faster.

This means the evidence cannot support a global correction, interpolation, or extrapolation. The next modeling layer should stay topology-specific and focus on cadence and boundary mechanisms.

Flags: diagnostic_only=true, valid_for_default=false, perf_database=false.
"""
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase287", type=Path, default=DEFAULT_PHASE287)
    parser.add_argument("--phase301", type=Path, default=DEFAULT_PHASE301)
    parser.add_argument("--out", type=Path, default=DEFAULT_CSV_OUT)
    parser.add_argument("--doc-out", type=Path, default=DEFAULT_DOC_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = analyze_deeper_trace_topology_family(args.phase287, args.phase301)
    write_deeper_trace_topology_family_csv(args.out, rows)
    write_deeper_trace_topology_family_doc(args.doc_out, rows)
    print(f"wrote {args.out}")
    print(f"wrote {args.doc_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
