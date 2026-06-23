from __future__ import annotations

import argparse
import csv
from pathlib import Path


SOURCE = "phase333_topology_holdout_matrix_spec"
PHASE331_SOURCE = "phase331_topology_candidate_validation_gate"
CANDIDATE = "topology_specific_cadence_boundary_candidate"
DEFAULT_READINESS = "No-Go"

DEFAULT_PHASE331_CSV = Path(
    "docs/iter_gap_investigation/phase331_topology_candidate_validation_gate.csv"
)
DEFAULT_OUTPUT_CSV = Path(
    "docs/iter_gap_investigation/phase333_topology_holdout_matrix_spec.csv"
)
DEFAULT_OUTPUT_MD = Path(
    "docs/iter_gap_investigation/phase333_topology_holdout_matrix_spec.md"
)

FIELDNAMES = [
    "source",
    "candidate",
    "topology_key",
    "shape_key",
    "control_scenario",
    "holdout_scenario",
    "required_features",
    "pass_condition",
    "fail_condition",
    "error_threshold",
    "run_status",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

MATRIX = [
    (
        "tp8_dp1_ep8",
        "isl12000_osl2000_batch128",
        "tp8ep8-12k2k-bt12000",
        "tp8ep8-12k2k-bt65536",
    ),
    (
        "tp4_dp2_ep8",
        "isl12000_osl2000_batch128",
        "tp4dp2ep8-12k2k-bt12000",
        "tp4dp2ep8-12k2k-bt65536",
    ),
    (
        "tp8_dp1_ep8",
        "isl4000_osl2000_batch128",
        "tp8ep8-4k2k-bt12000",
        "tp8ep8-4k2k-bt65536",
    ),
    (
        "tp4_dp2_ep8",
        "isl4000_osl2000_batch128",
        "tp4dp2ep8-4k2k-bt12000",
        "tp4dp2ep8-4k2k-bt65536",
    ),
]

COMMON_VALUES = {
    "required_features": (
        "trace_integrity,worker_payload_alignment,phase_mix,"
        "boundary_cadence,actual_scheduled_tokens"
    ),
    "pass_condition": (
        "complete_trace_and_aligned_worker_rows_and_required_phase_cadence_boundary_fields_"
        "and_reproducible_direction_judgment"
    ),
    "fail_condition": (
        "missing_artifact_or_worker_payload_divergence_or_posthoc_threshold_or_"
        "unknown_key_or_direction_conflict_unresolved"
    ),
    "error_threshold": "must_be_defined_before_gpu_run",
    "run_status": "not_run",
    "default_readiness": DEFAULT_READINESS,
    "diagnostic_only": "true",
    "valid_for_default": "false",
    "perf_database": "false",
}

EXPECTED_PHASE331 = {
    "source": PHASE331_SOURCE,
    "candidate": CANDIDATE,
    "promotion_status": "blocked_pending_validation",
    "default_readiness": DEFAULT_READINESS,
    "diagnostic_only": "true",
    "valid_for_default": "false",
    "perf_database": "false",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return rows


def _require_phase331_gate(path: Path) -> None:
    rows = _read_csv(path)
    if len(rows) != 1:
        raise ValueError(f"Phase331 gate must have exactly 1 row, got {len(rows)}")
    row = rows[0]
    for field, expected in EXPECTED_PHASE331.items():
        if row.get(field) != expected:
            raise ValueError(
                f"Phase331 gate {field}={row.get(field)}, expected {expected}"
            )


def analyze_topology_holdout_matrix_spec(
    phase331_csv: Path = DEFAULT_PHASE331_CSV,
) -> list[dict[str, str]]:
    _require_phase331_gate(phase331_csv)
    rows: list[dict[str, str]] = []
    for topology_key, shape_key, control_scenario, holdout_scenario in MATRIX:
        rows.append(
            {
                "source": SOURCE,
                "candidate": CANDIDATE,
                "topology_key": topology_key,
                "shape_key": shape_key,
                "control_scenario": control_scenario,
                "holdout_scenario": holdout_scenario,
                **COMMON_VALUES,
            }
        )
    return rows


def _require_four_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 4:
        raise ValueError(f"Phase333 output must contain exactly 4 rows, got {len(rows)}")


def write_topology_holdout_matrix_spec_csv(
    output_path: Path,
    rows: list[dict[str, str]],
) -> None:
    _require_four_rows(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_topology_holdout_matrix_spec_doc(
    output_path: Path,
    rows: list[dict[str, str]],
) -> None:
    _require_four_rows(rows)
    lines = [
        "# Phase333 Topology Holdout Matrix Spec",
        "",
        "| Item | Decision |",
        "|---|---|",
        f"| Candidate | {CANDIDATE} |",
        "| Default AIC | No-Go |",
        "| Runtime integration | Not allowed in this phase |",
        "| PerfDatabase | Not written |",
        "| Run status | not_run |",
        "",
        "This file defines the next validation matrix only. It does not run any scenario and does not create evidence artifacts.",
        "",
        "The error threshold is must_be_defined_before_gpu_run. That label is not a pass threshold and cannot be treated as validation success.",
        "",
        "Failure is explicit: missing_artifact, worker_payload_divergence, posthoc_threshold, unknown_key, or direction_conflict_unresolved.",
        "",
        "| Topology | Shape | Control | Holdout |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {topology_key} | {shape_key} | {control_scenario} | {holdout_scenario} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "| Flag | Value |",
            "|---|---|",
            "| diagnostic_only | true |",
            "| valid_for_default | false |",
            "| perf_database | false |",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Phase333 topology holdout matrix spec artifacts."
    )
    parser.add_argument("--phase331-csv", type=Path, default=DEFAULT_PHASE331_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = analyze_topology_holdout_matrix_spec(args.phase331_csv)
    write_topology_holdout_matrix_spec_csv(args.output_csv, rows)
    write_topology_holdout_matrix_spec_doc(args.output_md, rows)


if __name__ == "__main__":
    main()
