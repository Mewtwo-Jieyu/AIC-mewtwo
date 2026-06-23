from __future__ import annotations

import argparse
import csv
from pathlib import Path


SOURCE = "phase331_topology_candidate_validation_gate"
PHASE329_SOURCE = "phase329_default_model_form_candidates"
RETAINED_CANDIDATE = "topology_specific_cadence_boundary_candidate"
RETAINED_STATUS = "candidate_needs_more_evidence"
REJECTED_STATUS = "rejected"
DEFAULT_READINESS = "No-Go"

DEFAULT_PHASE329_CSV = Path(
    "docs/iter_gap_investigation/phase329_default_model_form_candidates.csv"
)
DEFAULT_OUTPUT_CSV = Path(
    "docs/iter_gap_investigation/phase331_topology_candidate_validation_gate.csv"
)
DEFAULT_OUTPUT_MD = Path(
    "docs/iter_gap_investigation/phase331_topology_candidate_validation_gate.md"
)

FIELDNAMES = [
    "source",
    "candidate",
    "required_input_features",
    "eligible_scope",
    "required_holdout_matrix",
    "error_threshold_policy",
    "failure_policy",
    "promotion_target",
    "promotion_status",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

EXPECTED_PHASE329_FLAGS = {
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


def _require_phase329_contract(rows: list[dict[str, str]]) -> dict[str, str]:
    for row in rows:
        if row.get("source") != PHASE329_SOURCE:
            raise ValueError(f"unexpected Phase329 source: {row.get('source')}")
        for field, expected in EXPECTED_PHASE329_FLAGS.items():
            if row.get(field) != expected:
                raise ValueError(
                    f"Phase329 {row.get('candidate')} has {field}={row.get(field)}, "
                    f"expected {expected}"
                )

    retained = [row for row in rows if row.get("candidate") == RETAINED_CANDIDATE]
    if len(retained) != 1:
        raise ValueError(f"expected exactly one {RETAINED_CANDIDATE} row")

    if len(rows) != 5:
        raise ValueError(f"Phase329 candidate matrix must have exactly 5 rows, got {len(rows)}")

    retained_row = retained[0]
    if retained_row.get("candidate_status") != RETAINED_STATUS:
        raise ValueError(
            f"{RETAINED_CANDIDATE} must stay {RETAINED_STATUS}, "
            f"got {retained_row.get('candidate_status')}"
        )
    if retained_row.get("eligible_scope") != "future_exact_topology_shape_budget_keys_only":
        raise ValueError("retained candidate eligible_scope drifted")
    if (
        retained_row.get("required_next_validation")
        != "independent_topology_specific_holdout_with_error_threshold"
    ):
        raise ValueError("retained candidate required_next_validation drifted")

    revived = [
        row.get("candidate", "")
        for row in rows
        if row.get("candidate") != RETAINED_CANDIDATE
        and row.get("candidate_status") != REJECTED_STATUS
    ]
    if revived:
        raise ValueError(f"rejected routes revived as candidates: {', '.join(revived)}")

    return retained_row


def analyze_topology_candidate_validation_gate(
    phase329_csv: Path = DEFAULT_PHASE329_CSV,
) -> list[dict[str, str]]:
    _require_phase329_contract(_read_csv(phase329_csv))
    return [
        {
            "source": SOURCE,
            "candidate": RETAINED_CANDIDATE,
            "required_input_features": (
                "topology_key,shape_key,control_bt,holdout_bt,"
                "actual_scheduled_tokens,phase_mix,boundary_cadence"
            ),
            "eligible_scope": "topology_shape_budget_exact_key_only",
            "required_holdout_matrix": (
                "independent_holdouts_covering_tp8_dp1_ep8_and_"
                "tp4_dp2_ep8_opposite_directions"
            ),
            "error_threshold_policy": "define_before_running_holdout_no_posthoc_threshold",
            "failure_policy": "fail_fast_no_interpolation_no_extrapolation",
            "promotion_target": "model_experiment_candidate_only",
            "promotion_status": "blocked_pending_validation",
            "default_readiness": DEFAULT_READINESS,
            "diagnostic_only": "true",
            "valid_for_default": "false",
            "perf_database": "false",
        }
    ]


def _require_single_gate_row(rows: list[dict[str, str]]) -> dict[str, str]:
    if len(rows) != 1:
        raise ValueError(f"Phase331 output must contain exactly 1 row, got {len(rows)}")
    return rows[0]


def write_topology_candidate_validation_gate_csv(
    output_path: Path,
    rows: list[dict[str, str]],
) -> None:
    _require_single_gate_row(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_topology_candidate_validation_gate_doc(
    output_path: Path,
    rows: list[dict[str, str]],
) -> None:
    row = _require_single_gate_row(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                "# Phase331 Topology Candidate Validation Gate",
                "",
                "| Item | Decision |",
                "|---|---|",
                f"| Candidate | {row['candidate']} |",
                f"| Promotion status | {row['promotion_status']} |",
                "| Default AIC | No-Go |",
                "| Runtime integration | Not allowed in this phase |",
                "| PerfDatabase | Not written |",
                "",
                "The retained route is still a diagnostic-only candidate. It can only move toward a model experiment after an independent holdout matrix covers both opposite-direction topologies.",
                "",
                "The error threshold must be defined before running holdout. It cannot be adjusted after looking at outcomes.",
                "",
                "No interpolation or extrapolation is allowed. Unknown topology, shape, or budget keys must fail fast.",
                "",
                "Rejected routes remain rejected and are not listed as live candidates in this gate.",
                "",
                "| Flag | Value |",
                "|---|---|",
                f"| diagnostic_only | {row['diagnostic_only']} |",
                f"| valid_for_default | {row['valid_for_default']} |",
                f"| perf_database | {row['perf_database']} |",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Phase331 topology candidate validation gate artifacts."
    )
    parser.add_argument("--phase329-csv", type=Path, default=DEFAULT_PHASE329_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = analyze_topology_candidate_validation_gate(args.phase329_csv)
    write_topology_candidate_validation_gate_csv(args.output_csv, rows)
    write_topology_candidate_validation_gate_doc(args.output_md, rows)


if __name__ == "__main__":
    main()
