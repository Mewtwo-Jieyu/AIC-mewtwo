from __future__ import annotations

import argparse
import csv
from pathlib import Path


SOURCE = "phase335_model_output_threshold_contract"
PHASE333_SOURCE = "phase333_topology_holdout_matrix_spec"
CANDIDATE = "topology_specific_cadence_boundary_candidate"
DEFAULT_READINESS = "No-Go"

DEFAULT_PHASE333_CSV = Path(
    "docs/iter_gap_investigation/phase333_topology_holdout_matrix_spec.csv"
)
DEFAULT_OUTPUT_CSV = Path(
    "docs/iter_gap_investigation/phase335_model_output_threshold_contract.csv"
)
DEFAULT_OUTPUT_MD = Path(
    "docs/iter_gap_investigation/phase335_model_output_threshold_contract.md"
)

FIELDNAMES = [
    "source",
    "contract",
    "model_output",
    "required_inputs",
    "pass_rule",
    "fail_rule",
    "threshold_status",
    "gpu_run_allowed",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

EXPECTED_PHASE333_VALUES = {
    "source": PHASE333_SOURCE,
    "candidate": CANDIDATE,
    "error_threshold": "must_be_defined_before_gpu_run",
    "run_status": "not_run",
    "default_readiness": DEFAULT_READINESS,
    "diagnostic_only": "true",
    "valid_for_default": "false",
    "perf_database": "false",
}

EXPECTED_PHASE333_PAIRS = [
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

COMMON_FLAGS = {
    "gpu_run_allowed": "false",
    "default_readiness": DEFAULT_READINESS,
    "diagnostic_only": "true",
    "valid_for_default": "false",
    "perf_database": "false",
}

CONTRACT_ROWS = [
    {
        "source": SOURCE,
        "contract": "direction_prediction_contract",
        "model_output": "holdout_faster_or_holdout_slower_or_inconclusive",
        "required_inputs": (
            "actual_scheduled_tokens,phase_mix,boundary_cadence,"
            "trace_integrity,worker_payload_alignment"
        ),
        "pass_rule": "all_4_matrix_pairs_have_reproducible_direction_explanation",
        "fail_rule": "any_pair_direction_conflict_unresolved",
        "threshold_status": "direction_only_contract_registered",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "contract": "cadence_boundary_feature_contract",
        "model_output": "cadence_boundary_direction_features",
        "required_inputs": (
            "actual_scheduled_tokens,phase_mix,boundary_cadence,"
            "trace_integrity,worker_payload_alignment"
        ),
        "pass_rule": "uses_only_phase333_registered_features",
        "fail_rule": "posthoc_feature_introduced",
        "threshold_status": "feature_set_registered_no_posthoc_features",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "contract": "numeric_error_threshold_contract",
        "model_output": "numeric_error_threshold",
        "required_inputs": "numeric_prediction_formula",
        "pass_rule": "numeric_formula_defined_before_gpu_run",
        "fail_rule": "no_numeric_formula_or_threshold_changed_after_gpu_run",
        "threshold_status": "blocked_until_numeric_model_form_exists",
        **COMMON_FLAGS,
    },
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return rows


def _require_phase333_matrix(path: Path) -> None:
    rows = _read_csv(path)
    if len(rows) != 4:
        raise ValueError(f"Phase333 matrix must have exactly 4 rows, got {len(rows)}")

    observed_pairs = [
        (
            row.get("topology_key", ""),
            row.get("shape_key", ""),
            row.get("control_scenario", ""),
            row.get("holdout_scenario", ""),
        )
        for row in rows
    ]
    if observed_pairs != EXPECTED_PHASE333_PAIRS:
        raise ValueError("Phase333 topology holdout matrix changed")

    for row in rows:
        for field, expected in EXPECTED_PHASE333_VALUES.items():
            if row.get(field) != expected:
                raise ValueError(
                    f"Phase333 {field}={row.get(field)}, expected {expected}"
                )


def analyze_model_output_threshold_contract(
    phase333_csv: Path = DEFAULT_PHASE333_CSV,
) -> list[dict[str, str]]:
    _require_phase333_matrix(phase333_csv)
    return [dict(row) for row in CONTRACT_ROWS]


def _require_three_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 3:
        raise ValueError(f"Phase335 output must contain exactly 3 rows, got {len(rows)}")


def write_model_output_threshold_contract_csv(
    output_path: Path,
    rows: list[dict[str, str]],
) -> None:
    _require_three_rows(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_model_output_threshold_contract_doc(
    output_path: Path,
    rows: list[dict[str, str]],
) -> None:
    _require_three_rows(rows)
    lines = [
        "# Phase335 Model Output And Threshold Contract",
        "",
        "| Item | Decision |",
        "|---|---|",
        f"| Candidate | {CANDIDATE} |",
        "| Default AIC | No-Go |",
        "| GPU run allowed | false |",
        "| Runtime integration | Not allowed in this phase |",
        "| PerfDatabase | Not written |",
        "",
        "The direction output is limited to holdout_faster / holdout_slower / inconclusive.",
        "",
        "No posthoc feature is allowed. The feature set is limited to Phase333 registered inputs: actual_scheduled_tokens, phase_mix, boundary_cadence, trace_integrity, and worker_payload_alignment.",
        "",
        "The numeric threshold remains blocked_until_numeric_model_form_exists because there is no numeric prediction formula yet.",
        "",
        "| Contract | Model output | Threshold status |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['contract']} | {row['model_output']} | {row['threshold_status']} |"
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
        description="Build Phase335 model output and threshold contract artifacts."
    )
    parser.add_argument("--phase333-csv", type=Path, default=DEFAULT_PHASE333_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = analyze_model_output_threshold_contract(args.phase333_csv)
    write_model_output_threshold_contract_csv(args.output_csv, rows)
    write_model_output_threshold_contract_doc(args.output_md, rows)


if __name__ == "__main__":
    main()
