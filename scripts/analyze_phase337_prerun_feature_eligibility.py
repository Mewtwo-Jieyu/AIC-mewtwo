from __future__ import annotations

import argparse
import csv
from pathlib import Path


SOURCE = "phase337_prerun_feature_eligibility"
PHASE335_SOURCE = "phase335_model_output_threshold_contract"
DEFAULT_READINESS = "No-Go"

DEFAULT_PHASE335_CSV = Path(
    "docs/iter_gap_investigation/phase335_model_output_threshold_contract.csv"
)
DEFAULT_OUTPUT_CSV = Path(
    "docs/iter_gap_investigation/phase337_prerun_feature_eligibility.csv"
)
DEFAULT_OUTPUT_MD = Path(
    "docs/iter_gap_investigation/phase337_prerun_feature_eligibility.md"
)

FIELDNAMES = [
    "source",
    "feature",
    "availability",
    "eligible_for_default_prediction",
    "eligible_for_diagnostic_audit",
    "leakage_risk",
    "decision",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

EXPECTED_CONTRACTS = [
    "direction_prediction_contract",
    "cadence_boundary_feature_contract",
    "numeric_error_threshold_contract",
]

EXPECTED_FLAGS = {
    "source": PHASE335_SOURCE,
    "gpu_run_allowed": "false",
    "default_readiness": DEFAULT_READINESS,
    "diagnostic_only": "true",
    "valid_for_default": "false",
    "perf_database": "false",
}

REGISTERED_TRACE_INPUTS = {
    "actual_scheduled_tokens",
    "phase_mix",
    "boundary_cadence",
    "trace_integrity",
    "worker_payload_alignment",
}

COMMON_FLAGS = {
    "default_readiness": DEFAULT_READINESS,
    "diagnostic_only": "true",
    "valid_for_default": "false",
    "perf_database": "false",
}

FEATURE_ROWS = [
    {
        "source": SOURCE,
        "feature": "topology_key",
        "availability": "pre_run_available",
        "eligible_for_default_prediction": "false",
        "eligible_for_diagnostic_audit": "true",
        "leakage_risk": "false",
        "decision": "scope_key_only_not_standalone_throughput_predictor",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "feature": "shape_key",
        "availability": "pre_run_available",
        "eligible_for_default_prediction": "false",
        "eligible_for_diagnostic_audit": "true",
        "leakage_risk": "false",
        "decision": "scope_key_only_not_standalone_throughput_predictor",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "feature": "control_bt_holdout_bt",
        "availability": "pre_run_available",
        "eligible_for_default_prediction": "false",
        "eligible_for_diagnostic_audit": "true",
        "leakage_risk": "false",
        "decision": "scope_key_only_not_standalone_throughput_predictor",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "feature": "actual_scheduled_tokens",
        "availability": "post_run_trace_derived",
        "eligible_for_default_prediction": "false",
        "eligible_for_diagnostic_audit": "true",
        "leakage_risk": "true",
        "decision": "diagnostic_audit_only_default_prediction_leakage",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "feature": "phase_mix_boundary_cadence",
        "availability": "post_run_trace_derived",
        "eligible_for_default_prediction": "false",
        "eligible_for_diagnostic_audit": "true",
        "leakage_risk": "true",
        "decision": "diagnostic_audit_only_default_prediction_leakage",
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


def _split_inputs(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def _require_phase335_contract(path: Path) -> None:
    rows = _read_csv(path)
    if [row.get("contract") for row in rows] != EXPECTED_CONTRACTS:
        raise ValueError("Phase335 contract rows changed")

    for row in rows:
        for field, expected in EXPECTED_FLAGS.items():
            if row.get(field) != expected:
                raise ValueError(
                    f"Phase335 {field}={row.get(field)}, expected {expected}"
                )

    feature_contract = rows[1]
    missing = REGISTERED_TRACE_INPUTS - _split_inputs(feature_contract.get("required_inputs", ""))
    if missing:
        raise ValueError(
            "Phase335 required_inputs missing registered trace feature(s): "
            + ", ".join(sorted(missing))
        )

    numeric_contract = rows[2]
    if numeric_contract.get("threshold_status") != "blocked_until_numeric_model_form_exists":
        raise ValueError("Phase335 numeric threshold must stay blocked")


def analyze_prerun_feature_eligibility(
    phase335_csv: Path = DEFAULT_PHASE335_CSV,
) -> list[dict[str, str]]:
    _require_phase335_contract(phase335_csv)
    return [dict(row) for row in FEATURE_ROWS]


def _require_five_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 5:
        raise ValueError(f"Phase337 output must contain exactly 5 rows, got {len(rows)}")


def write_prerun_feature_eligibility_csv(
    output_path: Path,
    rows: list[dict[str, str]],
) -> None:
    _require_five_rows(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_prerun_feature_eligibility_doc(
    output_path: Path,
    rows: list[dict[str, str]],
) -> None:
    _require_five_rows(rows)
    lines = [
        "# Phase337 Pre-Run Feature Eligibility",
        "",
        "| Item | Decision |",
        "|---|---|",
        "| Default AIC | No-Go |",
        "| Runtime integration | Not allowed in this phase |",
        "| PerfDatabase | Not written |",
        "",
        "The scope keys only define exact-key coverage. They are not standalone throughput predictors.",
        "",
        "post-run trace features cannot be default prediction inputs. They are diagnostic audit evidence only.",
        "",
        "| Feature | Availability | Default prediction | Leakage risk |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {feature} | {availability} | {eligible_for_default_prediction} | {leakage_risk} |".format(
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
        description="Build Phase337 pre-run feature eligibility artifacts."
    )
    parser.add_argument("--phase335-csv", type=Path, default=DEFAULT_PHASE335_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = analyze_prerun_feature_eligibility(args.phase335_csv)
    write_prerun_feature_eligibility_csv(args.output_csv, rows)
    write_prerun_feature_eligibility_doc(args.output_md, rows)


if __name__ == "__main__":
    main()
