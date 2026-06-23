from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE337_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase337_prerun_feature_eligibility.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase339_prerun_model_feasibility.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase339_prerun_model_feasibility.md"
)

SOURCE = "phase339_prerun_model_feasibility"
PHASE337_SOURCE = "phase337_prerun_feature_eligibility"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"

FIELDNAMES = [
    "source",
    "candidate",
    "verdict",
    "reason",
    "gpu_run_allowed",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

EXPECTED_FEATURES = [
    "topology_key",
    "shape_key",
    "control_bt_holdout_bt",
    "actual_scheduled_tokens",
    "phase_mix_boundary_cadence",
]

SCOPE_KEY_DECISION = "scope_key_only_not_standalone_throughput_predictor"
POSTRUN_DECISION = "diagnostic_audit_only_default_prediction_leakage"


def _read_phase337_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _require(row: dict[str, str], field: str, expected: str, feature: str) -> None:
    actual = row.get(field)
    if actual != expected:
        raise ValueError(
            f"{feature} {field} expected {expected!r}, got {actual!r}"
        )


def _validate_phase337_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != len(EXPECTED_FEATURES):
        raise ValueError(f"Phase337 must have exactly {len(EXPECTED_FEATURES)} rows")

    features = [row.get("feature") for row in rows]
    if features != EXPECTED_FEATURES:
        raise ValueError(f"Phase337 feature order mismatch: {features!r}")

    for row in rows:
        feature = row["feature"]
        _require(row, "source", PHASE337_SOURCE, feature)
        _require(row, "default_readiness", DEFAULT_READINESS, feature)
        _require(row, "diagnostic_only", TRUE, feature)
        _require(row, "valid_for_default", FALSE, feature)
        _require(row, "perf_database", FALSE, feature)
        _require(row, "eligible_for_default_prediction", FALSE, feature)
        _require(row, "eligible_for_diagnostic_audit", TRUE, feature)

    for row in rows[:3]:
        feature = row["feature"]
        _require(row, "availability", "pre_run_available", feature)
        _require(row, "leakage_risk", FALSE, feature)
        _require(row, "decision", SCOPE_KEY_DECISION, feature)

    for row in rows[3:]:
        feature = row["feature"]
        _require(row, "availability", "post_run_trace_derived", feature)
        _require(row, "leakage_risk", TRUE, feature)
        _require(row, "decision", POSTRUN_DECISION, feature)


def analyze_prerun_model_feasibility(
    phase337_csv: Path = DEFAULT_PHASE337_CSV,
) -> list[dict[str, str]]:
    rows = _read_phase337_rows(phase337_csv)
    _validate_phase337_rows(rows)

    return [
        {
            "source": SOURCE,
            "candidate": "scope_key_only_candidate",
            "verdict": "rejected_scope_only_no_throughput_prediction",
            "reason": (
                "pre_run_scope_keys_only_define_exact_key_scope_and_"
                "cannot_predict_throughput"
            ),
            "gpu_run_allowed": FALSE,
            "default_readiness": DEFAULT_READINESS,
            "diagnostic_only": TRUE,
            "valid_for_default": FALSE,
            "perf_database": FALSE,
        },
        {
            "source": SOURCE,
            "candidate": "postrun_trace_feature_candidate",
            "verdict": "rejected_postrun_trace_leakage",
            "reason": (
                "post_run_trace_features_have_leakage_risk_and_cannot_be_"
                "default_prediction_inputs"
            ),
            "gpu_run_allowed": FALSE,
            "default_readiness": DEFAULT_READINESS,
            "diagnostic_only": TRUE,
            "valid_for_default": FALSE,
            "perf_database": FALSE,
        },
        {
            "source": SOURCE,
            "candidate": "topology_specific_cadence_boundary_candidate",
            "verdict": "diagnostic_only_postrun_explanation",
            "reason": (
                "actual_scheduled_tokens_and_phase_mix_boundary_cadence_are_"
                "post_run_only_and_not_a_default_model"
            ),
            "gpu_run_allowed": FALSE,
            "default_readiness": DEFAULT_READINESS,
            "diagnostic_only": TRUE,
            "valid_for_default": FALSE,
            "perf_database": FALSE,
        },
    ]


def _validate_output_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 3:
        raise ValueError(f"Phase339 output must have exactly 3 rows, got {len(rows)}")

    expected_candidates = [
        "scope_key_only_candidate",
        "postrun_trace_feature_candidate",
        "topology_specific_cadence_boundary_candidate",
    ]
    candidates = [row.get("candidate") for row in rows]
    if candidates != expected_candidates:
        raise ValueError(f"Phase339 candidate order mismatch: {candidates!r}")

    for row in rows:
        candidate = row["candidate"]
        _require(row, "source", SOURCE, candidate)
        _require(row, "gpu_run_allowed", FALSE, candidate)
        _require(row, "default_readiness", DEFAULT_READINESS, candidate)
        _require(row, "diagnostic_only", TRUE, candidate)
        _require(row, "valid_for_default", FALSE, candidate)
        _require(row, "perf_database", FALSE, candidate)


def write_prerun_model_feasibility_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_prerun_model_feasibility_doc(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    _validate_output_rows(rows)
    lines = [
        "# Phase339 Pre-Run-Only Model Feasibility",
        "",
        "| Item | Status |",
        "|---|---|",
        "| Default AIC | No-Go |",
        "| GPU run allowed | false |",
        "| Diagnostic only | true |",
        "| Valid for default | false |",
        "| PerfDatabase | false |",
        "",
        "Phase337 leaves only topology_key, shape_key, and control_bt_holdout_bt "
        "as pre-run inputs. Those pre-run inputs only define exact-key scope; "
        "they do not predict throughput.",
        "",
        "The useful explanatory fields, actual_scheduled_tokens and "
        "phase_mix_boundary_cadence, are post-run trace features. Using them "
        "for default prediction would leak the answer. In short, post-run "
        "trace features would leak the answer.",
        "",
        "| Candidate | Verdict | Reason |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {row['verdict']} | {row['reason']} |"
        )
    lines.extend(
        [
            "",
            "Conclusion: the topology-specific cadence/boundary path remains a "
            "diagnostic-only post-run explanation, not a default model.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether Phase337 pre-run inputs can form a default model."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_PHASE337_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = analyze_prerun_model_feasibility(args.input_csv)
    write_prerun_model_feasibility_csv(args.output_csv, rows)
    write_prerun_model_feasibility_doc(args.output_md, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
