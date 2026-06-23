#!/usr/bin/env python3
"""List default model form candidates and reject unsafe routes."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


SOURCE = "phase329_default_model_form_candidates"
DEFAULT_READINESS = "No-Go"
DEFAULT_PHASE327 = Path("docs/iter_gap_investigation/phase327_default_model_candidate_spec_audit.csv")
DEFAULT_CSV_OUT = Path("docs/iter_gap_investigation/phase329_default_model_form_candidates.csv")
DEFAULT_DOC_OUT = Path("docs/iter_gap_investigation/phase329_default_model_form_candidates.md")

FIELDNAMES = [
    "source",
    "candidate",
    "input_evidence",
    "model_form",
    "eligible_scope",
    "reject_reason",
    "required_next_validation",
    "candidate_status",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return rows


def _require_values(path: Path, rows: list[dict[str, str]], key: str, expected: set[str]) -> None:
    values = {row.get(key, "") for row in rows}
    if values != expected:
        raise ValueError(f"{path} {key} mismatch: {sorted(values)} expected={sorted(expected)}")


def _audit_phase327(path: Path) -> None:
    rows = _read_csv(path)
    if len(rows) != 5:
        raise ValueError(f"Phase327 expects exactly 5 rows, got {len(rows)}")
    _require_values(path, rows, "source", {"phase327_default_model_candidate_spec_audit"})
    _require_values(path, rows, "default_readiness", {DEFAULT_READINESS})
    _require_values(path, rows, "diagnostic_only", {"true"})
    _require_values(path, rows, "valid_for_default", {"false"})
    _require_values(path, rows, "perf_database", {"false"})
    modules = {row["module"] for row in rows}
    required = {
        "phase258_actual_scheduled_token_evidence",
        "phase303_deeper_trace_topology_evidence",
        "phase305_phase309_diagnostic_exact_key_api",
        "phase323_runner_lifecycle_canary",
        "default_model_candidate_gate",
    }
    if modules != required:
        raise ValueError(f"Phase327 module mismatch: {sorted(modules)} expected={sorted(required)}")
    gate = [row for row in rows if row["module"] == "default_model_candidate_gate"][0]
    if gate["verdict"] != "default_model_candidate_spec_gap":
        raise ValueError(f"default_model_candidate_gate verdict mismatch: {gate['verdict']}")


def _candidate(
    *,
    candidate: str,
    input_evidence: str,
    model_form: str,
    eligible_scope: str,
    reject_reason: str,
    required_next_validation: str,
    candidate_status: str,
) -> dict[str, str]:
    return {
        "source": SOURCE,
        "candidate": candidate,
        "input_evidence": input_evidence,
        "model_form": model_form,
        "eligible_scope": eligible_scope,
        "reject_reason": reject_reason,
        "required_next_validation": required_next_validation,
        "candidate_status": candidate_status,
        "default_readiness": DEFAULT_READINESS,
        "diagnostic_only": "true",
        "valid_for_default": "false",
        "perf_database": "false",
    }


def analyze_default_model_form_candidates(phase327_csv: Path) -> list[dict[str, str]]:
    _audit_phase327(phase327_csv)
    rows = [
        _candidate(
            candidate="global_correction",
            input_evidence="phase303_deeper_trace_topology_family",
            model_form="single_multiplier_or_global_penalty",
            eligible_scope="none",
            reject_reason="Phase303 topology directions conflict: tp8 slower while tp4dp2 faster",
            required_next_validation="none_route_rejected",
            candidate_status="rejected",
        ),
        _candidate(
            candidate="budget_ceiling_model",
            input_evidence="phase258_actual_scheduled_tokens_and_phase303_deeper_trace",
            model_form="configured_max_num_batched_tokens_as_linear_cost",
            eligible_scope="none",
            reject_reason="Phase258 and Phase303 reject configured budget ceiling as actual cost",
            required_next_validation="none_route_rejected",
            candidate_status="rejected",
        ),
        _candidate(
            candidate="diagnostic_lookup_as_model",
            input_evidence="phase305_phase309_exact_key_api",
            model_form="reuse_exact_key_lookup_for_default_prediction",
            eligible_scope="none",
            reject_reason="diagnostic lookup is not a prediction API",
            required_next_validation="none_route_rejected",
            candidate_status="rejected",
        ),
        _candidate(
            candidate="lifecycle_canary_as_evidence",
            input_evidence="phase323_runner_lifecycle_pair_canary",
            model_form="treat_worker_canary_ratio_as_model_evidence",
            eligible_scope="none",
            reject_reason="Phase323 proves runner lifecycle only, not model behavior",
            required_next_validation="none_route_rejected",
            candidate_status="rejected",
        ),
        _candidate(
            candidate="topology_specific_cadence_boundary_candidate",
            input_evidence="phase258_phase303_phase327",
            model_form="topology_shape_budget_exact_model_with_cadence_boundary_features",
            eligible_scope="future_exact_topology_shape_budget_keys_only",
            reject_reason="",
            required_next_validation="independent_topology_specific_holdout_with_error_threshold",
            candidate_status="candidate_needs_more_evidence",
        ),
    ]
    if len(rows) != 5:
        raise ValueError("Phase329 expects exactly 5 candidate rows")
    return rows


def write_default_model_form_candidates_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if len(rows) != 5:
        raise ValueError("Phase329 CSV expects exactly 5 rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_default_model_form_candidates_doc(path: Path, rows: list[dict[str, str]]) -> None:
    if len(rows) != 5:
        raise ValueError("Phase329 document expects exactly 5 rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    by_candidate = {row["candidate"]: row for row in rows}
    kept = by_candidate["topology_specific_cadence_boundary_candidate"]
    table = "\n".join(
        f"| {row['candidate']} | {row['model_form']} | {row['candidate_status']} | "
        f"{row['reject_reason'] or row['required_next_validation']} |"
        for row in rows
    )
    content = f"""# Phase329 Default Model Form Candidates

This diagnostic audit lists possible default model candidate forms and rejects routes that are already contradicted by committed evidence.

| Item | Value |
|---|---|
| Default AIC | No-Go |
| PerfDatabase | No-Go |
| diagnostic_only | true |
| valid_for_default | false |
| perf_database | false |
| retained candidate | {kept["candidate"]} |
| retained status | {kept["candidate_status"]} |

| Candidate | Model form | Status | Reason or next validation |
|---|---|---|---|
{table}

The global correction is rejected because Phase303 has opposite throughput directions across topology. The budget ceiling model is rejected because Phase258 and Phase303 show that configured max_num_batched_tokens is a ceiling, not actual scheduled-token cost. The diagnostic exact-key lookup cannot be reused as a default prediction API. Phase323 is runner lifecycle evidence only.

Only topology_specific_cadence_boundary_candidate remains as a future diagnostic candidate, and it still needs independent topology-specific holdout validation plus a defined error threshold before any default AIC discussion. Default readiness remains No-Go.
"""
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase327", type=Path, default=DEFAULT_PHASE327)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV_OUT)
    parser.add_argument("--doc-out", type=Path, default=DEFAULT_DOC_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = analyze_default_model_form_candidates(args.phase327)
    write_default_model_form_candidates_csv(args.csv_out, rows)
    write_default_model_form_candidates_doc(args.doc_out, rows)


if __name__ == "__main__":
    main()
