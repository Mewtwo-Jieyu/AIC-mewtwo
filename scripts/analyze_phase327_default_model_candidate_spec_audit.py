#!/usr/bin/env python3
"""Audit the gap between diagnostic evidence and default model readiness."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


SOURCE = "phase327_default_model_candidate_spec_audit"
DEFAULT_READINESS = "No-Go"
DEFAULT_PHASE258 = Path("docs/iter_gap_investigation/phase258_actual_scheduled_token_full_family.csv")
DEFAULT_PHASE303 = Path("docs/iter_gap_investigation/phase303_deeper_trace_topology_family.csv")
DEFAULT_PHASE309 = Path("docs/iter_gap_investigation/phase309_diagnostic_exact_key_api_consistency.csv")
DEFAULT_PHASE323 = Path("docs/iter_gap_investigation/phase323_runner_lifecycle_pair_canary.csv")
DEFAULT_CSV_OUT = Path("docs/iter_gap_investigation/phase327_default_model_candidate_spec_audit.csv")
DEFAULT_DOC_OUT = Path("docs/iter_gap_investigation/phase327_default_model_candidate_spec_audit.md")

FIELDNAMES = [
    "source",
    "module",
    "candidate_input",
    "candidate_output",
    "eligible_keys",
    "reject_keys",
    "required_holdout",
    "error_threshold",
    "failure_policy",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
    "verdict",
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


def _require_common_no_go(path: Path, rows: list[dict[str, str]]) -> None:
    _require_values(path, rows, "default_readiness", {DEFAULT_READINESS})
    _require_values(path, rows, "diagnostic_only", {"true"})
    _require_values(path, rows, "valid_for_default", {"false"})
    _require_values(path, rows, "perf_database", {"false"})


def _audit_phase258(path: Path) -> dict[str, str]:
    rows = _read_csv(path)
    if len(rows) != 8:
        raise ValueError(f"Phase258 expects 8 rows, got {len(rows)}")
    _require_values(path, rows, "source", {"phase258_actual_scheduled_token_full_family"})
    _require_common_no_go(path, rows)
    _require_values(path, rows, "mechanism_hypothesis", {"actual_scheduled_tokens_not_configured_budget"})
    topologies = {row["topology_key"] for row in rows}
    shapes = {row["shape_key"] for row in rows}
    if topologies != {"tp8_dp1_ep8", "tp4_dp2_ep8"}:
        raise ValueError(f"Phase258 topology mismatch: {sorted(topologies)}")
    if shapes != {"isl4000_osl2000_batch128", "isl12000_osl2000_batch128"}:
        raise ValueError(f"Phase258 shape mismatch: {sorted(shapes)}")
    return _row(
        module="phase258_actual_scheduled_token_evidence",
        candidate_input="actual_scheduled_tokens_phase_family",
        candidate_output="rank_sum_scheduled_token_budget_fill_evidence",
        eligible_keys="4_exact_pairs_4k2k_12k2k_tp8_tp4dp2",
        reject_keys="unknown_topology_shape_budget_or_missing_pair",
        required_holdout="independent_holdout_required_before_default_model_use",
        error_threshold="not_defined_for_default_aic",
        failure_policy="diagnostic_evidence_only_no_prediction",
        verdict="evidence_source_only",
    )


def _audit_phase303(path: Path) -> dict[str, str]:
    rows = _read_csv(path)
    if len(rows) != 2:
        raise ValueError(f"Phase303 expects 2 rows, got {len(rows)}")
    _require_values(path, rows, "source", {"phase303_deeper_trace_topology_family"})
    _require_common_no_go(path, rows)
    _require_values(path, rows, "budget_ceiling_rejected", {"true"})
    directions = {row["throughput_direction"] for row in rows}
    if directions != {"holdout_slower", "holdout_faster"}:
        raise ValueError(f"Phase303 must remain topology-dependent, got {sorted(directions)}")
    return _row(
        module="phase303_deeper_trace_topology_evidence",
        candidate_input="deeper_trace_topology_family",
        candidate_output="topology_dependent_cadence_boundary_evidence",
        eligible_keys="2_exact_12k2k_topology_pairs",
        reject_keys="global_correction_interpolation_extrapolation",
        required_holdout="additional_topology_specific_holdout_before_modeling",
        error_threshold="not_defined_for_default_aic",
        failure_policy="keep_topology_dependent_evidence_separate",
        verdict="evidence_source_only",
    )


def _audit_phase309(path: Path) -> dict[str, str]:
    rows = _read_csv(path)
    if len(rows) != 2:
        raise ValueError(f"Phase309 expects 2 rows, got {len(rows)}")
    _require_values(path, rows, "source", {"phase309_diagnostic_exact_key_api_consistency"})
    _require_values(path, rows, "api_family", {"actual_scheduled_token_family", "deeper_trace_topology_family"})
    _require_values(path, rows, "unknown_key_rejects", {"true"})
    _require_values(path, rows, "diagnostic_only_all", {"true"})
    _require_values(path, rows, "valid_for_default_any", {"false"})
    _require_values(path, rows, "perf_database_any", {"false"})
    _require_values(path, rows, "default_readiness", {DEFAULT_READINESS})
    return _row(
        module="phase305_phase309_diagnostic_exact_key_api",
        candidate_input="fixed_evidence_csv_rows",
        candidate_output="diagnostic_exact_key_lookup_only",
        eligible_keys="4_actual_scheduled_keys_plus_2_deeper_trace_keys",
        reject_keys="unknown_key_must_raise_keyerror",
        required_holdout="not_a_holdout_model_api",
        error_threshold="not_applicable_to_lookup_api",
        failure_policy="fail_fast_no_interpolation_no_extrapolation",
        verdict="lookup_api_not_prediction_api",
    )


def _audit_phase323(path: Path) -> dict[str, str]:
    rows = _read_csv(path)
    if len(rows) != 1:
        raise ValueError(f"Phase323 expects 1 row, got {len(rows)}")
    _require_values(path, rows, "source", {"phase323_runner_lifecycle_pair_canary"})
    _require_values(path, rows, "diagnostic_only", {"true"})
    _require_values(path, rows, "valid_for_default", {"false"})
    _require_values(path, rows, "perf_database", {"false"})
    _require_values(path, rows, "evidence_status", {"not_replacement_evidence"})
    _require_values(path, rows, "verdict", {"runner_lifecycle_pair_canary_pass"})
    return _row(
        module="phase323_runner_lifecycle_canary",
        candidate_input="worker_wrzh8_control_holdout_lifecycle_canary",
        candidate_output="runner_lifecycle_closure_evidence",
        eligible_keys="none_for_modeling",
        reject_keys="all_default_model_or_evidence_replacement_use",
        required_holdout="not_model_holdout_evidence",
        error_threshold="not_applicable_to_lifecycle_canary",
        failure_policy="do_not_update_phase303_phase309_or_default_model",
        verdict="lifecycle_canary_not_model_evidence",
    )


def _row(
    *,
    module: str,
    candidate_input: str,
    candidate_output: str,
    eligible_keys: str,
    reject_keys: str,
    required_holdout: str,
    error_threshold: str,
    failure_policy: str,
    verdict: str,
) -> dict[str, str]:
    return {
        "source": SOURCE,
        "module": module,
        "candidate_input": candidate_input,
        "candidate_output": candidate_output,
        "eligible_keys": eligible_keys,
        "reject_keys": reject_keys,
        "required_holdout": required_holdout,
        "error_threshold": error_threshold,
        "failure_policy": failure_policy,
        "default_readiness": DEFAULT_READINESS,
        "diagnostic_only": "true",
        "valid_for_default": "false",
        "perf_database": "false",
        "verdict": verdict,
    }


def analyze_default_model_candidate_spec_audit(
    phase258_csv: Path,
    phase303_csv: Path,
    phase309_csv: Path,
    phase323_csv: Path,
) -> list[dict[str, str]]:
    rows = [
        _audit_phase258(phase258_csv),
        _audit_phase303(phase303_csv),
        _audit_phase309(phase309_csv),
        _audit_phase323(phase323_csv),
        _row(
            module="default_model_candidate_gate",
            candidate_input="explicit_model_form_plus_topology_shape_budget_keyspace",
            candidate_output="bounded_default_aic_prediction_or_fail_fast",
            eligible_keys="only_keys_with_model_spec_and_independent_holdout",
            reject_keys="unknown_or_missing_holdout_or_conflicting_topology_direction",
            required_holdout="independent_holdout_per_topology_shape_and_budget_family",
            error_threshold="must_be_defined_before_default_integration",
            failure_policy="fail_fast_no_prediction_no_perf_database_write",
            verdict="default_model_candidate_spec_gap",
        ),
    ]
    if len(rows) != 5:
        raise ValueError("Phase327 expects exactly 5 readiness rows")
    return rows


def write_default_model_candidate_spec_audit_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if len(rows) != 5:
        raise ValueError("Phase327 CSV expects exactly 5 rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_default_model_candidate_spec_audit_doc(path: Path, rows: list[dict[str, str]]) -> None:
    if len(rows) != 5:
        raise ValueError("Phase327 document expects exactly 5 rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = "\n".join(
        f"| {row['module']} | {row['candidate_input']} | {row['candidate_output']} | "
        f"{row['eligible_keys']} | {row['reject_keys']} | {row['default_readiness']} |"
        for row in rows
    )
    content = f"""# Phase327 Default Model Candidate Spec Audit

This audit records what is still missing before any diagnostic evidence can be considered for default AIC readiness.

| Item | Value |
|---|---|
| Default AIC | No-Go |
| diagnostic_only | true |
| valid_for_default | false |
| perf_database | false |
| Runtime integration | No-Go |
| PerfDatabase write | No-Go |

| Module | Candidate input | Candidate output | Eligible keys | Reject keys | Default readiness |
|---|---|---|---|---|---|
{table}

Phase258, Phase303, Phase305, Phase309, and Phase323 are evidence sources for a future candidate specification only. The diagnostic lookup cannot be reused as a prediction API.

Phase323 is lifecycle canary evidence, not model evidence. It does not replace Phase283/285 evidence and does not update Phase303 or Phase309.

Default model readiness still requires an explicit model form, a bounded topology/shape/budget keyspace, independent holdout validation, an error threshold, a fail-fast policy, and a defined PerfDatabase write strategy. Until those exist, default_readiness remains No-Go.
"""
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase258", type=Path, default=DEFAULT_PHASE258)
    parser.add_argument("--phase303", type=Path, default=DEFAULT_PHASE303)
    parser.add_argument("--phase309", type=Path, default=DEFAULT_PHASE309)
    parser.add_argument("--phase323", type=Path, default=DEFAULT_PHASE323)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV_OUT)
    parser.add_argument("--doc-out", type=Path, default=DEFAULT_DOC_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = analyze_default_model_candidate_spec_audit(
        args.phase258,
        args.phase303,
        args.phase309,
        args.phase323,
    )
    write_default_model_candidate_spec_audit_csv(args.csv_out, rows)
    write_default_model_candidate_spec_audit_doc(args.doc_out, rows)


if __name__ == "__main__":
    main()
