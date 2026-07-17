#!/usr/bin/env python3
"""Validate the Phase466 residual-attribution report against Phase463."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
from typing import Any


FAILED_SCENARIOS = (
    "K2.5-tp4ep8dp2-32k3k",
    "K2.5-tp8ep8-8k2k-bt65536",
    "K2.5-tp4ep8dp2-8k2k-bt65536",
)
CANDIDATES = (
    "schedule_merged_batch_composition",
    "iteration_cost_serving_state_coverage",
    "dp_rank_synchronization_asymmetry",
)
BASE_COMMIT = "9ba5ad93ca8805a1eb427593ced8cee01aea6804"

SOURCE_PATHS = (
    "docs/iter_gap_investigation/phase465_parallel_exploration_charter.md",
    "docs/iter_gap_investigation/phase462_baseline_handoff.md",
    "docs/iter_gap_investigation/phase462_bt65536_metric_and_composition_audit/phase462_bt65536_metric_and_composition_audit.md",
    "docs/iter_gap_investigation/phase462_exit_review.md",
    "docs/iter_gap_investigation/phase462_step3f/phase462_step3f.md",
    "docs/iter_gap_investigation/phase462_dual_replica_metric_consistency/phase462_dual_replica_metric_consistency.md",
    "docs/iter_gap_investigation/phase462_engine_loop_arc_closure.md",
    "docs/iter_gap_investigation/phase462_step2c9_six_point_ab.md",
    "docs/iter_gap_investigation/phase463_six_point_latency_recollect.md",
    "docs/iter_gap_investigation/phase463_six_point_latency_recollect.csv",
)

NUMERIC_FIELDS = (
    "throughput_abs_error_pct",
    "throughput_sim_over_real",
    "ttft_sim_over_real",
    "tpot_sim_over_real",
    "e2e_sim_over_real",
    "prefill_reqs_sim_over_real_delta_pct",
    "decode_reqs_sim_over_real_delta_pct",
)
REQUIRED_TEXT_FIELDS = (
    "existing_support",
    "existing_counterevidence",
    "missing_observation",
    "phase466b_required_fields",
    "expected_signal",
    "disproof_condition",
    "evidence_sources",
)
ALIGNMENT_FIELDS = {
    "run_id",
    "source",
    "rank_id",
    "iteration_start_offset_ms",
    "iteration_end_offset_ms",
    "workload_cohort_digest",
    "cumulative_scheduled_tokens",
    "progress_window_id",
}
VALID_EVIDENCE_SOURCE = (
    "Phase462 admissible closeout/field-audit evidence;"
    "Phase463 gated N512/C128 evidence"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid_numeric_field:{row.get('scenario')}:{key}") from exc


def _delta_pct(sim: float, real: float) -> float:
    if real <= 0:
        raise ValueError(f"nonpositive_real_value:{real}")
    return (sim / real - 1.0) * 100.0


def load_failed_scenarios(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    if len(all_rows) != 7:
        raise ValueError(f"phase463_row_count:{len(all_rows)}!=7")

    formal_rows = [row for row in all_rows if row.get("role") == "formal"]
    if len(formal_rows) != 6:
        raise ValueError(f"phase463_formal_row_count:{len(formal_rows)}!=6")
    failed = {
        row["scenario"]: row
        for row in formal_rows
        if row.get("throughput_gate") == "fail"
    }
    if set(failed) != set(FAILED_SCENARIOS):
        raise ValueError(f"phase463_failed_set:{sorted(failed)}")

    for row in all_rows:
        for key, expected in {
            "diagnostic_only": "True",
            "valid_for_default": "False",
            "perf_database": "False",
            "default_readiness": "No-Go",
        }.items():
            if row.get(key) != expected:
                raise ValueError(f"phase463_boundary:{row.get('scenario')}:{key}")
    return [failed[scenario] for scenario in FAILED_SCENARIOS]


def build_scenario_evidence(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for row in rows:
        evidence.append(
            {
                "scenario": row["scenario"],
                "throughput_abs_error_pct": _float(
                    row, "throughput_abs_error_pct"
                ),
                "throughput_sim_over_real": _float(row, "throughput_sim_over_real"),
                "ttft_sim_over_real": _float(row, "ttft_sim_over_real"),
                "tpot_sim_over_real": _float(row, "tpot_sim_over_real"),
                "e2e_sim_over_real": _float(row, "e2e_sim_over_real"),
                "prefill_reqs_sim_over_real_delta_pct": _delta_pct(
                    _float(row, "sim_avg_prefill_reqs_per_iter"),
                    _float(row, "real_avg_context_reqs_per_logged_iteration"),
                ),
                "decode_reqs_sim_over_real_delta_pct": _delta_pct(
                    _float(row, "sim_avg_decode_reqs_per_iter"),
                    _float(row, "real_avg_generation_reqs_per_logged_iteration"),
                ),
            }
        )
    return evidence


def expected_status(
    scenario: str, candidate: str, evidence: dict[str, Any]
) -> str:
    """Derive the matrix status from topology and measured Phase463 facts."""
    if candidate == "schedule_merged_batch_composition":
        return "open_unseparated"
    if candidate == "iteration_cost_serving_state_coverage":
        if math.isclose(
            float(evidence["tpot_sim_over_real"]), 1.0, rel_tol=0.0, abs_tol=0.02
        ):
            return "open_with_decode_counterevidence"
        return "open_unseparated"
    if candidate == "dp_rank_synchronization_asymmetry":
        return (
            "topology_negative_control"
            if "tp8ep8" in scenario
            else "open_topology_only"
        )
    raise ValueError(f"unknown_candidate:{candidate}")


def _validate_hypothesis_contract(row: dict[str, str]) -> None:
    key = (row["scenario"], row["candidate"])
    fields = set(row["phase466b_required_fields"].split(";"))
    if not ALIGNMENT_FIELDS <= fields:
        raise ValueError(f"alignment_fields:{key}:{sorted(ALIGNMENT_FIELDS - fields)}")
    if row["evidence_sources"] != VALID_EVIDENCE_SOURCE:
        raise ValueError(f"evidence_sources:{key}")

    support = row["existing_support"].lower()
    counter = row["existing_counterevidence"].lower()
    signal = row["expected_signal"].lower()
    disproof = row["disproof_condition"].lower()
    contracts = {
        "schedule_merged_batch_composition": (
            (support, ("phase463", "phase462")),
            (counter, ("composition", "logging", "engine-loop")),
            (signal, ("composition", "prefill", "request-state")),
            (disproof, ("composition",)),
        ),
        "iteration_cost_serving_state_coverage": (
            (support, ("phase463",)),
            (counter, ("composition", "latency", "tpot")),
            (signal, ("cost", "deficit")),
            (disproof, ("cost", "mismatch", "iteration")),
        ),
        "dp_rank_synchronization_asymmetry": (
            (support, ("rank", "topology", "dp2")),
            (counter, ("rank", "dp", "tpot")),
            (signal, ("rank", "dp-specific")),
            (disproof, ("rank", "residual")),
        ),
    }[row["candidate"]]
    for value, alternatives in contracts:
        if not any(token in value for token in alternatives):
            raise ValueError(f"hypothesis_contract:{key}:{alternatives}")


def load_attribution_rows(path: Path) -> list[dict[str, str]]:
    if b"\r" in path.read_bytes():
        raise ValueError(f"crlf:{path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_attribution_rows(
    rows: list[dict[str, str]], evidence: list[dict[str, Any]]
) -> None:
    expected_keys = {
        (scenario, candidate)
        for scenario in FAILED_SCENARIOS
        for candidate in CANDIDATES
    }
    row_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row.get("scenario", ""), row.get("candidate", ""))
        if key in row_by_key:
            raise ValueError(f"duplicate_attribution:{key}")
        row_by_key[key] = row
    if set(row_by_key) != expected_keys:
        raise ValueError(f"attribution_matrix:{sorted(row_by_key)}")

    evidence_by_scenario = {row["scenario"]: row for row in evidence}
    for key, row in row_by_key.items():
        scenario_evidence = evidence_by_scenario[key[0]]
        if row.get("candidate_status") != expected_status(
            key[0], key[1], scenario_evidence
        ):
            raise ValueError(f"candidate_status:{key}")
        if row.get("outcome") != "INCONCLUSIVE":
            raise ValueError(f"outcome:{key}")
        for field in NUMERIC_FIELDS:
            actual = _float(row, field)
            expected = float(scenario_evidence[field])
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"numeric_mismatch:{key}:{field}")
        for field in REQUIRED_TEXT_FIELDS:
            if not row.get(field, "").strip():
                raise ValueError(f"missing_attribution_field:{key}:{field}")
        _validate_hypothesis_contract(row)
        for field, expected in {
            "diagnostic_only": "true",
            "valid_for_default": "false",
            "perf_database": "false",
            "default_readiness": "No-Go",
        }.items():
            if row.get(field) != expected:
                raise ValueError(f"attribution_boundary:{key}:{field}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_report(
    path: Path,
    evidence: list[dict[str, Any]],
    attribution_rows: list[dict[str, str]],
) -> None:
    raw = path.read_bytes()
    if b"\r" in raw:
        raise ValueError(f"crlf:{path}")
    text = raw.decode("utf-8")
    if any(line.endswith((" ", "\t")) for line in text.splitlines()):
        raise ValueError(f"trailing_whitespace:{path}")

    required = {
        "结论：`INCONCLUSIVE`",
        "Phase462 composition logging v2/v3/v4",
        "13.8486% / 8.7915% / 8.2043%",
        "DP2-bt65536 off/on",
        BASE_COMMIT,
        "Default AIC=No-Go",
        "human-reviewed preregistration",
        "workload_cohort_digest",
        "cumulative_scheduled_tokens",
        "progress_window_id",
        "iteration_start_offset_ms",
        "iteration_end_offset_ms",
        *FAILED_SCENARIOS,
        *CANDIDATES,
    }
    missing = sorted(item for item in required if item not in text)
    if missing:
        raise ValueError(f"report_missing:{missing}")

    for row in evidence:
        expected_table_row = (
            f'| {row["scenario"]} | '
            f'{row["throughput_abs_error_pct"]:.3f}% | '
            f'{row["ttft_sim_over_real"]:.3f} | '
            f'{row["tpot_sim_over_real"]:.3f} | '
            f'{row["e2e_sim_over_real"]:.3f} | '
            f'{row["prefill_reqs_sim_over_real_delta_pct"]:.3f}% | '
            f'{row["decode_reqs_sim_over_real_delta_pct"]:.3f}% |'
        )
        if expected_table_row not in text:
            raise ValueError(f"report_numeric_mismatch:{row['scenario']}")

    for row in attribution_rows:
        matrix_row = (
            f"| {row['scenario']} | {row['candidate']} | "
            f"{row['candidate_status']} |"
        )
        if matrix_row not in text:
            raise ValueError(
                f"report_attribution_mismatch:{row['scenario']}:{row['candidate']}"
            )

    root = _repo_root()
    for source in SOURCE_PATHS:
        if f"| {source} | {_sha256(root / source)} |" not in text:
            raise ValueError(f"report_source_hash:{source}")


def validate_outputs(source_csv: Path, attribution_csv: Path, report_md: Path) -> None:
    evidence = build_scenario_evidence(load_failed_scenarios(source_csv))
    attribution_rows = load_attribution_rows(attribution_csv)
    validate_attribution_rows(attribution_rows, evidence)
    validate_report(report_md, evidence, attribution_rows)


def main() -> int:
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=root
        / "docs/iter_gap_investigation/phase463_six_point_latency_recollect.csv",
    )
    parser.add_argument(
        "--attribution-csv",
        type=Path,
        default=root / "docs/iter_gap_investigation/phase466_residual_attribution.csv",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=root / "docs/iter_gap_investigation/phase466_residual_attribution.md",
    )
    args = parser.parse_args()
    validate_outputs(args.source_csv, args.attribution_csv, args.report_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
