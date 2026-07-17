#!/usr/bin/env python3
"""Validate Phase466 real/simulator evidence and enforce single-route exit semantics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CANDIDATES = (
    "schedule_merged_batch_composition",
    "iteration_cost_serving_state_coverage",
    "dp_rank_synchronization_asymmetry",
)
BASE_FIELDS = {
    "run_id",
    "source",
    "rank_id",
    "rank_scope",
    "workload_cohort_digest",
    "progress_window_id",
    "iteration_elapsed_ms",
    "scheduled_prefill_tokens",
    "scheduled_decode_tokens",
    "prefill_request_count",
    "decode_request_count",
}
SCHEDULE_FIELDS = {
    "prefill_chunk_token_histogram",
    "fresh_prefill_tokens",
    "recompute_prefill_tokens",
    "resume_prefill_tokens",
    "decode_kv_token_sum",
    "running_count",
    "waiting_count",
    "cudagraph_mode",
}
STATE_FIELDS = {"running_count", "waiting_count", "completed_request_count"}
REAL_CANDIDATE_FIELDS = {
    "schedule_merged_batch_composition": BASE_FIELDS | SCHEDULE_FIELDS,
    "iteration_cost_serving_state_coverage": BASE_FIELDS | SCHEDULE_FIELDS,
    "dp_rank_synchronization_asymmetry": BASE_FIELDS | STATE_FIELDS,
}
SIM_CANDIDATE_FIELDS = {
    "schedule_merged_batch_composition": BASE_FIELDS | SCHEDULE_FIELDS,
    "iteration_cost_serving_state_coverage": BASE_FIELDS
    | {
        "sim_serving_state_key",
        "sim_predicted_iteration_ms",
        "sim_component_cost_ms",
    },
    "dp_rank_synchronization_asymmetry": BASE_FIELDS | STATE_FIELDS,
}
VALID_JUDGEMENTS = {"PASS", "DISPROVED", "INCONCLUSIVE"}


def _present_fields(rows: list[dict[str, Any]]) -> set[str]:
    if not rows:
        return set()
    keys = set.intersection(*(set(row) for row in rows))
    return {
        key
        for key in keys
        if all(
            row[key] is not None
            and (not isinstance(row[key], str) or bool(row[key].strip()))
            for row in rows
        )
    }


def evaluate_evidence_coverage(
    real_rows: list[dict[str, Any]],
    sim_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not real_rows or not sim_rows:
        raise ValueError("missing_real_or_sim_rows")
    if any(row.get("source") != "real" for row in real_rows):
        raise ValueError("real_source_mismatch")
    if any(row.get("rank_scope") != "dp_rank" for row in real_rows):
        raise ValueError("real_rank_scope_mismatch")
    if any(row.get("source") != "sim" for row in sim_rows):
        raise ValueError("sim_source_mismatch")
    if any(row.get("rank_scope") != "global_simulator" for row in sim_rows):
        raise ValueError("sim_rank_scope_mismatch")

    real_digests = {str(row.get("workload_cohort_digest", "")) for row in real_rows}
    sim_digests = {str(row.get("workload_cohort_digest", "")) for row in sim_rows}
    if len(real_digests) != 1 or real_digests != sim_digests:
        raise ValueError("workload_cohort_digest_mismatch")
    real_windows = {int(row["progress_window_id"]) for row in real_rows}
    sim_windows = {int(row["progress_window_id"]) for row in sim_rows}
    joined_windows = real_windows & sim_windows
    if not joined_windows:
        raise ValueError("missing_joined_progress_window")

    real_fields = _present_fields(real_rows)
    sim_fields = _present_fields(sim_rows)
    candidate_coverage: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATES:
        missing_real = sorted(REAL_CANDIDATE_FIELDS[candidate] - real_fields)
        missing_sim = sorted(SIM_CANDIDATE_FIELDS[candidate] - sim_fields)
        missing = sorted(set(missing_real) | set(missing_sim))
        candidate_coverage[candidate] = {
            "status": "EVALUABLE" if not missing else "INCONCLUSIVE_MISSING_FIELDS",
            "missing_fields": missing,
            "missing_real_fields": missing_real,
            "missing_sim_fields": missing_sim,
        }
    return {
        "workload_cohort_digest": next(iter(real_digests)),
        "joined_progress_windows": len(joined_windows),
        "real_rank_ids": sorted({int(row["rank_id"]) for row in real_rows}),
        "real_rank_scope": "dp_rank",
        "sim_rank_scope": "global_simulator",
        "candidate_coverage": candidate_coverage,
    }


def select_route(judgements: dict[str, str]) -> dict[str, Any]:
    if set(judgements) != set(CANDIDATES):
        raise ValueError("candidate_judgement_set_mismatch")
    invalid = {value for value in judgements.values() if value not in VALID_JUDGEMENTS}
    if invalid:
        raise ValueError(f"invalid_candidate_judgement:{sorted(invalid)}")
    passed = [candidate for candidate in CANDIDATES if judgements[candidate] == "PASS"]
    if len(passed) == 1:
        return {"status": "SELECTED", "selected_route": passed[0], "pass_count": 1}
    return {"status": "INCONCLUSIVE", "selected_route": None, "pass_count": len(passed)}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--judgements-json", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    real_paths = sorted(args.real_root.glob("*/iteration_rows.csv"))
    sim_paths = sorted(args.sim_root.glob("*/iteration_rows.csv"))
    real_by_scenario = {path.parent.name: _read_csv(path) for path in real_paths}
    sim_by_scenario = {path.parent.name: _read_csv(path) for path in sim_paths}
    if set(real_by_scenario) != set(sim_by_scenario) or not real_by_scenario:
        raise ValueError("real_sim_scenario_set_mismatch")
    coverage = {
        scenario: evaluate_evidence_coverage(real_by_scenario[scenario], sim_by_scenario[scenario])
        for scenario in sorted(real_by_scenario)
    }
    judgements = {candidate: "INCONCLUSIVE" for candidate in CANDIDATES}
    if args.judgements_json is not None:
        raw = json.loads(args.judgements_json.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("candidate_judgements_must_be_object")
        judgements = {str(key): str(value) for key, value in raw.items()}
    result = {
        "schema": "phase466_exit_review_v1",
        "coverage": coverage,
        "human_reviewed_judgements": judgements,
        "route_selection": select_route(judgements),
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
        "default_readiness": "No-Go",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
