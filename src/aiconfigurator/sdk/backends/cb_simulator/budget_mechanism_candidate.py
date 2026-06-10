"""Diagnostic-only budget mechanism candidates for cb_sim analysis."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


SOURCE = "phase171_budget_mechanism_candidate"
ACCEPTED_CANDIDATE_KEYS = {
    "tp8_dp1_ep8:isl8000_osl2000_batch128:max_bt65536",
    "tp4_dp2_ep8:isl8000_osl2000_batch128:max_bt65536",
}


@dataclass(frozen=True)
class VLLMBudgetMechanismCandidate:
    source: str
    candidate_key: str
    topology_key: str
    shape_key: str
    max_num_batched_tokens: int
    baseline_breakdown_name: str
    budget_breakdown_name: str
    clean_budget_effect: float
    sim_budget_effect: float
    steady_state_time_ratio: float
    depenalized_budget_effect: float
    raw_error_ratio: float
    depenalized_error_ratio: float
    depenalized_is_closer: bool
    exact_gap_upper_bound: float
    baseline_steady_state_time_ms: float
    budget_steady_state_time_ms: float
    diagnostic_only: bool
    valid_for_default: bool
    perf_database: bool


def _require(row: Mapping[str, str], field: str) -> str:
    value = row.get(field)
    if value is None or value == "":
        raise ValueError(f"missing {field}")
    return value


def _as_float(row: Mapping[str, str], field: str) -> float:
    value = _require(row, field)
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be float: {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{field} must be positive: {parsed!r}")
    return parsed


def _as_bool(row: Mapping[str, str], field: str) -> bool:
    value = _require(row, field).lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"{field} must be true/false: {value!r}")


def _candidate_key(
    topology_key: str,
    shape_key: str,
    max_num_batched_tokens: int,
) -> str:
    return f"{topology_key}:{shape_key}:max_bt{max_num_batched_tokens}"


def _require_close(field: str, actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-5):
        raise ValueError(f"{field} mismatch: got {actual!r}, expected {expected!r}")


def budget_mechanism_candidate_from_row(
    row: Mapping[str, str],
) -> VLLMBudgetMechanismCandidate:
    source = _require(row, "source")
    if source != SOURCE:
        raise ValueError(f"source must be {SOURCE}: {source!r}")

    diagnostic_only = _as_bool(row, "diagnostic_only")
    valid_for_default = _as_bool(row, "valid_for_default")
    perf_database = _as_bool(row, "perf_database")
    if diagnostic_only is not True:
        raise ValueError("diagnostic_only must be true")
    if valid_for_default is not False:
        raise ValueError("valid_for_default must be false")
    if perf_database is not False:
        raise ValueError("perf_database must be false")

    topology_key = _require(row, "topology_key")
    shape_key = _require(row, "shape_key")
    max_num_batched_tokens = 65536
    candidate_key = _candidate_key(topology_key, shape_key, max_num_batched_tokens)
    if candidate_key not in ACCEPTED_CANDIDATE_KEYS:
        raise ValueError(f"candidate_key is not accepted Phase172 evidence: {candidate_key}")

    clean_budget_effect = _as_float(row, "clean_budget_effect")
    sim_budget_effect = _as_float(row, "sim_budget_effect")
    steady_state_time_ratio = _as_float(row, "steady_state_time_ratio")
    depenalized_budget_effect = _as_float(row, "depenalized_budget_effect")
    raw_error_ratio = _as_float(row, "raw_error_ratio")
    depenalized_error_ratio = _as_float(row, "depenalized_error_ratio")
    exact_gap_upper_bound = _as_float(row, "exact_gap_upper_bound")
    baseline_steady_state_time_ms = _as_float(row, "baseline_steady_state_time_ms")
    budget_steady_state_time_ms = _as_float(row, "budget_steady_state_time_ms")
    depenalized_is_closer = _as_bool(row, "depenalized_is_closer")
    if depenalized_is_closer is not True:
        raise ValueError("depenalized_is_closer must be true")

    _require_close(
        "steady_state_time_ratio",
        steady_state_time_ratio,
        budget_steady_state_time_ms / baseline_steady_state_time_ms,
    )
    _require_close(
        "depenalized_budget_effect",
        depenalized_budget_effect,
        sim_budget_effect * steady_state_time_ratio,
    )
    _require_close(
        "raw_error_ratio",
        raw_error_ratio,
        sim_budget_effect / clean_budget_effect,
    )
    _require_close(
        "depenalized_error_ratio",
        depenalized_error_ratio,
        depenalized_budget_effect / clean_budget_effect,
    )

    return VLLMBudgetMechanismCandidate(
        source=source,
        candidate_key=candidate_key,
        topology_key=topology_key,
        shape_key=shape_key,
        max_num_batched_tokens=max_num_batched_tokens,
        baseline_breakdown_name=_require(row, "baseline_breakdown_name"),
        budget_breakdown_name=_require(row, "budget_breakdown_name"),
        clean_budget_effect=clean_budget_effect,
        sim_budget_effect=sim_budget_effect,
        steady_state_time_ratio=steady_state_time_ratio,
        depenalized_budget_effect=depenalized_budget_effect,
        raw_error_ratio=raw_error_ratio,
        depenalized_error_ratio=depenalized_error_ratio,
        depenalized_is_closer=depenalized_is_closer,
        exact_gap_upper_bound=exact_gap_upper_bound,
        baseline_steady_state_time_ms=baseline_steady_state_time_ms,
        budget_steady_state_time_ms=budget_steady_state_time_ms,
        diagnostic_only=diagnostic_only,
        valid_for_default=valid_for_default,
        perf_database=perf_database,
    )


def load_budget_mechanism_candidates(
    path: Path | str,
) -> dict[str, VLLMBudgetMechanismCandidate]:
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(ACCEPTED_CANDIDATE_KEYS):
        raise ValueError(
            f"budget mechanism candidate csv must contain "
            f"{len(ACCEPTED_CANDIDATE_KEYS)} rows: got {len(rows)}"
        )

    candidates: dict[str, VLLMBudgetMechanismCandidate] = {}
    for row in rows:
        candidate = budget_mechanism_candidate_from_row(row)
        if candidate.candidate_key in candidates:
            raise ValueError(f"duplicate candidate_key: {candidate.candidate_key}")
        candidates[candidate.candidate_key] = candidate

    missing = sorted(ACCEPTED_CANDIDATE_KEYS - set(candidates))
    extra = sorted(set(candidates) - ACCEPTED_CANDIDATE_KEYS)
    if missing or extra:
        raise ValueError(f"candidate key set mismatch: missing={missing} extra={extra}")
    return candidates


def get_budget_mechanism_candidate(
    candidates: Mapping[str, VLLMBudgetMechanismCandidate],
    topology_key: str,
    shape_key: str,
    max_num_batched_tokens: int,
) -> VLLMBudgetMechanismCandidate:
    candidate_key = _candidate_key(topology_key, shape_key, max_num_batched_tokens)
    try:
        return candidates[candidate_key]
    except KeyError as exc:
        raise KeyError(f"exact budget mechanism candidate not found: {candidate_key}") from exc
