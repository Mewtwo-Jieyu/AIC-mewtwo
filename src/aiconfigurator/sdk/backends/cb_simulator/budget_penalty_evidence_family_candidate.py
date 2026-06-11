"""Diagnostic-only Phase222 budget penalty evidence family candidates."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


SOURCE = "phase222_budget_penalty_evidence_family"
DEFAULT_READINESS = "No-Go"
RECOMMENDED_MODEL_BOUNDARY = "diagnostic_only_exact_key"
ACCEPTED_CANDIDATE_METADATA = {
    "tp8_dp1_ep8:isl4000_osl2000_batch128:control_bt4000:holdout_bt65536": {
        "topology_key": "tp8_dp1_ep8",
        "shape_key": "isl4000_osl2000_batch128",
        "control_max_num_batched_tokens": 4000,
        "holdout_max_num_batched_tokens": 65536,
        "control_scenario": "tp8ep8-4k2k-bt4000",
        "holdout_scenario": "tp8ep8-4k2k-bt65536",
    },
    "tp8_dp1_ep8:isl8000_osl2000_batch128:control_bt8000:holdout_bt65536": {
        "topology_key": "tp8_dp1_ep8",
        "shape_key": "isl8000_osl2000_batch128",
        "control_max_num_batched_tokens": 8000,
        "holdout_max_num_batched_tokens": 65536,
        "control_scenario": "tp8ep8-bt8000",
        "holdout_scenario": "tp8ep8-bt65536",
    },
    "tp8_dp1_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536": {
        "topology_key": "tp8_dp1_ep8",
        "shape_key": "isl12000_osl2000_batch128",
        "control_max_num_batched_tokens": 12000,
        "holdout_max_num_batched_tokens": 65536,
        "control_scenario": "tp8ep8-12k2k-bt12000",
        "holdout_scenario": "tp8ep8-12k2k-bt65536",
    },
    "tp4_dp2_ep8:isl4000_osl2000_batch128:control_bt4000:holdout_bt65536": {
        "topology_key": "tp4_dp2_ep8",
        "shape_key": "isl4000_osl2000_batch128",
        "control_max_num_batched_tokens": 4000,
        "holdout_max_num_batched_tokens": 65536,
        "control_scenario": "tp4dp2ep8-4k2k-bt4000",
        "holdout_scenario": "tp4dp2ep8-4k2k-bt65536",
    },
    "tp4_dp2_ep8:isl8000_osl2000_batch128:control_bt8000:holdout_bt65536": {
        "topology_key": "tp4_dp2_ep8",
        "shape_key": "isl8000_osl2000_batch128",
        "control_max_num_batched_tokens": 8000,
        "holdout_max_num_batched_tokens": 65536,
        "control_scenario": "tp4dp2ep8-bt8000",
        "holdout_scenario": "tp4dp2ep8-bt65536",
    },
    "tp4_dp2_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536": {
        "topology_key": "tp4_dp2_ep8",
        "shape_key": "isl12000_osl2000_batch128",
        "control_max_num_batched_tokens": 12000,
        "holdout_max_num_batched_tokens": 65536,
        "control_scenario": "tp4dp2ep8-12k2k-bt12000",
        "holdout_scenario": "tp4dp2ep8-12k2k-bt65536",
    },
}
ACCEPTED_CANDIDATE_KEYS = set(ACCEPTED_CANDIDATE_METADATA)


@dataclass(frozen=True)
class VLLMBudgetPenaltyEvidenceFamilyCandidate:
    source: str
    candidate_key: str
    topology_key: str
    shape_key: str
    control_scenario: str
    holdout_scenario: str
    control_max_num_batched_tokens: int
    holdout_max_num_batched_tokens: int
    clean_budget_effect: float
    no_budget_penalty_error: float
    evidence_source: str
    min_clean_effect: float
    max_clean_effect: float
    clean_effect_spread: float
    shape_count: int
    recommended_model_boundary: str
    default_readiness: str
    diagnostic_only: bool
    valid_for_default: bool
    perf_database: bool


def _require(row: Mapping[str, str], field: str) -> str:
    value = row.get(field)
    if value is None or value == "":
        raise ValueError(f"missing {field}")
    return value


def _as_int(row: Mapping[str, str], field: str) -> int:
    value = _require(row, field)
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be int: {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{field} must be positive: {parsed!r}")
    return parsed


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
    control_max_num_batched_tokens: int,
    holdout_max_num_batched_tokens: int,
) -> str:
    return (
        f"{topology_key}:{shape_key}:"
        f"control_bt{control_max_num_batched_tokens}:"
        f"holdout_bt{holdout_max_num_batched_tokens}"
    )


def _symmetric_error(predicted: float, observed: float, label: str) -> float:
    if predicted <= 0 or observed <= 0:
        raise ValueError(f"{label} values must be positive")
    ratio = predicted / observed
    return ratio if ratio >= 1.0 else 1.0 / ratio


def _require_close(field: str, actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-5):
        raise ValueError(f"{field} mismatch: got {actual!r}, expected {expected!r}")


def _require_str_match(field: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise ValueError(f"{field} mismatch: got {actual!r}, expected {expected!r}")


def _require_int_match(field: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise ValueError(f"{field} mismatch: got {actual!r}, expected {expected!r}")


def budget_penalty_evidence_family_candidate_from_row(
    row: Mapping[str, str],
) -> VLLMBudgetPenaltyEvidenceFamilyCandidate:
    source = _require(row, "source")
    if source != SOURCE:
        raise ValueError(f"source must be {SOURCE}: {source!r}")

    recommended_model_boundary = _require(row, "recommended_model_boundary")
    if recommended_model_boundary != RECOMMENDED_MODEL_BOUNDARY:
        raise ValueError(
            "recommended_model_boundary must be "
            f"{RECOMMENDED_MODEL_BOUNDARY}: {recommended_model_boundary!r}"
        )
    default_readiness = _require(row, "default_readiness")
    if default_readiness != DEFAULT_READINESS:
        raise ValueError(f"default_readiness must be {DEFAULT_READINESS}: {default_readiness!r}")

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
    control_max_num_batched_tokens = _as_int(row, "control_bt")
    holdout_max_num_batched_tokens = _as_int(row, "holdout_bt")
    if holdout_max_num_batched_tokens != 65536:
        raise ValueError(f"holdout_bt must be 65536: {holdout_max_num_batched_tokens!r}")
    candidate_key = _candidate_key(
        topology_key,
        shape_key,
        control_max_num_batched_tokens,
        holdout_max_num_batched_tokens,
    )
    if candidate_key not in ACCEPTED_CANDIDATE_KEYS:
        raise ValueError(f"candidate_key is not accepted Phase225 evidence: {candidate_key}")

    metadata = ACCEPTED_CANDIDATE_METADATA[candidate_key]
    _require_str_match("topology_key", topology_key, metadata["topology_key"])
    _require_str_match("shape_key", shape_key, metadata["shape_key"])
    control_scenario = _require(row, "control_scenario")
    holdout_scenario = _require(row, "holdout_scenario")
    _require_str_match("control_scenario", control_scenario, metadata["control_scenario"])
    _require_str_match("holdout_scenario", holdout_scenario, metadata["holdout_scenario"])
    _require_int_match(
        "control_bt",
        control_max_num_batched_tokens,
        metadata["control_max_num_batched_tokens"],
    )
    _require_int_match(
        "holdout_bt",
        holdout_max_num_batched_tokens,
        metadata["holdout_max_num_batched_tokens"],
    )

    clean_budget_effect = _as_float(row, "clean_budget_effect")
    no_budget_penalty_error = _as_float(row, "no_budget_penalty_error")
    min_clean_effect = _as_float(row, "min_clean_effect")
    max_clean_effect = _as_float(row, "max_clean_effect")
    clean_effect_spread = _as_float(row, "clean_effect_spread")
    shape_count = _as_int(row, "shape_count")
    if shape_count != 3:
        raise ValueError(f"shape_count must be 3: {shape_count!r}")
    if min_clean_effect > clean_budget_effect or clean_budget_effect > max_clean_effect:
        raise ValueError("clean_budget_effect must be within min/max clean effect")
    _require_close("clean_effect_spread", clean_effect_spread, max_clean_effect / min_clean_effect)
    _require_close(
        "no_budget_penalty_error",
        no_budget_penalty_error,
        _symmetric_error(1.0, clean_budget_effect, "no_budget_penalty_error"),
    )

    return VLLMBudgetPenaltyEvidenceFamilyCandidate(
        source=source,
        candidate_key=candidate_key,
        topology_key=topology_key,
        shape_key=shape_key,
        control_scenario=control_scenario,
        holdout_scenario=holdout_scenario,
        control_max_num_batched_tokens=control_max_num_batched_tokens,
        holdout_max_num_batched_tokens=holdout_max_num_batched_tokens,
        clean_budget_effect=clean_budget_effect,
        no_budget_penalty_error=no_budget_penalty_error,
        evidence_source=_require(row, "evidence_source"),
        min_clean_effect=min_clean_effect,
        max_clean_effect=max_clean_effect,
        clean_effect_spread=clean_effect_spread,
        shape_count=shape_count,
        recommended_model_boundary=recommended_model_boundary,
        default_readiness=default_readiness,
        diagnostic_only=diagnostic_only,
        valid_for_default=valid_for_default,
        perf_database=perf_database,
    )


def load_budget_penalty_evidence_family_candidates(
    path: Path | str,
) -> dict[str, VLLMBudgetPenaltyEvidenceFamilyCandidate]:
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(ACCEPTED_CANDIDATE_KEYS):
        raise ValueError(
            "budget penalty evidence family candidate csv must contain "
            f"{len(ACCEPTED_CANDIDATE_KEYS)} rows: got {len(rows)}"
        )

    row_keys = [
        _candidate_key(
            _require(row, "topology_key"),
            _require(row, "shape_key"),
            _as_int(row, "control_bt"),
            _as_int(row, "holdout_bt"),
        )
        for row in rows
    ]
    duplicate_keys = sorted({key for key in row_keys if row_keys.count(key) > 1})
    if duplicate_keys:
        raise ValueError(f"duplicate candidate_key: {duplicate_keys[0]}")

    candidates: dict[str, VLLMBudgetPenaltyEvidenceFamilyCandidate] = {}
    for row in rows:
        candidate = budget_penalty_evidence_family_candidate_from_row(row)
        if candidate.candidate_key in candidates:
            raise ValueError(f"duplicate candidate_key: {candidate.candidate_key}")
        candidates[candidate.candidate_key] = candidate

    missing = sorted(ACCEPTED_CANDIDATE_KEYS - set(candidates))
    extra = sorted(set(candidates) - ACCEPTED_CANDIDATE_KEYS)
    if missing or extra:
        raise ValueError(f"candidate key set mismatch: missing={missing} extra={extra}")
    return candidates


def get_budget_penalty_evidence_family_candidate(
    candidates: Mapping[str, VLLMBudgetPenaltyEvidenceFamilyCandidate],
    topology_key: str,
    shape_key: str,
    control_max_num_batched_tokens: int,
    holdout_max_num_batched_tokens: int,
) -> VLLMBudgetPenaltyEvidenceFamilyCandidate:
    candidate_key = _candidate_key(
        topology_key,
        shape_key,
        control_max_num_batched_tokens,
        holdout_max_num_batched_tokens,
    )
    try:
        return candidates[candidate_key]
    except KeyError as exc:
        raise KeyError(
            f"exact budget penalty evidence family candidate not found: {candidate_key}"
        ) from exc
