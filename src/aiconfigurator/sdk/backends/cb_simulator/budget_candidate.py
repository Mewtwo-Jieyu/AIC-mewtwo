"""Diagnostic-only clean budget gap candidates for cb_sim analysis."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


SOURCE = "phase164_clean_gpu_benchmark"
ACCEPTED_CANDIDATE_METADATA = {
    "tp8_dp1_ep8:isl8000_osl2000_batch128:max_bt65536": {
        "tp": 8,
        "dp": 1,
        "ep": 8,
        "isl": 8000,
        "osl": 2000,
        "batch_size": 128,
        "baseline_max_num_batched_tokens": 8000,
        "budget_max_num_batched_tokens": 65536,
    },
    "tp4_dp2_ep8:isl8000_osl2000_batch128:max_bt65536": {
        "tp": 4,
        "dp": 2,
        "ep": 8,
        "isl": 8000,
        "osl": 2000,
        "batch_size": 128,
        "baseline_max_num_batched_tokens": 8000,
        "budget_max_num_batched_tokens": 65536,
    },
}
ACCEPTED_CANDIDATE_KEYS = set(ACCEPTED_CANDIDATE_METADATA)


@dataclass(frozen=True)
class VLLMCleanBudgetGapCandidate:
    source: str
    candidate_key: str
    topology_key: str
    shape_key: str
    tp: int
    dp: int
    ep: int
    isl: int
    osl: int
    batch_size: int
    baseline_max_num_batched_tokens: int
    max_num_batched_tokens: int
    clean_budget_effect: float
    sim_budget_effect: float
    budget_gap: float
    budget_rank: int
    budget_real_rank: int
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
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be int: {value!r}") from exc


def _as_float(row: Mapping[str, str], field: str) -> float:
    value = _require(row, field)
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be float: {value!r}") from exc


def _as_bool(row: Mapping[str, str], field: str) -> bool:
    value = _require(row, field).lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"{field} must be true/false: {value!r}")


def _candidate_key(topology_key: str, shape_key: str, max_num_batched_tokens: int) -> str:
    return f"{topology_key}:{shape_key}:max_bt{max_num_batched_tokens}"


def _require_int_match(field: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise ValueError(f"{field} mismatch: got {actual}, expected {expected}")


def clean_budget_gap_candidate_from_row(
    row: Mapping[str, str],
) -> VLLMCleanBudgetGapCandidate:
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
    max_num_batched_tokens = _as_int(row, "budget_max_num_batched_tokens")
    expected_key = _candidate_key(topology_key, shape_key, max_num_batched_tokens)
    candidate_key = _require(row, "candidate_key")
    if candidate_key != expected_key:
        raise ValueError(
            f"candidate_key mismatch: got {candidate_key!r}, expected {expected_key!r}"
        )
    if candidate_key not in ACCEPTED_CANDIDATE_KEYS:
        raise ValueError(f"candidate_key is not accepted Phase166 evidence: {candidate_key}")

    metadata = ACCEPTED_CANDIDATE_METADATA[candidate_key]
    tp = _as_int(row, "tp")
    dp = _as_int(row, "dp")
    ep = _as_int(row, "ep")
    isl = _as_int(row, "isl")
    osl = _as_int(row, "osl")
    batch_size = _as_int(row, "batch_size")
    baseline_max_num_batched_tokens = _as_int(
        row,
        "baseline_max_num_batched_tokens",
    )
    for field, actual in (
        ("tp", tp),
        ("dp", dp),
        ("ep", ep),
        ("isl", isl),
        ("osl", osl),
        ("batch_size", batch_size),
        ("baseline_max_num_batched_tokens", baseline_max_num_batched_tokens),
        ("budget_max_num_batched_tokens", max_num_batched_tokens),
    ):
        _require_int_match(field, actual, metadata[field])

    clean_budget_effect = _as_float(row, "clean_budget_effect")
    sim_budget_effect = _as_float(row, "sim_budget_effect")
    budget_gap = _as_float(row, "budget_gap")
    for field, value in (
        ("clean_budget_effect", clean_budget_effect),
        ("sim_budget_effect", sim_budget_effect),
        ("budget_gap", budget_gap),
    ):
        if value <= 0:
            raise ValueError(f"{field} must be positive: {value!r}")
    expected_budget_gap = clean_budget_effect / sim_budget_effect
    if not math.isclose(
        budget_gap,
        expected_budget_gap,
        rel_tol=1e-6,
        abs_tol=1e-5,
    ):
        raise ValueError(
            f"budget_gap mismatch: got {budget_gap!r}, "
            f"expected clean_budget_effect/sim_budget_effect={expected_budget_gap!r}"
        )

    return VLLMCleanBudgetGapCandidate(
        source=source,
        candidate_key=candidate_key,
        topology_key=topology_key,
        shape_key=shape_key,
        tp=tp,
        dp=dp,
        ep=ep,
        isl=isl,
        osl=osl,
        batch_size=batch_size,
        baseline_max_num_batched_tokens=baseline_max_num_batched_tokens,
        max_num_batched_tokens=max_num_batched_tokens,
        clean_budget_effect=clean_budget_effect,
        sim_budget_effect=sim_budget_effect,
        budget_gap=budget_gap,
        budget_rank=_as_int(row, "budget_rank"),
        budget_real_rank=_as_int(row, "budget_real_rank"),
        diagnostic_only=diagnostic_only,
        valid_for_default=valid_for_default,
        perf_database=perf_database,
    )


def load_clean_budget_gap_candidates(
    path: Path | str,
) -> dict[str, VLLMCleanBudgetGapCandidate]:
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(ACCEPTED_CANDIDATE_KEYS):
        raise ValueError(
            f"clean budget gap candidate csv must contain "
            f"{len(ACCEPTED_CANDIDATE_KEYS)} rows: got {len(rows)}"
        )

    candidates: dict[str, VLLMCleanBudgetGapCandidate] = {}
    for row in rows:
        candidate = clean_budget_gap_candidate_from_row(row)
        if candidate.candidate_key in candidates:
            raise ValueError(f"duplicate candidate_key: {candidate.candidate_key}")
        candidates[candidate.candidate_key] = candidate

    missing = sorted(ACCEPTED_CANDIDATE_KEYS - set(candidates))
    extra = sorted(set(candidates) - ACCEPTED_CANDIDATE_KEYS)
    if missing or extra:
        raise ValueError(f"candidate key set mismatch: missing={missing} extra={extra}")
    return candidates


def get_clean_budget_gap_candidate(
    candidates: Mapping[str, VLLMCleanBudgetGapCandidate],
    topology_key: str,
    shape_key: str,
    max_num_batched_tokens: int,
) -> VLLMCleanBudgetGapCandidate:
    candidate_key = _candidate_key(topology_key, shape_key, max_num_batched_tokens)
    try:
        return candidates[candidate_key]
    except KeyError as exc:
        raise KeyError(
            f"exact clean budget gap candidate not found: {candidate_key}"
        ) from exc
