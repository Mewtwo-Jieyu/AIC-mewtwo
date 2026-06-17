"""Diagnostic-only Phase303 deeper trace topology family candidates."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


SOURCE = "phase303_deeper_trace_topology_family"
FAMILY_KEY = "deeper_trace_12k2k_topology_family"
DEFAULT_READINESS = "No-Go"
SHAPE_KEY = "isl12000_osl2000_batch128"
CONTROL_BT = 12000
HOLDOUT_BT = 65536
MAX_HOLDOUT_FILL_FOR_EVIDENCE = 0.50
ACCEPTED_CANDIDATE_METADATA = {
    "tp8_dp1_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536": {
        "topology_key": "tp8_dp1_ep8",
        "shape_key": SHAPE_KEY,
        "control_scenario": "tp8ep8-12k2k-bt12000",
        "holdout_scenario": "tp8ep8-12k2k-bt65536",
        "control_bt": CONTROL_BT,
        "holdout_bt": HOLDOUT_BT,
        "output_ratio": 0.969300,
        "throughput_direction": "holdout_slower",
        "holdout_scheduled_p99": 128,
        "holdout_scheduled_max": 12000,
        "holdout_max_fill": 0.183105,
        "verdict": "partial_only",
        "mechanism_conclusion": "boundary_mixed_overhead_partial_only",
    },
    "tp4_dp2_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536": {
        "topology_key": "tp4_dp2_ep8",
        "shape_key": SHAPE_KEY,
        "control_scenario": "tp4dp2ep8-12k2k-bt12000",
        "holdout_scenario": "tp4dp2ep8-12k2k-bt65536",
        "control_bt": CONTROL_BT,
        "holdout_bt": HOLDOUT_BT,
        "output_ratio": 1.321996,
        "throughput_direction": "holdout_faster",
        "holdout_scheduled_p99": 128,
        "holdout_scheduled_max": 24736,
        "holdout_max_fill": 0.188721,
        "verdict": "boundary_timeline_explains_direction",
        "mechanism_conclusion": "wall_span_iteration_cadence_diagnostic",
    },
}
ACCEPTED_CANDIDATE_KEYS = set(ACCEPTED_CANDIDATE_METADATA)


@dataclass(frozen=True)
class VLLMDeeperTraceTopologyFamilyCandidate:
    source: str
    family_key: str
    candidate_key: str
    topology_key: str
    shape_key: str
    control_scenario: str
    holdout_scenario: str
    control_bt: int
    holdout_bt: int
    output_ratio: float
    throughput_direction: str
    holdout_scheduled_p99: int
    holdout_scheduled_max: int
    holdout_max_fill: float
    budget_ceiling_rejected: bool
    verdict: str
    mechanism_conclusion: str
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
    control_bt: int,
    holdout_bt: int,
) -> str:
    return f"{topology_key}:{shape_key}:control_bt{control_bt}:holdout_bt{holdout_bt}"


def _require_close(field: str, actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"{field} mismatch: got {actual!r}, expected {expected!r}")


def _require_str_match(field: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise ValueError(f"{field} mismatch: got {actual!r}, expected {expected!r}")


def _require_int_match(field: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise ValueError(f"{field} mismatch: got {actual!r}, expected {expected!r}")


def _direction(output_ratio: float) -> str:
    if output_ratio < 1.0:
        return "holdout_slower"
    if output_ratio > 1.0:
        return "holdout_faster"
    return "flat"


def _require_common_flags(row: Mapping[str, str]) -> tuple[bool, bool, bool]:
    source = _require(row, "source")
    if source != SOURCE:
        raise ValueError(f"source must be {SOURCE}: {source!r}")
    family_key = _require(row, "family_key")
    if family_key != FAMILY_KEY:
        raise ValueError(f"family_key must be {FAMILY_KEY}: {family_key!r}")
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
    return diagnostic_only, valid_for_default, perf_database


def deeper_trace_topology_family_candidate_from_row(
    row: Mapping[str, str],
) -> VLLMDeeperTraceTopologyFamilyCandidate:
    diagnostic_only, valid_for_default, perf_database = _require_common_flags(row)

    holdout_bt = _as_int(row, "holdout_bt")
    if holdout_bt != HOLDOUT_BT:
        raise ValueError(f"holdout_bt must be {HOLDOUT_BT}: {holdout_bt!r}")
    control_bt = _as_int(row, "control_bt")
    candidate_key = _candidate_key(
        _require(row, "topology_key"),
        _require(row, "shape_key"),
        control_bt,
        holdout_bt,
    )
    if candidate_key not in ACCEPTED_CANDIDATE_KEYS:
        raise ValueError(f"candidate_key is not accepted Phase305 evidence: {candidate_key}")
    metadata = ACCEPTED_CANDIDATE_METADATA[candidate_key]

    topology_key = _require(row, "topology_key")
    shape_key = _require(row, "shape_key")
    control_scenario = _require(row, "control_scenario")
    holdout_scenario = _require(row, "holdout_scenario")
    _require_str_match("topology_key", topology_key, metadata["topology_key"])
    _require_str_match("shape_key", shape_key, metadata["shape_key"])
    _require_str_match("control_scenario", control_scenario, metadata["control_scenario"])
    _require_str_match("holdout_scenario", holdout_scenario, metadata["holdout_scenario"])
    _require_int_match("control_bt", control_bt, metadata["control_bt"])
    _require_int_match("holdout_bt", holdout_bt, metadata["holdout_bt"])

    output_ratio = _as_float(row, "output_ratio")
    _require_close("output_ratio", output_ratio, metadata["output_ratio"])
    throughput_direction = _require(row, "throughput_direction")
    _require_str_match("throughput_direction", throughput_direction, metadata["throughput_direction"])
    _require_str_match("throughput_direction", _direction(output_ratio), throughput_direction)

    holdout_scheduled_p99 = _as_int(row, "holdout_scheduled_p99")
    holdout_scheduled_max = _as_int(row, "holdout_scheduled_max")
    holdout_max_fill = _as_float(row, "holdout_max_fill")
    _require_int_match("holdout_scheduled_p99", holdout_scheduled_p99, metadata["holdout_scheduled_p99"])
    _require_int_match("holdout_scheduled_max", holdout_scheduled_max, metadata["holdout_scheduled_max"])
    _require_close("holdout_max_fill", holdout_max_fill, metadata["holdout_max_fill"])
    if holdout_max_fill >= MAX_HOLDOUT_FILL_FOR_EVIDENCE:
        raise ValueError(f"holdout_max_fill too high: {holdout_max_fill!r}")

    budget_ceiling_rejected = _as_bool(row, "budget_ceiling_rejected")
    if budget_ceiling_rejected is not True:
        raise ValueError("budget_ceiling_rejected must be true")
    verdict = _require(row, "verdict")
    mechanism_conclusion = _require(row, "mechanism_conclusion")
    _require_str_match("verdict", verdict, metadata["verdict"])
    _require_str_match(
        "mechanism_conclusion",
        mechanism_conclusion,
        metadata["mechanism_conclusion"],
    )

    return VLLMDeeperTraceTopologyFamilyCandidate(
        source=SOURCE,
        family_key=FAMILY_KEY,
        candidate_key=candidate_key,
        topology_key=topology_key,
        shape_key=shape_key,
        control_scenario=control_scenario,
        holdout_scenario=holdout_scenario,
        control_bt=control_bt,
        holdout_bt=holdout_bt,
        output_ratio=output_ratio,
        throughput_direction=throughput_direction,
        holdout_scheduled_p99=holdout_scheduled_p99,
        holdout_scheduled_max=holdout_scheduled_max,
        holdout_max_fill=holdout_max_fill,
        budget_ceiling_rejected=budget_ceiling_rejected,
        verdict=verdict,
        mechanism_conclusion=mechanism_conclusion,
        default_readiness=DEFAULT_READINESS,
        diagnostic_only=diagnostic_only,
        valid_for_default=valid_for_default,
        perf_database=perf_database,
    )


def load_deeper_trace_topology_family_candidates(
    path: Path | str,
) -> dict[str, VLLMDeeperTraceTopologyFamilyCandidate]:
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 2:
        raise ValueError(f"deeper trace topology family candidate csv must contain 2 rows: got {len(rows)}")

    candidates: dict[str, VLLMDeeperTraceTopologyFamilyCandidate] = {}
    for row in rows:
        candidate = deeper_trace_topology_family_candidate_from_row(row)
        if candidate.candidate_key in candidates:
            raise ValueError(f"duplicate candidate_key: {candidate.candidate_key}")
        candidates[candidate.candidate_key] = candidate

    missing = sorted(ACCEPTED_CANDIDATE_KEYS - set(candidates))
    extra = sorted(set(candidates) - ACCEPTED_CANDIDATE_KEYS)
    if missing or extra:
        raise ValueError(f"candidate key set mismatch: missing={missing} extra={extra}")
    return candidates


def get_deeper_trace_topology_family_candidate(
    candidates: Mapping[str, VLLMDeeperTraceTopologyFamilyCandidate],
    topology_key: str,
    shape_key: str,
    control_bt: int,
    holdout_bt: int,
) -> VLLMDeeperTraceTopologyFamilyCandidate:
    candidate_key = _candidate_key(topology_key, shape_key, control_bt, holdout_bt)
    try:
        return candidates[candidate_key]
    except KeyError as exc:
        raise KeyError(
            f"exact deeper trace topology family candidate not found: {candidate_key}"
        ) from exc
