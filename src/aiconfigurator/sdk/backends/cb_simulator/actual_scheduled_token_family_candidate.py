"""Diagnostic-only Phase258 actual scheduled token family candidates."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


SOURCE = "phase258_actual_scheduled_token_full_family"
MECHANISM_HYPOTHESIS = "actual_scheduled_tokens_not_configured_budget"
DEFAULT_READINESS = "No-Go"
HOLDOUT_MAX_BT = 65536
MAX_HOLDOUT_FILL_FOR_MECHANISM = 0.50
ACCEPTED_CANDIDATE_METADATA = {
    "tp8_dp1_ep8:isl4000_osl2000_batch128:control_bt4000:holdout_bt65536": {
        "topology_key": "tp8_dp1_ep8",
        "shape_key": "isl4000_osl2000_batch128",
        "control_scenario": "tp8ep8-4k2k-bt4000",
        "holdout_scenario": "tp8ep8-4k2k-bt65536",
        "control_max_num_batched_tokens": 4000,
        "holdout_max_num_batched_tokens": HOLDOUT_MAX_BT,
        "control_configured_budget_aggregate": 4000,
        "holdout_configured_budget_aggregate": HOLDOUT_MAX_BT,
    },
    "tp4_dp2_ep8:isl4000_osl2000_batch128:control_bt4000:holdout_bt65536": {
        "topology_key": "tp4_dp2_ep8",
        "shape_key": "isl4000_osl2000_batch128",
        "control_scenario": "tp4dp2ep8-4k2k-bt4000",
        "holdout_scenario": "tp4dp2ep8-4k2k-bt65536",
        "control_max_num_batched_tokens": 4000,
        "holdout_max_num_batched_tokens": HOLDOUT_MAX_BT,
        "control_configured_budget_aggregate": 8000,
        "holdout_configured_budget_aggregate": HOLDOUT_MAX_BT * 2,
    },
    "tp8_dp1_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536": {
        "topology_key": "tp8_dp1_ep8",
        "shape_key": "isl12000_osl2000_batch128",
        "control_scenario": "tp8ep8-12k2k-bt12000",
        "holdout_scenario": "tp8ep8-12k2k-bt65536",
        "control_max_num_batched_tokens": 12000,
        "holdout_max_num_batched_tokens": HOLDOUT_MAX_BT,
        "control_configured_budget_aggregate": 12000,
        "holdout_configured_budget_aggregate": HOLDOUT_MAX_BT,
    },
    "tp4_dp2_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536": {
        "topology_key": "tp4_dp2_ep8",
        "shape_key": "isl12000_osl2000_batch128",
        "control_scenario": "tp4dp2ep8-12k2k-bt12000",
        "holdout_scenario": "tp4dp2ep8-12k2k-bt65536",
        "control_max_num_batched_tokens": 12000,
        "holdout_max_num_batched_tokens": HOLDOUT_MAX_BT,
        "control_configured_budget_aggregate": 24000,
        "holdout_configured_budget_aggregate": HOLDOUT_MAX_BT * 2,
    },
}
ACCEPTED_CANDIDATE_KEYS = set(ACCEPTED_CANDIDATE_METADATA)


@dataclass(frozen=True)
class VLLMActualScheduledTokenFamilyCandidate:
    source: str
    candidate_key: str
    topology_key: str
    shape_key: str
    control_scenario: str
    holdout_scenario: str
    control_max_num_batched_tokens: int
    holdout_max_num_batched_tokens: int
    control_configured_budget_aggregate: int
    holdout_configured_budget_aggregate: int
    control_rank_sum_p99_scheduled_total_tokens: float
    control_rank_sum_max_scheduled_total_tokens: int
    holdout_rank_sum_p99_scheduled_total_tokens: float
    holdout_rank_sum_max_scheduled_total_tokens: int
    control_rank_sum_p99_fill_ratio: float
    control_rank_sum_max_fill_ratio: float
    holdout_rank_sum_p99_fill_ratio: float
    holdout_rank_sum_max_fill_ratio: float
    control_output_tok_s: float
    holdout_output_tok_s: float
    output_ratio_vs_control: float
    mechanism_hypothesis: str
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


def _require_close(field: str, actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-5):
        raise ValueError(f"{field} mismatch: got {actual!r}, expected {expected!r}")


def _require_str_match(field: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise ValueError(f"{field} mismatch: got {actual!r}, expected {expected!r}")


def _require_int_match(field: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise ValueError(f"{field} mismatch: got {actual!r}, expected {expected!r}")


def _require_common_flags(row: Mapping[str, str]) -> tuple[bool, bool, bool]:
    source = _require(row, "source")
    if source != SOURCE:
        raise ValueError(f"source must be {SOURCE}: {source!r}")
    mechanism_hypothesis = _require(row, "mechanism_hypothesis")
    if mechanism_hypothesis != MECHANISM_HYPOTHESIS:
        raise ValueError(
            f"mechanism_hypothesis must be {MECHANISM_HYPOTHESIS}: "
            f"{mechanism_hypothesis!r}"
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
    return diagnostic_only, valid_for_default, perf_database


def _validate_row_budget(row: Mapping[str, str], role: str, expected_aggregate: int) -> None:
    max_bt = _as_int(row, "max_bt")
    configured_budget_per_rank = _as_int(row, "configured_budget_per_rank")
    configured_budget_aggregate = _as_int(row, "configured_budget_aggregate")
    if configured_budget_per_rank != max_bt:
        raise ValueError(
            "configured_budget_per_rank mismatch: "
            f"got {configured_budget_per_rank!r}, expected {max_bt!r}"
        )
    _require_int_match(f"{role}_configured_budget_aggregate", configured_budget_aggregate, expected_aggregate)


def _validate_fill(row: Mapping[str, str], prefix: str, configured_budget_aggregate: int) -> tuple[float, int, float, float]:
    p99_scheduled = _as_float(row, "rank_sum_p99_scheduled_total_tokens")
    max_scheduled = _as_int(row, "rank_sum_max_scheduled_total_tokens")
    p99_fill = _as_float(row, "rank_sum_p99_fill_ratio")
    max_fill = _as_float(row, "rank_sum_max_fill_ratio")
    _require_close(f"{prefix}_rank_sum_p99_fill_ratio", p99_fill, p99_scheduled / configured_budget_aggregate)
    _require_close(f"{prefix}_rank_sum_max_fill_ratio", max_fill, max_scheduled / configured_budget_aggregate)
    return p99_scheduled, max_scheduled, p99_fill, max_fill


def actual_scheduled_token_family_candidate_from_pair(
    control_row: Mapping[str, str],
    holdout_row: Mapping[str, str],
) -> VLLMActualScheduledTokenFamilyCandidate:
    control_flags = _require_common_flags(control_row)
    holdout_flags = _require_common_flags(holdout_row)
    if control_flags != holdout_flags:
        raise ValueError("pair flags mismatch")

    if _require(control_row, "role") != "control":
        raise ValueError("control row role must be control")
    if _require(holdout_row, "role") != "holdout":
        raise ValueError("holdout row role must be holdout")

    topology_key = _require(control_row, "topology_key")
    shape_key = _require(control_row, "shape_key")
    _require_str_match("topology_key", _require(holdout_row, "topology_key"), topology_key)
    _require_str_match("shape_key", _require(holdout_row, "shape_key"), shape_key)

    control_max_num_batched_tokens = _as_int(control_row, "control_max_bt")
    holdout_max_num_batched_tokens = _as_int(control_row, "holdout_max_bt")
    _require_int_match("control_max_bt", _as_int(holdout_row, "control_max_bt"), control_max_num_batched_tokens)
    _require_int_match("holdout_max_bt", _as_int(holdout_row, "holdout_max_bt"), holdout_max_num_batched_tokens)
    if holdout_max_num_batched_tokens != HOLDOUT_MAX_BT:
        raise ValueError(f"holdout_max_bt must be {HOLDOUT_MAX_BT}: {holdout_max_num_batched_tokens!r}")
    _require_int_match("control max_bt", _as_int(control_row, "max_bt"), control_max_num_batched_tokens)
    _require_int_match("holdout max_bt", _as_int(holdout_row, "max_bt"), holdout_max_num_batched_tokens)

    candidate_key = _candidate_key(
        topology_key,
        shape_key,
        control_max_num_batched_tokens,
        holdout_max_num_batched_tokens,
    )
    if candidate_key not in ACCEPTED_CANDIDATE_KEYS:
        raise ValueError(f"candidate_key is not accepted Phase261 evidence: {candidate_key}")
    metadata = ACCEPTED_CANDIDATE_METADATA[candidate_key]

    _require_str_match("topology_key", topology_key, metadata["topology_key"])
    _require_str_match("shape_key", shape_key, metadata["shape_key"])
    control_scenario = _require(control_row, "scenario")
    holdout_scenario = _require(holdout_row, "scenario")
    _require_str_match("control_scenario", control_scenario, metadata["control_scenario"])
    _require_str_match("holdout_scenario", holdout_scenario, metadata["holdout_scenario"])
    _require_int_match(
        "control_max_bt",
        control_max_num_batched_tokens,
        metadata["control_max_num_batched_tokens"],
    )
    _require_int_match(
        "holdout_max_bt",
        holdout_max_num_batched_tokens,
        metadata["holdout_max_num_batched_tokens"],
    )

    control_aggregate = metadata["control_configured_budget_aggregate"]
    holdout_aggregate = metadata["holdout_configured_budget_aggregate"]
    _validate_row_budget(control_row, "control", control_aggregate)
    _validate_row_budget(holdout_row, "holdout", holdout_aggregate)

    control_p99_scheduled, control_max_scheduled, control_p99_fill, control_max_fill = _validate_fill(
        control_row,
        "control",
        control_aggregate,
    )
    holdout_p99_scheduled, holdout_max_scheduled, holdout_p99_fill, holdout_max_fill = _validate_fill(
        holdout_row,
        "holdout",
        holdout_aggregate,
    )
    if holdout_p99_fill >= MAX_HOLDOUT_FILL_FOR_MECHANISM:
        raise ValueError(f"holdout rank_sum_p99_fill_ratio too high: {holdout_p99_fill!r}")
    if holdout_max_fill >= MAX_HOLDOUT_FILL_FOR_MECHANISM:
        raise ValueError(f"holdout rank_sum_max_fill_ratio too high: {holdout_max_fill!r}")

    control_output_tok_s = _as_float(control_row, "output_tok_s")
    holdout_output_tok_s = _as_float(holdout_row, "output_tok_s")
    _require_close("control output_ratio_vs_control", _as_float(control_row, "output_ratio_vs_control"), 1.0)
    output_ratio_vs_control = _as_float(holdout_row, "output_ratio_vs_control")
    _require_close("output_ratio_vs_control", output_ratio_vs_control, holdout_output_tok_s / control_output_tok_s)

    diagnostic_only, valid_for_default, perf_database = control_flags
    return VLLMActualScheduledTokenFamilyCandidate(
        source=SOURCE,
        candidate_key=candidate_key,
        topology_key=topology_key,
        shape_key=shape_key,
        control_scenario=control_scenario,
        holdout_scenario=holdout_scenario,
        control_max_num_batched_tokens=control_max_num_batched_tokens,
        holdout_max_num_batched_tokens=holdout_max_num_batched_tokens,
        control_configured_budget_aggregate=control_aggregate,
        holdout_configured_budget_aggregate=holdout_aggregate,
        control_rank_sum_p99_scheduled_total_tokens=control_p99_scheduled,
        control_rank_sum_max_scheduled_total_tokens=control_max_scheduled,
        holdout_rank_sum_p99_scheduled_total_tokens=holdout_p99_scheduled,
        holdout_rank_sum_max_scheduled_total_tokens=holdout_max_scheduled,
        control_rank_sum_p99_fill_ratio=control_p99_fill,
        control_rank_sum_max_fill_ratio=control_max_fill,
        holdout_rank_sum_p99_fill_ratio=holdout_p99_fill,
        holdout_rank_sum_max_fill_ratio=holdout_max_fill,
        control_output_tok_s=control_output_tok_s,
        holdout_output_tok_s=holdout_output_tok_s,
        output_ratio_vs_control=output_ratio_vs_control,
        mechanism_hypothesis=MECHANISM_HYPOTHESIS,
        default_readiness=DEFAULT_READINESS,
        diagnostic_only=diagnostic_only,
        valid_for_default=valid_for_default,
        perf_database=perf_database,
    )


def load_actual_scheduled_token_family_candidates(
    path: Path | str,
) -> dict[str, VLLMActualScheduledTokenFamilyCandidate]:
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 8:
        raise ValueError(f"actual scheduled token family candidate csv must contain 8 rows: got {len(rows)}")

    grouped_rows: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        holdout_max_num_batched_tokens = _as_int(row, "holdout_max_bt")
        if holdout_max_num_batched_tokens != HOLDOUT_MAX_BT:
            raise ValueError(
                f"holdout_max_bt must be {HOLDOUT_MAX_BT}: "
                f"{holdout_max_num_batched_tokens!r}"
            )
        candidate_key = _candidate_key(
            _require(row, "topology_key"),
            _require(row, "shape_key"),
            _as_int(row, "control_max_bt"),
            holdout_max_num_batched_tokens,
        )
        if candidate_key not in ACCEPTED_CANDIDATE_KEYS:
            raise ValueError(f"candidate_key is not accepted Phase261 evidence: {candidate_key}")
        grouped_rows.setdefault(candidate_key, []).append(row)

    candidates: dict[str, VLLMActualScheduledTokenFamilyCandidate] = {}
    for candidate_key, pair_rows in grouped_rows.items():
        controls = [row for row in pair_rows if row.get("role") == "control"]
        holdouts = [row for row in pair_rows if row.get("role") == "holdout"]
        if len(controls) != 1 or len(holdouts) != 1:
            raise ValueError(f"pair must contain exactly one control and one holdout: {candidate_key}")
        candidate = actual_scheduled_token_family_candidate_from_pair(controls[0], holdouts[0])
        if candidate.candidate_key in candidates:
            raise ValueError(f"duplicate candidate_key: {candidate.candidate_key}")
        candidates[candidate.candidate_key] = candidate

    missing = sorted(ACCEPTED_CANDIDATE_KEYS - set(candidates))
    extra = sorted(set(candidates) - ACCEPTED_CANDIDATE_KEYS)
    if missing or extra:
        raise ValueError(f"candidate key set mismatch: missing={missing} extra={extra}")
    return candidates


def get_actual_scheduled_token_family_candidate(
    candidates: Mapping[str, VLLMActualScheduledTokenFamilyCandidate],
    topology_key: str,
    shape_key: str,
    control_max_num_batched_tokens: int,
    holdout_max_num_batched_tokens: int,
) -> VLLMActualScheduledTokenFamilyCandidate:
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
            f"exact actual scheduled token family candidate not found: {candidate_key}"
        ) from exc
