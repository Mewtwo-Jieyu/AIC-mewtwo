from __future__ import annotations

import csv
from pathlib import Path

import pytest

from aiconfigurator.sdk.backends.cb_simulator import (
    VLLMBudgetMechanismCandidate,
    get_budget_mechanism_candidate,
    load_budget_mechanism_candidates,
)


FIELDNAMES = [
    "source",
    "topology_key",
    "shape_key",
    "baseline_breakdown_name",
    "budget_breakdown_name",
    "clean_budget_effect",
    "sim_budget_effect",
    "steady_state_time_ratio",
    "depenalized_budget_effect",
    "raw_error_ratio",
    "depenalized_error_ratio",
    "depenalized_is_closer",
    "exact_gap_upper_bound",
    "baseline_steady_state_time_ms",
    "budget_steady_state_time_ms",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]


def _row(
    *,
    topology_key: str = "tp8_dp1_ep8",
    shape_key: str = "isl8000_osl2000_batch128",
    diagnostic_only: str = "true",
    valid_for_default: str = "false",
    perf_database: str = "false",
) -> dict[str, str]:
    if topology_key == "tp8_dp1_ep8":
        return {
            "source": "phase171_budget_mechanism_candidate",
            "topology_key": topology_key,
            "shape_key": shape_key,
            "baseline_breakdown_name": "K2.5-tp8ep8-8k2k",
            "budget_breakdown_name": "K2.5-tp8ep8-8k2k-bt65536",
            "clean_budget_effect": "1.000000",
            "sim_budget_effect": "0.250000",
            "steady_state_time_ratio": "4.000000",
            "depenalized_budget_effect": "1.000000",
            "raw_error_ratio": "0.250000",
            "depenalized_error_ratio": "1.000000",
            "depenalized_is_closer": "true",
            "exact_gap_upper_bound": "4.000000",
            "baseline_steady_state_time_ms": "1000.000000",
            "budget_steady_state_time_ms": "4000.000000",
            "diagnostic_only": diagnostic_only,
            "valid_for_default": valid_for_default,
            "perf_database": perf_database,
        }
    return {
        "source": "phase171_budget_mechanism_candidate",
        "topology_key": topology_key,
        "shape_key": shape_key,
        "baseline_breakdown_name": "K2.5-tp4ep8dp2-8k2k",
        "budget_breakdown_name": "K2.5-tp4ep8dp2-8k2k-bt65536",
        "clean_budget_effect": "1.250000",
        "sim_budget_effect": "0.250000",
        "steady_state_time_ratio": "5.000000",
        "depenalized_budget_effect": "1.250000",
        "raw_error_ratio": "0.200000",
        "depenalized_error_ratio": "1.000000",
        "depenalized_is_closer": "true",
        "exact_gap_upper_bound": "5.000000",
        "baseline_steady_state_time_ms": "2000.000000",
        "budget_steady_state_time_ms": "10000.000000",
        "diagnostic_only": diagnostic_only,
        "valid_for_default": valid_for_default,
        "perf_database": perf_database,
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_load_and_query_two_exact_mechanism_candidates(tmp_path: Path) -> None:
    path = tmp_path / "candidate.csv"
    _write_csv(path, [_row(), _row(topology_key="tp4_dp2_ep8")])

    candidates = load_budget_mechanism_candidates(path)
    first = get_budget_mechanism_candidate(
        candidates,
        "tp8_dp1_ep8",
        "isl8000_osl2000_batch128",
        65536,
    )
    second = get_budget_mechanism_candidate(
        candidates,
        "tp4_dp2_ep8",
        "isl8000_osl2000_batch128",
        65536,
    )

    assert isinstance(first, VLLMBudgetMechanismCandidate)
    assert first.candidate_key == "tp8_dp1_ep8:isl8000_osl2000_batch128:max_bt65536"
    assert second.candidate_key == "tp4_dp2_ep8:isl8000_osl2000_batch128:max_bt65536"
    assert first.depenalized_budget_effect == pytest.approx(1.0)
    assert first.diagnostic_only is True
    assert first.valid_for_default is False
    assert first.perf_database is False


def test_missing_key_does_not_interpolate_or_extrapolate(tmp_path: Path) -> None:
    path = tmp_path / "candidate.csv"
    _write_csv(path, [_row(), _row(topology_key="tp4_dp2_ep8")])
    candidates = load_budget_mechanism_candidates(path)

    with pytest.raises(KeyError, match="exact budget mechanism candidate not found"):
        get_budget_mechanism_candidate(
            candidates,
            "tp8_dp1_ep8",
            "isl8000_osl2000_batch128",
            8000,
        )
    with pytest.raises(KeyError, match="exact budget mechanism candidate not found"):
        get_budget_mechanism_candidate(
            candidates,
            "tp2_dp4_ep8",
            "isl8000_osl2000_batch128",
            65536,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("diagnostic_only", "false", "diagnostic_only"),
        ("valid_for_default", "true", "valid_for_default"),
        ("perf_database", "true", "perf_database"),
    ],
)
def test_boundary_flags_fail_fast(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    path = tmp_path / "candidate.csv"
    first = _row()
    first[field] = value
    _write_csv(path, [first, _row(topology_key="tp4_dp2_ep8")])

    with pytest.raises(ValueError, match=message):
        load_budget_mechanism_candidates(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("depenalized_budget_effect", "2.0", "depenalized_budget_effect mismatch"),
        ("raw_error_ratio", "2.0", "raw_error_ratio mismatch"),
        ("depenalized_error_ratio", "2.0", "depenalized_error_ratio mismatch"),
        ("depenalized_is_closer", "false", "depenalized_is_closer"),
    ],
)
def test_ratio_guards_fail_fast(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    path = tmp_path / "candidate.csv"
    first = _row()
    first[field] = value
    _write_csv(path, [first, _row(topology_key="tp4_dp2_ep8")])

    with pytest.raises(ValueError, match=message):
        load_budget_mechanism_candidates(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("budget_steady_state_time_ms", "5000.000000"),
        ("baseline_steady_state_time_ms", "500.000000"),
    ],
)
def test_steady_state_time_ratio_guard_fails_fast(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    path = tmp_path / "candidate.csv"
    first = _row()
    first[field] = value
    _write_csv(path, [first, _row(topology_key="tp4_dp2_ep8")])

    with pytest.raises(ValueError, match="steady_state_time_ratio mismatch"):
        load_budget_mechanism_candidates(path)


def test_duplicate_key_fails(tmp_path: Path) -> None:
    path = tmp_path / "candidate.csv"
    _write_csv(path, [_row(), _row()])

    with pytest.raises(ValueError, match="duplicate candidate_key"):
        load_budget_mechanism_candidates(path)
