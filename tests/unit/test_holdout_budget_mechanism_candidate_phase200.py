from __future__ import annotations

import csv
from pathlib import Path

import pytest

from aiconfigurator.sdk.backends.cb_simulator import (
    VLLMHoldoutBudgetMechanismCandidate,
    get_holdout_budget_mechanism_candidate,
    holdout_budget_mechanism_candidate_from_row,
    load_holdout_budget_mechanism_candidates,
)


FIELDNAMES = [
    "source",
    "topology_key",
    "shape_key",
    "control_scenario",
    "holdout_scenario",
    "control_max_bt",
    "holdout_max_bt",
    "control_output_tok_s",
    "holdout_output_tok_s",
    "clean_high_over_control",
    "clean_delta_from_1",
    "control_steady_state_time_ms",
    "holdout_steady_state_time_ms",
    "steady_state_time_ratio",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]


ACCEPTED_KEYS = [
    "tp8_dp1_ep8:isl4000_osl2000_batch128:control_bt4000:holdout_bt65536",
    "tp4_dp2_ep8:isl4000_osl2000_batch128:control_bt4000:holdout_bt65536",
    "tp8_dp1_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536",
    "tp4_dp2_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536",
]


def _rows() -> list[dict[str, str]]:
    base = {
        "source": "phase199_holdout_budget_mechanism_analysis",
        "holdout_max_bt": "65536",
        "holdout_scenario": "holdout",
        "diagnostic_only": "true",
        "valid_for_default": "false",
        "perf_database": "false",
    }
    return [
        {
            **base,
            "topology_key": "tp8_dp1_ep8",
            "shape_key": "isl4000_osl2000_batch128",
            "control_scenario": "tp8ep8-4k2k-bt4000",
            "holdout_scenario": "tp8ep8-4k2k-bt65536",
            "control_max_bt": "4000",
            "control_output_tok_s": "100.000000",
            "holdout_output_tok_s": "105.000000",
            "clean_high_over_control": "1.050000",
            "clean_delta_from_1": "0.050000",
            "control_steady_state_time_ms": "10.000000",
            "holdout_steady_state_time_ms": "40.000000",
            "steady_state_time_ratio": "4.000000",
        },
        {
            **base,
            "topology_key": "tp4_dp2_ep8",
            "shape_key": "isl4000_osl2000_batch128",
            "control_scenario": "tp4dp2ep8-4k2k-bt4000",
            "holdout_scenario": "tp4dp2ep8-4k2k-bt65536",
            "control_max_bt": "4000",
            "control_output_tok_s": "200.000000",
            "holdout_output_tok_s": "240.000000",
            "clean_high_over_control": "1.200000",
            "clean_delta_from_1": "0.200000",
            "control_steady_state_time_ms": "20.000000",
            "holdout_steady_state_time_ms": "100.000000",
            "steady_state_time_ratio": "5.000000",
        },
        {
            **base,
            "topology_key": "tp8_dp1_ep8",
            "shape_key": "isl12000_osl2000_batch128",
            "control_scenario": "tp8ep8-12k2k-bt12000",
            "holdout_scenario": "tp8ep8-12k2k-bt65536",
            "control_max_bt": "12000",
            "control_output_tok_s": "80.000000",
            "holdout_output_tok_s": "84.000000",
            "clean_high_over_control": "1.050000",
            "clean_delta_from_1": "0.050000",
            "control_steady_state_time_ms": "25.000000",
            "holdout_steady_state_time_ms": "50.000000",
            "steady_state_time_ratio": "2.000000",
        },
        {
            **base,
            "topology_key": "tp4_dp2_ep8",
            "shape_key": "isl12000_osl2000_batch128",
            "control_scenario": "tp4dp2ep8-12k2k-bt12000",
            "holdout_scenario": "tp4dp2ep8-12k2k-bt65536",
            "control_max_bt": "12000",
            "control_output_tok_s": "300.000000",
            "holdout_output_tok_s": "297.000000",
            "clean_high_over_control": "0.990000",
            "clean_delta_from_1": "-0.010000",
            "control_steady_state_time_ms": "30.000000",
            "holdout_steady_state_time_ms": "120.000000",
            "steady_state_time_ratio": "4.000000",
        },
    ]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_load_and_query_four_exact_holdout_candidates(tmp_path: Path) -> None:
    path = tmp_path / "candidates.csv"
    _write_csv(path, _rows())

    candidates = load_holdout_budget_mechanism_candidates(path)

    assert set(candidates) == set(ACCEPTED_KEYS)
    for key in ACCEPTED_KEYS:
        topology_key, shape_key, control_part, holdout_part = key.split(":")
        candidate = get_holdout_budget_mechanism_candidate(
            candidates,
            topology_key,
            shape_key,
            int(control_part.removeprefix("control_bt")),
            int(holdout_part.removeprefix("holdout_bt")),
        )
        assert isinstance(candidate, VLLMHoldoutBudgetMechanismCandidate)
        assert candidate.candidate_key == key
        assert candidate.holdout_max_num_batched_tokens == 65536
        assert candidate.diagnostic_only is True
        assert candidate.valid_for_default is False
        assert candidate.perf_database is False


def test_from_row_builds_candidate_key_and_preserves_values() -> None:
    candidate = holdout_budget_mechanism_candidate_from_row(_rows()[0])

    assert candidate.candidate_key == ACCEPTED_KEYS[0]
    assert candidate.control_max_num_batched_tokens == 4000
    assert candidate.clean_high_over_control == pytest.approx(1.05)
    assert candidate.clean_delta_from_1 == pytest.approx(0.05)
    assert candidate.steady_state_time_ratio == pytest.approx(4.0)


def test_unknown_key_does_not_interpolate_or_extrapolate(tmp_path: Path) -> None:
    path = tmp_path / "candidates.csv"
    _write_csv(path, _rows())
    candidates = load_holdout_budget_mechanism_candidates(path)

    with pytest.raises(KeyError, match="exact holdout budget mechanism candidate not found"):
        get_holdout_budget_mechanism_candidate(
            candidates,
            "tp8_dp1_ep8",
            "isl4000_osl2000_batch128",
            8000,
            65536,
        )
    with pytest.raises(KeyError, match="exact holdout budget mechanism candidate not found"):
        get_holdout_budget_mechanism_candidate(
            candidates,
            "tp8_dp1_ep8",
            "isl4000_osl2000_batch128",
            4000,
            32768,
        )
    with pytest.raises(KeyError, match="exact holdout budget mechanism candidate not found"):
        get_holdout_budget_mechanism_candidate(
            candidates,
            "tp2_dp4_ep8",
            "isl4000_osl2000_batch128",
            4000,
            65536,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows.pop(), "4 rows"),
        (lambda rows: rows.__setitem__(1, dict(rows[0])), "duplicate candidate_key"),
        (lambda rows: rows[0].__setitem__("diagnostic_only", "false"), "diagnostic_only"),
        (lambda rows: rows[0].__setitem__("valid_for_default", "true"), "valid_for_default"),
        (lambda rows: rows[0].__setitem__("perf_database", "true"), "perf_database"),
        (lambda rows: rows[0].__setitem__("source", "wrong"), "source"),
        (lambda rows: rows[0].__setitem__("clean_high_over_control", "2.0"), "clean_high_over_control mismatch"),
        (lambda rows: rows[0].__setitem__("steady_state_time_ratio", "2.0"), "steady_state_time_ratio mismatch"),
        (lambda rows: rows[0].__setitem__("clean_delta_from_1", "2.0"), "clean_delta_from_1 mismatch"),
        (lambda rows: rows[0].__setitem__("holdout_max_bt", "32768"), "holdout_max_bt"),
    ],
)
def test_row_guards_fail_fast(tmp_path: Path, mutate, message: str) -> None:
    path = tmp_path / "candidates.csv"
    rows = _rows()
    mutate(rows)
    _write_csv(path, rows)

    with pytest.raises(ValueError, match=message):
        load_holdout_budget_mechanism_candidates(path)


def test_actual_phase199_csv_loads_four_exact_keys() -> None:
    csv_path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "iter_gap_investigation"
        / "phase199_holdout_budget_mechanism_analysis.csv"
    )

    candidates = load_holdout_budget_mechanism_candidates(csv_path)

    assert set(candidates) == set(ACCEPTED_KEYS)
    assert candidates[ACCEPTED_KEYS[0]].clean_high_over_control == pytest.approx(1.012212)
    assert candidates[ACCEPTED_KEYS[3]].clean_delta_from_1 == pytest.approx(-0.006191)
