from __future__ import annotations

import csv
from pathlib import Path

import pytest

from aiconfigurator.sdk.backends.cb_simulator import (
    VLLMBudgetPenaltyMechanismCandidate,
    budget_penalty_mechanism_candidate_from_row,
    get_budget_penalty_mechanism_candidate,
    load_budget_penalty_mechanism_candidates,
)


FIELDNAMES = [
    "source",
    "topology_key",
    "shape_key",
    "control_scenario",
    "holdout_scenario",
    "control_max_bt",
    "holdout_max_bt",
    "clean_budget_effect",
    "steady_state_time_ratio",
    "steady_linear_error",
    "no_budget_penalty_error",
    "topology_shape_spread",
    "recommended_model_boundary",
    "default_readiness",
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
        "source": "phase215_budget_penalty_mechanism",
        "holdout_max_bt": "65536",
        "recommended_model_boundary": "diagnostic_only_exact_key",
        "default_readiness": "No-Go",
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
            "clean_budget_effect": "1.050000",
            "steady_state_time_ratio": "4.000000",
            "steady_linear_error": "3.809524",
            "no_budget_penalty_error": "1.050000",
            "topology_shape_spread": "1.047619",
        },
        {
            **base,
            "topology_key": "tp4_dp2_ep8",
            "shape_key": "isl4000_osl2000_batch128",
            "control_scenario": "tp4dp2ep8-4k2k-bt4000",
            "holdout_scenario": "tp4dp2ep8-4k2k-bt65536",
            "control_max_bt": "4000",
            "clean_budget_effect": "1.200000",
            "steady_state_time_ratio": "5.000000",
            "steady_linear_error": "4.166667",
            "no_budget_penalty_error": "1.200000",
            "topology_shape_spread": "1.212121",
        },
        {
            **base,
            "topology_key": "tp8_dp1_ep8",
            "shape_key": "isl12000_osl2000_batch128",
            "control_scenario": "tp8ep8-12k2k-bt12000",
            "holdout_scenario": "tp8ep8-12k2k-bt65536",
            "control_max_bt": "12000",
            "clean_budget_effect": "1.100000",
            "steady_state_time_ratio": "2.000000",
            "steady_linear_error": "1.818182",
            "no_budget_penalty_error": "1.100000",
            "topology_shape_spread": "1.047619",
        },
        {
            **base,
            "topology_key": "tp4_dp2_ep8",
            "shape_key": "isl12000_osl2000_batch128",
            "control_scenario": "tp4dp2ep8-12k2k-bt12000",
            "holdout_scenario": "tp4dp2ep8-12k2k-bt65536",
            "control_max_bt": "12000",
            "clean_budget_effect": "0.990000",
            "steady_state_time_ratio": "4.000000",
            "steady_linear_error": "4.040404",
            "no_budget_penalty_error": "1.010101",
            "topology_shape_spread": "1.212121",
        },
    ]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_load_and_query_four_exact_budget_penalty_candidates(tmp_path: Path) -> None:
    path = tmp_path / "phase215.csv"
    _write_csv(path, _rows())

    candidates = load_budget_penalty_mechanism_candidates(path)

    assert set(candidates) == set(ACCEPTED_KEYS)
    for key in ACCEPTED_KEYS:
        topology_key, shape_key, control_part, holdout_part = key.split(":")
        candidate = get_budget_penalty_mechanism_candidate(
            candidates,
            topology_key,
            shape_key,
            int(control_part.removeprefix("control_bt")),
            int(holdout_part.removeprefix("holdout_bt")),
        )
        assert isinstance(candidate, VLLMBudgetPenaltyMechanismCandidate)
        assert candidate.candidate_key == key
        assert candidate.recommended_model_boundary == "diagnostic_only_exact_key"
        assert candidate.default_readiness == "No-Go"
        assert candidate.diagnostic_only is True
        assert candidate.valid_for_default is False
        assert candidate.perf_database is False


def test_from_row_builds_candidate_key_and_preserves_values() -> None:
    candidate = budget_penalty_mechanism_candidate_from_row(_rows()[0])

    assert candidate.candidate_key == ACCEPTED_KEYS[0]
    assert candidate.control_max_num_batched_tokens == 4000
    assert candidate.holdout_max_num_batched_tokens == 65536
    assert candidate.clean_budget_effect == pytest.approx(1.05)
    assert candidate.steady_state_time_ratio == pytest.approx(4.0)
    assert candidate.steady_linear_error == pytest.approx(3.809524)
    assert candidate.no_budget_penalty_error == pytest.approx(1.05)
    assert candidate.topology_shape_spread == pytest.approx(1.047619)


def test_unknown_key_does_not_interpolate_or_extrapolate(tmp_path: Path) -> None:
    path = tmp_path / "phase215.csv"
    _write_csv(path, _rows())
    candidates = load_budget_penalty_mechanism_candidates(path)

    with pytest.raises(KeyError, match="exact budget penalty mechanism candidate not found"):
        get_budget_penalty_mechanism_candidate(
            candidates,
            "tp8_dp1_ep8",
            "isl4000_osl2000_batch128",
            8000,
            65536,
        )
    with pytest.raises(KeyError, match="exact budget penalty mechanism candidate not found"):
        get_budget_penalty_mechanism_candidate(
            candidates,
            "tp8_dp1_ep8",
            "isl4000_osl2000_batch999",
            4000,
            65536,
        )
    with pytest.raises(KeyError, match="exact budget penalty mechanism candidate not found"):
        get_budget_penalty_mechanism_candidate(
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
        (lambda rows: rows[0].__setitem__("default_readiness", "Go"), "default_readiness"),
        (
            lambda rows: rows[0].__setitem__("recommended_model_boundary", "global_constant"),
            "recommended_model_boundary",
        ),
        (lambda rows: rows[0].__setitem__("steady_linear_error", "1.000000"), "steady_linear_error mismatch"),
        (lambda rows: rows[0].__setitem__("no_budget_penalty_error", "2.000000"), "no_budget_penalty_error mismatch"),
        (lambda rows: rows[0].__setitem__("topology_shape_spread", "9.000000"), "topology_shape_spread mismatch"),
        (lambda rows: rows[0].__setitem__("holdout_max_bt", "32768"), "holdout_max_bt"),
    ],
)
def test_row_guards_fail_fast(tmp_path: Path, mutate, message: str) -> None:
    path = tmp_path / "phase215.csv"
    rows = _rows()
    mutate(rows)
    _write_csv(path, rows)

    with pytest.raises(ValueError, match=message):
        load_budget_penalty_mechanism_candidates(path)


def test_actual_phase215_csv_loads_four_exact_keys() -> None:
    csv_path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "iter_gap_investigation"
        / "phase215_budget_penalty_mechanism.csv"
    )

    candidates = load_budget_penalty_mechanism_candidates(csv_path)

    assert set(candidates) == set(ACCEPTED_KEYS)
    assert candidates[ACCEPTED_KEYS[0]].steady_linear_error == pytest.approx(3.644068)
    assert candidates[ACCEPTED_KEYS[3]].no_budget_penalty_error == pytest.approx(1.00623)
