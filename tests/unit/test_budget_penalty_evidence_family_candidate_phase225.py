from __future__ import annotations

import csv
from pathlib import Path

import pytest

from aiconfigurator.sdk.backends.cb_simulator import (
    VLLMBudgetPenaltyEvidenceFamilyCandidate,
    budget_penalty_evidence_family_candidate_from_row,
    get_budget_penalty_evidence_family_candidate,
    load_budget_penalty_evidence_family_candidates,
)


FIELDNAMES = [
    "source",
    "topology_key",
    "shape_key",
    "control_scenario",
    "holdout_scenario",
    "control_bt",
    "holdout_bt",
    "clean_budget_effect",
    "no_budget_penalty_error",
    "evidence_source",
    "min_clean_effect",
    "max_clean_effect",
    "clean_effect_spread",
    "shape_count",
    "recommended_model_boundary",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

ACCEPTED_KEYS = [
    "tp8_dp1_ep8:isl4000_osl2000_batch128:control_bt4000:holdout_bt65536",
    "tp8_dp1_ep8:isl8000_osl2000_batch128:control_bt8000:holdout_bt65536",
    "tp8_dp1_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536",
    "tp4_dp2_ep8:isl4000_osl2000_batch128:control_bt4000:holdout_bt65536",
    "tp4_dp2_ep8:isl8000_osl2000_batch128:control_bt8000:holdout_bt65536",
    "tp4_dp2_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536",
]


def _rows() -> list[dict[str, str]]:
    base = {
        "source": "phase222_budget_penalty_evidence_family",
        "holdout_bt": "65536",
        "shape_count": "3",
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
            "control_bt": "4000",
            "clean_budget_effect": "1.012212",
            "no_budget_penalty_error": "1.012212",
            "evidence_source": "phase215_budget_penalty_mechanism",
            "min_clean_effect": "0.987382",
            "max_clean_effect": "1.074279",
            "clean_effect_spread": "1.088007",
        },
        {
            **base,
            "topology_key": "tp8_dp1_ep8",
            "shape_key": "isl8000_osl2000_batch128",
            "control_scenario": "tp8ep8-bt8000",
            "holdout_scenario": "tp8ep8-bt65536",
            "control_bt": "8000",
            "clean_budget_effect": "0.987382",
            "no_budget_penalty_error": "1.012779",
            "evidence_source": "phase164_clean_gpu_budget_manifest",
            "min_clean_effect": "0.987382",
            "max_clean_effect": "1.074279",
            "clean_effect_spread": "1.088007",
        },
        {
            **base,
            "topology_key": "tp8_dp1_ep8",
            "shape_key": "isl12000_osl2000_batch128",
            "control_scenario": "tp8ep8-12k2k-bt12000",
            "holdout_scenario": "tp8ep8-12k2k-bt65536",
            "control_bt": "12000",
            "clean_budget_effect": "1.074279",
            "no_budget_penalty_error": "1.074279",
            "evidence_source": "phase215_budget_penalty_mechanism",
            "min_clean_effect": "0.987382",
            "max_clean_effect": "1.074279",
            "clean_effect_spread": "1.088007",
        },
        {
            **base,
            "topology_key": "tp4_dp2_ep8",
            "shape_key": "isl4000_osl2000_batch128",
            "control_scenario": "tp4dp2ep8-4k2k-bt4000",
            "holdout_scenario": "tp4dp2ep8-4k2k-bt65536",
            "control_bt": "4000",
            "clean_budget_effect": "1.133187",
            "no_budget_penalty_error": "1.133187",
            "evidence_source": "phase215_budget_penalty_mechanism",
            "min_clean_effect": "0.993809",
            "max_clean_effect": "1.167497",
            "clean_effect_spread": "1.174770",
        },
        {
            **base,
            "topology_key": "tp4_dp2_ep8",
            "shape_key": "isl8000_osl2000_batch128",
            "control_scenario": "tp4dp2ep8-bt8000",
            "holdout_scenario": "tp4dp2ep8-bt65536",
            "control_bt": "8000",
            "clean_budget_effect": "1.167497",
            "no_budget_penalty_error": "1.167497",
            "evidence_source": "phase164_clean_gpu_budget_manifest",
            "min_clean_effect": "0.993809",
            "max_clean_effect": "1.167497",
            "clean_effect_spread": "1.174770",
        },
        {
            **base,
            "topology_key": "tp4_dp2_ep8",
            "shape_key": "isl12000_osl2000_batch128",
            "control_scenario": "tp4dp2ep8-12k2k-bt12000",
            "holdout_scenario": "tp4dp2ep8-12k2k-bt65536",
            "control_bt": "12000",
            "clean_budget_effect": "0.993809",
            "no_budget_penalty_error": "1.006230",
            "evidence_source": "phase215_budget_penalty_mechanism",
            "min_clean_effect": "0.993809",
            "max_clean_effect": "1.167497",
            "clean_effect_spread": "1.174770",
        },
    ]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_load_and_query_six_exact_budget_penalty_evidence_family_candidates(tmp_path: Path) -> None:
    path = tmp_path / "phase222.csv"
    _write_csv(path, _rows())

    candidates = load_budget_penalty_evidence_family_candidates(path)

    assert set(candidates) == set(ACCEPTED_KEYS)
    for key in ACCEPTED_KEYS:
        topology_key, shape_key, control_part, holdout_part = key.split(":")
        candidate = get_budget_penalty_evidence_family_candidate(
            candidates,
            topology_key,
            shape_key,
            int(control_part.removeprefix("control_bt")),
            int(holdout_part.removeprefix("holdout_bt")),
        )
        assert isinstance(candidate, VLLMBudgetPenaltyEvidenceFamilyCandidate)
        assert candidate.candidate_key == key
        assert candidate.shape_count == 3
        assert candidate.recommended_model_boundary == "diagnostic_only_exact_key"
        assert candidate.default_readiness == "No-Go"
        assert candidate.diagnostic_only is True
        assert candidate.valid_for_default is False
        assert candidate.perf_database is False


def test_from_row_builds_candidate_key_and_preserves_values() -> None:
    candidate = budget_penalty_evidence_family_candidate_from_row(_rows()[1])

    assert candidate.candidate_key == ACCEPTED_KEYS[1]
    assert candidate.control_max_num_batched_tokens == 8000
    assert candidate.holdout_max_num_batched_tokens == 65536
    assert candidate.clean_budget_effect == pytest.approx(0.987382)
    assert candidate.no_budget_penalty_error == pytest.approx(1.012779)
    assert candidate.evidence_source == "phase164_clean_gpu_budget_manifest"
    assert candidate.clean_effect_spread == pytest.approx(1.088007)


def test_unknown_key_does_not_interpolate_or_extrapolate(tmp_path: Path) -> None:
    path = tmp_path / "phase222.csv"
    _write_csv(path, _rows())
    candidates = load_budget_penalty_evidence_family_candidates(path)

    with pytest.raises(KeyError, match="exact budget penalty evidence family candidate not found"):
        get_budget_penalty_evidence_family_candidate(
            candidates,
            "tp8_dp1_ep8",
            "isl4000_osl2000_batch128",
            8000,
            65536,
        )
    with pytest.raises(KeyError, match="exact budget penalty evidence family candidate not found"):
        get_budget_penalty_evidence_family_candidate(
            candidates,
            "tp8_dp1_ep8",
            "isl9000_osl2000_batch128",
            8000,
            65536,
        )
    with pytest.raises(KeyError, match="exact budget penalty evidence family candidate not found"):
        get_budget_penalty_evidence_family_candidate(
            candidates,
            "tp2_dp4_ep8",
            "isl8000_osl2000_batch128",
            8000,
            65536,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows.pop(), "6 rows"),
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
        (lambda rows: rows[0].__setitem__("shape_count", "2"), "shape_count"),
        (lambda rows: rows[0].__setitem__("clean_effect_spread", "9.000000"), "clean_effect_spread mismatch"),
        (lambda rows: rows[0].__setitem__("holdout_bt", "32768"), "holdout_bt"),
    ],
)
def test_row_guards_fail_fast(tmp_path: Path, mutate, message: str) -> None:
    path = tmp_path / "phase222.csv"
    rows = _rows()
    mutate(rows)
    _write_csv(path, rows)

    with pytest.raises(ValueError, match=message):
        load_budget_penalty_evidence_family_candidates(path)


def test_actual_phase222_csv_loads_six_exact_keys() -> None:
    csv_path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "iter_gap_investigation"
        / "phase222_budget_penalty_evidence_family.csv"
    )

    candidates = load_budget_penalty_evidence_family_candidates(csv_path)

    assert set(candidates) == set(ACCEPTED_KEYS)
    assert candidates[ACCEPTED_KEYS[1]].evidence_source == "phase164_clean_gpu_budget_manifest"
    assert candidates[ACCEPTED_KEYS[5]].no_budget_penalty_error == pytest.approx(1.00623)
