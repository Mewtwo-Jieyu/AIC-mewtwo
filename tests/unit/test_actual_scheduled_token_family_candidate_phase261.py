from __future__ import annotations

import csv
from pathlib import Path

import pytest

from aiconfigurator.sdk.backends.cb_simulator import (
    VLLMActualScheduledTokenFamilyCandidate,
    actual_scheduled_token_family_candidate_from_pair,
    get_actual_scheduled_token_family_candidate,
    load_actual_scheduled_token_family_candidates,
)


FIELDNAMES = [
    "source",
    "topology_key",
    "scenario",
    "role",
    "shape_key",
    "control_max_bt",
    "holdout_max_bt",
    "max_bt",
    "configured_budget_per_rank",
    "configured_budget_aggregate",
    "output_tok_s",
    "output_ratio_vs_control",
    "raw_trace_rows",
    "unique_iterations",
    "rank_min_p50_scheduled_total_tokens",
    "rank_min_p95_scheduled_total_tokens",
    "rank_min_p99_scheduled_total_tokens",
    "rank_min_max_scheduled_total_tokens",
    "rank_max_p50_scheduled_total_tokens",
    "rank_max_p95_scheduled_total_tokens",
    "rank_max_p99_scheduled_total_tokens",
    "rank_max_max_scheduled_total_tokens",
    "rank_sum_p50_scheduled_total_tokens",
    "rank_sum_p95_scheduled_total_tokens",
    "rank_sum_p99_scheduled_total_tokens",
    "rank_sum_max_scheduled_total_tokens",
    "rank_sum_mean_fill_ratio",
    "rank_sum_p99_fill_ratio",
    "rank_sum_max_fill_ratio",
    "mechanism_hypothesis",
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


def _row(
    *,
    topology_key: str,
    scenario: str,
    role: str,
    shape_key: str,
    control_max_bt: int,
    holdout_max_bt: int,
    max_bt: int,
    configured_budget_aggregate: int,
    output_tok_s: float,
    output_ratio_vs_control: float,
    rank_sum_p99: int,
    rank_sum_max: int,
    rank_sum_p99_fill: float,
    rank_sum_max_fill: float,
) -> dict[str, str]:
    return {
        "source": "phase258_actual_scheduled_token_full_family",
        "topology_key": topology_key,
        "scenario": scenario,
        "role": role,
        "shape_key": shape_key,
        "control_max_bt": str(control_max_bt),
        "holdout_max_bt": str(holdout_max_bt),
        "max_bt": str(max_bt),
        "configured_budget_per_rank": str(max_bt),
        "configured_budget_aggregate": str(configured_budget_aggregate),
        "output_tok_s": f"{output_tok_s:.6f}",
        "output_ratio_vs_control": f"{output_ratio_vs_control:.6f}",
        "raw_trace_rows": "16000",
        "unique_iterations": "2000",
        "rank_min_p50_scheduled_total_tokens": "64.000000",
        "rank_min_p95_scheduled_total_tokens": "64.000000",
        "rank_min_p99_scheduled_total_tokens": "64.000000",
        "rank_min_max_scheduled_total_tokens": str(max_bt),
        "rank_max_p50_scheduled_total_tokens": "64.000000",
        "rank_max_p95_scheduled_total_tokens": "64.000000",
        "rank_max_p99_scheduled_total_tokens": "64.000000",
        "rank_max_max_scheduled_total_tokens": str(max_bt),
        "rank_sum_p50_scheduled_total_tokens": "128.000000",
        "rank_sum_p95_scheduled_total_tokens": "128.000000",
        "rank_sum_p99_scheduled_total_tokens": f"{rank_sum_p99:.6f}",
        "rank_sum_max_scheduled_total_tokens": str(rank_sum_max),
        "rank_sum_mean_fill_ratio": "0.001000",
        "rank_sum_p99_fill_ratio": f"{rank_sum_p99_fill:.6f}",
        "rank_sum_max_fill_ratio": f"{rank_sum_max_fill:.6f}",
        "mechanism_hypothesis": "actual_scheduled_tokens_not_configured_budget",
        "default_readiness": "No-Go",
        "diagnostic_only": "true",
        "valid_for_default": "false",
        "perf_database": "false",
    }


def _rows() -> list[dict[str, str]]:
    return [
        _row(
            topology_key="tp8_dp1_ep8",
            scenario="tp8ep8-4k2k-bt4000",
            role="control",
            shape_key="isl4000_osl2000_batch128",
            control_max_bt=4000,
            holdout_max_bt=65536,
            max_bt=4000,
            configured_budget_aggregate=4000,
            output_tok_s=4071.651448,
            output_ratio_vs_control=1.0,
            rank_sum_p99=128,
            rank_sum_max=4000,
            rank_sum_p99_fill=0.032,
            rank_sum_max_fill=1.0,
        ),
        _row(
            topology_key="tp8_dp1_ep8",
            scenario="tp8ep8-4k2k-bt65536",
            role="holdout",
            shape_key="isl4000_osl2000_batch128",
            control_max_bt=4000,
            holdout_max_bt=65536,
            max_bt=65536,
            configured_budget_aggregate=65536,
            output_tok_s=4027.060291,
            output_ratio_vs_control=0.989048,
            rank_sum_p99=128,
            rank_sum_max=4000,
            rank_sum_p99_fill=0.001953,
            rank_sum_max_fill=0.061035,
        ),
        _row(
            topology_key="tp4_dp2_ep8",
            scenario="tp4dp2ep8-4k2k-bt4000",
            role="control",
            shape_key="isl4000_osl2000_batch128",
            control_max_bt=4000,
            holdout_max_bt=65536,
            max_bt=4000,
            configured_budget_aggregate=8000,
            output_tok_s=3143.891608,
            output_ratio_vs_control=1.0,
            rank_sum_p99=128,
            rank_sum_max=8000,
            rank_sum_p99_fill=0.016,
            rank_sum_max_fill=1.0,
        ),
        _row(
            topology_key="tp4_dp2_ep8",
            scenario="tp4dp2ep8-4k2k-bt65536",
            role="holdout",
            shape_key="isl4000_osl2000_batch128",
            control_max_bt=4000,
            holdout_max_bt=65536,
            max_bt=65536,
            configured_budget_aggregate=131072,
            output_tok_s=3445.355421,
            output_ratio_vs_control=1.095889,
            rank_sum_p99=128,
            rank_sum_max=8240,
            rank_sum_p99_fill=0.000977,
            rank_sum_max_fill=0.062866,
        ),
        _row(
            topology_key="tp8_dp1_ep8",
            scenario="tp8ep8-12k2k-bt12000",
            role="control",
            shape_key="isl12000_osl2000_batch128",
            control_max_bt=12000,
            holdout_max_bt=65536,
            max_bt=12000,
            configured_budget_aggregate=12000,
            output_tok_s=2777.513790,
            output_ratio_vs_control=1.0,
            rank_sum_p99=128,
            rank_sum_max=12000,
            rank_sum_p99_fill=0.010667,
            rank_sum_max_fill=1.0,
        ),
        _row(
            topology_key="tp8_dp1_ep8",
            scenario="tp8ep8-12k2k-bt65536",
            role="holdout",
            shape_key="isl12000_osl2000_batch128",
            control_max_bt=12000,
            holdout_max_bt=65536,
            max_bt=65536,
            configured_budget_aggregate=65536,
            output_tok_s=2687.912821,
            output_ratio_vs_control=0.967741,
            rank_sum_p99=128,
            rank_sum_max=12000,
            rank_sum_p99_fill=0.001953,
            rank_sum_max_fill=0.183105,
        ),
        _row(
            topology_key="tp4_dp2_ep8",
            scenario="tp4dp2ep8-12k2k-bt12000",
            role="control",
            shape_key="isl12000_osl2000_batch128",
            control_max_bt=12000,
            holdout_max_bt=65536,
            max_bt=12000,
            configured_budget_aggregate=24000,
            output_tok_s=2091.977129,
            output_ratio_vs_control=1.0,
            rank_sum_p99=128,
            rank_sum_max=24000,
            rank_sum_p99_fill=0.005333,
            rank_sum_max_fill=1.0,
        ),
        _row(
            topology_key="tp4_dp2_ep8",
            scenario="tp4dp2ep8-12k2k-bt65536",
            role="holdout",
            shape_key="isl12000_osl2000_batch128",
            control_max_bt=12000,
            holdout_max_bt=65536,
            max_bt=65536,
            configured_budget_aggregate=131072,
            output_tok_s=2802.211032,
            output_ratio_vs_control=1.339504,
            rank_sum_p99=128,
            rank_sum_max=24000,
            rank_sum_p99_fill=0.000977,
            rank_sum_max_fill=0.183105,
        ),
    ]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_load_and_query_four_actual_scheduled_token_family_candidates(tmp_path: Path) -> None:
    path = tmp_path / "phase258.csv"
    _write_csv(path, _rows())

    candidates = load_actual_scheduled_token_family_candidates(path)

    assert set(candidates) == set(ACCEPTED_KEYS)
    for key in ACCEPTED_KEYS:
        topology_key, shape_key, control_part, holdout_part = key.split(":")
        candidate = get_actual_scheduled_token_family_candidate(
            candidates,
            topology_key,
            shape_key,
            int(control_part.removeprefix("control_bt")),
            int(holdout_part.removeprefix("holdout_bt")),
        )
        assert isinstance(candidate, VLLMActualScheduledTokenFamilyCandidate)
        assert candidate.candidate_key == key
        assert candidate.mechanism_hypothesis == "actual_scheduled_tokens_not_configured_budget"
        assert candidate.default_readiness == "No-Go"
        assert candidate.diagnostic_only is True
        assert candidate.valid_for_default is False
        assert candidate.perf_database is False


def test_from_pair_builds_candidate_key_and_preserves_pair_values() -> None:
    candidate = actual_scheduled_token_family_candidate_from_pair(_rows()[0], _rows()[1])

    assert candidate.candidate_key == ACCEPTED_KEYS[0]
    assert candidate.control_scenario == "tp8ep8-4k2k-bt4000"
    assert candidate.holdout_scenario == "tp8ep8-4k2k-bt65536"
    assert candidate.control_max_num_batched_tokens == 4000
    assert candidate.holdout_max_num_batched_tokens == 65536
    assert candidate.control_configured_budget_aggregate == 4000
    assert candidate.holdout_configured_budget_aggregate == 65536
    assert candidate.control_output_tok_s == pytest.approx(4071.651448)
    assert candidate.holdout_output_tok_s == pytest.approx(4027.060291)
    assert candidate.output_ratio_vs_control == pytest.approx(0.989048)
    assert candidate.holdout_rank_sum_p99_fill_ratio == pytest.approx(0.001953)
    assert candidate.holdout_rank_sum_max_fill_ratio == pytest.approx(0.061035)


def test_unknown_key_does_not_interpolate_or_extrapolate(tmp_path: Path) -> None:
    path = tmp_path / "phase258.csv"
    _write_csv(path, _rows())
    candidates = load_actual_scheduled_token_family_candidates(path)

    for topology_key, shape_key, control_bt, holdout_bt in [
        ("tp8_dp1_ep8", "isl8000_osl2000_batch128", 8000, 65536),
        ("tp2_dp4_ep8", "isl4000_osl2000_batch128", 4000, 65536),
        ("tp8_dp1_ep8", "isl4000_osl2000_batch128", 12000, 65536),
        ("tp8_dp1_ep8", "isl4000_osl2000_batch128", 4000, 32768),
    ]:
        with pytest.raises(KeyError, match="exact actual scheduled token family candidate not found"):
            get_actual_scheduled_token_family_candidate(
                candidates,
                topology_key,
                shape_key,
                control_bt,
                holdout_bt,
            )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows.pop(), "8 rows"),
        (lambda rows: rows.__setitem__(1, dict(rows[0])), "pair must contain exactly one control and one holdout"),
        (lambda rows: rows[0].__setitem__("diagnostic_only", "false"), "diagnostic_only"),
        (lambda rows: rows[0].__setitem__("valid_for_default", "true"), "valid_for_default"),
        (lambda rows: rows[0].__setitem__("perf_database", "true"), "perf_database"),
        (lambda rows: rows[0].__setitem__("source", "wrong"), "source"),
        (lambda rows: rows[0].__setitem__("default_readiness", "Go"), "default_readiness"),
        (lambda rows: rows[0].__setitem__("mechanism_hypothesis", "global_constant"), "mechanism_hypothesis"),
        (lambda rows: rows[1].__setitem__("rank_sum_p99_fill_ratio", "9.000000"), "rank_sum_p99_fill_ratio mismatch"),
        (lambda rows: rows[1].__setitem__("rank_sum_max_fill_ratio", "9.000000"), "rank_sum_max_fill_ratio mismatch"),
        (lambda rows: rows[1].__setitem__("configured_budget_aggregate", "65535"), "configured_budget_aggregate"),
        (lambda rows: rows[1].__setitem__("holdout_max_bt", "32768"), "holdout_max_bt"),
    ],
)
def test_row_guards_fail_fast(tmp_path: Path, mutate, message: str) -> None:
    path = tmp_path / "phase258.csv"
    rows = _rows()
    mutate(rows)
    _write_csv(path, rows)

    with pytest.raises(ValueError, match=message):
        load_actual_scheduled_token_family_candidates(path)


def test_actual_phase258_csv_loads_four_exact_keys() -> None:
    csv_path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "iter_gap_investigation"
        / "phase258_actual_scheduled_token_full_family.csv"
    )

    candidates = load_actual_scheduled_token_family_candidates(csv_path)

    assert set(candidates) == set(ACCEPTED_KEYS)
    assert candidates[ACCEPTED_KEYS[1]].holdout_rank_sum_max_scheduled_total_tokens == 8240
    assert candidates[ACCEPTED_KEYS[3]].output_ratio_vs_control == pytest.approx(1.339504)
