from __future__ import annotations

import csv
from pathlib import Path

import pytest

from aiconfigurator.sdk.backends.cb_simulator import (
    VLLMDeeperTraceTopologyFamilyCandidate,
    deeper_trace_topology_family_candidate_from_row,
    get_deeper_trace_topology_family_candidate,
    load_deeper_trace_topology_family_candidates,
)


FIELDNAMES = [
    "source",
    "family_key",
    "topology_key",
    "shape_key",
    "control_scenario",
    "holdout_scenario",
    "control_bt",
    "holdout_bt",
    "output_ratio",
    "throughput_direction",
    "holdout_scheduled_p99",
    "holdout_scheduled_max",
    "holdout_max_fill",
    "budget_ceiling_rejected",
    "verdict",
    "mechanism_conclusion",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

ACCEPTED_KEYS = [
    "tp8_dp1_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536",
    "tp4_dp2_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536",
]


def _row(
    *,
    topology_key: str,
    control_scenario: str,
    holdout_scenario: str,
    output_ratio: str,
    throughput_direction: str,
    holdout_scheduled_max: int,
    holdout_max_fill: str,
    verdict: str,
    mechanism_conclusion: str,
) -> dict[str, str]:
    return {
        "source": "phase303_deeper_trace_topology_family",
        "family_key": "deeper_trace_12k2k_topology_family",
        "topology_key": topology_key,
        "shape_key": "isl12000_osl2000_batch128",
        "control_scenario": control_scenario,
        "holdout_scenario": holdout_scenario,
        "control_bt": "12000",
        "holdout_bt": "65536",
        "output_ratio": output_ratio,
        "throughput_direction": throughput_direction,
        "holdout_scheduled_p99": "128",
        "holdout_scheduled_max": str(holdout_scheduled_max),
        "holdout_max_fill": holdout_max_fill,
        "budget_ceiling_rejected": "true",
        "verdict": verdict,
        "mechanism_conclusion": mechanism_conclusion,
        "default_readiness": "No-Go",
        "diagnostic_only": "true",
        "valid_for_default": "false",
        "perf_database": "false",
    }


def _rows() -> list[dict[str, str]]:
    return [
        _row(
            topology_key="tp8_dp1_ep8",
            control_scenario="tp8ep8-12k2k-bt12000",
            holdout_scenario="tp8ep8-12k2k-bt65536",
            output_ratio="0.969300",
            throughput_direction="holdout_slower",
            holdout_scheduled_max=12000,
            holdout_max_fill="0.183105",
            verdict="partial_only",
            mechanism_conclusion="boundary_mixed_overhead_partial_only",
        ),
        _row(
            topology_key="tp4_dp2_ep8",
            control_scenario="tp4dp2ep8-12k2k-bt12000",
            holdout_scenario="tp4dp2ep8-12k2k-bt65536",
            output_ratio="1.321996",
            throughput_direction="holdout_faster",
            holdout_scheduled_max=24736,
            holdout_max_fill="0.188721",
            verdict="boundary_timeline_explains_direction",
            mechanism_conclusion="wall_span_iteration_cadence_diagnostic",
        ),
    ]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_loads_two_deeper_trace_topology_family_candidates(tmp_path: Path) -> None:
    path = tmp_path / "phase303.csv"
    _write_csv(path, _rows())

    candidates = load_deeper_trace_topology_family_candidates(path)

    assert set(candidates) == set(ACCEPTED_KEYS)
    tp8 = get_deeper_trace_topology_family_candidate(
        candidates,
        "tp8_dp1_ep8",
        "isl12000_osl2000_batch128",
        12000,
        65536,
    )
    tp4 = get_deeper_trace_topology_family_candidate(
        candidates,
        "tp4_dp2_ep8",
        "isl12000_osl2000_batch128",
        12000,
        65536,
    )

    assert isinstance(tp8, VLLMDeeperTraceTopologyFamilyCandidate)
    assert tp8.output_ratio == pytest.approx(0.969300)
    assert tp8.throughput_direction == "holdout_slower"
    assert tp8.holdout_scheduled_p99 == 128
    assert tp8.holdout_scheduled_max == 12000
    assert tp8.holdout_max_fill == pytest.approx(0.183105)
    assert tp8.budget_ceiling_rejected is True
    assert tp8.verdict == "partial_only"
    assert tp8.mechanism_conclusion == "boundary_mixed_overhead_partial_only"
    assert tp8.default_readiness == "No-Go"
    assert tp8.diagnostic_only is True
    assert tp8.valid_for_default is False
    assert tp8.perf_database is False

    assert tp4.output_ratio == pytest.approx(1.321996)
    assert tp4.throughput_direction == "holdout_faster"
    assert tp4.holdout_scheduled_max == 24736
    assert tp4.holdout_max_fill == pytest.approx(0.188721)


def test_from_row_builds_exact_candidate_key_and_preserves_values() -> None:
    candidate = deeper_trace_topology_family_candidate_from_row(_rows()[1])

    assert candidate.candidate_key == ACCEPTED_KEYS[1]
    assert candidate.source == "phase303_deeper_trace_topology_family"
    assert candidate.family_key == "deeper_trace_12k2k_topology_family"
    assert candidate.topology_key == "tp4_dp2_ep8"
    assert candidate.shape_key == "isl12000_osl2000_batch128"
    assert candidate.control_scenario == "tp4dp2ep8-12k2k-bt12000"
    assert candidate.holdout_scenario == "tp4dp2ep8-12k2k-bt65536"
    assert candidate.control_bt == 12000
    assert candidate.holdout_bt == 65536
    assert candidate.output_ratio == pytest.approx(1.321996)
    assert candidate.throughput_direction == "holdout_faster"
    assert candidate.budget_ceiling_rejected is True


def test_unknown_key_rejects_interpolation_and_extrapolation(tmp_path: Path) -> None:
    path = tmp_path / "phase303.csv"
    _write_csv(path, _rows())
    candidates = load_deeper_trace_topology_family_candidates(path)

    for topology_key, shape_key, control_bt, holdout_bt in [
        ("tp8_dp1_ep8", "isl8000_osl2000_batch128", 8000, 65536),
        ("tp2_dp4_ep8", "isl12000_osl2000_batch128", 12000, 65536),
        ("tp8_dp1_ep8", "isl4000_osl2000_batch128", 4000, 65536),
        ("tp8_dp1_ep8", "isl12000_osl2000_batch128", 4000, 65536),
        ("tp8_dp1_ep8", "isl12000_osl2000_batch128", 12000, 32768),
    ]:
        with pytest.raises(KeyError, match="exact deeper trace topology family candidate not found"):
            get_deeper_trace_topology_family_candidate(
                candidates,
                topology_key,
                shape_key,
                control_bt,
                holdout_bt,
            )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows.pop(), "2 rows"),
        (lambda rows: rows.__setitem__(1, dict(rows[0])), "duplicate candidate_key"),
        (lambda rows: rows[0].__setitem__("source", "wrong"), "source"),
        (lambda rows: rows[0].__setitem__("family_key", "wrong"), "family_key"),
        (lambda rows: rows[0].__setitem__("shape_key", "isl8000_osl2000_batch128"), "candidate_key"),
        (lambda rows: rows[0].__setitem__("control_bt", "4000"), "candidate_key"),
        (lambda rows: rows[0].__setitem__("holdout_bt", "32768"), "holdout_bt"),
        (lambda rows: rows[0].__setitem__("diagnostic_only", "false"), "diagnostic_only"),
        (lambda rows: rows[0].__setitem__("valid_for_default", "true"), "valid_for_default"),
        (lambda rows: rows[0].__setitem__("perf_database", "true"), "perf_database"),
        (lambda rows: rows[0].__setitem__("default_readiness", "Go"), "default_readiness"),
        (lambda rows: rows[0].__setitem__("budget_ceiling_rejected", "false"), "budget_ceiling_rejected"),
        (lambda rows: rows[0].__setitem__("throughput_direction", "holdout_faster"), "throughput_direction"),
        (lambda rows: rows[0].__setitem__("output_ratio", "1.100000"), "output_ratio"),
        (lambda rows: rows[0].__setitem__("holdout_max_fill", "0.600000"), "holdout_max_fill"),
    ],
)
def test_row_guards_fail_fast(tmp_path: Path, mutate, message: str) -> None:
    path = tmp_path / "phase303.csv"
    rows = _rows()
    mutate(rows)
    _write_csv(path, rows)

    with pytest.raises(ValueError, match=message):
        load_deeper_trace_topology_family_candidates(path)


def test_actual_phase303_csv_loads_two_exact_keys() -> None:
    csv_path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "iter_gap_investigation"
        / "phase303_deeper_trace_topology_family.csv"
    )

    candidates = load_deeper_trace_topology_family_candidates(csv_path)

    assert set(candidates) == set(ACCEPTED_KEYS)
    assert candidates[ACCEPTED_KEYS[0]].throughput_direction == "holdout_slower"
    assert candidates[ACCEPTED_KEYS[0]].output_ratio == pytest.approx(0.969300)
    assert candidates[ACCEPTED_KEYS[1]].throughput_direction == "holdout_faster"
    assert candidates[ACCEPTED_KEYS[1]].output_ratio == pytest.approx(1.321996)
