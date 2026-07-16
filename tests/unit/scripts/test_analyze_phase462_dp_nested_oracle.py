from __future__ import annotations

import math

import pytest

import scripts.analyze_phase462_dp_nested_oracle as analysis


def _row(kind: str, ordinal: int, rank: int, ts: int, step: int = 0):
    row = {
        "kind": kind,
        "arrival_ordinal": ordinal,
        "wall_ts_ns": ts,
    }
    if kind == "route":
        row["chosen_rank"] = rank
    else:
        row["dp_rank"] = rank
    if kind == "admit":
        row["schedule_seq"] = step
    return row


def test_build_oracle_inputs_uses_measured_chain_without_free_parameters() -> None:
    rows = [
        _row("route", 0, 1, 100),
        _row("receive", 0, 1, 160),
        _row("admit", 0, 1, 220, 7),
        _row("route", 1, 0, 110),
        _row("receive", 1, 0, 190),
        _row("admit", 1, 0, 260, 9),
    ]

    oracle = analysis.build_oracle_inputs(rows, expected_requests=2)

    assert oracle[0] == analysis.OracleInput(1, 0.00006, 7)
    assert oracle[1] == analysis.OracleInput(0, 0.00008, 9)


def test_build_oracle_inputs_rejects_rank_mismatch() -> None:
    rows = [
        _row("route", 0, 1, 100),
        _row("receive", 0, 0, 160),
        _row("admit", 0, 1, 220, 7),
    ]

    with pytest.raises(AssertionError, match="rank_mismatch"):
        analysis.build_oracle_inputs(rows, expected_requests=1)


def test_cross_rank_spread_uses_per_rank_cell_medians() -> None:
    trace = [
        {
            "replica_id": 0,
            "prefill_tokens": 92,
            "total_tokens": 100,
            "decode_reqs": 8,
            "latency_ms": 10.0,
        },
        {
            "replica_id": 0,
            "prefill_tokens": 92,
            "total_tokens": 100,
            "decode_reqs": 8,
            "latency_ms": 14.0,
        },
        {
            "replica_id": 1,
            "prefill_tokens": 92,
            "total_tokens": 100,
            "decode_reqs": 8,
            "latency_ms": 3.0,
        },
        {
            "replica_id": 1,
            "prefill_tokens": 92,
            "total_tokens": 100,
            "decode_reqs": 8,
            "latency_ms": 5.0,
        },
    ]

    result = analysis.same_cell_cross_rank_spread(trace)

    assert result.cell == (100, 8)
    assert result.max_over_min == 3.0


def test_attribution_keeps_official_bridge_outside_route_chain() -> None:
    measurements = [
        analysis.OracleMeasurement("O0", 1.09, 1.0, 1.0, 0.0),
        analysis.OracleMeasurement("O1", 1.08, 0.5, 2.0, 0.0),
        analysis.OracleMeasurement("O2", 1.06, 0.25, 4.0, 0.0),
        analysis.OracleMeasurement("O3", 1.01, 0.0, 8.0, 1.0),
    ]

    rows = analysis.attribution_rows(official_error=1.20, measurements=measurements)

    assert rows[0].segment == "official_to_multi_replica_architecture"
    assert rows[0].delta_error == pytest.approx(-0.11)
    assert rows[1].segment == "route_rank_sequence"
    assert rows[1].delta_error == pytest.approx(-0.01)
    assert math.isclose(sum(row.delta_error for row in rows[1:]), -0.08)


def test_admit_lag_summary_exposes_unclosed_oracle() -> None:
    oracle = {
        0: analysis.OracleInput(0, 0.0, 3),
        1: analysis.OracleInput(1, 0.0, 5),
        2: analysis.OracleInput(0, 0.0, 7),
    }

    result = analysis.admit_lag_summary({0: 3, 1: 6, 2: 10}, oracle)

    assert result == {
        "exact": 1,
        "early": 0,
        "late": 2,
        "median_lag_steps": 1.0,
        "max_lag_steps": 3,
    }
