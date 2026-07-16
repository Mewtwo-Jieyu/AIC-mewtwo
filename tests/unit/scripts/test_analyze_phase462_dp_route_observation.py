from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.analyze_phase462_dp_route_observation as analysis


def _chain(ordinal: int, rank: int, admit_step: int) -> list[dict[str, object]]:
    trace_id = f"phase462-dp-route-{ordinal:06d}"
    base = ordinal * 1000
    return [
        {
            "kind": "route",
            "trace_id": trace_id,
            "arrival_ordinal": ordinal,
            "chosen_rank": rank,
            "wall_ts_ns": base,
            "score_snapshot": [
                {"rank": 0, "waiting": 0, "running": 0, "score": 0},
                {"rank": 1, "waiting": 0, "running": 0, "score": 0},
            ],
        },
        {
            "kind": "receive",
            "trace_id": trace_id,
            "arrival_ordinal": ordinal,
            "dp_rank": rank,
            "wall_ts_ns": base + 100,
        },
        {
            "kind": "admit",
            "trace_id": trace_id,
            "arrival_ordinal": ordinal,
            "dp_rank": rank,
            "schedule_seq": admit_step,
            "wall_ts_ns": base + 200,
        },
    ]


@pytest.mark.parametrize(
    ("chains", "expected"),
    [
        (
            _chain(0, 0, 10)
            + _chain(1, 0, 11)
            + _chain(2, 0, 12)
            + _chain(3, 0, 13)
            + _chain(4, 1, 20)
            + _chain(5, 1, 21),
            "H-route",
        ),
        (
            _chain(0, 0, 10)
            + _chain(1, 1, 20)
            + _chain(2, 0, 12)
            + _chain(3, 1, 21),
            "H-drain",
        ),
        (
            _chain(0, 0, 10)
            + _chain(1, 0, 11)
            + _chain(2, 1, 20)
            + _chain(3, 1, 21),
            "H-route",
        ),
        (
            _chain(0, 0, 10)
            + _chain(1, 0, 12)
            + _chain(2, 0, 14)
            + _chain(3, 0, 16)
            + _chain(4, 1, 20)
            + _chain(5, 1, 21),
            "H-mixed",
        ),
    ],
)
def test_verdict_follows_locked_route_then_drain_order(
    chains: list[dict[str, object]], expected: str
) -> None:
    result = analysis.judge_rows(chains, expected_requests=len(chains) // 3)
    assert result.verdict == expected


def test_integrity_rejects_a_missing_admit() -> None:
    rows = _chain(0, 0, 10) + _chain(1, 1, 20)
    rows = [
        row
        for row in rows
        if not (row["kind"] == "admit" and row["arrival_ordinal"] == 1)
    ]

    with pytest.raises(AssertionError, match="admit_ordinals"):
        analysis.judge_rows(rows, expected_requests=2)


def test_route_clustering_uses_route_timestamp_order() -> None:
    rows = (
        _chain(0, 0, 10)
        + _chain(1, 0, 11)
        + _chain(2, 1, 20)
        + _chain(3, 1, 21)
    )
    route_times = {0: 0, 1: 2000, 2: 1000, 3: 3000}
    for row in rows:
        ordinal = int(row["arrival_ordinal"])
        offset = {"route": 0, "receive": 100, "admit": 200}[str(row["kind"])]
        row["wall_ts_ns"] = route_times[ordinal] + offset

    result = analysis.judge_rows(rows, expected_requests=4)

    assert result.route_strict_alternation is True
    assert result.score_tie_routes == 4
    assert result.rank0_route_receive_median_ms == pytest.approx(0.0001)
    assert result.rank1_route_receive_median_ms == pytest.approx(0.0001)
    assert result.rank0_receive_admit_median_ms == pytest.approx(0.0001)
    assert result.rank1_receive_admit_median_ms == pytest.approx(0.0001)
    assert result.verdict == "no_rank_asymmetry_observed"


def test_drain_cadence_sorts_schedule_seq_and_allows_co_admit() -> None:
    rows = (
        _chain(0, 0, 12)
        + _chain(1, 1, 20)
        + _chain(2, 0, 10)
        + _chain(3, 1, 20)
    )

    result = analysis.judge_rows(rows, expected_requests=4)

    assert result.rank0_admit_gap_distribution == "2:1/1"
    assert result.rank1_admit_gap_distribution == "0:1/1"
    assert result.verdict == "H-drain"


def test_trace_loader_requires_complete_footers(tmp_path: Path) -> None:
    trace = tmp_path / "route-1.jsonl"
    rows = _chain(0, 0, 10)[:1]
    rows.append(
        {
            "kind": "trace_footer",
            "flush_complete": True,
            "event_count": 1,
            "component": "route",
        }
    )
    trace.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    assert analysis.load_trace_rows([trace])[0]["kind"] == "route"


def test_writer_uses_lf_line_endings(tmp_path: Path) -> None:
    result = analysis.judge_rows(
        _chain(0, 0, 10)
        + _chain(1, 1, 20)
        + _chain(2, 0, 11)
        + _chain(3, 1, 21),
        expected_requests=4,
    )
    output_csv = tmp_path / "result.csv"

    analysis.write_outputs(
        result,
        output_csv=output_csv,
        output_report=tmp_path / "result.md",
    )

    assert b"\r\n" not in output_csv.read_bytes()
