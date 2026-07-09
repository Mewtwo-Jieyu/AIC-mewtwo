from scripts.analyze_phase455_tp8_scope_collision import (
    CandidateRow,
    _no_scope_perfdb_rows,
    classify_delta,
    regime_scope_rows,
    summarize_candidate_surface,
)


def test_candidate_surface_reports_bucket_and_batch_ranges() -> None:
    rows = [
        CandidateRow("s", "tp8ep8", "decode", "forward_total", "forward_total", 1, 1, 9.0, 8),
        CandidateRow("s", "tp8ep8", "decode", "forward_total", "forward_total", 67, 67, 32.0, 4),
        CandidateRow("s", "tp8ep8", "mixed_prefill", "forward_total", "forward_total", 8000, 66, 604.0, 4),
    ]

    surface = summarize_candidate_surface(rows)

    assert surface["decode"]["rows"] == 2
    assert surface["decode"]["bucket_min"] == 1
    assert surface["decode"]["bucket_max"] == 67
    assert surface["mixed_prefill"]["decode_batch_max"] == 66


def test_classify_delta_marks_scope_pollution_regression() -> None:
    assert classify_delta(1.163, 1.703) == "regressed"
    assert classify_delta(1.439, 1.443) == "unchanged"
    assert classify_delta(1.439, 1.100) == "improved"


def test_regime_scope_uses_configurable_max_bt_not_scenario_label() -> None:
    rows = regime_scope_rows(
        {
            "K2.5-tp8ep8-8k2k": 8000,
            "K2.5-tp8ep8-32k3k": 32000,
            "K2.5-tp8ep8-8k2k-bt65536": 65536,
        }
    )

    assert {row["regime_scope"] for row in rows} == {
        "max_num_batched_tokens=8000",
        "max_num_batched_tokens=32000",
        "max_num_batched_tokens=65536",
    }
    assert all("K2.5" not in row["regime_scope"] for row in rows)


def test_no_scope_simulation_fans_tp8_row_to_all_max_bt_regimes() -> None:
    row = CandidateRow(
        "K2.5-tp8ep8-8k2k",
        "tp8ep8",
        "decode",
        "forward_total",
        "forward_total",
        32,
        32,
        20.0,
        8,
    )

    perf_rows = _no_scope_perfdb_rows(row)

    assert {item["max_num_batched_tokens"] for item in perf_rows} == {8000, 32000, 65536}
