from types import SimpleNamespace

import pytest

from scripts import analyze_phase401_preempt_admission as phase401


def _fake_budget_row(name: str, sim_output: float, avg_decode: float, peak_decode: float):
    return SimpleNamespace(
        name=name,
        sim_output_tok_s_gpu=sim_output,
        avg_prefill_reqs_per_iter=0.0,
        avg_decode_reqs_per_iter=avg_decode,
        avg_tokens_per_iter=avg_decode,
        peak_prefill_reqs_per_iter=0.0,
        peak_decode_reqs_per_iter=peak_decode,
        peak_tokens_per_iter=peak_decode,
        steady_state_iterations=10,
        steady_state_time_ms=100.0,
    )


def test_phase401_builds_four_clean_rows() -> None:
    rows = phase401.build_phase401_rows(
        budget_rows=[
            _fake_budget_row("K2.5-tp8ep8-8k2k", 130.0, 60.0, 67.0),
            _fake_budget_row("K2.5-tp8ep8-32k3k", 55.0, 13.0, 14.0),
            _fake_budget_row("K2.5-tp4ep8dp2-8k2k", 140.0, 40.0, 50.0),
            _fake_budget_row("K2.5-tp4ep8dp2-32k3k", 45.0, 9.0, 10.0),
        ]
    )

    assert {row["scenario"] for row in rows} == set(phase401.CLEAN_SCENARIOS)
    assert all(row["runtime_modified"] == "true" for row in rows)
    assert all(row["valid_for_default"] == "false" for row in rows)
    assert all(row["default_readiness"] == "No-Go" for row in rows)


def test_phase401_locks_tp8_running_batch_targets() -> None:
    rows = {
        row["scenario"]: row
        for row in phase401.build_phase401_rows(
            budget_rows=[
                _fake_budget_row("K2.5-tp8ep8-8k2k", 130.0, 60.0, 67.0),
                _fake_budget_row("K2.5-tp8ep8-32k3k", 55.0, 13.0, 14.0),
                _fake_budget_row("K2.5-tp4ep8dp2-8k2k", 140.0, 40.0, 50.0),
                _fake_budget_row("K2.5-tp4ep8dp2-32k3k", 45.0, 9.0, 10.0),
            ]
        )
    }

    assert rows["K2.5-tp8ep8-8k2k"]["phase400_running_reqs_max"] == "67"
    assert rows["K2.5-tp8ep8-32k3k"]["phase400_running_reqs_max"] == "14"
    assert rows["K2.5-tp8ep8-32k3k"]["sim_peak_decode_vs_phase400_running_max_ratio"] == "1.000000"


def test_phase401_writer_rejects_default_upgrade(tmp_path) -> None:
    rows = phase401.build_phase401_rows(
        budget_rows=[
            _fake_budget_row("K2.5-tp8ep8-8k2k", 130.0, 60.0, 67.0),
            _fake_budget_row("K2.5-tp8ep8-32k3k", 55.0, 13.0, 14.0),
            _fake_budget_row("K2.5-tp4ep8dp2-8k2k", 140.0, 40.0, 50.0),
            _fake_budget_row("K2.5-tp4ep8dp2-32k3k", 45.0, 9.0, 10.0),
        ]
    )
    rows[0]["valid_for_default"] = "true"

    with pytest.raises(ValueError, match="valid_for_default"):
        phase401.write_phase401_csv(tmp_path / "bad.csv", rows)
