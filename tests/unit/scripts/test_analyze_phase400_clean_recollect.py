from pathlib import Path

import pytest

from scripts import analyze_phase400_clean_recollect as phase400


def test_phase400_builds_four_clean_prefix_off_rows() -> None:
    rows = phase400.build_phase400_rows()
    scenarios = {row["scenario"]: row for row in rows}

    assert set(scenarios) == {
        "K2.5-tp8ep8-8k2k",
        "K2.5-tp8ep8-32k3k",
        "K2.5-tp4ep8dp2-8k2k",
        "K2.5-tp4ep8dp2-32k3k",
    }
    assert all(row["ok_requests"] == "128" for row in scenarios.values())
    assert all(row["failed_requests"] == "0" for row in scenarios.values())
    assert all(row["serve_enable_prefix_caching"] == "false" for row in scenarios.values())
    assert all(row["valid_for_default"] == "false" for row in scenarios.values())
    assert all(row["default_readiness"] == "No-Go" for row in scenarios.values())


def test_phase400_locks_running_batch_observations() -> None:
    rows = {row["scenario"]: row for row in phase400.build_phase400_rows()}

    assert rows["K2.5-tp8ep8-8k2k"]["serve_running_reqs_max"] == "67"
    assert rows["K2.5-tp8ep8-32k3k"]["serve_running_reqs_max"] == "14"
    assert rows["K2.5-tp4ep8dp2-8k2k"]["stats_status"] == "missing_dp_stats"
    assert rows["K2.5-tp4ep8dp2-32k3k"]["stats_status"] == "missing_dp_stats"


def test_phase400_records_cleanup_and_override_semantics() -> None:
    rows = {row["scenario"]: row for row in phase400.build_phase400_rows()}

    assert all(row["gpu_compute_apps_after_empty"] == "true" for row in rows.values())
    assert all(row["process_residual_after_empty"] == "true" for row in rows.values())
    assert all(row["override_num_gpu_blocks"] == "512" for row in rows.values())
    assert rows["K2.5-tp8ep8-8k2k"]["gpu_kv_cache_tokens"] == "546160"
    assert rows["K2.5-tp4ep8dp2-32k3k"]["gpu_kv_cache_tokens"] == "320800"


def test_phase400_writer_rejects_default_upgrade(tmp_path: Path) -> None:
    rows = phase400.build_phase400_rows()
    rows[0]["valid_for_default"] = "true"

    with pytest.raises(ValueError, match="valid_for_default"):
        phase400.write_phase400_csv(tmp_path / "bad.csv", rows)


def test_phase400_markdown_keeps_phase401_as_next_fix() -> None:
    md = phase400.render_phase400_md(phase400.build_phase400_rows())

    assert "Default AIC remains No-Go" in md
    assert "Phase401" in md
    assert "prefix hit is 0.0%" in md
