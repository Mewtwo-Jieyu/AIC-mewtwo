from pathlib import Path

import pytest

from scripts import analyze_phase399_kv_capacity_semantics as phase399


def test_phase399_little_law_identifies_dirty_bt65536_capacity() -> None:
    rows = phase399.build_phase399_rows()
    scenarios = {row["scenario"]: row for row in rows if row["row_type"] == "scenario"}

    row = scenarios["K2.5-tp4ep8dp2-8k2k-bt65536"]

    assert float(row["little_law_in_system_global"]) == pytest.approx(91.51, rel=1e-3)
    assert float(row["real_effective_decode_batch_global"]) == pytest.approx(85.6, rel=2e-2)
    assert float(row["full_sequence_capacity_global"]) < 6.0
    assert row["config_cleanliness"] == "reference_dirty"
    assert row["classification"] == "dirty_bt65536_capacity_conflict"
    assert row["default_readiness"] == "No-Go"


def test_phase399_source_facts_lock_vllm_kv_semantics() -> None:
    rows = phase399.build_phase399_rows()
    facts = {row["row_type"]: row for row in rows if row["row_type"].startswith("vllm_source_")}

    assert facts["vllm_source_override"]["source_line_range"] == "823-836"
    assert facts["vllm_source_override"]["classification"] == "override_replaces_num_blocks"
    assert facts["vllm_source_gpu_kv_size"]["source_line_range"] == "1303-1319"
    assert facts["vllm_source_gpu_kv_size"]["classification"] == "logged_token_capacity_after_override"


def test_phase399_writer_rejects_default_upgrade(tmp_path: Path) -> None:
    row = phase399._base_row("scenario")
    row["valid_for_default"] = True

    with pytest.raises(ValueError, match="valid_for_default"):
        phase399.write_phase399_csv(tmp_path / "bad.csv", [row])


def test_phase399_markdown_recommends_four_clean_scenario_gate() -> None:
    md = phase399.render_phase399_md(phase399.build_phase399_rows())

    assert "Default AIC remains No-Go" in md
    assert "gate shrink to four clean scenarios" in md
    assert "K2.5-tp4ep8dp2-8k2k-bt65536" in md
