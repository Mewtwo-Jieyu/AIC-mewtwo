from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts import analyze_phase397u_marlin_config_gate as phase397u_gate


def _find(rows: list[dict[str, str]], **match: str) -> dict[str, str]:
    for row in rows:
        if all(row[key] == value for key, value in match.items()):
            return row
    raise AssertionError(match)


def test_phase397u_records_both_last_shots_as_failed() -> None:
    rows = phase397u_gate.build_rows()
    shot_rows = [row for row in rows if row["row_type"].startswith("shot_gate_")]

    assert {row["shot"] for row in shot_rows} == {"local48_direct", "force_block64"}
    assert all(row["gate_passed"] == "false" for row in shot_rows)
    assert all(float(row["latency_ms"]) > float(row["gate_upper_ms"]) for row in shot_rows)

    local48 = _find(rows, shot="local48_direct")
    assert local48["diag_global_num_experts"] == "48"
    assert local48["diag_block_size_m"] == "16/32"
    assert float(local48["latency_ms"]) == pytest.approx(0.693781, rel=1e-6)

    block64 = _find(rows, shot="force_block64")
    assert block64["diag_global_num_experts"] == "384"
    assert block64["diag_block_size_m"] == "64"
    assert float(block64["latency_ms"]) == pytest.approx(0.742794, rel=1e-6)


def test_phase397u_decision_stops_before_table_validate_or_default() -> None:
    rows = phase397u_gate.build_rows()
    decision = _find(rows, row_type="decision")

    assert decision["verdict"] == "structure_path_exhausted_switch_to_option_a"
    assert decision["write_moe_perf"] == "false"
    assert decision["validate_repoint"] == "false"
    assert decision["retire_k"] == "false"
    assert decision["valid_for_default"] == "false"
    assert decision["default_readiness"] == "No-Go"
    assert decision["next_allowed_phase"] == "option_a_profiler_anchored_calibration_spec"


def test_phase397u_writer_rejects_upgrade_flags() -> None:
    rows = phase397u_gate.build_rows()
    rows[-1] = dict(rows[-1])
    rows[-1]["write_moe_perf"] = "true"
    with pytest.raises(ValueError, match="write_moe_perf=false"):
        phase397u_gate.validate_rows(rows)

    rows = phase397u_gate.build_rows()
    rows[-1] = dict(rows[-1])
    rows[-1]["validate_repoint"] = "true"
    with pytest.raises(ValueError, match="validate_repoint=false"):
        phase397u_gate.validate_rows(rows)


def test_phase397u_main_writes_csv_and_md(tmp_path: Path) -> None:
    out_csv = tmp_path / "phase397u.csv"
    out_md = tmp_path / "phase397u.md"

    assert phase397u_gate.main(["--out-csv", str(out_csv), "--out-md", str(out_md)]) == 0

    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert _find(rows, row_type="decision")["retire_k"] == "false"

    text = out_md.read_text(encoding="utf-8")
    assert "structure_path_exhausted_switch_to_option_a" in text
    assert "Default AIC remains No-Go" in text
    assert b"\r" not in out_csv.read_bytes()
