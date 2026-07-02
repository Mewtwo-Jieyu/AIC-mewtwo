from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts import analyze_phase397t_marlin_mechanism_gate as phase397t_gate


def _find(rows: list[dict[str, str]], **match: str) -> dict[str, str]:
    for row in rows:
        if all(row[key] == value for key, value in match.items()):
            return row
    raise AssertionError(match)


def test_phase397t_gate_records_all_candidate_routes_as_failed() -> None:
    rows = phase397t_gate.build_rows()
    route_rows = [row for row in rows if row["row_type"].startswith("route_gate_")]
    assert {row["route"] for row in route_rows} == {
        "power_law",
        "power_law_rank0_compact",
        "power_law_eplb",
        "balanced",
        "power_law_batched_rank0",
    }
    assert all(row["verdict"] == "route_gate_failed" for row in route_rows)
    assert all(row["gate_passed"] == "false" for row in route_rows)

    batched = _find(rows, route="power_law_batched_rank0")
    assert float(batched["latency_ms"]) == pytest.approx(0.749925, rel=1e-6)
    assert float(batched["latency_ms"]) > float(batched["gate_upper_ms"])
    assert batched["diag_block_size_m"] == "64"
    assert float(batched["diag_rank0_tokens_mean"]) == pytest.approx(93.7)


def test_phase397t_gate_decision_stops_before_table_or_validate_changes() -> None:
    rows = phase397t_gate.build_rows()
    decision = _find(rows, row_type="decision")
    assert decision["verdict"] == "mechanism_gate_failed_report_only"
    assert decision["write_moe_perf"] == "false"
    assert decision["validate_repoint"] == "false"
    assert decision["runtime_modified"] == "false"
    assert decision["default_readiness"] == "No-Go"
    assert decision["gpu_process_residue"] == "false"
    assert decision["process_residue"] == "false"


def test_phase397t_gate_writer_rejects_upgrade_flags() -> None:
    rows = phase397t_gate.build_rows()
    rows[-1] = dict(rows[-1])
    rows[-1]["write_moe_perf"] = "true"
    with pytest.raises(ValueError, match="write_moe_perf=false"):
        phase397t_gate.validate_rows(rows)

    rows = phase397t_gate.build_rows()
    rows[-1] = dict(rows[-1])
    rows[-1]["validate_repoint"] = "true"
    with pytest.raises(ValueError, match="validate_repoint=false"):
        phase397t_gate.validate_rows(rows)


def test_phase397t_gate_main_writes_csv_and_md(tmp_path: Path) -> None:
    out_csv = tmp_path / "phase397t_gate.csv"
    out_md = tmp_path / "phase397t_gate.md"
    assert phase397t_gate.main(["--out-csv", str(out_csv), "--out-md", str(out_md)]) == 0

    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert _find(rows, row_type="decision")["valid_for_default"] == "false"

    text = out_md.read_text(encoding="utf-8")
    assert "mechanism_gate_failed_report_only" in text
    assert "Default AIC remains No-Go" in text
    assert b"\r" not in out_csv.read_bytes()
