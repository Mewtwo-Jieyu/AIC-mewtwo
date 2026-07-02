from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts import analyze_phase397s_int4_wo_route_gate as phase397s


def _find(rows: list[dict[str, str]], **match: str) -> dict[str, str]:
    for row in rows:
        if all(row[key] == value for key, value in match.items()):
            return row
    raise AssertionError(match)


def test_phase397s_records_missing_tp16_ep1_family_and_no_table_write() -> None:
    rows = phase397s.build_rows()

    coverage = _find(rows, row_type="coverage_check")
    assert coverage["verdict"] == "unique_missing_family_confirmed"
    assert coverage["existing_ep8_int4_wo_rows"] == "58"
    assert coverage["existing_tp16_ep1_int4_wo_rows"] == "0"
    assert coverage["missing_family"] == "moonshotai/Kimi-K2.5:int4_wo:moe_tp16:moe_ep1"

    decision = _find(rows, row_type="decision")
    assert decision["verdict"] == "route_gate_failed_report_only"
    assert decision["write_moe_perf"] == "false"
    assert decision["validate_repoint_persisted"] == "false"
    assert decision["default_readiness"] == "No-Go"


def test_phase397s_route_gate_latencies_all_fail_anchor_band() -> None:
    rows = phase397s.build_rows()
    expected = {
        "power_law": 0.470627,
        "balanced": 0.566186,
        "power_law_eplb": 0.426923,
    }

    for route, latency in expected.items():
        row = _find(rows, route=route)
        assert row["verdict"] == "route_gate_failed"
        assert row["gate_passed"] == "false"
        assert float(row["latency_ms"]) == pytest.approx(latency, rel=1e-6)
        assert float(row["latency_ms"]) > float(row["gate_upper_ms"])


def test_phase397s_writer_rejects_upgrades() -> None:
    rows = phase397s.build_rows()
    rows[-1] = dict(rows[-1])
    rows[-1]["write_moe_perf"] = "true"
    with pytest.raises(ValueError, match="write_moe_perf=false"):
        phase397s.write_csv(rows, Path("/tmp/phase397s_should_not_write.csv"))

    rows = phase397s.build_rows()
    rows[-1] = dict(rows[-1])
    rows[-1]["validate_repoint_persisted"] = "true"
    with pytest.raises(ValueError, match="validate_repoint_persisted=false"):
        phase397s.validate_rows(rows)

    rows = phase397s.build_rows()
    rows[2] = dict(rows[2])
    rows[2]["gate_passed"] = "true"
    with pytest.raises(ValueError, match="gate_passed=false"):
        phase397s.validate_rows(rows)


def test_phase397s_main_writes_csv_and_md(tmp_path: Path) -> None:
    out_csv = tmp_path / "phase397s.csv"
    out_md = tmp_path / "phase397s.md"
    assert phase397s.main(["--out-csv", str(out_csv), "--out-md", str(out_md)]) == 0

    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert _find(rows, row_type="decision")["valid_for_default"] == "false"

    text = out_md.read_text(encoding="utf-8")
    assert "route_gate_failed_report_only" in text
    assert "Default AIC remains No-Go" in text
    assert b"\r" not in out_csv.read_bytes()
