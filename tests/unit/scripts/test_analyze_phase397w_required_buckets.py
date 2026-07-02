from __future__ import annotations

from pathlib import Path

import pytest

from scripts import analyze_phase397w_required_buckets as phase397w


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_phase397w_enumerates_runtime_bucket_requests() -> None:
    rows = phase397w.analyze_phase397w_required_buckets(REPO_ROOT)

    by_key = {(row["scenario"], row["module_boundary"]): row for row in rows}

    ep8_8k = by_key[("K2.5-tp4ep8dp2-8k2k", "ep8_comm_dispatch_combine")]
    fused_8k = by_key[("K2.5-tp4ep8dp2-8k2k", "fusedmoe_runner_compute")]
    fused_bt = by_key[("K2.5-tp4ep8dp2-8k2k-bt65536", "fusedmoe_runner_compute")]

    assert "2000" in ep8_8k["missing_buckets"].split("/")
    assert "4000" in fused_8k["missing_buckets"].split("/")
    assert "32768" in fused_bt["missing_buckets"].split("/")
    assert fused_bt["missing_count"] != "0"


def test_phase397w_records_missing_32k3k_bt65536_real_measurement() -> None:
    rows = phase397w.analyze_phase397w_required_buckets(REPO_ROOT)

    planned_rows = [
        row
        for row in rows
        if row["scenario"] == "K2.5-tp4ep8dp2-32k3k-bt65536"
    ]

    assert planned_rows
    assert {row["real_measurement_status"] for row in planned_rows} == {
        "planned_missing_real_measurement"
    }
    assert {row["acceptance_gate_eligible"] for row in planned_rows} == {"false"}


def test_phase397w_exposes_bucket_128_contract_conflict() -> None:
    rows = phase397w.analyze_phase397w_required_buckets(REPO_ROOT)

    fused_rows = [row for row in rows if row["module_boundary"] == "fusedmoe_runner_compute"]
    assert any("128" in row["requested_buckets"].split("/") for row in fused_rows)
    assert any(row["bucket_128_required_by_runtime"] == "true" for row in fused_rows)


def test_phase397w_writer_rejects_upgrade_flags(tmp_path: Path) -> None:
    rows = phase397w.analyze_phase397w_required_buckets(REPO_ROOT)
    rows[0]["valid_for_default"] = "true"

    with pytest.raises(ValueError, match="valid_for_default"):
        phase397w.write_phase397w_csv(tmp_path / "bad.csv", rows)


def test_phase397w_main_writes_csv_and_md(tmp_path: Path) -> None:
    out_csv = tmp_path / "phase397w.csv"
    out_md = tmp_path / "phase397w.md"

    phase397w.main([
        "--repo-root",
        str(REPO_ROOT),
        "--out-csv",
        str(out_csv),
        "--out-md",
        str(out_md),
    ])

    csv_text = out_csv.read_text(encoding="utf-8")
    md_text = out_md.read_text(encoding="utf-8")

    assert "K2.5-tp4ep8dp2-8k2k" in csv_text
    assert "2000" in csv_text
    assert "bucket 128" in md_text
    assert "GPU collection is not started" in md_text
