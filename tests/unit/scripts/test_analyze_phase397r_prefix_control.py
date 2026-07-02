from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts import analyze_phase397r_prefix_control as phase397r


def _find(rows: list[dict[str, str]], **match: str) -> dict[str, str]:
    for row in rows:
        if all(row[key] == value for key, value in match.items()):
            return row
    raise AssertionError(match)


def test_phase397r_prefix_off_control_is_kv_capacity_limited() -> None:
    rows = phase397r.build_rows()

    prefix = _find(rows, row_type="prefix_cache_check")
    assert prefix["verdict"] == "prefix_cache_disabled_confirmed"
    assert float(prefix["max_prefix_hit_pct"]) == pytest.approx(0.0)

    kv = _find(rows, row_type="kv_capacity_check")
    assert kv["verdict"] == "kv_capacity_limited_effective_batch"
    assert int(kv["profile_decode_bs"]) == 67
    assert float(kv["max_gpu_kv_cache_usage_pct"]) >= 99.0

    attention = _find(rows, row_type="attention_measurement")
    assert attention["verdict"] == "matches_b64_not_b128"
    assert float(attention["attention_ms_per_iter"]) == pytest.approx(16.141, rel=1e-4)
    assert 0.24 <= float(attention["attention_ms_per_layer"]) <= 0.28


def test_phase397r_decision_does_not_close_attention_or_write_table() -> None:
    rows = phase397r.build_rows()
    decision = _find(rows, row_type="decision")
    assert decision["verdict"] == "prefix_off_control_inconclusive_due_kv_capacity_batch_drop"
    assert decision["prefix_control_sufficient"] == "false"
    assert decision["attention_line_closed"] == "false"
    assert decision["write_generation_mla_perf"] == "false"
    assert decision["runtime_modified"] == "false"
    assert decision["gate_modified"] == "false"
    assert decision["default_readiness"] == "No-Go"


def test_phase397r_writer_rejects_table_write_upgrade() -> None:
    rows = phase397r.build_rows()
    rows[-1] = dict(rows[-1])
    rows[-1]["write_generation_mla_perf"] = "true"
    with pytest.raises(ValueError, match="write_generation_mla_perf=false"):
        phase397r.validate_rows(rows)


def test_phase397r_main_writes_csv_and_md(tmp_path: Path) -> None:
    out_csv = tmp_path / "phase397r.csv"
    out_md = tmp_path / "phase397r.md"
    assert phase397r.main(["--out-csv", str(out_csv), "--out-md", str(out_md)]) == 0

    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert _find(rows, row_type="decision")["valid_for_default"] == "false"

    text = out_md.read_text(encoding="utf-8")
    assert "prefix_off_control_inconclusive_due_kv_capacity_batch_drop" in text
    assert "Default AIC remains No-Go" in text
    assert b"\r" not in out_csv.read_bytes()
