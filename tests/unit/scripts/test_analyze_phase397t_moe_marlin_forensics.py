from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts import analyze_phase397t_moe_marlin_forensics as phase397t


def _find(rows: list[dict[str, str]], **match: str) -> dict[str, str]:
    for row in rows:
        if all(row[key] == value for key, value in match.items()):
            return row
    raise AssertionError(match)


def test_phase397t_summarizes_serve_marlin_per_call() -> None:
    rows = phase397t.build_rows()
    summary = _find(rows, row_type="serve_marlin_summary")

    assert summary["verdict"] == "serve_marlin_anchor_confirmed"
    assert int(summary["ranks"]) == 8
    assert int(summary["calls_per_rank"]) == 3118
    assert float(summary["serve_mean_us_per_call"]) == pytest.approx(64.305125)
    assert float(summary["serve_ms_per_layer"]) == pytest.approx(0.128610, rel=1e-5)
    assert float(summary["phase397l_anchor_ms_per_layer"]) == pytest.approx(0.133668, rel=1e-5)


def test_phase397t_records_collector_and_vllm_path_mismatch_candidates() -> None:
    rows = phase397t.build_rows()

    collector = _find(rows, row_type="collector_static_path")
    assert collector["collector_hidden_m_source"] == "hidden_states[:tw.shape[0]]"
    assert collector["collector_effective_m"] == "128"
    assert collector["collector_uses_expert_map"] == "true"

    vllm = _find(rows, row_type="vllm_static_path")
    assert vllm["vllm_batched_experts_path"] == "true"
    assert vllm["vllm_align_ignore_invalid_experts"] == "true"
    assert vllm["vllm_block_size_candidates"] == "8/16/32/48/64"

    top = _find(rows, row_type="mechanism_rank_1")
    assert top["mechanism"] == "activation_format_or_effective_m_mismatch"
    assert top["decision"] == "test_on_h200"


def test_phase397t_report_only_guards_reject_perfdb_upgrade() -> None:
    rows = phase397t.build_rows()
    rows[-1] = dict(rows[-1])
    rows[-1]["write_moe_perf"] = "true"
    with pytest.raises(ValueError, match="write_moe_perf=false"):
        phase397t.validate_rows(rows)

    rows = phase397t.build_rows()
    rows[-1] = dict(rows[-1])
    rows[-1]["valid_for_default"] = "true"
    with pytest.raises(ValueError, match="valid_for_default=false"):
        phase397t.validate_rows(rows)


def test_phase397t_main_writes_csv_and_md(tmp_path: Path) -> None:
    out_csv = tmp_path / "phase397t.csv"
    out_md = tmp_path / "phase397t.md"
    assert phase397t.main(["--out-csv", str(out_csv), "--out-md", str(out_md)]) == 0

    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert _find(rows, row_type="decision")["default_readiness"] == "No-Go"

    text = out_md.read_text(encoding="utf-8")
    assert "activation_format_or_effective_m_mismatch" in text
    assert "Default AIC remains No-Go" in text
    assert b"\r" not in out_csv.read_bytes()
