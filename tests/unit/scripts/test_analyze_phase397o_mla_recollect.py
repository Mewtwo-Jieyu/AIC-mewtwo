from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts import analyze_phase397o_mla_recollect as phase397o


def _load_raw() -> list[dict[str, str]]:
    with phase397o.RAW_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _find(rows: list[dict[str, str]], **match: str) -> dict[str, str]:
    for row in rows:
        if all(row[key] == value for key, value in match.items()):
            return row
    raise AssertionError(match)


def test_phase397o_raw_has_exact_48_shape_grid() -> None:
    indexed = phase397o.validate_raw_rows(_load_raw())
    assert len(indexed) == 48
    assert {key.local_num_heads for key in indexed} == {8, 16}
    assert {key.kv_cache_dtype for key in indexed} == {"float16", "fp8"}
    assert {key.batch_size for key in indexed} == {64, 128}
    assert {key.target_seq_len for key in indexed} == {8192, 9001, 16384}
    assert {key.randomize_blocks for key in indexed} == {"true", "false"}


def test_phase397o_triage_selects_b2_not_graph_or_randomize_fix() -> None:
    rows = phase397o.build_triage_rows(_load_raw())
    assert len([row for row in rows if row["row_type"] == "shape_compare"]) == 48
    key = _find(
        rows,
        row_type="shape_compare",
        local_num_heads="8",
        kv_cache_dtype="float16",
        batch_size="128",
        target_seq_len="9001",
        randomize_blocks="true",
    )
    assert float(key["graph_over_serve"]) > 1.7
    assert 0.98 < float(key["graph_over_eager6"]) < 1.0
    assert 0.99 < float(key["randomize_false_over_true"]) < 1.01

    timing = _find(rows, row_type="decision", candidate="cuda_graph_launch_overhead")
    assert timing["verdict"] == "timing_method_not_sufficient"
    layout = _find(rows, row_type="decision", candidate="randomized_block_layout")
    assert layout["verdict"] == "randomized_block_layout_not_primary"
    final = _find(rows, row_type="decision", candidate="phase397q_trigger")
    assert final["verdict"] == "B2_kernel_workload_or_serve_seed_difference"
    assert final["default_readiness"] == "No-Go"


def test_phase397o_fp8_backend_is_recorded_as_not_db_flashmla() -> None:
    rows = phase397o.build_triage_rows(_load_raw())
    fp8_shape = _find(
        rows,
        row_type="shape_compare",
        local_num_heads="8",
        kv_cache_dtype="fp8",
        batch_size="128",
        target_seq_len="9001",
        randomize_blocks="true",
    )
    assert fp8_shape["target_backend"] == "TritonMLAImpl"
    assert fp8_shape["kernel_source"] == "vllm_triton_mla"
    fp8_decision = _find(rows, row_type="decision", candidate="fp8_dtype_control")
    assert fp8_decision["verdict"] == "fp8_backend_not_comparable_to_db_flashmla"


def test_phase397o_writer_rejects_default_readiness_upgrade() -> None:
    raw = _load_raw()
    raw[0] = dict(raw[0])
    raw[0]["default_readiness"] = "Go"
    with pytest.raises(ValueError, match="default_readiness=No-Go"):
        phase397o.validate_raw_rows(raw)


def test_phase397o_main_writes_csv_and_md(tmp_path: Path) -> None:
    out_csv = tmp_path / "phase397o.csv"
    out_md = tmp_path / "phase397o.md"
    assert phase397o.main(["--out-csv", str(out_csv), "--out-md", str(out_md)]) == 0
    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 52
    assert _find(rows, row_type="decision", candidate="phase397q_trigger")["perf_database"] == "false"
    assert "B2_kernel_workload_or_serve_seed_difference" in out_md.read_text(encoding="utf-8")
