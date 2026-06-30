from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase384_paired_bucket_expansion_spec.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase384_paired_bucket_expansion_spec",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE383_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase383_runtime_bucket_semantics_decision_spec.csv"
)
VLLM_MODULE_PERF = (
    REPO_ROOT / "src/aiconfigurator/systems/data/h200_sxm/vllm/0.19.0/vllm_module_perf.txt"
)
CHECKED_IN_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase384_paired_bucket_expansion_spec.csv"
)

EXPECTED_ROW_TYPES = [
    "phase383_prerequisite",
    "existing_vllm_module_rows_preserved",
    "ep8_bucket_set_preserved",
    "fusedmoe_paired_bucket_expansion_planned",
    "future_total_row_count_planned",
    "bucket_128_blocked",
    "lookup_policy_exact_only",
    "runtime_change_blocked",
    "gpu_data_collection_deferred",
    "perfdb_write_blocked",
    "default_aic_blocked",
    "next_phase",
]


def _rows() -> list[dict[str, str]]:
    return analyzer.analyze_phase384_paired_bucket_expansion_spec(
        phase383_csv=PHASE383_CSV,
        vllm_module_perf=VLLM_MODULE_PERF,
    )


def test_phase384_outputs_paired_bucket_expansion_spec_rows() -> None:
    rows = _rows()

    assert [row["row_type"] for row in rows] == EXPECTED_ROW_TYPES
    assert len(rows) == 12


def test_phase384_reads_phase383_decision() -> None:
    prereq = next(row for row in _rows() if row["row_type"] == "phase383_prerequisite")

    assert prereq["phase383_prerequisite"] == "paired_bucket_data_expansion_first"
    assert prereq["runtime_bucket_semantics_change_allowed"] == "false"
    assert prereq["next_allowed_phase_from_phase383"] == "phase384_paired_bucket_expansion_spec"


def test_phase384_preserves_existing_rows_and_ep8_bucket_set() -> None:
    preserve = next(row for row in _rows() if row["row_type"] == "existing_vllm_module_rows_preserved")
    ep8 = next(row for row in _rows() if row["row_type"] == "ep8_bucket_set_preserved")

    assert preserve["existing_rows_preserved"] == "true"
    assert preserve["existing_module_row_count"] == "14"
    assert preserve["delete_existing_rows"] == "false"
    assert preserve["replace_existing_rows"] == "false"
    assert ep8["ep8_existing_buckets"] == "1/15/16/241/1808/2048/8192"
    assert ep8["ep8_bucket_change_allowed"] == "false"


def test_phase384_plans_only_fusedmoe_paired_bucket_expansion() -> None:
    expansion = next(
        row for row in _rows() if row["row_type"] == "fusedmoe_paired_bucket_expansion_planned"
    )
    total = next(row for row in _rows() if row["row_type"] == "future_total_row_count_planned")

    assert expansion["fusedmoe_existing_buckets"] == "1/15/16/241/1808/2048/8192"
    assert expansion["fusedmoe_paired_bucket_expansion"] == "2/30/32/482/3616/4096/16384"
    assert expansion["new_planned_row_count"] == "7"
    assert expansion["write_real_data_file"] == "false"
    assert total["future_total_row_count"] == "21"


def test_phase384_blocks_bucket_128_and_non_exact_lookup() -> None:
    bucket = next(row for row in _rows() if row["row_type"] == "bucket_128_blocked")
    lookup = next(row for row in _rows() if row["row_type"] == "lookup_policy_exact_only")

    assert bucket["bucket_128_allowed"] == "false"
    assert lookup["exact_lookup_only"] == "true"
    assert lookup["interpolation_allowed"] == "false"
    assert lookup["extrapolation_allowed"] == "false"
    assert lookup["nearest_bucket_allowed"] == "false"


def test_phase384_keeps_gpu_runtime_perfdb_and_default_blocked() -> None:
    for row in _rows():
        assert row["gpu_allowed"] == "false"
        assert row["ssh_allowed"] == "false"
        assert row["runtime_change"] == "false"
        assert row["new_perfdb_data"] == "false"
        assert row["write_real_data_file"] == "false"
        assert row["default_aic_allowed"] == "false"
        assert row["diagnostic_only"] == "true"
        assert row["valid_for_default"] == "false"
        assert row["perf_database"] == "false"
        assert row["default_readiness"] == "No-Go"


def test_checked_in_csv_matches_analyzer_output() -> None:
    expected = _rows()

    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        checked_in = list(csv.DictReader(handle))

    assert checked_in == expected


def test_writers_reject_real_data_runtime_or_default_upgrade(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["write_real_data_file"] = "true"
    with pytest.raises(ValueError, match="write_real_data_file"):
        analyzer.write_phase384_csv(tmp_path / "bad.csv", rows)

    rows = _rows()
    rows[0]["runtime_change"] = "true"
    with pytest.raises(ValueError, match="runtime_change"):
        analyzer.write_phase384_csv(tmp_path / "bad.csv", rows)

    rows = _rows()
    rows[0]["valid_for_default"] = "true"
    with pytest.raises(ValueError, match="valid_for_default"):
        analyzer.write_phase384_csv(tmp_path / "bad.csv", rows)


def test_writers_emit_csv_and_md(tmp_path: Path) -> None:
    rows = _rows()
    csv_path = tmp_path / "phase384.csv"
    md_path = tmp_path / "phase384.md"

    analyzer.write_phase384_csv(csv_path, rows)
    analyzer.write_phase384_md(md_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert written == rows
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = md_path.read_text(encoding="utf-8")
    assert "paired bucket expansion spec" in doc
    assert "2/30/32/482/3616/4096/16384" in doc
    assert "Do not write vllm_module_perf.txt in Phase384" in doc
    assert "phase385_fusedmoe_paired_bucket_gpu_data_spec" in doc
