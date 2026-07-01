from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "scripts"
    / "analyze_phase394_merged_granularity_runtime_change.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase394_merged_granularity_runtime_change", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)

CHECKED_IN_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase394_merged_granularity_runtime_change.csv"
)
CHECKED_IN_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase394_merged_granularity_runtime_change.md"
)

EXPECTED_ROW_TYPES = [
    "runtime_change",
    "unit_tests",
    "decision_b_global",
    "gate_throughput",
    "gate_multi_config",
    "gate_ttft",
    "regression_root_cause",
    "followup_tp16_overcount",
    "followup_unify_backends",
    "followup_measure_0190_attention_gemm",
    "followup_bare_error_harness",
    "followup_remove_fudge",
    "verdict",
    "next_phase",
]


def _rows() -> list[dict[str, str]]:
    return analyzer.analyze_phase394_merged_granularity_runtime_change()


def test_row_order() -> None:
    assert [r["row_type"] for r in _rows()] == EXPECTED_ROW_TYPES


def test_all_rows_use_fieldnames() -> None:
    for row in _rows():
        assert set(row) == set(analyzer.FIELDNAMES)


def test_decision_is_b_global() -> None:
    row = next(r for r in _rows() if r["row_type"] == "decision_b_global")
    assert row["decision"] == "adopt_merged_globally_no_scope_gating"


def test_throughput_gate_records_fail_honestly() -> None:
    row = next(r for r in _rows() if r["row_type"] == "gate_throughput")
    assert row["baseline_error"] == "1.50x"
    assert row["p394_error"] == "1.52x"
    assert row["gate_status"] == "FAIL_threshold_1.50x"


def test_multi_config_gate_records_improvement() -> None:
    row = next(r for r in _rows() if r["row_type"] == "gate_multi_config")
    assert row["baseline_error"] == "1.47x"
    assert row["p394_error"] == "1.43x"
    assert row["gate_status"] == "PASS_IMPROVED"


def test_ttft_gate_unchanged() -> None:
    row = next(r for r in _rows() if r["row_type"] == "gate_ttft")
    assert row["baseline_error"] == row["p394_error"] == "1.79x"
    assert row["gate_status"] == "PASS_SAME"


def test_unit_tests_pass() -> None:
    row = next(r for r in _rows() if r["row_type"] == "unit_tests")
    assert row["p394_error"] == analyzer.UNIT_TESTS
    assert row["gate_status"] == "PASS"


def test_followups_present() -> None:
    types = {r["row_type"] for r in _rows()}
    for expected in (
        "followup_tp16_overcount",
        "followup_unify_backends",
        "followup_measure_0190_attention_gemm",
        "followup_bare_error_harness",
        "followup_remove_fudge",
    ):
        assert expected in types


def test_no_go_discipline_flags() -> None:
    forbidden_false = [
        "nearest_lookup_allowed",
        "interpolation_allowed",
        "extrapolation_allowed",
        "fudge_factor_tuning_used",
        "scope_gating_used",
        "write_real_data_file",
        "gpu_allowed",
        "ssh_allowed",
        "default_aic_allowed",
        "valid_for_default",
        "perf_database",
        "diagnostic_only",
    ]
    for row in _rows():
        for field in forbidden_false:
            assert row[field] == "false", (row["row_type"], field)
        assert row["exact_lookup_only"] == "true"
        # this is the runtime-change phase
        assert row["runtime_modified"] == "true"
        assert row["default_readiness"] == "No-Go"


def test_verdict_next_phase() -> None:
    verdict = next(r for r in _rows() if r["row_type"] == "verdict")
    assert verdict["next_allowed_phase"] == analyzer.NEXT_PHASE
    assert "phase395" in analyzer.NEXT_PHASE


def test_checked_in_csv_matches_analyzer(tmp_path: Path) -> None:
    if not CHECKED_IN_CSV.exists():
        pytest.skip("phase394 csv not generated yet")
    out = tmp_path / "phase394.csv"
    analyzer.write_phase394_csv(out, _rows())
    assert CHECKED_IN_CSV.read_text(encoding="utf-8") == out.read_text(
        encoding="utf-8"
    )


def test_checked_in_csv_parseable() -> None:
    if not CHECKED_IN_CSV.exists():
        pytest.skip("phase394 csv not generated yet")
    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == analyzer.FIELDNAMES
        rows = list(reader)
    assert [r["row_type"] for r in rows] == EXPECTED_ROW_TYPES


def test_checked_in_md_has_verdict() -> None:
    if not CHECKED_IN_MD.exists():
        pytest.skip("phase394 md not generated yet")
    text = CHECKED_IN_MD.read_text(encoding="utf-8")
    assert "Phase394" in text
    assert "Decision B" in text
    assert "merged" in text
