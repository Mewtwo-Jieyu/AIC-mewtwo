from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "analyze_phase382_runtime_binding_probe_sufficiency_gate.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase382_runtime_binding_probe_sufficiency_gate",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE381_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase381_local_end_to_end_exact_bucket_probe.csv"
)
VLLM_MODULE_PERF = (
    REPO_ROOT / "src/aiconfigurator/systems/data/h200_sxm/vllm/0.19.0/vllm_module_perf.txt"
)
CHECKED_IN_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase382_runtime_binding_probe_sufficiency_gate.csv"
)

EXPECTED_ROW_TYPES = [
    "phase381_prerequisite",
    "module_table_exact_key_inventory",
    "runtime_bucket_formula_current",
    "ep8_runtime_bucket_reachability",
    "fusedmoe_runtime_bucket_reachability",
    "full_model_paired_lookup_requirement",
    "paired_lookup_candidate_gap",
    "op_level_runtime_binding_sufficient",
    "full_model_runtime_exact_lookup_blocked",
    "default_aic_blocked",
    "next_phase",
]


def _rows() -> list[dict[str, str]]:
    return analyzer.analyze_phase382_runtime_binding_probe_sufficiency_gate(
        phase381_csv=PHASE381_CSV,
        vllm_module_perf=VLLM_MODULE_PERF,
    )


def test_phase382_outputs_exact_sufficiency_gate_rows() -> None:
    rows = _rows()

    assert [row["row_type"] for row in rows] == EXPECTED_ROW_TYPES
    assert len(rows) == 11


def test_phase382_reads_phase381_as_op_level_only_prerequisite() -> None:
    prereq = next(row for row in _rows() if row["row_type"] == "phase381_prerequisite")

    assert prereq["phase381_prerequisite"] == "local_exact_lookup_passed"
    assert prereq["op_level_runtime_binding_sufficient"] == "true"
    assert prereq["full_model_runtime_exact_lookup_sufficient"] == "false"
    assert prereq["combined_full_model_status"] == "blocked_by_bucket_mapping"


def test_phase382_locks_module_table_inventory() -> None:
    inventory = next(row for row in _rows() if row["row_type"] == "module_table_exact_key_inventory")

    assert inventory["table_row_count"] == "14"
    assert inventory["module_count"] == "2"
    assert inventory["bucket_count"] == "7"
    assert inventory["module_boundaries"] == "fusedmoe_runner_compute;ep8_comm_dispatch_combine"
    assert inventory["allowed_bucket_tokens"] == "1/15/16/241/1808/2048/8192"
    assert inventory["module_table_exact_keys"] == "2_modules_x_7_buckets"


def test_phase382_records_current_runtime_bucket_formulas() -> None:
    formula = next(row for row in _rows() if row["row_type"] == "runtime_bucket_formula_current")

    assert formula["ep8_bucket_formula"] == "max(1, raw_tokens//4)"
    assert formula["fusedmoe_bucket_formula"] == "max(1, raw_tokens//4)*2"
    assert formula["verdict"] == "current_formula_captured"


def test_phase382_identifies_fusedmoe_partial_reachability() -> None:
    ep8 = next(row for row in _rows() if row["row_type"] == "ep8_runtime_bucket_reachability")
    fused = next(row for row in _rows() if row["row_type"] == "fusedmoe_runtime_bucket_reachability")

    assert ep8["ep8_reachable_buckets"] == "1/15/16/241/1808/2048/8192"
    assert ep8["verdict"] == "all_registered_buckets_reachable"
    assert fused["fusedmoe_reachable_buckets"] == "16/1808/2048/8192"
    assert fused["fusedmoe_unreachable_buckets"] == "1/15/241"
    assert fused["verdict"] == "partial_registered_bucket_reachability"


def test_phase382_blocks_full_model_exact_lookup_without_bucket_pairs() -> None:
    requirement = next(row for row in _rows() if row["row_type"] == "full_model_paired_lookup_requirement")
    gap = next(row for row in _rows() if row["row_type"] == "paired_lookup_candidate_gap")
    blocked = next(row for row in _rows() if row["row_type"] == "full_model_runtime_exact_lookup_blocked")

    assert requirement["paired_lookup_required"] == "bucket_b_and_2b_both_registered"
    assert gap["paired_lookup_candidate_count"] == "0"
    assert gap["paired_lookup_candidates"] == ""
    assert gap["verdict"] == "no_common_raw_token_pair"
    assert blocked["full_model_runtime_exact_lookup_sufficient"] == "false"
    assert blocked["next_allowed_phase"] == "phase383_runtime_bucket_semantics_decision_spec"


def test_phase382_op_level_is_sufficient_but_default_aic_stays_no_go() -> None:
    op = next(row for row in _rows() if row["row_type"] == "op_level_runtime_binding_sufficient")
    default = next(row for row in _rows() if row["row_type"] == "default_aic_blocked")
    next_phase = next(row for row in _rows() if row["row_type"] == "next_phase")

    assert op["op_level_runtime_binding_sufficient"] == "true"
    assert op["full_model_runtime_exact_lookup_sufficient"] == "false"
    assert default["default_aic_allowed"] == "false"
    assert default["default_readiness"] == "No-Go"
    assert next_phase["next_allowed_phase"] == "phase383_runtime_bucket_semantics_decision_spec"


def test_phase382_preserves_diagnostic_only_flags() -> None:
    for row in _rows():
        assert row["gpu_allowed"] == "false"
        assert row["ssh_allowed"] == "false"
        assert row["new_perfdb_data"] == "false"
        assert row["default_aic_allowed"] == "false"
        assert row["diagnostic_only"] == "true"
        assert row["valid_for_default"] == "false"
        assert row["perf_database"] == "false"
        assert row["default_readiness"] == "No-Go"
        assert row["interpolation_allowed"] == "false"
        assert row["extrapolation_allowed"] == "false"
        assert row["nearest_bucket_allowed"] == "false"


def test_checked_in_csv_matches_analyzer_output() -> None:
    expected = _rows()

    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        checked_in = list(csv.DictReader(handle))

    assert checked_in == expected


def test_writers_reject_default_or_perfdb_upgrade(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["valid_for_default"] = "true"
    with pytest.raises(ValueError, match="valid_for_default"):
        analyzer.write_phase382_csv(tmp_path / "bad.csv", rows)

    rows = _rows()
    rows[0]["perf_database"] = "true"
    with pytest.raises(ValueError, match="perf_database"):
        analyzer.write_phase382_csv(tmp_path / "bad.csv", rows)


def test_writers_emit_csv_and_md(tmp_path: Path) -> None:
    rows = _rows()
    csv_path = tmp_path / "phase382.csv"
    md_path = tmp_path / "phase382.md"

    analyzer.write_phase382_csv(csv_path, rows)
    analyzer.write_phase382_md(md_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert written == rows
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = md_path.read_text(encoding="utf-8")
    assert "op-level binding is sufficient" in doc
    assert "full-model exact lookup remains blocked" in doc
    assert "max(1, ...) 不改变当前 paired gap，只修正公式表达。" in doc
    assert "phase383_runtime_bucket_semantics_decision_spec" in doc
