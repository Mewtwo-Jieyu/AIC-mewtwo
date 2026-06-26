from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "analyze_phase380_vllm_module_runtime_binding_sufficiency_gate.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase380_vllm_module_runtime_binding_sufficiency_gate",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE377_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase377_vllm_module_runtime_binding_spec.csv"
OPERATIONS_PY = REPO_ROOT / "src/aiconfigurator/sdk/operations.py"
BASE_BACKEND_PY = REPO_ROOT / "src/aiconfigurator/sdk/backends/base_backend.py"
PHASE378_TEST = REPO_ROOT / "tests/unit/sdk/backends/test_cb_simulator.py"
CHECKED_IN_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase380_vllm_module_runtime_binding_sufficiency_gate.csv"
)

EXPECTED_ROW_TYPES = [
    "phase377_contract_prerequisite",
    "phase378_runtime_binding_implemented",
    "phase379_topology_guard_implemented",
    "phase379_post_dispatch_bucket_guard_implemented",
    "exact_lookup_guard_complete",
    "unit_coverage_present",
    "validator_stability_preserved",
    "perfdb_data_unchanged",
    "default_aic_blocked",
    "next_phase",
]
EXPECTED_GUARD_FIELDS = (
    "model;hardware;vllm_version;topology;bucket_tokens;module_boundary;quant_runtime"
)


def test_phase380_outputs_exact_sufficiency_gate_rows() -> None:
    rows = analyzer.analyze_phase380_vllm_module_runtime_binding_sufficiency_gate(
        phase377_csv=PHASE377_CSV,
        operations_py=OPERATIONS_PY,
        base_backend_py=BASE_BACKEND_PY,
        phase378_test=PHASE378_TEST,
    )

    assert [row["row_type"] for row in rows] == EXPECTED_ROW_TYPES
    assert len(rows) == 10


def test_phase380_records_runtime_binding_implemented_but_not_default_ready() -> None:
    rows = analyzer.analyze_phase380_vllm_module_runtime_binding_sufficiency_gate(
        phase377_csv=PHASE377_CSV,
        operations_py=OPERATIONS_PY,
        base_backend_py=BASE_BACKEND_PY,
        phase378_test=PHASE378_TEST,
    )
    binding = next(row for row in rows if row["row_type"] == "phase378_runtime_binding_implemented")
    default_gate = next(row for row in rows if row["row_type"] == "default_aic_blocked")

    assert binding["runtime_binding"] == "implemented"
    assert binding["integration_level"] == "unit_plus_validator_evidence_only"
    assert default_gate["default_aic_allowed"] == "false"
    assert default_gate["default_readiness"] == "No-Go"


def test_phase380_locks_exact_lookup_guard_and_fail_fast_semantics() -> None:
    rows = analyzer.analyze_phase380_vllm_module_runtime_binding_sufficiency_gate(
        phase377_csv=PHASE377_CSV,
        operations_py=OPERATIONS_PY,
        base_backend_py=BASE_BACKEND_PY,
        phase378_test=PHASE378_TEST,
    )
    guard = next(row for row in rows if row["row_type"] == "exact_lookup_guard_complete")

    assert guard["exact_lookup_guard"] == EXPECTED_GUARD_FIELDS
    assert guard["model_required"] == "moonshotai/Kimi-K2.5"
    assert guard["perfdb_model_key"] == "kimi-k2.5"
    assert guard["hardware_required"] == "h200_sxm"
    assert guard["vllm_version_required"] == "0.19.0"
    assert guard["topology_required"] == "tp4dp2ep8"
    assert guard["allowed_bucket_tokens"] == "1/15/16/241/1808/2048/8192"
    assert guard["excluded_bucket_tokens"] == "128"
    assert guard["nearest_bucket_allowed"] == "false"
    assert guard["interpolation_allowed"] == "false"
    assert guard["extrapolation_allowed"] == "false"


def test_phase380_verifies_topology_and_post_dispatch_guards() -> None:
    rows = analyzer.analyze_phase380_vllm_module_runtime_binding_sufficiency_gate(
        phase377_csv=PHASE377_CSV,
        operations_py=OPERATIONS_PY,
        base_backend_py=BASE_BACKEND_PY,
        phase378_test=PHASE378_TEST,
    )
    topology = next(row for row in rows if row["row_type"] == "phase379_topology_guard_implemented")
    post_dispatch = next(
        row for row in rows if row["row_type"] == "phase379_post_dispatch_bucket_guard_implemented"
    )

    assert topology["topology_guard"] == "implemented"
    assert topology["topology_source"] == "vllm_module_topology"
    assert topology["topology_required"] == "tp4dp2ep8"
    assert post_dispatch["post_dispatch_bucket_guard"] == "implemented"
    assert post_dispatch["post_dispatch_combined_row_policy"] == "zero_after_bucket_check"


def test_phase380_preserves_no_gpu_no_perfdb_no_default_flags() -> None:
    rows = analyzer.analyze_phase380_vllm_module_runtime_binding_sufficiency_gate(
        phase377_csv=PHASE377_CSV,
        operations_py=OPERATIONS_PY,
        base_backend_py=BASE_BACKEND_PY,
        phase378_test=PHASE378_TEST,
    )

    for row in rows:
        assert row["gpu_allowed"] == "false"
        assert row["ssh_allowed"] == "false"
        assert row["perf_database"] == "false"
        assert row["new_data_rows_allowed"] == "false"
        assert row["schema_changed"] == "false"
        assert row["default_readiness"] == "No-Go"
        assert row["diagnostic_only"] == "true"
        assert row["valid_for_default"] == "false"


def test_phase380_rejects_missing_guard_text(tmp_path: Path) -> None:
    bad_operations = tmp_path / "operations.py"
    bad_operations.write_text(OPERATIONS_PY.read_text(encoding="utf-8").replace("tp4dp2ep8", ""), encoding="utf-8")

    with pytest.raises(ValueError, match="topology"):
        analyzer.analyze_phase380_vllm_module_runtime_binding_sufficiency_gate(
            phase377_csv=PHASE377_CSV,
            operations_py=bad_operations,
            base_backend_py=BASE_BACKEND_PY,
            phase378_test=PHASE378_TEST,
        )


def test_checked_in_csv_matches_analyzer_output() -> None:
    expected = analyzer.analyze_phase380_vllm_module_runtime_binding_sufficiency_gate(
        phase377_csv=PHASE377_CSV,
        operations_py=OPERATIONS_PY,
        base_backend_py=BASE_BACKEND_PY,
        phase378_test=PHASE378_TEST,
    )

    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        checked_in = list(csv.DictReader(handle))

    assert checked_in == expected


def test_writer_rejects_default_or_perfdb_upgrade(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase380_vllm_module_runtime_binding_sufficiency_gate(
        phase377_csv=PHASE377_CSV,
        operations_py=OPERATIONS_PY,
        base_backend_py=BASE_BACKEND_PY,
        phase378_test=PHASE378_TEST,
    )
    rows[0]["valid_for_default"] = "true"
    with pytest.raises(ValueError, match="valid_for_default"):
        analyzer.write_phase380_csv(tmp_path / "bad.csv", rows)

    rows = analyzer.analyze_phase380_vllm_module_runtime_binding_sufficiency_gate(
        phase377_csv=PHASE377_CSV,
        operations_py=OPERATIONS_PY,
        base_backend_py=BASE_BACKEND_PY,
        phase378_test=PHASE378_TEST,
    )
    rows[1]["perf_database"] = "true"
    with pytest.raises(ValueError, match="perf_database"):
        analyzer.write_phase380_csv(tmp_path / "bad.csv", rows)


def test_writers_emit_csv_and_md(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase380_vllm_module_runtime_binding_sufficiency_gate(
        phase377_csv=PHASE377_CSV,
        operations_py=OPERATIONS_PY,
        base_backend_py=BASE_BACKEND_PY,
        phase378_test=PHASE378_TEST,
    )
    csv_path = tmp_path / "phase380.csv"
    md_path = tmp_path / "phase380.md"

    analyzer.write_phase380_csv(csv_path, rows)
    analyzer.write_phase380_md(md_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 10
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = md_path.read_text(encoding="utf-8")
    assert "runtime binding implemented" in doc
    assert "Default AIC | No-Go" in doc
    assert "tp4dp2ep8" in doc
    assert "Phase381" in doc
