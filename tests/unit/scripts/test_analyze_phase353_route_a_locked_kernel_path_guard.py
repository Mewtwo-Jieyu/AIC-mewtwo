from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "analyze_phase353_route_a_locked_kernel_path_guard.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase353_route_a_locked_kernel_path_guard",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE351 = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase351_route_a_kernel_source_key_contract.csv"
)


def _copy_csv_with_mutation(src: Path, dst: Path, mutate) -> Path:
    with src.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames is not None
    mutate(rows)
    with dst.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return dst


def test_locked_kernel_path_guard_outputs_seven_rows() -> None:
    rows = analyzer.analyze_route_a_locked_kernel_path_guard(PHASE351)

    assert [row["guard_row"] for row in rows] == [
        "route_choice",
        "perfdb_schema_update",
        "kernel_source_value_set",
        "source_check_guard",
        "guard_failure_policy",
        "measurement_row_eligibility",
        "gpu_smoke_readiness",
    ]
    assert [row["decision"] for row in rows] == [
        "locked_kernel_path_guard_first",
        "deferred_until_measurement_value_set_exists",
        "blocked_pending_source_checked_kernel_name",
        "required_before_any_gpu_smoke",
        "fail_fast_no_gpu_run",
        "diagnostic_smoke_only_not_perfdb",
        "blocked_until_locked_guard_clears",
    ]
    assert {row["source"] for row in rows} == {
        "phase353_route_a_locked_kernel_path_guard"
    }
    assert {row["gpu_allowed"] for row in rows} == {"false"}
    assert {row["default_readiness"] for row in rows} == {"No-Go"}
    assert {row["diagnostic_only"] for row in rows} == {"true"}
    assert {row["valid_for_default"] for row in rows} == {"false"}
    assert {row["perf_database"] for row in rows} == {"false"}
    assert rows[0]["guard_contract"] == "choose_locked_kernel_path_guard_before_schema_update"
    assert rows[1]["guard_contract"] == "do_not_change_perfdb_schema_in_phase353"
    assert rows[-1]["next_required_action"] == (
        "run_h_source_check_only_after_guard_spec_is_committed"
    )


def test_writers_emit_csv_and_doc(tmp_path: Path) -> None:
    rows = analyzer.analyze_route_a_locked_kernel_path_guard(PHASE351)
    csv_path = tmp_path / "out.csv"
    doc_path = tmp_path / "out.md"

    analyzer.write_route_a_locked_kernel_path_guard_csv(csv_path, rows)
    analyzer.write_route_a_locked_kernel_path_guard_doc(doc_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 7
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = doc_path.read_text(encoding="utf-8")
    assert "Default AIC | No-Go" in doc
    assert "GPU allowed | false" in doc
    assert "locked kernel path guard first" in doc
    assert "PerfDatabase schema update is deferred" in doc
    assert "source-check guard" in doc
    assert "fail fast and do not run GPU" in doc
    assert "default-ready" not in doc


def test_phase351_input_must_require_kernel_source_identity(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, str]]) -> None:
        for row in rows:
            if row["contract_row"] == "kernel_source_identity":
                row["decision"] = "optional"

    phase351 = _copy_csv_with_mutation(PHASE351, tmp_path / "phase351.csv", mutate)

    with pytest.raises(ValueError, match="kernel_source_identity"):
        analyzer.analyze_route_a_locked_kernel_path_guard(phase351)


def test_phase351_input_must_keep_schema_update_candidate_not_done(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, str]]) -> None:
        for row in rows:
            if row["contract_row"] == "perfdb_schema_update_option":
                row["decision"] = "schema_update_complete"

    phase351 = _copy_csv_with_mutation(PHASE351, tmp_path / "phase351.csv", mutate)

    with pytest.raises(ValueError, match="perfdb_schema_update_option"):
        analyzer.analyze_route_a_locked_kernel_path_guard(phase351)


def test_phase351_input_must_keep_gpu_smoke_blocked(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, str]]) -> None:
        for row in rows:
            if row["contract_row"] == "gpu_smoke_readiness":
                row["decision"] = "cleared_for_gpu_smoke"

    phase351 = _copy_csv_with_mutation(PHASE351, tmp_path / "phase351.csv", mutate)

    with pytest.raises(ValueError, match="gpu_smoke_readiness"):
        analyzer.analyze_route_a_locked_kernel_path_guard(phase351)


def test_output_rejects_default_or_perfdb_upgrade(tmp_path: Path) -> None:
    rows = analyzer.analyze_route_a_locked_kernel_path_guard(PHASE351)
    rows[0]["perf_database"] = "true"

    with pytest.raises(ValueError, match="perf_database"):
        analyzer.write_route_a_locked_kernel_path_guard_csv(
            tmp_path / "bad.csv", rows
        )


def test_output_requires_exactly_seven_rows(tmp_path: Path) -> None:
    rows = analyzer.analyze_route_a_locked_kernel_path_guard(PHASE351)

    with pytest.raises(ValueError, match="exactly 7"):
        analyzer.write_route_a_locked_kernel_path_guard_csv(
            tmp_path / "bad.csv", rows[:6]
        )
