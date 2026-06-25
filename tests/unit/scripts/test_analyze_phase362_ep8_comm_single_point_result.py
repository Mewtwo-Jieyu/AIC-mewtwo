from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "analyze_phase362_ep8_comm_single_point_result.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase362_ep8_comm_single_point_result",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE360 = (
    REPO_ROOT / "docs/iter_gap_investigation/phase360_ep8_alltoall_single_point_spec.csv"
)


def test_phase362_outputs_exact_single_result_row() -> None:
    rows = analyzer.analyze_phase362_ep8_comm_single_point_result(PHASE360)

    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "phase362_ep8_comm_single_point_result"
    assert row["phase361_artifact_dir"] == (
        "/mnt/shared-storage-user/ailab-sys/zhaojieyu/aic/"
        "phase361_ep8_comm_single_point_c68cfa1"
    )
    assert row["ok"] == "true"
    assert row["worker"] == "worker-892rz"
    assert row["vllm_version"] == "0.19.0"
    assert row["source_root"] == "/usr/local/lib/python3.12/dist-packages/vllm"
    assert row["measurement_boundary"] == (
        "vllm_ep_group_dispatch_router_logits_plus_combine"
    )
    assert row["backend"] == "allgather_reducescatter"
    assert row["manager"] == "AgRsAll2AllManager"
    assert row["latency_ms"] == "0.590688"
    assert row["cleanup"] == "true"
    assert row["gpu_process_residue"] == "false"
    assert row["rank_error"] == "0"


def test_phase362_preserves_no_go_diagnostic_flags() -> None:
    row = analyzer.analyze_phase362_ep8_comm_single_point_result(PHASE360)[0]

    assert row["default_readiness"] == "No-Go"
    assert row["diagnostic_only"] == "true"
    assert row["valid_for_default"] == "false"
    assert row["perf_database"] == "false"


def test_phase362_backend_capture_is_explicit_not_generic_alltoall() -> None:
    row = analyzer.analyze_phase362_ep8_comm_single_point_result(PHASE360)[0]

    assert row["route_label"] == "ep8_alltoall_single_point_result"
    assert row["backend"] == "allgather_reducescatter"
    assert row["backend"] != "alltoall"
    assert row["manager"] == "AgRsAll2AllManager"
    assert row["result_interpretation"] == (
        "runner_level_ep_comm_single_point_pass_not_perfdb_or_default_aic_evidence"
    )


def test_phase362_requires_phase360_runtime_backend_spec(tmp_path: Path) -> None:
    with PHASE360.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames is not None
    rows[2]["generic_backend_allowed"] = "true"

    bad = tmp_path / "phase360.csv"
    with bad.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="generic_backend_allowed"):
        analyzer.analyze_phase362_ep8_comm_single_point_result(bad)


def test_output_rejects_default_or_perfdb_upgrade(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase362_ep8_comm_single_point_result(PHASE360)
    rows[0]["perf_database"] = "true"

    with pytest.raises(ValueError, match="perf_database"):
        analyzer.write_phase362_csv(tmp_path / "bad.csv", rows)


def test_writers_emit_csv_and_md(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase362_ep8_comm_single_point_result(PHASE360)
    csv_path = tmp_path / "phase362.csv"
    md_path = tmp_path / "phase362.md"

    analyzer.write_phase362_csv(csv_path, rows)
    analyzer.write_phase362_md(md_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 1
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = md_path.read_text(encoding="utf-8")
    assert "EP8 comm single point pass" in doc
    assert "actual backend is `allgather_reducescatter`" in doc
    assert "`alltoall` is only the route label" in doc
    assert "`0.590688` ms is not a PerfDatabase row" in doc
    assert "Default AIC | No-Go" in doc
    assert "PerfDatabase | not written" in doc
