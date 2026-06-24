from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase344_route_a_source_schema_audit.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase344_route_a_source_schema_audit",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE342 = REPO_ROOT / "docs/iter_gap_investigation/phase342_route_a_module_perf_table_spec.csv"
PERF_DATABASE = REPO_ROOT / "src/aiconfigurator/sdk/perf_database.py"
OPERATIONS = REPO_ROOT / "src/aiconfigurator/sdk/operations.py"
H200_VLLM_DIR = REPO_ROOT / "src/aiconfigurator/systems/data/h200_sxm/vllm"


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


def test_route_a_source_schema_audit_outputs_six_gates() -> None:
    rows = analyzer.analyze_route_a_source_schema_audit(
        phase342_csv=PHASE342,
        perf_database_py=PERF_DATABASE,
        h200_vllm_dir=H200_VLLM_DIR,
    )

    assert [row["gate"] for row in rows] == [
        "vllm_0190_moe_kernel_api",
        "perfdb_moe_kernel_source_schema",
        "vllm_ep8_comm_scope",
        "legacy_attention_gemm_reuse_assumption",
        "cb_sim_token_shape_fidelity_input",
        "full_trace_feedback_guard",
    ]
    assert [row["decision"] for row in rows] == [
        "blocked_pending_vllm_0190_source_check",
        "blocked_perfdb_schema_missing_kernel_source_key",
        "blocked_ep8_comm_schema_pending_source_check",
        "assumption_pending_validation",
        "retained_as_prerun_input_dependency",
        "rejected_data_leakage",
    ]
    assert {row["source"] for row in rows} == {"phase344_route_a_source_schema_audit"}
    assert {row["gpu_allowed"] for row in rows} == {"false"}
    assert {row["default_readiness"] for row in rows} == {"No-Go"}
    assert {row["diagnostic_only"] for row in rows} == {"true"}
    assert {row["valid_for_default"] for row in rows} == {"false"}
    assert {row["perf_database"] for row in rows} == {"false"}
    assert rows[0]["next_allowed_phase"] == "phase345_blocked_until_source_schema_clears"
    assert rows[5]["blocking_condition"] == "full_trace_must_not_generate_perfdb_rows"


def test_writers_emit_csv_and_doc(tmp_path: Path) -> None:
    rows = analyzer.analyze_route_a_source_schema_audit(
        phase342_csv=PHASE342,
        perf_database_py=PERF_DATABASE,
        h200_vllm_dir=H200_VLLM_DIR,
    )
    csv_path = tmp_path / "out.csv"
    doc_path = tmp_path / "out.md"

    analyzer.write_route_a_source_schema_audit_csv(csv_path, rows)
    analyzer.write_route_a_source_schema_audit_doc(doc_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 6
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = doc_path.read_text(encoding="utf-8")
    assert "Default AIC | No-Go" in doc
    assert "GPU allowed | false" in doc
    assert "vLLM 0.19.0 source/schema is not cleared" in doc
    assert "PerfDatabase vLLM MoE query does not use kernel_source as a key" in doc
    assert "H200 vLLM local perf table version | 0.12.0 only" in doc
    assert "full trace must stay validation-only" in doc
    assert "default-ready" not in doc


def test_phase342_must_keep_route_a_primary_candidate(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, str]]) -> None:
        rows[0]["decision"] = "rejected"

    phase342 = _copy_csv_with_mutation(PHASE342, tmp_path / "phase342.csv", mutate)

    with pytest.raises(ValueError, match="route_a_module_perf_table_candidate"):
        analyzer.analyze_route_a_source_schema_audit(
            phase342_csv=phase342,
            perf_database_py=PERF_DATABASE,
            h200_vllm_dir=H200_VLLM_DIR,
        )


def test_perfdb_vllm_moe_kernel_source_query_key_change_fails(tmp_path: Path) -> None:
    text = PERF_DATABASE.read_text(encoding="utf-8")
    marker = "elif self.backend == common.BackendName.vllm.value:"
    branch_start = text.index(marker)
    mutated_tail = text[branch_start:].replace(
        "moe_dict = self._moe_data[quant_mode][used_workload_distribution]",
        "moe_dict = self._moe_data[kernel_source][quant_mode][used_workload_distribution]",
        1,
    )
    mutated = text[:branch_start] + mutated_tail
    perfdb = tmp_path / "perf_database.py"
    perfdb.write_text(mutated, encoding="utf-8")

    with pytest.raises(ValueError, match="kernel_source"):
        analyzer.analyze_route_a_source_schema_audit(
            phase342_csv=PHASE342,
            perf_database_py=perfdb,
            h200_vllm_dir=H200_VLLM_DIR,
        )


def test_operations_vllm_moe_dispatch_comm_schema_change_fails(tmp_path: Path) -> None:
    text = OPERATIONS.read_text(encoding="utf-8")
    start_marker = "elif database.backend == common.BackendName.vllm.value:"
    end_marker = "elif database.backend == common.BackendName.sglang.value:"
    branch_start = text.index(start_marker)
    branch_end = text.index(end_marker, branch_start)
    branch = text[branch_start:branch_end]
    mutated_branch = branch.replace(
        "database.query_nccl(",
        "database.query_wideep_alltoall(",
        1,
    ).replace(
        '"all_gather" if self._pre_dispatch else "reduce_scatter"',
        '"alltoall"',
        1,
    )
    operations = tmp_path / "operations.py"
    operations.write_text(
        text[:branch_start] + mutated_branch + text[branch_end:],
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="MoEDispatch"):
        analyzer.analyze_route_a_source_schema_audit(
            phase342_csv=PHASE342,
            perf_database_py=PERF_DATABASE,
            operations_py=operations,
            h200_vllm_dir=H200_VLLM_DIR,
        )


def test_h200_vllm_local_version_change_requires_new_audit(tmp_path: Path) -> None:
    (tmp_path / "0.12.0").mkdir()
    (tmp_path / "0.19.0").mkdir()

    with pytest.raises(ValueError, match="0.12.0 only"):
        analyzer.analyze_route_a_source_schema_audit(
            phase342_csv=PHASE342,
            perf_database_py=PERF_DATABASE,
            h200_vllm_dir=tmp_path,
        )


def test_writer_requires_exactly_six_rows(tmp_path: Path) -> None:
    rows = analyzer.analyze_route_a_source_schema_audit(
        phase342_csv=PHASE342,
        perf_database_py=PERF_DATABASE,
        h200_vllm_dir=H200_VLLM_DIR,
    )

    with pytest.raises(ValueError, match="exactly 6"):
        analyzer.write_route_a_source_schema_audit_csv(tmp_path / "bad.csv", rows[:5])
