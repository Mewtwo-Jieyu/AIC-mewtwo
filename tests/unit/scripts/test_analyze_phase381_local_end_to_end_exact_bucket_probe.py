from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "analyze_phase381_local_end_to_end_exact_bucket_probe.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase381_local_end_to_end_exact_bucket_probe",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


SYSTEMS_ROOT = REPO_ROOT / "src/aiconfigurator/systems"
PHASE380_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase380_vllm_module_runtime_binding_sufficiency_gate.csv"
)
CHECKED_IN_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase381_local_end_to_end_exact_bucket_probe.csv"
)

EXPECTED_ROW_TYPES = [
    "phase380_prerequisite",
    "fusedmoe_runner_compute_run_static_exact_bucket_probe",
    "ep8_comm_dispatch_combine_run_static_pre_post_probe",
    "bucket_128_fail_fast_probe",
    "non_kimi_scope_probe",
    "non_topology_scope_probe",
    "combined_full_model_exact_bucket_probe",
    "default_aic_blocked",
    "next_phase",
]


def _rows() -> list[dict[str, str]]:
    return analyzer.analyze_phase381_local_end_to_end_exact_bucket_probe(
        phase380_csv=PHASE380_CSV,
        systems_root=SYSTEMS_ROOT,
    )


def test_phase381_outputs_exact_probe_rows() -> None:
    rows = _rows()

    assert [row["row_type"] for row in rows] == EXPECTED_ROW_TYPES
    assert len(rows) == 9


def test_phase381_fusedmoe_probe_runs_through_basebackend_to_exact_vllm_module_key() -> None:
    rows = _rows()
    fused = next(
        row
        for row in rows
        if row["row_type"] == "fusedmoe_runner_compute_run_static_exact_bucket_probe"
    )

    assert fused["verdict"] == "pass"
    assert fused["run_static_used"] == "true"
    assert fused["probe_level"] == "basebackend_run_static"
    assert fused["measurement_boundary"] == "fusedmoe_forward_runner_level"
    assert fused["module_boundary"] == "fusedmoe_runner_compute"
    assert fused["raw_tokens"] == "32"
    assert fused["bucket_tokens"] == "16"
    assert fused["query_vllm_module_call_count"] == "1"
    assert fused["query_vllm_module_key"] == (
        "kimi-k2.5|h200_sxm|0.19.0|tp4dp2ep8|16|"
        "fusedmoe_runner_compute|CompressedTensorsWNA16MarlinMoEMethod"
    )
    assert fused["latency_ms"] == "0.239764"


def test_phase381_ep8_dispatch_probe_counts_pre_once_and_post_zero() -> None:
    rows = _rows()
    ep8 = next(
        row
        for row in rows
        if row["row_type"] == "ep8_comm_dispatch_combine_run_static_pre_post_probe"
    )

    assert ep8["verdict"] == "pass"
    assert ep8["run_static_used"] == "true"
    assert ep8["measurement_boundary"] == "vllm_ep_group_dispatch_router_logits_plus_combine"
    assert ep8["module_boundary"] == "ep8_comm_dispatch_combine"
    assert ep8["raw_tokens"] == "64"
    assert ep8["bucket_tokens"] == "16"
    assert ep8["query_vllm_module_call_count"] == "1"
    assert ep8["post_dispatch_policy"] == "zero_after_bucket_check"
    assert ep8["post_dispatch_latency_ms"] == "0.000000"
    assert ep8["query_vllm_module_key"] == (
        "kimi-k2.5|h200_sxm|0.19.0|tp4dp2ep8|16|"
        "ep8_comm_dispatch_combine|CompressedTensorsWNA16MarlinMoEMethod"
    )
    assert ep8["latency_ms"] == "0.083776"


def test_phase381_bucket_128_fails_fast_inside_kimi_scope() -> None:
    rows = _rows()
    bucket = next(row for row in rows if row["row_type"] == "bucket_128_fail_fast_probe")

    assert bucket["verdict"] == "pass"
    assert bucket["raw_tokens"] == "512"
    assert bucket["bucket_tokens"] == "128"
    assert bucket["fail_fast"] == "true"
    assert "bucket_tokens must be one of" in bucket["error"]
    assert bucket["query_vllm_module_call_count"] == "0"


def test_phase381_non_scope_does_not_use_vllm_module_table() -> None:
    rows = _rows()
    non_kimi = next(row for row in rows if row["row_type"] == "non_kimi_scope_probe")
    non_topology = next(row for row in rows if row["row_type"] == "non_topology_scope_probe")

    assert non_kimi["verdict"] == "pass"
    assert non_kimi["non_scope_reason"] == "model_name_not_kimi"
    assert non_kimi["raw_tokens"] == "32"
    assert non_kimi["bucket_tokens"] == "16"
    assert non_kimi["query_vllm_module_call_count"] == "0"
    assert non_kimi["fallback_query"] == "query_moe"
    assert non_topology["verdict"] == "pass"
    assert non_topology["non_scope_reason"] == "topology_not_tp4dp2ep8"
    assert non_topology["raw_tokens"] == "32"
    assert non_topology["bucket_tokens"] == "16"
    assert non_topology["query_vllm_module_call_count"] == "0"
    assert non_topology["fallback_query"] == "query_moe"


def test_phase381_combined_full_model_probe_is_blocked_not_faked() -> None:
    rows = _rows()
    combined = next(
        row for row in rows if row["row_type"] == "combined_full_model_exact_bucket_probe"
    )

    assert combined["verdict"] == "blocked_by_bucket_mapping"
    assert combined["combined_full_model_status"] == "blocked_by_bucket_mapping"
    assert combined["run_static_used"] == "false"
    assert combined["fallback_query"] == "not_bypassed"
    assert combined["error"] == (
        "same_raw_tokens_maps_to_ep8_bucket_raw_div_4_and_"
        "fusedmoe_bucket_raw_div_4_times_2"
    )


def test_phase381_preserves_no_gpu_no_perfdb_no_default_flags() -> None:
    for row in _rows():
        assert row["gpu_allowed"] == "false"
        assert row["ssh_allowed"] == "false"
        assert row["new_perfdb_data"] == "false"
        assert row["perf_database"] == "false"
        assert row["default_readiness"] == "No-Go"
        assert row["diagnostic_only"] == "true"
        assert row["valid_for_default"] == "false"
        assert row["default_aic_allowed"] == "false"
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
        analyzer.write_phase381_csv(tmp_path / "bad.csv", rows)

    rows = _rows()
    rows[0]["perf_database"] = "true"
    with pytest.raises(ValueError, match="perf_database"):
        analyzer.write_phase381_csv(tmp_path / "bad.csv", rows)


def test_writers_emit_csv_and_md(tmp_path: Path) -> None:
    rows = _rows()
    csv_path = tmp_path / "phase381.csv"
    md_path = tmp_path / "phase381.md"

    analyzer.write_phase381_csv(csv_path, rows)
    analyzer.write_phase381_md(md_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert written == rows
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = md_path.read_text(encoding="utf-8")
    assert "BaseBackend.run_static" in doc
    assert "blocked-by-bucket-mapping" in doc
    assert "Default AIC | No-Go" in doc
