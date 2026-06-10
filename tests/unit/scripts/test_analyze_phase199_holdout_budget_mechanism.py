from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "analyze_phase199_holdout_budget_mechanism.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase199_holdout_budget_mechanism",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


FIELDNAMES = [
    "source",
    "scenario",
    "topology_key",
    "shape_key",
    "role",
    "tp",
    "dp",
    "ep",
    "world_size",
    "gpu_count",
    "gpu_model",
    "gpu_memory_gb",
    "driver_version",
    "cuda_version",
    "model_path",
    "vllm_version",
    "dtype",
    "quantization",
    "isl",
    "osl",
    "batch_size",
    "max_num_batched_tokens",
    "max_num_seqs",
    "num_prompts",
    "max_concurrency",
    "request_success_count",
    "request_fail_count",
    "real_output_tok_s",
    "real_total_tok_s",
    "real_output_tok_s_gpu",
    "real_total_tok_s_gpu",
    "steady_state_time_ms",
    "serve_command",
    "benchmark_command",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]


def _manifest_rows() -> list[dict[str, str]]:
    base = {
        "source": "phase178_budget_mechanism_holdout",
        "world_size": "8",
        "gpu_count": "8",
        "gpu_model": "NVIDIA H200",
        "gpu_memory_gb": "141",
        "driver_version": "570.133.20",
        "cuda_version": "12.8",
        "model_path": "/models/kimi",
        "vllm_version": "0.19.0",
        "dtype": "auto",
        "quantization": "fp8",
        "osl": "2000",
        "batch_size": "128",
        "max_num_seqs": "256",
        "num_prompts": "128",
        "max_concurrency": "128",
        "request_success_count": "128",
        "request_fail_count": "0",
        "real_output_tok_s_gpu": "1.000000",
        "real_total_tok_s_gpu": "1.000000",
        "serve_command": "serve",
        "benchmark_command": "bench",
        "diagnostic_only": "true",
        "valid_for_default": "false",
        "perf_database": "false",
    }
    rows = [
        {
            **base,
            "scenario": "tp8ep8-4k2k-bt4000",
            "topology_key": "tp8_dp1_ep8",
            "shape_key": "isl4000_osl2000_batch128",
            "role": "control",
            "tp": "8",
            "dp": "1",
            "ep": "8",
            "isl": "4000",
            "max_num_batched_tokens": "4000",
            "real_output_tok_s": "100.000000",
            "real_total_tok_s": "300.000000",
            "steady_state_time_ms": "10.000000",
        },
        {
            **base,
            "scenario": "tp8ep8-4k2k-bt65536",
            "topology_key": "tp8_dp1_ep8",
            "shape_key": "isl4000_osl2000_batch128",
            "role": "holdout",
            "tp": "8",
            "dp": "1",
            "ep": "8",
            "isl": "4000",
            "max_num_batched_tokens": "65536",
            "real_output_tok_s": "105.000000",
            "real_total_tok_s": "315.000000",
            "steady_state_time_ms": "40.000000",
        },
        {
            **base,
            "scenario": "tp4dp2ep8-4k2k-bt4000",
            "topology_key": "tp4_dp2_ep8",
            "shape_key": "isl4000_osl2000_batch128",
            "role": "control",
            "tp": "4",
            "dp": "2",
            "ep": "8",
            "isl": "4000",
            "max_num_batched_tokens": "4000",
            "real_output_tok_s": "200.000000",
            "real_total_tok_s": "600.000000",
            "steady_state_time_ms": "20.000000",
        },
        {
            **base,
            "scenario": "tp4dp2ep8-4k2k-bt65536",
            "topology_key": "tp4_dp2_ep8",
            "shape_key": "isl4000_osl2000_batch128",
            "role": "holdout",
            "tp": "4",
            "dp": "2",
            "ep": "8",
            "isl": "4000",
            "max_num_batched_tokens": "65536",
            "real_output_tok_s": "240.000000",
            "real_total_tok_s": "720.000000",
            "steady_state_time_ms": "100.000000",
        },
        {
            **base,
            "scenario": "tp8ep8-12k2k-bt12000",
            "topology_key": "tp8_dp1_ep8",
            "shape_key": "isl12000_osl2000_batch128",
            "role": "control",
            "tp": "8",
            "dp": "1",
            "ep": "8",
            "isl": "12000",
            "max_num_batched_tokens": "12000",
            "real_output_tok_s": "80.000000",
            "real_total_tok_s": "560.000000",
            "steady_state_time_ms": "25.000000",
        },
        {
            **base,
            "scenario": "tp8ep8-12k2k-bt65536",
            "topology_key": "tp8_dp1_ep8",
            "shape_key": "isl12000_osl2000_batch128",
            "role": "holdout",
            "tp": "8",
            "dp": "1",
            "ep": "8",
            "isl": "12000",
            "max_num_batched_tokens": "65536",
            "real_output_tok_s": "84.000000",
            "real_total_tok_s": "588.000000",
            "steady_state_time_ms": "50.000000",
        },
        {
            **base,
            "scenario": "tp4dp2ep8-12k2k-bt12000",
            "topology_key": "tp4_dp2_ep8",
            "shape_key": "isl12000_osl2000_batch128",
            "role": "control",
            "tp": "4",
            "dp": "2",
            "ep": "8",
            "isl": "12000",
            "max_num_batched_tokens": "12000",
            "real_output_tok_s": "300.000000",
            "real_total_tok_s": "2100.000000",
            "steady_state_time_ms": "30.000000",
        },
        {
            **base,
            "scenario": "tp4dp2ep8-12k2k-bt65536",
            "topology_key": "tp4_dp2_ep8",
            "shape_key": "isl12000_osl2000_batch128",
            "role": "holdout",
            "tp": "4",
            "dp": "2",
            "ep": "8",
            "isl": "12000",
            "max_num_batched_tokens": "65536",
            "real_output_tok_s": "297.000000",
            "real_total_tok_s": "2079.000000",
            "steady_state_time_ms": "120.000000",
        },
    ]
    return rows


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_analyze_holdout_pairs_computes_four_pair_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    out_csv = tmp_path / "analysis.csv"
    out_doc = tmp_path / "interpretation.md"
    _write_manifest(manifest, _manifest_rows())

    rows = analyzer.analyze_holdout_budget_mechanism(manifest)
    analyzer.write_analysis_csv(out_csv, rows)
    analyzer.write_interpretation_doc(out_doc, rows)

    written = _read_csv(out_csv)
    assert len(written) == 4
    by_pair = {(row["topology_key"], row["shape_key"]): row for row in written}
    assert by_pair[("tp8_dp1_ep8", "isl4000_osl2000_batch128")]["clean_high_over_control"] == "1.050000"
    assert by_pair[("tp8_dp1_ep8", "isl4000_osl2000_batch128")]["clean_delta_from_1"] == "0.050000"
    assert by_pair[("tp8_dp1_ep8", "isl4000_osl2000_batch128")]["steady_state_time_ratio"] == "4.000000"
    assert by_pair[("tp4_dp2_ep8", "isl12000_osl2000_batch128")]["clean_high_over_control"] == "0.990000"
    assert by_pair[("tp4_dp2_ep8", "isl12000_osl2000_batch128")]["clean_delta_from_1"] == "-0.010000"
    assert by_pair[("tp4_dp2_ep8", "isl12000_osl2000_batch128")]["steady_state_time_ratio"] == "4.000000"
    assert {row["diagnostic_only"] for row in written} == {"true"}
    assert {row["valid_for_default"] for row in written} == {"false"}
    assert {row["perf_database"] for row in written} == {"false"}
    text = out_doc.read_text(encoding="utf-8")
    assert "diagnostic mechanism candidate only" in text
    assert "Default AIC | No-Go" in text
    assert "PerfDatabase | No-Go" in text
    assert "not a global constant" in text


def test_missing_pair_row_fails_fast(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, _manifest_rows()[:-1])

    with pytest.raises(ValueError, match="exactly 8 rows"):
        analyzer.analyze_holdout_budget_mechanism(manifest)


def test_missing_holdout_role_fails_fast(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    rows = _manifest_rows()
    rows[-1]["role"] = "control"
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match="control/holdout pair"):
        analyzer.analyze_holdout_budget_mechanism(manifest)


def test_flag_mismatch_fails_fast(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    rows = _manifest_rows()
    rows[0]["valid_for_default"] = "true"
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match="valid_for_default"):
        analyzer.analyze_holdout_budget_mechanism(manifest)


def test_failed_request_fails_fast(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    rows = _manifest_rows()
    rows[0]["request_fail_count"] = "1"
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match="request_fail_count"):
        analyzer.analyze_holdout_budget_mechanism(manifest)
