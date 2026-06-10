from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "build_phase178_budget_mechanism_holdout_manifest.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_phase178_budget_mechanism_holdout_manifest",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


SCENARIOS = [
    ("tp8ep8-4k2k-bt4000", "tp8_dp1_ep8", "isl4000_osl2000_batch128", 8, 1, 8, 4000, 4000),
    ("tp8ep8-4k2k-bt65536", "tp8_dp1_ep8", "isl4000_osl2000_batch128", 8, 1, 8, 65536, 4000),
    ("tp4dp2ep8-4k2k-bt4000", "tp4_dp2_ep8", "isl4000_osl2000_batch128", 4, 2, 8, 4000, 4000),
    ("tp4dp2ep8-4k2k-bt65536", "tp4_dp2_ep8", "isl4000_osl2000_batch128", 4, 2, 8, 65536, 4000),
    ("tp8ep8-12k2k-bt12000", "tp8_dp1_ep8", "isl12000_osl2000_batch128", 8, 1, 8, 12000, 12000),
    ("tp8ep8-12k2k-bt65536", "tp8_dp1_ep8", "isl12000_osl2000_batch128", 8, 1, 8, 65536, 12000),
    ("tp4dp2ep8-12k2k-bt12000", "tp4_dp2_ep8", "isl12000_osl2000_batch128", 4, 2, 8, 12000, 12000),
    ("tp4dp2ep8-12k2k-bt65536", "tp4_dp2_ep8", "isl12000_osl2000_batch128", 4, 2, 8, 65536, 12000),
]


def _write_result(
    path: Path,
    *,
    scenario: str,
    topology_key: str,
    shape_key: str,
    tp: int,
    dp: int,
    ep: int,
    max_bt: int,
    isl: int,
    failed_requests: int = 0,
    diagnostic_only: bool = True,
    valid_for_default: bool = False,
    perf_database: bool = False,
) -> None:
    payload = {
        "source": "phase178_budget_mechanism_holdout",
        "scenario": scenario,
        "topology_key": topology_key,
        "shape_key": shape_key,
        "bench_result": {
            "ok_requests": 128,
            "failed_requests": failed_requests,
            "input_len": isl,
            "output_len": 2000,
            "num_prompts": 128,
            "max_concurrency": 128,
            "output_tok_s": 2048.0,
            "total_tok_s": 4096.0,
        },
        "shape": {
            "isl": isl,
            "osl": 2000,
            "batch_size": 128,
            "max_num_batched_tokens": max_bt,
            "max_num_seqs": 256,
        },
        "parallelism": {
            "tp": tp,
            "dp": dp,
            "ep": ep,
            "world_size": 8,
            "gpu_count": 8,
        },
        "scheduler": {
            "steady_state_time_ms": 1234.5 + max_bt,
        },
        "runtime": {
            "vllm_version": "0.19.0",
            "model_path": "/mnt/cfs/models/kimi-2.5-fp8",
            "dtype": "auto",
            "quantization": "fp8",
            "serve_command": "python3 -m vllm.entrypoints.cli.main serve ...",
            "benchmark_command": "python3 scripts/run_openai_fixed_shape_benchmark.py ...",
        },
        "hardware": {
            "gpu_model": "NVIDIA H200",
            "gpu_memory_gb": 141,
            "driver_version": "570.133.20",
            "cuda_version": "12.8",
        },
        "diagnostic_only": diagnostic_only,
        "valid_for_default": valid_for_default,
        "perf_database": perf_database,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_all_results(root: Path) -> list[Path]:
    paths = []
    for scenario, topology_key, shape_key, tp, dp, ep, max_bt, isl in SCENARIOS:
        path = root / scenario / "phase178_result.json"
        path.parent.mkdir(parents=True)
        _write_result(
            path,
            scenario=scenario,
            topology_key=topology_key,
            shape_key=shape_key,
            tp=tp,
            dp=dp,
            ep=ep,
            max_bt=max_bt,
            isl=isl,
        )
        paths.append(path)
    return paths


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_build_manifest_writes_eight_holdout_rows(tmp_path: Path) -> None:
    result_paths = _write_all_results(tmp_path)
    out_csv = tmp_path / "manifest.csv"

    rows = builder.build_manifest(result_paths)
    builder.write_manifest(out_csv, rows)

    written = _read_csv(out_csv)
    assert len(written) == 8
    by_name = {row["scenario"]: row for row in written}
    row = by_name["tp4dp2ep8-12k2k-bt65536"]
    assert row["topology_key"] == "tp4_dp2_ep8"
    assert row["shape_key"] == "isl12000_osl2000_batch128"
    assert row["tp"] == "4"
    assert row["dp"] == "2"
    assert row["ep"] == "8"
    assert row["max_num_batched_tokens"] == "65536"
    assert row["real_output_tok_s_gpu"] == "256.000000"
    assert row["steady_state_time_ms"] == "66770.500000"
    assert {row["diagnostic_only"] for row in written} == {"true"}
    assert {row["valid_for_default"] for row in written} == {"false"}
    assert {row["perf_database"] for row in written} == {"false"}


def test_manifest_rejects_missing_row(tmp_path: Path) -> None:
    result_paths = _write_all_results(tmp_path)[:-1]

    with pytest.raises(ValueError, match="exactly 8 rows"):
        builder.build_manifest(result_paths)


def test_manifest_rejects_duplicate_scenario(tmp_path: Path) -> None:
    result_paths = _write_all_results(tmp_path)
    duplicate = tmp_path / "duplicate-tp8ep8-4k2k-bt4000" / "phase178_result.json"
    duplicate.parent.mkdir(parents=True)
    _write_result(
        duplicate,
        scenario="tp8ep8-4k2k-bt4000",
        topology_key="tp8_dp1_ep8",
        shape_key="isl4000_osl2000_batch128",
        tp=8,
        dp=1,
        ep=8,
        max_bt=4000,
        isl=4000,
    )

    with pytest.raises(ValueError, match="duplicate scenario"):
        builder.build_manifest([*result_paths, duplicate])


def test_manifest_rejects_flag_error(tmp_path: Path) -> None:
    path = tmp_path / "tp8ep8-4k2k-bt4000" / "phase178_result.json"
    path.parent.mkdir(parents=True)
    _write_result(
        path,
        scenario="tp8ep8-4k2k-bt4000",
        topology_key="tp8_dp1_ep8",
        shape_key="isl4000_osl2000_batch128",
        tp=8,
        dp=1,
        ep=8,
        max_bt=4000,
        isl=4000,
        valid_for_default=True,
    )

    with pytest.raises(ValueError, match="valid_for_default"):
        builder.build_manifest([path])


def test_manifest_rejects_missing_control_pair(tmp_path: Path) -> None:
    result_paths = _write_all_results(tmp_path)
    missing_control = [
        path for path in result_paths if "tp8ep8-4k2k-bt4000" not in str(path)
    ]
    extra = tmp_path / "replacement" / "phase178_result.json"
    extra.parent.mkdir(parents=True)
    _write_result(
        extra,
        scenario="tp8ep8-4k2k-bt4000",
        topology_key="tp8_dp1_ep8",
        shape_key="isl4000_osl2000_batch128",
        tp=8,
        dp=1,
        ep=8,
        max_bt=65536,
        isl=4000,
    )

    with pytest.raises(ValueError, match="control/high-budget pair"):
        builder.build_manifest([*missing_control, extra])
