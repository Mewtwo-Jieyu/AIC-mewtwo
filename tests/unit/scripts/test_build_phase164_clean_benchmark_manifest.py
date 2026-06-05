from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "build_phase164_clean_benchmark_manifest.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_phase164_clean_benchmark_manifest",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


SCENARIOS = [
    ("tp8ep8-bt8000", 8, 1, 8, 8000),
    ("tp8ep8-bt65536", 8, 1, 8, 65536),
    ("tp4dp2ep8-bt8000", 4, 2, 8, 8000),
    ("tp4dp2ep8-bt65536", 4, 2, 8, 65536),
]


def _write_result(
    path: Path,
    *,
    scenario: str,
    tp: int,
    dp: int,
    ep: int,
    max_bt: int,
    failed_requests: int = 0,
    omit: str | None = None,
) -> None:
    payload = {
        "scenario": scenario,
        "bench_result": {
            "ok_requests": 128,
            "failed_requests": failed_requests,
            "input_len": 8000,
            "output_len": 2000,
            "num_prompts": 128,
            "max_concurrency": 128,
            "output_tok_s": 1024.0,
            "total_tok_s": 5120.0,
        },
        "shape": {
            "isl": 8000,
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
            "driver_version": "550.54.15",
            "cuda_version": "12.4",
        },
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
    }
    if omit is not None:
        section, key = omit.split(".", 1)
        del payload[section][key]
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_all_results(root: Path) -> list[Path]:
    paths = []
    for scenario, tp, dp, ep, max_bt in SCENARIOS:
        path = root / scenario / "phase164_result.json"
        path.parent.mkdir(parents=True)
        _write_result(
            path,
            scenario=scenario,
            tp=tp,
            dp=dp,
            ep=ep,
            max_bt=max_bt,
        )
        paths.append(path)
    return paths


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_build_manifest_writes_four_clean_rows(tmp_path: Path) -> None:
    result_paths = _write_all_results(tmp_path)
    out_csv = tmp_path / "manifest.csv"

    rows = builder.build_manifest(result_paths)
    builder.write_manifest(out_csv, rows)

    written = _read_csv(out_csv)
    assert len(written) == 4
    by_name = {row["scenario"]: row for row in written}
    assert by_name["tp8ep8-bt65536"]["tp"] == "8"
    assert by_name["tp8ep8-bt65536"]["dp"] == "1"
    assert by_name["tp8ep8-bt65536"]["ep"] == "8"
    assert by_name["tp8ep8-bt65536"]["max_num_batched_tokens"] == "65536"
    assert by_name["tp8ep8-bt65536"]["real_output_tok_s_gpu"] == "128.000000"
    assert by_name["tp8ep8-bt65536"]["real_total_tok_s_gpu"] == "640.000000"
    assert {row["diagnostic_only"] for row in written} == {"true"}
    assert {row["valid_for_default"] for row in written} == {"false"}
    assert {row["perf_database"] for row in written} == {"false"}


def test_manifest_rejects_duplicate_scenario(tmp_path: Path) -> None:
    result_paths = _write_all_results(tmp_path)
    duplicate = tmp_path / "duplicate-tp8ep8-bt8000" / "phase164_result.json"
    duplicate.parent.mkdir(parents=True)
    _write_result(
        duplicate,
        scenario="tp8ep8-bt8000",
        tp=8,
        dp=1,
        ep=8,
        max_bt=8000,
    )

    with pytest.raises(ValueError, match="duplicate scenario"):
        builder.build_manifest([*result_paths, duplicate])


def test_manifest_rejects_failed_requests(tmp_path: Path) -> None:
    path = tmp_path / "tp8ep8-bt8000" / "phase164_result.json"
    path.parent.mkdir(parents=True)
    _write_result(
        path,
        scenario="tp8ep8-bt8000",
        tp=8,
        dp=1,
        ep=8,
        max_bt=8000,
        failed_requests=1,
    )

    with pytest.raises(ValueError, match="failed_requests"):
        builder.build_manifest([path])


def test_manifest_rejects_missing_clean_schema_field(tmp_path: Path) -> None:
    path = tmp_path / "tp8ep8-bt8000" / "phase164_result.json"
    path.parent.mkdir(parents=True)
    _write_result(
        path,
        scenario="tp8ep8-bt8000",
        tp=8,
        dp=1,
        ep=8,
        max_bt=8000,
        omit="runtime.serve_command",
    )

    with pytest.raises(ValueError, match="runtime.serve_command"):
        builder.build_manifest([path])
