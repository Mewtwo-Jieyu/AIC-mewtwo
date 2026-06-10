#!/usr/bin/env python3
"""Build Phase178 budget mechanism holdout manifest from per-scenario JSON results."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


EXPECTED_SCENARIOS: dict[str, dict[str, Any]] = {
    "tp8ep8-4k2k-bt4000": {
        "topology_key": "tp8_dp1_ep8",
        "shape_key": "isl4000_osl2000_batch128",
        "tp": 8,
        "dp": 1,
        "ep": 8,
        "isl": 4000,
        "osl": 2000,
        "batch_size": 128,
        "max_num_batched_tokens": 4000,
        "role": "control",
    },
    "tp8ep8-4k2k-bt65536": {
        "topology_key": "tp8_dp1_ep8",
        "shape_key": "isl4000_osl2000_batch128",
        "tp": 8,
        "dp": 1,
        "ep": 8,
        "isl": 4000,
        "osl": 2000,
        "batch_size": 128,
        "max_num_batched_tokens": 65536,
        "role": "holdout",
    },
    "tp4dp2ep8-4k2k-bt4000": {
        "topology_key": "tp4_dp2_ep8",
        "shape_key": "isl4000_osl2000_batch128",
        "tp": 4,
        "dp": 2,
        "ep": 8,
        "isl": 4000,
        "osl": 2000,
        "batch_size": 128,
        "max_num_batched_tokens": 4000,
        "role": "control",
    },
    "tp4dp2ep8-4k2k-bt65536": {
        "topology_key": "tp4_dp2_ep8",
        "shape_key": "isl4000_osl2000_batch128",
        "tp": 4,
        "dp": 2,
        "ep": 8,
        "isl": 4000,
        "osl": 2000,
        "batch_size": 128,
        "max_num_batched_tokens": 65536,
        "role": "holdout",
    },
    "tp8ep8-12k2k-bt12000": {
        "topology_key": "tp8_dp1_ep8",
        "shape_key": "isl12000_osl2000_batch128",
        "tp": 8,
        "dp": 1,
        "ep": 8,
        "isl": 12000,
        "osl": 2000,
        "batch_size": 128,
        "max_num_batched_tokens": 12000,
        "role": "control",
    },
    "tp8ep8-12k2k-bt65536": {
        "topology_key": "tp8_dp1_ep8",
        "shape_key": "isl12000_osl2000_batch128",
        "tp": 8,
        "dp": 1,
        "ep": 8,
        "isl": 12000,
        "osl": 2000,
        "batch_size": 128,
        "max_num_batched_tokens": 65536,
        "role": "holdout",
    },
    "tp4dp2ep8-12k2k-bt12000": {
        "topology_key": "tp4_dp2_ep8",
        "shape_key": "isl12000_osl2000_batch128",
        "tp": 4,
        "dp": 2,
        "ep": 8,
        "isl": 12000,
        "osl": 2000,
        "batch_size": 128,
        "max_num_batched_tokens": 12000,
        "role": "control",
    },
    "tp4dp2ep8-12k2k-bt65536": {
        "topology_key": "tp4_dp2_ep8",
        "shape_key": "isl12000_osl2000_batch128",
        "tp": 4,
        "dp": 2,
        "ep": 8,
        "isl": 12000,
        "osl": 2000,
        "batch_size": 128,
        "max_num_batched_tokens": 65536,
        "role": "holdout",
    },
}

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


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid json: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"result json must be an object: {path}")
    return payload


def _section(payload: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"missing {key} object in {path}")
    return value


def _required(mapping: dict[str, Any], key: str, prefix: str, path: Path) -> Any:
    if key not in mapping or mapping[key] in (None, ""):
        raise ValueError(f"missing {prefix}.{key} in {path}")
    return mapping[key]


def _as_int(value: Any, field: str, path: Path) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be int in {path}: {value!r}") from exc


def _as_float(value: Any, field: str, path: Path) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be float in {path}: {value!r}") from exc


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _validate_boundary_flags(payload: dict[str, Any], path: Path) -> None:
    expected = {
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
    }
    for key, expected_value in expected.items():
        if payload.get(key) is not expected_value:
            raise ValueError(f"{key} must be {expected_value} in {path}")


def _build_row(path: Path) -> dict[str, str]:
    payload = _load_json(path)
    _validate_boundary_flags(payload, path)
    bench = _section(payload, "bench_result", path)
    shape = _section(payload, "shape", path)
    parallelism = _section(payload, "parallelism", path)
    runtime = _section(payload, "runtime", path)
    hardware = _section(payload, "hardware", path)
    scheduler = _section(payload, "scheduler", path)

    scenario = str(_required(payload, "scenario", "root", path))
    if scenario not in EXPECTED_SCENARIOS:
        raise ValueError(f"unexpected scenario in {path}: {scenario}")
    expected = EXPECTED_SCENARIOS[scenario]

    topology_key = str(_required(payload, "topology_key", "root", path))
    shape_key = str(_required(payload, "shape_key", "root", path))
    tp = _as_int(_required(parallelism, "tp", "parallelism", path), "parallelism.tp", path)
    dp = _as_int(_required(parallelism, "dp", "parallelism", path), "parallelism.dp", path)
    ep = _as_int(_required(parallelism, "ep", "parallelism", path), "parallelism.ep", path)
    isl = _as_int(_required(shape, "isl", "shape", path), "shape.isl", path)
    osl = _as_int(_required(shape, "osl", "shape", path), "shape.osl", path)
    batch_size = _as_int(
        _required(shape, "batch_size", "shape", path),
        "shape.batch_size",
        path,
    )
    max_bt = _as_int(
        _required(shape, "max_num_batched_tokens", "shape", path),
        "shape.max_num_batched_tokens",
        path,
    )

    if (topology_key, shape_key, tp, dp, ep, isl, osl, batch_size) != (
        expected["topology_key"],
        expected["shape_key"],
        expected["tp"],
        expected["dp"],
        expected["ep"],
        expected["isl"],
        expected["osl"],
        expected["batch_size"],
    ):
        raise ValueError(f"scenario metadata mismatch for {scenario} in {path}")

    failed_requests = _as_int(
        _required(bench, "failed_requests", "bench_result", path),
        "bench_result.failed_requests",
        path,
    )
    if failed_requests != 0:
        raise ValueError(f"failed_requests must be 0 in {path}: {failed_requests}")

    gpu_count = _as_int(
        _required(parallelism, "gpu_count", "parallelism", path),
        "parallelism.gpu_count",
        path,
    )
    if gpu_count <= 0:
        raise ValueError(f"parallelism.gpu_count must be positive in {path}")

    output_tok_s = _as_float(
        _required(bench, "output_tok_s", "bench_result", path),
        "bench_result.output_tok_s",
        path,
    )
    total_tok_s = _as_float(
        _required(bench, "total_tok_s", "bench_result", path),
        "bench_result.total_tok_s",
        path,
    )
    steady_state_time_ms = _as_float(
        _required(scheduler, "steady_state_time_ms", "scheduler", path),
        "scheduler.steady_state_time_ms",
        path,
    )

    return {
        "source": "phase178_budget_mechanism_holdout",
        "scenario": scenario,
        "topology_key": topology_key,
        "shape_key": shape_key,
        "role": str(expected["role"]),
        "tp": str(tp),
        "dp": str(dp),
        "ep": str(ep),
        "world_size": str(
            _as_int(
                _required(parallelism, "world_size", "parallelism", path),
                "parallelism.world_size",
                path,
            )
        ),
        "gpu_count": str(gpu_count),
        "gpu_model": str(_required(hardware, "gpu_model", "hardware", path)),
        "gpu_memory_gb": str(_required(hardware, "gpu_memory_gb", "hardware", path)),
        "driver_version": str(_required(hardware, "driver_version", "hardware", path)),
        "cuda_version": str(_required(hardware, "cuda_version", "hardware", path)),
        "model_path": str(_required(runtime, "model_path", "runtime", path)),
        "vllm_version": str(_required(runtime, "vllm_version", "runtime", path)),
        "dtype": str(_required(runtime, "dtype", "runtime", path)),
        "quantization": str(_required(runtime, "quantization", "runtime", path)),
        "isl": str(isl),
        "osl": str(osl),
        "batch_size": str(batch_size),
        "max_num_batched_tokens": str(max_bt),
        "max_num_seqs": str(
            _as_int(
                _required(shape, "max_num_seqs", "shape", path),
                "shape.max_num_seqs",
                path,
            )
        ),
        "num_prompts": str(
            _as_int(
                _required(bench, "num_prompts", "bench_result", path),
                "bench_result.num_prompts",
                path,
            )
        ),
        "max_concurrency": str(
            _as_int(
                _required(bench, "max_concurrency", "bench_result", path),
                "bench_result.max_concurrency",
                path,
            )
        ),
        "request_success_count": str(
            _as_int(
                _required(bench, "ok_requests", "bench_result", path),
                "bench_result.ok_requests",
                path,
            )
        ),
        "request_fail_count": str(failed_requests),
        "real_output_tok_s": _format_float(output_tok_s),
        "real_total_tok_s": _format_float(total_tok_s),
        "real_output_tok_s_gpu": _format_float(output_tok_s / gpu_count),
        "real_total_tok_s_gpu": _format_float(total_tok_s / gpu_count),
        "steady_state_time_ms": _format_float(steady_state_time_ms),
        "serve_command": str(_required(runtime, "serve_command", "runtime", path)),
        "benchmark_command": str(_required(runtime, "benchmark_command", "runtime", path)),
        "diagnostic_only": "true",
        "valid_for_default": "false",
        "perf_database": "false",
    }


def _validate_pairs(rows: list[dict[str, str]]) -> None:
    by_pair: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        pair_key = (row["topology_key"], row["shape_key"])
        role = "holdout" if row["max_num_batched_tokens"] == "65536" else "control"
        by_pair.setdefault(pair_key, set()).add(role)
    for pair_key, roles in sorted(by_pair.items()):
        if roles != {"control", "holdout"}:
            raise ValueError(f"control/high-budget pair missing for {pair_key}: {sorted(roles)}")


def _validate_scenario_roles(rows: list[dict[str, str]]) -> None:
    for row in rows:
        expected = EXPECTED_SCENARIOS[row["scenario"]]
        if row["max_num_batched_tokens"] != str(expected["max_num_batched_tokens"]):
            raise ValueError(f"scenario max_bt mismatch for {row['scenario']}")
        if row["role"] != expected["role"]:
            raise ValueError(f"scenario role mismatch for {row['scenario']}")


def build_manifest(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows = [_build_row(Path(path)) for path in paths]
    scenario_counts = Counter(row["scenario"] for row in rows)
    duplicate_scenarios = sorted(
        scenario for scenario, count in scenario_counts.items() if count > 1
    )
    if duplicate_scenarios:
        raise ValueError(f"duplicate scenario rows in manifest: {duplicate_scenarios}")

    expected_row_count = len(EXPECTED_SCENARIOS)
    if len(rows) != expected_row_count:
        raise ValueError(
            f"manifest must contain exactly {expected_row_count} rows: got {len(rows)}"
        )

    seen = {row["scenario"] for row in rows}
    expected = set(EXPECTED_SCENARIOS)
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(f"manifest scenario set mismatch: missing={missing} extra={extra}")

    _validate_pairs(rows)
    _validate_scenario_roles(rows)
    rows.sort(key=lambda row: list(EXPECTED_SCENARIOS).index(row["scenario"]))
    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("no manifest rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _discover_result_jsons(input_root: Path) -> list[Path]:
    return sorted(input_root.glob("*/phase178_result.json"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build Phase178 budget mechanism holdout manifest."
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        action="append",
        default=[],
        help="Per-scenario phase178_result.json. May be repeated.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Root containing */phase178_result.json files.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    paths = list(args.result_json)
    if args.input_root is not None:
        paths.extend(_discover_result_jsons(args.input_root))
    if not paths:
        raise SystemExit("no result json files provided")

    rows = build_manifest(paths)
    write_manifest(args.out, rows)
    print(f"wrote_phase178_manifest={args.out}")
    print(f"phase178_manifest_rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
