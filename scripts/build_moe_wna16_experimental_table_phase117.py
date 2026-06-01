#!/usr/bin/env python3
"""Build a diagnostic-only exact-key MoE WNA16 table for Phase 117."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


EXPECTED_TOKENS = (128, 248, 512, 1024)
CONFIG_SHA256 = "15c797eed1dd441e1d6d50fcf7584b2264d248d5ebc1144e1c95e6c4e885d853"
MODEL_USE = "experimental_table_prototype_only"
QUERY_POLICY = "exact_match_only"

REQUIRED_FIELDS = {
    "tokens_actual",
    "run_count",
    "cuda_event_ms_mean",
    "wall_ms_mean",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
}

FORBIDDEN_FIELD_PARTS = (
    "latency",
    "residual",
    "profiler",
    "profiled",
    "nccl",
    "sync",
    "throughput",
)

TABLE_FIELDS = [
    "backend",
    "vllm_version",
    "device",
    "model_family",
    "module",
    "experts_impl",
    "kernel",
    "dtype",
    "activation_dtype",
    "hidden",
    "intermediate",
    "global_experts",
    "local_experts",
    "topk",
    "topology",
    "config_sha256",
    "input_kind",
    "tokens_actual",
    "cuda_event_ms_mean",
    "wall_ms_mean",
    "run_count",
    "query_policy",
    "model_use",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]


@dataclass(frozen=True)
class ShapeRow:
    tokens_actual: int
    run_count: int
    cuda_event_ms_mean: float
    wall_ms_mean: float
    diagnostic_only: bool
    valid_for_default: bool
    perf_database: bool


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Phase117 exact-key experimental MoE WNA16 table."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase114_shape_trend_summary.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "docs/iter_gap_investigation/"
            "phase117_moe_wna16_experimental_table.csv"
        ),
    )
    return parser.parse_args(argv)


def _bool_value(value: str, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{field} must be boolean-like, got {value!r}")


def _check_header(fieldnames: Sequence[str] | None, path: Path) -> list[str]:
    if fieldnames is None:
        raise ValueError(f"{path} has no CSV header")
    fields = list(fieldnames)
    forbidden = [
        field
        for field in fields
        if any(part in field.lower() for part in FORBIDDEN_FIELD_PARTS)
    ]
    if forbidden:
        raise ValueError(f"{path} contains forbidden fields: {', '.join(forbidden)}")
    missing = sorted(REQUIRED_FIELDS - set(fields))
    if missing:
        raise ValueError(f"{path} missing required fields: {', '.join(missing)}")
    return fields


def _int_value(row: dict[str, str], field: str, path: Path) -> int:
    try:
        return int(row[field])
    except ValueError as exc:
        raise ValueError(f"{path}: {field} must be an integer") from exc


def _float_value(row: dict[str, str], field: str, path: Path) -> float:
    try:
        return float(row[field])
    except ValueError as exc:
        raise ValueError(f"{path}: {field} must be a number") from exc


def _format_float(value: float) -> str:
    return f"{value:.12g}"


def read_shape_summary(path: Path) -> list[ShapeRow]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        _check_header(reader.fieldnames, path)
        rows = [
            ShapeRow(
                tokens_actual=_int_value(row, "tokens_actual", path),
                run_count=_int_value(row, "run_count", path),
                cuda_event_ms_mean=_float_value(row, "cuda_event_ms_mean", path),
                wall_ms_mean=_float_value(row, "wall_ms_mean", path),
                diagnostic_only=_bool_value(row["diagnostic_only"], "diagnostic_only"),
                valid_for_default=_bool_value(
                    row["valid_for_default"], "valid_for_default"
                ),
                perf_database=_bool_value(row["perf_database"], "perf_database"),
            )
            for row in reader
        ]
    if not rows:
        raise ValueError(f"{path} must contain rows")
    tokens = tuple(row.tokens_actual for row in rows)
    if tokens != EXPECTED_TOKENS:
        raise ValueError(
            f"{path} must contain exact token shapes {EXPECTED_TOKENS}, got {tokens}"
        )
    for row in rows:
        if not row.diagnostic_only:
            raise ValueError("diagnostic_only must be true")
        if row.valid_for_default:
            raise ValueError("valid_for_default must be false")
        if row.perf_database:
            raise ValueError("perf_database must be false")
    return rows


def build_table_rows(rows: Sequence[ShapeRow]) -> list[dict[str, str]]:
    return [
        {
            "backend": "vllm",
            "vllm_version": "0.19.0",
            "device": "NVIDIA_H200",
            "model_family": "KimiK25",
            "module": "DeepseekV2MoE",
            "experts_impl": "SharedFusedMoE",
            "kernel": "WNA16",
            "dtype": "int4_w4a16",
            "activation_dtype": "bfloat16",
            "hidden": "7168",
            "intermediate": "2048",
            "global_experts": "384",
            "local_experts": "48",
            "topk": "8",
            "topology": "tp4dp2ep8",
            "config_sha256": CONFIG_SHA256,
            "input_kind": "synthetic_random_hidden_states",
            "tokens_actual": str(row.tokens_actual),
            "cuda_event_ms_mean": _format_float(row.cuda_event_ms_mean),
            "wall_ms_mean": _format_float(row.wall_ms_mean),
            "run_count": str(row.run_count),
            "query_policy": QUERY_POLICY,
            "model_use": MODEL_USE,
            "diagnostic_only": "true",
            "valid_for_default": "false",
            "perf_database": "false",
        }
        for row in rows
    ]


def query_exact_token(table_rows: Sequence[dict[str, str]], tokens_actual: int) -> dict[str, str]:
    matches = [
        row for row in table_rows if int(row["tokens_actual"]) == int(tokens_actual)
    ]
    if len(matches) != 1:
        raise KeyError(f"exact token key missing: {tokens_actual}")
    row = matches[0]
    if row.get("query_policy") != QUERY_POLICY:
        raise ValueError("query_policy must be exact_match_only")
    if row.get("valid_for_default") != "false" or row.get("perf_database") != "false":
        raise ValueError("table row must not be valid for default or perf database")
    return row


def _write_table(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    table_rows = build_table_rows(read_shape_summary(args.input))
    _write_table(args.out, table_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
