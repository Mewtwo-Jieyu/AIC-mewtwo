#!/usr/bin/env python3
"""Query the Phase118 diagnostic-only MoE WNA16 exact-key table."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_TABLE = Path(
    "docs/iter_gap_investigation/phase117_moe_wna16_experimental_table.csv"
)

REQUIRED_QUERY_FIELDS = [
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
]

RESULT_FIELDS = [
    *REQUIRED_QUERY_FIELDS,
    "cuda_event_ms_mean",
    "wall_ms_mean",
    "run_count",
    "query_policy",
    "model_use",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

FORBIDDEN_FIELD_PARTS = (
    "latency",
    "residual",
    "profiler",
    "profiled",
    "nccl",
    "sync",
    "throughput",
)


class QueryError(ValueError):
    """Fail-fast query error for unsafe or non-exact Phase118 inputs."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the diagnostic-only Phase118 MoE WNA16 exact-key table."
    )
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--key-json", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def _check_header(fieldnames: Sequence[str] | None, path: Path) -> list[str]:
    if fieldnames is None:
        raise QueryError(f"{path} has no CSV header")
    fields = list(fieldnames)
    forbidden = [
        field
        for field in fields
        if any(part in field.lower() for part in FORBIDDEN_FIELD_PARTS)
    ]
    if forbidden:
        raise QueryError(f"{path} contains forbidden fields: {', '.join(forbidden)}")
    missing = [field for field in RESULT_FIELDS if field not in fields]
    if missing:
        raise QueryError(f"{path} missing required fields: {', '.join(missing)}")
    return fields


def _require_safe_row(row: dict[str, str]) -> None:
    if row.get("diagnostic_only") != "true":
        raise QueryError("diagnostic_only must be true")
    if row.get("valid_for_default") != "false":
        raise QueryError("valid_for_default must be false")
    if row.get("perf_database") != "false":
        raise QueryError("perf_database must be false")
    if row.get("query_policy") != "exact_match_only":
        raise QueryError("query_policy must be exact_match_only")
    if row.get("model_use") != "experimental_table_prototype_only":
        raise QueryError("model_use must be experimental_table_prototype_only")


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = _check_header(reader.fieldnames, path)
        rows = [
            {field: row[field] for field in fields}
            for row in reader
        ]
    if not rows:
        raise QueryError(f"{path} must contain rows")
    for row in rows:
        _require_safe_row(row)
    return rows


def read_key_json(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise QueryError(f"{path} must contain a JSON object")
    missing = [field for field in REQUIRED_QUERY_FIELDS if field not in data]
    if missing:
        raise QueryError(f"missing query fields: {', '.join(missing)}")
    unexpected = sorted(set(data) - set(REQUIRED_QUERY_FIELDS))
    if unexpected:
        raise QueryError(f"unexpected query fields: {', '.join(unexpected)}")
    return {field: str(data[field]) for field in REQUIRED_QUERY_FIELDS}


def _normalize_query_key(query_key: dict[str, object]) -> dict[str, str]:
    missing = [field for field in REQUIRED_QUERY_FIELDS if field not in query_key]
    if missing:
        raise QueryError(f"missing query fields: {', '.join(missing)}")
    unexpected = sorted(set(query_key) - set(REQUIRED_QUERY_FIELDS))
    if unexpected:
        raise QueryError(f"unexpected query fields: {', '.join(unexpected)}")
    return {field: str(query_key[field]) for field in REQUIRED_QUERY_FIELDS}


def query_exact_key(
    table_rows: Sequence[dict[str, str]],
    query_key: dict[str, object],
) -> dict[str, str]:
    normalized_key = _normalize_query_key(query_key)
    matches = [
        row
        for row in table_rows
        if all(row[field] == normalized_key[field] for field in REQUIRED_QUERY_FIELDS)
    ]
    if not matches:
        same_token_rows = [
            row
            for row in table_rows
            if row["tokens_actual"] == normalized_key["tokens_actual"]
        ]
        if len(same_token_rows) == 1:
            mismatched = [
                field
                for field in REQUIRED_QUERY_FIELDS
                if same_token_rows[0][field] != normalized_key[field]
            ]
            raise QueryError(
                "no exact row for full key; mismatched fields: "
                + ", ".join(mismatched)
            )
        raise QueryError("no exact row for full key")
    if len(matches) > 1:
        raise QueryError("duplicate exact row for full key")
    row = matches[0]
    _require_safe_row(row)
    return {field: row[field] for field in RESULT_FIELDS}


def write_result(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    row = query_exact_key(read_table(args.table), read_key_json(args.key_json))
    if args.out is not None:
        write_result(args.out, row)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QueryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
