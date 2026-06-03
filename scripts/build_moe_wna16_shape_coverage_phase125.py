#!/usr/bin/env python3
"""Build the Phase125 exact-shape MoE WNA16 coverage manifest."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REQUIRED_FIELDS = {
    "tokens_actual",
    "occurrence_count",
    "phase117_exact_key_covered",
}
FORBIDDEN_FIELD_RE = re.compile(
    r"(^|_)(timing|latency|cuda_event|wall|residual|profiler|profiled|nccl|sync|throughput)($|_)|_ms$",
    re.IGNORECASE,
)
MANIFEST_FIELDS = [
    "tokens_actual",
    "occurrence_count",
    "phase117_exact_key_covered",
    "phase125_action",
    "valid_for_time_gate",
    "valid_for_default",
    "perf_database",
    "notes",
]


@dataclass(frozen=True)
class CoverageRow:
    tokens_actual: int
    occurrence_count: int
    phase117_exact_key_covered: bool


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Phase125 exact-shape MoE WNA16 coverage manifest."
    )
    parser.add_argument("--coverage", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def _check_header(fieldnames: Sequence[str] | None, path: Path) -> None:
    if fieldnames is None:
        raise ValueError(f"{path} has no CSV header")
    fields = list(fieldnames)
    forbidden = [field for field in fields if FORBIDDEN_FIELD_RE.search(field)]
    if forbidden:
        raise ValueError(f"{path} contains forbidden field: {', '.join(forbidden)}")
    missing = sorted(REQUIRED_FIELDS - set(fields))
    if missing:
        raise ValueError(f"{path} missing required fields: {', '.join(missing)}")


def _int_value(row: dict[str, str], field: str, path: Path) -> int:
    value = row[field]
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {field} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{path}: {field} must be positive")
    return parsed


def _bool_value(value: str, field: str, path: Path) -> bool:
    if value is None:
        raise ValueError(f"{path}: {field} must be true or false")
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{path}: {field} must be true or false")


def read_coverage(path: Path) -> list[CoverageRow]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        _check_header(reader.fieldnames, path)
        rows = [
            CoverageRow(
                tokens_actual=_int_value(row, "tokens_actual", path),
                occurrence_count=_int_value(row, "occurrence_count", path),
                phase117_exact_key_covered=_bool_value(
                    row["phase117_exact_key_covered"],
                    "phase117_exact_key_covered",
                    path,
                ),
            )
            for row in reader
        ]
    if not rows:
        raise ValueError(f"{path} must contain rows")

    seen: set[int] = set()
    duplicates: list[int] = []
    for row in rows:
        if row.tokens_actual in seen:
            duplicates.append(row.tokens_actual)
        seen.add(row.tokens_actual)
    if duplicates:
        duplicate_list = ", ".join(str(token) for token in sorted(set(duplicates)))
        raise ValueError(f"{path} duplicate tokens_actual: {duplicate_list}")

    return sorted(rows, key=lambda row: row.tokens_actual)


def build_manifest_rows(rows: Sequence[CoverageRow]) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []
    for row in rows:
        if row.phase117_exact_key_covered:
            action = "already_has_phase117_exact_shape"
            notes = "descriptor_only"
        else:
            action = "needs_exact_shape_evidence"
            notes = "no_interpolation_or_extrapolation"
        manifest.append(
            {
                "tokens_actual": str(row.tokens_actual),
                "occurrence_count": str(row.occurrence_count),
                "phase117_exact_key_covered": str(
                    row.phase117_exact_key_covered
                ).lower(),
                "phase125_action": action,
                "valid_for_time_gate": "false",
                "valid_for_default": "false",
                "perf_database": "false",
                "notes": notes,
            }
        )
    return manifest


def _write_manifest(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=MANIFEST_FIELDS,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_rows = build_manifest_rows(read_coverage(args.coverage))
    _write_manifest(args.out, manifest_rows)
    total_count = sum(int(row["occurrence_count"]) for row in manifest_rows)
    print(f"phase125_manifest_rows={len(manifest_rows)}")
    print(f"phase125_manifest_total_count={total_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
