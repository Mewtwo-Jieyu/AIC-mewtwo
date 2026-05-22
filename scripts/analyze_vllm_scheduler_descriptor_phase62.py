#!/usr/bin/env python3
"""Parse Phase 62 vLLM scheduler descriptor alignment markers."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from aiconfigurator.sdk.backends.cb_simulator.forward_descriptor import (
    VLLMSchedulerRuntimeDescriptor,
    scheduler_runtime_descriptor_from_scheduled,
)


MARKER = "AIC_SCHEDULER_ALIGNMENT_DESCRIPTOR_ROW"
MARKER_RE = re.compile(rf"{MARKER}\s+(?P<body>{{.*}})$")

FORBIDDEN_KEY_RE = re.compile(
    r"(^|_)(latency|duration|residual|profiler|nccl|sync|throughput)($|_)|_ms$",
    re.IGNORECASE,
)

ALLOWED_ALIGNMENT_KEY_TYPES = {
    "engine_core_dp_step",
    "request_set_hash",
    "phase_ordinal",
}

RAW_IDENTITY_FIELDS = [
    "line_no",
    "alignment_key",
    "alignment_key_type",
    "engine_step_id",
    "rank",
    "local_rank",
    "dp_rank",
    "tp_rank",
    "ep_rank",
]
DESCRIPTOR_FIELDS = [field.name for field in fields(VLLMSchedulerRuntimeDescriptor)]
ROW_FIELDS = RAW_IDENTITY_FIELDS + DESCRIPTOR_FIELDS
DEDUP_FIELDS = [
    key
    for key in ROW_FIELDS
    if key not in {"line_no", "rank", "local_rank", "tp_rank", "ep_rank"}
]

INT_FIELDS = {
    "engine_step_id",
    "iteration",
    "scheduled_context_tokens",
    "scheduled_decode_tokens",
    "scheduled_total_tokens",
    "scheduled_context_reqs",
    "scheduled_decode_reqs",
    "scheduled_total_reqs",
    "max_num_batched_tokens",
    "max_num_seqs",
    "forward_token_count",
    "tp",
    "dp",
    "moe_tp",
    "moe_ep",
    "rank",
    "local_rank",
    "dp_rank",
    "tp_rank",
    "ep_rank",
}

BOOL_FIELDS = {
    "valid_for_default",
    "perf_database",
    "diagnostic_only",
}


def _reject_forbidden_keys(payload: dict[str, Any], line_no: int) -> None:
    for key in payload:
        if FORBIDDEN_KEY_RE.search(key):
            raise ValueError(f"forbidden field {key!r} at line {line_no}")


def _coerce_bool(value: object, field: str, line_no: int) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"invalid bool for {field} at line {line_no}: {value!r}")


def _coerce_int(value: object, field: str, line_no: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"invalid int for {field} at line {line_no}: {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid int for {field} at line {line_no}: {value!r}") from exc


def _require_fields(payload: dict[str, Any], line_no: int) -> None:
    required = set(DESCRIPTOR_FIELDS) | set(RAW_IDENTITY_FIELDS) - {"line_no"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"missing {','.join(missing)} at line {line_no}")


def _normalize_payload(payload: dict[str, Any], line_no: int) -> dict[str, Any]:
    _reject_forbidden_keys(payload, line_no)
    _require_fields(payload, line_no)

    normalized: dict[str, Any] = {"line_no": line_no}
    for key, value in payload.items():
        if key in INT_FIELDS:
            normalized[key] = _coerce_int(value, key, line_no)
        elif key in BOOL_FIELDS:
            normalized[key] = _coerce_bool(value, key, line_no)
        else:
            normalized[key] = str(value)

    if normalized["source"] != "vllm_scheduler":
        raise ValueError(f"source must be vllm_scheduler at line {line_no}")
    if normalized["alignment_key_type"] not in ALLOWED_ALIGNMENT_KEY_TYPES:
        raise ValueError(f"invalid alignment_key_type at line {line_no}")
    if not normalized["alignment_key"]:
        raise ValueError(f"alignment_key must be non-empty at line {line_no}")
    if normalized["alignment_key_type"] == "engine_core_dp_step":
        expected_key = (
            f"engine_dp:{normalized['dp_rank']}:step:{normalized['engine_step_id']}"
        )
        if normalized["alignment_key"] != expected_key:
            raise ValueError(
                f"alignment_key mismatch at line {line_no}: "
                f"{normalized['alignment_key']!r} != {expected_key!r}"
            )
        if normalized["iteration"] != normalized["engine_step_id"]:
            raise ValueError(f"iteration must equal engine_step_id at line {line_no}")
    if normalized["valid_for_default"]:
        raise ValueError(f"valid_for_default must be false at line {line_no}")
    if normalized["perf_database"]:
        raise ValueError(f"perf_database must be false at line {line_no}")
    if not normalized["diagnostic_only"]:
        raise ValueError(f"diagnostic_only must be true at line {line_no}")

    descriptor = scheduler_runtime_descriptor_from_scheduled(
        source=str(normalized["source"]),
        scenario=str(normalized["scenario"]),
        iteration=int(normalized["iteration"]),
        phase=str(normalized["phase"]),
        scheduled_context_tokens=int(normalized["scheduled_context_tokens"]),
        scheduled_decode_tokens=int(normalized["scheduled_decode_tokens"]),
        scheduled_context_reqs=int(normalized["scheduled_context_reqs"]),
        scheduled_decode_reqs=int(normalized["scheduled_decode_reqs"]),
        max_num_batched_tokens=int(normalized["max_num_batched_tokens"]),
        max_num_seqs=int(normalized["max_num_seqs"]),
        forward_token_count=int(normalized["forward_token_count"]),
        cudagraph_runtime_mode=str(normalized["cudagraph_runtime_mode"]),
        topology_key=str(normalized["topology_key"]),
        tp=int(normalized["tp"]),
        dp=int(normalized["dp"]),
        moe_tp=int(normalized["moe_tp"]),
        moe_ep=int(normalized["moe_ep"]),
    )
    descriptor_values = asdict(descriptor)
    for key in DESCRIPTOR_FIELDS:
        if normalized[key] != descriptor_values[key]:
            raise ValueError(
                f"{key} mismatch at line {line_no}: "
                f"{normalized[key]!r} != {descriptor_values[key]!r}"
            )

    return {key: normalized[key] for key in ROW_FIELDS}


def parse_rows(log_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(log_path.read_text(errors="ignore").splitlines(), 1):
        match = MARKER_RE.search(line)
        if not match:
            continue
        try:
            payload = json.loads(match.group("body"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON marker at line {line_no}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"marker JSON must be object at line {line_no}")
        rows.append(_normalize_payload(payload, line_no))
    if not rows:
        raise ValueError(f"no {MARKER} rows found")
    return rows


def dedupe_rows_by_alignment_dp(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["alignment_key"]), int(row["dp_rank"]))
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group = grouped[key]
        shape_values = {tuple(row[field] for field in DEDUP_FIELDS) for row in group}
        if len(shape_values) != 1:
            raise ValueError(f"inconsistent TP rank rows for alignment/dp {key}")
        preferred_rank = int(key[1]) * int(group[0]["tp"])
        selected = next(
            (row for row in group if int(row["rank"]) == preferred_rank),
            group[0],
        )
        output.append({field: selected[field] for field in DEDUP_FIELDS})
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--rows-out", required=True, type=Path)
    parser.add_argument("--dedup-by-alignment-dp-out", type=Path)
    args = parser.parse_args(argv)

    rows = parse_rows(args.log)
    _write_csv(args.rows_out, rows, ROW_FIELDS)
    if args.dedup_by_alignment_dp_out:
        dedup_rows = dedupe_rows_by_alignment_dp(rows)
        _write_csv(args.dedup_by_alignment_dp_out, dedup_rows, DEDUP_FIELDS)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
