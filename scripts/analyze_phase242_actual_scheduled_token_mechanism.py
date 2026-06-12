#!/usr/bin/env python3
"""Summarize the Phase242 actual scheduled token mechanism."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


SOURCE = "phase242_actual_scheduled_token_mechanism"
TRACE_SOURCE = "phase234_vllm_scheduler_trace"
MECHANISM_HYPOTHESIS = "actual_scheduled_tokens_not_configured_budget"
DEFAULT_READINESS = "No-Go"
CONTROL_SCENARIO = "tp8ep8-4k2k-bt4000"
HOLDOUT_SCENARIO = "tp8ep8-4k2k-bt65536"
CONTROL_MAX_BT = 4000
HOLDOUT_MAX_BT = 65536
MAX_BUDGET_FILL_FOR_MECHANISM = 0.25

FIELDNAMES = [
    "source",
    "scenario",
    "role",
    "topology_key",
    "shape_key",
    "max_num_batched_tokens",
    "max_scheduled_total_tokens",
    "p50_scheduled_total_tokens",
    "p99_scheduled_total_tokens",
    "mean_budget_fill_ratio",
    "max_budget_fill_ratio",
    "output_tok_s",
    "output_ratio_vs_control",
    "mechanism_hypothesis",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

REQUIRED_TRACE_FIELDS = {
    "source",
    "scenario",
    "iteration",
    "phase",
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
    "ep",
    "topology_key",
    "shape_key",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
}


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected object json: {path}")
    return data


def _read_trace(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"trace row must be object in {path}")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty trace: {path}")
    return rows


def _required_int(row: dict[str, object], field: str) -> int:
    value = row.get(field)
    if not isinstance(value, int):
        raise ValueError(f"{field} must be int: {value!r}")
    if value < 0:
        raise ValueError(f"{field} must be non-negative: {value!r}")
    return value


def _percentile(values: list[int], q: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo))


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _signature(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row["phase"],
        row["scheduled_context_tokens"],
        row["scheduled_decode_tokens"],
        row["scheduled_total_tokens"],
        row["scheduled_context_reqs"],
        row["scheduled_decode_reqs"],
        row["scheduled_total_reqs"],
        row["forward_token_count"],
        row["max_num_batched_tokens"],
        row["max_num_seqs"],
        row["tp"],
        row["dp"],
        row["ep"],
        row["topology_key"],
        row["shape_key"],
        row["diagnostic_only"],
        row["valid_for_default"],
        row["perf_database"],
    )


def _validate_result(
    result: dict[str, object],
    scenario: str,
    max_bt: int,
    result_path: Path,
) -> tuple[str, str]:
    if result.get("scenario") != scenario:
        raise ValueError(f"result scenario mismatch in {result_path}")
    topology_key = result.get("topology_key")
    shape_key = result.get("shape_key")
    if not isinstance(topology_key, str) or not topology_key:
        raise ValueError(f"missing topology_key in {result_path}")
    if not isinstance(shape_key, str) or not shape_key:
        raise ValueError(f"missing shape_key in {result_path}")
    shape = result.get("shape")
    if not isinstance(shape, dict):
        raise ValueError(f"missing shape in {result_path}")
    expected_shape = {
        "isl": 4000,
        "osl": 2000,
        "batch_size": 128,
        "max_num_batched_tokens": max_bt,
        "max_num_seqs": 256,
    }
    for field, expected in expected_shape.items():
        if shape.get(field) != expected:
            raise ValueError(f"shape {field} mismatch in {result_path}: {shape.get(field)!r}")
    parallelism = result.get("parallelism")
    if parallelism != {"tp": 8, "dp": 1, "ep": 8}:
        raise ValueError(f"parallelism mismatch in {result_path}: {parallelism!r}")
    if result.get("diagnostic_only") is not True:
        raise ValueError(f"result diagnostic_only must be true in {result_path}")
    if result.get("valid_for_default") is not False:
        raise ValueError(f"result valid_for_default must be false in {result_path}")
    if result.get("perf_database") is not False:
        raise ValueError(f"result perf_database must be false in {result_path}")
    return topology_key, shape_key


def _validate_trace_row(
    row: dict[str, object],
    scenario: str,
    topology_key: str,
    shape_key: str,
    max_bt: int,
    trace_path: Path,
) -> None:
    missing = REQUIRED_TRACE_FIELDS - row.keys()
    if missing:
        raise ValueError(f"trace row missing fields in {trace_path}: {sorted(missing)}")
    if row.get("source") != TRACE_SOURCE:
        raise ValueError(f"trace source mismatch in {trace_path}")
    if row.get("scenario") != scenario:
        raise ValueError(f"trace scenario mismatch in {trace_path}")
    if row.get("topology_key") != topology_key or row.get("shape_key") != shape_key:
        raise ValueError(f"trace topology/shape mismatch in {trace_path}")
    if row.get("max_num_batched_tokens") != max_bt:
        raise ValueError(f"trace max_num_batched_tokens mismatch in {trace_path}")
    if row.get("max_num_seqs") != 256:
        raise ValueError(f"trace max_num_seqs mismatch in {trace_path}")
    if row.get("tp") != 8 or row.get("dp") != 1 or row.get("ep") != 8:
        raise ValueError(f"trace parallelism mismatch in {trace_path}")
    if row.get("diagnostic_only") is not True or row.get("valid_for_default") is not False or row.get("perf_database") is not False:
        raise ValueError(f"trace flag mismatch in {trace_path}")
    if row.get("phase") not in {"prefill", "mixed", "pure_decode"}:
        raise ValueError(f"trace phase mismatch in {trace_path}")

    context_tokens = _required_int(row, "scheduled_context_tokens")
    decode_tokens = _required_int(row, "scheduled_decode_tokens")
    total_tokens = _required_int(row, "scheduled_total_tokens")
    context_reqs = _required_int(row, "scheduled_context_reqs")
    decode_reqs = _required_int(row, "scheduled_decode_reqs")
    total_reqs = _required_int(row, "scheduled_total_reqs")
    _required_int(row, "forward_token_count")
    _required_int(row, "iteration")
    if total_tokens != context_tokens + decode_tokens:
        raise ValueError(f"trace token sum mismatch in {trace_path}")
    if total_reqs != context_reqs + decode_reqs:
        raise ValueError(f"trace request sum mismatch in {trace_path}")


def _dedupe_trace(rows: list[dict[str, object]], trace_path: Path) -> list[dict[str, object]]:
    by_iteration: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        iteration = _required_int(row, "iteration")
        by_iteration[iteration].append(row)
    iteration_ids = sorted(by_iteration)
    if iteration_ids != list(range(iteration_ids[0], iteration_ids[-1] + 1)):
        raise ValueError(f"trace iterations must be contiguous in {trace_path}")
    for group in by_iteration.values():
        if len({_signature(row) for row in group}) != 1:
            raise ValueError(f"duplicate worker trace payload mismatch in {trace_path}")
    return [by_iteration[iteration][0] for iteration in iteration_ids]


def _analyze_case(input_root: Path, scenario: str, max_bt: int, role: str) -> dict[str, object]:
    case_dir = input_root / scenario
    if not case_dir.exists():
        raise ValueError(f"missing scenario: {scenario}")
    bench_path = case_dir / "bench_result.json"
    records_path = case_dir / "bench_records.jsonl"
    trace_path = case_dir / "scheduler_trace.jsonl"
    result_path = case_dir / "phase234_result.json"
    for path in (bench_path, records_path, trace_path, result_path):
        if not path.exists():
            raise ValueError(f"missing artifact: {path}")

    bench = _read_json(bench_path)
    result = _read_json(result_path)
    topology_key, shape_key = _validate_result(result, scenario, max_bt, result_path)
    if bench.get("ok_requests") != 128 or bench.get("failed_requests") != 0:
        raise ValueError(f"bench requests must be 128/0 in {bench_path}")
    records = records_path.read_text(encoding="utf-8").splitlines()
    if len(records) != 128:
        raise ValueError(f"bench_records.jsonl must contain 128 rows: {records_path}")

    trace_rows = _read_trace(trace_path)
    for row in trace_rows:
        _validate_trace_row(row, scenario, topology_key, shape_key, max_bt, trace_path)
    deduped_rows = _dedupe_trace(trace_rows, trace_path)
    scheduled_tokens = [_required_int(row, "scheduled_total_tokens") for row in deduped_rows]
    p50_tokens = _percentile(scheduled_tokens, 0.50)
    p99_tokens = _percentile(scheduled_tokens, 0.99)
    if p99_tokens <= 0:
        raise ValueError("p99_scheduled_total_tokens must be positive")
    max_tokens = max(scheduled_tokens)
    max_fill = max_tokens / max_bt
    output_tok_s = bench.get("output_tok_s")
    if not isinstance(output_tok_s, (int, float)) or output_tok_s <= 0:
        raise ValueError(f"output_tok_s must be positive in {bench_path}")
    return {
        "scenario": scenario,
        "role": role,
        "topology_key": topology_key,
        "shape_key": shape_key,
        "max_num_batched_tokens": max_bt,
        "max_scheduled_total_tokens": max_tokens,
        "p50_scheduled_total_tokens": p50_tokens,
        "p99_scheduled_total_tokens": p99_tokens,
        "mean_budget_fill_ratio": sum(token / max_bt for token in scheduled_tokens) / len(scheduled_tokens),
        "max_budget_fill_ratio": max_fill,
        "output_tok_s": float(output_tok_s),
    }


def analyze_actual_scheduled_token_mechanism(input_root: Path | str) -> list[dict[str, str]]:
    root = Path(input_root)
    cases = [
        _analyze_case(root, CONTROL_SCENARIO, CONTROL_MAX_BT, "control"),
        _analyze_case(root, HOLDOUT_SCENARIO, HOLDOUT_MAX_BT, "holdout"),
    ]
    control, holdout = cases
    if (control["topology_key"], control["shape_key"]) != (holdout["topology_key"], holdout["shape_key"]):
        raise ValueError("control and holdout must share topology/shape")
    if control["max_num_batched_tokens"] != CONTROL_MAX_BT or holdout["max_num_batched_tokens"] != HOLDOUT_MAX_BT:
        raise ValueError("control/holdout budget mismatch")
    if float(holdout["max_budget_fill_ratio"]) >= MAX_BUDGET_FILL_FOR_MECHANISM:
        raise ValueError("holdout max scheduled tokens must stay far below configured budget")

    control_output = float(control["output_tok_s"])
    rows: list[dict[str, str]] = []
    for case in cases:
        output_ratio = float(case["output_tok_s"]) / control_output
        rows.append(
            {
                "source": SOURCE,
                "scenario": str(case["scenario"]),
                "role": str(case["role"]),
                "topology_key": str(case["topology_key"]),
                "shape_key": str(case["shape_key"]),
                "max_num_batched_tokens": str(case["max_num_batched_tokens"]),
                "max_scheduled_total_tokens": str(case["max_scheduled_total_tokens"]),
                "p50_scheduled_total_tokens": _format_float(float(case["p50_scheduled_total_tokens"])),
                "p99_scheduled_total_tokens": _format_float(float(case["p99_scheduled_total_tokens"])),
                "mean_budget_fill_ratio": _format_float(float(case["mean_budget_fill_ratio"])),
                "max_budget_fill_ratio": _format_float(float(case["max_budget_fill_ratio"])),
                "output_tok_s": _format_float(float(case["output_tok_s"])),
                "output_ratio_vs_control": _format_float(output_ratio),
                "mechanism_hypothesis": MECHANISM_HYPOTHESIS,
                "default_readiness": DEFAULT_READINESS,
                "diagnostic_only": "true",
                "valid_for_default": "false",
                "perf_database": "false",
            }
        )
    return rows


def write_actual_scheduled_token_csv(path: Path | str, rows: list[dict[str, str]]) -> None:
    if len(rows) != 2:
        raise ValueError(f"phase242 mechanism csv must contain 2 rows: got {len(rows)}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_actual_scheduled_token_doc(path: Path | str, rows: list[dict[str, str]]) -> None:
    if len(rows) != 2:
        raise ValueError(f"phase242 doc expects 2 rows: got {len(rows)}")
    by_role = {row["role"]: row for row in rows}
    control = by_role["control"]
    holdout = by_role["holdout"]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase242: Actual Scheduled Token Mechanism",
        "",
        "## Decision",
        "",
        "| Item | Result |",
        "|---|---|",
        f"| Mechanism hypothesis | {MECHANISM_HYPOTHESIS} |",
        "| Default AIC | No-Go |",
        "| PerfDatabase | No-Go |",
        "| GPU benchmark | No-Go in Phase242 |",
        "| Boundary flags | diagnostic_only=true valid_for_default=false perf_database=false |",
        "",
        "## Pair Summary",
        "",
        "| Role | Scenario | max_num_batched_tokens | max scheduled | p50 scheduled | p99 scheduled | mean fill | max fill | output tok/s | output ratio |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['role']} | {row['scenario']} | {row['max_num_batched_tokens']} | "
            f"{row['max_scheduled_total_tokens']} | {row['p50_scheduled_total_tokens']} | "
            f"{row['p99_scheduled_total_tokens']} | {row['mean_budget_fill_ratio']} | "
            f"{row['max_budget_fill_ratio']} | {row['output_tok_s']} | "
            f"{row['output_ratio_vs_control']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "max_num_batched_tokens is a ceiling, not the actual per-iteration scheduler cost.",
            f"The holdout config sets max_num_batched_tokens={holdout['max_num_batched_tokens']}, "
            f"but max scheduled total tokens stays at {holdout['max_scheduled_total_tokens']}.",
            f"The steady decode distribution stays near p50={holdout['p50_scheduled_total_tokens']} "
            f"and p99={holdout['p99_scheduled_total_tokens']} scheduled tokens.",
            "This supports modeling scheduler cost from scheduled_total_tokens, phase mix, and decode batch tokens instead of applying a linear penalty to the configured budget ceiling.",
            "The evidence is still diagnostic-only and must not be wired into VLLMBackend.run_agg, default AIC, or PerfDatabase.",
            "",
            "## Guard",
            "",
            f"Control and holdout share topology_key={control['topology_key']} and shape_key={control['shape_key']}.",
            "The pair keeps diagnostic_only=true, valid_for_default=false, and perf_database=false.",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase234_scheduler_trace"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase242_actual_scheduled_token_mechanism.csv"),
    )
    parser.add_argument(
        "--doc-out",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase242_actual_scheduled_token_mechanism.md"),
    )
    args = parser.parse_args()

    rows = analyze_actual_scheduled_token_mechanism(args.input_root)
    write_actual_scheduled_token_csv(args.out, rows)
    write_actual_scheduled_token_doc(args.doc_out, rows)
    print(f"wrote_phase242_actual_scheduled_token_mechanism={args.out}")
    print(f"phase242_rows={len(rows)}")
    print(f"wrote_phase242_interpretation={args.doc_out}")
    print(f"mechanism_hypothesis={MECHANISM_HYPOTHESIS}")
    print(f"default_readiness={DEFAULT_READINESS}")


if __name__ == "__main__":
    main()
