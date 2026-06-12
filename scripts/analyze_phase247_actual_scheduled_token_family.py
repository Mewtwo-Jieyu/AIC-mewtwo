#!/usr/bin/env python3
"""Summarize the Phase247 4k2k actual scheduled token family."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple


SOURCE = "phase247_actual_scheduled_token_family"
TRACE_SOURCE = "phase234_vllm_scheduler_trace"
MECHANISM_HYPOTHESIS = "actual_scheduled_tokens_not_configured_budget"
DEFAULT_READINESS = "No-Go"
SHAPE_KEY = "isl4000_osl2000_batch128"
CONTROL_MAX_BT = 4000
HOLDOUT_MAX_BT = 65536
MAX_BUDGET_FILL_FOR_MECHANISM = 0.25


class ScenarioSpec(NamedTuple):
    scenario: str
    topology_key: str
    role: str
    max_bt: int
    tp: int
    dp: int
    ep: int


SCENARIOS = [
    ScenarioSpec("tp8ep8-4k2k-bt4000", "tp8_dp1_ep8", "control", CONTROL_MAX_BT, 8, 1, 8),
    ScenarioSpec("tp8ep8-4k2k-bt65536", "tp8_dp1_ep8", "holdout", HOLDOUT_MAX_BT, 8, 1, 8),
    ScenarioSpec("tp4dp2ep8-4k2k-bt4000", "tp4_dp2_ep8", "control", CONTROL_MAX_BT, 4, 2, 8),
    ScenarioSpec("tp4dp2ep8-4k2k-bt65536", "tp4_dp2_ep8", "holdout", HOLDOUT_MAX_BT, 4, 2, 8),
]

FIELDNAMES = [
    "source",
    "topology_key",
    "scenario",
    "role",
    "shape_key",
    "max_bt",
    "output_tok_s",
    "output_ratio_vs_control",
    "raw_trace_rows",
    "unique_iterations",
    "rank_min_p50_scheduled_total_tokens",
    "rank_min_p99_scheduled_total_tokens",
    "rank_min_max_scheduled_total_tokens",
    "rank_max_p50_scheduled_total_tokens",
    "rank_max_p99_scheduled_total_tokens",
    "rank_max_max_scheduled_total_tokens",
    "rank_sum_p50_scheduled_total_tokens",
    "rank_sum_p99_scheduled_total_tokens",
    "rank_sum_max_scheduled_total_tokens",
    "rank_sum_mean_fill_ratio",
    "rank_sum_max_fill_ratio",
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

PAYLOAD_SIGNATURE_FIELDS = (
    "phase",
    "scheduled_context_tokens",
    "scheduled_decode_tokens",
    "scheduled_total_tokens",
    "scheduled_context_reqs",
    "scheduled_decode_reqs",
    "scheduled_total_reqs",
    "forward_token_count",
)


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


def _required_int(row: dict[str, object], field: str, path: Path) -> int:
    value = row.get(field)
    if not isinstance(value, int):
        raise ValueError(f"{field} must be int in {path}: {value!r}")
    if value < 0:
        raise ValueError(f"{field} must be non-negative in {path}: {value!r}")
    return value


def _percentile(values: list[int], q: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    sorted_values = sorted(values)
    index = math.ceil(q * len(sorted_values)) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return float(sorted_values[index])


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _payload_signature(row: dict[str, object]) -> tuple[object, ...]:
    return tuple(row[field] for field in PAYLOAD_SIGNATURE_FIELDS)


def _validate_result(result: dict[str, object], spec: ScenarioSpec, result_path: Path) -> None:
    if result.get("scenario") != spec.scenario:
        raise ValueError(f"result scenario mismatch in {result_path}")
    if result.get("topology_key") != spec.topology_key:
        raise ValueError(f"result topology mismatch in {result_path}")
    if result.get("shape_key") != SHAPE_KEY:
        raise ValueError(f"result shape_key mismatch in {result_path}")
    shape = result.get("shape")
    if not isinstance(shape, dict):
        raise ValueError(f"missing shape in {result_path}")
    expected_shape = {
        "isl": 4000,
        "osl": 2000,
        "batch_size": 128,
        "max_num_batched_tokens": spec.max_bt,
        "max_num_seqs": 256,
    }
    for field, expected in expected_shape.items():
        if shape.get(field) != expected:
            raise ValueError(f"shape {field} mismatch in {result_path}: {shape.get(field)!r}")
    expected_parallelism = {"tp": spec.tp, "dp": spec.dp, "ep": spec.ep}
    if result.get("parallelism") != expected_parallelism:
        raise ValueError(f"parallelism mismatch in {result_path}: {result.get('parallelism')!r}")
    if result.get("diagnostic_only") is not True:
        raise ValueError(f"result diagnostic_only must be true in {result_path}")
    if result.get("valid_for_default") is not False:
        raise ValueError(f"result valid_for_default must be false in {result_path}")
    if result.get("perf_database") is not False:
        raise ValueError(f"result perf_database must be false in {result_path}")


def _validate_trace_row(row: dict[str, object], spec: ScenarioSpec, trace_path: Path) -> None:
    missing = REQUIRED_TRACE_FIELDS - row.keys()
    if missing:
        raise ValueError(f"trace row missing fields in {trace_path}: {sorted(missing)}")
    if row.get("source") != TRACE_SOURCE:
        raise ValueError(f"trace source mismatch in {trace_path}")
    if row.get("scenario") != spec.scenario:
        raise ValueError(f"trace scenario mismatch in {trace_path}")
    if row.get("topology_key") != spec.topology_key or row.get("shape_key") != SHAPE_KEY:
        raise ValueError(f"trace topology/shape mismatch in {trace_path}")
    if row.get("max_num_batched_tokens") != spec.max_bt:
        raise ValueError(f"trace max_num_batched_tokens mismatch in {trace_path}")
    if row.get("max_num_seqs") != 256:
        raise ValueError(f"trace max_num_seqs mismatch in {trace_path}")
    if row.get("tp") != spec.tp or row.get("dp") != spec.dp or row.get("ep") != spec.ep:
        raise ValueError(f"trace parallelism mismatch in {trace_path}")
    if row.get("diagnostic_only") is not True or row.get("valid_for_default") is not False or row.get("perf_database") is not False:
        raise ValueError(f"trace flag mismatch in {trace_path}")
    if row.get("phase") not in {"prefill", "mixed", "pure_decode"}:
        raise ValueError(f"trace phase mismatch in {trace_path}")

    context_tokens = _required_int(row, "scheduled_context_tokens", trace_path)
    decode_tokens = _required_int(row, "scheduled_decode_tokens", trace_path)
    total_tokens = _required_int(row, "scheduled_total_tokens", trace_path)
    context_reqs = _required_int(row, "scheduled_context_reqs", trace_path)
    decode_reqs = _required_int(row, "scheduled_decode_reqs", trace_path)
    total_reqs = _required_int(row, "scheduled_total_reqs", trace_path)
    _required_int(row, "forward_token_count", trace_path)
    _required_int(row, "iteration", trace_path)
    if total_tokens != context_tokens + decode_tokens:
        raise ValueError(f"trace token sum mismatch in {trace_path}")
    if total_reqs != context_reqs + decode_reqs:
        raise ValueError(f"trace request sum mismatch in {trace_path}")


def _distinct_iteration_payloads(
    rows: list[dict[str, object]],
    spec: ScenarioSpec,
    trace_path: Path,
) -> list[list[dict[str, object]]]:
    by_iteration: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_iteration[_required_int(row, "iteration", trace_path)].append(row)
    iteration_ids = sorted(by_iteration)
    if iteration_ids != list(range(iteration_ids[0], iteration_ids[-1] + 1)):
        raise ValueError(f"trace iterations must be contiguous in {trace_path}")

    grouped: list[list[dict[str, object]]] = []
    for iteration in iteration_ids:
        distinct: dict[tuple[object, ...], dict[str, object]] = {}
        for row in by_iteration[iteration]:
            distinct.setdefault(_payload_signature(row), row)
        payloads = list(distinct.values())
        if spec.topology_key == "tp8_dp1_ep8" and len(payloads) != 1:
            raise ValueError(f"tp8 duplicate worker trace payload mismatch in {trace_path}")
        if spec.topology_key == "tp4_dp2_ep8" and len(payloads) > spec.dp:
            raise ValueError(f"tp4dp2 trace has more distinct DP payloads than dp in {trace_path}")
        if not payloads:
            raise ValueError(f"empty iteration payload in {trace_path}")
        grouped.append(payloads)
    return grouped


def _analyze_case(input_root: Path, spec: ScenarioSpec) -> dict[str, object]:
    case_dir = input_root / spec.scenario
    if not case_dir.exists():
        raise ValueError(f"missing scenario: {spec.scenario}")
    bench_path = case_dir / "bench_result.json"
    records_path = case_dir / "bench_records.jsonl"
    result_path = case_dir / "phase234_result.json"
    trace_path = case_dir / "scheduler_trace.jsonl"
    for path in (bench_path, records_path, result_path, trace_path):
        if not path.exists():
            raise ValueError(f"missing artifact: {path}")

    bench = _read_json(bench_path)
    result = _read_json(result_path)
    _validate_result(result, spec, result_path)
    if bench.get("ok_requests") != 128 or bench.get("failed_requests") != 0:
        raise ValueError(f"bench requests must be 128/0 in {bench_path}")
    output_tok_s = bench.get("output_tok_s")
    if not isinstance(output_tok_s, (int, float)) or output_tok_s <= 0:
        raise ValueError(f"output_tok_s must be positive in {bench_path}")
    records = records_path.read_text(encoding="utf-8").splitlines()
    if len(records) != 128:
        raise ValueError(f"bench_records.jsonl must contain 128 rows: {records_path}")

    trace_rows = _read_trace(trace_path)
    for row in trace_rows:
        _validate_trace_row(row, spec, trace_path)
    payload_groups = _distinct_iteration_payloads(trace_rows, spec, trace_path)
    rank_min_tokens: list[int] = []
    rank_max_tokens: list[int] = []
    rank_sum_tokens: list[int] = []
    for payloads in payload_groups:
        scheduled_tokens = [
            _required_int(payload, "scheduled_total_tokens", trace_path)
            for payload in payloads
        ]
        rank_min_tokens.append(min(scheduled_tokens))
        rank_max_tokens.append(max(scheduled_tokens))
        rank_sum_tokens.append(sum(scheduled_tokens))
    if _percentile(rank_sum_tokens, 0.99) <= 0:
        raise ValueError(f"rank_sum_p99_scheduled_total_tokens must be positive: {spec.scenario}")

    return {
        "spec": spec,
        "output_tok_s": float(output_tok_s),
        "raw_trace_rows": len(trace_rows),
        "unique_iterations": len(payload_groups),
        "rank_min_p50": _percentile(rank_min_tokens, 0.50),
        "rank_min_p99": _percentile(rank_min_tokens, 0.99),
        "rank_min_max": max(rank_min_tokens),
        "rank_max_p50": _percentile(rank_max_tokens, 0.50),
        "rank_max_p99": _percentile(rank_max_tokens, 0.99),
        "rank_max_max": max(rank_max_tokens),
        "rank_sum_p50": _percentile(rank_sum_tokens, 0.50),
        "rank_sum_p99": _percentile(rank_sum_tokens, 0.99),
        "rank_sum_max": max(rank_sum_tokens),
        "rank_sum_mean_fill": sum(token / spec.max_bt for token in rank_sum_tokens) / len(rank_sum_tokens),
        "rank_sum_max_fill": max(rank_sum_tokens) / spec.max_bt,
    }


def analyze_actual_scheduled_token_family(input_root: Path | str) -> list[dict[str, str]]:
    root = Path(input_root)
    cases = [_analyze_case(root, spec) for spec in SCENARIOS]
    by_topology_role: dict[tuple[str, str], dict[str, object]] = {}
    for case in cases:
        spec = case["spec"]
        assert isinstance(spec, ScenarioSpec)
        key = (spec.topology_key, spec.role)
        if key in by_topology_role:
            raise ValueError(f"duplicate topology/role: {key}")
        by_topology_role[key] = case

    expected_keys = {(spec.topology_key, spec.role) for spec in SCENARIOS}
    if set(by_topology_role) != expected_keys:
        raise ValueError(f"pair completeness mismatch: {sorted(by_topology_role)}")

    rows: list[dict[str, str]] = []
    for topology_key in ("tp8_dp1_ep8", "tp4_dp2_ep8"):
        control = by_topology_role[(topology_key, "control")]
        holdout = by_topology_role[(topology_key, "holdout")]
        control_spec = control["spec"]
        holdout_spec = holdout["spec"]
        assert isinstance(control_spec, ScenarioSpec)
        assert isinstance(holdout_spec, ScenarioSpec)
        if control_spec.max_bt != CONTROL_MAX_BT or holdout_spec.max_bt != HOLDOUT_MAX_BT:
            raise ValueError(f"control/holdout budget mismatch: {topology_key}")
        if float(holdout["rank_sum_max_fill"]) >= MAX_BUDGET_FILL_FOR_MECHANISM:
            raise ValueError("holdout rank-sum max scheduled tokens must stay far below configured budget")
        control_output = float(control["output_tok_s"])
        for case in (control, holdout):
            spec = case["spec"]
            assert isinstance(spec, ScenarioSpec)
            output_ratio = float(case["output_tok_s"]) / control_output
            rows.append(
                {
                    "source": SOURCE,
                    "topology_key": spec.topology_key,
                    "scenario": spec.scenario,
                    "role": spec.role,
                    "shape_key": SHAPE_KEY,
                    "max_bt": str(spec.max_bt),
                    "output_tok_s": _format_float(float(case["output_tok_s"])),
                    "output_ratio_vs_control": _format_float(output_ratio),
                    "raw_trace_rows": str(case["raw_trace_rows"]),
                    "unique_iterations": str(case["unique_iterations"]),
                    "rank_min_p50_scheduled_total_tokens": _format_float(float(case["rank_min_p50"])),
                    "rank_min_p99_scheduled_total_tokens": _format_float(float(case["rank_min_p99"])),
                    "rank_min_max_scheduled_total_tokens": str(case["rank_min_max"]),
                    "rank_max_p50_scheduled_total_tokens": _format_float(float(case["rank_max_p50"])),
                    "rank_max_p99_scheduled_total_tokens": _format_float(float(case["rank_max_p99"])),
                    "rank_max_max_scheduled_total_tokens": str(case["rank_max_max"]),
                    "rank_sum_p50_scheduled_total_tokens": _format_float(float(case["rank_sum_p50"])),
                    "rank_sum_p99_scheduled_total_tokens": _format_float(float(case["rank_sum_p99"])),
                    "rank_sum_max_scheduled_total_tokens": str(case["rank_sum_max"]),
                    "rank_sum_mean_fill_ratio": _format_float(float(case["rank_sum_mean_fill"])),
                    "rank_sum_max_fill_ratio": _format_float(float(case["rank_sum_max_fill"])),
                    "mechanism_hypothesis": MECHANISM_HYPOTHESIS,
                    "default_readiness": DEFAULT_READINESS,
                    "diagnostic_only": "true",
                    "valid_for_default": "false",
                    "perf_database": "false",
                }
            )
    if len(rows) != 4:
        raise ValueError(f"phase247 family must contain 4 rows: got {len(rows)}")
    return rows


def write_actual_scheduled_token_family_csv(path: Path | str, rows: list[dict[str, str]]) -> None:
    if len(rows) != 4:
        raise ValueError(f"phase247 family csv must contain 4 rows: got {len(rows)}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_actual_scheduled_token_family_doc(path: Path | str, rows: list[dict[str, str]]) -> None:
    if len(rows) != 4:
        raise ValueError(f"phase247 family doc expects 4 rows: got {len(rows)}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase247: 4k2k Actual Scheduled Token Family",
        "",
        "## Decision",
        "",
        "| Item | Result |",
        "|---|---|",
        f"| Mechanism hypothesis | {MECHANISM_HYPOTHESIS} |",
        "| Default AIC | No-Go |",
        "| PerfDatabase | No-Go |",
        "| GPU benchmark | No-Go in Phase247 |",
        "| Boundary flags | diagnostic_only=true valid_for_default=false perf_database=false |",
        "",
        "## Family Summary",
        "",
        "| Topology | Role | Scenario | max_bt | output tok/s | output ratio | rank min p50/p99/max | rank max p50/p99/max | rank sum p50/p99/max | rank sum mean fill | rank sum max fill |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['topology_key']} | {row['role']} | {row['scenario']} | "
            f"{row['max_bt']} | {row['output_tok_s']} | {row['output_ratio_vs_control']} | "
            f"{row['rank_min_p50_scheduled_total_tokens']}/{row['rank_min_p99_scheduled_total_tokens']}/{row['rank_min_max_scheduled_total_tokens']} | "
            f"{row['rank_max_p50_scheduled_total_tokens']}/{row['rank_max_p99_scheduled_total_tokens']}/{row['rank_max_max_scheduled_total_tokens']} | "
            f"{row['rank_sum_p50_scheduled_total_tokens']}/{row['rank_sum_p99_scheduled_total_tokens']}/{row['rank_sum_max_scheduled_total_tokens']} | "
            f"{row['rank_sum_mean_fill_ratio']} | {row['rank_sum_max_fill_ratio']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Both 4k2k topology pairs support actual_scheduled_tokens_not_configured_budget.",
            "The high-budget rows configure max_bt=65536, but actual scheduled rank-sum tokens stay far below that ceiling.",
            "For tp8_dp1_ep8, worker payloads are expected to be identical within an iteration and are deduplicated by iteration.",
            "For tp4_dp2_ep8, DP=2 rows are aggregated as rank min/max/sum because different DP payloads can appear in the same iteration.",
            "This keeps the mechanism diagnostic-only: model scheduler cost from actual scheduled tokens, phase mix, and decode batch tokens, not from a linear penalty on max_num_batched_tokens.",
            "Do not wire this into VLLMBackend.run_agg, default AIC, or PerfDatabase.",
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
        default=Path("docs/iter_gap_investigation/phase247_actual_scheduled_token_family.csv"),
    )
    parser.add_argument(
        "--doc-out",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase247_actual_scheduled_token_family.md"),
    )
    args = parser.parse_args()

    rows = analyze_actual_scheduled_token_family(args.input_root)
    write_actual_scheduled_token_family_csv(args.out, rows)
    write_actual_scheduled_token_family_doc(args.doc_out, rows)
    print(f"wrote_phase247_actual_scheduled_token_family={args.out}")
    print(f"phase247_rows={len(rows)}")
    print(f"wrote_phase247_interpretation={args.doc_out}")
    print(f"mechanism_hypothesis={MECHANISM_HYPOTHESIS}")
    print(f"default_readiness={DEFAULT_READINESS}")


if __name__ == "__main__":
    main()
