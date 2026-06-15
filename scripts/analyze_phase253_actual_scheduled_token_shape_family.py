#!/usr/bin/env python3
"""Summarize the Phase253 actual scheduled token shape family."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple


SOURCE = "phase253_actual_scheduled_token_shape_family"
TRACE_SOURCE = "phase234_vllm_scheduler_trace"
MECHANISM_HYPOTHESIS = "actual_scheduled_tokens_not_configured_budget"
DEFAULT_READINESS = "No-Go"
HOLDOUT_MAX_BT = 65536
MAX_BUDGET_FILL_FOR_MECHANISM = 0.25


class ScenarioSpec(NamedTuple):
    scenario: str
    topology_key: str
    role: str
    shape_key: str
    isl: int
    control_max_bt: int
    holdout_max_bt: int
    max_bt: int
    tp: int
    dp: int
    ep: int


SCENARIOS = [
    ScenarioSpec("tp8ep8-4k2k-bt4000", "tp8_dp1_ep8", "control", "isl4000_osl2000_batch128", 4000, 4000, HOLDOUT_MAX_BT, 4000, 8, 1, 8),
    ScenarioSpec("tp8ep8-4k2k-bt65536", "tp8_dp1_ep8", "holdout", "isl4000_osl2000_batch128", 4000, 4000, HOLDOUT_MAX_BT, HOLDOUT_MAX_BT, 8, 1, 8),
    ScenarioSpec("tp4dp2ep8-4k2k-bt4000", "tp4_dp2_ep8", "control", "isl4000_osl2000_batch128", 4000, 4000, HOLDOUT_MAX_BT, 4000, 4, 2, 8),
    ScenarioSpec("tp4dp2ep8-4k2k-bt65536", "tp4_dp2_ep8", "holdout", "isl4000_osl2000_batch128", 4000, 4000, HOLDOUT_MAX_BT, HOLDOUT_MAX_BT, 4, 2, 8),
    ScenarioSpec("tp8ep8-12k2k-bt12000", "tp8_dp1_ep8", "control", "isl12000_osl2000_batch128", 12000, 12000, HOLDOUT_MAX_BT, 12000, 8, 1, 8),
    ScenarioSpec("tp8ep8-12k2k-bt65536", "tp8_dp1_ep8", "holdout", "isl12000_osl2000_batch128", 12000, 12000, HOLDOUT_MAX_BT, HOLDOUT_MAX_BT, 8, 1, 8),
]

FIELDNAMES = [
    "source",
    "topology_key",
    "scenario",
    "role",
    "shape_key",
    "control_max_bt",
    "holdout_max_bt",
    "max_bt",
    "output_tok_s",
    "output_ratio_vs_control",
    "raw_trace_rows",
    "unique_iterations",
    "rank_min_p50_scheduled_total_tokens",
    "rank_min_p95_scheduled_total_tokens",
    "rank_min_p99_scheduled_total_tokens",
    "rank_min_max_scheduled_total_tokens",
    "rank_max_p50_scheduled_total_tokens",
    "rank_max_p95_scheduled_total_tokens",
    "rank_max_p99_scheduled_total_tokens",
    "rank_max_max_scheduled_total_tokens",
    "rank_sum_p50_scheduled_total_tokens",
    "rank_sum_p95_scheduled_total_tokens",
    "rank_sum_p99_scheduled_total_tokens",
    "rank_sum_max_scheduled_total_tokens",
    "rank_sum_mean_fill_ratio",
    "rank_sum_p99_fill_ratio",
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
    if result.get("shape_key") != spec.shape_key:
        raise ValueError(f"result shape_key mismatch in {result_path}")
    shape = result.get("shape")
    if not isinstance(shape, dict):
        raise ValueError(f"missing shape in {result_path}")
    expected_shape = {
        "isl": spec.isl,
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
    if row.get("topology_key") != spec.topology_key or row.get("shape_key") != spec.shape_key:
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
        "rank_min_p95": _percentile(rank_min_tokens, 0.95),
        "rank_min_p99": _percentile(rank_min_tokens, 0.99),
        "rank_min_max": max(rank_min_tokens),
        "rank_max_p50": _percentile(rank_max_tokens, 0.50),
        "rank_max_p95": _percentile(rank_max_tokens, 0.95),
        "rank_max_p99": _percentile(rank_max_tokens, 0.99),
        "rank_max_max": max(rank_max_tokens),
        "rank_sum_p50": _percentile(rank_sum_tokens, 0.50),
        "rank_sum_p95": _percentile(rank_sum_tokens, 0.95),
        "rank_sum_p99": _percentile(rank_sum_tokens, 0.99),
        "rank_sum_max": max(rank_sum_tokens),
        "rank_sum_mean_fill": sum(token / spec.max_bt for token in rank_sum_tokens) / len(rank_sum_tokens),
        "rank_sum_p99_fill": _percentile(rank_sum_tokens, 0.99) / spec.max_bt,
        "rank_sum_max_fill": max(rank_sum_tokens) / spec.max_bt,
    }


def analyze_actual_scheduled_token_shape_family(input_root: Path | str) -> list[dict[str, str]]:
    root = Path(input_root)
    cases = [_analyze_case(root, spec) for spec in SCENARIOS]
    by_pair_role: dict[tuple[str, str, str], dict[str, object]] = {}
    for case in cases:
        spec = case["spec"]
        assert isinstance(spec, ScenarioSpec)
        key = (spec.topology_key, spec.shape_key, spec.role)
        if key in by_pair_role:
            raise ValueError(f"duplicate topology/shape/role: {key}")
        by_pair_role[key] = case

    expected_keys = {(spec.topology_key, spec.shape_key, spec.role) for spec in SCENARIOS}
    if set(by_pair_role) != expected_keys:
        raise ValueError(f"pair completeness mismatch: {sorted(by_pair_role)}")

    rows: list[dict[str, str]] = []
    for spec in SCENARIOS:
        case = by_pair_role[(spec.topology_key, spec.shape_key, spec.role)]
        if spec.role == "holdout" and spec.max_bt != HOLDOUT_MAX_BT:
            raise ValueError(f"holdout budget mismatch: {spec.scenario}")
        if spec.role == "holdout" and float(case["rank_sum_max_fill"]) >= MAX_BUDGET_FILL_FOR_MECHANISM:
            raise ValueError("holdout rank-sum max scheduled tokens must stay far below configured budget")
        control = by_pair_role[(spec.topology_key, spec.shape_key, "control")]
        control_output = float(control["output_tok_s"])
        output_ratio = float(case["output_tok_s"]) / control_output
        rows.append(
            {
                "source": SOURCE,
                "topology_key": spec.topology_key,
                "scenario": spec.scenario,
                "role": spec.role,
                "shape_key": spec.shape_key,
                "control_max_bt": str(spec.control_max_bt),
                "holdout_max_bt": str(spec.holdout_max_bt),
                "max_bt": str(spec.max_bt),
                "output_tok_s": _format_float(float(case["output_tok_s"])),
                "output_ratio_vs_control": _format_float(output_ratio),
                "raw_trace_rows": str(case["raw_trace_rows"]),
                "unique_iterations": str(case["unique_iterations"]),
                "rank_min_p50_scheduled_total_tokens": _format_float(float(case["rank_min_p50"])),
                "rank_min_p95_scheduled_total_tokens": _format_float(float(case["rank_min_p95"])),
                "rank_min_p99_scheduled_total_tokens": _format_float(float(case["rank_min_p99"])),
                "rank_min_max_scheduled_total_tokens": str(case["rank_min_max"]),
                "rank_max_p50_scheduled_total_tokens": _format_float(float(case["rank_max_p50"])),
                "rank_max_p95_scheduled_total_tokens": _format_float(float(case["rank_max_p95"])),
                "rank_max_p99_scheduled_total_tokens": _format_float(float(case["rank_max_p99"])),
                "rank_max_max_scheduled_total_tokens": str(case["rank_max_max"]),
                "rank_sum_p50_scheduled_total_tokens": _format_float(float(case["rank_sum_p50"])),
                "rank_sum_p95_scheduled_total_tokens": _format_float(float(case["rank_sum_p95"])),
                "rank_sum_p99_scheduled_total_tokens": _format_float(float(case["rank_sum_p99"])),
                "rank_sum_max_scheduled_total_tokens": str(case["rank_sum_max"]),
                "rank_sum_mean_fill_ratio": _format_float(float(case["rank_sum_mean_fill"])),
                "rank_sum_p99_fill_ratio": _format_float(float(case["rank_sum_p99_fill"])),
                "rank_sum_max_fill_ratio": _format_float(float(case["rank_sum_max_fill"])),
                "mechanism_hypothesis": MECHANISM_HYPOTHESIS,
                "default_readiness": DEFAULT_READINESS,
                "diagnostic_only": "true",
                "valid_for_default": "false",
                "perf_database": "false",
            }
        )
    if len(rows) != 6:
        raise ValueError(f"phase253 family must contain 6 rows: got {len(rows)}")
    return rows


def _validate_output_row(row: dict[str, str]) -> None:
    if row.get("source") != SOURCE:
        raise ValueError("source mismatch")
    if row.get("mechanism_hypothesis") != MECHANISM_HYPOTHESIS:
        raise ValueError("mechanism_hypothesis mismatch")
    if row.get("default_readiness") != DEFAULT_READINESS:
        raise ValueError("default_readiness mismatch")
    if row.get("diagnostic_only") != "true" or row.get("valid_for_default") != "false" or row.get("perf_database") != "false":
        raise ValueError("flag mismatch")
    for field in ("max_bt", "control_max_bt", "holdout_max_bt", "rank_sum_max_scheduled_total_tokens"):
        value = int(row[field])
        if value <= 0:
            raise ValueError(f"{field} must be positive")
    for field in ("output_tok_s", "output_ratio_vs_control", "rank_sum_p99_fill_ratio", "rank_sum_max_fill_ratio"):
        value = float(row[field])
        if value <= 0:
            raise ValueError(f"{field} must be positive")


def read_actual_scheduled_token_shape_family_csv(path: Path | str) -> list[dict[str, str]]:
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 6:
        raise ValueError(f"phase253 family csv must contain 6 rows: got {len(rows)}")
    scenarios = [row.get("scenario") for row in rows]
    expected = [spec.scenario for spec in SCENARIOS]
    if scenarios != expected:
        raise ValueError(f"scenario set/order mismatch: {scenarios}")
    for row in rows:
        _validate_output_row(row)
    return rows


def write_actual_scheduled_token_shape_family_csv(path: Path | str, rows: list[dict[str, str]]) -> None:
    if len(rows) != 6:
        raise ValueError(f"phase253 family csv must contain 6 rows: got {len(rows)}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    read_actual_scheduled_token_shape_family_csv(out)


def write_actual_scheduled_token_shape_family_doc(path: Path | str, rows: list[dict[str, str]]) -> None:
    if len(rows) != 6:
        raise ValueError(f"phase253 family doc expects 6 rows: got {len(rows)}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase253: Actual Scheduled Token Shape Family",
        "",
        "## Decision",
        "",
        "| Item | Result |",
        "|---|---|",
        f"| Mechanism hypothesis | {MECHANISM_HYPOTHESIS} |",
        "| Default AIC | No-Go |",
        "| PerfDatabase | No-Go |",
        "| GPU benchmark | No-Go in Phase253 |",
        "| Boundary flags | diagnostic_only=true valid_for_default=false perf_database=false |",
        "",
        "## Family Summary",
        "",
        "| Topology / shape | Role | Scenario | max_bt | output tok/s | output ratio | rank sum p50/p95/p99/max | rank sum mean/p99/max fill |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        shape_label = "4k2k" if row["shape_key"] == "isl4000_osl2000_batch128" else "12k2k"
        lines.append(
            f"| {row['topology_key']} / {shape_label} | {row['role']} | {row['scenario']} | "
            f"{row['max_bt']} | {row['output_tok_s']} | {row['output_ratio_vs_control']} | "
            f"{row['rank_sum_p50_scheduled_total_tokens']}/{row['rank_sum_p95_scheduled_total_tokens']}/{row['rank_sum_p99_scheduled_total_tokens']}/{row['rank_sum_max_scheduled_total_tokens']} | "
            f"{row['rank_sum_mean_fill_ratio']}/{row['rank_sum_p99_fill_ratio']}/{row['rank_sum_max_fill_ratio']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The Phase253 family adds the tp8_dp1_ep8 / 12k2k pair to the Phase247 4k2k trace evidence.",
            "The high-budget rows configure max_bt=65536, but actual scheduled rank-sum tokens remain far below that ceiling.",
            "For tp8_dp1_ep8, worker payloads are deduplicated by iteration because TP workers should report the same scheduler payload.",
            "For tp4_dp2_ep8, DP=2 rows are aggregated as rank min/max/sum because different DP payloads can appear in the same iteration.",
            "This supports modeling scheduler cost from actual scheduled tokens, phase mix, and decode batch tokens instead of a linear cost on max_num_batched_tokens.",
            "This remains diagnostic-only evidence. Do not wire it into VLLMBackend.run_agg, default AIC, or PerfDatabase.",
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
        default=Path("docs/iter_gap_investigation/phase253_actual_scheduled_token_shape_family.csv"),
    )
    parser.add_argument(
        "--doc-out",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase253_actual_scheduled_token_shape_family.md"),
    )
    args = parser.parse_args()

    rows = analyze_actual_scheduled_token_shape_family(args.input_root)
    write_actual_scheduled_token_shape_family_csv(args.out, rows)
    write_actual_scheduled_token_shape_family_doc(args.doc_out, rows)
    print(f"wrote_phase253_actual_scheduled_token_shape_family={args.out}")
    print(f"phase253_rows={len(rows)}")
    print(f"wrote_phase253_interpretation={args.doc_out}")
    print(f"mechanism_hypothesis={MECHANISM_HYPOTHESIS}")
    print(f"default_readiness={DEFAULT_READINESS}")


if __name__ == "__main__":
    main()
