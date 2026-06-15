#!/usr/bin/env python3
"""Summarize Phase264 phase mix and decode batch diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple


SOURCE = "phase264_phase_mix_decode_batch"
TRACE_SOURCE = "phase234_vllm_scheduler_trace"
MECHANISM_HYPOTHESIS = "phase_mix_decode_batch_diagnostic"
DEFAULT_READINESS = "No-Go"
HOLDOUT_MAX_BT = 65536
MAX_AGGREGATE_FILL_FOR_MECHANISM = 0.50
VALID_PHASES = {"prefill", "mixed", "pure_decode"}


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
    ScenarioSpec("tp4dp2ep8-12k2k-bt12000", "tp4_dp2_ep8", "control", "isl12000_osl2000_batch128", 12000, 12000, HOLDOUT_MAX_BT, 12000, 4, 2, 8),
    ScenarioSpec("tp4dp2ep8-12k2k-bt65536", "tp4_dp2_ep8", "holdout", "isl12000_osl2000_batch128", 12000, 12000, HOLDOUT_MAX_BT, HOLDOUT_MAX_BT, 4, 2, 8),
]

FIELDNAMES = [
    "source",
    "topology_key",
    "shape_key",
    "scenario",
    "role",
    "max_bt",
    "dp",
    "configured_budget_aggregate",
    "prefill_iterations",
    "mixed_iterations",
    "pure_decode_iterations",
    "total_iterations",
    "mixed_ratio",
    "pure_decode_rank_sum_p50_decode_tokens",
    "pure_decode_rank_sum_p95_decode_tokens",
    "pure_decode_rank_sum_p99_decode_tokens",
    "pure_decode_rank_sum_max_decode_tokens",
    "mixed_rank_sum_p50_total_tokens",
    "mixed_rank_sum_p95_total_tokens",
    "mixed_rank_sum_p99_total_tokens",
    "mixed_rank_sum_max_total_tokens",
    "rank_sum_p99_total_tokens",
    "rank_sum_max_total_tokens",
    "rank_sum_p99_fill_ratio",
    "rank_sum_max_fill_ratio",
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
    if row.get("phase") not in VALID_PHASES:
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


def _aggregate_iteration_rows(
    rows: list[dict[str, object]],
    spec: ScenarioSpec,
    trace_path: Path,
) -> list[dict[str, int | str]]:
    by_iteration: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_iteration[_required_int(row, "iteration", trace_path)].append(row)
    iteration_ids = sorted(by_iteration)
    if iteration_ids != list(range(iteration_ids[0], iteration_ids[-1] + 1)):
        raise ValueError(f"trace iterations must be contiguous in {trace_path}")

    aggregated: list[dict[str, int | str]] = []
    for iteration in iteration_ids:
        iteration_rows = by_iteration[iteration]
        signatures = {_payload_signature(row) for row in iteration_rows}
        if len(signatures) > spec.dp:
            raise ValueError(f"trace has more distinct DP payloads than dp in {trace_path}")
        raw_context = sum(_required_int(row, "scheduled_context_tokens", trace_path) for row in iteration_rows)
        raw_decode = sum(_required_int(row, "scheduled_decode_tokens", trace_path) for row in iteration_rows)
        raw_total = sum(_required_int(row, "scheduled_total_tokens", trace_path) for row in iteration_rows)
        if raw_context % spec.tp != 0 or raw_decode % spec.tp != 0 or raw_total % spec.tp != 0:
            raise ValueError(f"trace scheduled token raw sum must be divisible by tp in {trace_path}")
        context_tokens = raw_context // spec.tp
        decode_tokens = raw_decode // spec.tp
        total_tokens = raw_total // spec.tp
        if total_tokens != context_tokens + decode_tokens:
            raise ValueError(f"aggregate token sum mismatch in {trace_path}")
        if context_tokens > 0 and decode_tokens > 0:
            phase = "mixed"
        elif context_tokens > 0:
            phase = "prefill"
        elif decode_tokens > 0:
            phase = "pure_decode"
        else:
            raise ValueError(f"aggregate scheduled tokens must be positive in {trace_path}")
        aggregated.append(
            {
                "iteration": iteration,
                "phase": phase,
                "rank_sum_context_tokens": context_tokens,
                "rank_sum_decode_tokens": decode_tokens,
                "rank_sum_total_tokens": total_tokens,
            }
        )
    return aggregated


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
    aggregated = _aggregate_iteration_rows(trace_rows, spec, trace_path)
    phases = [str(row["phase"]) for row in aggregated]
    if "prefill" not in phases or "pure_decode" not in phases:
        raise ValueError(f"trace must contain prefill and pure_decode iterations: {spec.scenario}")

    rank_sum_totals = [int(row["rank_sum_total_tokens"]) for row in aggregated]
    pure_decode_decode_tokens = [
        int(row["rank_sum_decode_tokens"])
        for row in aggregated
        if row["phase"] == "pure_decode"
    ]
    mixed_total_tokens = [
        int(row["rank_sum_total_tokens"])
        for row in aggregated
        if row["phase"] == "mixed"
    ]
    if not pure_decode_decode_tokens:
        raise ValueError(f"missing pure_decode decode batch: {spec.scenario}")
    if not mixed_total_tokens:
        raise ValueError(f"missing mixed iteration: {spec.scenario}")

    configured_budget_aggregate = spec.max_bt * spec.dp
    rank_sum_max_fill = max(rank_sum_totals) / configured_budget_aggregate
    if spec.role == "holdout" and rank_sum_max_fill >= MAX_AGGREGATE_FILL_FOR_MECHANISM:
        raise ValueError("holdout rank-sum max scheduled tokens must stay below aggregate configured budget")
    return {
        "spec": spec,
        "output_tok_s": float(output_tok_s),
        "configured_budget_aggregate": configured_budget_aggregate,
        "prefill_iterations": phases.count("prefill"),
        "mixed_iterations": phases.count("mixed"),
        "pure_decode_iterations": phases.count("pure_decode"),
        "total_iterations": len(aggregated),
        "pure_decode_p50": _percentile(pure_decode_decode_tokens, 0.50),
        "pure_decode_p95": _percentile(pure_decode_decode_tokens, 0.95),
        "pure_decode_p99": _percentile(pure_decode_decode_tokens, 0.99),
        "pure_decode_max": max(pure_decode_decode_tokens),
        "mixed_p50": _percentile(mixed_total_tokens, 0.50),
        "mixed_p95": _percentile(mixed_total_tokens, 0.95),
        "mixed_p99": _percentile(mixed_total_tokens, 0.99),
        "mixed_max": max(mixed_total_tokens),
        "rank_sum_p99": _percentile(rank_sum_totals, 0.99),
        "rank_sum_max": max(rank_sum_totals),
        "rank_sum_p99_fill": _percentile(rank_sum_totals, 0.99) / configured_budget_aggregate,
        "rank_sum_max_fill": rank_sum_max_fill,
    }


def analyze_phase_mix_decode_batch(input_root: Path | str) -> list[dict[str, str]]:
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
        control = by_pair_role[(spec.topology_key, spec.shape_key, "control")]
        output_ratio = float(case["output_tok_s"]) / float(control["output_tok_s"])
        total_iterations = int(case["total_iterations"])
        rows.append(
            {
                "source": SOURCE,
                "topology_key": spec.topology_key,
                "shape_key": spec.shape_key,
                "scenario": spec.scenario,
                "role": spec.role,
                "max_bt": str(spec.max_bt),
                "dp": str(spec.dp),
                "configured_budget_aggregate": str(case["configured_budget_aggregate"]),
                "prefill_iterations": str(case["prefill_iterations"]),
                "mixed_iterations": str(case["mixed_iterations"]),
                "pure_decode_iterations": str(case["pure_decode_iterations"]),
                "total_iterations": str(total_iterations),
                "mixed_ratio": _format_float(int(case["mixed_iterations"]) / total_iterations),
                "pure_decode_rank_sum_p50_decode_tokens": _format_float(float(case["pure_decode_p50"])),
                "pure_decode_rank_sum_p95_decode_tokens": _format_float(float(case["pure_decode_p95"])),
                "pure_decode_rank_sum_p99_decode_tokens": _format_float(float(case["pure_decode_p99"])),
                "pure_decode_rank_sum_max_decode_tokens": str(case["pure_decode_max"]),
                "mixed_rank_sum_p50_total_tokens": _format_float(float(case["mixed_p50"])),
                "mixed_rank_sum_p95_total_tokens": _format_float(float(case["mixed_p95"])),
                "mixed_rank_sum_p99_total_tokens": _format_float(float(case["mixed_p99"])),
                "mixed_rank_sum_max_total_tokens": str(case["mixed_max"]),
                "rank_sum_p99_total_tokens": _format_float(float(case["rank_sum_p99"])),
                "rank_sum_max_total_tokens": str(case["rank_sum_max"]),
                "rank_sum_p99_fill_ratio": _format_float(float(case["rank_sum_p99_fill"])),
                "rank_sum_max_fill_ratio": _format_float(float(case["rank_sum_max_fill"])),
                "output_tok_s": _format_float(float(case["output_tok_s"])),
                "output_ratio_vs_control": _format_float(output_ratio),
                "mechanism_hypothesis": MECHANISM_HYPOTHESIS,
                "default_readiness": DEFAULT_READINESS,
                "diagnostic_only": "true",
                "valid_for_default": "false",
                "perf_database": "false",
            }
        )
    if len(rows) != 8:
        raise ValueError(f"phase264 family must contain 8 rows: got {len(rows)}")
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
    for field in (
        "max_bt",
        "dp",
        "configured_budget_aggregate",
        "prefill_iterations",
        "pure_decode_iterations",
        "total_iterations",
        "rank_sum_max_total_tokens",
    ):
        value = int(row[field])
        if value <= 0:
            raise ValueError(f"{field} must be positive")
    if int(row["mixed_iterations"]) <= 0:
        raise ValueError("mixed_iterations must be positive")
    for field in (
        "mixed_ratio",
        "pure_decode_rank_sum_p99_decode_tokens",
        "mixed_rank_sum_p99_total_tokens",
        "rank_sum_p99_fill_ratio",
        "rank_sum_max_fill_ratio",
        "output_tok_s",
        "output_ratio_vs_control",
    ):
        value = float(row[field])
        if value <= 0:
            raise ValueError(f"{field} must be positive")


def read_phase_mix_decode_batch_csv(path: Path | str) -> list[dict[str, str]]:
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 8:
        raise ValueError(f"phase264 csv must contain 8 rows: got {len(rows)}")
    scenarios = [row.get("scenario") for row in rows]
    expected = [spec.scenario for spec in SCENARIOS]
    if scenarios != expected:
        raise ValueError(f"scenario set/order mismatch: {scenarios}")
    for row in rows:
        _validate_output_row(row)
    return rows


def write_phase_mix_decode_batch_csv(path: Path | str, rows: list[dict[str, str]]) -> None:
    if len(rows) != 8:
        raise ValueError(f"phase264 csv must contain 8 rows: got {len(rows)}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    read_phase_mix_decode_batch_csv(out)


def write_phase_mix_decode_batch_doc(path: Path | str, rows: list[dict[str, str]]) -> None:
    if len(rows) != 8:
        raise ValueError(f"phase264 doc expects 8 rows: got {len(rows)}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase264: Phase Mix / Decode Batch Diagnostic",
        "",
        "## Decision",
        "",
        "| Item | Result |",
        "|---|---|",
        f"| Mechanism hypothesis | {MECHANISM_HYPOTHESIS} |",
        "| Default AIC | No-Go |",
        "| PerfDatabase | No-Go |",
        "| GPU benchmark | No-Go in Phase264 |",
        "| Boundary flags | diagnostic_only=true valid_for_default=false perf_database=false |",
        "",
        "## Family Summary",
        "",
        "| Topology / shape | Role | Scenario | max_bt | phase mix prefill/mixed/decode | pure decode p99/max | mixed p99/max | output ratio | rank sum p99/max fill |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        shape_label = "4k2k" if row["shape_key"] == "isl4000_osl2000_batch128" else "12k2k"
        lines.append(
            f"| {row['topology_key']} / {shape_label} | {row['role']} | {row['scenario']} | "
            f"{row['max_bt']} | {row['prefill_iterations']}/{row['mixed_iterations']}/{row['pure_decode_iterations']} | "
            f"{row['pure_decode_rank_sum_p99_decode_tokens']}/{row['pure_decode_rank_sum_max_decode_tokens']} | "
            f"{row['mixed_rank_sum_p99_total_tokens']}/{row['mixed_rank_sum_max_total_tokens']} | "
            f"{row['output_ratio_vs_control']} | {row['rank_sum_p99_fill_ratio']}/{row['rank_sum_max_fill_ratio']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Phase264 keeps the Phase258 actual scheduled token boundary and adds phase mix and decode batch readouts.",
            "The analyzer classifies each iteration after rank-sum aggregation, so tp4_dp2_ep8 can have different DP payload phases without being reduced to the first raw worker row.",
            "The high-budget rows still keep rank-sum p99/max fill far below the aggregate configured budget, so max_num_batched_tokens remains a ceiling rather than a direct linear scheduler cost.",
            "The next diagnostic question is whether throughput spread is better explained by phase mix and decode batch shape than by the configured budget ceiling.",
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
        default=Path("docs/iter_gap_investigation/phase264_phase_mix_decode_batch.csv"),
    )
    parser.add_argument(
        "--doc-out",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase264_phase_mix_decode_batch.md"),
    )
    args = parser.parse_args()

    rows = analyze_phase_mix_decode_batch(args.input_root)
    write_phase_mix_decode_batch_csv(args.out, rows)
    write_phase_mix_decode_batch_doc(args.doc_out, rows)
    print(f"wrote_phase264_phase_mix_decode_batch={args.out}")
    print(f"phase264_rows={len(rows)}")
    print(f"wrote_phase264_interpretation={args.doc_out}")
    print(f"mechanism_hypothesis={MECHANISM_HYPOTHESIS}")
    print(f"default_readiness={DEFAULT_READINESS}")


if __name__ == "__main__":
    main()
