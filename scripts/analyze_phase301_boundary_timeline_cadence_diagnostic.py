#!/usr/bin/env python3
"""Summarize Phase300 tp4dp2 12k2k deeper trace cadence diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, NamedTuple


SOURCE = "phase301_boundary_timeline_cadence_diagnostic"
TRACE_SOURCE = "phase274_deeper_scheduler_trace"
VERDICT = "boundary_timeline_explains_direction"
MECHANISM_CONCLUSION = "wall_span_iteration_cadence_diagnostic"
DEFAULT_READINESS = "No-Go"
SOURCE_SHA_COMPARISON = "startup_diagnostics_only"

CONTROL_SCENARIO = "tp4dp2ep8-12k2k-bt12000"
HOLDOUT_SCENARIO = "tp4dp2ep8-12k2k-bt65536"
CONTROL_SHA_FILE = ".phase291_source_sha"
HOLDOUT_SHA_FILE = ".phase298_source_sha"
CONTROL_SOURCE_SHA = "65372730f59a99cbb237e951a9a27871a0f78bdb"
HOLDOUT_SOURCE_SHA = "d357f2d428a35272acc3c5e0d9af09b792dfa4b0"
TOPOLOGY_KEY = "tp4_dp2_ep8"
SHAPE_KEY = "isl12000_osl2000_batch128"
CONTROL_BT = 12000
HOLDOUT_BT = 65536
TP = 4
DP = 2
EP = 8
MAX_NUM_SEQS = 256
REQUESTS = 128
VALID_PHASES = {"prefill", "mixed", "pure_decode"}

DEFAULT_CSV_OUT = Path("docs/iter_gap_investigation/phase301_boundary_timeline_cadence_diagnostic.csv")
DEFAULT_DOC_OUT = Path("docs/iter_gap_investigation/phase301_boundary_timeline_cadence_diagnostic.md")

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
    "iteration_start_ns",
    "forward_start_ns",
    "forward_end_ns",
    "iteration_end_ns",
    "iteration_elapsed_ns",
    "forward_elapsed_ns",
    "active_request_count",
    "scheduled_new_req_count",
    "scheduled_cached_req_count",
    "scheduled_resumed_req_count",
    "finished_req_count",
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
    "max_num_batched_tokens",
    "max_num_seqs",
    "forward_token_count",
    "tp",
    "dp",
    "ep",
    "topology_key",
    "shape_key",
    "active_request_count",
    "scheduled_new_req_count",
    "scheduled_cached_req_count",
    "scheduled_resumed_req_count",
    "finished_req_count",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
)

RANK_SUM_FIELDS = (
    "scheduled_context_tokens",
    "scheduled_decode_tokens",
    "scheduled_total_tokens",
    "scheduled_context_reqs",
    "scheduled_decode_reqs",
    "scheduled_total_reqs",
    "forward_token_count",
    "active_request_count",
    "scheduled_new_req_count",
    "scheduled_cached_req_count",
    "scheduled_resumed_req_count",
    "finished_req_count",
)

FIELDNAMES = [
    "source",
    "pair_key",
    "topology_key",
    "shape_key",
    "control_scenario",
    "holdout_scenario",
    "control_bt",
    "holdout_bt",
    "control_source_sha",
    "holdout_source_sha",
    "source_sha_comparison",
    "control_output_tok_s",
    "holdout_output_tok_s",
    "output_ratio",
    "control_unique_iterations",
    "holdout_unique_iterations",
    "unique_iteration_delta",
    "control_phase_counts",
    "holdout_phase_counts",
    "control_mixed_sequence",
    "holdout_mixed_sequence",
    "mixed_iteration_delta",
    "control_first_pure_decode",
    "holdout_first_pure_decode",
    "first_pure_decode_delta",
    "control_tail_decode_tokens",
    "holdout_tail_decode_tokens",
    "control_scheduled_p50",
    "control_scheduled_p95",
    "control_scheduled_p99",
    "control_scheduled_max",
    "holdout_scheduled_p50",
    "holdout_scheduled_p95",
    "holdout_scheduled_p99",
    "holdout_scheduled_max",
    "control_forward_p50",
    "control_forward_p95",
    "control_forward_p99",
    "control_forward_max",
    "holdout_forward_p50",
    "holdout_forward_p95",
    "holdout_forward_p99",
    "holdout_forward_max",
    "control_scheduled_max_fill",
    "holdout_scheduled_max_fill",
    "control_trace_wall_span_s",
    "holdout_trace_wall_span_s",
    "trace_wall_span_delta_s",
    "control_forward_wall_span_s",
    "holdout_forward_wall_span_s",
    "forward_wall_span_delta_s",
    "control_iteration_start_delta_p50_ms",
    "holdout_iteration_start_delta_p50_ms",
    "iteration_start_delta_p50_delta_ms",
    "control_forward_elapsed_p50_ms",
    "control_forward_elapsed_p95_ms",
    "holdout_forward_elapsed_p50_ms",
    "holdout_forward_elapsed_p95_ms",
    "forward_elapsed_p50_delta_ms",
    "forward_elapsed_p95_delta_ms",
    "control_overhead_p50_ms",
    "control_overhead_p95_ms",
    "holdout_overhead_p50_ms",
    "holdout_overhead_p95_ms",
    "overhead_p50_delta_ms",
    "overhead_p95_delta_ms",
    "verdict",
    "mechanism_conclusion",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]


class ScenarioSpec(NamedTuple):
    scenario: str
    role: str
    max_bt: int
    sha_file: str
    source_sha: str


class CaseSummary(NamedTuple):
    scenario: str
    role: str
    max_bt: int
    source_sha: str
    output_tok_s: float
    raw_rows: int
    unique_iterations: int
    phase_counts: Counter[str]
    mixed_sequence: tuple[int, ...]
    first_pure_decode: int
    tail_decode_tokens: tuple[int, ...]
    scheduled_stats: dict[str, float]
    forward_stats: dict[str, float]
    scheduled_max_fill: float
    trace_wall_span_s: float
    forward_wall_span_s: float
    iteration_start_delta_stats_ms: dict[str, float]
    forward_elapsed_stats_ms: dict[str, float]
    overhead_stats_ms: dict[str, float]


SCENARIOS = {
    CONTROL_SCENARIO: ScenarioSpec(
        CONTROL_SCENARIO,
        "control",
        CONTROL_BT,
        CONTROL_SHA_FILE,
        CONTROL_SOURCE_SHA,
    ),
    HOLDOUT_SCENARIO: ScenarioSpec(
        HOLDOUT_SCENARIO,
        "holdout",
        HOLDOUT_BT,
        HOLDOUT_SHA_FILE,
        HOLDOUT_SOURCE_SHA,
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"json object expected: {path}")
    return value


def _read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"trace row must be object: {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"trace is empty: {path}")
    return rows


def _resolve_case_dir(path: Path, scenario: str) -> Path:
    if (path / "phase274_result.json").exists():
        return path
    candidate = path / "docs" / "iter_gap_investigation" / "phase274_deeper_scheduler_trace" / scenario
    if (candidate / "phase274_result.json").exists():
        return candidate
    raise FileNotFoundError(f"scenario artifact directory not found: {path} scenario={scenario}")


def _find_source_sha(case_dir: Path, spec: ScenarioSpec) -> str:
    for current in (case_dir, *case_dir.parents):
        sha_path = current / spec.sha_file
        if sha_path.exists():
            value = sha_path.read_text(encoding="utf-8").strip()
            if value != spec.source_sha:
                raise ValueError(f"source sha mismatch: {spec.scenario}")
            return value
    raise FileNotFoundError(f"source sha file not found: {spec.sha_file}")


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}"


def _format_sequence(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def _format_phase_counts(counts: Counter[str]) -> str:
    return ",".join(f"{phase}={counts[phase]}" for phase in ("prefill", "mixed", "pure_decode"))


def _as_int(row: dict[str, Any], key: str, label: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be integer: {key}")
    return value


def _require_nonnegative_int(row: dict[str, Any], key: str, label: str) -> int:
    value = _as_int(row, key, label)
    if value < 0:
        raise ValueError(f"{label} must be non-negative: {key}")
    return value


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        raise ValueError("percentile requires non-empty values")
    sorted_values = sorted(values)
    if percentile == 50:
        midpoint = len(sorted_values) // 2
        if len(sorted_values) % 2:
            return float(sorted_values[midpoint])
        return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2
    rank = math.ceil((percentile / 100) * len(sorted_values)) - 1
    return float(sorted_values[max(0, min(rank, len(sorted_values) - 1))])


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("stats requires non-empty values")
    return {
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": max(values),
    }


def _validate_flags(source: str, value: dict[str, Any]) -> None:
    if (
        value.get("diagnostic_only") is not True
        or value.get("valid_for_default") is not False
        or value.get("perf_database") is not False
    ):
        raise ValueError(f"{source} flag mismatch")


def _validate_result(case_dir: Path, spec: ScenarioSpec) -> None:
    result = _read_json(case_dir / "phase274_result.json")
    if result.get("source") != TRACE_SOURCE:
        raise ValueError(f"result source mismatch: {spec.scenario}")
    if result.get("scenario") != spec.scenario:
        raise ValueError(f"result scenario mismatch: {spec.scenario}")
    if result.get("topology_key") != TOPOLOGY_KEY or result.get("shape_key") != SHAPE_KEY:
        raise ValueError(f"result topology/shape mismatch: {spec.scenario}")
    _validate_flags("result", result)

    if result.get("parallelism") != {"tp": TP, "dp": DP, "ep": EP}:
        raise ValueError(f"result parallelism mismatch: {spec.scenario}")

    expected_shape = {
        "isl": 12000,
        "osl": 2000,
        "batch_size": REQUESTS,
        "max_num_batched_tokens": spec.max_bt,
        "max_num_seqs": MAX_NUM_SEQS,
    }
    if result.get("shape") != expected_shape:
        raise ValueError(f"result shape mismatch: {spec.scenario}")


def _validate_bench(case_dir: Path, spec: ScenarioSpec) -> float:
    bench = _read_json(case_dir / "bench_result.json")
    if bench.get("ok_requests") != REQUESTS or bench.get("failed_requests") != 0:
        raise ValueError(f"benchmark request mismatch: {spec.scenario}")
    output_tok_s = bench.get("output_tok_s")
    if not isinstance(output_tok_s, (int, float)) or output_tok_s <= 0:
        raise ValueError(f"benchmark output_tok_s mismatch: {spec.scenario}")

    records_path = case_dir / "bench_records.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(records_path)
    with records_path.open(encoding="utf-8") as handle:
        record_count = sum(1 for line in handle if line.strip())
    if record_count != REQUESTS:
        raise ValueError(f"benchmark record count mismatch: {spec.scenario}")
    return float(output_tok_s)


def _validate_trace_row(row: dict[str, Any], spec: ScenarioSpec, index: int) -> None:
    missing = REQUIRED_TRACE_FIELDS - row.keys()
    if missing:
        raise ValueError(f"trace schema mismatch: {spec.scenario} row={index} missing={sorted(missing)}")
    if row.get("source") != TRACE_SOURCE:
        raise ValueError(f"trace source mismatch: {spec.scenario}")
    if row.get("scenario") != spec.scenario:
        raise ValueError(f"trace scenario mismatch: {spec.scenario}")
    if row.get("phase") not in VALID_PHASES:
        raise ValueError(f"trace phase mismatch: {spec.scenario}")
    if row.get("topology_key") != TOPOLOGY_KEY or row.get("shape_key") != SHAPE_KEY:
        raise ValueError(f"trace topology/shape mismatch: {spec.scenario}")
    if row.get("tp") != TP or row.get("dp") != DP or row.get("ep") != EP:
        raise ValueError(f"trace parallelism mismatch: {spec.scenario}")
    if row.get("max_num_batched_tokens") != spec.max_bt or row.get("max_num_seqs") != MAX_NUM_SEQS:
        raise ValueError(f"trace budget mismatch: {spec.scenario}")
    _validate_flags("trace", row)

    iteration = _require_nonnegative_int(row, "iteration", "trace")
    context_tokens = _require_nonnegative_int(row, "scheduled_context_tokens", "trace")
    decode_tokens = _require_nonnegative_int(row, "scheduled_decode_tokens", "trace")
    total_tokens = _require_nonnegative_int(row, "scheduled_total_tokens", "trace")
    context_reqs = _require_nonnegative_int(row, "scheduled_context_reqs", "trace")
    decode_reqs = _require_nonnegative_int(row, "scheduled_decode_reqs", "trace")
    total_reqs = _require_nonnegative_int(row, "scheduled_total_reqs", "trace")
    _require_nonnegative_int(row, "forward_token_count", "trace")
    active_count = _require_nonnegative_int(row, "active_request_count", "trace")
    _require_nonnegative_int(row, "scheduled_new_req_count", "trace")
    _require_nonnegative_int(row, "scheduled_cached_req_count", "trace")
    _require_nonnegative_int(row, "scheduled_resumed_req_count", "trace")
    _require_nonnegative_int(row, "finished_req_count", "trace")

    if total_tokens != context_tokens + decode_tokens:
        raise ValueError(f"trace token sum mismatch: {spec.scenario} iteration={iteration}")
    if total_reqs != context_reqs + decode_reqs:
        raise ValueError(f"trace request sum mismatch: {spec.scenario} iteration={iteration}")
    if active_count > REQUESTS:
        raise ValueError(f"trace active request count mismatch: {spec.scenario} iteration={iteration}")

    iteration_start = _require_nonnegative_int(row, "iteration_start_ns", "trace")
    forward_start = _require_nonnegative_int(row, "forward_start_ns", "trace")
    forward_end = _require_nonnegative_int(row, "forward_end_ns", "trace")
    iteration_end = _require_nonnegative_int(row, "iteration_end_ns", "trace")
    iteration_elapsed = _require_nonnegative_int(row, "iteration_elapsed_ns", "trace")
    forward_elapsed = _require_nonnegative_int(row, "forward_elapsed_ns", "trace")
    if not (iteration_start <= forward_start <= forward_end <= iteration_end):
        raise ValueError(f"trace timing order mismatch: {spec.scenario} iteration={iteration}")
    if forward_elapsed != forward_end - forward_start:
        raise ValueError(f"trace forward elapsed mismatch: {spec.scenario} iteration={iteration}")
    if iteration_elapsed < forward_elapsed:
        raise ValueError(f"trace iteration elapsed mismatch: {spec.scenario} iteration={iteration}")


def _payload_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in PAYLOAD_SIGNATURE_FIELDS)


def _aggregate_trace(rows: list[dict[str, Any]], spec: ScenarioSpec) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        _validate_trace_row(row, spec, index)
        grouped[_as_int(row, "iteration", "trace")].append(row)

    iterations = sorted(grouped)
    if iterations != list(range(iterations[0], iterations[-1] + 1)):
        raise ValueError(f"trace iterations must be contiguous: {spec.scenario}")
    if iterations[0] != 0:
        raise ValueError(f"trace iterations must start at zero: {spec.scenario}")

    aggregated: list[dict[str, Any]] = []
    for iteration in iterations:
        worker_rows = grouped[iteration]
        if len(worker_rows) != TP * DP:
            raise ValueError(f"trace worker count mismatch: {spec.scenario} iteration={iteration}")
        payload_counts = Counter(_payload_signature(row) for row in worker_rows)
        if len(payload_counts) > DP:
            raise ValueError(f"payload count mismatch: {spec.scenario} iteration={iteration}")
        if any(count % TP != 0 for count in payload_counts.values()):
            raise ValueError(f"payload worker count mismatch: {spec.scenario} iteration={iteration}")
        if sum(count // TP for count in payload_counts.values()) != DP:
            raise ValueError(f"payload rank count mismatch: {spec.scenario} iteration={iteration}")

        sums = {field: 0 for field in RANK_SUM_FIELDS}
        for signature, count in payload_counts.items():
            payload = dict(zip(PAYLOAD_SIGNATURE_FIELDS, signature))
            rank_count = count // TP
            for field in RANK_SUM_FIELDS:
                sums[field] += int(payload[field]) * rank_count

        if sums["scheduled_context_tokens"] > 0 and sums["scheduled_decode_tokens"] > 0:
            phase = "mixed"
        elif sums["scheduled_context_tokens"] > 0:
            phase = "prefill"
        else:
            phase = "pure_decode"

        aggregated.append(
            {
                "iteration": iteration,
                "phase": phase,
                "iteration_start_ns_min": min(_as_int(row, "iteration_start_ns", "trace") for row in worker_rows),
                "iteration_end_ns_max": max(_as_int(row, "iteration_end_ns", "trace") for row in worker_rows),
                "forward_start_ns_min": min(_as_int(row, "forward_start_ns", "trace") for row in worker_rows),
                "forward_end_ns_max": max(_as_int(row, "forward_end_ns", "trace") for row in worker_rows),
                "forward_elapsed_ns_rank_max": max(
                    _as_int(row, "forward_elapsed_ns", "trace") for row in worker_rows
                ),
                "iteration_elapsed_ns_rank_max": max(
                    _as_int(row, "iteration_elapsed_ns", "trace") for row in worker_rows
                ),
                "overhead_ns_rank_max": max(
                    _as_int(row, "iteration_elapsed_ns", "trace") - _as_int(row, "forward_elapsed_ns", "trace")
                    for row in worker_rows
                ),
                **sums,
            }
        )
    return aggregated


def _case_summary(input_dir: Path, spec: ScenarioSpec) -> CaseSummary:
    case_dir = _resolve_case_dir(input_dir, spec.scenario)
    source_sha = _find_source_sha(case_dir, spec)
    _validate_result(case_dir, spec)
    output_tok_s = _validate_bench(case_dir, spec)
    raw_rows = _read_trace(case_dir / "deeper_scheduler_trace.jsonl")
    rows = _aggregate_trace(raw_rows, spec)

    phases = Counter(str(row["phase"]) for row in rows)
    if phases["prefill"] < 1 or phases["pure_decode"] < 1:
        raise ValueError(f"trace phase coverage mismatch: {spec.scenario}")
    mixed_sequence = tuple(_as_int(row, "iteration", "trace") for row in rows if row["phase"] == "mixed")
    pure_iterations = [_as_int(row, "iteration", "trace") for row in rows if row["phase"] == "pure_decode"]
    first_pure_decode = min(pure_iterations)
    tail_decode_tokens = tuple(_as_int(row, "scheduled_decode_tokens", "trace") for row in rows[-5:])

    scheduled = [float(_as_int(row, "scheduled_total_tokens", "trace")) for row in rows]
    forward = [float(_as_int(row, "forward_token_count", "trace")) for row in rows]
    iteration_starts = [float(_as_int(row, "iteration_start_ns_min", "trace")) for row in rows]
    iteration_ends = [float(_as_int(row, "iteration_end_ns_max", "trace")) for row in rows]
    forward_starts = [float(_as_int(row, "forward_start_ns_min", "trace")) for row in rows]
    forward_ends = [float(_as_int(row, "forward_end_ns_max", "trace")) for row in rows]
    start_delta_ms = [
        (iteration_starts[index] - iteration_starts[index - 1]) / 1_000_000
        for index in range(1, len(iteration_starts))
    ]
    if not start_delta_ms:
        raise ValueError(f"trace must contain at least two iterations: {spec.scenario}")

    aggregate_budget = spec.max_bt * DP
    return CaseSummary(
        scenario=spec.scenario,
        role=spec.role,
        max_bt=spec.max_bt,
        source_sha=source_sha,
        output_tok_s=output_tok_s,
        raw_rows=len(raw_rows),
        unique_iterations=len(rows),
        phase_counts=phases,
        mixed_sequence=mixed_sequence,
        first_pure_decode=first_pure_decode,
        tail_decode_tokens=tail_decode_tokens,
        scheduled_stats=_stats(scheduled),
        forward_stats=_stats(forward),
        scheduled_max_fill=max(scheduled) / aggregate_budget,
        trace_wall_span_s=(max(iteration_ends) - min(iteration_starts)) / 1_000_000_000,
        forward_wall_span_s=(max(forward_ends) - min(forward_starts)) / 1_000_000_000,
        iteration_start_delta_stats_ms=_stats(start_delta_ms),
        forward_elapsed_stats_ms=_stats(
            [_as_int(row, "forward_elapsed_ns_rank_max", "trace") / 1_000_000 for row in rows]
        ),
        overhead_stats_ms=_stats([_as_int(row, "overhead_ns_rank_max", "trace") / 1_000_000 for row in rows]),
    )


def _delta(holdout: dict[str, float], control: dict[str, float], key: str) -> float:
    return holdout[key] - control[key]


def _pair_row(control: CaseSummary, holdout: CaseSummary) -> dict[str, str]:
    if control.scenario != CONTROL_SCENARIO or holdout.scenario != HOLDOUT_SCENARIO:
        raise ValueError("scenario pair mismatch")
    output_ratio = holdout.output_tok_s / control.output_tok_s
    if holdout.scheduled_max_fill >= 0.5:
        raise ValueError("holdout max fill is too high for cadence diagnostic conclusion")

    return {
        "source": SOURCE,
        "pair_key": f"{TOPOLOGY_KEY}:{SHAPE_KEY}:control_bt{CONTROL_BT}:holdout_bt{HOLDOUT_BT}",
        "topology_key": TOPOLOGY_KEY,
        "shape_key": SHAPE_KEY,
        "control_scenario": control.scenario,
        "holdout_scenario": holdout.scenario,
        "control_bt": str(control.max_bt),
        "holdout_bt": str(holdout.max_bt),
        "control_source_sha": control.source_sha,
        "holdout_source_sha": holdout.source_sha,
        "source_sha_comparison": SOURCE_SHA_COMPARISON,
        "control_output_tok_s": _format_float(control.output_tok_s),
        "holdout_output_tok_s": _format_float(holdout.output_tok_s),
        "output_ratio": _format_float(output_ratio),
        "control_unique_iterations": str(control.unique_iterations),
        "holdout_unique_iterations": str(holdout.unique_iterations),
        "unique_iteration_delta": str(holdout.unique_iterations - control.unique_iterations),
        "control_phase_counts": _format_phase_counts(control.phase_counts),
        "holdout_phase_counts": _format_phase_counts(holdout.phase_counts),
        "control_mixed_sequence": _format_sequence(control.mixed_sequence),
        "holdout_mixed_sequence": _format_sequence(holdout.mixed_sequence),
        "mixed_iteration_delta": str(len(holdout.mixed_sequence) - len(control.mixed_sequence)),
        "control_first_pure_decode": str(control.first_pure_decode),
        "holdout_first_pure_decode": str(holdout.first_pure_decode),
        "first_pure_decode_delta": str(holdout.first_pure_decode - control.first_pure_decode),
        "control_tail_decode_tokens": _format_sequence(control.tail_decode_tokens),
        "holdout_tail_decode_tokens": _format_sequence(holdout.tail_decode_tokens),
        "control_scheduled_p50": _format_number(control.scheduled_stats["p50"]),
        "control_scheduled_p95": _format_number(control.scheduled_stats["p95"]),
        "control_scheduled_p99": _format_number(control.scheduled_stats["p99"]),
        "control_scheduled_max": _format_number(control.scheduled_stats["max"]),
        "holdout_scheduled_p50": _format_number(holdout.scheduled_stats["p50"]),
        "holdout_scheduled_p95": _format_number(holdout.scheduled_stats["p95"]),
        "holdout_scheduled_p99": _format_number(holdout.scheduled_stats["p99"]),
        "holdout_scheduled_max": _format_number(holdout.scheduled_stats["max"]),
        "control_forward_p50": _format_number(control.forward_stats["p50"]),
        "control_forward_p95": _format_number(control.forward_stats["p95"]),
        "control_forward_p99": _format_number(control.forward_stats["p99"]),
        "control_forward_max": _format_number(control.forward_stats["max"]),
        "holdout_forward_p50": _format_number(holdout.forward_stats["p50"]),
        "holdout_forward_p95": _format_number(holdout.forward_stats["p95"]),
        "holdout_forward_p99": _format_number(holdout.forward_stats["p99"]),
        "holdout_forward_max": _format_number(holdout.forward_stats["max"]),
        "control_scheduled_max_fill": _format_float(control.scheduled_max_fill),
        "holdout_scheduled_max_fill": _format_float(holdout.scheduled_max_fill),
        "control_trace_wall_span_s": _format_float(control.trace_wall_span_s),
        "holdout_trace_wall_span_s": _format_float(holdout.trace_wall_span_s),
        "trace_wall_span_delta_s": _format_float(holdout.trace_wall_span_s - control.trace_wall_span_s),
        "control_forward_wall_span_s": _format_float(control.forward_wall_span_s),
        "holdout_forward_wall_span_s": _format_float(holdout.forward_wall_span_s),
        "forward_wall_span_delta_s": _format_float(holdout.forward_wall_span_s - control.forward_wall_span_s),
        "control_iteration_start_delta_p50_ms": _format_float(control.iteration_start_delta_stats_ms["p50"]),
        "holdout_iteration_start_delta_p50_ms": _format_float(holdout.iteration_start_delta_stats_ms["p50"]),
        "iteration_start_delta_p50_delta_ms": _format_float(
            _delta(holdout.iteration_start_delta_stats_ms, control.iteration_start_delta_stats_ms, "p50")
        ),
        "control_forward_elapsed_p50_ms": _format_float(control.forward_elapsed_stats_ms["p50"]),
        "control_forward_elapsed_p95_ms": _format_float(control.forward_elapsed_stats_ms["p95"]),
        "holdout_forward_elapsed_p50_ms": _format_float(holdout.forward_elapsed_stats_ms["p50"]),
        "holdout_forward_elapsed_p95_ms": _format_float(holdout.forward_elapsed_stats_ms["p95"]),
        "forward_elapsed_p50_delta_ms": _format_float(
            _delta(holdout.forward_elapsed_stats_ms, control.forward_elapsed_stats_ms, "p50")
        ),
        "forward_elapsed_p95_delta_ms": _format_float(
            _delta(holdout.forward_elapsed_stats_ms, control.forward_elapsed_stats_ms, "p95")
        ),
        "control_overhead_p50_ms": _format_float(control.overhead_stats_ms["p50"]),
        "control_overhead_p95_ms": _format_float(control.overhead_stats_ms["p95"]),
        "holdout_overhead_p50_ms": _format_float(holdout.overhead_stats_ms["p50"]),
        "holdout_overhead_p95_ms": _format_float(holdout.overhead_stats_ms["p95"]),
        "overhead_p50_delta_ms": _format_float(
            _delta(holdout.overhead_stats_ms, control.overhead_stats_ms, "p50")
        ),
        "overhead_p95_delta_ms": _format_float(
            _delta(holdout.overhead_stats_ms, control.overhead_stats_ms, "p95")
        ),
        "verdict": VERDICT,
        "mechanism_conclusion": MECHANISM_CONCLUSION,
        "default_readiness": DEFAULT_READINESS,
        "diagnostic_only": _bool_text(True),
        "valid_for_default": _bool_text(False),
        "perf_database": _bool_text(False),
    }


def analyze_boundary_timeline_cadence_diagnostic(control_dir: Path, holdout_dir: Path) -> list[dict[str, str]]:
    """Return one pair-level cadence diagnostic row for the Phase300 trace pair."""
    control = _case_summary(control_dir, SCENARIOS[CONTROL_SCENARIO])
    holdout = _case_summary(holdout_dir, SCENARIOS[HOLDOUT_SCENARIO])
    return [_pair_row(control, holdout)]


def write_boundary_timeline_cadence_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_boundary_timeline_cadence_doc(path: Path, rows: list[dict[str, str]]) -> None:
    if len(rows) != 1:
        raise ValueError("Phase301 document expects exactly one row")
    row = rows[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Phase301 Boundary Timeline Cadence Diagnostic

This diagnostic compares the tp4dp2 12k2k deeper trace control/holdout pair.

| Item | Value |
|---|---|
| Pair | {row["pair_key"]} |
| Source SHA comparison | {row["source_sha_comparison"]} |
| Output ratio | {row["output_ratio"]} |
| Mixed sequence | {row["control_mixed_sequence"]} -> {row["holdout_mixed_sequence"]} |
| First pure decode | {row["control_first_pure_decode"]} -> {row["holdout_first_pure_decode"]} |
| Trace wall-span | {row["control_trace_wall_span_s"]}s -> {row["holdout_trace_wall_span_s"]}s |
| Iteration cadence p50 | {row["control_iteration_start_delta_p50_ms"]}ms -> {row["holdout_iteration_start_delta_p50_ms"]}ms |
| Scheduled max fill | {row["control_scheduled_max_fill"]} -> {row["holdout_scheduled_max_fill"]} |
| Verdict | {row["verdict"]} |
| Mechanism | {row["mechanism_conclusion"]} |
| Default AIC | {row["default_readiness"]} |

The main signal is shorter trace wall-span and faster iteration-start cadence. Mixed phase and first-pure-decode movement explain the direction, while overhead p50/p95 is not promoted into a standalone model.

The result stays diagnostic-only. It must not be used for default AIC, interpolation, extrapolation, or PerfDatabase writes.

Flags: diagnostic_only={row["diagnostic_only"]}, valid_for_default={row["valid_for_default"]}, perf_database={row["perf_database"]}.
"""
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--holdout-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_CSV_OUT)
    parser.add_argument("--doc-out", type=Path, default=DEFAULT_DOC_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = analyze_boundary_timeline_cadence_diagnostic(args.control_dir, args.holdout_dir)
    write_boundary_timeline_cadence_csv(args.out, rows)
    write_boundary_timeline_cadence_doc(args.doc_out, rows)
    print(f"wrote {args.out}")
    print(f"wrote {args.doc_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
