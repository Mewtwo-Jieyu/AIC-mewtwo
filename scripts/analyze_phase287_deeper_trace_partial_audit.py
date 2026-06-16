#!/usr/bin/env python3
"""Audit Phase286 deeper trace pair as partial-only diagnostic evidence."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, NamedTuple


SOURCE = "phase287_deeper_trace_partial_audit"
TRACE_SOURCE = "phase274_deeper_scheduler_trace"
MECHANISM_CONCLUSION = "boundary_mixed_overhead_partial_only"
VERDICT = "partial_only"
DEFAULT_READINESS = "No-Go"

CONTROL_SCENARIO = "tp8ep8-12k2k-bt12000"
HOLDOUT_SCENARIO = "tp8ep8-12k2k-bt65536"
TOPOLOGY_KEY = "tp8_dp1_ep8"
SHAPE_KEY = "isl12000_osl2000_batch128"
CONTROL_BT = 12000
HOLDOUT_BT = 65536
TP = 8
DP = 1
EP = 8
MAX_NUM_SEQS = 256
REQUESTS = 128
VALID_PHASES = {"prefill", "mixed", "pure_decode"}

DEFAULT_INPUT_ROOT = Path("docs/iter_gap_investigation/phase274_deeper_scheduler_trace")
DEFAULT_CSV_OUT = Path("docs/iter_gap_investigation/phase287_deeper_trace_partial_audit.csv")
DEFAULT_DOC_OUT = Path("docs/iter_gap_investigation/phase287_deeper_trace_partial_audit.md")

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
    "active_request_count",
    "scheduled_new_req_count",
    "scheduled_cached_req_count",
    "scheduled_resumed_req_count",
    "finished_req_count",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
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
    "control_output_tok_s",
    "holdout_output_tok_s",
    "output_ratio",
    "control_unique_iterations",
    "holdout_unique_iterations",
    "unique_iteration_delta",
    "control_mixed_iterations",
    "holdout_mixed_iterations",
    "mixed_iteration_delta",
    "control_pure_decode_iterations",
    "holdout_pure_decode_iterations",
    "pure_decode_iteration_delta",
    "control_first_pure_iteration",
    "holdout_first_pure_iteration",
    "control_tail_decode_tokens",
    "holdout_tail_decode_tokens",
    "control_scheduled_p99",
    "holdout_scheduled_p99",
    "control_scheduled_max",
    "holdout_scheduled_max",
    "holdout_max_fill",
    "control_forward_token_max",
    "holdout_forward_token_max",
    "forward_elapsed_p50_delta_ns",
    "forward_elapsed_p95_delta_ns",
    "forward_elapsed_p99_delta_ns",
    "forward_elapsed_max_delta_ns",
    "iteration_elapsed_p50_delta_ns",
    "iteration_elapsed_p95_delta_ns",
    "iteration_elapsed_p99_delta_ns",
    "iteration_elapsed_max_delta_ns",
    "overhead_p50_delta_ns",
    "overhead_p95_delta_ns",
    "overhead_p99_delta_ns",
    "overhead_max_delta_ns",
    "control_active_request_max",
    "holdout_active_request_max",
    "control_scheduled_new_req_sum",
    "holdout_scheduled_new_req_sum",
    "control_scheduled_cached_req_sum",
    "holdout_scheduled_cached_req_sum",
    "control_scheduled_resumed_req_sum",
    "holdout_scheduled_resumed_req_sum",
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


class CaseSummary(NamedTuple):
    scenario: str
    role: str
    max_bt: int
    output_tok_s: float
    raw_rows: int
    unique_iterations: int
    phase_counts: Counter[str]
    mixed_iterations: tuple[int, ...]
    first_pure_iteration: int
    tail_decode_tokens: tuple[int, ...]
    scheduled_stats: dict[str, int]
    forward_token_stats: dict[str, int]
    forward_elapsed_stats: dict[str, int]
    iteration_elapsed_stats: dict[str, int]
    overhead_stats: dict[str, int]
    active_request_max: int
    scheduled_new_req_sum: int
    scheduled_cached_req_sum: int
    scheduled_resumed_req_sum: int


SCENARIOS = {
    CONTROL_SCENARIO: ScenarioSpec(CONTROL_SCENARIO, "control", CONTROL_BT),
    HOLDOUT_SCENARIO: ScenarioSpec(HOLDOUT_SCENARIO, "holdout", HOLDOUT_BT),
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


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _format_sequence(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


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


def _percentile(values: list[int], percentile: int) -> int:
    if not values:
        raise ValueError("percentile requires non-empty values")
    sorted_values = sorted(values)
    rank = math.ceil((percentile / 100) * len(sorted_values)) - 1
    return sorted_values[max(0, min(rank, len(sorted_values) - 1))]


def _stats(values: list[int]) -> dict[str, int]:
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

    parallelism = result.get("parallelism")
    if parallelism != {"tp": TP, "dp": DP, "ep": EP}:
        raise ValueError(f"result parallelism mismatch: {spec.scenario}")

    shape = result.get("shape")
    expected_shape = {
        "isl": 12000,
        "osl": 2000,
        "batch_size": REQUESTS,
        "max_num_batched_tokens": spec.max_bt,
        "max_num_seqs": MAX_NUM_SEQS,
    }
    if shape != expected_shape:
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
    forward_token_count = _require_nonnegative_int(row, "forward_token_count", "trace")
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


def _dedup_trace(rows: list[dict[str, Any]], spec: ScenarioSpec) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        _validate_trace_row(row, spec, index)
        grouped[_as_int(row, "iteration", "trace")].append(row)

    iterations = sorted(grouped)
    if iterations != list(range(iterations[0], iterations[-1] + 1)):
        raise ValueError(f"trace iterations must be contiguous: {spec.scenario}")
    if iterations[0] != 0:
        raise ValueError(f"trace iterations must start at zero: {spec.scenario}")

    deduped: list[dict[str, Any]] = []
    for iteration in iterations:
        worker_rows = grouped[iteration]
        if len(worker_rows) != TP:
            raise ValueError(f"trace worker count mismatch: {spec.scenario} iteration={iteration}")
        signatures = {_payload_signature(row) for row in worker_rows}
        if len(signatures) != 1:
            raise ValueError(f"scheduler payload divergence: {spec.scenario} iteration={iteration}")

        base = dict(worker_rows[0])
        base["forward_elapsed_ns_rank_max"] = max(_as_int(row, "forward_elapsed_ns", "trace") for row in worker_rows)
        base["iteration_elapsed_ns_rank_max"] = max(
            _as_int(row, "iteration_elapsed_ns", "trace") for row in worker_rows
        )
        base["overhead_ns_rank_max"] = max(
            _as_int(row, "iteration_elapsed_ns", "trace") - _as_int(row, "forward_elapsed_ns", "trace")
            for row in worker_rows
        )
        deduped.append(base)
    return deduped


def _case_summary(input_root: Path, spec: ScenarioSpec) -> CaseSummary:
    case_dir = input_root / spec.scenario
    if not case_dir.exists():
        raise FileNotFoundError(case_dir)
    _validate_result(case_dir, spec)
    output_tok_s = _validate_bench(case_dir, spec)
    raw_rows = _read_trace(case_dir / "deeper_scheduler_trace.jsonl")
    deduped = _dedup_trace(raw_rows, spec)

    phases = Counter(str(row["phase"]) for row in deduped)
    if phases["prefill"] < 1 or phases["pure_decode"] < 1:
        raise ValueError(f"trace phase coverage mismatch: {spec.scenario}")
    mixed_iterations = tuple(_as_int(row, "iteration", "trace") for row in deduped if row["phase"] == "mixed")
    pure_iterations = [_as_int(row, "iteration", "trace") for row in deduped if row["phase"] == "pure_decode"]
    first_pure_iteration = min(pure_iterations)
    tail_decode_tokens = tuple(_as_int(row, "scheduled_decode_tokens", "trace") for row in deduped[-5:])

    scheduled_tokens = [_as_int(row, "scheduled_total_tokens", "trace") for row in deduped]
    forward_tokens = [_as_int(row, "forward_token_count", "trace") for row in deduped]
    forward_elapsed = [_as_int(row, "forward_elapsed_ns_rank_max", "trace") for row in deduped]
    iteration_elapsed = [_as_int(row, "iteration_elapsed_ns_rank_max", "trace") for row in deduped]
    overhead = [_as_int(row, "overhead_ns_rank_max", "trace") for row in deduped]

    return CaseSummary(
        scenario=spec.scenario,
        role=spec.role,
        max_bt=spec.max_bt,
        output_tok_s=output_tok_s,
        raw_rows=len(raw_rows),
        unique_iterations=len(deduped),
        phase_counts=phases,
        mixed_iterations=mixed_iterations,
        first_pure_iteration=first_pure_iteration,
        tail_decode_tokens=tail_decode_tokens,
        scheduled_stats=_stats(scheduled_tokens),
        forward_token_stats=_stats(forward_tokens),
        forward_elapsed_stats=_stats(forward_elapsed),
        iteration_elapsed_stats=_stats(iteration_elapsed),
        overhead_stats=_stats(overhead),
        active_request_max=max(_as_int(row, "active_request_count", "trace") for row in deduped),
        scheduled_new_req_sum=sum(_as_int(row, "scheduled_new_req_count", "trace") for row in deduped),
        scheduled_cached_req_sum=sum(_as_int(row, "scheduled_cached_req_count", "trace") for row in deduped),
        scheduled_resumed_req_sum=sum(_as_int(row, "scheduled_resumed_req_count", "trace") for row in deduped),
    )


def _delta(holdout: dict[str, int], control: dict[str, int], key: str) -> str:
    return str(holdout[key] - control[key])


def _pair_row(control: CaseSummary, holdout: CaseSummary) -> dict[str, str]:
    if control.scenario != CONTROL_SCENARIO or holdout.scenario != HOLDOUT_SCENARIO:
        raise ValueError("scenario pair mismatch")
    output_ratio = holdout.output_tok_s / control.output_tok_s
    holdout_max_fill = holdout.scheduled_stats["max"] / holdout.max_bt
    if holdout_max_fill >= 0.5:
        raise ValueError("holdout max fill is too high for partial-only conclusion")

    return {
        "source": SOURCE,
        "pair_key": f"{TOPOLOGY_KEY}:{SHAPE_KEY}:control_bt{CONTROL_BT}:holdout_bt{HOLDOUT_BT}",
        "topology_key": TOPOLOGY_KEY,
        "shape_key": SHAPE_KEY,
        "control_scenario": control.scenario,
        "holdout_scenario": holdout.scenario,
        "control_bt": str(control.max_bt),
        "holdout_bt": str(holdout.max_bt),
        "control_output_tok_s": _format_float(control.output_tok_s),
        "holdout_output_tok_s": _format_float(holdout.output_tok_s),
        "output_ratio": _format_float(output_ratio),
        "control_unique_iterations": str(control.unique_iterations),
        "holdout_unique_iterations": str(holdout.unique_iterations),
        "unique_iteration_delta": str(holdout.unique_iterations - control.unique_iterations),
        "control_mixed_iterations": _format_sequence(control.mixed_iterations),
        "holdout_mixed_iterations": _format_sequence(holdout.mixed_iterations),
        "mixed_iteration_delta": str(len(holdout.mixed_iterations) - len(control.mixed_iterations)),
        "control_pure_decode_iterations": str(control.phase_counts["pure_decode"]),
        "holdout_pure_decode_iterations": str(holdout.phase_counts["pure_decode"]),
        "pure_decode_iteration_delta": str(holdout.phase_counts["pure_decode"] - control.phase_counts["pure_decode"]),
        "control_first_pure_iteration": str(control.first_pure_iteration),
        "holdout_first_pure_iteration": str(holdout.first_pure_iteration),
        "control_tail_decode_tokens": _format_sequence(control.tail_decode_tokens),
        "holdout_tail_decode_tokens": _format_sequence(holdout.tail_decode_tokens),
        "control_scheduled_p99": str(control.scheduled_stats["p99"]),
        "holdout_scheduled_p99": str(holdout.scheduled_stats["p99"]),
        "control_scheduled_max": str(control.scheduled_stats["max"]),
        "holdout_scheduled_max": str(holdout.scheduled_stats["max"]),
        "holdout_max_fill": _format_float(holdout_max_fill),
        "control_forward_token_max": str(control.forward_token_stats["max"]),
        "holdout_forward_token_max": str(holdout.forward_token_stats["max"]),
        "forward_elapsed_p50_delta_ns": _delta(holdout.forward_elapsed_stats, control.forward_elapsed_stats, "p50"),
        "forward_elapsed_p95_delta_ns": _delta(holdout.forward_elapsed_stats, control.forward_elapsed_stats, "p95"),
        "forward_elapsed_p99_delta_ns": _delta(holdout.forward_elapsed_stats, control.forward_elapsed_stats, "p99"),
        "forward_elapsed_max_delta_ns": _delta(holdout.forward_elapsed_stats, control.forward_elapsed_stats, "max"),
        "iteration_elapsed_p50_delta_ns": _delta(
            holdout.iteration_elapsed_stats, control.iteration_elapsed_stats, "p50"
        ),
        "iteration_elapsed_p95_delta_ns": _delta(
            holdout.iteration_elapsed_stats, control.iteration_elapsed_stats, "p95"
        ),
        "iteration_elapsed_p99_delta_ns": _delta(
            holdout.iteration_elapsed_stats, control.iteration_elapsed_stats, "p99"
        ),
        "iteration_elapsed_max_delta_ns": _delta(
            holdout.iteration_elapsed_stats, control.iteration_elapsed_stats, "max"
        ),
        "overhead_p50_delta_ns": _delta(holdout.overhead_stats, control.overhead_stats, "p50"),
        "overhead_p95_delta_ns": _delta(holdout.overhead_stats, control.overhead_stats, "p95"),
        "overhead_p99_delta_ns": _delta(holdout.overhead_stats, control.overhead_stats, "p99"),
        "overhead_max_delta_ns": _delta(holdout.overhead_stats, control.overhead_stats, "max"),
        "control_active_request_max": str(control.active_request_max),
        "holdout_active_request_max": str(holdout.active_request_max),
        "control_scheduled_new_req_sum": str(control.scheduled_new_req_sum),
        "holdout_scheduled_new_req_sum": str(holdout.scheduled_new_req_sum),
        "control_scheduled_cached_req_sum": str(control.scheduled_cached_req_sum),
        "holdout_scheduled_cached_req_sum": str(holdout.scheduled_cached_req_sum),
        "control_scheduled_resumed_req_sum": str(control.scheduled_resumed_req_sum),
        "holdout_scheduled_resumed_req_sum": str(holdout.scheduled_resumed_req_sum),
        "verdict": VERDICT,
        "mechanism_conclusion": MECHANISM_CONCLUSION,
        "default_readiness": DEFAULT_READINESS,
        "diagnostic_only": _bool_text(True),
        "valid_for_default": _bool_text(False),
        "perf_database": _bool_text(False),
    }


def analyze_deeper_trace_partial_audit(input_root: Path) -> list[dict[str, str]]:
    """Return one pair-level partial-only audit row for the Phase286 trace pair."""
    summaries = {
        scenario: _case_summary(input_root, spec)
        for scenario, spec in SCENARIOS.items()
    }
    row = _pair_row(summaries[CONTROL_SCENARIO], summaries[HOLDOUT_SCENARIO])
    return [row]


def write_deeper_trace_partial_audit_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_deeper_trace_partial_audit_doc(path: Path, rows: list[dict[str, str]]) -> None:
    if len(rows) != 1:
        raise ValueError("Phase287 document expects exactly one row")
    row = rows[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Phase287 Deeper Trace Partial Audit

This diagnostic audit compares the tp8 12k2k deeper trace control/holdout pair.

| Item | Value |
|---|---|
| Pair | {row["pair_key"]} |
| Output ratio | {row["output_ratio"]} |
| Unique iteration delta | {row["unique_iteration_delta"]} |
| Mixed iteration delta | {row["mixed_iteration_delta"]} |
| Control mixed iterations | {row["control_mixed_iterations"]} |
| Holdout mixed iterations | {row["holdout_mixed_iterations"]} |
| Control tail decode tokens | {row["control_tail_decode_tokens"]} |
| Holdout tail decode tokens | {row["holdout_tail_decode_tokens"]} |
| Holdout max fill | {row["holdout_max_fill"]} |
| Verdict | {row["verdict"]} |
| Mechanism | {row["mechanism_conclusion"]} |
| Default AIC | {row["default_readiness"]} |

The conclusion remains diagnostic-only: boundary, mixed iteration count, and rank-max timing overhead explain only part of the observed throughput delta. The holdout still schedules at most {row["holdout_scheduled_max"]} tokens against a configured max budget of {row["holdout_bt"]}, so the configured budget is not a linear runtime cost.

Flags: diagnostic_only={row["diagnostic_only"]}, valid_for_default={row["valid_for_default"]}, perf_database={row["perf_database"]}.
"""
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_CSV_OUT)
    parser.add_argument("--doc-out", type=Path, default=DEFAULT_DOC_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = analyze_deeper_trace_partial_audit(args.input_root)
    write_deeper_trace_partial_audit_csv(args.out, rows)
    write_deeper_trace_partial_audit_doc(args.doc_out, rows)
    print(f"wrote {args.out}")
    print(f"wrote {args.doc_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
