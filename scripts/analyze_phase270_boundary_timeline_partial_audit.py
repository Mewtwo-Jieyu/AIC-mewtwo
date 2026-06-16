#!/usr/bin/env python3
"""Audit Phase269 boundary/timeline signals as partial-only evidence."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple


SOURCE = "phase270_boundary_timeline_partial_audit"
TRACE_SOURCE = "phase234_vllm_scheduler_trace"
PHASE264_SOURCE = "phase264_phase_mix_decode_batch"
PHASE267_SOURCE = "phase267_mixed_phase_partial_audit"
CONCLUSION = "boundary_timeline_partial_only"
NEXT_PARTIAL = "boundary_timeline_partial_only"
NEXT_DEEPER = "deeper_trace_for_tp8_12k2k"
DEFAULT_READINESS = "No-Go"
VALID_PHASES = {"prefill", "mixed", "pure_decode"}

FIELDNAMES = [
    "source",
    "pair_key",
    "topology_key",
    "shape_key",
    "control_scenario",
    "holdout_scenario",
    "control_boundary",
    "holdout_boundary",
    "control_mixed_iterations",
    "holdout_mixed_iterations",
    "control_first_pure_iteration",
    "holdout_first_pure_iteration",
    "total_iteration_delta",
    "tail_decode_tokens_control",
    "tail_decode_tokens_holdout",
    "boundary_verdict",
    "conclusion",
    "next_mechanism",
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
)


class ScenarioSpec(NamedTuple):
    scenario: str
    topology_key: str
    shape_key: str
    role: str
    control_bt: int
    max_bt: int
    tp: int
    dp: int
    ep: int
    total_iterations: int
    mixed_iterations: tuple[int, ...]
    first_pure_iteration: int
    tail_decode_tokens: tuple[int, ...]
    context_iterations: tuple[tuple[int, int, int, int], ...]


SCENARIOS = [
    ScenarioSpec(
        "tp8ep8-4k2k-bt4000",
        "tp8_dp1_ep8",
        "isl4000_osl2000_batch128",
        "control",
        4000,
        4000,
        8,
        1,
        8,
        2002,
        (2,),
        1,
        (128, 128, 128, 127, 127),
        ((0, 4000, 0, 4000), (2, 2032, 1, 2033)),
    ),
    ScenarioSpec(
        "tp8ep8-4k2k-bt65536",
        "tp8_dp1_ep8",
        "isl4000_osl2000_batch128",
        "holdout",
        4000,
        65536,
        8,
        1,
        8,
        2003,
        (2, 3),
        1,
        (128, 128, 127, 127, 30),
        ((0, 4000, 0, 4000), (2, 1552, 1, 1553), (3, 480, 98, 578)),
    ),
    ScenarioSpec(
        "tp4dp2ep8-4k2k-bt4000",
        "tp4_dp2_ep8",
        "isl4000_osl2000_batch128",
        "control",
        4000,
        4000,
        4,
        2,
        8,
        2002,
        (1, 2),
        3,
        (128, 128, 128, 126, 111),
        ((0, 8000, 0, 8000), (1, 240, 2, 242), (2, 1776, 17, 1793)),
    ),
    ScenarioSpec(
        "tp4dp2ep8-4k2k-bt65536",
        "tp4_dp2_ep8",
        "isl4000_osl2000_batch128",
        "holdout",
        4000,
        65536,
        4,
        2,
        8,
        2002,
        (2,),
        1,
        (128, 128, 128, 111, 111),
        ((0, 8240, 0, 8240), (2, 1776, 17, 1793)),
    ),
    ScenarioSpec(
        "tp8ep8-12k2k-bt12000",
        "tp8_dp1_ep8",
        "isl12000_osl2000_batch128",
        "control",
        12000,
        12000,
        8,
        1,
        8,
        2008,
        (2, 4, 7, 8),
        1,
        (62, 30, 30, 30, 20),
        ((0, 12000, 0, 12000), (2, 1040, 1, 1041), (4, 512, 66, 578), (7, 160, 98, 258), (8, 320, 108, 428)),
    ),
    ScenarioSpec(
        "tp8ep8-12k2k-bt65536",
        "tp8_dp1_ep8",
        "isl12000_osl2000_batch128",
        "holdout",
        12000,
        65536,
        8,
        1,
        8,
        2004,
        (2, 3, 4),
        1,
        (128, 127, 127, 48, 30),
        ((0, 12000, 0, 12000), (2, 1264, 1, 1265), (3, 288, 80, 368), (4, 480, 98, 578)),
    ),
    ScenarioSpec(
        "tp4dp2ep8-12k2k-bt12000",
        "tp4_dp2_ep8",
        "isl12000_osl2000_batch128",
        "control",
        12000,
        12000,
        4,
        2,
        8,
        2002,
        (1, 2),
        3,
        (128, 128, 128, 126, 72),
        ((0, 24000, 0, 24000), (1, 864, 2, 866), (2, 1152, 56, 1208)),
    ),
    ScenarioSpec(
        "tp4dp2ep8-12k2k-bt65536",
        "tp4_dp2_ep8",
        "isl12000_osl2000_batch128",
        "holdout",
        12000,
        65536,
        4,
        2,
        8,
        2003,
        (2, 3),
        1,
        (128, 128, 126, 126, 15),
        ((0, 24000, 0, 24000), (2, 1776, 2, 1778), (3, 240, 113, 353)),
    ),
]

PAIR_SPECS = [
    ("tp8_dp1_ep8", "isl4000_osl2000_batch128", 4000, "boundary_explains", NEXT_PARTIAL),
    ("tp4_dp2_ep8", "isl4000_osl2000_batch128", 4000, "boundary_explains", NEXT_PARTIAL),
    ("tp8_dp1_ep8", "isl12000_osl2000_batch128", 12000, "needs_deeper_trace", NEXT_DEEPER),
    ("tp4_dp2_ep8", "isl12000_osl2000_batch128", 12000, "boundary_partial", NEXT_PARTIAL),
]


def _pair_key(topology_key: str, shape_key: str, control_bt: int) -> str:
    return f"{topology_key}:{shape_key}:control_bt{control_bt}:holdout_bt65536"


PHASE267_MIXED_VERDICTS = {
    _pair_key("tp8_dp1_ep8", "isl4000_osl2000_batch128", 4000): "partial",
    _pair_key("tp4_dp2_ep8", "isl4000_osl2000_batch128", 4000): "explains",
    _pair_key("tp8_dp1_ep8", "isl12000_osl2000_batch128", 12000): "partial",
    _pair_key("tp4_dp2_ep8", "isl12000_osl2000_batch128", 12000): "does_not_explain",
}


def _csv_ints(values: tuple[int, ...] | list[int]) -> str:
    return ",".join(str(value) for value in values)


def _payload_signature(row: dict[str, object]) -> tuple[object, ...]:
    return tuple(row[field] for field in PAYLOAD_SIGNATURE_FIELDS)


def _required_int(row: dict[str, object], field: str, path: Path) -> int:
    value = row.get(field)
    if not isinstance(value, int):
        raise ValueError(f"{field} must be int in {path}: {value!r}")
    if value < 0:
        raise ValueError(f"{field} must be non-negative in {path}: {value!r}")
    return value


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected object json: {path}")
    return data


def _validate_phase264_csv(path: Path | str) -> None:
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 8:
        raise ValueError(f"phase264 csv must contain 8 rows: got {len(rows)}")
    expected = [spec.scenario for spec in SCENARIOS]
    if [row.get("scenario") for row in rows] != expected:
        raise ValueError("phase264 scenario order mismatch")
    for row in rows:
        if row.get("source") != PHASE264_SOURCE:
            raise ValueError("phase264 source mismatch")
        if row.get("default_readiness") != DEFAULT_READINESS:
            raise ValueError("phase264 default_readiness mismatch")
        if row.get("diagnostic_only") != "true" or row.get("valid_for_default") != "false" or row.get("perf_database") != "false":
            raise ValueError("phase264 flag mismatch")


def _validate_phase267_csv(path: Path | str) -> None:
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 4:
        raise ValueError(f"phase267 csv must contain 4 rows: got {len(rows)}")
    if [row.get("pair_key") for row in rows] != list(PHASE267_MIXED_VERDICTS):
        raise ValueError("phase267 pair key order mismatch")
    for row in rows:
        if row.get("source") != PHASE267_SOURCE:
            raise ValueError("phase267 source mismatch")
        if row.get("verdict") != PHASE267_MIXED_VERDICTS[row["pair_key"]]:
            raise ValueError(f"phase267 verdict mismatch: {row['pair_key']}")
        if row.get("default_readiness") != DEFAULT_READINESS:
            raise ValueError("phase267 default_readiness mismatch")
        if row.get("diagnostic_only") != "true" or row.get("valid_for_default") != "false" or row.get("perf_database") != "false":
            raise ValueError("phase267 flag mismatch")


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


def _read_trace_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"trace row must be object in {path}")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty trace: {path}")
    return rows


def _aggregate_trace(input_root: Path, spec: ScenarioSpec) -> dict[str, object]:
    case_dir = input_root / spec.scenario
    bench_path = case_dir / "bench_result.json"
    result_path = case_dir / "phase234_result.json"
    trace_path = case_dir / "scheduler_trace.jsonl"
    for path in (bench_path, result_path, trace_path):
        if not path.exists():
            raise ValueError(f"missing artifact: {path}")
    bench = _read_json(bench_path)
    result = _read_json(result_path)
    if result.get("scenario") != spec.scenario:
        raise ValueError(f"result scenario mismatch in {result_path}")
    if result.get("diagnostic_only") is not True or result.get("valid_for_default") is not False or result.get("perf_database") is not False:
        raise ValueError(f"result flag mismatch in {result_path}")
    if not isinstance(bench.get("output_tok_s"), (int, float)):
        raise ValueError(f"bench output_tok_s mismatch in {bench_path}")

    by_iteration: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in _read_trace_rows(trace_path):
        _validate_trace_row(row, spec, trace_path)
        by_iteration[_required_int(row, "iteration", trace_path)].append(row)
    iteration_ids = sorted(by_iteration)
    if iteration_ids != list(range(iteration_ids[0], iteration_ids[-1] + 1)):
        raise ValueError(f"trace iterations must be contiguous in {trace_path}")

    aggregated: list[dict[str, int | str]] = []
    max_distinct_payloads = 0
    for iteration in iteration_ids:
        iteration_rows = by_iteration[iteration]
        signatures = {_payload_signature(row) for row in iteration_rows}
        max_distinct_payloads = max(max_distinct_payloads, len(signatures))
        if len(signatures) > spec.dp:
            raise ValueError(f"trace has more distinct DP payloads than dp in {trace_path}")
        raw_context = sum(_required_int(row, "scheduled_context_tokens", trace_path) for row in iteration_rows)
        raw_decode = sum(_required_int(row, "scheduled_decode_tokens", trace_path) for row in iteration_rows)
        raw_total = sum(_required_int(row, "scheduled_total_tokens", trace_path) for row in iteration_rows)
        if raw_context % spec.tp or raw_decode % spec.tp or raw_total % spec.tp:
            raise ValueError(f"trace scheduled token raw sum must be divisible by tp in {trace_path}")
        context = raw_context // spec.tp
        decode = raw_decode // spec.tp
        total = raw_total // spec.tp
        if context and decode:
            phase = "mixed"
        elif context:
            phase = "prefill"
        elif decode:
            phase = "pure_decode"
        else:
            raise ValueError(f"aggregate scheduled tokens must be positive in {trace_path}")
        aggregated.append({"iteration": iteration, "phase": phase, "context": context, "decode": decode, "total": total})

    mixed_iterations = tuple(int(row["iteration"]) for row in aggregated if row["phase"] == "mixed")
    pure_iterations = [int(row["iteration"]) for row in aggregated if row["phase"] == "pure_decode"]
    tail_decode_tokens = tuple(int(row["decode"]) for row in aggregated[-5:])
    context_iterations = tuple(
        (int(row["iteration"]), int(row["context"]), int(row["decode"]), int(row["total"]))
        for row in aggregated
        if int(row["context"]) > 0
    )
    if len(aggregated) != spec.total_iterations:
        raise ValueError(f"total iterations mismatch for {spec.scenario}")
    if mixed_iterations != spec.mixed_iterations:
        raise ValueError(f"mixed iteration sequence mismatch for {spec.scenario}")
    if not pure_iterations or pure_iterations[0] != spec.first_pure_iteration:
        raise ValueError(f"first pure iteration mismatch for {spec.scenario}")
    if tail_decode_tokens != spec.tail_decode_tokens:
        raise ValueError(f"tail decode mismatch for {spec.scenario}")
    if context_iterations != spec.context_iterations:
        raise ValueError(f"context drain pattern mismatch for {spec.scenario}")
    return {
        "scenario": spec.scenario,
        "total_iterations": len(aggregated),
        "mixed_iterations": mixed_iterations,
        "first_pure_iteration": pure_iterations[0],
        "tail_decode_tokens": tail_decode_tokens,
        "context_iterations": context_iterations,
        "max_distinct_payloads": max_distinct_payloads,
    }


def _boundary_summary(timeline: dict[str, object]) -> str:
    return (
        f"mixed={_csv_ints(timeline['mixed_iterations'])};"
        f"first_pure={timeline['first_pure_iteration']};"
        f"tail={_csv_ints(timeline['tail_decode_tokens'])};"
        f"iters={timeline['total_iterations']}"
    )


def analyze_boundary_timeline_partial_audit(
    input_root: Path | str,
    phase264_csv: Path | str,
    phase267_csv: Path | str,
) -> list[dict[str, str]]:
    _validate_phase264_csv(phase264_csv)
    _validate_phase267_csv(phase267_csv)
    root = Path(input_root)
    timelines = {spec.scenario: _aggregate_trace(root, spec) for spec in SCENARIOS}

    by_role: dict[tuple[str, str, str], ScenarioSpec] = {}
    for spec in SCENARIOS:
        by_role[(spec.topology_key, spec.shape_key, spec.role)] = spec
    rows: list[dict[str, str]] = []
    for topology_key, shape_key, control_bt, verdict, next_mechanism in PAIR_SPECS:
        control = by_role[(topology_key, shape_key, "control")]
        holdout = by_role[(topology_key, shape_key, "holdout")]
        control_timeline = timelines[control.scenario]
        holdout_timeline = timelines[holdout.scenario]
        row = {
            "source": SOURCE,
            "pair_key": _pair_key(topology_key, shape_key, control_bt),
            "topology_key": topology_key,
            "shape_key": shape_key,
            "control_scenario": control.scenario,
            "holdout_scenario": holdout.scenario,
            "control_boundary": _boundary_summary(control_timeline),
            "holdout_boundary": _boundary_summary(holdout_timeline),
            "control_mixed_iterations": _csv_ints(control_timeline["mixed_iterations"]),
            "holdout_mixed_iterations": _csv_ints(holdout_timeline["mixed_iterations"]),
            "control_first_pure_iteration": str(control_timeline["first_pure_iteration"]),
            "holdout_first_pure_iteration": str(holdout_timeline["first_pure_iteration"]),
            "total_iteration_delta": str(int(holdout_timeline["total_iterations"]) - int(control_timeline["total_iterations"])),
            "tail_decode_tokens_control": _csv_ints(control_timeline["tail_decode_tokens"]),
            "tail_decode_tokens_holdout": _csv_ints(holdout_timeline["tail_decode_tokens"]),
            "boundary_verdict": verdict,
            "conclusion": CONCLUSION,
            "next_mechanism": next_mechanism,
            "default_readiness": DEFAULT_READINESS,
            "diagnostic_only": "true",
            "valid_for_default": "false",
            "perf_database": "false",
        }
        rows.append(row)
    _validate_output_rows(rows)
    return rows


def _validate_output_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 4:
        raise ValueError(f"phase270 audit must contain 4 rows: got {len(rows)}")
    expected_keys = [_pair_key(topology, shape, control_bt) for topology, shape, control_bt, _verdict, _next in PAIR_SPECS]
    if [row.get("pair_key") for row in rows] != expected_keys:
        raise ValueError("phase270 pair key order mismatch")
    verdicts = [row.get("boundary_verdict") for row in rows]
    if verdicts.count("boundary_explains") != 2 or verdicts.count("boundary_partial") != 1 or verdicts.count("needs_deeper_trace") != 1:
        raise ValueError(f"unexpected boundary verdict distribution: {verdicts}")
    for row in rows:
        if row.get("source") != SOURCE:
            raise ValueError("phase270 source mismatch")
        if row.get("conclusion") != CONCLUSION:
            raise ValueError("phase270 conclusion mismatch")
        if row.get("default_readiness") != DEFAULT_READINESS:
            raise ValueError("phase270 default_readiness mismatch")
        if row.get("diagnostic_only") != "true" or row.get("valid_for_default") != "false" or row.get("perf_database") != "false":
            raise ValueError("phase270 flag mismatch")


def write_boundary_timeline_partial_audit_csv(path: Path | str, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with out.open(newline="", encoding="utf-8") as f:
        _validate_output_rows(list(csv.DictReader(f)))


def write_boundary_timeline_partial_audit_doc(path: Path | str, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase270: Boundary / Timeline Partial Audit",
        "",
        "## Decision",
        "",
        "| Item | Result |",
        "|---|---|",
        f"| Conclusion | {CONCLUSION} |",
        "| Boundary / timeline | Partial-only, not a model |",
        "| tp8 12k2k needs deeper trace | true |",
        "| Default AIC | No-Go |",
        "| PerfDatabase | No-Go |",
        "| Boundary flags | diagnostic_only=true valid_for_default=false perf_database=false |",
        "",
        "## Pair Audit",
        "",
        "| Pair | control boundary | holdout boundary | total iteration delta | verdict | next |",
        "|---|---|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['pair_key']} | {row['control_boundary']} | {row['holdout_boundary']} | "
            f"{row['total_iteration_delta']} | {row['boundary_verdict']} | {row['next_mechanism']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Phase270 makes the Phase269 boundary/timeline readout machine-checkable.",
            "Boundary timing explains both 4k2k pairs, but tp8 12k2k moves in the wrong direction and cannot be explained by this layer.",
            "tp4dp2 12k2k has useful boundary signal, but the signal is not strong enough to explain the full throughput ratio.",
            "The next diagnostic step is deeper trace instrumentation for tp8 12k2k before any default model discussion.",
            "This remains diagnostic-only evidence. Do not wire it into VLLMBackend.run_agg, default AIC, or PerfDatabase.",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("docs/iter_gap_investigation/phase234_scheduler_trace"))
    parser.add_argument("--phase264", type=Path, default=Path("docs/iter_gap_investigation/phase264_phase_mix_decode_batch.csv"))
    parser.add_argument("--phase267", type=Path, default=Path("docs/iter_gap_investigation/phase267_mixed_phase_partial_audit.csv"))
    parser.add_argument("--out", type=Path, default=Path("docs/iter_gap_investigation/phase270_boundary_timeline_partial_audit.csv"))
    parser.add_argument("--doc-out", type=Path, default=Path("docs/iter_gap_investigation/phase270_boundary_timeline_partial_audit.md"))
    args = parser.parse_args()

    rows = analyze_boundary_timeline_partial_audit(args.input_root, args.phase264, args.phase267)
    write_boundary_timeline_partial_audit_csv(args.out, rows)
    write_boundary_timeline_partial_audit_doc(args.doc_out, rows)
    print(f"wrote_phase270_boundary_timeline_partial_audit={args.out}")
    print(f"phase270_rows={len(rows)}")
    print("boundary_verdict_explains=2")
    print("boundary_verdict_partial=1")
    print("boundary_verdict_needs_deeper_trace=1")
    print(f"default_readiness={DEFAULT_READINESS}")


if __name__ == "__main__":
    main()
