#!/usr/bin/env python3
"""Summarize the Phase319/321 runner lifecycle pair canary."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple


SOURCE = "phase323_runner_lifecycle_pair_canary"
SOURCE_SHA = "13cbb5c650fd622e2655fed844d11464b5c2371c"
WORKER = "worker-wrzh8"
VERDICT = "runner_lifecycle_pair_canary_pass"
EVIDENCE_STATUS = "not_replacement_evidence"
CONTROL_SCENARIO = "tp8ep8-12k2k-bt12000"
HOLDOUT_SCENARIO = "tp8ep8-12k2k-bt65536"
SHAPE_KEY = "isl12000_osl2000_batch128"
TOPOLOGY_KEY = "tp8_dp1_ep8"
DEFAULT_ARTIFACT_ROOT = Path("/private/tmp/aic_phase323_runner_lifecycle_pair_canary_13cbb5c")
DEFAULT_CSV_OUT = Path("docs/iter_gap_investigation/phase323_runner_lifecycle_pair_canary.csv")
DEFAULT_DOC_OUT = Path("docs/iter_gap_investigation/phase323_runner_lifecycle_pair_canary.md")

FIELDNAMES = [
    "source",
    "source_sha",
    "worker",
    "control_scenario",
    "holdout_scenario",
    "control_ok_requests",
    "control_failed_requests",
    "holdout_ok_requests",
    "holdout_failed_requests",
    "control_output_tok_s",
    "holdout_output_tok_s",
    "output_ratio",
    "control_trace_rows",
    "holdout_trace_rows",
    "control_unique_iterations",
    "holdout_unique_iterations",
    "control_prefill_iterations",
    "control_mixed_iterations",
    "control_pure_decode_iterations",
    "holdout_prefill_iterations",
    "holdout_mixed_iterations",
    "holdout_pure_decode_iterations",
    "control_max_bt",
    "holdout_max_bt",
    "control_lock_released",
    "holdout_lock_released",
    "lock_released",
    "control_benchmark_failure_count",
    "holdout_benchmark_failure_count",
    "benchmark_failure_count",
    "control_cleanup_trap_done",
    "holdout_cleanup_trap_done",
    "cleanup_trap_done",
    "verdict",
    "evidence_status",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

REQUIRED_TRACE_FIELDS = set(
    """
    source scenario iteration phase scheduled_context_tokens scheduled_decode_tokens
    scheduled_total_tokens scheduled_context_reqs scheduled_decode_reqs scheduled_total_reqs
    max_num_batched_tokens max_num_seqs forward_token_count tp dp ep topology_key shape_key
    iteration_start_ns forward_start_ns forward_end_ns iteration_end_ns iteration_elapsed_ns
    forward_elapsed_ns active_request_count scheduled_new_req_count scheduled_cached_req_count
    scheduled_resumed_req_count finished_req_count diagnostic_only valid_for_default perf_database
    """.split()
)
COUNT_FIELDS = [
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
]
PAYLOAD_KEYS = [
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
]


class ScenarioSummary(NamedTuple):
    scenario: str
    ok_requests: int
    failed_requests: int
    output_tok_s: float
    records: int
    trace_rows: int
    unique_iterations: int
    prefill_iterations: int
    mixed_iterations: int
    pure_decode_iterations: int
    max_bt: int
    lock_released: bool
    benchmark_failure_count: int
    cleanup_trap_done: bool


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _resolve_scenario_dir(artifact_root: Path, scenario: str) -> Path:
    direct = artifact_root / scenario
    nested = artifact_root / "docs/iter_gap_investigation/phase274_deeper_scheduler_trace" / scenario
    for candidate in (direct, nested):
        if (candidate / "bench_result.json").exists():
            return candidate
    raise FileNotFoundError(f"missing scenario artifact: {scenario} under {artifact_root}")


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_lines(path: Path) -> list[str]:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8").splitlines()


def _phase_counts(rows: list[dict[str, object]], scenario: str) -> Counter[str]:
    phase_by_iter: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        phase_by_iter[int(row["iteration"])].add(str(row["phase"]))
    if any(len(phases) != 1 for phases in phase_by_iter.values()):
        raise ValueError(f"phase divergence: {scenario}")
    return Counter(next(iter(phases)) for phases in phase_by_iter.values())


def _validate_trace(rows: list[dict[str, object]], scenario: str, expected_max_bt: int) -> tuple[int, Counter[str]]:
    if not rows:
        raise ValueError(f"empty trace: {scenario}")
    missing = set().union(*(REQUIRED_TRACE_FIELDS - set(row) for row in rows))
    if missing:
        raise ValueError(f"missing trace schema for {scenario}: {sorted(missing)}")

    flags = {(row["diagnostic_only"], row["valid_for_default"], row["perf_database"]) for row in rows}
    if flags != {(True, False, False)}:
        raise ValueError(f"flags mismatch: {scenario}")
    if {row["source"] for row in rows} != {"phase274_deeper_scheduler_trace"}:
        raise ValueError(f"source mismatch: {scenario}")
    if {row["scenario"] for row in rows} != {scenario}:
        raise ValueError(f"scenario mismatch: {scenario}")
    if {row["tp"] for row in rows} != {8} or {row["dp"] for row in rows} != {1} or {row["ep"] for row in rows} != {8}:
        raise ValueError(f"topology shape mismatch: {scenario}")
    if {row["shape_key"] for row in rows} != {SHAPE_KEY} or {row["topology_key"] for row in rows} != {TOPOLOGY_KEY}:
        raise ValueError(f"key mismatch: {scenario}")
    if {row["max_num_batched_tokens"] for row in rows} != {expected_max_bt}:
        raise ValueError(f"max_bt mismatch: {scenario}")

    iterations = sorted({int(row["iteration"]) for row in rows})
    if iterations != list(range(iterations[0], iterations[-1] + 1)):
        raise ValueError(f"iteration continuity mismatch: {scenario}")
    worker_counts = Counter(int(row["iteration"]) for row in rows)
    if min(worker_counts.values()) != 8 or max(worker_counts.values()) != 8:
        raise ValueError(f"worker row count mismatch: {scenario}")

    payload_by_iter: dict[int, set[tuple[tuple[str, object], ...]]] = defaultdict(set)
    for row in rows:
        payload_by_iter[int(row["iteration"])].add(tuple((key, row[key]) for key in PAYLOAD_KEYS))
        if row["scheduled_total_tokens"] != row["scheduled_context_tokens"] + row["scheduled_decode_tokens"]:
            raise ValueError(f"token sum mismatch: {scenario}")
        if row["scheduled_total_reqs"] != row["scheduled_context_reqs"] + row["scheduled_decode_reqs"]:
            raise ValueError(f"request sum mismatch: {scenario}")
        if not row["iteration_start_ns"] <= row["forward_start_ns"] <= row["forward_end_ns"] <= row["iteration_end_ns"]:
            raise ValueError(f"timing order mismatch: {scenario}")
        if row["iteration_elapsed_ns"] < 0 or row["forward_elapsed_ns"] < 0:
            raise ValueError(f"negative timing mismatch: {scenario}")
        if any(row[key] < 0 for key in COUNT_FIELDS):
            raise ValueError(f"negative count mismatch: {scenario}")
    if any(len(payloads) != 1 for payloads in payload_by_iter.values()):
        raise ValueError(f"worker payload divergence: {scenario}")

    return len(iterations), _phase_counts(rows, scenario)


def _validate_result(result: dict[str, object], scenario: str) -> None:
    if result.get("scenario") != scenario:
        raise ValueError(f"result scenario mismatch: {scenario}")
    if result.get("shape_key") != SHAPE_KEY or result.get("topology_key") != TOPOLOGY_KEY:
        raise ValueError(f"result key mismatch: {scenario}")
    if (
        result.get("diagnostic_only") is not True
        or result.get("valid_for_default") is not False
        or result.get("perf_database") is not False
    ):
        raise ValueError(f"result flags mismatch: {scenario}")


def _summarize_scenario(artifact_root: Path, scenario: str, expected_max_bt: int) -> ScenarioSummary:
    directory = _resolve_scenario_dir(artifact_root, scenario)
    bench = _load_json(directory / "bench_result.json")
    records = _read_lines(directory / "bench_records.jsonl")
    rows = [json.loads(line) for line in _read_lines(directory / "deeper_scheduler_trace.jsonl")]
    result = _load_json(directory / "phase274_result.json")
    _validate_result(result, scenario)

    if bench.get("ok_requests") != 128 or bench.get("failed_requests") != 0:
        raise ValueError(f"benchmark request mismatch: {scenario}")
    if len(records) != 128:
        raise ValueError(f"records mismatch: {scenario}")
    unique_iterations, phase_counts = _validate_trace(rows, scenario, expected_max_bt)

    run_log = _read_lines(directory / f"run_one_{scenario}.log")
    cleanup_log = _read_lines(directory / f"cleanup_{scenario}.log")
    if not any("run_id=" in line for line in run_log) or not any("runner_pid=" in line for line in run_log):
        raise ValueError(f"run identity missing: {scenario}")
    cleanup_trap_done = any("cleanup_trap_done" in line for line in cleanup_log)
    if not cleanup_trap_done:
        raise ValueError(f"cleanup trap missing: {scenario}")
    lock_released = not (directory / ".phase274_run.lock").exists()
    if not lock_released:
        raise ValueError(f"run lock still exists: {scenario}")
    benchmark_failure_count = len(list(directory.glob("benchmark_failure_*")))
    if benchmark_failure_count != 0:
        raise ValueError(f"benchmark failure artifacts present: {scenario}")
    if (directory / "gpu_compute_apps_after.txt").stat().st_size != 0:
        raise ValueError(f"gpu after file not empty: {scenario}")
    drain_text = (directory / "gpu_compute_apps_drain.log").read_text(encoding="utf-8").lower()
    if "stable" not in drain_text:
        raise ValueError(f"gpu drain not stable: {scenario}")

    return ScenarioSummary(
        scenario=scenario,
        ok_requests=int(bench["ok_requests"]),
        failed_requests=int(bench["failed_requests"]),
        output_tok_s=float(bench["output_tok_s"]),
        records=len(records),
        trace_rows=len(rows),
        unique_iterations=unique_iterations,
        prefill_iterations=phase_counts.get("prefill", 0),
        mixed_iterations=phase_counts.get("mixed", 0),
        pure_decode_iterations=phase_counts.get("pure_decode", 0),
        max_bt=expected_max_bt,
        lock_released=lock_released,
        benchmark_failure_count=benchmark_failure_count,
        cleanup_trap_done=cleanup_trap_done,
    )


def analyze_runner_lifecycle_pair_canary(artifact_root: Path) -> list[dict[str, str]]:
    """Return one pair-level row for the worker-wrzh8 lifecycle canary."""
    control = _summarize_scenario(artifact_root, CONTROL_SCENARIO, 12000)
    holdout = _summarize_scenario(artifact_root, HOLDOUT_SCENARIO, 65536)
    output_ratio = holdout.output_tok_s / control.output_tok_s
    return [
        {
            "source": SOURCE,
            "source_sha": SOURCE_SHA,
            "worker": WORKER,
            "control_scenario": control.scenario,
            "holdout_scenario": holdout.scenario,
            "control_ok_requests": str(control.ok_requests),
            "control_failed_requests": str(control.failed_requests),
            "holdout_ok_requests": str(holdout.ok_requests),
            "holdout_failed_requests": str(holdout.failed_requests),
            "control_output_tok_s": f"{control.output_tok_s:.6f}",
            "holdout_output_tok_s": f"{holdout.output_tok_s:.6f}",
            "output_ratio": f"{output_ratio:.6f}",
            "control_trace_rows": str(control.trace_rows),
            "holdout_trace_rows": str(holdout.trace_rows),
            "control_unique_iterations": str(control.unique_iterations),
            "holdout_unique_iterations": str(holdout.unique_iterations),
            "control_prefill_iterations": str(control.prefill_iterations),
            "control_mixed_iterations": str(control.mixed_iterations),
            "control_pure_decode_iterations": str(control.pure_decode_iterations),
            "holdout_prefill_iterations": str(holdout.prefill_iterations),
            "holdout_mixed_iterations": str(holdout.mixed_iterations),
            "holdout_pure_decode_iterations": str(holdout.pure_decode_iterations),
            "control_max_bt": str(control.max_bt),
            "holdout_max_bt": str(holdout.max_bt),
            "control_lock_released": _bool_text(control.lock_released),
            "holdout_lock_released": _bool_text(holdout.lock_released),
            "lock_released": _bool_text(control.lock_released and holdout.lock_released),
            "control_benchmark_failure_count": str(control.benchmark_failure_count),
            "holdout_benchmark_failure_count": str(holdout.benchmark_failure_count),
            "benchmark_failure_count": str(control.benchmark_failure_count + holdout.benchmark_failure_count),
            "control_cleanup_trap_done": _bool_text(control.cleanup_trap_done),
            "holdout_cleanup_trap_done": _bool_text(holdout.cleanup_trap_done),
            "cleanup_trap_done": _bool_text(control.cleanup_trap_done and holdout.cleanup_trap_done),
            "verdict": VERDICT,
            "evidence_status": EVIDENCE_STATUS,
            "diagnostic_only": "true",
            "valid_for_default": "false",
            "perf_database": "false",
        }
    ]


def write_runner_lifecycle_pair_canary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if len(rows) != 1:
        raise ValueError("Phase323 CSV expects exactly one row")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_runner_lifecycle_pair_canary_doc(path: Path, rows: list[dict[str, str]]) -> None:
    if len(rows) != 1:
        raise ValueError("Phase323 document expects exactly one row")
    row = rows[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Phase323 Runner Lifecycle Pair Canary

This diagnostic records the worker-wrzh8 control/holdout lifecycle canary for the Phase317 runner guard.

| Item | Value |
|---|---|
| Source SHA | {row["source_sha"]} |
| Worker | {row["worker"]} |
| Control | {row["control_scenario"]} |
| Holdout | {row["holdout_scenario"]} |
| Output ratio | {row["output_ratio"]} |
| Verdict | {row["verdict"]} |
| Evidence status | {row["evidence_status"]} |
| Default AIC | No-Go |
| diagnostic_only | true |
| valid_for_default | false |
| perf_database | false |

The canary proves that both control and holdout can run to completion on worker-wrzh8 with the new lifecycle guard: run locks are released, cleanup traps run, and no benchmark failure artifacts remain.

This is not replacement evidence for Phase283/285, does not update Phase303 or Phase309, does not support model correction, and does not write PerfDatabase data.
"""
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_CSV_OUT)
    parser.add_argument("--doc-out", type=Path, default=DEFAULT_DOC_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = analyze_runner_lifecycle_pair_canary(args.artifact_root)
    write_runner_lifecycle_pair_canary_csv(args.out, rows)
    write_runner_lifecycle_pair_canary_doc(args.doc_out, rows)
    print(f"wrote {args.out}")
    print(f"wrote {args.doc_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
