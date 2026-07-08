#!/usr/bin/env python3
"""Phase451-G: ground truth for wave desynchronization and arrival replay."""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

SCENARIO = "K2.5-tp4ep8dp2-8k2k"
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase446_b2b_event_timing/"
    / "overhead_gate_20260708_075726/overhead_on"
)
DEFAULT_SCENARIO_DIR = DEFAULT_RUN_DIR / SCENARIO
DEFAULT_RECORDS = DEFAULT_SCENARIO_DIR / "bench_records.jsonl"
DEFAULT_BENCH = DEFAULT_SCENARIO_DIR / "bench_result.json"
DEFAULT_METRICS = DEFAULT_SCENARIO_DIR / "metrics.jsonl"
DEFAULT_SERVE_LOG = DEFAULT_SCENARIO_DIR / "serve.log"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase451g_wave_groundtruth.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase451g_wave_groundtruth.md"
DEFAULT_READINESS = "No-Go"
MIXED_SHARE_TARGET = 0.0243
MIXED_SHARE_TOLERANCE = 0.005

CSV_FIELDS = [
    "section",
    "scenario",
    "metric",
    "value",
    "target",
    "status",
    "note",
]


@dataclass
class ReplicaState:
    replica_id: int
    scheduler: object
    waiting: list
    running: list
    total_iters: int = 0


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def _load_validate_module():
    path = REPO_ROOT / "scripts" / "validate_cb_simulator.py"
    spec = importlib.util.spec_from_file_location("validate_cb_simulator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_phase451_module():
    import scripts.analyze_phase451_preemption_ledger as phase451

    return phase451


def _load_phase451d_module():
    import scripts.analyze_phase451d_preemption_forensics as phase451d

    return phase451d


def _fmt(value: object) -> str:
    if value == "":
        return ""
    if isinstance(value, bool):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "nan"
    if abs(number) >= 1000:
        return f"{number:.0f}"
    if number.is_integer():
        return f"{number:.0f}"
    return f"{number:.6f}".rstrip("0").rstrip(".")


def _row(
    section: str,
    metric: str,
    value: object,
    *,
    target: object = "",
    status: str = "",
    note: str = "",
    scenario: str = SCENARIO,
) -> dict[str, object]:
    return {
        "section": section,
        "scenario": scenario,
        "metric": metric,
        "value": value,
        "target": target,
        "status": status,
        "note": note,
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with _open_text(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def reconstruct_closed_loop_timeline(
    records: Iterable[dict[str, object]],
    *,
    max_concurrency: int,
) -> list[dict[str, float | int]]:
    """Reconstruct client request start/finish times from closed-loop records.

    The benchmark has ``max_concurrency`` workers sharing one monotonically
    increasing request index.  The next request starts when the earliest worker
    finishes its previous request, so latency records are enough to reconstruct
    the client-side arrival schedule.
    """
    worker_heap: list[tuple[float, int]] = [
        (0.0, worker_id) for worker_id in range(max_concurrency)
    ]
    heapq.heapify(worker_heap)
    timeline: list[dict[str, float | int]] = []
    for record in sorted(records, key=lambda item: int(item["request_index"])):
        if int(record.get("ok", 1)) != 1:
            continue
        start_ms, worker_id = heapq.heappop(worker_heap)
        latency_ms = float(record["latency_ms"])
        finish_ms = start_ms + latency_ms
        request_index = int(record["request_index"])
        timeline.append(
            {
                "request_index": request_index,
                "worker_id": worker_id,
                "start_ms": start_ms,
                "finish_ms": finish_ms,
                "latency_ms": latency_ms,
                "prompt_tokens": int(record.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(record.get("completion_tokens", 0) or 0),
            }
        )
        heapq.heappush(worker_heap, (finish_ms, worker_id))
    return timeline


def summarize_client_waves(
    timeline: Iterable[dict[str, float | int]],
    *,
    max_concurrency: int,
) -> list[dict[str, float | int]]:
    by_wave: dict[int, list[dict[str, float | int]]] = {}
    for row in timeline:
        wave_id = int(row["request_index"]) // max_concurrency
        by_wave.setdefault(wave_id, []).append(row)
    rows: list[dict[str, float | int]] = []
    wave0_span = math.nan
    for wave_id in sorted(by_wave):
        items = by_wave[wave_id]
        starts = [float(item["start_ms"]) for item in items]
        finishes = [float(item["finish_ms"]) for item in items]
        finish_span = max(finishes) - min(finishes)
        if wave_id == 0:
            wave0_span = finish_span
        growth = (
            finish_span / wave0_span
            if wave0_span and not math.isnan(wave0_span)
            else math.nan
        )
        rows.append(
            {
                "wave_id": wave_id,
                "count": len(items),
                "start_min_ms": min(starts),
                "start_p50_ms": _percentile(starts, 50),
                "start_max_ms": max(starts),
                "start_span_ms": max(starts) - min(starts),
                "finish_min_ms": min(finishes),
                "finish_p50_ms": _percentile(finishes, 50),
                "finish_max_ms": max(finishes),
                "finish_span_ms": finish_span,
                "desync_growth_vs_wave0": growth,
            }
        )
    return rows


def summarize_interarrival(timeline: Iterable[dict[str, float | int]]) -> dict[str, float]:
    starts = sorted(float(row["start_ms"]) for row in timeline)
    nonzero = [value for value in starts if value > 0]
    gaps = [
        right - left
        for left, right in zip(nonzero, nonzero[1:])
        if right >= left
    ]
    return {
        "nonzero_start_count": float(len(nonzero)),
        "start_gap_p50_ms": _percentile(gaps, 50),
        "start_gap_p90_ms": _percentile(gaps, 90),
        "start_gap_max_ms": max(gaps) if gaps else math.nan,
    }


def _phase_from_schedule(schedule) -> str:
    if schedule.total_prefill_tokens > 0 and schedule.decode_reqs:
        return "mixed"
    if schedule.total_prefill_tokens > 0:
        return "prefill"
    if schedule.decode_reqs:
        return "decode"
    return "empty"


def _trace_mixed_share(trace: list[dict[str, object]]) -> float:
    if not trace:
        return 0.0
    return sum(1 for row in trace if bool(row.get("is_mixed"))) / len(trace)


def _count_preemptions_for_replay(
    *,
    point,
    backend,
    model,
    db,
    config,
    arrival_times_ms: list[float],
) -> dict[str, object]:
    from aiconfigurator.sdk.backends.cb_simulator import CBSimulator
    from aiconfigurator.sdk.backends.cb_simulator.datatypes import Request, RequestState
    from aiconfigurator.sdk.backends.cb_simulator.dp_admission import (
        DPAdmissionRouter,
        DPReplicaCounts,
    )
    from aiconfigurator.sdk.backends.cb_simulator.scheduler import CBScheduler

    sim = CBSimulator(backend, model, db, config)
    latency_calc = sim._create_latency_calc(0)
    replicas = [
        ReplicaState(idx, CBScheduler(config), [], [])
        for idx in range(point.dp)
    ]
    router = DPAdmissionRouter(point.dp)
    pending = list(sorted(enumerate(arrival_times_ms), key=lambda item: (item[1], item[0])))
    pending_index = 0
    completed = 0
    completed_by_replica = Counter()
    global_clock_ms = 0.0
    trace: list[dict[str, object]] = []
    preemptions: list[dict[str, object]] = []

    original_preempt = CBScheduler._preempt

    def actual_counts() -> list[DPReplicaCounts]:
        return [
            DPReplicaCounts(waiting=len(replica.waiting), running=len(replica.running))
            for replica in replicas
        ]

    def route_due(now_ms: float) -> None:
        nonlocal pending_index
        while pending_index < len(pending) and pending[pending_index][1] <= now_ms + 1e-6:
            request_id, arrival_ms = pending[pending_index]
            replica_idx = router.route(now_ms=arrival_ms, actual_counts=actual_counts())
            replicas[replica_idx].waiting.append(
                Request(
                    request_id=request_id,
                    isl=point.isl,
                    osl=point.osl,
                    arrival_time_ms=arrival_ms,
                )
            )
            pending_index += 1

    def wrapped_preempt(self, victim, waiting, running, result, preempted_ids):
        preemptions.append(
            {
                "victim_req_id": victim.request_id,
                "victim_preemptions_before": victim.num_preemptions,
                "generated_tokens": victim.generated_tokens,
            }
        )
        return original_preempt(self, victim, waiting, running, result, preempted_ids)

    CBScheduler._preempt = wrapped_preempt
    try:
        max_iters = max(1, len(arrival_times_ms)) * max(1, point.dp) * (
            point.osl + point.isl // config.max_num_batched_tokens + 10
        )
        total_iters = 0
        while completed < len(arrival_times_ms) and total_iters < max_iters:
            route_due(global_clock_ms)
            active_exists = any(replica.waiting or replica.running for replica in replicas)
            if not active_exists:
                if pending_index >= len(pending):
                    break
                global_clock_ms = max(global_clock_ms, pending[pending_index][1])
                route_due(global_clock_ms)

            cycle = []
            for replica in replicas:
                if not replica.waiting and not replica.running:
                    continue
                schedule = replica.scheduler.schedule(replica.waiting, replica.running)
                if schedule.is_empty:
                    continue
                decode_kv = [req.kv_cache_len for req in schedule.decode_reqs]
                avg_kv = int(statistics.mean(decode_kv)) if decode_kv else 0
                charge_ms = latency_calc.compute(
                    prefill_tokens=schedule.total_prefill_tokens,
                    prefill_batch_size=len(schedule.prefill_reqs),
                    prefill_seq_len=point.isl,
                    decode_batch_size=len(schedule.decode_reqs),
                    decode_avg_kv_len=avg_kv,
                )
                cycle.append((replica, schedule, charge_ms, avg_kv))
            if not cycle:
                break

            step_start_ms = global_clock_ms
            step_ms = max(item[2] for item in cycle)
            step_end_ms = step_start_ms + step_ms
            global_clock_ms = step_end_ms

            for replica, schedule, charge_ms, avg_kv in cycle:
                replica.total_iters += 1
                total_iters += 1
                trace.append(
                    {
                        "replica_id": replica.replica_id,
                        "local_iter": replica.total_iters,
                        "start_ms": step_start_ms,
                        "end_ms": step_end_ms,
                        "phase": _phase_from_schedule(schedule),
                        "prefill_reqs": len(schedule.prefill_reqs),
                        "prefill_tokens": schedule.total_prefill_tokens,
                        "decode_reqs": len(schedule.decode_reqs),
                        "total_tokens": schedule.total_tokens,
                        "avg_kv": avg_kv,
                        "charge_ms": charge_ms,
                        "is_mixed": bool(schedule.prefill_reqs and schedule.decode_reqs),
                    }
                )

            for replica, schedule, _charge_ms, _avg_kv in cycle:
                for req in schedule.prefill_reqs:
                    tokens = schedule.prefill_tokens[req.request_id]
                    if req.state in {RequestState.WAITING, RequestState.PREEMPTED}:
                        req.state = RequestState.PREFILLING
                        if req in replica.waiting:
                            replica.waiting.remove(req)
                        replica.running.append(req)
                    req.prefill_tokens_remaining -= tokens
                    if req.prefill_tokens_remaining <= 0:
                        req.prefill_tokens_remaining = 0
                        req.state = RequestState.DECODING

                newly_done = []
                for req in schedule.decode_reqs:
                    req.generated_tokens += 1
                    if req.generated_tokens >= req.osl - 1:
                        req.state = RequestState.DONE
                        req.finish_ms = step_end_ms
                        newly_done.append(req)
                for req in newly_done:
                    if req in replica.running:
                        replica.running.remove(req)
                    completed += 1
                    completed_by_replica[replica.replica_id] += 1
    finally:
        CBScheduler._preempt = original_preempt

    return {
        "completed": completed,
        "wall_ms": global_clock_ms,
        "preemptions": len(preemptions),
        "preemptions_per_request": len(preemptions) / len(arrival_times_ms)
        if arrival_times_ms else 0.0,
        "mixed_share": _trace_mixed_share(trace),
        "trace_rows": len(trace),
        "completed_by_replica": dict(completed_by_replica),
    }


def run_real_arrival_replay(
    *,
    scenario: str,
    arrival_times_ms: list[float],
) -> dict[str, object]:
    validate = _load_validate_module()
    point = next(pt for pt in validate.MULTI_CONFIG_DATA if pt.name == scenario)
    model, db, backend = validate._load_model_and_db(
        tp=point.tp,
        dp=point.dp,
        moe_tp=point.moe_tp,
        moe_ep=point.moe_ep,
    )
    config = validate._make_multi_config_cb_config(
        point,
        overlap_factor=0.0,
        ep8_per_iteration_overhead_ms=validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
    )
    config = replace(config, num_requests=len(arrival_times_ms), warmup_requests=0)
    return _count_preemptions_for_replay(
        point=point,
        backend=backend,
        model=model,
        db=db,
        config=config,
        arrival_times_ms=arrival_times_ms,
    )


def decide_replay_responsibility(
    *,
    real_arrival_preemptions: int,
    real_arrival_mixed_share: float,
    closed_loop_preemptions: int,
    closed_loop_mixed_share: float,
    target_mixed_share: float,
) -> dict[str, str]:
    target_delta = abs(real_arrival_mixed_share - target_mixed_share)
    closed_delta = abs(closed_loop_mixed_share - target_mixed_share)
    if (
        real_arrival_preemptions <= max(2, int(0.05 * max(closed_loop_preemptions, 1)))
        and target_delta <= MIXED_SHARE_TOLERANCE * 2
        and target_delta < closed_delta
    ):
        return {
            "responsibility": "arrival_timing_model",
            "wave_model_gate": "ready_for_nonparametric_arrival_replay_design",
            "reason": "real reconstructed arrivals remove thrash and match mixed-share target",
        }
    if (
        real_arrival_preemptions >= int(0.50 * max(closed_loop_preemptions, 1))
        or target_delta >= closed_delta * 0.75
    ):
        return {
            "responsibility": "scheduler_dynamics_internal",
            "wave_model_gate": "blocked_pending_scheduler_mechanism",
            "reason": "real reconstructed arrivals still preserve most thrash or mixed-share error",
        }
    return {
        "responsibility": "inconclusive",
        "wave_model_gate": "blocked_pending_more_evidence",
        "reason": "arrival replay improves but does not isolate one mechanism",
    }


def count_literal_preemption_lines(serve_log: Path) -> int:
    count = 0
    with _open_text(serve_log) as f:
        for line in f:
            lower = line.lower()
            if "preempt" in lower or "recompute" in lower:
                count += 1
    return count


def metric_total(deltas: dict[str, dict[str, object]], metric: str) -> float:
    return sum(value.delta for value in deltas.get(metric, {}).values())


def build_report_rows(
    *,
    records_path: Path = DEFAULT_RECORDS,
    bench_path: Path = DEFAULT_BENCH,
    metrics_path: Path = DEFAULT_METRICS,
    serve_log: Path = DEFAULT_SERVE_LOG,
    scenario: str = SCENARIO,
) -> list[dict[str, object]]:
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    records = read_jsonl(records_path)
    concurrency = int(bench["max_concurrency"])
    timeline = reconstruct_closed_loop_timeline(records, max_concurrency=concurrency)
    waves = summarize_client_waves(timeline, max_concurrency=concurrency)
    interarrival = summarize_interarrival(timeline)
    reconstructed_wall_s = max(float(row["finish_ms"]) for row in timeline) / 1000.0
    wall_error = abs(reconstructed_wall_s - float(bench["wall_s"])) / float(bench["wall_s"])

    phase451 = _load_phase451_module()
    real_iterations = phase451.summarize_real_iterations(serve_log)
    deltas = phase451.collect_metric_deltas(metrics_path)
    bench_request_count = float(len(timeline))
    metric_success_count = metric_total(deltas, "request_success_length")
    preemptions = metric_total(deltas, "num_preemptions_total")
    recomputed_tokens = metric_total(deltas, "prompt_tokens_recomputed_total")
    prompt_tokens = metric_total(deltas, "prompt_tokens_total")
    prefill_kv_tokens = metric_total(deltas, "request_prefill_kv_computed_tokens_sum")
    serve_preemption_lines = count_literal_preemption_lines(serve_log)

    phase451d = _load_phase451d_module()
    closed_loop = phase451d.run_sim_preemption_forensics(
        scenario=scenario,
        num_requests=len(timeline),
    )
    closed_loop_mixed_share = _trace_mixed_share(list(closed_loop["trace"]))
    real_arrival = run_real_arrival_replay(
        scenario=scenario,
        arrival_times_ms=[float(row["start_ms"]) for row in timeline],
    )
    verdict = decide_replay_responsibility(
        real_arrival_preemptions=int(real_arrival["preemptions"]),
        real_arrival_mixed_share=float(real_arrival["mixed_share"]),
        closed_loop_preemptions=len(closed_loop["events"]),
        closed_loop_mixed_share=closed_loop_mixed_share,
        target_mixed_share=MIXED_SHARE_TARGET,
    )

    rows: list[dict[str, object]] = [
        _row("g1_groundtruth", "bench_records", len(timeline), target="all ok records"),
        _row("g1_groundtruth", "max_concurrency", concurrency),
        _row(
            "g1_groundtruth",
            "reconstructed_wall_error",
            wall_error,
            target="<=0.02",
            status="pass" if wall_error <= 0.02 else "blocked",
            note=f"reconstructed={reconstructed_wall_s:.3f}s; bench={float(bench['wall_s']):.3f}s",
        ),
        _row("g1_groundtruth", "wave_count", len(waves)),
        _row("g1_groundtruth", "start_gap_p50_ms", interarrival["start_gap_p50_ms"]),
        _row("g1_groundtruth", "start_gap_p90_ms", interarrival["start_gap_p90_ms"]),
    ]
    for wave in waves[: min(6, len(waves))]:
        rows.extend(
            [
                _row(
                    "g1_wave",
                    f"wave_{wave['wave_id']}_start_span_ms",
                    wave["start_span_ms"],
                    note=f"count={wave['count']}",
                ),
                _row(
                    "g1_wave",
                    f"wave_{wave['wave_id']}_finish_span_ms",
                    wave["finish_span_ms"],
                    note=f"growth_vs_wave0={_fmt(wave['desync_growth_vs_wave0'])}",
                ),
            ]
        )

    rows.extend(
        [
            _row("g2_puzzle", "bench_ok_request_count", bench_request_count),
            _row(
                "g2_puzzle",
                "metrics_success_count_delta",
                metric_success_count,
                note="metrics polling can miss already-finished requests at the first sample",
            ),
            _row(
                "g2_puzzle",
                "num_preemptions_total",
                preemptions,
                target="explain 54-style counter",
                note="vllm/v1/core/sched/scheduler.py:957-973 increments request.num_preemptions",
            ),
            *[
                _row(
                    "g2_puzzle",
                    f"engine_{engine}_num_preemptions_total",
                    delta.delta,
                    target="54-style per-engine counter",
                )
                for engine, delta in sorted(deltas.get("num_preemptions_total", {}).items())
            ],
            _row(
                "g2_puzzle",
                "preemptions_per_request",
                preemptions / bench_request_count if bench_request_count else math.nan,
            ),
            _row(
                "g2_puzzle",
                "prompt_tokens_recomputed_total",
                recomputed_tokens,
                target="not full preemption recompute",
                note="vllm/v1/metrics/stats.py:249-296 defines recomputed as cached-token accounting",
            ),
            _row(
                "g2_puzzle",
                "prompt_tokens_total_per_request",
                prompt_tokens / bench_request_count if bench_request_count else math.nan,
                target="~8000 if no full prompt replay is visible in counters",
            ),
            _row(
                "g2_puzzle",
                "prefill_kv_computed_tokens_per_request",
                prefill_kv_tokens / bench_request_count if bench_request_count else math.nan,
                target="~8000",
            ),
            _row(
                "g2_puzzle",
                "mixed_steps_per_request",
                float(real_iterations["mixed_steps"]) / bench_request_count if bench_request_count else math.nan,
                target="~1.04",
                note=f"mixed_steps={real_iterations['mixed_steps']}",
            ),
            _row(
                "g2_puzzle",
                "serve_preemption_log_lines",
                serve_preemption_lines,
                target="victim context unavailable if 0",
                status="blocked" if serve_preemption_lines == 0 else "pass",
            ),
            _row(
                "g2_puzzle",
                "puzzle_resolution",
                "partial_counter_semantics_only",
                status="blocked",
                note=(
                    "preemption counter and recomputed-token counter measure different things; "
                    "serve.log lacks victim context, so recovery path remains unproven"
                ),
            ),
        ]
    )
    rows.extend(
        [
            _row("g3_replay", "closed_loop_preemptions", len(closed_loop["events"])),
            _row("g3_replay", "closed_loop_mixed_share", closed_loop_mixed_share),
            _row("g3_replay", "real_arrival_replay_preemptions", real_arrival["preemptions"]),
            _row(
                "g3_replay",
                "real_arrival_replay_preemptions_per_request",
                real_arrival["preemptions_per_request"],
            ),
            _row("g3_replay", "real_arrival_replay_mixed_share", real_arrival["mixed_share"]),
            _row(
                "g3_replay",
                "real_arrival_replay_completed",
                real_arrival["completed"],
                target=len(timeline),
                status="pass" if int(real_arrival["completed"]) == len(timeline) else "blocked",
                note=f"completed_by_replica={real_arrival['completed_by_replica']}",
            ),
            _row("g3_replay", "responsibility", verdict["responsibility"], status="pass" if verdict["responsibility"] != "inconclusive" else "blocked", note=verdict["reason"]),
            _row("g4_decision", "wave_model_gate", verdict["wave_model_gate"], status="blocked" if verdict["wave_model_gate"].startswith("blocked") else "open"),
            _row(
                "g4_decision",
                "runtime_patch",
                "not_applied",
                status="blocked",
                note="Phase451-G is report-only until a non-parametric mechanism passes declared gates",
            ),
            _row("g4_decision", "default_aic", DEFAULT_READINESS, status="blocked"),
        ]
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def write_md(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_metric = {str(row["metric"]): row for row in rows}
    responsibility = by_metric.get("responsibility", {}).get("value", "unknown")
    gate = by_metric.get("wave_model_gate", {}).get("value", "unknown")
    lines = [
        "# Phase451-G wave ground truth",
        "",
        f"结论: `{responsibility}`; wave_model_gate=`{gate}`。本报告只做 G1-G3 取证,不改 runtime/PerfDB/gate。",
        "",
        f"- Default AIC remains `{DEFAULT_READINESS}`.",
        "",
        "## Summary",
        "",
        "| section | metric | value | target | status | note |",
        "|---|---|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['section']} | {row['metric']} | {_fmt(row['value'])} | "
            f"{row.get('target', '')} | {row.get('status', '')} | {row.get('note', '')} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- `bench_records` has latency only; start times are reconstructed from the benchmark worker queue semantics.",
            "- `prompt_tokens_recomputed_total` is cached-token accounting, not a direct full-prompt preemption recovery counter.",
            "- `serve.log` has no victim-level preemption records in this run, so victim recovery path remains unproven.",
            "- No runtime patch is applied in this phase.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    rows = build_report_rows()
    write_csv(args.csv_out, rows)
    write_md(args.md_out, rows)
    print(f"wrote {args.csv_out}")
    print(f"wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
