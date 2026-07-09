#!/usr/bin/env python3
"""Phase451-H: first-divergence localization for ramp dynamics."""

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
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

SCENARIO = "K2.5-tp4ep8dp2-8k2k"
RUN_DIR = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase446_b2b_event_timing/"
    / "overhead_gate_20260708_075726/overhead_on"
)
SCENARIO_DIR = RUN_DIR / SCENARIO
DEFAULT_EVENT = RUN_DIR / "event_timing.jsonl"
DEFAULT_RECORDS = SCENARIO_DIR / "bench_records.jsonl"
DEFAULT_BENCH = SCENARIO_DIR / "bench_result.json"
DEFAULT_METRICS = SCENARIO_DIR / "metrics.jsonl"
DEFAULT_SERVE_LOG = SCENARIO_DIR / "serve.log"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase451h_first_divergence.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase451h_first_divergence.md"
DEFAULT_TP_WIDTH = 4
DEFAULT_CAPACITY_BLOCKS = 28633
DEFAULT_BLOCK_SIZE = 16
DEFAULT_READINESS = "No-Go"
MIXED_SHARE_TARGET = 0.0243
MIXED_SHARE_TOLERANCE = 0.005

CSV_FIELDS = [
    "section",
    "scenario",
    "side",
    "metric",
    "value",
    "target",
    "status",
    "note",
]


@dataclass(frozen=True)
class RealRawEvent:
    dp_rank: str
    ctx_requests: int
    ctx_tokens: int
    generation_requests: int
    generation_tokens: int
    num_tokens_unpadded: int
    forward_busy_ms: float


@dataclass(frozen=True)
class RealStep:
    dp_rank: str
    local_step: int
    ctx_requests: int
    ctx_tokens: int
    generation_requests: int
    generation_tokens: int
    bucket_tokens: int
    forward_busy_ms: float

    @property
    def phase(self) -> str:
        if self.ctx_tokens > 0 and self.generation_requests > 0:
            return "mixed_prefill"
        if self.ctx_tokens > 0:
            return "prefill"
        if self.generation_requests > 0:
            return "decode"
        return "empty"


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
    side: str,
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
        "side": side,
        "metric": metric,
        "value": value,
        "target": target,
        "status": status,
        "note": note,
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _load_validate_module():
    path = REPO_ROOT / "scripts" / "validate_cb_simulator.py"
    spec = importlib.util.spec_from_file_location("validate_cb_simulator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_phase451g_module():
    import scripts.analyze_phase451g_wave_groundtruth as phase451g

    return phase451g


def _load_phase451_module():
    import scripts.analyze_phase451_preemption_ledger as phase451

    return phase451


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with _open_text(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_real_raw_events(path: Path) -> list[RealRawEvent]:
    events: list[RealRawEvent] = []
    with _open_text(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != "phase446_graph_outer_event_v2":
                continue
            events.append(
                RealRawEvent(
                    dp_rank=str(row.get("dp_rank")),
                    ctx_requests=int(row.get("ctx_requests") or 0),
                    ctx_tokens=int(row.get("ctx_tokens") or 0),
                    generation_requests=int(row.get("generation_requests") or 0),
                    generation_tokens=int(row.get("generation_tokens") or 0),
                    num_tokens_unpadded=int(row.get("num_tokens_unpadded") or 0),
                    forward_busy_ms=float(row["forward_busy_ms"]),
                )
            )
    if not events:
        raise ValueError(f"no phase446 event rows found in {path}")
    return events


def group_real_steps(
    events: Iterable[RealRawEvent],
    *,
    tp_width: int = DEFAULT_TP_WIDTH,
) -> list[RealStep]:
    by_dp: dict[str, list[RealRawEvent]] = defaultdict(list)
    for event in events:
        by_dp[event.dp_rank].append(event)

    steps: list[RealStep] = []
    for dp_rank, items in sorted(by_dp.items()):
        width = tp_width
        if len(items) < tp_width:
            width = 1
        else:
            first_chunk = items[:tp_width]
            first_keys = {
                (
                    item.ctx_requests,
                    item.ctx_tokens,
                    item.generation_requests,
                    item.generation_tokens,
                    item.num_tokens_unpadded,
                )
                for item in first_chunk
            }
            # Phase446 v2 event rows are already one row per EngineCore step
            # (tp_rank is absent/null).  Older traces can have TP-duplicate rows.
            width = tp_width if len(first_keys) == 1 else 1
        if len(items) % width != 0:
            raise ValueError(
                f"dp_rank={dp_rank} row count {len(items)} not divisible by width={width}"
            )
        local_step = 0
        for idx in range(0, len(items), width):
            chunk = items[idx : idx + width]
            keys = {
                (
                    item.ctx_requests,
                    item.ctx_tokens,
                    item.generation_requests,
                    item.generation_tokens,
                    item.num_tokens_unpadded,
                )
                for item in chunk
            }
            if len(keys) != 1:
                raise ValueError(f"mixed TP chunk for dp_rank={dp_rank} offset={idx}")
            (
                ctx_requests,
                ctx_tokens,
                generation_requests,
                generation_tokens,
                num_tokens_unpadded,
            ) = next(iter(keys))
            bucket_tokens = num_tokens_unpadded or (ctx_tokens + generation_requests)
            steps.append(
                RealStep(
                    dp_rank=dp_rank,
                    local_step=local_step,
                    ctx_requests=ctx_requests,
                    ctx_tokens=ctx_tokens,
                    generation_requests=generation_requests,
                    generation_tokens=generation_tokens,
                    bucket_tokens=bucket_tokens,
                    forward_busy_ms=max(item.forward_busy_ms for item in chunk),
                )
            )
            local_step += 1
    return steps


def build_real_ramp_trajectory(
    steps: Iterable[RealStep],
    *,
    capacity_blocks: int = DEFAULT_CAPACITY_BLOCKS,
    block_size: int = DEFAULT_BLOCK_SIZE,
    max_steps_per_rank: int = 4096,
) -> list[dict[str, object]]:
    """Build a ramp trajectory from real event rows.

    Request identity and finish releases are not present in the event rows.  The
    block trajectory is therefore a ramp-only lower-fidelity ledger; it is used
    to locate early divergence, not to claim exact late-run free blocks.
    """
    used_blocks_by_rank: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    for step in sorted(steps, key=lambda item: (item.local_step, item.dp_rank)):
        if step.local_step >= max_steps_per_rank:
            continue
        delta_tokens = step.ctx_tokens + step.generation_tokens
        used_blocks_by_rank[step.dp_rank] += math.ceil(delta_tokens / block_size) if delta_tokens > 0 else 0
        running = step.ctx_requests + step.generation_requests
        rows.append(
            {
                "axis": step.local_step,
                "replica_id": int(step.dp_rank),
                "phase": step.phase,
                "running": running,
                "decode_reqs": step.generation_requests,
                "prefill_reqs": step.ctx_requests,
                "prefill_tokens": step.ctx_tokens,
                "total_tokens": step.bucket_tokens,
                "waiting": math.nan,
                "used_blocks": used_blocks_by_rank[step.dp_rank],
                "free_blocks": capacity_blocks - used_blocks_by_rank[step.dp_rank],
                "note": "real event rows lack request ids; free_blocks is ramp-only cumulative token ledger",
            }
        )
    return rows


def reconstruct_closed_loop_timeline(
    records: Iterable[dict[str, object]],
    *,
    max_concurrency: int,
) -> list[dict[str, float | int]]:
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
            }
        )
        heapq.heappush(worker_heap, (finish_ms, worker_id))
    return timeline


def _phase_from_schedule(schedule) -> str:
    if schedule.total_prefill_tokens > 0 and schedule.decode_reqs:
        return "mixed_prefill"
    if schedule.total_prefill_tokens > 0:
        return "prefill"
    if schedule.decode_reqs:
        return "decode"
    return "empty"


def _empty_schedule():
    from aiconfigurator.sdk.backends.cb_simulator.datatypes import ScheduleResult

    return ScheduleResult()


def run_real_arrival_sim_trace(
    *,
    scenario: str,
    arrival_times_ms: list[float],
    max_steps_per_rank: int = 4096,
) -> dict[str, object]:
    """Run current sim with reconstructed real arrivals and return ramp state."""
    from dataclasses import replace

    from aiconfigurator.sdk.backends.cb_simulator import CBSimulator
    from aiconfigurator.sdk.backends.cb_simulator.datatypes import Request, RequestState
    from aiconfigurator.sdk.backends.cb_simulator.dp_admission import (
        DPAdmissionRouter,
        DPReplicaCounts,
    )
    from aiconfigurator.sdk.backends.cb_simulator.scheduler import CBScheduler

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
    completed_by_replica: Counter[int] = Counter()
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
                "replica_id": next(
                    (
                        replica.replica_id
                        for replica in replicas
                        if replica.scheduler is self
                    ),
                    -1,
                ),
                "victim_req_id": victim.request_id,
                "victim_generated_tokens": victim.generated_tokens,
                "victim_preemptions_before": victim.num_preemptions,
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
                blocks_before = replica.scheduler._total_blocks(replica.running, _empty_schedule())
                schedule = replica.scheduler.schedule(replica.waiting, replica.running)
                if schedule.is_empty:
                    continue
                blocks_after = replica.scheduler._total_blocks(replica.running, schedule)
                decode_kv = [req.kv_cache_len for req in schedule.decode_reqs]
                avg_kv = int(statistics.mean(decode_kv)) if decode_kv else 0
                charge_ms = latency_calc.compute(
                    prefill_tokens=schedule.total_prefill_tokens,
                    prefill_batch_size=len(schedule.prefill_reqs),
                    prefill_seq_len=point.isl,
                    decode_batch_size=len(schedule.decode_reqs),
                    decode_avg_kv_len=avg_kv,
                )
                cycle.append((replica, schedule, charge_ms, avg_kv, blocks_before, blocks_after))
            if not cycle:
                break

            step_start_ms = global_clock_ms
            step_ms = max(item[2] for item in cycle)
            step_end_ms = step_start_ms + step_ms
            global_clock_ms = step_end_ms

            for replica, schedule, charge_ms, avg_kv, blocks_before, blocks_after in cycle:
                replica.total_iters += 1
                total_iters += 1
                if replica.total_iters <= max_steps_per_rank:
                    trace.append(
                        {
                            "axis": replica.total_iters - 1,
                            "replica_id": replica.replica_id,
                            "phase": _phase_from_schedule(schedule),
                            "running": len(replica.running) + sum(
                                1 for req in schedule.prefill_reqs if req not in replica.running
                            ),
                            "decode_reqs": len(schedule.decode_reqs),
                            "prefill_reqs": len(schedule.prefill_reqs),
                            "prefill_tokens": schedule.total_prefill_tokens,
                            "total_tokens": schedule.total_tokens,
                            "waiting": len(replica.waiting),
                            "used_blocks": blocks_after,
                            "free_blocks": config.num_gpu_blocks - blocks_after,
                            "blocks_before": blocks_before,
                            "charge_ms": charge_ms,
                            "avg_kv": avg_kv,
                        }
                    )

            for replica, schedule, _charge_ms, _avg_kv, _before, _after in cycle:
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
        "trace": trace,
        "preemptions": preemptions,
        "completed": completed,
        "wall_ms": global_clock_ms,
        "completed_by_replica": dict(completed_by_replica),
    }


def _field_differs(a: object, b: object) -> bool:
    if isinstance(a, float) and math.isnan(a) and isinstance(b, float) and math.isnan(b):
        return False
    if isinstance(a, float) or isinstance(b, float):
        try:
            af = float(a)
            bf = float(b)
        except (TypeError, ValueError):
            return a != b
        if math.isnan(af) and math.isnan(bf):
            return False
        if math.isnan(af) != math.isnan(bf):
            return True
        return abs(af - bf) > 1e-9
    return a != b


def _is_unobservable(value: object) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def semantic_hint_for_fields(fields: list[str]) -> str:
    field_set = set(fields)
    if {"running", "free_blocks"} & field_set:
        return "admission_or_capacity_gate"
    if {"prefill_tokens", "decode_reqs", "phase"} & field_set:
        return "budget_packing_or_phase"
    if "waiting" in field_set:
        return "admission_queue_or_routing"
    return "unknown"


def find_first_divergence(
    real_trace: Iterable[dict[str, object]],
    sim_trace: Iterable[dict[str, object]],
    *,
    compare_fields: tuple[str, ...] = (
        "running",
        "free_blocks",
        "waiting",
        "phase",
        "decode_reqs",
        "prefill_reqs",
        "prefill_tokens",
        "total_tokens",
    ),
) -> dict[str, object]:
    real_by_axis = {int(row["axis"]): row for row in real_trace}
    sim_by_axis = {int(row["axis"]): row for row in sim_trace}
    for axis in sorted(set(real_by_axis) & set(sim_by_axis)):
        real = real_by_axis[axis]
        sim = sim_by_axis[axis]
        differing = [
            field for field in compare_fields
            if not _is_unobservable(real.get(field, ""))
            and not _is_unobservable(sim.get(field, ""))
            and _field_differs(real.get(field, ""), sim.get(field, ""))
        ]
        if differing:
            return {
                "status": "found",
                "axis": axis,
                "first_differing_fields": differing,
                "semantic_hint": semantic_hint_for_fields(differing),
                "real": real,
                "sim": sim,
            }
    return {
        "status": "not_found",
        "axis": "",
        "first_differing_fields": [],
        "semantic_hint": "matched_within_available_axis",
        "real": {},
        "sim": {},
    }


def classify_latency_outlier(
    *,
    latency_ms: float,
    baseline_ms: float,
    prefill_ms: float,
    completion_wave_ms: float,
) -> str:
    excess = latency_ms - baseline_ms
    if excess <= 0:
        return "normal"
    if abs(excess - prefill_ms) <= max(prefill_ms * 0.50, 1.0):
        return "immediate_refill_like"
    if abs(excess - completion_wave_ms) <= max(completion_wave_ms * 0.35, 1.0):
        return "wait_for_completion_wave_like"
    return "other_outlier"


def summarize_latency_outliers(
    latencies_ms: Iterable[float],
    *,
    baseline_ms: float,
    prefill_ms: float,
    completion_wave_ms: float,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for latency_ms in latencies_ms:
        label = classify_latency_outlier(
            latency_ms=float(latency_ms),
            baseline_ms=baseline_ms,
            prefill_ms=prefill_ms,
            completion_wave_ms=completion_wave_ms,
        )
        if label != "normal":
            counts[label] += 1
    return {
        "immediate_refill_like": counts["immediate_refill_like"],
        "wait_for_completion_wave_like": counts["wait_for_completion_wave_like"],
        "other_outlier": counts["other_outlier"],
    }


def summarize_real_victim_recovery(records: list[dict[str, object]]) -> dict[str, object]:
    latencies = [float(row["latency_ms"]) for row in records if int(row.get("ok", 1)) == 1]
    baseline = _percentile(latencies, 10)
    p50 = _percentile(latencies, 50)
    p90 = _percentile(latencies, 90)
    p99 = _percentile(latencies, 99)
    # One 8k prefill is on the order of 8-10 seconds in the client latency
    # distribution; one completion wave is about the spread between early and
    # late first-wave completions.
    outliers = summarize_latency_outliers(
        latencies,
        baseline_ms=baseline,
        prefill_ms=9000.0,
        completion_wave_ms=80000.0,
    )
    return {
        "latency_p10_ms": baseline,
        "latency_p50_ms": p50,
        "latency_p90_ms": p90,
        "latency_p99_ms": p99,
        "immediate_refill_like": outliers.get("immediate_refill_like", 0),
        "wait_for_completion_wave_like": outliers.get("wait_for_completion_wave_like", 0),
        "other_outlier": outliers.get("other_outlier", 0),
        "classification_status": "blocked",
        "note": "bench_records has request latency but no victim id; recovery path remains suggestive, not proven",
    }


def _phase_share(trace: list[dict[str, object]], phase: str) -> float:
    return sum(1 for row in trace if row.get("phase") == phase) / len(trace) if trace else 0.0


def _first_rows_for_replica(
    trace: list[dict[str, object]],
    *,
    replica_id: int,
) -> list[dict[str, object]]:
    return [
        row for row in sorted(trace, key=lambda item: (int(item["axis"]), int(item.get("replica_id", 0))))
        if int(row.get("replica_id", -1)) == replica_id
    ]


def _metrics_preemption_total(metrics_path: Path) -> float:
    phase451 = _load_phase451_module()
    deltas = phase451.collect_metric_deltas(metrics_path)
    return sum(delta.delta for delta in deltas.get("num_preemptions_total", {}).values())


def count_preemption_log_lines(serve_log: Path) -> int:
    count = 0
    with _open_text(serve_log) as f:
        for line in f:
            lower = line.lower()
            if "preempt" in lower or "recompute" in lower:
                count += 1
    return count


def build_report_rows(
    *,
    event_path: Path = DEFAULT_EVENT,
    records_path: Path = DEFAULT_RECORDS,
    bench_path: Path = DEFAULT_BENCH,
    metrics_path: Path = DEFAULT_METRICS,
    serve_log: Path = DEFAULT_SERVE_LOG,
    scenario: str = SCENARIO,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    records = read_jsonl(records_path)
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    timeline = reconstruct_closed_loop_timeline(
        records,
        max_concurrency=int(bench["max_concurrency"]),
    )
    real_steps = group_real_steps(read_real_raw_events(event_path))
    real_trace = build_real_ramp_trajectory(real_steps)
    sim = run_real_arrival_sim_trace(
        scenario=scenario,
        arrival_times_ms=[float(row["start_ms"]) for row in timeline],
    )
    sim_trace = list(sim["trace"])

    real_replica0 = _first_rows_for_replica(real_trace, replica_id=0)
    sim_replica0 = _first_rows_for_replica(sim_trace, replica_id=0)
    divergence = find_first_divergence(real_replica0, sim_replica0)
    victim = summarize_real_victim_recovery(records)
    real_mixed_share = _phase_share(real_trace, "mixed_prefill")
    sim_mixed_share = _phase_share(sim_trace, "mixed_prefill")
    real_preemptions = _metrics_preemption_total(metrics_path)
    serve_preemption_lines = count_preemption_log_lines(serve_log)

    rows: list[dict[str, object]] = [
        _row("h1_real_ramp", "real", "real_trace_steps", len(real_trace)),
        _row("h1_real_ramp", "real", "real_mixed_share", real_mixed_share, target=f"{MIXED_SHARE_TARGET:.4f}±{MIXED_SHARE_TOLERANCE:.4f}"),
        _row("h1_real_ramp", "real", "metrics_preemptions_total", real_preemptions, target="overlay only; no exact step timestamps"),
        _row("h1_real_ramp", "real", "serve_preemption_log_lines", serve_preemption_lines, target=">0 needed for victim context", status="blocked" if serve_preemption_lines == 0 else "informational"),
        _row("h1_real_ramp", "real", "latency_p10_ms", victim["latency_p10_ms"]),
        _row("h1_real_ramp", "real", "latency_p50_ms", victim["latency_p50_ms"]),
        _row("h1_real_ramp", "real", "latency_p90_ms", victim["latency_p90_ms"]),
        _row("h1_real_ramp", "real", "immediate_refill_like_outliers", victim["immediate_refill_like"], status="informational"),
        _row("h1_real_ramp", "real", "wait_for_completion_wave_like_outliers", victim["wait_for_completion_wave_like"], status="informational"),
        _row("h1_real_ramp", "real", "victim_recovery_path_gate", victim["classification_status"], status="blocked", note=victim["note"]),
        _row("h2_sim_ramp", "sim", "sim_trace_steps", len(sim_trace)),
        _row("h2_sim_ramp", "sim", "sim_preemptions", len(sim["preemptions"])),
        _row("h2_sim_ramp", "sim", "sim_completed", sim["completed"]),
        _row("h2_sim_ramp", "sim", "sim_mixed_share", sim_mixed_share, target=f"{MIXED_SHARE_TARGET:.4f}±{MIXED_SHARE_TOLERANCE:.4f}"),
        _row("h3_divergence", "compare", "first_divergence_status", divergence["status"], status="pass" if divergence["status"] == "found" else "blocked"),
        _row("h3_divergence", "compare", "first_divergence_axis", divergence["axis"]),
        _row("h3_divergence", "compare", "first_differing_fields", ",".join(divergence["first_differing_fields"])),
        _row("h3_divergence", "compare", "semantic_hint", divergence["semantic_hint"], status="informational"),
        _row(
            "h3_divergence",
            "compare",
            "first_divergence_observability_gate",
            "blocked",
            status="blocked",
            note="real event rows have no per-step waiting queue; axis=1 may be engine-arrival visibility, not scheduler admission semantics",
        ),
        _row("h3_source", "vllm", "waiting_admission", "scheduler.py:575-700; kv_cache_manager.py:327-334", note="source point to inspect only if first divergence is admission/capacity"),
        _row("h3_source", "vllm", "running_then_waiting_order", "scheduler.py:530-700", note="vLLM does attempt WAITING scheduling after RUNNING when token budget remains"),
        _row("h3_source", "vllm", "preemption_reentry", "scheduler.py:956-971", note="source point to inspect only if victim recovery is proven"),
        _row("h4_fix", "decision", "runtime_patch", "not_applied", status="blocked", note="report-only H1-H3; current logs do not identify a single source-level fix"),
        _row("h4_fix", "decision", "default_aic", DEFAULT_READINESS, status="blocked"),
    ]
    sample_rows: list[dict[str, object]] = []
    if divergence["status"] == "found":
        real = dict(divergence["real"])
        sim_row = dict(divergence["sim"])
        sample_rows = [
            {
                "label": "real_first_divergence",
                **real,
            },
            {
                "label": "sim_first_divergence",
                **sim_row,
            },
        ]
    return rows, sample_rows, sim_trace[:80]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def write_md(
    path: Path,
    rows: list[dict[str, object]],
    divergence_rows: list[dict[str, object]],
    sim_sample: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_metric = {str(row["metric"]): row for row in rows}
    semantic_hint = by_metric.get("semantic_hint", {}).get("value", "unknown")
    divergence_status = by_metric.get("first_divergence_status", {}).get("value", "")
    lines = [
        "# Phase451-H first divergence",
        "",
        (
            f"结论: 首次分歧状态 `{divergence_status}`, 初步语义指向 `{semantic_hint}`。"
            "分岔点是 axis=1:真实只 decode,sim 同步 admit 下一条 prefill。"
            "但真实日志没有 per-step waiting 队列和 victim request id,所以不能把它硬判成 scheduler 源码错误。"
        ),
        "",
        f"- Default AIC remains `{DEFAULT_READINESS}`.",
        "- Report-only: no scheduler/runtime/PerfDB/validate gate changes.",
        "",
        "## Summary",
        "",
        "| section | side | metric | value | target | status | note |",
        "|---|---|---|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['section']} | {row['side']} | {row['metric']} | {_fmt(row['value'])} | "
            f"{row.get('target', '')} | {row.get('status', '')} | {row.get('note', '')} |"
        )
    lines.extend(
        [
            "",
            "## First Divergence Row",
            "",
            "| label | axis | replica | phase | running | decode | prefill_reqs | prefill_tokens | free_blocks | waiting | note |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in divergence_rows:
        lines.append(
            f"| {row.get('label', '')} | {row.get('axis', '')} | {row.get('replica_id', '')} | "
            f"{row.get('phase', '')} | {row.get('running', '')} | {row.get('decode_reqs', '')} | "
            f"{row.get('prefill_reqs', '')} | {row.get('prefill_tokens', '')} | "
            f"{_fmt(row.get('free_blocks', ''))} | {_fmt(row.get('waiting', ''))} | {row.get('note', '')} |"
        )
    lines.extend(
        [
            "",
            "## Sim Ramp Sample",
            "",
            "| axis | replica | phase | running | waiting | decode | prefill_tokens | free_blocks |",
            "|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sim_sample[:20]:
        lines.append(
            f"| {row.get('axis', '')} | {row.get('replica_id', '')} | {row.get('phase', '')} | "
            f"{row.get('running', '')} | {row.get('waiting', '')} | {row.get('decode_reqs', '')} | "
            f"{row.get('prefill_tokens', '')} | {_fmt(row.get('free_blocks', ''))} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Real event rows identify step composition, not request identity.",
            "- Metrics expose total preemptions, not exact victim context or exact step timestamp.",
            "- A runtime fix remains blocked until the first divergent decision can be mapped to one source-level semantic with victim recovery evidence.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=Path, default=DEFAULT_EVENT)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--serve-log", type=Path, default=DEFAULT_SERVE_LOG)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows, divergence_rows, sim_sample = build_report_rows(
        event_path=args.event,
        records_path=args.records,
        bench_path=args.bench,
        metrics_path=args.metrics,
        serve_log=args.serve_log,
    )
    write_csv(args.csv_out, rows)
    write_md(args.md_out, rows, divergence_rows, sim_sample)
    print(f"wrote {args.csv_out}")
    print(f"wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
