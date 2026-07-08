#!/usr/bin/env python3
"""Phase451-F: distinguish block deficit from wave overshoot thrash dynamics."""

from __future__ import annotations

import argparse
import bisect
import csv
import importlib.util
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

SCENARIO = "K2.5-tp4ep8dp2-8k2k"
DEFAULT_NUM_REQUESTS = 512
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase451f_thrash_dynamics.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase451f_thrash_dynamics.md"
DEFAULT_READINESS = "No-Go"

CSV_FIELDS = [
    "section",
    "scenario",
    "metric",
    "value",
    "target",
    "status",
    "note",
]


def _load_validate_module():
    path = REPO_ROOT / "scripts" / "validate_cb_simulator.py"
    spec = importlib.util.spec_from_file_location("validate_cb_simulator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    if len(values) == 1:
        return values[0]
    return float(statistics.quantiles(values, n=100, method="inclusive")[int(pct) - 1])


def _safe_cv(values: list[float]) -> float:
    if len(values) < 2:
        return math.nan
    mean = statistics.mean(values)
    if mean == 0:
        return math.nan
    return statistics.pstdev(values) / mean


def distances_to_nearest_mixed(
    *,
    events: Iterable[dict[str, object]],
    trace: Iterable[dict[str, object]],
) -> list[int]:
    """Return same-replica iteration distance from each event to nearest mixed step."""
    mixed_iters: dict[int, list[int]] = defaultdict(list)
    for row in trace:
        if bool(row.get("is_mixed")):
            mixed_iters[int(row["replica_id"])].append(int(row["local_iter"]))
    for values in mixed_iters.values():
        values.sort()

    distances: list[int] = []
    for event in events:
        replica = int(event.get("replica_id", -1))
        local_iter = int(event.get("local_iter", -1))
        candidates = mixed_iters.get(replica, [])
        if not candidates:
            continue
        pos = bisect.bisect_left(candidates, local_iter)
        local_distances: list[int] = []
        if pos < len(candidates):
            local_distances.append(abs(candidates[pos] - local_iter))
        if pos > 0:
            local_distances.append(abs(candidates[pos - 1] - local_iter))
        if local_distances:
            distances.append(min(local_distances))
    return distances


def summarize_loop_dynamics(
    *,
    events: list[dict[str, object]],
    trace: list[dict[str, object]],
    near_mixed_window: int = 3,
) -> dict[str, object]:
    repeat_events = [
        event for event in events
        if int(event.get("victim_preemptions_before", 0)) > 0
    ]
    distances = distances_to_nearest_mixed(events=repeat_events, trace=trace)
    near_mixed = [value for value in distances if value <= near_mixed_window]

    by_victim: dict[tuple[int, int], list[int]] = defaultdict(list)
    for event in repeat_events:
        key = (
            int(event.get("replica_id", -1)),
            int(event.get("victim_req_id", -1)),
        )
        by_victim[key].append(int(event.get("local_iter", -1)))
    intervals: list[float] = []
    for iters in by_victim.values():
        ordered = sorted(iters)
        intervals.extend(
            float(right - left)
            for left, right in zip(ordered, ordered[1:])
            if right > left
        )

    over_blocks = [
        float(event.get("over_blocks_before", 0))
        for event in events
        if "over_blocks_before" in event
    ]
    return {
        "total_preemptions": len(events),
        "repeat_preemptions": len(repeat_events),
        "repeat_share": len(repeat_events) / len(events) if events else 0.0,
        "near_mixed_count": len(near_mixed),
        "near_mixed_share": len(near_mixed) / len(distances) if distances else 0.0,
        "nearest_mixed_distance_median": statistics.median(distances) if distances else math.nan,
        "loop_interval_count": len(intervals),
        "loop_interval_median": statistics.median(intervals) if intervals else math.nan,
        "loop_interval_cv": _safe_cv(intervals),
        "over_blocks_median": statistics.median(over_blocks) if over_blocks else math.nan,
        "over_blocks_p90": _percentile(over_blocks, 90),
    }


def block_ledger(
    *,
    num_gpu_blocks: int,
    block_size: int,
    isl: int,
    osl: int,
    max_num_batched_tokens: int,
    real_capacity_blocks: int,
) -> dict[str, object]:
    prefill_chunk_tokens = min(isl, max_num_batched_tokens)
    prefill_chunk_blocks = math.ceil(prefill_chunk_tokens / block_size)
    full_request_blocks = math.ceil((isl + osl) / block_size)
    decode_growth_blocks = full_request_blocks - prefill_chunk_blocks
    capacity_ratio = num_gpu_blocks / real_capacity_blocks if real_capacity_blocks else math.nan
    return {
        "sim_capacity_blocks": num_gpu_blocks,
        "real_capacity_blocks": real_capacity_blocks,
        "capacity_ratio": capacity_ratio,
        "prefill_chunk_blocks": prefill_chunk_blocks,
        "decode_growth_blocks": decode_growth_blocks,
        "full_request_blocks": full_request_blocks,
        "capacity_full_request_ceiling": num_gpu_blocks / full_request_blocks,
        "block_deficit_detected": abs(capacity_ratio - 1.0) > 0.01,
    }


def decide_hypothesis(
    *,
    near_mixed_share: float,
    interval_cv: float,
    counterfactual_preemptions_per_request: float,
    block_deficit_detected: bool,
) -> dict[str, str]:
    """Classify the dominant thrash mechanism without using fit coefficients."""
    if block_deficit_detected or (
        counterfactual_preemptions_per_request >= 0.20
        and near_mixed_share < 0.50
    ):
        return {
            "hypothesis": "H1_block_deficit",
            "fix_gate": "ready_for_block_accounting_fix",
            "reason": "staggered lifecycle still preempts or block ledger is mismatched",
        }
    if near_mixed_share >= 0.80 and counterfactual_preemptions_per_request <= 0.05:
        return {
            "hypothesis": "H2_wave_overshoot",
            "fix_gate": "blocked_pending_wave_model",
            "reason": "preemptions cluster near mixed waves and disappear when lifecycle is staggered",
        }
    return {
        "hypothesis": "inconclusive",
        "fix_gate": "blocked_pending_more_forensics",
        "reason": "loop, ledger, and counterfactual do not identify one mechanism",
    }


def _make_decoding_request(request_id: int, *, isl: int, osl: int, generated_tokens: int):
    from aiconfigurator.sdk.backends.cb_simulator.datatypes import Request, RequestState

    req = Request(request_id=request_id, isl=isl, osl=osl, arrival_time_ms=0.0)
    req.state = RequestState.DECODING
    req.prefill_tokens_remaining = 0
    req.generated_tokens = max(1, min(generated_tokens, osl - 2))
    req.first_token_ms = 0.0
    return req


def run_staggered_counterfactual(
    *,
    scenario: str = SCENARIO,
    running_per_replica: int = 46,
    steps: int = 4096,
) -> dict[str, object]:
    """Run a no-wave decode-lifecycle counterfactual with uniform request offsets."""
    from aiconfigurator.sdk.backends.cb_simulator.datatypes import RequestState
    from aiconfigurator.sdk.backends.cb_simulator.scheduler import CBScheduler
    from aiconfigurator.sdk.backends.cb_simulator import scheduler as scheduler_module

    validate = _load_validate_module()
    point = next(pt for pt in validate.MULTI_CONFIG_DATA if pt.name == scenario)
    config = validate._make_multi_config_cb_config(
        point,
        overlap_factor=0.0,
        ep8_per_iteration_overhead_ms=validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
    )
    schedulers = [CBScheduler(config) for _ in range(point.dp)]
    waiting = [[] for _ in range(point.dp)]
    running = [[] for _ in range(point.dp)]
    next_id = 0
    for replica_id in range(point.dp):
        for idx in range(running_per_replica):
            generated = 1 + int(idx * max(point.osl - 2, 1) / running_per_replica)
            running[replica_id].append(
                _make_decoding_request(next_id, isl=point.isl, osl=point.osl, generated_tokens=generated)
            )
            next_id += 1

    original_preempt = scheduler_module.CBScheduler._preempt
    preemptions: list[dict[str, object]] = []
    scheduler_to_replica = {id(scheduler): idx for idx, scheduler in enumerate(schedulers)}
    local_iters: Counter[int] = Counter()

    def wrapped_preempt(self, victim, waiting_queue, running_queue, result, preempted_ids):
        replica_id = scheduler_to_replica.get(id(self), -1)
        preemptions.append(
            {
                "replica_id": replica_id,
                "local_iter": local_iters[replica_id],
                "victim_req_id": victim.request_id,
                "victim_preemptions_before": victim.num_preemptions,
                "victim_generated_tokens": victim.generated_tokens,
            }
        )
        return original_preempt(self, victim, waiting_queue, running_queue, result, preempted_ids)

    scheduler_module.CBScheduler._preempt = wrapped_preempt
    try:
        completed = 0
        for _ in range(steps):
            for replica_id, scheduler in enumerate(schedulers):
                local_iters[replica_id] += 1
                schedule = scheduler.schedule(waiting[replica_id], running[replica_id])
                for req in schedule.prefill_reqs:
                    tokens = schedule.prefill_tokens[req.request_id]
                    if req.state in {RequestState.WAITING, RequestState.PREEMPTED}:
                        req.state = RequestState.PREFILLING
                        if req in waiting[replica_id]:
                            waiting[replica_id].remove(req)
                        running[replica_id].append(req)
                    req.prefill_tokens_remaining -= tokens
                    if req.prefill_tokens_remaining <= 0:
                        req.prefill_tokens_remaining = 0
                        req.state = RequestState.DECODING
                done = []
                for req in schedule.decode_reqs:
                    req.generated_tokens += 1
                    if req.generated_tokens >= req.osl - 1:
                        done.append(req)
                for req in done:
                    if req in running[replica_id]:
                        running[replica_id].remove(req)
                    completed += 1
                    generated = 1 + ((next_id * 37) % max(point.osl - 2, 1))
                    running[replica_id].append(
                        _make_decoding_request(
                            next_id,
                            isl=point.isl,
                            osl=point.osl,
                            generated_tokens=generated,
                        )
                    )
                    next_id += 1
    finally:
        scheduler_module.CBScheduler._preempt = original_preempt

    request_slots = running_per_replica * point.dp
    return {
        "preemptions": len(preemptions),
        "preemptions_per_request_slot": len(preemptions) / request_slots if request_slots else 0.0,
        "completed": completed,
        "request_slots": request_slots,
        "sample_events": preemptions[:20],
    }


def build_report_rows() -> list[dict[str, object]]:
    phase451d = _load_phase451d_module()
    validate = _load_validate_module()
    point = next(pt for pt in validate.MULTI_CONFIG_DATA if pt.name == SCENARIO)
    config = validate._make_multi_config_cb_config(
        point,
        overlap_factor=0.0,
        ep8_per_iteration_overhead_ms=validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
    )

    sim = phase451d.run_sim_preemption_forensics(
        scenario=SCENARIO,
        num_requests=DEFAULT_NUM_REQUESTS,
    )
    events = list(sim["events"])
    trace = list(sim["trace"])
    loop = summarize_loop_dynamics(events=events, trace=trace)
    ledger = block_ledger(
        num_gpu_blocks=config.num_gpu_blocks,
        block_size=config.block_size,
        isl=point.isl,
        osl=point.osl,
        max_num_batched_tokens=config.max_num_batched_tokens,
        real_capacity_blocks=28633,
    )
    counterfactual = run_staggered_counterfactual(scenario=SCENARIO)
    verdict = decide_hypothesis(
        near_mixed_share=float(loop["near_mixed_share"]),
        interval_cv=float(loop["loop_interval_cv"]),
        counterfactual_preemptions_per_request=float(
            counterfactual["preemptions_per_request_slot"]
        ),
        block_deficit_detected=bool(ledger["block_deficit_detected"]),
    )

    rows = [
        _row("f1_loop", "total_preemptions", loop["total_preemptions"]),
        _row("f1_loop", "repeat_share", loop["repeat_share"], target="dominant repeat-victim signature"),
        _row("f1_loop", "near_mixed_share", loop["near_mixed_share"], target=">=0.80 supports H2"),
        _row("f1_loop", "nearest_mixed_distance_median", loop["nearest_mixed_distance_median"]),
        _row("f1_loop", "loop_interval_count", loop["loop_interval_count"]),
        _row("f1_loop", "loop_interval_median", loop["loop_interval_median"]),
        _row("f1_loop", "loop_interval_cv", loop["loop_interval_cv"], target="low supports H1; high supports H2"),
        _row("f1_loop", "over_blocks_median", loop["over_blocks_median"]),
        _row("f1_loop", "over_blocks_p90", loop["over_blocks_p90"]),
        _row("f2_ledger", "sim_capacity_blocks", ledger["sim_capacity_blocks"], target="28633"),
        _row("f2_ledger", "real_capacity_blocks", ledger["real_capacity_blocks"], target="28633"),
        _row("f2_ledger", "capacity_ratio", ledger["capacity_ratio"], target="1.0"),
        _row("f2_ledger", "prefill_chunk_blocks", ledger["prefill_chunk_blocks"], target="8000/16=500"),
        _row("f2_ledger", "decode_growth_blocks", ledger["decode_growth_blocks"], target="about 120-125"),
        _row("f2_ledger", "full_request_blocks", ledger["full_request_blocks"], target="(8000+2000)/16=625"),
        _row("f2_ledger", "capacity_full_request_ceiling", ledger["capacity_full_request_ceiling"]),
        _row("f2_ledger", "block_deficit_detected", ledger["block_deficit_detected"], target="false for H2"),
        _row("f3_counterfactual", "staggered_preemptions", counterfactual["preemptions"], target="0 supports H2"),
        _row(
            "f3_counterfactual",
            "staggered_preemptions_per_request_slot",
            counterfactual["preemptions_per_request_slot"],
            target="<=0.05 supports H2; >=0.20 supports H1",
        ),
        _row("decision", "hypothesis", verdict["hypothesis"], status="pass" if verdict["hypothesis"] != "inconclusive" else "blocked", note=verdict["reason"]),
        _row("decision", "phase451f_fix_gate", verdict["fix_gate"], status="blocked" if verdict["fix_gate"].startswith("blocked") else "open"),
        _row("decision", "phase451e_runtime_patch", "not_applied", status="blocked", note="report-only F1-F3; H2 requires wave model design before runtime changes"),
    ]
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
    hypothesis = by_metric.get("hypothesis", {}).get("value", "unknown")
    fix_gate = by_metric.get("phase451f_fix_gate", {}).get("value", "unknown")
    lines = [
        "# Phase451-F thrash dynamics",
        "",
        (
            f"结论: `{hypothesis}`。F1-F3 只做动力学判别;fix gate=`{fix_gate}`。"
            "本报告不改 scheduler/runtime/PerfDB/validate gate。"
        ),
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
            "- Step 1-3 are report-only.",
            "- H2 does not authorize a runtime patch by itself; it requires a non-parametric wave model design and a separate red-green phase.",
            "- Default AIC remains No-Go.",
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
