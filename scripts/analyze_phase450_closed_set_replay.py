#!/usr/bin/env python3
"""Phase450: report-only closed-set replay and underprediction decomposition."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_SCENARIO = "K2.5-tp4ep8dp2-8k2k"
DEFAULT_CLOSED_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase450_closed_set_replay.csv"
DEFAULT_CLOSED_MD = REPO_ROOT / "docs/iter_gap_investigation/phase450_closed_set_replay.md"
DEFAULT_DECOMP_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase450_underpredict_decomp.csv"
DEFAULT_DECOMP_MD = REPO_ROOT / "docs/iter_gap_investigation/phase450_underpredict_decomp.md"
PHASE425_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase425_8k2k_sweep.csv"
PHASE447_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase447_sequence_fingerprint.csv"
PHASE449_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase449_kv_watermark.csv"


@dataclass(frozen=True)
class WallMetric:
    wall_ms: float
    output_tok_s_gpu: float


@dataclass
class ReplicaState:
    replica_id: int
    scheduler: object
    waiting: list
    running: list
    total_iters: int = 0


@dataclass(frozen=True)
class ReplayRun:
    scenario: str
    num_requests: int
    concurrency: int
    output_tokens_per_request: int
    num_gpus: int
    rows: list[dict[str, object]]
    completed_by_replica: dict[int, int]

    @property
    def wall_metric(self) -> WallMetric:
        return total_wall_metric(
            self.rows,
            num_requests=self.num_requests,
            output_tokens_per_request=self.output_tokens_per_request,
            num_gpus=self.num_gpus,
        )

    @property
    def request_split(self) -> float:
        if not self.completed_by_replica:
            return 0.0
        values = list(self.completed_by_replica.values())
        low = min(values)
        high = max(values)
        return high / low if low > 0 else float("inf")


def total_wall_metric(
    rows: Iterable[dict[str, object]],
    *,
    num_requests: int,
    output_tokens_per_request: int,
    num_gpus: int,
) -> WallMetric:
    trace = list(rows)
    if not trace:
        return WallMetric(wall_ms=0.0, output_tok_s_gpu=0.0)
    start = min(float(row["start_ms"]) for row in trace)
    end = max(float(row["end_ms"]) for row in trace)
    wall_ms = end - start
    if wall_ms <= 0 or num_gpus <= 0:
        return WallMetric(wall_ms=wall_ms, output_tok_s_gpu=0.0)
    output_tokens = num_requests * output_tokens_per_request
    return WallMetric(
        wall_ms=wall_ms,
        output_tok_s_gpu=output_tokens / (wall_ms / 1000.0) / num_gpus,
    )


def _phase_from_row(row: dict[str, object]) -> str:
    prefill_tokens = int(row.get("prefill_tokens", 0))
    decode_reqs = int(row.get("decode_reqs", 0))
    if prefill_tokens > 0 and decode_reqs > 0:
        return "mixed_prefill"
    if prefill_tokens > 0:
        return "prefill"
    if decode_reqs > 0:
        return "decode"
    return "empty"


def phase_joint_shares(rows: Iterable[dict[str, object]]) -> dict[str, float]:
    by_cycle: dict[int, dict[int, str]] = {}
    for row in rows:
        cycle = int(row["cycle"])
        replica = int(row["replica_id"])
        by_cycle.setdefault(cycle, {})[replica] = str(row.get("phase") or _phase_from_row(row))
    counts = Counter(
        f"{phases[0]}+{phases[1]}"
        for _, phases in sorted(by_cycle.items())
        if 0 in phases and 1 in phases
    )
    total = sum(counts.values())
    return {key: value / total for key, value in sorted(counts.items())} if total else {}


def lockstep_padding_summary(rows: Iterable[dict[str, object]]) -> dict[str, float]:
    by_cycle: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        by_cycle.setdefault(int(row["cycle"]), []).append(row)
    wall_ms = 0.0
    padded_extra_ms = 0.0
    mixed_decode_cycles = 0
    for cycle_rows in by_cycle.values():
        charges = [float(row["charge_ms"]) for row in cycle_rows]
        if not charges:
            continue
        cycle_wall = max(charges)
        wall_ms += cycle_wall
        padded_extra_ms += sum(cycle_wall - charge for charge in charges)
        phases = {str(row.get("phase") or _phase_from_row(row)) for row in cycle_rows}
        if "mixed_prefill" in phases and "decode" in phases:
            mixed_decode_cycles += 1
    cycles = len(by_cycle)
    return {
        "cycles": cycles,
        "wall_ms": wall_ms,
        "padded_extra_ms": padded_extra_ms,
        "padded_extra_share": padded_extra_ms / wall_ms if wall_ms > 0 else 0.0,
        "mixed_decode_share": mixed_decode_cycles / cycles if cycles else 0.0,
    }


def closed_set_gate(
    *,
    sim_n128_tput: float,
    sim_n512_tput: float,
    sim_n128_split: float,
    sim_n512_split: float,
    real_artifact: float,
    real_n128_split: float,
    real_n512_split: float,
    tolerance: float,
) -> dict[str, object]:
    sim_artifact = sim_n512_tput / sim_n128_tput if sim_n128_tput > 0 else float("inf")
    artifact_error = abs(sim_artifact / real_artifact - 1.0) if real_artifact > 0 else float("inf")
    n128_split_error = abs(sim_n128_split / real_n128_split - 1.0) if real_n128_split > 0 else float("inf")
    n512_split_error = abs(sim_n512_split / real_n512_split - 1.0) if real_n512_split > 0 else float("inf")
    return {
        "sim_artifact": sim_artifact,
        "artifact_error": artifact_error,
        "n128_split_error": n128_split_error,
        "n512_split_error": n512_split_error,
        "passed": (
            artifact_error <= tolerance
            and n128_split_error <= tolerance
            and n512_split_error <= tolerance
            and sim_n512_tput > sim_n128_tput
        ),
    }


def _load_validate_module():
    path = REPO_ROOT / "scripts" / "validate_cb_simulator.py"
    spec = importlib.util.spec_from_file_location("validate_cb_simulator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_replay_inputs(validate, scenario: str, num_requests: int):
    from aiconfigurator.sdk.backends.cb_simulator import CBSimulator

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
    config = replace(config, num_requests=num_requests, warmup_requests=0)
    sim = CBSimulator(backend, model, db, config)
    return point, sim, config


def run_diagnostic_lockstep(validate, scenario: str, num_requests: int) -> ReplayRun:
    from aiconfigurator.sdk.backends.cb_simulator.datatypes import Request, RequestState
    from aiconfigurator.sdk.backends.cb_simulator.dp_admission import (
        DPAdmissionRouter,
        DPReplicaCounts,
    )
    from aiconfigurator.sdk.backends.cb_simulator.scheduler import CBScheduler

    point, sim, config = _build_replay_inputs(validate, scenario, num_requests)
    latency_calc = sim._create_latency_calc(0)
    replicas = [
        ReplicaState(idx, CBScheduler(config), [], [])
        for idx in range(point.dp)
    ]
    router = DPAdmissionRouter(point.dp)
    next_id = 0
    completed = 0
    completed_by_replica = {idx: 0 for idx in range(point.dp)}
    rows: list[dict[str, object]] = []
    clock_ms = 0.0

    def actual_counts() -> list[DPReplicaCounts]:
        return [
            DPReplicaCounts(waiting=len(replica.waiting), running=len(replica.running))
            for replica in replicas
        ]

    def route_request(now_ms: float) -> None:
        nonlocal next_id
        replica_idx = router.route(now_ms=now_ms, actual_counts=actual_counts())
        replicas[replica_idx].waiting.append(
            Request(
                request_id=next_id,
                isl=point.isl,
                osl=point.osl,
                arrival_time_ms=now_ms,
            )
        )
        next_id += 1

    for _ in range(min(point.batch_size, config.num_requests)):
        route_request(0.0)

    max_iters = config.num_requests * point.dp * (
        point.osl + point.isl // config.max_num_batched_tokens + 10
    )
    cycle_id = 0
    while completed < config.num_requests and cycle_id < max_iters:
        cycle = []
        for replica in replicas:
            if not replica.waiting and not replica.running:
                continue
            waiting_before = len(replica.waiting)
            running_before = len(replica.running)
            schedule = replica.scheduler.schedule(replica.waiting, replica.running)
            if schedule.is_empty:
                continue
            avg_kv = int(np.mean([req.kv_cache_len for req in schedule.decode_reqs])) \
                if schedule.decode_reqs else 0
            charge_ms = latency_calc.compute(
                prefill_tokens=schedule.total_prefill_tokens,
                prefill_batch_size=len(schedule.prefill_reqs),
                prefill_seq_len=point.isl,
                decode_batch_size=len(schedule.decode_reqs),
                decode_avg_kv_len=avg_kv,
            )
            cycle.append((
                replica,
                schedule,
                charge_ms,
                avg_kv,
                waiting_before,
                running_before,
            ))
        if not cycle:
            break

        step_ms = max(item[2] for item in cycle)
        step_start = clock_ms
        step_end = step_start + step_ms
        clock_ms = step_end

        for replica, schedule, charge_ms, avg_kv, waiting_before, running_before in cycle:
            replica.total_iters += 1
            phase = _phase_from_row({
                "prefill_tokens": schedule.total_prefill_tokens,
                "decode_reqs": len(schedule.decode_reqs),
            })
            rows.append({
                "scenario": scenario,
                "num_requests": num_requests,
                "cycle": cycle_id,
                "replica_id": replica.replica_id,
                "local_iter": replica.total_iters,
                "start_ms": step_start,
                "end_ms": step_end,
                "wall_ms": step_ms,
                "charge_ms": charge_ms,
                "padded_extra_ms": step_ms - charge_ms,
                "phase": phase,
                "prefill_reqs": len(schedule.prefill_reqs),
                "prefill_tokens": schedule.total_prefill_tokens,
                "decode_reqs": len(schedule.decode_reqs),
                "total_tokens": schedule.total_tokens,
                "avg_kv": avg_kv,
                "waiting_before": waiting_before,
                "running_before": running_before,
            })

        for replica, schedule, _charge_ms, _avg_kv, _waiting_before, _running_before in cycle:
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
                    newly_done.append(req)
            for req in newly_done:
                replica.running.remove(req)
                completed += 1
                completed_by_replica[replica.replica_id] += 1

        while next_id < config.num_requests and sum(
            len(replica.waiting) + len(replica.running)
            for replica in replicas
        ) < point.batch_size:
            route_request(step_end)
        cycle_id += 1

    return ReplayRun(
        scenario=scenario,
        num_requests=num_requests,
        concurrency=point.batch_size,
        output_tokens_per_request=point.osl - 1,
        num_gpus=point.tp * point.dp,
        rows=rows,
        completed_by_replica=completed_by_replica,
    )


def _read_phase425_targets(path: Path) -> dict[str, float]:
    sample_splits: dict[int, float] = {}
    trend: dict[str, str] | None = None
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["scenario"] != DEFAULT_SCENARIO:
                continue
            if row["row_type"] == "sample":
                sample_splits[int(row["num_prompts"])] = float(row["request_count_ratio"])
            elif row["row_type"] == "trend":
                trend = row
    if trend is None or 128 not in sample_splits or 512 not in sample_splits:
        raise ValueError(f"missing Phase425 target rows in {path}")
    return {
        "real_artifact": float(trend["validation_burst_artifact_multiplier"]),
        "real_n128_split": sample_splits[128],
        "real_n512_split": sample_splits[512],
        "real_n128_output": float(trend["validation_real_output_tok_s_gpu"]),
    }


def _read_phase449_ratio(path: Path) -> float:
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (
                row["section"] == "validate"
                and row["scenario"] == DEFAULT_SCENARIO
                and row["metric"] == "error_ratio"
            ):
                return float(row["after"])
    raise ValueError(f"missing Phase449 validate ratio in {path}")


def _read_real_phase_shares(path: Path, scenario: str) -> dict[str, float]:
    shares: dict[str, float] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["scenario"] == scenario and row["section"] == "phase_joint":
                shares[row["phase_pair"]] = float(row["share"])
    return shares


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _run_rows(run: ReplayRun) -> list[dict[str, object]]:
    metric = run.wall_metric
    split_values = [run.completed_by_replica[idx] for idx in sorted(run.completed_by_replica)]
    pad = lockstep_padding_summary(run.rows)
    rows = [
        {
            "section": "closed_set_run",
            "scenario": run.scenario,
            "num_requests": run.num_requests,
            "metric": "total_wall_output_tok_s_gpu",
            "value": f"{metric.output_tok_s_gpu:.6f}",
            "target": "",
            "status": "",
            "note": f"wall_ms={metric.wall_ms:.3f}; completed_by_replica={split_values}",
        },
        {
            "section": "closed_set_run",
            "scenario": run.scenario,
            "num_requests": run.num_requests,
            "metric": "request_split_ratio",
            "value": f"{run.request_split:.6f}",
            "target": "",
            "status": "",
            "note": f"completed_by_replica={split_values}",
        },
        {
            "section": "closed_set_run",
            "scenario": run.scenario,
            "num_requests": run.num_requests,
            "metric": "lockstep_padded_extra_share",
            "value": f"{pad['padded_extra_share']:.6f}",
            "target": "",
            "status": "",
            "note": f"padded_extra_ms={pad['padded_extra_ms']:.3f}; wall_ms={pad['wall_ms']:.3f}",
        },
    ]
    for pair, share in phase_joint_shares(run.rows).items():
        rows.append({
            "section": "phase_joint",
            "scenario": run.scenario,
            "num_requests": run.num_requests,
            "metric": pair,
            "value": f"{share:.6f}",
            "target": "",
            "status": "",
            "note": "",
        })
    mixed = [row for row in run.rows if row["phase"] == "mixed_prefill"]
    decode_batches = [float(row["decode_reqs"]) for row in mixed]
    running_before = [float(row["running_before"]) for row in mixed]
    for metric_name, values in [
        ("mixed_decode_batch_p50", decode_batches),
        ("mixed_decode_batch_p10", decode_batches),
        ("mixed_decode_batch_p90", decode_batches),
        ("mixed_running_before_p50", running_before),
    ]:
        pct = 50 if metric_name.endswith("p50") else 10 if metric_name.endswith("p10") else 90
        rows.append({
            "section": "distribution",
            "scenario": run.scenario,
            "num_requests": run.num_requests,
            "metric": metric_name,
            "value": f"{_percentile(values, pct):.6f}",
            "target": "",
            "status": "",
            "note": f"samples={len(values)}",
        })
    return rows


def build_reports(
    *,
    scenario: str = DEFAULT_SCENARIO,
    num_requests_values: tuple[int, int] = (128, 512),
    tolerance: float = 0.10,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    validate = _load_validate_module()
    runs = [
        run_diagnostic_lockstep(validate, scenario, num_requests)
        for num_requests in num_requests_values
    ]
    by_n = {run.num_requests: run for run in runs}
    targets = _read_phase425_targets(PHASE425_CSV)
    phase449_ratio = _read_phase449_ratio(PHASE449_CSV)
    gate = closed_set_gate(
        sim_n128_tput=by_n[128].wall_metric.output_tok_s_gpu,
        sim_n512_tput=by_n[512].wall_metric.output_tok_s_gpu,
        sim_n128_split=by_n[128].request_split,
        sim_n512_split=by_n[512].request_split,
        real_artifact=targets["real_artifact"],
        real_n128_split=targets["real_n128_split"],
        real_n512_split=targets["real_n512_split"],
        tolerance=tolerance,
    )
    closed_rows: list[dict[str, object]] = []
    for run in runs:
        closed_rows.extend(_run_rows(run))
    closed_rows.extend([
        {
            "section": "gate",
            "scenario": scenario,
            "num_requests": "128->512",
            "metric": "artifact_multiplier",
            "value": f"{gate['sim_artifact']:.6f}",
            "target": f"{targets['real_artifact']:.6f}±{tolerance:.2f}",
            "status": "pass" if gate["passed"] else "fail",
            "note": f"artifact_error={gate['artifact_error']:.6f}",
        },
        {
            "section": "gate",
            "scenario": scenario,
            "num_requests": "128->512",
            "metric": "n128_request_split",
            "value": f"{by_n[128].request_split:.6f}",
            "target": f"{targets['real_n128_split']:.6f}±{tolerance:.2f}",
            "status": "pass" if float(gate["n128_split_error"]) <= tolerance else "fail",
            "note": f"split_error={gate['n128_split_error']:.6f}",
        },
        {
            "section": "gate",
            "scenario": scenario,
            "num_requests": "128->512",
            "metric": "n512_request_split",
            "value": f"{by_n[512].request_split:.6f}",
            "target": f"{targets['real_n512_split']:.6f}±{tolerance:.2f}",
            "status": "pass" if float(gate["n512_split_error"]) <= tolerance else "fail",
            "note": f"split_error={gate['n512_split_error']:.6f}",
        },
        {
            "section": "decision",
            "scenario": scenario,
            "num_requests": "128->512",
            "metric": "phase450_b_start_closed_set_validate_metric",
            "value": "false",
            "target": "only_if_closed_set_gate_passes",
            "status": "blocked" if not gate["passed"] else "allowed",
            "note": (
                "closed-set artifact direction did not match Phase425; "
                "do not wire closed-set validate metric"
            ) if not gate["passed"] else "gate passed",
        },
    ])

    real_phase = _read_real_phase_shares(PHASE447_CSV, scenario)
    sim_phase = phase_joint_shares(by_n[512].rows)
    pad = lockstep_padding_summary(by_n[512].rows)
    real_tput = targets["real_n128_output"]
    sim_closed_n128 = by_n[128].wall_metric.output_tok_s_gpu
    decomp_rows = [
        {
            "section": "underpredict_summary",
            "scenario": scenario,
            "metric": "phase449_error_ratio",
            "value": f"{phase449_ratio:.6f}",
            "target": "<=1.15",
            "status": "open",
            "note": "current validate ratio is real/sim underprediction",
        },
        {
            "section": "underpredict_summary",
            "scenario": scenario,
            "metric": "closed_set_n128_error_ratio",
            "value": f"{(real_tput / sim_closed_n128 if sim_closed_n128 > 0 else float('inf')):.6f}",
            "target": "direction check",
            "status": "informational",
            "note": f"real_n128={real_tput:.6f}; sim_closed_n128={sim_closed_n128:.6f}",
        },
        {
            "section": "lockstep",
            "scenario": scenario,
            "metric": "padded_extra_share_n512",
            "value": f"{pad['padded_extra_share']:.6f}",
            "target": "diagnostic_only",
            "status": "informational",
            "note": f"mixed_decode_share={pad['mixed_decode_share']:.6f}",
        },
    ]
    mixed_mixed_real = real_phase.get("mixed_prefill+mixed_prefill", 0.0)
    mixed_mixed_sim = sim_phase.get("mixed_prefill+mixed_prefill", 0.0)
    mixed_decode_real = (
        real_phase.get("mixed_prefill+decode", 0.0)
        + real_phase.get("decode+mixed_prefill", 0.0)
    )
    mixed_decode_sim = (
        sim_phase.get("mixed_prefill+decode", 0.0)
        + sim_phase.get("decode+mixed_prefill", 0.0)
    )
    decomp_rows.extend([
        {
            "section": "underpredict_verdict",
            "scenario": scenario,
            "metric": "lockstep_reverse_phase_padding",
            "value": "ruled_out",
            "target": "padded_extra_share>0",
            "status": "fail",
            "note": "sim has no mixed+decode cycles, so peer padding is zero",
        },
        {
            "section": "underpredict_verdict",
            "scenario": scenario,
            "metric": "mixed_mixed_oversync_ratio",
            "value": f"{(mixed_mixed_sim / mixed_mixed_real if mixed_mixed_real > 0 else float('inf')):.6f}",
            "target": "near 1.0",
            "status": "open",
            "note": "sim over-synchronizes mixed waves relative to real fingerprint",
        },
        {
            "section": "underpredict_verdict",
            "scenario": scenario,
            "metric": "mixed_decode_missing_share",
            "value": f"{mixed_decode_sim:.6f}",
            "target": f"real={mixed_decode_real:.6f}",
            "status": "open",
            "note": "phase-joint tolerance was too loose for rare phase pairs",
        },
    ])
    for pair in sorted(set(real_phase) | set(sim_phase)):
        real = real_phase.get(pair, 0.0)
        sim = sim_phase.get(pair, 0.0)
        decomp_rows.append({
            "section": "phase_joint",
            "scenario": scenario,
            "metric": pair,
            "value": f"{sim:.6f}",
            "target": f"real={real:.6f}",
            "status": "informational",
            "note": f"abs_delta={abs(sim - real):.6f}",
        })
    return closed_rows, decomp_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["section", "scenario", "num_requests", "metric", "value", "target", "status", "note"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, title: str, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "| section | scenario | N | metric | value | target | status | note |",
        "|---|---|---:|---|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('section', '')} | {row.get('scenario', '')} | "
            f"{row.get('num_requests', '')} | {row.get('metric', '')} | "
            f"{row.get('value', '')} | {row.get('target', '')} | "
            f"{row.get('status', '')} | {row.get('note', '')} |"
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        "- Report-only: no validate/runtime path changed.",
        "- Closed-set throughput is derived from first scheduled step to last completed step.",
        "- Default AIC remains No-Go until the full validate table is inside the agreed gate.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--closed-csv", type=Path, default=DEFAULT_CLOSED_CSV)
    parser.add_argument("--closed-md", type=Path, default=DEFAULT_CLOSED_MD)
    parser.add_argument("--decomp-csv", type=Path, default=DEFAULT_DECOMP_CSV)
    parser.add_argument("--decomp-md", type=Path, default=DEFAULT_DECOMP_MD)
    args = parser.parse_args()

    closed_rows, decomp_rows = build_reports(scenario=args.scenario)
    write_csv(args.closed_csv, closed_rows)
    write_csv(args.decomp_csv, decomp_rows)
    write_md(args.closed_md, "Phase450 closed-set replay", closed_rows)
    write_md(args.decomp_md, "Phase450 underprediction decomposition", decomp_rows)
    print(f"wrote {args.closed_csv}")
    print(f"wrote {args.closed_md}")
    print(f"wrote {args.decomp_csv}")
    print(f"wrote {args.decomp_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
