#!/usr/bin/env python3
"""Phase407: run a genuine two-replica DP lockstep joint simulation."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aiconfigurator.sdk.backends.cb_simulator import CBSimulator  # noqa: E402
from aiconfigurator.sdk.backends.cb_simulator.datatypes import (  # noqa: E402
    CBSimConfig,
    Request,
    RequestState,
    ScheduleResult,
)
from aiconfigurator.sdk.backends.cb_simulator.iteration_latency import (  # noqa: E402
    IterationLatencyBreakdown,
    IterationLatencyCalculator,
)
from aiconfigurator.sdk.backends.cb_simulator.scheduler import CBScheduler  # noqa: E402
from scripts import validate_cb_simulator as validate_cb  # noqa: E402


SOURCE = "phase407_dp_lockstep_joint_sim"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase407_dp_lockstep_joint_sim.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase407_dp_lockstep_joint_sim.md"
PHASE400_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase400_clean_recollect.csv"
PHASE401_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase401_preempt_admission.csv"
PHASE403_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase403_dp2_running_batch.csv"
DEFAULT_READINESS = "No-Go"

DP2_SCENARIOS = (
    "K2.5-tp4ep8dp2-8k2k",
    "K2.5-tp4ep8dp2-32k3k",
)
TP8_CONTROL = "K2.5-tp8ep8-32k3k"
ALL_SCENARIOS = DP2_SCENARIOS + (TP8_CONTROL,)
PENALTY_TARGETS = {
    "K2.5-tp4ep8dp2-8k2k": 1.78,
    "K2.5-tp4ep8dp2-32k3k": 1.88,
}

CSV_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "tp",
    "dp",
    "ep",
    "isl",
    "osl",
    "max_num_batched_tokens",
    "num_gpu_blocks",
    "real_output_tok_s_gpu",
    "phase401_sim_output_tok_s_gpu",
    "uncoupled_joint_output_tok_s_gpu",
    "uncoupled_vs_phase401_error_pct",
    "uncoupled_harness_gate",
    "coupled_joint_output_tok_s_gpu",
    "coupled_ratio",
    "coupled_error_pct",
    "coupled_convergence_gate",
    "joint_penalty",
    "penalty_target",
    "penalty_error_pct",
    "penalty_gate",
    "joint_avg_decode_reqs_global",
    "joint_peak_decode_reqs_global",
    "real_running_global_mean",
    "occupancy_error_pct",
    "occupancy_gate",
    "tp8_early_exit_gate",
    "steady_state_iterations",
    "steady_state_time_ms",
    "phase405_penalty_read",
    "paper_model_rule",
    "mechanism_verdict",
    "phase408_target",
    "gpu_allowed",
    "ssh_allowed",
    "runtime_modified",
    "perf_database",
    "valid_for_default",
    "diagnostic_only",
    "default_readiness",
]


@dataclass(frozen=True)
class JointSimResult:
    output_tok_s_gpu: float
    avg_decode_reqs_global: float
    peak_decode_reqs_global: int
    steady_state_iterations: int
    steady_state_time_ms: float


@dataclass
class _ReplicaRuntime:
    scheduler: CBScheduler
    latency_calc: IterationLatencyCalculator
    waiting: list[Request]
    running: list[Request]
    completed: list[Request]
    next_id: int

    @property
    def has_external_supply(self) -> bool:
        return bool(self.waiting) or self.next_id < self.scheduler._config.num_requests


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def _read_csv_by_scenario(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        return {row["scenario"]: row for row in csv.DictReader(f)}


def _point_by_name() -> dict[str, validate_cb.MultiConfigPoint]:
    return {point.name: point for point in validate_cb.MULTI_CONFIG_DATA}


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return math.inf
    return numerator / denominator


def _error_pct(predicted: float, target: float) -> float:
    return abs(_safe_ratio(predicted, target) - 1.0) * 100.0


def _gate(error_pct: float, threshold_pct: float) -> str:
    return "passed" if error_pct <= threshold_pct else "failed"


def _make_config(point: validate_cb.MultiConfigPoint, num_gpu_blocks: int) -> CBSimConfig:
    return validate_cb._make_cb_config(
        point.isl,
        point.batch_size,
        max_num_batched_tokens=point.max_num_batched_tokens,
        overlap_factor=0.0,
        per_iteration_overhead_ms=0.0,
        num_gpu_blocks=num_gpu_blocks,
    )


def _phase401_num_gpu_blocks(phase401: dict[str, dict[str, str]], scenario: str) -> int:
    try:
        return int(float(phase401[scenario]["num_gpu_blocks"]))
    except KeyError as exc:
        raise ValueError(f"missing Phase401 num_gpu_blocks for {scenario}") from exc


def _make_replica(
    *,
    backend,
    model,
    database,
    config: CBSimConfig,
    isl: int,
    osl: int,
    concurrency: int,
) -> _ReplicaRuntime:
    waiting = [
        Request(request_id=request_id, isl=isl, osl=osl, arrival_time_ms=0.0)
        for request_id in range(min(concurrency, config.num_requests))
    ]
    return _ReplicaRuntime(
        scheduler=CBScheduler(config),
        latency_calc=IterationLatencyCalculator(
            backend,
            model,
            database,
            overlap_factor=config.overlap_factor,
            per_iteration_overhead_ms=config.per_iteration_overhead_ms,
        ),
        waiting=waiting,
        running=[],
        completed=[],
        next_id=len(waiting),
    )


def _avg_kv(schedule: ScheduleResult) -> int:
    if not schedule.decode_reqs:
        return 0
    return int(sum(req.kv_cache_len for req in schedule.decode_reqs) / len(schedule.decode_reqs))


def _compute_breakdown(
    latency_calc: IterationLatencyCalculator,
    *,
    schedule: ScheduleResult,
    isl: int,
) -> IterationLatencyBreakdown:
    latency_calc.compute(
        prefill_tokens=schedule.total_prefill_tokens,
        prefill_batch_size=len(schedule.prefill_reqs),
        prefill_seq_len=isl,
        decode_batch_size=len(schedule.decode_reqs),
        decode_avg_kv_len=_avg_kv(schedule),
    )
    breakdown = latency_calc.get_last_breakdown()
    if breakdown is None:
        raise ValueError("missing iteration latency breakdown")
    return breakdown


def _attention_ms(breakdown: IterationLatencyBreakdown) -> float:
    return breakdown.context_attention_ms + breakdown.generation_attention_ms


def _padded_non_attention_ms(
    latency_calc: IterationLatencyCalculator,
    *,
    padded_tokens: int,
    has_prefill: bool,
    isl: int,
    decode_avg_kv_len: int,
) -> float:
    if padded_tokens <= 0:
        return 0.0
    if has_prefill:
        latency_calc.compute(
            prefill_tokens=padded_tokens,
            prefill_batch_size=1,
            prefill_seq_len=isl,
            decode_batch_size=0,
            decode_avg_kv_len=0,
        )
        breakdown = latency_calc.get_last_breakdown()
        if breakdown is None:
            raise ValueError("missing padded prefill breakdown")
        return breakdown.context_non_attention_ms
    latency_calc.compute(
        prefill_tokens=0,
        prefill_batch_size=0,
        prefill_seq_len=1,
        decode_batch_size=padded_tokens,
        decode_avg_kv_len=decode_avg_kv_len,
    )
    breakdown = latency_calc.get_last_breakdown()
    if breakdown is None:
        raise ValueError("missing padded decode breakdown")
    return breakdown.generation_non_attention_ms


def _apply_schedule(
    replica: _ReplicaRuntime,
    *,
    schedule: ScheduleResult,
    clock_ms: float,
    iter_lat: float,
    in_steady_state: bool,
) -> int:
    steady_output_tokens = 0
    for req in schedule.prefill_reqs:
        tokens = schedule.prefill_tokens[req.request_id]
        if req.state in {RequestState.WAITING, RequestState.PREEMPTED}:
            req.state = RequestState.PREFILLING
            if req.prefill_start_ms < 0:
                req.prefill_start_ms = clock_ms - iter_lat
            if req in replica.waiting:
                replica.waiting.remove(req)
            replica.running.append(req)
        req.prefill_tokens_remaining -= tokens
        if req.prefill_tokens_remaining <= 0:
            req.prefill_tokens_remaining = 0
            if req.first_token_ms < 0:
                req.first_token_ms = clock_ms
            req.state = RequestState.DECODING

    newly_done: list[Request] = []
    for req in schedule.decode_reqs:
        req.generated_tokens += 1
        if in_steady_state:
            steady_output_tokens += 1
        if req.generated_tokens >= req.osl - 1:
            req.state = RequestState.DONE
            req.finish_ms = clock_ms
            newly_done.append(req)

    for req in newly_done:
        if req in replica.running:
            replica.running.remove(req)
        replica.completed.append(req)
        config = replica.scheduler._config
        if replica.next_id < config.num_requests:
            replica.waiting.append(
                Request(
                    request_id=replica.next_id,
                    isl=req.isl,
                    osl=req.osl,
                    arrival_time_ms=clock_ms,
                )
            )
            replica.next_id += 1
    return steady_output_tokens


def run_uncoupled_reference(
    *,
    backend,
    model,
    database,
    config: CBSimConfig,
    point: validate_cb.MultiConfigPoint,
) -> JointSimResult:
    per_replica_concurrency = int(math.ceil(point.batch_size / point.dp))
    result = CBSimulator(backend, model, database, config).run(
        isl=point.isl,
        osl=point.osl,
        concurrency=per_replica_concurrency,
        num_gpus=point.tp,
    )
    return JointSimResult(
        output_tok_s_gpu=result.throughput_tok_s_gpu,
        avg_decode_reqs_global=result.avg_decode_reqs_per_iter * point.dp,
        peak_decode_reqs_global=result.peak_decode_reqs_per_iter * point.dp,
        steady_state_iterations=result.steady_state_iterations,
        steady_state_time_ms=result.steady_state_time_ms,
    )


def run_coupled_joint_sim(
    *,
    backend,
    model,
    database,
    config: CBSimConfig,
    point: validate_cb.MultiConfigPoint,
) -> JointSimResult:
    if point.dp == 1:
        return run_uncoupled_reference(
            backend=backend,
            model=model,
            database=database,
            config=config,
            point=point,
        )

    per_replica_concurrency = int(math.ceil(point.batch_size / point.dp))
    replicas = [
        _make_replica(
            backend=backend,
            model=model,
            database=database,
            config=config,
            isl=point.isl,
            osl=point.osl,
            concurrency=per_replica_concurrency,
        )
        for _ in range(point.dp)
    ]
    clock_ms = 0.0
    total_iters = 0
    steady_iters = 0
    steady_time_ms = 0.0
    steady_output_tokens = 0
    sum_decode_reqs = 0
    peak_decode_reqs = 0
    max_iters = config.num_requests * (
        point.osl + point.isl // max(config.max_num_batched_tokens, 1) + 10
    ) * point.dp

    while (
        any(len(rep.completed) < config.num_requests for rep in replicas)
        and total_iters < max_iters
    ):
        schedules = [rep.scheduler.schedule(rep.waiting, rep.running) for rep in replicas]
        if all(schedule.is_empty for schedule in schedules):
            break
        breakdowns = [
            _compute_breakdown(rep.latency_calc, schedule=schedule, isl=point.isl)
            if not schedule.is_empty
            else None
            for rep, schedule in zip(replicas, schedules, strict=True)
        ]
        tokens = [schedule.total_tokens for schedule in schedules]
        padded_tokens = max(tokens)
        has_prefill = any(schedule.total_prefill_tokens > 0 for schedule in schedules)
        max_avg_kv = max((_avg_kv(schedule) for schedule in schedules), default=0)
        padded_non_attn = _padded_non_attention_ms(
            replicas[0].latency_calc,
            padded_tokens=padded_tokens,
            has_prefill=has_prefill,
            isl=point.isl,
            decode_avg_kv_len=max_avg_kv,
        )
        max_attention = max(
            (_attention_ms(breakdown) for breakdown in breakdowns if breakdown is not None),
            default=0.0,
        )
        step_wall_ms = padded_non_attn + max_attention
        in_steady = all(
            len(rep.completed) >= config.warmup_requests and rep.has_external_supply
            for rep in replicas
        )
        clock_ms += step_wall_ms
        total_iters += 1
        decode_reqs_global = sum(len(schedule.decode_reqs) for schedule in schedules)
        sum_decode_reqs += decode_reqs_global
        peak_decode_reqs = max(peak_decode_reqs, decode_reqs_global)
        if in_steady:
            steady_iters += 1
            steady_time_ms += step_wall_ms
        for rep, schedule in zip(replicas, schedules, strict=True):
            steady_output_tokens += _apply_schedule(
                rep,
                schedule=schedule,
                clock_ms=clock_ms,
                iter_lat=step_wall_ms,
                in_steady_state=in_steady,
            )

    if steady_iters <= 0 or steady_time_ms <= 0:
        output_gpu = 0.0
    else:
        output_global = steady_output_tokens / (steady_time_ms / 1000.0)
        output_gpu = output_global / (point.tp * point.dp)
    return JointSimResult(
        output_tok_s_gpu=output_gpu,
        avg_decode_reqs_global=sum_decode_reqs / max(total_iters, 1),
        peak_decode_reqs_global=peak_decode_reqs,
        steady_state_iterations=steady_iters,
        steady_state_time_ms=steady_time_ms,
    )


def _load_real_targets(
    *,
    phase403_csv: Path,
    phase400_csv: Path,
) -> dict[str, tuple[float, float]]:
    phase403 = _read_csv_by_scenario(phase403_csv)
    phase400 = _read_csv_by_scenario(phase400_csv)
    targets: dict[str, tuple[float, float]] = {}
    for scenario in DP2_SCENARIOS:
        row = phase403[scenario]
        targets[scenario] = (
            float(row["phase403_output_tok_s_gpu"]),
            float(row["real_running_global_mean"]),
        )
    tp8 = phase400[TP8_CONTROL]
    targets[TP8_CONTROL] = (
        float(tp8["output_tok_s_gpu"]),
        float(tp8["serve_running_reqs_mean"]),
    )
    return targets


def build_phase407_rows_from_results(
    *,
    simulation_results: dict[tuple[str, str], JointSimResult],
    phase401_csv: Path = PHASE401_CSV,
    phase403_csv: Path = PHASE403_CSV,
    phase400_csv: Path = PHASE400_CSV,
) -> list[dict[str, str]]:
    phase401 = _read_csv_by_scenario(phase401_csv)
    real_targets = _load_real_targets(
        phase403_csv=phase403_csv,
        phase400_csv=phase400_csv,
    )
    points = _point_by_name()
    rows = [
        _build_gate_row(
            point=points[scenario],
            num_gpu_blocks=_phase401_num_gpu_blocks(phase401, scenario),
            phase401_sim=float(phase401[scenario]["phase401_sim_output_tok_s_gpu"]),
            real_output=real_targets[scenario][0],
            real_running_mean=real_targets[scenario][1],
            uncoupled=simulation_results[(scenario, "uncoupled")],
            coupled=simulation_results[(scenario, "coupled")],
        )
        for scenario in ALL_SCENARIOS
    ]
    _validate_rows(rows)
    return rows


def _build_gate_row(
    *,
    point: validate_cb.MultiConfigPoint,
    num_gpu_blocks: int,
    phase401_sim: float,
    real_output: float,
    real_running_mean: float,
    uncoupled: JointSimResult,
    coupled: JointSimResult,
) -> dict[str, str]:
    row_type = "tp8_early_exit_control" if point.dp == 1 else "dp2_joint_sim"
    uncoupled_error = _error_pct(uncoupled.output_tok_s_gpu, phase401_sim)
    coupled_ratio = _safe_ratio(coupled.output_tok_s_gpu, real_output)
    coupled_error = _error_pct(coupled.output_tok_s_gpu, real_output)
    penalty = _safe_ratio(uncoupled.output_tok_s_gpu, coupled.output_tok_s_gpu)
    penalty_target = PENALTY_TARGETS.get(point.name, 1.0)
    penalty_error = _error_pct(penalty, penalty_target)
    occupancy_error = _error_pct(coupled.avg_decode_reqs_global, real_running_mean)
    tp8_error = _error_pct(coupled.output_tok_s_gpu, uncoupled.output_tok_s_gpu)

    uncoupled_gate = _gate(uncoupled_error, 2.0)
    coupled_gate = _gate(coupled_error, 10.0) if point.dp > 1 else "not_applicable"
    penalty_gate = _gate(penalty_error, 10.0) if point.dp > 1 else "not_applicable"
    occupancy_gate = _gate(occupancy_error, 20.0) if point.dp > 1 else "not_applicable"
    tp8_gate = _gate(tp8_error, 2.0) if point.dp == 1 else "not_applicable"

    if point.dp == 1:
        verdict = "tp8_dp1_joint_sim_no_change" if tp8_gate == "passed" else "tp8_joint_sim_changed_recheck"
        target = "no_runtime_change"
    elif {uncoupled_gate, coupled_gate, penalty_gate, occupancy_gate} == {"passed"}:
        verdict = "independent_joint_sim_matches_dp2_real"
        target = "runtime_dp_lockstep_joint_sim"
    else:
        verdict = "independent_joint_sim_incomplete"
        target = "recheck_padded_nonattention_or_attention_max_cost"

    row = {
        "source": SOURCE,
        "row_type": row_type,
        "scenario": point.name,
        "tp": point.tp,
        "dp": point.dp,
        "ep": point.moe_ep,
        "isl": point.isl,
        "osl": point.osl,
        "max_num_batched_tokens": point.max_num_batched_tokens,
        "num_gpu_blocks": num_gpu_blocks,
        "real_output_tok_s_gpu": real_output,
        "phase401_sim_output_tok_s_gpu": phase401_sim,
        "uncoupled_joint_output_tok_s_gpu": uncoupled.output_tok_s_gpu,
        "uncoupled_vs_phase401_error_pct": uncoupled_error,
        "uncoupled_harness_gate": uncoupled_gate,
        "coupled_joint_output_tok_s_gpu": coupled.output_tok_s_gpu,
        "coupled_ratio": coupled_ratio,
        "coupled_error_pct": coupled_error,
        "coupled_convergence_gate": coupled_gate,
        "joint_penalty": penalty,
        "penalty_target": penalty_target,
        "penalty_error_pct": penalty_error,
        "penalty_gate": penalty_gate,
        "joint_avg_decode_reqs_global": coupled.avg_decode_reqs_global,
        "joint_peak_decode_reqs_global": coupled.peak_decode_reqs_global,
        "real_running_global_mean": real_running_mean,
        "occupancy_error_pct": occupancy_error,
        "occupancy_gate": occupancy_gate,
        "tp8_early_exit_gate": tp8_gate,
        "steady_state_iterations": coupled.steady_state_iterations,
        "steady_state_time_ms": coupled.steady_state_time_ms,
        "phase405_penalty_read": False,
        "paper_model_rule": "two_replica_joint_loop_step_tokens_max",
        "mechanism_verdict": verdict,
        "phase408_target": target,
        "gpu_allowed": False,
        "ssh_allowed": False,
        "runtime_modified": False,
        "perf_database": False,
        "valid_for_default": False,
        "diagnostic_only": True,
        "default_readiness": DEFAULT_READINESS,
    }
    return {field: _fmt(row.get(field)) for field in CSV_FIELDS}


def run_phase407_simulations(
    *,
    phase401_csv: Path = PHASE401_CSV,
) -> dict[tuple[str, str], JointSimResult]:
    phase401 = _read_csv_by_scenario(phase401_csv)
    points = _point_by_name()
    loaded: dict[tuple[int, int, int, int], tuple] = {}
    results: dict[tuple[str, str], JointSimResult] = {}
    for scenario in ALL_SCENARIOS:
        point = points[scenario]
        key = (point.tp, point.dp, point.moe_tp, point.moe_ep)
        if key not in loaded:
            loaded[key] = validate_cb._load_model_and_db(
                tp=point.tp,
                dp=point.dp,
                moe_tp=point.moe_tp,
                moe_ep=point.moe_ep,
            )
        model, db, backend = loaded[key]
        cb_config = _make_config(
            point,
            num_gpu_blocks=_phase401_num_gpu_blocks(phase401, scenario),
        )
        uncoupled = run_uncoupled_reference(
            backend=backend,
            model=model,
            database=db,
            config=cb_config,
            point=point,
        )
        if point.dp == 1:
            coupled = uncoupled
        else:
            coupled = run_coupled_joint_sim(
                backend=backend,
                model=model,
                database=db,
                config=cb_config,
                point=point,
            )
        results[(scenario, "uncoupled")] = uncoupled
        results[(scenario, "coupled")] = coupled
    return results


def build_phase407_rows(
    *,
    phase401_csv: Path = PHASE401_CSV,
    phase403_csv: Path = PHASE403_CSV,
    phase400_csv: Path = PHASE400_CSV,
) -> list[dict[str, str]]:
    return build_phase407_rows_from_results(
        simulation_results=run_phase407_simulations(phase401_csv=phase401_csv),
        phase401_csv=phase401_csv,
        phase403_csv=phase403_csv,
        phase400_csv=phase400_csv,
    )


def _validate_rows(rows: list[dict[str, str]]) -> None:
    for row in rows:
        if row.get("source") != SOURCE:
            raise ValueError("source")
        if row.get("phase405_penalty_read") != "false":
            raise ValueError("phase405_penalty_read")
        for field in ("gpu_allowed", "ssh_allowed", "runtime_modified", "perf_database", "valid_for_default"):
            if row.get(field) != "false":
                raise ValueError(field)
        if row.get("diagnostic_only") != "true":
            raise ValueError("diagnostic_only")
        if row.get("default_readiness") != DEFAULT_READINESS:
            raise ValueError("default_readiness")
    if len(rows) != 3:
        raise ValueError("Phase407 must contain two DP2 rows plus one TP8 control")
    if {row["scenario"] for row in rows} != set(ALL_SCENARIOS):
        raise ValueError("Phase407 scenario set mismatch")


def write_phase407_csv(path: Path, rows: list[dict[str, str]] | None = None) -> None:
    rows = build_phase407_rows() if rows is None else rows
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_phase407_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    dp_rows = [row for row in rows if row["row_type"] == "dp2_joint_sim"]
    all_passed = all(
        row["uncoupled_harness_gate"] == "passed"
        and row["coupled_convergence_gate"] == "passed"
        and row["penalty_gate"] == "passed"
        and row["occupancy_gate"] == "passed"
        for row in dp_rows
    )
    summary = (
        "Phase407 supersedes Phase406's circular penalty check with a genuine joint loop. "
        "The independent two-replica simulation passes all DP2 gates."
        if all_passed
        else (
            "Phase407 supersedes Phase406's circular penalty check, and the genuine joint "
            "loop does not pass the DP2 gates. The uncoupled harness matches Phase401, but "
            "the symmetric max-token coupling produces about 1.0x penalty instead of the "
            "required 1.78-1.88x, so pad-to-max alone is not a sufficient runtime fix."
        )
    )
    lines = [
        "# Phase407 DP lockstep joint simulation",
        "",
        summary,
        "",
        "## Rule",
        "",
        "- phase405_penalty_read=false: the model does not read Phase405 duty, active-iter, or reconstructed penalty fields.",
        "- uncoupled mode uses the existing CBSimulator path as the harness gate.",
        "- coupled mode runs two replicas with independent schedulers and shared wall time; each step charges padded non-attention at `max(tokens_A, tokens_B)` plus `max(attn_A, attn_B)`.",
        "",
        "## Gates",
        "",
        "| scenario | row_type | uncoupled gate | coupled ratio | coupled gate | penalty | penalty gate | occupancy gate | verdict |",
        "|---|---|---|---:|---|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {row_type} | {uncoupled_harness_gate} | {coupled_ratio} | {coupled_convergence_gate} | {joint_penalty} | {penalty_gate} | {occupancy_gate} | {mechanism_verdict} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Harness fidelity passed: uncoupled joint output matches Phase401 for both DP2 scenarios and the TP8 control.",
            "- The coupled loop stays near the uncoupled DP2 output, so the model does not independently reproduce the real duty loss.",
            "- This means Phase408 should not directly implement this symmetric lockstep rule. The next target is replica asymmetry / prefill occupancy behavior that creates the real running mean gap.",
            "",
            "## Boundary",
            "",
            "- report-only; runtime_modified=false, perf_database=false.",
            "- gpu_allowed=false, ssh_allowed=false.",
            "- valid_for_default=false, diagnostic_only=true, default_readiness=No-Go.",
            "- Phase408 should only touch runtime if this genuine joint simulation is accepted.",
            "",
        ]
    )
    return "\n".join(lines)


def write_phase407_md(path: Path, rows: list[dict[str, str]] | None = None) -> None:
    rows = build_phase407_rows() if rows is None else rows
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase407_md(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase401-csv", type=Path, default=PHASE401_CSV)
    parser.add_argument("--phase403-csv", type=Path, default=PHASE403_CSV)
    parser.add_argument("--phase400-csv", type=Path, default=PHASE400_CSV)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_phase407_rows(
        phase401_csv=args.phase401_csv,
        phase403_csv=args.phase403_csv,
        phase400_csv=args.phase400_csv,
    )
    write_phase407_csv(args.csv, rows)
    write_phase407_md(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
