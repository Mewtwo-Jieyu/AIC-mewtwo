#!/usr/bin/env python3
"""Run Phase462 DP Step 3c nested route/receive/admit oracles offline."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import scripts.analyze_phase462_dp_route_observation as route_analysis  # noqa: E402
import scripts.analyze_phase462_dp_step3_null_only as step3  # noqa: E402
import scripts.validate_cb_simulator as validate  # noqa: E402
from aiconfigurator.sdk.backends.cb_simulator import CBSimulator  # noqa: E402
from aiconfigurator.sdk.backends.cb_simulator.datatypes import (  # noqa: E402
    CBSimResult,
    Request,
    RequestState,
)
from aiconfigurator.sdk.backends.cb_simulator.dp_admission import (  # noqa: E402
    DPAdmissionRouter,
    DPReplicaCounts,
)
from aiconfigurator.sdk.backends.cb_simulator.scheduler import CBScheduler  # noqa: E402
from aiconfigurator.sdk.backends.cb_simulator.simulator import _ReplicaState  # noqa: E402


DP2_BT = step3.DP2_BT
DEFAULT_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase462_dp_step3c"
DEFAULT_CSV = DEFAULT_ROOT / "phase462_dp_nested_oracle.csv"
DEFAULT_REPORT = DEFAULT_ROOT / "phase462_dp_nested_oracle.md"
TRACE_GLOB = str(
    REPO_ROOT
    / "docs/iter_gap_investigation/phase462_dp_route_observation/capture/trace/*.jsonl"
)


@dataclass(frozen=True)
class OracleInput:
    rank: int
    route_receive_ms: float
    admit_step: int


@dataclass(frozen=True)
class SpreadSummary:
    max_over_min: float
    cell: tuple[int, int]


@dataclass(frozen=True)
class OracleMeasurement:
    name: str
    error_ratio: float
    rank_exact_fraction: float
    same_cell_spread: float
    admit_match_fraction: float


@dataclass(frozen=True)
class AttributionRow:
    segment: str
    parent: str
    child: str
    delta_error: float


@dataclass(frozen=True)
class ReplayResult:
    metrics: CBSimResult
    trace: list[dict[str, object]]
    first_admit: dict[int, int]


def _by_ordinal(
    rows: Iterable[Mapping[str, object]], kind: str
) -> dict[int, Mapping[str, object]]:
    result: dict[int, Mapping[str, object]] = {}
    for row in rows:
        if row.get("kind") != kind:
            continue
        ordinal = int(row["arrival_ordinal"])
        if ordinal in result:
            raise AssertionError(f"duplicate_{kind}:{ordinal}")
        result[ordinal] = row
    return result


def build_oracle_inputs(
    rows: Iterable[Mapping[str, object]], *, expected_requests: int
) -> dict[int, OracleInput]:
    rows = list(rows)
    expected = set(range(expected_requests))
    route = _by_ordinal(rows, "route")
    receive = _by_ordinal(rows, "receive")
    admit = _by_ordinal(rows, "admit")
    for kind, observed in (("route", route), ("receive", receive), ("admit", admit)):
        if not expected.issubset(observed):
            missing = sorted(expected - set(observed))
            raise AssertionError(f"missing_{kind}:{missing[:5]}")

    result: dict[int, OracleInput] = {}
    for ordinal in range(expected_requests):
        rank = int(route[ordinal]["chosen_rank"])
        receive_rank = int(receive[ordinal]["dp_rank"])
        admit_rank = int(admit[ordinal]["dp_rank"])
        if rank != receive_rank or rank != admit_rank:
            raise AssertionError(f"rank_mismatch:{ordinal}:{rank}/{receive_rank}/{admit_rank}")
        route_ts = int(route[ordinal]["wall_ts_ns"])
        receive_ts = int(receive[ordinal]["wall_ts_ns"])
        admit_ts = int(admit[ordinal]["wall_ts_ns"])
        if not route_ts <= receive_ts <= admit_ts:
            raise AssertionError(f"time_order:{ordinal}")
        admit_step = int(admit[ordinal]["schedule_seq"])
        if admit_step <= 0:
            raise AssertionError(f"invalid_admit_step:{ordinal}:{admit_step}")
        result[ordinal] = OracleInput(
            rank=rank,
            route_receive_ms=(receive_ts - route_ts) / 1_000_000,
            admit_step=admit_step,
        )
    return result


def same_cell_cross_rank_spread(
    trace: Iterable[Mapping[str, object]],
) -> SpreadSummary:
    cells: dict[tuple[int, int], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in trace:
        if int(row["prefill_tokens"]) <= 0 or int(row["decode_reqs"]) <= 0:
            continue
        key = (int(row["total_tokens"]), int(row["decode_reqs"]))
        cells[key][int(row["replica_id"])].append(float(row["latency_ms"]))
    candidates: list[tuple[float, tuple[int, int]]] = []
    for cell, by_rank in cells.items():
        if set(by_rank) != {0, 1}:
            continue
        medians = [statistics.median(by_rank[rank]) for rank in (0, 1)]
        if min(medians) <= 0:
            continue
        candidates.append((max(medians) / min(medians), cell))
    if not candidates:
        return SpreadSummary(1.0, (0, 0))
    spread, cell = max(candidates)
    return SpreadSummary(spread, cell)


def attribution_rows(
    *, official_error: float, measurements: Sequence[OracleMeasurement]
) -> list[AttributionRow]:
    expected = ["O0", "O1", "O2", "O3"]
    if [item.name for item in measurements] != expected:
        raise AssertionError("oracle_measurements_must_be_O0_to_O3")
    errors = {item.name: item.error_ratio for item in measurements}
    return [
        AttributionRow(
            "official_to_multi_replica_architecture",
            "official",
            "O0",
            errors["O0"] - official_error,
        ),
        AttributionRow("route_rank_sequence", "O0", "O1", errors["O1"] - errors["O0"]),
        AttributionRow("route_to_receive_delay", "O1", "O2", errors["O2"] - errors["O1"]),
        AttributionRow("receive_to_admit_step", "O2", "O3", errors["O3"] - errors["O2"]),
    ]


def admit_lag_summary(
    observed: Mapping[int, int], oracle: Mapping[int, OracleInput]
) -> dict[str, int | float]:
    if set(observed) != set(oracle):
        raise AssertionError("admit_lag_request_set_mismatch")
    lags = [observed[ordinal] - item.admit_step for ordinal, item in oracle.items()]
    return {
        "exact": sum(lag == 0 for lag in lags),
        "early": sum(lag < 0 for lag in lags),
        "late": sum(lag > 0 for lag in lags),
        "median_lag_steps": float(statistics.median(lags)),
        "max_lag_steps": max(lags),
    }


def _release_ready(
    replica: _ReplicaState,
    pending: list[Request],
    *,
    mode: str,
    oracle: Mapping[int, OracleInput],
) -> None:
    next_step = replica.total_iters + 1
    ready = [
        request
        for request in pending
        if request.arrival_time_ms <= replica.clock_ms
        and (mode != "O3" or oracle[request.request_id].admit_step <= next_step)
    ]
    for request in sorted(ready, key=lambda item: (item.arrival_time_ms, item.request_id)):
        pending.remove(request)
        replica.waiting.append(request)


def _next_pending_time(
    replica: _ReplicaState,
    pending: Sequence[Request],
    *,
    mode: str,
    oracle: Mapping[int, OracleInput],
) -> float | None:
    next_step = replica.total_iters + 1
    eligible = [
        request.arrival_time_ms
        for request in pending
        if mode != "O3" or oracle[request.request_id].admit_step <= next_step
    ]
    return min(eligible) if eligible else None


def replay_nested_oracle(
    sim: CBSimulator,
    *,
    isl: int,
    osl: int,
    concurrency: int,
    data_parallel_size: int,
    num_gpus: int,
    mode: str,
    oracle: Mapping[int, OracleInput],
) -> ReplayResult:
    if mode not in {"O0", "O1", "O2", "O3"}:
        raise ValueError(f"unknown oracle mode:{mode}")
    if data_parallel_size != 2:
        raise ValueError("Step3c nested oracle is frozen to dp=2")
    config = sim._config  # noqa: SLF001
    if mode != "O0" and set(oracle) != set(range(config.num_requests)):
        raise AssertionError("oracle_request_set_mismatch")

    latency_calc = sim._create_latency_calc(0)  # noqa: SLF001
    replicas = [
        _ReplicaState(replica_id=rank, scheduler=CBScheduler(config))
        for rank in range(data_parallel_size)
    ]
    pending: dict[int, list[Request]] = {rank: [] for rank in range(data_parallel_size)}
    router = DPAdmissionRouter(data_parallel_size)
    completed: list[Request] = []
    trace: list[dict[str, object]] = []
    first_admit: dict[int, int] = {}
    next_id = 0

    def counts() -> list[DPReplicaCounts]:
        return [
            DPReplicaCounts(
                waiting=len(replica.waiting) + len(pending[replica.replica_id]),
                running=len(replica.running),
            )
            for replica in replicas
        ]

    def route_request(now_ms: float) -> None:
        nonlocal next_id
        if mode == "O0":
            rank = router.route(now_ms=now_ms, actual_counts=counts())
            receive_ms = now_ms
        else:
            item = oracle[next_id]
            rank = item.rank
            receive_ms = now_ms + (item.route_receive_ms if mode in {"O2", "O3"} else 0.0)
        request = Request(next_id, isl, osl, receive_ms)
        if receive_ms <= replicas[rank].clock_ms and mode != "O3":
            replicas[rank].waiting.append(request)
        else:
            pending[rank].append(request)
        next_id += 1

    for _ in range(min(concurrency, config.num_requests)):
        route_request(0.0)

    max_iters = config.num_requests * data_parallel_size * (
        osl + isl // config.max_num_batched_tokens + 10
    )
    total_iters = sum_prefill = sum_decode = sum_tokens = 0
    steady_iters = steady_output_tokens = 0
    steady_start_ms: float | None = None
    steady_end_ms = 0.0
    peak_prefill = peak_decode = peak_tokens = 0

    while len(completed) < config.num_requests and total_iters < max_iters:
        for replica in replicas:
            _release_ready(replica, pending[replica.replica_id], mode=mode, oracle=oracle)

        candidates: list[tuple[float, _ReplicaState]] = []
        for replica in replicas:
            if replica.waiting or replica.running:
                candidates.append((replica.clock_ms, replica))
                continue
            next_time = _next_pending_time(
                replica,
                pending[replica.replica_id],
                mode=mode,
                oracle=oracle,
            )
            if next_time is not None:
                candidates.append((max(replica.clock_ms, next_time), replica))
        if not candidates:
            blocked = sum(len(items) for items in pending.values())
            raise RuntimeError(f"oracle_replay_stalled:{mode}:pending={blocked}")

        candidate_time, replica = min(
            candidates, key=lambda item: (item[0], item[1].replica_id)
        )
        replica.clock_ms = candidate_time
        _release_ready(replica, pending[replica.replica_id], mode=mode, oracle=oracle)
        schedule = replica.scheduler.schedule(replica.waiting, replica.running)
        if schedule.is_empty:
            raise RuntimeError(f"empty_schedule:{mode}:rank={replica.replica_id}")

        local_step = replica.total_iters + 1
        for request in schedule.prefill_reqs:
            if request.state == RequestState.WAITING:
                first_admit.setdefault(request.request_id, local_step)

        avg_kv = (
            int(
                sum(request.kv_cache_len for request in schedule.decode_reqs)
                / len(schedule.decode_reqs)
            )
            if schedule.decode_reqs
            else 0
        )
        latency_ms = latency_calc.compute(
            prefill_tokens=schedule.total_prefill_tokens,
            prefill_batch_size=len(schedule.prefill_reqs),
            prefill_seq_len=isl,
            decode_batch_size=len(schedule.decode_reqs),
            decode_avg_kv_len=avg_kv,
        )
        start_ms = replica.clock_ms
        end_ms = start_ms + latency_ms
        has_external_supply = next_id < config.num_requests
        in_steady = len(completed) >= config.warmup_requests and has_external_supply

        replica.clock_ms = end_ms
        replica.total_iters += 1
        total_iters += 1
        sum_prefill += len(schedule.prefill_reqs)
        sum_decode += len(schedule.decode_reqs)
        sum_tokens += schedule.total_tokens
        peak_prefill = max(peak_prefill, len(schedule.prefill_reqs))
        peak_decode = max(peak_decode, len(schedule.decode_reqs))
        peak_tokens = max(peak_tokens, schedule.total_tokens)
        if in_steady:
            steady_iters += 1
            if steady_start_ms is None:
                steady_start_ms = start_ms
            steady_end_ms = max(steady_end_ms, end_ms)

        trace.append(
            {
                "replica_id": replica.replica_id,
                "local_iter": replica.total_iters,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "latency_ms": latency_ms,
                "prefill_reqs": len(schedule.prefill_reqs),
                "prefill_tokens": schedule.total_prefill_tokens,
                "decode_reqs": len(schedule.decode_reqs),
                "total_tokens": schedule.total_tokens,
                "is_mixed": bool(schedule.prefill_reqs and schedule.decode_reqs),
            }
        )

        for request in schedule.prefill_reqs:
            tokens = schedule.prefill_tokens[request.request_id]
            if request.state in {RequestState.WAITING, RequestState.PREEMPTED}:
                request.state = RequestState.PREFILLING
                if request.prefill_start_ms < 0:
                    request.prefill_start_ms = start_ms
                if request in replica.waiting:
                    replica.waiting.remove(request)
                replica.running.append(request)
            request.prefill_tokens_remaining -= tokens
            if request.prefill_tokens_remaining <= 0:
                request.prefill_tokens_remaining = 0
                if request.first_token_ms < 0:
                    request.first_token_ms = end_ms
                request.state = RequestState.DECODING

        newly_done: list[Request] = []
        for request in schedule.decode_reqs:
            request.generated_tokens += 1
            if in_steady:
                steady_output_tokens += 1
            if request.generated_tokens >= request.osl - 1:
                request.state = RequestState.DONE
                request.finish_ms = end_ms
                newly_done.append(request)
        for request in newly_done:
            replica.running.remove(request)
            completed.append(request)
            if next_id < config.num_requests:
                route_request(end_ms)

    if len(completed) != config.num_requests:
        raise RuntimeError(f"incomplete_replay:{mode}:{len(completed)}/{config.num_requests}")
    completed.sort(key=lambda request: request.finish_ms)
    steady_time_ms = (
        steady_end_ms - steady_start_ms
        if steady_start_ms is not None and steady_end_ms > steady_start_ms
        else 0.0
    )
    metrics = sim._collect_metrics(  # noqa: SLF001
        completed,
        num_gpus,
        total_iters,
        sum_prefill,
        sum_decode,
        sum_tokens,
        steady_iters,
        steady_time_ms,
        steady_output_tokens,
        peak_prefill,
        peak_decode,
        peak_tokens,
    )
    return ReplayResult(metrics, trace, first_admit)


def _measurement(
    name: str,
    replay: ReplayResult,
    *,
    real_rate: float,
    oracle: Mapping[int, OracleInput],
) -> OracleMeasurement:
    phase = step3.summarize_paired_rank_phase(replay.trace)
    spread = same_cell_cross_rank_spread(replay.trace)
    matches = sum(
        replay.first_admit.get(ordinal) == item.admit_step
        for ordinal, item in oracle.items()
    )
    return OracleMeasurement(
        name=name,
        error_ratio=validate._abs_error(replay.metrics.throughput_tok_s_gpu, real_rate),
        rank_exact_fraction=phase.active_exact_fraction,
        same_cell_spread=spread.max_over_min,
        admit_match_fraction=matches / len(oracle),
    )


def _load_oracle(expected_requests: int) -> dict[int, OracleInput]:
    paths = [Path(path) for path in sorted(glob.glob(TRACE_GLOB))]
    if not paths:
        raise FileNotFoundError(TRACE_GLOB)
    rows = route_analysis.load_trace_rows(paths)
    return build_oracle_inputs(rows, expected_requests=expected_requests)


def run_analysis(
    *, output_csv: Path = DEFAULT_CSV, output_report: Path = DEFAULT_REPORT
) -> dict[str, object]:
    point = next(item for item in validate.MULTI_CONFIG_DATA if item.name == DP2_BT)
    model, database, backend = validate._load_model_and_db(
        tp=point.tp, dp=point.dp, moe_tp=point.moe_tp, moe_ep=point.moe_ep
    )
    config = validate._make_multi_config_cb_config(
        point,
        overlap_factor=0.0,
        ep8_per_iteration_overhead_ms=validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
    )
    oracle = _load_oracle(config.num_requests)
    measurements: list[OracleMeasurement] = []
    replays: dict[str, ReplayResult] = {}
    for mode in ("O0", "O1", "O2", "O3"):
        sim = CBSimulator(backend, model, database, config)
        replay = replay_nested_oracle(
            sim,
            isl=point.isl,
            osl=point.osl,
            concurrency=point.batch_size,
            data_parallel_size=point.dp,
            num_gpus=point.tp * point.dp,
            mode=mode,
            oracle=oracle,
        )
        replays[mode] = replay
        measurements.append(
            _measurement(mode, replay, real_rate=point.real_output_tok_s_gpu, oracle=oracle)
        )

    parity_sim = CBSimulator(backend, model, database, config)
    parity = parity_sim.run_multi_replica(
        isl=point.isl,
        osl=point.osl,
        concurrency=point.batch_size,
        data_parallel_size=point.dp,
        num_gpus=point.tp * point.dp,
        lockstep=False,
    )
    parity_trace = [dict(row) for row in parity_sim.get_last_schedule_trace()]
    parity_keys = (
        "replica_id",
        "local_iter",
        "start_ms",
        "end_ms",
        "prefill_reqs",
        "prefill_tokens",
        "decode_reqs",
        "total_tokens",
        "is_mixed",
    )
    projected_o0 = [
        {key: row[key] for key in parity_keys} for row in replays["O0"].trace
    ]
    if not math.isclose(
        parity.throughput_tok_s_gpu,
        replays["O0"].metrics.throughput_tok_s_gpu,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or projected_o0 != parity_trace:
        raise AssertionError("O0_builtin_parity_failed")

    _, _, official_error = step3._load_official_score()  # noqa: SLF001
    attribution = attribution_rows(
        official_error=official_error, measurements=measurements
    )
    o3_lag = admit_lag_summary(replays["O3"].first_admit, oracle)
    real_phase = step3.summarize_paired_rank_phase(step3._load_real_phase_rows())  # noqa: SLF001
    real_spread, _, real_cell = step3._load_dp_cost_evidence()  # noqa: SLF001

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(
            [
                "kind",
                "name",
                "error_ratio",
                "rank_exact_fraction",
                "same_cell_spread",
                "admit_match_fraction",
                "parent",
                "child",
                "delta_error",
            ]
        )
        for item in measurements:
            writer.writerow(
                [
                    "oracle",
                    item.name,
                    item.error_ratio,
                    item.rank_exact_fraction,
                    item.same_cell_spread,
                    item.admit_match_fraction,
                    "",
                    "",
                    "",
                ]
            )
        for item in attribution:
            writer.writerow(
                [
                    "attribution",
                    item.segment,
                    "",
                    "",
                    "",
                    "",
                    item.parent,
                    item.child,
                    item.delta_error,
                ]
            )

    rows = "\n".join(
        f"| {item.name} | {item.error_ratio:.6f} | {item.rank_exact_fraction:.2%} | "
        f"{item.same_cell_spread:.3f}x | {item.admit_match_fraction:.2%} |"
        for item in measurements
    )
    deltas = "\n".join(
        f"| {item.segment} | {item.parent}→{item.child} | {item.delta_error:+.6f} |"
        for item in attribution
    )
    o3 = measurements[-1]
    decision = (
        "nested_oracle_closed_ready_for_mechanism_review"
        if o3.admit_match_fraction == 1.0
        else "admit_oracle_not_closed_no_mechanism_decision"
    )
    output_report.write_text(
        f"""# Phase462 DP Step 3c 嵌套 oracle 归因

结论：官方 `1.20` 路径没有 DP route 链，不能机械拆成 route/receive/admit。以下 O1-O3 只在双副本 O0 母体内逐层注入；官方→O0 单列为架构口径差。裁决为 `{decision}`，`Default AIC=No-Go`。

| 层 | error | rank 构成全等率 | 同 cell spread | real admit step 命中率 |
|---|---:|---:|---:|---:|
{rows}
| real | - | {real_phase.active_exact_fraction:.2%} | {real_spread:.3f}x (`{real_cell}`) | 100.00% |

| 归属段 | 对照 | error 增量 |
|---|---|---:|
{deltas}

`official_to_multi_replica_architecture` 不属于路由链，禁止计入 route 收益。O2 注入的是逐请求实测 `route→receive` 延迟；O3 仅按真实 per-rank `schedule_seq` 放行请求，没有强改 scheduler 输出。若 O3 admit 命中率不足 100%，本步不得宣称链外无残余。

| O3 admit 闭合审计 | 数值 |
|---|---:|
| exact | {o3_lag['exact']} |
| early | {o3_lag['early']} |
| late | {o3_lag['late']} |
| median lag steps | {o3_lag['median_lag_steps']} |
| max lag steps | {o3_lag['max_lag_steps']} |

当前 trace 只有首次 admit step，没有该步每请求 scheduled token/chunk 和完整 scheduler 入参。强行把剩余请求塞进目标步会同时改 token budget、容量和队列语义，不再是单段 oracle；因此 O3 不闭合时只能停线，不能现场补齐。

本步全离线，不改 runtime、PerfDB 或 gate；O0 与内置双副本路径逐字数值一致。
""",
        encoding="utf-8",
    )
    return {
        "status": "completed_report_only",
        "decision": decision,
        "o0_builtin_parity": True,
        "o3_admit_lag": o3_lag,
        "measurements": [item.__dict__ for item in measurements],
        "runtime_changed": False,
        "perfdb_changed": False,
        "gate_changed": False,
        "gpu_used": False,
        "default_aic": "No-Go",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(
        json.dumps(
            run_analysis(output_csv=args.output_csv, output_report=args.output_report),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
