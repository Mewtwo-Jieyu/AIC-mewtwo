#!/usr/bin/env python3
"""Phase462 Step2a-3d EngineCore batch-queue report-only prototype."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from aiconfigurator.sdk.backends.cb_simulator.engine_loop import (  # noqa: E402
    EngineLoopBatch as PrototypeBatch,
    EngineLoopResult as QueueMachineResult,
    TimedInput,
    TimedInputSource,
    run_engine_loop,
)

ARRIVAL_JSONL = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase462_arrival_observation/32k3k.jsonl.gz"
)
PREEMPT_SERVE_LOG = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase462_preemption_observation/overhead_on/"
    "K2.5-tp8ep8-32k3k/serve.log.gz"
)
PREEMPT_JSONL = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase462_preemption_observation/"
    "preemption_observation.jsonl"
)
DP_DIAGNOSTIC_JSONL = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase461_cost_recollect/"
    "dp2_bt65536_diagnostic/overhead_on/event_timing.jsonl.gz"
)
DEFAULT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase462_engine_loop_state_machine.csv"
)
DEFAULT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase462_engine_loop_state_machine.md"
)
SCENARIO = "K2.5-tp8ep8-32k3k"
REQUEST_LIMIT = 128
PROTOTYPE_OSL = 1200
CSV_FIELDS = ["section", "mode", "metric", "value", "target", "status", "note"]

REPORT_BOUNDARIES = {
    "diagnostic_only": True,
    "valid_for_default": False,
    "perf_database": False,
    "default_aic": "No-Go",
}

SOURCE_HASHES = {
    "vllm/v1/engine/core.py": (
        "896730e749cbcabb487ce50c703974594d197fc31a1d3b26fe096197d142d2d5"
    ),
    "vllm/v1/executor/multiproc_executor.py": (
        "2011e7d3024f1f230db98b88acc63856ca4d1e74b77f676d8a8a66c4e0dc01ad"
    ),
}

ITERATION_RE = re.compile(
    r"Iteration\((?P<iteration>\d+)\): "
    r"(?P<context_requests>\d+) context requests, "
    r"(?P<context_tokens>\d+) context tokens, "
    r"(?P<generation_requests>\d+) generation requests, "
    r"(?P<generation_tokens>\d+) generation tokens, "
    r"iteration elapsed time: (?P<elapsed_ms>[0-9.]+) ms"
)


@dataclass(frozen=True)
class DeploymentQueueSpec:
    queue_depth: int
    step_fn: str
    source_rule: str


def derive_deployment_queue_spec(
    *,
    executor: str,
    pipeline_parallel_size: int,
    async_scheduling: bool,
) -> DeploymentQueueSpec:
    """Apply the deployed vLLM 0.19 executor rule to runtime values."""
    if executor != "multiproc":
        raise ValueError(f"unsupported deployed executor: {executor}")
    source_rule = "2 if pp_size <= 1 and async_scheduling else pp_size"
    queue_depth = (
        2
        if pipeline_parallel_size <= 1 and async_scheduling
        else pipeline_parallel_size
    )
    return DeploymentQueueSpec(
        queue_depth=queue_depth,
        step_fn="step_with_batch_queue" if queue_depth > 1 else "step",
        source_rule=source_rule,
    )


def run_batch_queue_machine(
    *,
    queue_depth: int,
    arrivals: Iterable[TimedInput],
    on_drain: Callable[[float, list[object]], None],
    schedule: Callable[[float], PrototypeBatch | None],
    on_complete: Callable[[PrototypeBatch], None],
) -> QueueMachineResult:
    """Replay the source-defined EngineCore busy-loop and batch queue.

    Scheduling advances logical scheduler state inside ``schedule`` before the
    callback returns. Model execution is serialized, while up to ``queue_depth``
    futures may be outstanding.
    """
    return run_engine_loop(
        queue_depth=queue_depth,
        input_source=TimedInputSource(list(arrivals)),
        on_drain=on_drain,
        schedule=schedule,
        on_complete=on_complete,
    )


def prototype_gate(
    *,
    drain_step_match: float,
    first_schedule_step_match: float,
    first_16_match: float,
    preemptions: int,
    self_preemptions: int,
    repeat_victim_events: int,
    target_preemptions: int,
    prediction_eligible: bool,
) -> dict[str, object]:
    tolerance = max(2, round(target_preemptions * 0.2))
    signatures_passed = (
        drain_step_match == 1.0
        and first_schedule_step_match == 1.0
        and first_16_match == 1.0
        and abs(preemptions - target_preemptions) <= tolerance
        and self_preemptions == 0
        and repeat_victim_events == 0
    )
    return {
        "passed": signatures_passed and prediction_eligible,
        "signatures_passed": signatures_passed,
        "prediction_eligible": prediction_eligible,
        "runtime_change_allowed": False,
        "preemption_tolerance": tolerance,
    }


def _trace_index(trace_id: str) -> int:
    return int(trace_id.rsplit("-", 1)[-1])


def build_predicted_tokenizer_inputs(
    rows: Iterable[dict[str, object]],
    *,
    predict_service_ms: Callable[[int, int], float],
    request_limit: int,
) -> list[TimedInput]:
    items = list(rows)
    completes = {
        str(row["batch_id"]): row
        for row in items
        if row.get("kind") == "tokenizer_batch_complete"
    }
    arrivals: list[TimedInput] = []
    for enter in items:
        if enter.get("kind") != "tokenizer_batch_enter":
            continue
        complete = completes.get(str(enter["batch_id"]))
        if complete is None:
            continue
        trace_ids = [str(value) for value in enter.get("trace_ids", [])]
        prompt_lengths = [
            int(value) for value in complete.get("prompt_token_lengths", [])
        ]
        if len(trace_ids) != len(prompt_lengths):
            raise ValueError(f"batch membership mismatch: {enter['batch_id']}")
        service_ms = predict_service_ms(sum(prompt_lengths), len(trace_ids))
        ready_ms = int(enter["batch_start_ns"]) / 1_000_000 + service_ms
        for trace_id in trace_ids:
            request_id = _trace_index(trace_id)
            if request_id < request_limit:
                arrivals.append(TimedInput(ready_ms, request_id))
    return sorted(arrivals, key=lambda item: (item.arrival_ms, int(item.payload)))


def normalize_timed_inputs(inputs: Iterable[TimedInput]) -> list[TimedInput]:
    ordered = sorted(inputs, key=lambda item: (item.arrival_ms, int(item.payload)))
    if not ordered:
        return []
    origin = ordered[0].arrival_ms
    return [replace(item, arrival_ms=item.arrival_ms - origin) for item in ordered]


def exact_match_fraction(expected: dict[int, int], actual: dict[int, int]) -> float:
    if not expected:
        return 0.0
    return sum(actual.get(key) == value for key, value in expected.items()) / len(expected)


def summarize_dp_queue_evidence(
    rows: Iterable[dict[str, object]],
) -> dict[str, object]:
    """Check queue events across the whole DP diagnostic stream."""
    ranks: set[int] = set()
    receive_ranks: set[int] = set()
    schedule_ranks: set[int] = set()
    for row in rows:
        rank_value = row.get("dp_rank")
        if rank_value is None:
            continue
        rank = int(rank_value)
        ranks.add(rank)
        if "ts_ns" not in row:
            continue
        if row.get("kind") == "engine_receive":
            receive_ranks.add(rank)
        elif row.get("kind") == "scheduler_step":
            schedule_ranks.add(rank)
    timestamped_queue_ranks = receive_ranks & schedule_ranks
    return {
        "ranks": sorted(ranks),
        "engine_receive_ranks": sorted(receive_ranks),
        "scheduler_step_ranks": sorted(schedule_ranks),
        "timestamped_queue_ranks": sorted(timestamped_queue_ranks),
        "has_per_rank_queue_timestamps": (
            len(ranks) >= 2 and timestamped_queue_ranks == ranks
        ),
    }


def dp_merge_verdict(
    *, single_rank_gate_passed: bool, has_per_rank_queue_timestamps: bool
) -> dict[str, object]:
    if not single_rank_gate_passed:
        return {
            "status": "blocked",
            "reason": "single_rank_state_machine_not_validated",
            "step3_inheritance_allowed": False,
        }
    if not has_per_rank_queue_timestamps:
        return {
            "status": "blocked",
            "reason": "per_rank_queue_timestamps_missing",
            "step3_inheritance_allowed": False,
        }
    return {
        "status": "pass",
        "reason": "per_rank_state_machine_observed",
        "step3_inheritance_allowed": True,
    }


def can_schedule_async_decode(
    *, sampled_output_tokens: int, output_placeholders: int, osl: int
) -> bool:
    return sampled_output_tokens + output_placeholders < osl


def recompute_tokens_after_preemption(
    *, isl: int, sampled_output_tokens: int
) -> int:
    return isl + sampled_output_tokens


def async_decode_kv_tokens(
    *,
    isl: int,
    computed_output_tokens: int,
    scheduled_tokens: int = 0,
) -> int:
    """Count KV tokens after this step without double-counting placeholders."""
    return isl + computed_output_tokens + scheduled_tokens


def waiting_admission_allowed(*, preemptions_this_step: int) -> bool:
    return preemptions_this_step == 0


def map_real_drain_steps(
    rows: Iterable[dict[str, object]], *, request_limit: int
) -> dict[int, int]:
    items = list(rows)
    schedule_rows = sorted(
        (
            (int(row["ts_ns"]), int(row["step"]))
            for row in items
            if row.get("kind") == "scheduler_step"
        ),
        key=lambda item: item[0],
    )
    result: dict[int, int] = {}
    for row in items:
        if row.get("kind") != "engine_receive":
            continue
        request_id = _trace_index(str(row["trace_id"]))
        if request_id >= request_limit:
            continue
        receive_ns = int(row["ts_ns"])
        next_step = next(
            (step for ts_ns, step in schedule_rows if ts_ns >= receive_ns),
            None,
        )
        if next_step is not None:
            result[request_id] = next_step
    return result


def summarize_preemption_signature(
    events: Iterable[dict[str, object]],
) -> dict[str, int]:
    rows = list(events)
    victims = Counter(row["victim_req_id"] for row in rows)
    return {
        "preemptions": len(rows),
        "unique_victims": len(victims),
        "repeat_victim_events": sum(count - 1 for count in victims.values()),
        "self_preemptions": sum(
            row.get("trigger_req_id") == row.get("victim_req_id") for row in rows
        ),
    }


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with _open_text(path) as source:
        return [json.loads(line) for line in source if line.strip()]


def parse_iterations(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with _open_text(path) as source:
        for line in source:
            match = ITERATION_RE.search(line)
            if match is None:
                continue
            rows.append(
                {
                    "iteration": int(match.group("iteration")),
                    "context_requests": int(match.group("context_requests")),
                    "context_tokens": int(match.group("context_tokens")),
                    "generation_requests": int(match.group("generation_requests")),
                    "generation_tokens": int(match.group("generation_tokens")),
                    "elapsed_ms": float(match.group("elapsed_ms")),
                }
            )
    if not rows:
        raise ValueError(f"no iteration rows in {path}")
    return rows


def parse_runtime_deployment(lines: Iterable[str]) -> dict[str, object]:
    items = list(lines)
    async_scheduling = any(
        "Asynchronous scheduling is enabled" in line for line in items
    )
    pp_matches = [
        int(match.group(1))
        for line in items
        if (match := re.search(r"pipeline_parallel_size=(\d+)", line))
    ]
    executor = (
        "multiproc"
        if any("multiproc_executor.py" in line for line in items)
        else "unknown"
    )
    if not async_scheduling or not pp_matches or executor == "unknown":
        raise ValueError("runtime deployment evidence incomplete")
    return {
        "async_scheduling": async_scheduling,
        "pipeline_parallel_size": pp_matches[0],
        "executor": executor,
    }


def build_actual_tokenizer_inputs(
    rows: Iterable[dict[str, object]], *, request_limit: int
) -> list[TimedInput]:
    arrivals: list[TimedInput] = []
    for row in rows:
        if row.get("kind") != "tokenizer_batch_complete":
            continue
        complete_ms = int(row["batch_complete_ns"]) / 1_000_000
        for trace_id in row.get("trace_ids", []):
            request_id = _trace_index(str(trace_id))
            if request_id < request_limit:
                arrivals.append(TimedInput(complete_ms, request_id))
    return normalize_timed_inputs(arrivals)


def build_real_drain_boundary_inputs(
    rows: Iterable[dict[str, object]], *, request_limit: int
) -> list[TimedInput]:
    items = list(rows)
    step_times = {
        int(row["step"]): int(row["ts_ns"]) / 1_000_000
        for row in items
        if row.get("kind") == "scheduler_step"
    }
    drain_steps = map_real_drain_steps(items, request_limit=request_limit)
    return normalize_timed_inputs(
        TimedInput(step_times[step], request_id)
        for request_id, step in drain_steps.items()
    )


def map_real_first_schedule_steps(
    rows: Iterable[dict[str, object]], *, request_limit: int
) -> dict[int, int]:
    result: dict[int, int] = {}
    for row in rows:
        if row.get("kind") != "scheduler_step":
            continue
        step = int(row["step"])
        for trace_id in row.get("new_context_trace_ids", []):
            request_id = _trace_index(str(trace_id))
            if request_id < request_limit:
                result.setdefault(request_id, step)
    return result


def derive_future_latencies_ms(
    rows: Iterable[dict[str, object]], *, queue_depth: int
) -> list[float]:
    """Infer queue-visible completion-boundary times from schedule edges."""
    if queue_depth < 1:
        raise ValueError("queue_depth must be positive")
    schedule_times = [
        int(row["ts_ns"]) / 1_000_000
        for row in sorted(
            (item for item in rows if item.get("kind") == "scheduler_step"),
            key=lambda item: int(item["step"]),
        )
    ]
    if len(schedule_times) <= queue_depth:
        return []
    latencies: list[float] = []
    for index in range(len(schedule_times) - queue_depth):
        launch_ms = schedule_times[index]
        completion_ms = schedule_times[index + queue_depth]
        previous_completion_ms = (
            schedule_times[index + queue_depth - 1]
            if index > 0
            else launch_ms
        )
        latency_ms = completion_ms - max(launch_ms, previous_completion_ms)
        if latency_ms <= 0:
            raise ValueError(
                f"non-positive inferred future latency at batch {index + 1}: "
                f"{latency_ms}"
            )
        latencies.append(latency_ms)
    return latencies


def _shape(row: dict[str, object], *, real: bool) -> tuple[int, int, int]:
    if real:
        return (
            int(row["context_requests"]),
            int(row["context_tokens"]),
            int(row["generation_requests"]),
        )
    return (
        int(row["prefill_reqs"]),
        int(row["prefill_tokens"]),
        int(row["decode_reqs"]),
    )


def first_n_shape_match(
    expected: list[dict[str, object]],
    actual: list[dict[str, object]],
    *,
    count: int,
) -> float:
    if len(expected) < count or len(actual) < count:
        return 0.0
    matches = sum(
        _shape(real_row, real=True) == _shape(sim_row, real=False)
        for real_row, sim_row in zip(expected[:count], actual[:count])
    )
    return matches / count


def _run_cb_queue_prototype(
    *,
    arrivals: list[TimedInput],
    measured_iteration_latencies_ms: list[float] | None,
    queue_depth: int,
    scenario: str = SCENARIO,
    request_limit: int = REQUEST_LIMIT,
    prototype_osl: int = PROTOTYPE_OSL,
) -> dict[str, object]:
    """Run the source-defined queue with current cb scheduler state updates."""
    import scripts.validate_cb_simulator as validate
    from aiconfigurator.sdk.backends.cb_simulator import CBSimulator
    from aiconfigurator.sdk.backends.cb_simulator.datatypes import Request, RequestState
    from aiconfigurator.sdk.backends.cb_simulator.scheduler import CBScheduler

    point = next(item for item in validate.MULTI_CONFIG_DATA if item.name == scenario)
    model, database, backend = validate._load_model_and_db(
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
    config = replace(config, num_requests=request_limit, warmup_requests=0)
    sim = CBSimulator(backend, model, database, config)
    latency_calc = sim._create_latency_calc(0)
    preemption_events: list[dict[str, object]] = []
    output_placeholders: Counter[int] = Counter()
    sampled_output_tokens: Counter[int] = Counter()
    computed_output_tokens: Counter[int] = Counter()

    class InstrumentedScheduler(CBScheduler):
        active_trigger: int = -1
        preemptions_this_step: int = 0

        def schedule(self, waiting, running):
            self.preemptions_this_step = 0
            return super().schedule(waiting, running)

        def _next_waiting_candidate(self, waiting, admitted_ids, preempted_ids):
            if not waiting_admission_allowed(
                preemptions_this_step=self.preemptions_this_step
            ):
                return None
            return super()._next_waiting_candidate(
                waiting, admitted_ids, preempted_ids
            )

        def _blocks_needed(self, req, scheduled_tokens=0):
            if req.state == RequestState.DECODING:
                total_tokens = async_decode_kv_tokens(
                    isl=req.isl,
                    computed_output_tokens=computed_output_tokens[req.request_id],
                    scheduled_tokens=scheduled_tokens,
                )
            else:
                total_tokens = req.kv_cache_len + scheduled_tokens
            if total_tokens <= 0 or self._config.block_size <= 0:
                return 0
            return (
                total_tokens + self._config.block_size - 1
            ) // self._config.block_size

        def _ensure_block_capacity(
            self, current_req, waiting, running, result, preempted_ids
        ):
            self.active_trigger = current_req.request_id
            try:
                return super()._ensure_block_capacity(
                    current_req, waiting, running, result, preempted_ids
                )
            finally:
                self.active_trigger = -1

        def _preempt(self, victim, waiting, running, result, preempted_ids):
            self.preemptions_this_step += 1
            preemption_events.append(
                {
                    "local_step": schedule_calls + 1,
                    "trigger_req_id": self.active_trigger,
                    "victim_req_id": victim.request_id,
                    "trigger_state": next(
                        (
                            req.state.name
                            for req in [*running, *waiting]
                            if req.request_id == self.active_trigger
                        ),
                        "missing",
                    ),
                    "victim_preemptions_before": victim.num_preemptions,
                    "victim_sampled_output_tokens": sampled_output_tokens[
                        victim.request_id
                    ],
                    "victim_computed_output_tokens": computed_output_tokens[
                        victim.request_id
                    ],
                    "victim_output_placeholders": output_placeholders[
                        victim.request_id
                    ],
                    "recompute_tokens": recompute_tokens_after_preemption(
                        isl=victim.isl,
                        sampled_output_tokens=sampled_output_tokens[victim.request_id],
                    ),
                }
            )
            released = super()._preempt(
                victim, waiting, running, result, preempted_ids
            )
            victim.prefill_tokens_remaining = recompute_tokens_after_preemption(
                isl=victim.isl,
                sampled_output_tokens=sampled_output_tokens[victim.request_id],
            )
            computed_output_tokens[victim.request_id] = 0
            return released

    scheduler = InstrumentedScheduler(config)
    waiting: list[Request] = []
    running: list[Request] = []
    first_schedule_steps: dict[int, int] = {}
    first_prefill_completion_steps: dict[int, int] = {}
    request_completion_steps: dict[int, int] = {}
    drain_steps: dict[int, int] = {}
    trace: list[dict[str, object]] = []
    schedule_calls = 0
    completed: set[int] = set()

    def on_drain(_now_ms: float, request_ids: list[object]) -> None:
        next_step = schedule_calls + 1
        for value in request_ids:
            request_id = int(value)
            waiting.append(
                Request(
                    request_id=request_id,
                    isl=point.isl,
                    osl=prototype_osl,
                    arrival_time_ms=_now_ms,
                )
            )
            drain_steps[request_id] = next_step

    def schedule(now_ms: float) -> PrototypeBatch | None:
        nonlocal schedule_calls
        if len(completed) >= request_limit:
            return None
        guarded: list[Request] = []
        for req in running:
            if (
                req.state == RequestState.DECODING
                and not can_schedule_async_decode(
                    sampled_output_tokens=sampled_output_tokens[req.request_id],
                    output_placeholders=output_placeholders[req.request_id],
                    osl=req.osl,
                )
            ):
                req.state = RequestState.DONE
                guarded.append(req)
        result = scheduler.schedule(waiting, running)
        for req in guarded:
            if req.state == RequestState.DONE:
                req.state = RequestState.DECODING
        schedule_calls += 1
        if result.is_empty:
            return None

        avg_kv = (
            int(
                statistics.mean(
                    req.isl + computed_output_tokens[req.request_id]
                    for req in result.decode_reqs
                )
            )
            if result.decode_reqs
            else 0
        )
        trace_row = {
            "step": schedule_calls,
            "prefill_reqs": len(result.prefill_reqs),
            "prefill_tokens": result.total_prefill_tokens,
            "recompute_prefill_tokens": sum(
                result.prefill_tokens[req.request_id]
                for req in result.prefill_reqs
                if req.num_preemptions > 0
            ),
            "decode_reqs": len(result.decode_reqs),
            "decode_phase_counts": dict(
                Counter(
                    computed_output_tokens[req.request_id] % config.block_size
                    for req in result.decode_reqs
                )
            ),
            "preemptions": scheduler.preemptions_this_step,
        }
        for req in result.prefill_reqs:
            if req.num_preemptions == 0:
                first_schedule_steps.setdefault(req.request_id, schedule_calls)

        completed_prefill_ids: list[int] = []
        for req in result.prefill_reqs:
            tokens = result.prefill_tokens[req.request_id]
            if req.state in {RequestState.WAITING, RequestState.PREEMPTED}:
                req.state = RequestState.PREFILLING
                if req in waiting:
                    waiting.remove(req)
                if req not in running:
                    running.append(req)
            req.prefill_tokens_remaining -= tokens
            if req.prefill_tokens_remaining <= 0:
                req.prefill_tokens_remaining = 0
                req.state = RequestState.DECODING
                computed_output_tokens[req.request_id] = sampled_output_tokens[
                    req.request_id
                ]
                completed_prefill_ids.append(req.request_id)
                output_placeholders[req.request_id] += 1
        for req in result.decode_reqs:
            computed_output_tokens[req.request_id] += 1
            output_placeholders[req.request_id] += 1

        if measured_iteration_latencies_ms is None:
            latency_ms = latency_calc.compute(
                prefill_tokens=result.total_prefill_tokens,
                prefill_batch_size=len(result.prefill_reqs),
                prefill_seq_len=point.isl,
                decode_batch_size=len(result.decode_reqs),
                decode_avg_kv_len=avg_kv,
            )
        else:
            if schedule_calls > len(measured_iteration_latencies_ms):
                raise ValueError("measured iteration latency sequence exhausted")
            latency_ms = measured_iteration_latencies_ms[schedule_calls - 1]
        trace_row["latency_ms"] = latency_ms
        trace.append(trace_row)
        return PrototypeBatch(
            batch_id=schedule_calls,
            launch_ms=now_ms,
            latency_ms=latency_ms,
            payload={
                "sampled_output_ids": completed_prefill_ids
                + [req.request_id for req in result.decode_reqs],
            },
        )

    def on_complete(batch: PrototypeBatch) -> None:
        for request_id in batch.payload.get("sampled_output_ids", []):
            if request_id in first_schedule_steps:
                first_prefill_completion_steps.setdefault(request_id, batch.batch_id)
        for request_id in batch.payload.get("sampled_output_ids", []):
            req = next(
                (
                    item
                    for item in [*running, *waiting]
                    if item.request_id == request_id
                ),
                None,
            )
            if req is None:
                continue
            output_placeholders[request_id] -= 1
            sampled_output_tokens[request_id] += 1
            req.generated_tokens = sampled_output_tokens[request_id]
            if req.state in {
                RequestState.WAITING,
                RequestState.PREEMPTED,
                RequestState.PREFILLING,
            }:
                req.prefill_tokens_remaining += 1
            if sampled_output_tokens[request_id] >= req.osl:
                if req in running:
                    running.remove(req)
                if req in waiting:
                    waiting.remove(req)
                req.state = RequestState.DONE
                completed.add(request_id)
                request_completion_steps[request_id] = batch.batch_id

    machine = run_batch_queue_machine(
        queue_depth=queue_depth,
        arrivals=arrivals,
        on_drain=on_drain,
        schedule=schedule,
        on_complete=on_complete,
    )
    return {
        "drain_steps": drain_steps,
        "first_schedule_steps": first_schedule_steps,
        "first_prefill_completion_steps": first_prefill_completion_steps,
        "request_completion_steps": request_completion_steps,
        "trace": trace,
        "preemption": summarize_preemption_signature(preemption_events),
        "preemption_events": preemption_events,
        "request_count": request_limit,
        "prototype_osl": prototype_osl,
        "block_size": config.block_size,
        "launched_batches": len(machine.launched),
        "completed_requests": len(completed),
        "wall_ms": machine.final_clock_ms,
        "first_batch_latency_ms": (
            machine.launched[0].latency_ms if machine.launched else 0.0
        ),
    }


def _evaluate_mode(
    *,
    mode: str,
    arrivals: list[TimedInput],
    measured_iteration_latencies_ms: list[float] | None,
    real_drain_steps: dict[int, int],
    real_first_schedule_steps: dict[int, int],
    real_iterations: list[dict[str, object]],
    queue_depth: int,
    target_preemptions: int,
    prediction_eligible: bool,
) -> dict[str, object]:
    result = _run_cb_queue_prototype(
        arrivals=arrivals,
        measured_iteration_latencies_ms=measured_iteration_latencies_ms,
        queue_depth=queue_depth,
    )
    signature = dict(result["preemption"])
    drain_match = exact_match_fraction(real_drain_steps, result["drain_steps"])
    schedule_match = exact_match_fraction(
        real_first_schedule_steps, result["first_schedule_steps"]
    )
    first_16_match = first_n_shape_match(
        real_iterations, result["trace"], count=16
    )
    gate = prototype_gate(
        drain_step_match=drain_match,
        first_schedule_step_match=schedule_match,
        first_16_match=first_16_match,
        preemptions=int(signature["preemptions"]),
        self_preemptions=int(signature["self_preemptions"]),
        repeat_victim_events=int(signature["repeat_victim_events"]),
        target_preemptions=target_preemptions,
        prediction_eligible=prediction_eligible,
    )
    first_schedule_mismatches = [
        {
            "request_id": request_id,
            "real_step": expected_step,
            "sim_step": result["first_schedule_steps"].get(request_id),
        }
        for request_id, expected_step in real_first_schedule_steps.items()
        if result["first_schedule_steps"].get(request_id) != expected_step
    ]
    self_preemption_events = [
        event
        for event in result["preemption_events"]
        if event["trigger_req_id"] == event["victim_req_id"]
    ]
    return {
        "mode": mode,
        "drain_step_match": drain_match,
        "first_schedule_step_match": schedule_match,
        "first_16_match": first_16_match,
        **signature,
        **gate,
        "completed_requests": result["completed_requests"],
        "launched_batches": result["launched_batches"],
        "first_batch_latency_ms": result["first_batch_latency_ms"],
        "first_schedule_mismatches": first_schedule_mismatches,
        "self_preemption_events": self_preemption_events,
    }


def _row(
    section: str,
    mode: str,
    metric: str,
    value: object,
    *,
    target: object = "",
    status: str = "",
    note: str = "",
) -> dict[str, object]:
    return {
        "section": section,
        "mode": mode,
        "metric": metric,
        "value": value,
        "target": target,
        "status": status,
        "note": note,
    }


def build_report() -> tuple[list[dict[str, object]], dict[str, object]]:
    import scripts.analyze_phase462_arrival_admission_prototype as phase462c
    import scripts.analyze_phase462_preemption_observation as phase462b

    arrival_rows = load_jsonl(ARRIVAL_JSONL)
    with _open_text(PREEMPT_SERVE_LOG) as source:
        runtime = parse_runtime_deployment(source)
    deployment_spec = derive_deployment_queue_spec(
        executor=str(runtime["executor"]),
        pipeline_parallel_size=int(runtime["pipeline_parallel_size"]),
        async_scheduling=bool(runtime["async_scheduling"]),
    )
    if deployment_spec.queue_depth != 2:
        raise ValueError(
            f"unsupported observed queue depth: {deployment_spec.queue_depth}"
        )
    samples, _ = phase462c.load_primitive_samples(phase462c.ARRIVAL_PATHS)
    fit = phase462c.fit_measured_primitive(samples)
    predicted = normalize_timed_inputs(
        build_predicted_tokenizer_inputs(
            arrival_rows,
            predict_service_ms=fit.predict,
            request_limit=REQUEST_LIMIT,
        )
    )
    tokenizer_oracle = build_actual_tokenizer_inputs(
        arrival_rows, request_limit=REQUEST_LIMIT
    )
    drain_boundary_oracle = build_real_drain_boundary_inputs(
        arrival_rows, request_limit=REQUEST_LIMIT
    )
    real_drain_steps = map_real_drain_steps(
        arrival_rows, request_limit=REQUEST_LIMIT
    )
    real_first_schedule_steps = map_real_first_schedule_steps(
        arrival_rows, request_limit=REQUEST_LIMIT
    )
    real_iterations = parse_iterations(PREEMPT_SERVE_LOG)
    real_pairs = phase462b.pair_records(load_jsonl(PREEMPT_JSONL))
    if not real_pairs:
        raise ValueError("real preemption observation has no complete pairs")
    real_preemption = phase462b.summarize_real_decisions(real_pairs)
    target_preemptions = int(real_preemption["preemptions"])
    real_first_decision = {
        "trigger_request_id": real_pairs[0].trigger_request_id,
        "victim_request_id": real_pairs[0].victim_request_id,
        "trigger_num_computed_tokens": real_pairs[0].trigger_num_computed_tokens,
        "victim_num_computed_tokens": real_pairs[0].victim_num_computed_tokens,
    }
    future_latencies = derive_future_latencies_ms(
        arrival_rows, queue_depth=deployment_spec.queue_depth
    )

    modes = [
        _evaluate_mode(
            mode="target_run_fit_replay_plus_cb_cost",
            arrivals=predicted,
            measured_iteration_latencies_ms=None,
            real_drain_steps=real_drain_steps,
            real_first_schedule_steps=real_first_schedule_steps,
            real_iterations=real_iterations,
            queue_depth=deployment_spec.queue_depth,
            target_preemptions=target_preemptions,
            prediction_eligible=False,
        ),
        _evaluate_mode(
            mode="target_run_fit_replay_plus_future_oracle",
            arrivals=predicted,
            measured_iteration_latencies_ms=future_latencies,
            real_drain_steps=real_drain_steps,
            real_first_schedule_steps=real_first_schedule_steps,
            real_iterations=real_iterations,
            queue_depth=deployment_spec.queue_depth,
            target_preemptions=target_preemptions,
            prediction_eligible=False,
        ),
        _evaluate_mode(
            mode="target_run_tokenizer_oracle_plus_future_oracle",
            arrivals=tokenizer_oracle,
            measured_iteration_latencies_ms=future_latencies,
            real_drain_steps=real_drain_steps,
            real_first_schedule_steps=real_first_schedule_steps,
            real_iterations=real_iterations,
            queue_depth=deployment_spec.queue_depth,
            target_preemptions=target_preemptions,
            prediction_eligible=False,
        ),
        _evaluate_mode(
            mode="target_run_drain_oracle_plus_future_oracle",
            arrivals=drain_boundary_oracle,
            measured_iteration_latencies_ms=future_latencies,
            real_drain_steps=real_drain_steps,
            real_first_schedule_steps=real_first_schedule_steps,
            real_iterations=real_iterations,
            queue_depth=deployment_spec.queue_depth,
            target_preemptions=target_preemptions,
            prediction_eligible=False,
        ),
    ]
    primary = modes[0]

    with _open_text(DP_DIAGNOSTIC_JSONL) as source:
        dp_evidence = summarize_dp_queue_evidence(
            json.loads(line) for line in source if line.strip()
        )
    has_per_rank_queue_timestamps = bool(
        dp_evidence["has_per_rank_queue_timestamps"]
    )
    dp_verdict = dp_merge_verdict(
        single_rank_gate_passed=bool(primary["passed"]),
        has_per_rank_queue_timestamps=has_per_rank_queue_timestamps,
    )

    rows = [
        _row(
            "deployment",
            "real",
            "pipeline_parallel_size",
            runtime["pipeline_parallel_size"],
            target="runtime value",
            status="pass",
            note="serve.log engine config",
        ),
        _row(
            "deployment",
            "real",
            "async_scheduling",
            runtime["async_scheduling"],
            target="runtime value",
            status="pass",
            note="serve.log: Asynchronous scheduling is enabled",
        ),
        _row(
            "deployment",
            "real",
            "executor",
            runtime["executor"],
            target="runtime path",
            status="pass",
            note="multiproc_executor startup lines",
        ),
        _row(
            "deployment",
            "real",
            "batch_queue_depth",
            deployment_spec.queue_depth,
            target="source rule applied to runtime values",
            status="pass",
            note="multiproc_executor.py:469-472",
        ),
        _row(
            "deployment",
            "real",
            "engine_step",
            deployment_spec.step_fn,
            target="queue_depth > 1",
            status="pass",
            note="core.py:185-211",
        ),
        _row(
            "source_semantics",
            "real",
            "async_output_progress",
            "output_placeholders_until_future_completion",
            target="AsyncScheduler",
            status="pass",
            note="async_scheduler.py:15-59",
        ),
        _row(
            "source_semantics",
            "real",
            "waiting_after_preemption",
            "skip_entire_waiting_phase",
            target="no waiting admission after any preemption",
            status="pass",
            note="scheduler.py:563-564; cb_sim base path differs after first preemption",
        ),
        _row(
            "primitive",
            "tokenizer",
            "in_sample_weighted_mape",
            fit.weighted_mape,
            target="diagnostic only; <=0.10",
            status="pass" if fit.weighted_mape <= 0.10 else "fail",
            note=(
                "target-run samples; confirms replay fidelity only, not "
                "prediction eligibility"
            ),
        ),
    ]
    for metric in (
        "preemptions",
        "unique_victims",
        "repeat_victim_events",
        "self_preemptions",
    ):
        rows.append(
            _row(
                "ground_truth",
                "phase462_preemption_observation",
                metric,
                real_preemption[metric],
                status="pass",
                note=str(PREEMPT_JSONL.relative_to(REPO_ROOT)),
            )
        )
    targets = {
        "drain_step_match": "1.0",
        "first_schedule_step_match": "1.0",
        "first_16_match": "1.0",
        "preemptions": f"{target_preemptions}±{max(2, round(target_preemptions * 0.2))}",
        "self_preemptions": "0",
        "repeat_victim_events": "0",
    }
    for mode in modes:
        for metric, target in targets.items():
            value = mode[metric]
            if metric == "preemptions":
                passed = (
                    abs(int(value) - target_preemptions)
                    <= int(mode["preemption_tolerance"])
                )
            elif metric.endswith("match"):
                passed = float(value) == 1.0
            else:
                passed = int(value) == 0
            rows.append(
                _row(
                    "prototype",
                    str(mode["mode"]),
                    metric,
                    value,
                    target=target,
                    status="pass" if passed else "fail",
                )
            )
        rows.append(
            _row(
                "prototype",
                str(mode["mode"]),
                "signature_gate",
                bool(mode["signatures_passed"]),
                target="all four signatures",
                status="pass" if mode["signatures_passed"] else "fail",
                note="target-run replay; never sufficient for runtime promotion",
            )
        )
        rows.append(
            _row(
                "prototype",
                str(mode["mode"]),
                "gate",
                bool(mode["passed"]),
                target="all four signatures",
                status="pass" if mode["passed"] else "fail",
                note="requires both signatures and prediction-eligible input",
            )
        )
    engine_oracle = modes[-1]
    rows.extend(
        [
            _row(
                "remaining_divergence",
                "target_run_drain_oracle_plus_future_oracle",
                "first_schedule_mismatches",
                len(engine_oracle["first_schedule_mismatches"]),
                target=0,
                status=(
                    "pass" if not engine_oracle["first_schedule_mismatches"] else "fail"
                ),
                note=json.dumps(
                    engine_oracle["first_schedule_mismatches"][:3],
                    ensure_ascii=False,
                ),
            ),
            _row(
                "remaining_divergence",
                "target_run_drain_oracle_plus_future_oracle",
                "self_preemption_events",
                len(engine_oracle["self_preemption_events"]),
                target=0,
                status=(
                    "pass" if not engine_oracle["self_preemption_events"] else "fail"
                ),
                note=json.dumps(
                    engine_oracle["self_preemption_events"][:3],
                    ensure_ascii=False,
                ),
            ),
        ]
    )
    rows.extend(
        [
            _row(
                "dp_merge",
                "phase461_diagnostic",
                "has_per_rank_queue_timestamps",
                has_per_rank_queue_timestamps,
                target=True,
                status="pass" if has_per_rank_queue_timestamps else "fail",
                note=json.dumps(dp_evidence, ensure_ascii=False),
            ),
            _row(
                "dp_merge",
                "phase461_diagnostic",
                "verdict",
                dp_verdict["reason"],
                target="per_rank_state_machine_observed",
                status=str(dp_verdict["status"]),
            ),
        ]
    )
    for key, value in REPORT_BOUNDARIES.items():
        rows.append(
            _row(
                "boundary",
                "phase462",
                key,
                value,
                status="pass",
            )
        )
    summary = {
        "modes": modes,
        "runtime": runtime,
        "deployment_spec": deployment_spec,
        "primary_gate_passed": bool(primary["passed"]),
        "runtime_change_allowed": False,
        "dp_verdict": dp_verdict,
        "dp_evidence": dp_evidence,
        "real_preemption": real_preemption,
        "real_first_decision": real_first_decision,
        "target_preemptions": target_preemptions,
        "predictive_gate_available": False,
        "tokenizer_wmape": fit.weighted_mape,
        "cb_first_batch_latency_ms": modes[0]["first_batch_latency_ms"],
        "future_oracle_first_batch_latency_ms": modes[1][
            "first_batch_latency_ms"
        ],
        **REPORT_BOUNDARIES,
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, object]) -> None:
    modes = summary["modes"]
    engine_oracle = modes[-1]
    runtime = summary["runtime"]
    deployment_spec = summary["deployment_spec"]
    mismatch = (
        engine_oracle["first_schedule_mismatches"][0]
        if engine_oracle["first_schedule_mismatches"]
        else None
    )
    self_event = (
        engine_oracle["self_preemption_events"][0]
        if engine_oracle["self_preemption_events"]
        else None
    )
    self_detail = (
        "原型仍有自抢占：step {local_step} 的 trigger/victim 都是 request "
        "{victim_req_id}，sampled output={victim_sampled_output_tokens}、"
        "computed output={victim_computed_output_tokens}、在途 placeholder="
        "{victim_output_placeholders}。".format(**self_event)
        if self_event
        else "target-run drain oracle 已无自抢占。"
    )
    mode_rows = "\n".join(
        "| {mode} | {drain:.2%} | {schedule:.2%} | {first16:.2%} | "
        "{preemptions} / {self_preemptions} / {repeat_victim_events} | "
        "{signature_gate} | {gate} |".format(
            mode=item["mode"],
            drain=float(item["drain_step_match"]),
            schedule=float(item["first_schedule_step_match"]),
            first16=float(item["first_16_match"]),
            preemptions=item["preemptions"],
            self_preemptions=item["self_preemptions"],
            repeat_victim_events=item["repeat_victim_events"],
            signature_gate="PASS" if item["signatures_passed"] else "FAIL",
            gate="PASS" if item["passed"] else "FAIL",
        )
        for item in modes
    )
    conclusion = (
        "本轮只有 target-run replay，没有 prediction-eligible 输入；无论结构签名"
        "是否通过，都不能解锁 2c。runtime、PerfDB、validate 与 gate 均不改。"
    )
    path.write_text(
        f"""# Phase462 Step 2a-3d EngineCore 状态机原型

## 结论

{conclusion}

## 部署状态机

| 项 | 运行时值 | 源码规则 |
|---|---:|---|
| pipeline parallel | {runtime['pipeline_parallel_size']} | serve.log 实际配置 |
| async scheduling | {str(runtime['async_scheduling']).lower()} | serve.log 明确启用 |
| executor | {runtime['executor']} | 启动路径 |
| batch queue 深度 | {deployment_spec.queue_depth} | `multiproc_executor.py:469-472` |
| EngineCore 路径 | `{deployment_spec.step_fn}` | `core.py:185-211` |

`core.py:421-494` 定义非阻塞预调度：队列未满且最老 future 未完成时立即返回；队列满或没有可调度 token 时才等待最老 future。`core.py:1136-1175` 定义每次 busy-loop 先 drain 全部 input，再进入 engine step。调度后的请求状态立即推进，因此第二个 batch 可在第一个 batch 完成前采样到新状态。

部署态实际使用 `AsyncScheduler`：`async_scheduler.py:15-59` 在完整 prefill 与 decode 调度后都登记 output placeholder；future 返回后才增加已采样 output。原型因此分开记录 sampled output、已进入 KV 的 computed output 和在途 placeholder。另一个此前 16 步审计覆盖不到的规则是 `scheduler.py:563-564`：本步只要发生过抢占，就整步跳过 WAITING admission。

## 判卷口径

| 读数 | 数据 |
|---|---|
| drain、首调度、future 完成边界 | Phase462 N=512 到达观测短跑，取前 128 请求 |
| 前 16 步、抢占目标 `{summary['target_preemptions']}/{summary['real_preemption']['self_preemptions']}/{summary['real_preemption']['repeat_victim_events']}` | Phase462 N=128/C=128 抢占观测短跑，脚本从 JSONL 重算 |
| DP 汇合 | Phase461 dp2-bt65536 双 rank 诊断流 |

两个 TP8 短跑只共享部署配置和工作负载形状；arrival run 提供输入/队列时间轴，preemption run 提供动态签名，未把两者当成同一条逐步时间线。当前 tokenizer fit 也使用了目标 32k run 的 batch 起点、成员与样本，因此以下模式全部是 replay，不具备预测资格。

## 四签名判卷

| 模式 | drain 步 | 首调度步 | 前 16 步 | 抢占 / 自抢占 / 重复 victim | 签名门 | 2c 门 |
|---|---:|---:|---:|---:|---|---|
{mode_rows}

模式说明：

- `target_run_fit_replay_plus_cb_cost`：复用目标 run 的 tokenizer 批边界与成员，单步时长为当前 cb_sim 成本。
- `target_run_fit_replay_plus_future_oracle`：再用同 run 的 queue-depth-shifted scheduler 时间戳恢复 future 完成边界。
- `target_run_tokenizer_oracle_plus_future_oracle`：进一步使用实测 tokenizer 完成时刻。
- `target_run_drain_oracle_plus_future_oracle`：把请求直接放到实测 drain 边界，是结构定位上界。

四种模式的 `prediction_eligible=false`，所以签名门即使通过，2c 门也必须保持 FAIL。

## 剩余首次分歧

| 项 | real | 原型 | 判定 |
|---|---:|---:|---|
| request {mismatch['request_id'] if mismatch else '-'} 首调度步 | {mismatch['real_step'] if mismatch else '-'} | {mismatch['sim_step'] if mismatch else '-'} | {'未闭合' if mismatch else '已闭合'} |
| 自抢占 | {summary['real_preemption']['self_preemptions']} | {engine_oracle['self_preemptions']} | {'未闭合' if engine_oracle['self_preemptions'] else '已闭合'} |

{self_detail} 真实第一次抢占的 trigger/victim 是否相同、computed token 分别为多少，均从 observation 重算：`{summary['real_first_decision']['trigger_request_id'] == summary['real_first_decision']['victim_request_id']}`，{summary['real_first_decision']['trigger_num_computed_tokens']}/{summary['real_first_decision']['victim_num_computed_tokens']}。不能用硬编码消除剩余分歧。

同时，cb-cost replay 的 drain 只有 {float(modes[0]['drain_step_match']):.1%}；换成 future completion oracle 后仍须按表判卷。当前 cb_sim 首个 32k prefill 为 {summary['cb_first_batch_latency_ms']:.3f}ms，同 run future 边界为 {summary['future_oracle_first_batch_latency_ms']:.3f}ms，成本差会直接改变 busy-loop 的 input drain 采样点。这里的 future 边界是 queue-depth-shifted scheduler 时间戳给出的可见完成上界，不冒充精确 CUDA 执行时长。这是独立阻断，不能由状态机补丁掩盖。

## DP 汇合

结论：`{summary['dp_verdict']['reason']}`。Phase461 双 rank 数据有每 rank 构成与 busy time，但没有每 rank 的 EngineCore receive、预调度和 drain 时间戳，不能从 8.54x spread 反推出状态机自然生成了相位差。Step 3 暂不继承。

## 边界

| 字段 | 值 |
|---|---|
| diagnostic_only | true |
| valid_for_default | false |
| perf_database | false |
| runtime / validate / gate 改动 | 无 |
| Default AIC | No-Go |

源码快照：`core.py={SOURCE_HASHES['vllm/v1/engine/core.py']}`，`multiproc_executor.py={SOURCE_HASHES['vllm/v1/executor/multiproc_executor.py']}`。本报告只做离线原型判卷。
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    rows, summary = build_report()
    write_csv(args.csv, rows)
    write_markdown(args.md, summary)
    print(
        json.dumps(
            {
                "primary_gate_passed": summary["primary_gate_passed"],
                "runtime_change_allowed": summary["runtime_change_allowed"],
                "dp_verdict": summary["dp_verdict"],
                "csv": str(args.csv),
                "md": str(args.md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
