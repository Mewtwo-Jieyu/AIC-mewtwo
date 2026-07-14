#!/usr/bin/env python3
"""Validate Phase462 queue traces and locate the first semantic divergence."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class CaptureIntegrity:
    passed: bool
    mapped_requests: int
    schedule_sequences: int
    linked_preemptions: int
    tokenizer_footers: int
    engine_footers: int


@dataclass(frozen=True)
class DivergenceVerdict:
    kind: str
    stage: str
    schedule_seq: int


def load_trace_paths(paths: Iterable[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        with path.open(encoding="utf-8") as source:
            rows.extend(json.loads(line) for line in source if line.strip())
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("ts_ns", 0)),
            int(row.get("pid", 0)),
            int(row.get("event_seq", 0)),
        ),
    )


def _open_trace(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def validate_capture_paths(
    paths: Iterable[Path], *, expected_requests: int
) -> CaptureIntegrity:
    trace_to_ordinal: dict[str, int] = {}
    ordinal_to_trace: dict[int, str] = {}
    inputs: set[int] = set()
    outputs: dict[int, set[object]] = {}
    futures: set[int] = set()
    preemptions: list[tuple[int, object]] = []
    tokenizer_footers = 0
    engine_footers = 0

    for path in paths:
        event_count = 0
        footer_seen = False
        with _open_trace(path) as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("kind") == "trace_footer":
                    if footer_seen:
                        raise ValueError(f"duplicate_trace_footer:{path}")
                    footer_seen = True
                    if int(row.get("event_count", -1)) != event_count:
                        raise ValueError(f"trace_event_count_mismatch:{path}")
                    if row.get("flush_complete") is not True:
                        raise ValueError(f"trace_flush_incomplete:{path}")
                    tokenizer_footers += row.get("component") == "tokenizer"
                    engine_footers += row.get("component") == "engine"
                    continue
                if footer_seen:
                    raise ValueError(f"event_after_trace_footer:{path}")
                event_count += 1

                kind = row.get("kind")
                if kind == "arrival_map":
                    trace_ids = list(row.get("trace_ids", []))
                    ordinals = list(row.get("arrival_ordinals", []))
                    positions = list(
                        row.get("batch_positions", range(len(trace_ids)))
                    )
                    if not (len(trace_ids) == len(ordinals) == len(positions)):
                        raise ValueError("arrival_map_length_mismatch")
                    if positions != list(range(len(trace_ids))):
                        raise ValueError("arrival_map_batch_order_mismatch")
                    for trace_id, ordinal in zip(trace_ids, ordinals):
                        key = str(trace_id)
                        value = int(ordinal)
                        if key in trace_to_ordinal or value in ordinal_to_trace:
                            raise ValueError("arrival_map_duplicate")
                        trace_to_ordinal[key] = value
                        ordinal_to_trace[value] = key
                elif kind == "scheduler_input":
                    sequence = int(row["schedule_seq"])
                    if sequence in inputs:
                        raise ValueError("duplicate_scheduler_input")
                    inputs.add(sequence)
                elif kind == "scheduler_output":
                    sequence = int(row["schedule_seq"])
                    if sequence in outputs:
                        raise ValueError("duplicate_scheduler_output")
                    outputs[sequence] = set(row.get("preempted_request_ids", []))
                elif kind == "future_complete":
                    sequence = int(row["schedule_seq"])
                    if sequence in futures:
                        raise ValueError("duplicate_future_complete")
                    futures.add(sequence)
                elif kind == "preempt_decision":
                    preemptions.append(
                        (int(row["schedule_seq"]), row.get("victim_request_id"))
                    )
        if not footer_seen:
            raise ValueError(f"trace_footer_missing:{path}")

    if len(trace_to_ordinal) != expected_requests:
        raise ValueError(
            f"arrival_map_incomplete:{len(trace_to_ordinal)}:{expected_requests}"
        )
    if set(ordinal_to_trace) != set(range(expected_requests)):
        raise ValueError("arrival_ordinal_not_contiguous")
    sequences = sorted(inputs)
    if not sequences or sequences != list(range(1, sequences[-1] + 1)):
        raise ValueError("schedule_seq_hole")
    if set(outputs) != inputs:
        raise ValueError("scheduler_output_link_incomplete")
    if futures != inputs:
        raise ValueError("future_completion_link_incomplete")
    for sequence, victim in preemptions:
        if sequence not in inputs:
            raise ValueError("preemption_without_input_state")
        if victim not in outputs[sequence]:
            raise ValueError("preemption_without_output_link")
    if tokenizer_footers < 1 or engine_footers < 1:
        raise ValueError("trace_flush_incomplete")
    return CaptureIntegrity(
        passed=True,
        mapped_requests=len(trace_to_ordinal),
        schedule_sequences=len(sequences),
        linked_preemptions=len(preemptions),
        tokenizer_footers=tokenizer_footers,
        engine_footers=engine_footers,
    )


def validate_capture(
    rows: Sequence[Mapping[str, object]], *, expected_requests: int
) -> CaptureIntegrity:
    trace_to_ordinal: dict[str, int] = {}
    ordinal_to_trace: dict[int, str] = {}
    for row in rows:
        if row.get("kind") != "arrival_map":
            continue
        trace_ids = list(row.get("trace_ids", []))
        ordinals = list(row.get("arrival_ordinals", []))
        positions = list(row.get("batch_positions", range(len(trace_ids))))
        if not (len(trace_ids) == len(ordinals) == len(positions)):
            raise ValueError("arrival_map_length_mismatch")
        if positions != list(range(len(trace_ids))):
            raise ValueError("arrival_map_batch_order_mismatch")
        for trace_id, ordinal in zip(trace_ids, ordinals):
            key = str(trace_id)
            value = int(ordinal)
            if key in trace_to_ordinal or value in ordinal_to_trace:
                raise ValueError("arrival_map_duplicate")
            trace_to_ordinal[key] = value
            ordinal_to_trace[value] = key
    if len(trace_to_ordinal) != expected_requests:
        raise ValueError(
            f"arrival_map_incomplete:{len(trace_to_ordinal)}:{expected_requests}"
        )
    if set(ordinal_to_trace) != set(range(expected_requests)):
        raise ValueError("arrival_ordinal_not_contiguous")

    inputs = {
        int(row["schedule_seq"]): row
        for row in rows
        if row.get("kind") == "scheduler_input"
    }
    outputs = {
        int(row["schedule_seq"]): row
        for row in rows
        if row.get("kind") == "scheduler_output"
    }
    futures = {
        int(row["schedule_seq"]): row
        for row in rows
        if row.get("kind") == "future_complete"
    }
    sequences = sorted(inputs)
    if not sequences or sequences != list(range(1, sequences[-1] + 1)):
        raise ValueError("schedule_seq_hole")
    if set(outputs) != set(inputs):
        raise ValueError("scheduler_output_link_incomplete")
    if set(futures) != set(inputs):
        raise ValueError("future_completion_link_incomplete")

    linked_preemptions = 0
    for row in rows:
        if row.get("kind") != "preempt_decision":
            continue
        schedule_seq = int(row["schedule_seq"])
        if schedule_seq not in inputs:
            raise ValueError("preemption_without_input_state")
        preempted = set(outputs[schedule_seq].get("preempted_request_ids", []))
        if row.get("victim_request_id") not in preempted:
            raise ValueError("preemption_without_output_link")
        linked_preemptions += 1

    tokenizer_footers = sum(
        row.get("kind") == "trace_footer"
        and row.get("component") == "tokenizer"
        and row.get("flush_complete") is True
        for row in rows
    )
    engine_footers = sum(
        row.get("kind") == "trace_footer"
        and row.get("component") == "engine"
        and row.get("flush_complete") is True
        for row in rows
    )
    if tokenizer_footers < 1 or engine_footers < 1:
        raise ValueError("trace_flush_incomplete")
    return CaptureIntegrity(
        passed=True,
        mapped_requests=len(trace_to_ordinal),
        schedule_sequences=len(sequences),
        linked_preemptions=linked_preemptions,
        tokenizer_footers=tokenizer_footers,
        engine_footers=engine_footers,
    )


def _phase_changed(real: Mapping[str, object], sim: Mapping[str, object]) -> bool:
    return real.get("request_phase") != sim.get("request_phase")


def _capacity_changed(real: Mapping[str, object], sim: Mapping[str, object]) -> bool:
    return real.get("free_blocks") != sim.get("free_blocks")


def _queue_changed(real: Mapping[str, object], sim: Mapping[str, object]) -> bool:
    fields = (
        "visible_ordinals",
        "running_order",
        "waiting_order",
        "skipped_waiting_order",
    )
    return any(real.get(field, []) != sim.get(field, []) for field in fields)


def classify_step(
    real: Mapping[str, object], sim: Mapping[str, object]
) -> DivergenceVerdict:
    schedule_seq = int(real["schedule_seq"])
    queue_changed = _queue_changed(real, sim)
    phase_changed = _phase_changed(real, sim)
    if queue_changed and phase_changed:
        return DivergenceVerdict(
            "completion_boundary_coupled_divergence",
            "queue_order_and_phase",
            schedule_seq,
        )
    if queue_changed:
        return DivergenceVerdict(
            "queue_order_divergence",
            "visible_set_and_queue_order",
            schedule_seq,
        )
    if _capacity_changed(real, sim):
        return DivergenceVerdict(
            "kv_capacity_divergence", "block_capacity", schedule_seq
        )
    if phase_changed:
        return DivergenceVerdict(
            "request_phase_divergence", "per_request_phase", schedule_seq
        )
    output_fields = (
        "scheduled_new",
        "scheduled_resumed",
        "scheduled_running",
        "preempted",
    )
    if any(real.get(field, []) != sim.get(field, []) for field in output_fields):
        return DivergenceVerdict(
            "scheduler_output_divergence", "scheduler_output", schedule_seq
        )
    return DivergenceVerdict("match", "none", schedule_seq)


def first_divergence(
    real_steps: Sequence[Mapping[str, object]],
    sim_steps: Sequence[Mapping[str, object]],
) -> DivergenceVerdict:
    if len(real_steps) != len(sim_steps):
        raise ValueError("step_count_mismatch")
    for real, sim in zip(real_steps, sim_steps):
        if int(real["schedule_seq"]) != int(sim["schedule_seq"]):
            raise ValueError("schedule_seq_alignment_mismatch")
        verdict = classify_step(real, sim)
        if verdict.kind != "match":
            return verdict
    return DivergenceVerdict("match", "none", len(real_steps))


def classify_schedule_boundary(
    real: Mapping[str, object], sim: Mapping[str, object]
) -> DivergenceVerdict:
    real_input = dict(real["input"])
    sim_input = dict(sim["input"])
    schedule_seq = int(real["schedule_seq"])
    real_input["schedule_seq"] = schedule_seq
    sim_input["schedule_seq"] = schedule_seq
    input_verdict = classify_step(real_input, sim_input)
    if input_verdict.kind != "match":
        return input_verdict
    for field in (
        "scheduled_new",
        "scheduled_resumed",
        "scheduled_running",
        "preempted",
    ):
        if list(real.get(field, [])) != list(sim.get(field, [])):
            return DivergenceVerdict(
                "scheduler_output_divergence", "scheduler_output", schedule_seq
            )
    return DivergenceVerdict("match", "none", schedule_seq)


def schedule_boundary_diff_stages(
    real: Mapping[str, object], sim: Mapping[str, object]
) -> list[str]:
    real_input = dict(real["input"])
    sim_input = dict(sim["input"])
    stages: list[str] = []
    if _queue_changed(real_input, sim_input):
        stages.append("queue")
    if _capacity_changed(real_input, sim_input):
        stages.append("capacity")
    if _phase_changed(real_input, sim_input):
        stages.append("phase")
    if any(
        list(real.get(field, [])) != list(sim.get(field, []))
        for field in (
            "scheduled_new",
            "scheduled_resumed",
            "scheduled_running",
            "preempted",
        )
    ):
        stages.append("output")
    return stages


def build_arrival_ordinal_map(
    rows: Iterable[Mapping[str, object]],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        if row.get("kind") != "arrival_map":
            continue
        for trace_id, ordinal in zip(
            row.get("trace_ids", []), row.get("arrival_ordinals", [])
        ):
            key = str(trace_id)
            if key in result:
                raise ValueError("arrival_trace_id_duplicate")
            result[key] = int(ordinal)
    return result


def iter_trace_rows(paths: Iterable[Path]):
    for path in paths:
        with _open_trace(path) as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)


def derive_execution_latency_oracle(
    rows: Iterable[Mapping[str, object]],
) -> list[float]:
    launches: dict[int, int] = {}
    completions: dict[int, int] = {}
    for row in rows:
        kind = row.get("kind")
        if kind == "scheduler_input":
            launches[int(row["schedule_seq"])] = int(row["ts_ns"])
        elif kind == "future_complete":
            completions[int(row["schedule_seq"])] = int(row["ts_ns"])
    if launches.keys() != completions.keys():
        raise ValueError("execution_oracle_boundary_mismatch")
    sequences = sorted(launches)
    if sequences != list(range(1, len(sequences) + 1)):
        raise ValueError("execution_oracle_schedule_hole")

    result: list[float] = []
    previous_completion_ns: int | None = None
    for sequence in sequences:
        start_ns = launches[sequence]
        if previous_completion_ns is not None:
            start_ns = max(start_ns, previous_completion_ns)
        duration_ns = completions[sequence] - start_ns
        if duration_ns < 0:
            raise ValueError(f"negative_execution_oracle_latency:{sequence}")
        result.append(duration_ns / 1_000_000)
        previous_completion_ns = completions[sequence]
    return result


def _canonical_queue_state(
    row: Mapping[str, object], trace_to_ordinal: Mapping[str, int]
) -> tuple[dict[str, object], dict[str, int]]:
    def ordinal(trace_id: object) -> int:
        key = str(trace_id)
        if key not in trace_to_ordinal:
            raise ValueError(f"unknown_trace_id:{key}")
        return int(trace_to_ordinal[key])

    request_to_ordinal: dict[str, int] = {}
    request_phase: dict[str, dict[str, object]] = {}
    for phase in row.get("request_phase", []):
        value = ordinal(phase["trace_id"])
        request_to_ordinal[str(phase["request_id"])] = value
        request_phase[str(value)] = {
            "num_computed_tokens": int(phase["num_computed_tokens"]),
            "num_output_placeholders": int(phase["num_output_placeholders"]),
            "block_counts": [int(item) for item in phase.get("block_counts", [])],
        }

    running = [ordinal(item) for item in row.get("running_trace_order", [])]
    waiting = [ordinal(item) for item in row.get("waiting_trace_order", [])]
    skipped = [
        ordinal(item) for item in row.get("skipped_waiting_trace_order", [])
    ]
    return (
        {
            "visible_ordinals": [*running, *waiting, *skipped],
            "running_order": running,
            "waiting_order": waiting,
            "skipped_waiting_order": skipped,
            "request_phase": request_phase,
            "free_blocks": int(row["free_blocks"]),
        },
        request_to_ordinal,
    )


def canonical_real_step(
    input_row: Mapping[str, object],
    output_row: Mapping[str, object],
    *,
    trace_to_ordinal: Mapping[str, int],
) -> dict[str, object]:
    input_state, input_request_map = _canonical_queue_state(
        input_row, trace_to_ordinal
    )
    output_state, output_request_map = _canonical_queue_state(
        output_row, trace_to_ordinal
    )
    request_map = {**input_request_map, **output_request_map}

    def scheduled(field: str) -> list[int]:
        values = []
        for request_id in output_row.get(field, []):
            key = str(request_id)
            if key not in request_map:
                raise ValueError(f"unknown_engine_request_id:{key}")
            values.append(request_map[key])
        return values

    return {
        "schedule_seq": int(input_row["schedule_seq"]),
        "input": input_state,
        "output": output_state,
        "scheduled_new": scheduled("scheduled_new_request_ids"),
        "scheduled_resumed": scheduled("scheduled_resumed_request_ids"),
        "scheduled_running": scheduled("scheduled_running_request_ids"),
        "preempted": scheduled("preempted_request_ids"),
    }


def iter_real_schedule_steps(
    path: Path, *, trace_to_ordinal: Mapping[str, int]
):
    pending: dict[int, dict[str, object]] = {}
    with _open_trace(path) as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            kind = row.get("kind")
            if kind == "scheduler_input":
                sequence = int(row["schedule_seq"])
                if sequence in pending:
                    raise ValueError("duplicate_pending_scheduler_input")
                pending[sequence] = row
            elif kind == "scheduler_output":
                sequence = int(row["schedule_seq"])
                if sequence not in pending:
                    raise ValueError("scheduler_output_without_input")
                yield canonical_real_step(
                    pending.pop(sequence),
                    row,
                    trace_to_ordinal=trace_to_ordinal,
                )
    if pending:
        raise ValueError("scheduler_input_without_output")


def judge_streaming_steps(
    real_steps: Iterable[Mapping[str, object]],
    *,
    run_sim: Callable[[Callable[[Mapping[str, object]], None]], Mapping[str, object]],
) -> dict[str, object]:
    real_iterator = iter(real_steps)
    first_mismatch: dict[str, object] | None = None
    first_by_stage: dict[str, dict[str, object]] = {}
    aligned_steps = 0
    sim_steps = 0

    def observe(sim_step: Mapping[str, object]) -> None:
        nonlocal aligned_steps, first_mismatch, sim_steps
        sim_steps += 1
        try:
            real_step = next(real_iterator)
        except StopIteration as error:
            raise ValueError("sim_schedule_without_real_step") from error
        real_seq = int(real_step["schedule_seq"])
        sim_seq = int(sim_step["schedule_seq"])
        if real_seq != sim_seq:
            raise ValueError(
                f"schedule_seq_alignment_mismatch:{real_seq}:{sim_seq}"
            )
        aligned_steps += 1
        verdict = classify_schedule_boundary(real_step, sim_step)
        for stage in schedule_boundary_diff_stages(real_step, sim_step):
            first_by_stage.setdefault(
                stage,
                {
                    "schedule_seq": real_seq,
                    "real": dict(real_step),
                    "sim": dict(sim_step),
                },
            )
        if first_mismatch is None and verdict.kind != "match":
            first_mismatch = {
                **asdict(verdict),
                "real": dict(real_step),
                "sim": dict(sim_step),
            }

    sim_result = dict(run_sim(observe))
    remaining_real_steps = list(real_iterator)
    terminal_empty_steps = 0
    if remaining_real_steps:
        terminal = remaining_real_steps[0]
        terminal_empty = (
            len(remaining_real_steps) == 1
            and int(terminal["schedule_seq"]) == sim_steps + 1
            and not terminal["input"].get("visible_ordinals")
            and all(
                not terminal.get(field)
                for field in (
                    "scheduled_new",
                    "scheduled_resumed",
                    "scheduled_running",
                    "preempted",
                )
            )
        )
        if not terminal_empty:
            raise ValueError(
                f"real_schedule_without_sim_step:{len(remaining_real_steps)}"
            )
        terminal_empty_steps = 1
    return {
        "aligned_steps": aligned_steps,
        "real_steps": aligned_steps + terminal_empty_steps,
        "sim_steps": sim_steps,
        "terminal_real_empty_schedules": terminal_empty_steps,
        "first_divergence": first_mismatch,
        "first_divergence_by_stage": first_by_stage,
        "sim_preemption": dict(sim_result.get("preemption", {})),
        "sim_preemption_events": list(
            sim_result.get("preemption_events", [])
        ),
    }


def run_oracle_first_divergence_judge(
    *,
    engine_trace: Path,
    tokenizer_traces: Sequence[Path],
    request_limit: int,
) -> dict[str, object]:
    import scripts.analyze_phase462_engine_loop_state_machine as engine

    tokenizer_rows = list(iter_trace_rows(tokenizer_traces))
    trace_to_ordinal = build_arrival_ordinal_map(tokenizer_rows)
    if sorted(trace_to_ordinal.values()) != list(range(request_limit)):
        raise ValueError("arrival_ordinal_coverage_mismatch")

    arrivals = []
    for row in tokenizer_rows:
        if row.get("kind") != "tokenizer_batch_complete":
            continue
        trace_ids = list(row.get("trace_ids", []))
        ordinals = list(row.get("arrival_ordinals", []))
        if len(trace_ids) != len(ordinals):
            raise ValueError("tokenizer_completion_membership_mismatch")
        ready_ms = int(row["batch_complete_ns"]) / 1_000_000
        for trace_id, ordinal in zip(trace_ids, ordinals):
            mapped = trace_to_ordinal.get(str(trace_id))
            if mapped != int(ordinal):
                raise ValueError("tokenizer_completion_ordinal_mismatch")
            if mapped < request_limit:
                arrivals.append(engine.TimedInput(ready_ms, mapped))
    arrivals = engine.normalize_timed_inputs(arrivals)
    if len(arrivals) != request_limit:
        raise ValueError("tokenizer_completion_coverage_mismatch")
    measured_iteration_latencies_ms = derive_execution_latency_oracle(
        iter_trace_rows([engine_trace])
    )

    def run_sim(observer):
        return engine._run_cb_queue_prototype(
            arrivals=arrivals,
            measured_iteration_latencies_ms=measured_iteration_latencies_ms,
            queue_depth=2,
            scenario="K2.5-tp8ep8-32k3k",
            request_limit=request_limit,
            prototype_osl=1200,
            schedule_observer=observer,
        )

    result = judge_streaming_steps(
        iter_real_schedule_steps(
            engine_trace, trace_to_ordinal=trace_to_ordinal
        ),
        run_sim=run_sim,
    )
    result["arrival_batches"] = sum(
        row.get("kind") == "tokenizer_batch_complete" for row in tokenizer_rows
    )
    result["request_limit"] = request_limit
    result["execution_latency_oracle"] = True
    result["diagnostic_only"] = True
    result["valid_for_default"] = False
    result["perf_database"] = False
    result["default_aic"] = "No-Go"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_paths", nargs="+", type=Path)
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--integrity-output", type=Path, required=True)
    parser.add_argument("--oracle-judge", action="store_true")
    parser.add_argument("--engine-trace", type=Path)
    parser.add_argument("--tokenizer-trace", action="append", type=Path, default=[])
    parser.add_argument("--judge-output", type=Path)
    args = parser.parse_args()
    verdict = validate_capture_paths(
        args.trace_paths, expected_requests=args.expected_requests
    )
    args.integrity_output.parent.mkdir(parents=True, exist_ok=True)
    args.integrity_output.write_text(
        json.dumps(asdict(verdict), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.oracle_judge:
        if args.engine_trace is None or not args.tokenizer_trace or args.judge_output is None:
            parser.error(
                "--oracle-judge requires --engine-trace, --tokenizer-trace, "
                "and --judge-output"
            )
        judge = run_oracle_first_divergence_judge(
            engine_trace=args.engine_trace,
            tokenizer_traces=args.tokenizer_trace,
            request_limit=args.expected_requests,
        )
        args.judge_output.parent.mkdir(parents=True, exist_ok=True)
        args.judge_output.write_text(
            json.dumps(judge, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(asdict(verdict), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
