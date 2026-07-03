#!/usr/bin/env python3
"""Phase410: recompute DP lockstep penalty from elapsed iteration trace."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import analyze_phase409_iter_trace as phase409

SOURCE = "phase410_elapsed_penalty"
DEFAULT_SCENARIO = "K2.5-tp4ep8dp2-32k3k"
DEFAULT_RAW_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase409_iter_trace"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase410_elapsed_penalty.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase410_elapsed_penalty.md"
PHASE407_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase407_dp_lockstep_joint_sim.csv"
DEFAULT_READINESS = "No-Go"

CSV_FIELDS = [
    "source",
    "scenario",
    "artifact_dir",
    "engine0_steps",
    "engine1_steps",
    "engine0_wall_ms",
    "engine1_wall_ms",
    "real_wall_ms",
    "counterfactual_wall_ms",
    "elapsed_penalty",
    "needed_penalty",
    "penalty_error_pct",
    "penalty_gate",
    "intrinsic_decode_ms",
    "clean_decode_steps",
    "stalled_decode_steps",
    "stalled_decode_wall_ms",
    "dp_lockstep_extra_ms",
    "prefill_step_wall_ms",
    "tail_single_engine_wall_ms",
    "real_wall_source",
    "counterfactual_rule",
    "mechanism_verdict",
    "phase411_target",
    "phase405_penalty_read",
    "gpu_allowed",
    "ssh_allowed",
    "runtime_modified",
    "perf_database",
    "valid_for_default",
    "diagnostic_only",
    "default_readiness",
]


@dataclass(frozen=True)
class TimedStep:
    step: phase409.IterationStep
    start_ms: float
    end_ms: float

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


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


def _safe_ratio(a: float, b: float) -> float:
    if b <= 0:
        return math.inf
    return a / b


def _error_pct(predicted: float, target: float) -> float:
    return abs(_safe_ratio(predicted, target) - 1.0) * 100.0


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot take median of empty list")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _read_csv_by_scenario(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        return {row["scenario"]: row for row in csv.DictReader(f)}


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_wallclock_timelines(
    steps: list[phase409.IterationStep],
) -> dict[str, list[TimedStep]]:
    by_engine: dict[str, list[phase409.IterationStep]] = {"0": [], "1": []}
    for step in steps:
        if step.engine in by_engine:
            by_engine[step.engine].append(step)

    timelines: dict[str, list[TimedStep]] = {}
    for engine, engine_steps in by_engine.items():
        wall_ms = 0.0
        timeline: list[TimedStep] = []
        for step in sorted(engine_steps, key=lambda item: item.iteration):
            start_ms = wall_ms
            wall_ms += step.elapsed_ms
            timeline.append(TimedStep(step=step, start_ms=start_ms, end_ms=wall_ms))
        timelines[engine] = timeline
    if not timelines["0"] or not timelines["1"]:
        raise ValueError("Phase410 requires both DP engines")
    return timelines


def _overlap_ms(left: TimedStep, right: TimedStep) -> float:
    return max(0.0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))


def _has_peer_prefill_overlap(step: TimedStep, peer_prefill_steps: list[TimedStep]) -> bool:
    return any(_overlap_ms(step, peer_step) > 0 for peer_step in peer_prefill_steps)


def _clean_decode_durations(
    timelines: dict[str, list[TimedStep]],
    *,
    clean_decode_max_ms: float,
) -> list[float]:
    values: list[float] = []
    for engine_steps in timelines.values():
        for timed_step in engine_steps:
            step = timed_step.step
            if step.ctx_tokens == 0 and step.generation_tokens > 0 and 0 < step.elapsed_ms <= clean_decode_max_ms:
                values.append(step.elapsed_ms)
    return values


def summarize_elapsed_penalty(
    steps: list[phase409.IterationStep],
    *,
    needed_penalty: float,
    clean_decode_max_ms: float = 50.0,
    stall_multiplier: float = 4.0,
) -> dict[str, float | int | str]:
    timelines = build_wallclock_timelines(steps)
    clean_decode = _clean_decode_durations(timelines, clean_decode_max_ms=clean_decode_max_ms)
    intrinsic_decode_ms = _median(clean_decode)
    stall_threshold_ms = intrinsic_decode_ms * stall_multiplier

    real_wall_by_engine = {
        engine: timeline[-1].end_ms
        for engine, timeline in timelines.items()
    }
    counterfactual_wall_by_engine: dict[str, float] = {}
    stalled_decode_steps = 0
    stalled_decode_wall_ms = 0.0
    dp_lockstep_extra_ms = 0.0
    prefill_step_wall_ms = 0.0

    for engine, timeline in timelines.items():
        peer = "1" if engine == "0" else "0"
        peer_prefill_steps = [
            timed_step
            for timed_step in timelines[peer]
            if timed_step.step.ctx_tokens > 0 or timed_step.step.ctx_requests > 0
        ]
        counterfactual_wall = 0.0
        for timed_step in timeline:
            step = timed_step.step
            if step.ctx_tokens > 0 or step.ctx_requests > 0:
                prefill_step_wall_ms += step.elapsed_ms
            is_decode_only = step.ctx_tokens == 0 and step.generation_tokens > 0
            is_stalled = (
                is_decode_only
                and step.elapsed_ms > stall_threshold_ms
                and _has_peer_prefill_overlap(timed_step, peer_prefill_steps)
            )
            if is_stalled:
                stalled_decode_steps += 1
                stalled_decode_wall_ms += step.elapsed_ms
                dp_lockstep_extra_ms += step.elapsed_ms - intrinsic_decode_ms
                counterfactual_wall += intrinsic_decode_ms
            else:
                counterfactual_wall += step.elapsed_ms
        counterfactual_wall_by_engine[engine] = counterfactual_wall

    real_wall_ms = max(real_wall_by_engine.values())
    counterfactual_wall_ms = max(counterfactual_wall_by_engine.values())
    elapsed_penalty = _safe_ratio(real_wall_ms, counterfactual_wall_ms)
    penalty_error = _error_pct(elapsed_penalty, needed_penalty)
    penalty_gate = "passed" if penalty_error <= 10.0 else "failed"
    if penalty_gate == "passed":
        verdict = "elapsed_lockstep_reproduces_needed"
        phase411_target = "runtime_two_engine_elapsed_lockstep_model"
    elif elapsed_penalty < needed_penalty:
        verdict = "elapsed_lockstep_still_insufficient"
        phase411_target = "recheck_tail_imbalance_or_missing_dp_cost"
    else:
        verdict = "elapsed_lockstep_overpredicts_needed"
        phase411_target = "tighten_stall_detection_before_runtime_model"

    return {
        "engine0_steps": len(timelines["0"]),
        "engine1_steps": len(timelines["1"]),
        "engine0_wall_ms": real_wall_by_engine["0"],
        "engine1_wall_ms": real_wall_by_engine["1"],
        "real_wall_ms": real_wall_ms,
        "counterfactual_wall_ms": counterfactual_wall_ms,
        "elapsed_penalty": elapsed_penalty,
        "needed_penalty": needed_penalty,
        "penalty_error_pct": penalty_error,
        "penalty_gate": penalty_gate,
        "intrinsic_decode_ms": intrinsic_decode_ms,
        "clean_decode_steps": len(clean_decode),
        "stalled_decode_steps": stalled_decode_steps,
        "stalled_decode_wall_ms": stalled_decode_wall_ms,
        "dp_lockstep_extra_ms": dp_lockstep_extra_ms,
        "prefill_step_wall_ms": prefill_step_wall_ms,
        "tail_single_engine_wall_ms": abs(real_wall_by_engine["0"] - real_wall_by_engine["1"]),
        "real_wall_source": "serve_log_cumulative_elapsed_max_engine",
        "counterfactual_rule": "replace_peer_prefill_stalled_decode_with_intrinsic_decode",
        "mechanism_verdict": verdict,
        "phase411_target": phase411_target,
    }


def build_phase410_rows(
    *,
    raw_root: Path = DEFAULT_RAW_ROOT,
    phase407_csv: Path = PHASE407_CSV,
    scenario: str = DEFAULT_SCENARIO,
) -> list[dict[str, str]]:
    artifact_dir = raw_root / scenario
    steps = phase409.parse_iteration_steps(artifact_dir / "serve.log")
    phase407 = _read_csv_by_scenario(phase407_csv)[scenario]
    needed_penalty = _safe_ratio(
        float(phase407["uncoupled_joint_output_tok_s_gpu"]),
        float(phase407["real_output_tok_s_gpu"]),
    )
    summary = summarize_elapsed_penalty(steps, needed_penalty=needed_penalty)
    row = {
        "source": SOURCE,
        "scenario": scenario,
        "artifact_dir": _display_path(artifact_dir),
        **summary,
        "phase405_penalty_read": False,
        "gpu_allowed": False,
        "ssh_allowed": False,
        "runtime_modified": False,
        "perf_database": False,
        "valid_for_default": False,
        "diagnostic_only": True,
        "default_readiness": DEFAULT_READINESS,
    }
    return [{field: _fmt(row.get(field, "")) for field in CSV_FIELDS}]


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 1:
        raise ValueError("Phase410 expects exactly one scenario row")
    for row in rows:
        if row.get("source") != SOURCE:
            raise ValueError("unexpected source")
        if row.get("phase405_penalty_read") != "false":
            raise ValueError("phase405_penalty_read must stay false")
        if row.get("gpu_allowed") != "false" or row.get("ssh_allowed") != "false":
            raise ValueError("Phase410 is offline and must not use GPU/SSH")
        if row.get("runtime_modified") != "false":
            raise ValueError("runtime_modified must stay false")
        if row.get("perf_database") != "false":
            raise ValueError("perf_database must stay false")
        if row.get("valid_for_default") != "false":
            raise ValueError("valid_for_default must stay false")
        if row.get("diagnostic_only") != "true":
            raise ValueError("diagnostic_only must stay true")
        if row.get("default_readiness") != DEFAULT_READINESS:
            raise ValueError("default_readiness must stay No-Go")


def write_phase410_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_phase410_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    row = rows[0]
    return "\n".join(
        [
            "# Phase410 Elapsed Penalty",
            "",
            "Phase410 复用 Phase409 的同一份 GPU trace，只做离线 elapsed 归因；不改 runtime、PerfDatabase 或 gate。",
            "",
            "## Verdict",
            "",
            f"- verdict: `{row['mechanism_verdict']}`",
            f"- elapsed penalty: `{row['elapsed_penalty']}`",
            f"- needed penalty: `{row['needed_penalty']}`",
            f"- penalty gate: `{row['penalty_gate']}`",
            f"- Phase411 target: `{row['phase411_target']}`",
            "",
            "## Summary",
            "",
            "| scenario | real wall ms | counterfactual wall ms | intrinsic decode ms | stalled decode steps | dp extra ms | tail single-engine ms |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| {row['scenario']} | {row['real_wall_ms']} | {row['counterfactual_wall_ms']} | {row['intrinsic_decode_ms']} | {row['stalled_decode_steps']} | {row['dp_lockstep_extra_ms']} | {row['tail_single_engine_wall_ms']} |",
            "",
            "## Interpretation",
            "",
            "- elapsed 里确实能看到 peer-prefill-stalled decode：这些 decode-only step 的耗时远高于 intrinsic decode。",
            "- 但把这些 stalled decode 换回 intrinsic 后，elapsed penalty 仍远低于 needed penalty；这条证据不足以证明 DP lockstep elapsed alone 解释 1.887。",
            "- 当前最大结构信号是 engine wall 不对称和长 tail：后续应先查 tail imbalance 或其他缺失 DP cost，而不是直接落 runtime lockstep 模型。",
            "",
            "## Boundary",
            "",
            "- GPU/SSH: not used in Phase410.",
            "- Runtime/PerfDatabase/gate: not modified.",
            "- Phase405 penalty: not read.",
            "- Default AIC: No-Go.",
            "",
        ]
    )


def write_phase410_md(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase410_md(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--phase407-csv", type=Path, default=PHASE407_CSV)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_phase410_rows(
        raw_root=args.raw_root,
        phase407_csv=args.phase407_csv,
        scenario=args.scenario,
    )
    write_phase410_csv(args.csv, rows)
    write_phase410_md(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
