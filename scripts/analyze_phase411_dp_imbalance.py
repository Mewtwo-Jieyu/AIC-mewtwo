#!/usr/bin/env python3
"""Phase411: attribute DP2 penalty to finite-workload tail and imbalance."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import analyze_phase409_iter_trace as phase409


SOURCE = "phase411_dp_imbalance"
DEFAULT_SCENARIO = "K2.5-tp4ep8dp2-32k3k"
DEFAULT_RAW_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase409_iter_trace"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase411_dp_imbalance.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase411_dp_imbalance.md"
PHASE407_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase407_dp_lockstep_joint_sim.csv"
DEFAULT_READINESS = "No-Go"

CSV_FIELDS = [
    "source",
    "scenario",
    "artifact_dir",
    "engine0_wall_ms",
    "engine1_wall_ms",
    "balanced_wall_ms",
    "tail_wall_ms",
    "tail_engine",
    "needed_penalty",
    "tail_penalty",
    "balanced_penalty",
    "product_penalty",
    "product_error_pct",
    "decomposition_gate",
    "tail_excess_share",
    "engine0_prompt_tokens",
    "engine1_prompt_tokens",
    "engine0_generation_tokens",
    "engine1_generation_tokens",
    "engine0_request_success_count",
    "engine1_request_success_count",
    "engine0_request_count_est",
    "engine1_request_count_est",
    "request_count_ratio",
    "engine0_avg_decode_len",
    "engine1_avg_decode_len",
    "decode_length_ratio",
    "engine0_decode_batch_mean_balanced",
    "engine1_decode_batch_mean_balanced",
    "decode_batch_ratio_balanced",
    "engine0_decode_batch_mean_tail",
    "tail_generation_step_count",
    "dominant_imbalance",
    "mechanism_verdict",
    "phase412_target",
    "phase405_penalty_read",
    "gpu_allowed",
    "ssh_allowed",
    "runtime_modified",
    "perf_database",
    "valid_for_default",
    "diagnostic_only",
    "default_readiness",
]

SUCCESS_RE = re.compile(
    r"^vllm:request_success_total\{(?P<labels>[^}]*)\}\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TimedStep:
    step: phase409.IterationStep
    start_ms: float
    end_ms: float


@dataclass(frozen=True)
class EngineMetrics:
    prompt_tokens: dict[str, float]
    generation_tokens: dict[str, float]
    request_success_count: dict[str, float]


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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _read_csv_by_scenario(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        return {row["scenario"]: row for row in csv.DictReader(f)}


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _labels(label_text: str) -> dict[str, str]:
    return {m.group("key"): m.group("value") for m in phase409.LABEL_RE.finditer(label_text)}


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
        raise ValueError("Phase411 requires both DP engines")
    return timelines


def parse_engine_metrics(metrics_path: Path) -> EngineMetrics:
    snapshots = phase409.parse_metric_snapshots(metrics_path)
    if not snapshots:
        raise ValueError(f"missing metric snapshots in {metrics_path}")
    final_snapshot = snapshots[-1]
    request_success_count = {"0": 0.0, "1": 0.0}
    last_record = json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[-1])
    for line in str(last_record.get("body", "")).splitlines():
        match = SUCCESS_RE.match(line.strip())
        if not match:
            continue
        labels = _labels(match.group("labels"))
        engine = labels.get("engine")
        if engine not in request_success_count:
            continue
        if labels.get("finished_reason") != "length":
            continue
        request_success_count[engine] += float(match.group("value"))
    return EngineMetrics(
        prompt_tokens=final_snapshot.prompt_tokens,
        generation_tokens=final_snapshot.generation_tokens,
        request_success_count=request_success_count,
    )


def _generation_request_values(timed_steps: list[TimedStep]) -> list[float]:
    return [
        float(timed_step.step.generation_requests)
        for timed_step in timed_steps
        if timed_step.step.generation_tokens > 0 or timed_step.step.generation_requests > 0
    ]


def _dominant_imbalance(
    *,
    request_count_ratio: float,
    decode_length_ratio: float,
    decode_batch_ratio_balanced: float,
) -> str:
    deltas = {
        "request_count": abs(request_count_ratio - 1.0),
        "decode_length": abs(decode_length_ratio - 1.0),
        "decode_batch": abs(decode_batch_ratio_balanced - 1.0),
    }
    return max(deltas, key=deltas.get)


def summarize_dp_imbalance(
    steps: list[phase409.IterationStep],
    metrics: EngineMetrics,
    *,
    needed_penalty: float,
    isl: float,
) -> dict[str, float | str]:
    timelines = build_wallclock_timelines(steps)
    engine_wall = {engine: timeline[-1].end_ms for engine, timeline in timelines.items()}
    tail_engine = "0" if engine_wall["0"] >= engine_wall["1"] else "1"
    balanced_wall_ms = min(engine_wall.values())
    tail_wall_ms = max(engine_wall.values()) - balanced_wall_ms
    tail_penalty = _safe_ratio(max(engine_wall.values()), balanced_wall_ms)
    balanced_penalty = _safe_ratio(needed_penalty, tail_penalty)
    product_penalty = tail_penalty * balanced_penalty
    product_error = _error_pct(product_penalty, needed_penalty)
    decomposition_gate = "passed" if product_error <= 10.0 else "failed"
    tail_excess_share = _safe_ratio(tail_penalty - 1.0, needed_penalty - 1.0)

    request_count_est = {
        engine: _safe_ratio(metrics.prompt_tokens[engine], isl)
        for engine in ("0", "1")
    }
    avg_decode_len = {
        engine: _safe_ratio(metrics.generation_tokens[engine], request_count_est[engine])
        for engine in ("0", "1")
    }
    request_count_ratio = _safe_ratio(
        max(request_count_est.values()),
        min(request_count_est.values()),
    )
    decode_length_ratio = _safe_ratio(
        max(avg_decode_len.values()),
        min(avg_decode_len.values()),
    )

    balanced_steps = {
        engine: [
            timed_step
            for timed_step in timeline
            if timed_step.start_ms < balanced_wall_ms
        ]
        for engine, timeline in timelines.items()
    }
    tail_steps = [
        timed_step
        for timed_step in timelines[tail_engine]
        if timed_step.end_ms > balanced_wall_ms
    ]
    balanced_batch_mean = {
        engine: _mean(_generation_request_values(timed_steps))
        for engine, timed_steps in balanced_steps.items()
    }
    tail_batch_mean = _mean(_generation_request_values(tail_steps))
    decode_batch_ratio_balanced = _safe_ratio(
        max(balanced_batch_mean.values()),
        min(balanced_batch_mean.values()),
    )
    dominant_imbalance = _dominant_imbalance(
        request_count_ratio=request_count_ratio,
        decode_length_ratio=decode_length_ratio,
        decode_batch_ratio_balanced=decode_batch_ratio_balanced,
    )

    if tail_excess_share >= 0.8:
        verdict = "closed_set_finite_workload_tail_dominates"
        phase412_target = "steady_state_recollect_or_larger_workload_before_runtime_model"
    elif balanced_penalty > 1.25:
        verdict = "steady_state_dp_underscaling_with_closed_set_tail"
        phase412_target = "separate_request_assignment_tail_from_balanced_dp_cost"
    else:
        verdict = "dp_imbalance_mostly_explained"
        phase412_target = "validate_imbalance_model_before_runtime_change"

    return {
        "engine0_wall_ms": engine_wall["0"],
        "engine1_wall_ms": engine_wall["1"],
        "balanced_wall_ms": balanced_wall_ms,
        "tail_wall_ms": tail_wall_ms,
        "tail_engine": tail_engine,
        "needed_penalty": needed_penalty,
        "tail_penalty": tail_penalty,
        "balanced_penalty": balanced_penalty,
        "product_penalty": product_penalty,
        "product_error_pct": product_error,
        "decomposition_gate": decomposition_gate,
        "tail_excess_share": tail_excess_share,
        "engine0_prompt_tokens": metrics.prompt_tokens["0"],
        "engine1_prompt_tokens": metrics.prompt_tokens["1"],
        "engine0_generation_tokens": metrics.generation_tokens["0"],
        "engine1_generation_tokens": metrics.generation_tokens["1"],
        "engine0_request_success_count": metrics.request_success_count["0"],
        "engine1_request_success_count": metrics.request_success_count["1"],
        "engine0_request_count_est": request_count_est["0"],
        "engine1_request_count_est": request_count_est["1"],
        "request_count_ratio": request_count_ratio,
        "engine0_avg_decode_len": avg_decode_len["0"],
        "engine1_avg_decode_len": avg_decode_len["1"],
        "decode_length_ratio": decode_length_ratio,
        "engine0_decode_batch_mean_balanced": balanced_batch_mean["0"],
        "engine1_decode_batch_mean_balanced": balanced_batch_mean["1"],
        "decode_batch_ratio_balanced": decode_batch_ratio_balanced,
        "engine0_decode_batch_mean_tail": tail_batch_mean if tail_engine == "0" else math.nan,
        "tail_generation_step_count": len(_generation_request_values(tail_steps)),
        "dominant_imbalance": dominant_imbalance,
        "mechanism_verdict": verdict,
        "phase412_target": phase412_target,
    }


def build_phase411_rows(
    *,
    raw_root: Path = DEFAULT_RAW_ROOT,
    phase407_csv: Path = PHASE407_CSV,
    scenario: str = DEFAULT_SCENARIO,
) -> list[dict[str, str]]:
    artifact_dir = raw_root / scenario
    steps = phase409.parse_iteration_steps(artifact_dir / "serve.log")
    metrics = parse_engine_metrics(artifact_dir / "metrics.jsonl")
    phase407 = _read_csv_by_scenario(phase407_csv)[scenario]
    needed_penalty = _safe_ratio(
        float(phase407["uncoupled_joint_output_tok_s_gpu"]),
        float(phase407["real_output_tok_s_gpu"]),
    )
    isl = float(phase407.get("isl") or 32000.0)
    summary = summarize_dp_imbalance(
        steps,
        metrics,
        needed_penalty=needed_penalty,
        isl=isl,
    )
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
        raise ValueError("Phase411 expects exactly one scenario row")
    for row in rows:
        if row.get("source") != SOURCE:
            raise ValueError("unexpected source")
        if row.get("phase405_penalty_read") != "false":
            raise ValueError("phase405_penalty_read must stay false")
        if row.get("gpu_allowed") != "false" or row.get("ssh_allowed") != "false":
            raise ValueError("Phase411 is offline and must not use GPU/SSH")
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


def write_phase411_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_phase411_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    row = rows[0]
    return "\n".join(
        [
            "# Phase411 DP Imbalance",
            "",
            "Phase411 复用 Phase409 trace 和 metrics，只做离线 DP 副本不对称归因；不改 runtime、PerfDatabase 或 gate。",
            "",
            "## Verdict",
            "",
            f"- verdict: `{row['mechanism_verdict']}`",
            f"- Phase412 target: `{row['phase412_target']}`",
            f"- decomposition: `{row['tail_penalty']} x {row['balanced_penalty']} = {row['product_penalty']}`",
            f"- needed penalty: `{row['needed_penalty']}`",
            f"- decomposition gate: `{row['decomposition_gate']}`",
            "",
            "## Summary",
            "",
            "| scenario | tail engine | tail wall ms | tail penalty | balanced penalty | tail excess share | request ratio | decode length ratio | balanced batch ratio | dominant imbalance |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            f"| {row['scenario']} | {row['tail_engine']} | {row['tail_wall_ms']} | {row['tail_penalty']} | {row['balanced_penalty']} | {row['tail_excess_share']} | {row['request_count_ratio']} | {row['decode_length_ratio']} | {row['decode_batch_ratio_balanced']} | {row['dominant_imbalance']} |",
            "",
            "## Interpretation",
            "",
            "- tail 存在且很大，但 tail_excess_share 没到 80%，不能把 1.887 主要归成闭集 tail 伪影。",
            "- prompt-token 反推请求分配约为 engine0/engine1 = request ratio；decode length ratio 接近 1，说明不是输出长度不均。",
            "- balanced 阶段 generation_requests 均值接近，batch 分布不均不是主导；剩余 balanced penalty 仍需要单独查 DP steady-state cost。",
            "- Phase412 应把 request-assignment tail 和 balanced DP cost 分开，不应直接给 simulator 加一个闭集 penalty。",
            "",
            "## Boundary",
            "",
            "- GPU/SSH: not used in Phase411.",
            "- Runtime/PerfDatabase/gate: not modified.",
            "- Phase405 penalty: not read.",
            "- Default AIC: No-Go.",
            "",
        ]
    )


def write_phase411_md(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase411_md(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--phase407-csv", type=Path, default=PHASE407_CSV)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_phase411_rows(
        raw_root=args.raw_root,
        phase407_csv=args.phase407_csv,
        scenario=args.scenario,
    )
    write_phase411_csv(args.csv, rows)
    write_phase411_md(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
