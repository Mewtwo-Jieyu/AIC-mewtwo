#!/usr/bin/env python3
"""Phase432: quantify execution bubble structure from Phase429 traces."""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import analyze_phase427_kernel_profile as phase427  # noqa: E402


SOURCE = "phase432_bubble_structure"
DEFAULT_SCENARIO = "K2.5-tp4ep8dp2-8k2k"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase429_reprofile"
PHASE429_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase429_reprofile.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase432_bubble_structure.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase432_bubble_structure.md"
DEFAULT_READINESS = "No-Go"
WINDOWS = ("w0_prefill", "w1_decode_c16", "w2_decode_c64", "w3_decode_c128")
DECODE_WINDOWS = ("w1_decode_c16", "w2_decode_c64", "w3_decode_c128")
TRUE = "true"
FALSE = "false"
LAYER_COUNT = 60
STRUCTURE_TOLERANCE = 0.20
PHASE431_DECODE_CONSTANT_TARGET_MS = 42.849884 - 30.001457
PHASE431_EP_A2A_EXCESS_SLOPE_TARGET_MS_PER_REQUEST = 0.070353 - 0.012944
PHASE414_32K_PREFILL_EXECUTION_GAP_TARGET_MS = 4718.954849 - 1818.637996
PHASE414_32K_PREFILL_GEN_REQS_MEAN = 6.451807
MIN_PROFILER_STEP_MS = 1.0

EXECUTE_CONTEXT_RE = re.compile(
    r"execute_context_(?P<ctx_tokens>\d+)\(\d+\)_generation_(?P<gen_tokens>\d+)\(\d+\)"
)

CSV_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "artifact_dir",
    "window",
    "step_type",
    "step_count",
    "prefill_step_count",
    "decode_step_count",
    "decode_batch_mean",
    "decode_batch_min",
    "decode_batch_max",
    "gap_ms_per_step_mean",
    "wall_bubble_ms_per_step_mean",
    "a2a_adjacent_gap_ms_per_step_mean",
    "gap_segments_per_step_mean",
    "a2a_gap_segments_per_step_mean",
    "ep_a2a_kernels_per_step_mean",
    "hypothesis",
    "parameter_ms",
    "prefill_target_ms",
    "prefill_predicted_ms",
    "prefill_error_ratio",
    "decode_constant_target_ms",
    "decode_constant_predicted_ms",
    "decode_constant_error_ratio",
    "ep_a2a_excess_slope_target_ms_per_request",
    "ep_a2a_excess_slope_predicted_ms_per_request",
    "ep_a2a_excess_error_ratio",
    "cross_32k3k_target_ms",
    "cross_32k3k_predicted_ms",
    "cross_32k3k_error_ratio",
    "consistency_gate",
    "selected_hypothesis",
    "phase432_verdict",
    "phase433_target",
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
class TraceEvent:
    name: str
    cat: str
    ts_us: float
    dur_us: float

    @property
    def end_us(self) -> float:
        return self.ts_us + self.dur_us

    @property
    def mid_us(self) -> float:
        return self.ts_us + self.dur_us / 2.0


@dataclass(frozen=True)
class ProfilerStep:
    window: str
    trace_file: str
    step_index: int
    ts_us: float
    end_us: float
    wall_ms: float
    ctx_tokens: int
    gen_tokens: int
    union_busy_ms: float
    internal_gap_ms: float
    internal_gap_count: int
    a2a_adjacent_gap_ms: float
    a2a_adjacent_gap_count: int
    ep_a2a_kernel_count: int

    @property
    def step_type(self) -> str:
        return "prefill_or_mixed" if self.ctx_tokens > 0 else "decode_only"

    @property
    def wall_bubble_ms(self) -> float:
        return max(0.0, self.wall_ms - self.union_busy_ms)


@dataclass(frozen=True)
class StepAggregate:
    window: str
    step_index: int
    step_type: str
    decode_batch: float
    wall_ms: float
    union_busy_ms: float
    internal_gap_ms: float
    wall_bubble_ms: float
    internal_gap_count: float
    a2a_adjacent_gap_ms: float
    a2a_adjacent_gap_count: float
    ep_a2a_kernel_count: float


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return TRUE if value else FALSE
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _safe_ratio(value: float, target: float) -> float:
    if target == 0:
        return 0.0 if value == 0 else math.inf
    return abs(value - target) / abs(target)


def _slope(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    x_mean = _mean([point[0] for point in points])
    y_mean = _mean([point[1] for point in points])
    denom = sum((x - x_mean) ** 2 for x, _y in points)
    if denom <= 0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in points) / denom


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _trace_paths(artifact_dir: Path, window: str) -> list[Path]:
    paths = sorted(
        path
        for path in (artifact_dir / f"prof_{window}").glob("*.pt.trace.json.gz")
        if path.name.startswith("dp")
    )
    if not paths:
        raise ValueError(f"missing trace files for {window} under {artifact_dir}")
    return paths


def _read_phase429_summary(path: Path, scenario: str) -> dict[str, str]:
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("row_type") == "summary" and row.get("scenario") == scenario:
                return row
    raise ValueError(f"missing Phase429 summary for {scenario}")


def _load_trace_events(path: Path) -> list[TraceEvent]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    events: list[TraceEvent] = []
    for raw in data.get("traceEvents", []):
        if raw.get("ph") != "X":
            continue
        try:
            ts_us = float(raw["ts"])
            dur_us = float(raw.get("dur", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if dur_us <= 0:
            continue
        events.append(
            TraceEvent(
                name=str(raw.get("name", "")),
                cat=str(raw.get("cat", "")),
                ts_us=ts_us,
                dur_us=dur_us,
            )
        )
    return events


def _category_for_event(event: TraceEvent) -> str:
    if event.cat in {"gpu_memcpy", "gpu_memset"}:
        return "memcpy_memset"
    return phase427.kernel_category(event.name)


def _merge_intervals(intervals: list[tuple[float, float, str]]) -> list[tuple[float, float, set[str]]]:
    merged: list[tuple[float, float, set[str]]] = []
    for start_us, end_us, category in sorted(intervals, key=lambda item: item[0]):
        if end_us <= start_us:
            continue
        if not merged or start_us > merged[-1][1]:
            merged.append((start_us, end_us, {category}))
            continue
        previous_start, previous_end, categories = merged[-1]
        categories.add(category)
        merged[-1] = (previous_start, max(previous_end, end_us), categories)
    return merged


def _infer_tokens(events: list[TraceEvent]) -> tuple[int, int]:
    best: tuple[float, int, int] | None = None
    for event in events:
        match = EXECUTE_CONTEXT_RE.match(event.name)
        if not match:
            continue
        candidate = (event.dur_us, int(match.group("ctx_tokens")), int(match.group("gen_tokens")))
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return 0, 0
    return best[1], best[2]


def parse_trace_steps(path: Path, window: str) -> list[ProfilerStep]:
    events = _load_trace_events(path)
    raw_steps = [
        event
        for event in events
        if event.name.startswith("ProfilerStep") and event.dur_us / 1000.0 >= MIN_PROFILER_STEP_MS
    ]
    raw_steps.sort(key=lambda event: event.ts_us)
    if not raw_steps:
        raise ValueError(f"missing profiler steps in {path}")

    step_starts = [step.ts_us for step in raw_steps]
    step_payloads: list[list[TraceEvent]] = [[] for _step in raw_steps]
    step_intervals: list[list[tuple[float, float, str]]] = [[] for _step in raw_steps]
    for event in events:
        if event.name.startswith("ProfilerStep"):
            continue
        idx = bisect.bisect_right(step_starts, event.mid_us) - 1
        if idx < 0:
            continue
        step = raw_steps[idx]
        if event.mid_us > step.end_us:
            continue
        step_payloads[idx].append(event)
        if event.cat not in {"kernel", "gpu_memcpy", "gpu_memset"}:
            continue
        category = _category_for_event(event)
        if category in {"profiler_step", "cuda_graph_envelope"}:
            continue
        step_intervals[idx].append(
            (max(event.ts_us, step.ts_us), min(event.end_us, step.end_us), category)
        )

    parsed_steps: list[ProfilerStep] = []
    for step_index, step in enumerate(raw_steps):
        ctx_tokens, gen_tokens = _infer_tokens(step_payloads[step_index])
        intervals = _merge_intervals(step_intervals[step_index])
        union_busy_us = sum(end_us - start_us for start_us, end_us, _cats in intervals)
        internal_gap_us = 0.0
        internal_gap_count = 0
        a2a_gap_us = 0.0
        a2a_gap_count = 0
        for left, right in zip(intervals, intervals[1:]):
            gap_us = max(0.0, right[0] - left[1])
            if gap_us <= 0:
                continue
            internal_gap_us += gap_us
            internal_gap_count += 1
            if "ep_a2a" in left[2] or "ep_a2a" in right[2]:
                a2a_gap_us += gap_us
                a2a_gap_count += 1
        ep_a2a_kernel_count = sum(1 for _start, _end, cats in intervals if "ep_a2a" in cats)
        parsed_steps.append(
            ProfilerStep(
                window=window,
                trace_file=path.name,
                step_index=step_index,
                ts_us=step.ts_us,
                end_us=step.end_us,
                wall_ms=step.dur_us / 1000.0,
                ctx_tokens=ctx_tokens,
                gen_tokens=gen_tokens,
                union_busy_ms=union_busy_us / 1000.0,
                internal_gap_ms=internal_gap_us / 1000.0,
                internal_gap_count=internal_gap_count,
                a2a_adjacent_gap_ms=a2a_gap_us / 1000.0,
                a2a_adjacent_gap_count=a2a_gap_count,
                ep_a2a_kernel_count=ep_a2a_kernel_count,
            )
        )
    return parsed_steps


def _aggregate_steps(steps: list[ProfilerStep]) -> list[StepAggregate]:
    grouped: dict[tuple[str, int], list[ProfilerStep]] = {}
    for step in steps:
        grouped.setdefault((step.window, step.step_index), []).append(step)
    aggregates: list[StepAggregate] = []
    for (window, step_index), group in sorted(grouped.items()):
        ctx_tokens = max(step.ctx_tokens for step in group)
        aggregates.append(
            StepAggregate(
                window=window,
                step_index=step_index,
                step_type="prefill_or_mixed" if ctx_tokens > 0 else "decode_only",
                decode_batch=max(float(step.gen_tokens) for step in group),
                wall_ms=_mean([step.wall_ms for step in group]),
                union_busy_ms=_mean([step.union_busy_ms for step in group]),
                internal_gap_ms=_mean([step.internal_gap_ms for step in group]),
                wall_bubble_ms=_mean([step.wall_bubble_ms for step in group]),
                internal_gap_count=_mean([float(step.internal_gap_count) for step in group]),
                a2a_adjacent_gap_ms=_mean([step.a2a_adjacent_gap_ms for step in group]),
                a2a_adjacent_gap_count=_mean([float(step.a2a_adjacent_gap_count) for step in group]),
                ep_a2a_kernel_count=_mean([float(step.ep_a2a_kernel_count) for step in group]),
            )
        )
    return aggregates


def _window_gap_summary(steps: list[StepAggregate], window: str) -> dict[str, object]:
    selected = [step for step in steps if step.window == window]
    if not selected:
        raise ValueError(f"missing parsed steps for {window}")
    batches = [step.decode_batch for step in selected if step.decode_batch > 0]
    return {
        "window": window,
        "step_count": len(selected),
        "prefill_step_count": sum(1 for step in selected if step.step_type == "prefill_or_mixed"),
        "decode_step_count": sum(1 for step in selected if step.step_type == "decode_only"),
        "decode_batch_mean": _mean(batches),
        "decode_batch_min": min(batches) if batches else 0,
        "decode_batch_max": max(batches) if batches else 0,
        "gap_ms_per_step_mean": _mean([step.internal_gap_ms for step in selected]),
        "wall_bubble_ms_per_step_mean": _mean([step.wall_bubble_ms for step in selected]),
        "a2a_adjacent_gap_ms_per_step_mean": _mean([step.a2a_adjacent_gap_ms for step in selected]),
        "gap_segments_per_step_mean": _mean([step.internal_gap_count for step in selected]),
        "a2a_gap_segments_per_step_mean": _mean([step.a2a_adjacent_gap_count for step in selected]),
        "ep_a2a_kernels_per_step_mean": _mean([step.ep_a2a_kernel_count for step in selected]),
    }


def _fit_hypotheses(steps: list[StepAggregate]) -> list[dict[str, object]]:
    prefill_steps = [step for step in steps if step.step_type == "prefill_or_mixed"]
    decode_steps = [step for step in steps if step.step_type == "decode_only" and step.decode_batch > 0]
    if not prefill_steps or not decode_steps:
        raise ValueError("Phase432 needs both prefill and decode steps")

    prefill_target = _mean([step.internal_gap_ms for step in prefill_steps])
    prefill_a2a_count = _mean([step.a2a_adjacent_gap_count for step in prefill_steps])
    decode_a2a_count = _mean([step.a2a_adjacent_gap_count for step in decode_steps])
    decode_batch_mean = _mean([step.decode_batch for step in decode_steps])
    a2a_points = [(step.decode_batch, step.a2a_adjacent_gap_count) for step in decode_steps]
    token_points = [(step.decode_batch, step.internal_gap_ms) for step in decode_steps]

    total_a2a_gap = sum(step.a2a_adjacent_gap_ms for step in steps)
    total_a2a_count = sum(step.a2a_adjacent_gap_count for step in steps)
    h1_param = total_a2a_gap / total_a2a_count if total_a2a_count > 0 else 0.0
    h2_param = sum(step.internal_gap_ms for step in steps) / (len(steps) * LAYER_COUNT)
    h3_param = _slope(token_points)

    specs = [
        {
            "hypothesis": "H1_per_a2a_call",
            "parameter_ms": h1_param,
            "prefill_predicted_ms": h1_param * prefill_a2a_count,
            "decode_constant_predicted_ms": h1_param * decode_a2a_count,
            "ep_a2a_excess_slope_predicted_ms_per_request": h1_param * _slope(a2a_points),
            "cross_32k3k_predicted_ms": h1_param * prefill_a2a_count,
        },
        {
            "hypothesis": "H2_per_layer",
            "parameter_ms": h2_param,
            "prefill_predicted_ms": h2_param * LAYER_COUNT,
            "decode_constant_predicted_ms": h2_param * LAYER_COUNT,
            "ep_a2a_excess_slope_predicted_ms_per_request": 0.0,
            "cross_32k3k_predicted_ms": h2_param * LAYER_COUNT,
        },
        {
            "hypothesis": "H3_per_decode_request",
            "parameter_ms": h3_param,
            "prefill_predicted_ms": h3_param * _mean([step.decode_batch for step in prefill_steps]),
            "decode_constant_predicted_ms": h3_param * decode_batch_mean,
            "ep_a2a_excess_slope_predicted_ms_per_request": h3_param,
            "cross_32k3k_predicted_ms": h3_param * PHASE414_32K_PREFILL_GEN_REQS_MEAN,
        },
    ]
    rows: list[dict[str, object]] = []
    for spec in specs:
        prefill_error = _safe_ratio(float(spec["prefill_predicted_ms"]), prefill_target)
        decode_error = _safe_ratio(
            float(spec["decode_constant_predicted_ms"]),
            PHASE431_DECODE_CONSTANT_TARGET_MS,
        )
        ep_error = _safe_ratio(
            float(spec["ep_a2a_excess_slope_predicted_ms_per_request"]),
            PHASE431_EP_A2A_EXCESS_SLOPE_TARGET_MS_PER_REQUEST,
        )
        cross_error = _safe_ratio(
            float(spec["cross_32k3k_predicted_ms"]),
            PHASE414_32K_PREFILL_EXECUTION_GAP_TARGET_MS,
        )
        passed = all(error <= STRUCTURE_TOLERANCE for error in (prefill_error, decode_error, ep_error))
        rows.append(
            {
                **spec,
                "prefill_target_ms": prefill_target,
                "prefill_error_ratio": prefill_error,
                "decode_constant_target_ms": PHASE431_DECODE_CONSTANT_TARGET_MS,
                "decode_constant_error_ratio": decode_error,
                "ep_a2a_excess_slope_target_ms_per_request": (
                    PHASE431_EP_A2A_EXCESS_SLOPE_TARGET_MS_PER_REQUEST
                ),
                "ep_a2a_excess_error_ratio": ep_error,
                "cross_32k3k_target_ms": PHASE414_32K_PREFILL_EXECUTION_GAP_TARGET_MS,
                "cross_32k3k_error_ratio": cross_error,
                "consistency_gate": "passed" if passed else "failed",
            }
        )
    return rows


def _common_flags() -> dict[str, object]:
    return {
        "phase405_penalty_read": False,
        "gpu_allowed": False,
        "ssh_allowed": False,
        "runtime_modified": False,
        "perf_database": False,
        "valid_for_default": False,
        "diagnostic_only": True,
        "default_readiness": DEFAULT_READINESS,
    }


def build_phase432_rows(
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    phase429_csv: Path = PHASE429_CSV,
    scenario: str = DEFAULT_SCENARIO,
    max_ranks_per_window: int | None = None,
) -> list[dict[str, str]]:
    artifact_dir = artifact_root / scenario
    if not artifact_dir.exists():
        raise ValueError(f"missing Phase429 artifact dir: {artifact_dir}")
    phase429 = _read_phase429_summary(phase429_csv, scenario)

    parsed_steps: list[ProfilerStep] = []
    for window in WINDOWS:
        paths = _trace_paths(artifact_dir, window)
        if max_ranks_per_window is not None:
            paths = paths[:max_ranks_per_window]
        for path in paths:
            parsed_steps.extend(parse_trace_steps(path, window))
    step_aggs = _aggregate_steps(parsed_steps)
    gap_summaries = [_window_gap_summary(step_aggs, window) for window in WINDOWS]
    hypothesis_rows = _fit_hypotheses(step_aggs)

    passed = [row for row in hypothesis_rows if row["consistency_gate"] == "passed"]
    if passed:
        best = min(
            passed,
            key=lambda row: (
                float(row["prefill_error_ratio"])
                + float(row["decode_constant_error_ratio"])
                + float(row["ep_a2a_excess_error_ratio"])
            ),
        )
        selected = str(best["hypothesis"])
        verdict = f"execution_overhead_{selected}_structural"
        phase433_target = "runtime_add_structural_execution_overhead_entry"
    else:
        selected = "none"
        verdict = "execution_overhead_structure_unresolved"
        phase433_target = "gpu_nsys_cpu_side_trace_before_modeling"

    common = {
        "source": SOURCE,
        "scenario": scenario,
        "artifact_dir": _display_path(artifact_dir),
        "prefill_step_count": phase429.get("prefill_step_count", ""),
        "decode_step_count": phase429.get("decode_step_count", ""),
        "selected_hypothesis": selected,
        "phase432_verdict": verdict,
        "phase433_target": phase433_target,
        **_common_flags(),
    }

    rows: list[dict[str, object]] = [{"row_type": "summary", **common}]
    for summary in gap_summaries:
        rows.append({"row_type": "gap_summary", **common, **summary})
    for row in hypothesis_rows:
        rows.append({"row_type": "hypothesis", **common, **row})
    return [{field: _fmt(row.get(field, "")) for field in CSV_FIELDS} for row in rows]


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Phase432 rows must not be empty")
    summaries = [row for row in rows if row.get("row_type") == "summary"]
    if len(summaries) != 1:
        raise ValueError("Phase432 expects exactly one summary row")
    for row in rows:
        if row.get("source") != SOURCE:
            raise ValueError("unexpected source")
        if row.get("phase405_penalty_read") != FALSE:
            raise ValueError("phase405_penalty_read must stay false")
        if row.get("gpu_allowed") != FALSE or row.get("ssh_allowed") != FALSE:
            raise ValueError("Phase432 is offline and must not declare GPU/SSH use")
        if row.get("runtime_modified") != FALSE:
            raise ValueError("runtime_modified must stay false")
        if row.get("perf_database") != FALSE:
            raise ValueError("perf_database must stay false")
        if row.get("valid_for_default") != FALSE:
            raise ValueError("valid_for_default must stay false")
        if row.get("diagnostic_only") != TRUE:
            raise ValueError("diagnostic_only must stay true")
        if row.get("default_readiness") != DEFAULT_READINESS:
            raise ValueError("default_readiness must stay No-Go")


def write_phase432_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_phase432_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    summary = [row for row in rows if row["row_type"] == "summary"][0]
    gap_rows = [row for row in rows if row["row_type"] == "gap_summary"]
    hypothesis_rows = [row for row in rows if row["row_type"] == "hypothesis"]
    lines = [
        "# Phase432 Bubble Structure",
        "",
        "Phase432 is report-only. It reads Phase429 traces and does not change runtime, PerfDatabase, or gate.",
        "",
        "## Verdict",
        "",
        f"- verdict: `{summary['phase432_verdict']}`",
        f"- selected hypothesis: `{summary['selected_hypothesis']}`",
        f"- Phase433 target: `{summary['phase433_target']}`",
        "",
        "## Gap Timeline",
        "",
        "| window | steps | prefill | decode | batch mean | internal gap ms/step | wall bubble ms/step | a2a-adjacent gap ms/step | a2a gap segments/step | ep kernels/step |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in gap_rows:
        lines.append(
            f"| {row['window']} | {row['step_count']} | {row['prefill_step_count']} | "
            f"{row['decode_step_count']} | {row['decode_batch_mean']} | "
            f"{row['gap_ms_per_step_mean']} | {row['wall_bubble_ms_per_step_mean']} | "
            f"{row['a2a_adjacent_gap_ms_per_step_mean']} | "
            f"{row['a2a_gap_segments_per_step_mean']} | {row['ep_a2a_kernels_per_step_mean']} |"
        )
    lines.extend(
        [
            "",
            "## Hypothesis Consistency",
            "",
            "| hypothesis | parameter ms | prefill err | decode const err | ep excess err | gate |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in hypothesis_rows:
        lines.append(
            f"| {row['hypothesis']} | {row['parameter_ms']} | {row['prefill_error_ratio']} | "
            f"{row['decode_constant_error_ratio']} | {row['ep_a2a_excess_error_ratio']} | "
            f"{row['consistency_gate']} |"
        )
    lines.extend(
        [
            "",
            "## 32k3k Cross-Check",
            "",
            "| hypothesis | target ms/step | predicted ms/step | error ratio |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in hypothesis_rows:
        lines.append(
            f"| {row['hypothesis']} | {row['cross_32k3k_target_ms']} | "
            f"{row['cross_32k3k_predicted_ms']} | {row['cross_32k3k_error_ratio']} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- A hypothesis only qualifies if the same parameter explains prefill gap, decode constant gap, and ep_a2a serving excess within 20%.",
            "- No hypothesis passed all three checks, so Phase432 does not authorize adding a runtime overhead term.",
            "- GPU/SSH: not used in Phase432.",
            "- Runtime/PerfDatabase/gate: not modified.",
            "- Default AIC: No-Go.",
            "",
        ]
    )
    return "\n".join(lines)


def write_phase432_md(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase432_md(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--phase429-csv", type=Path, default=PHASE429_CSV)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--max-ranks-per-window", type=int)
    args = parser.parse_args()

    rows = build_phase432_rows(
        artifact_root=args.artifact_root,
        phase429_csv=args.phase429_csv,
        scenario=args.scenario,
        max_ranks_per_window=args.max_ranks_per_window,
    )
    write_phase432_csv(args.csv, rows)
    write_phase432_md(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
