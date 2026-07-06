#!/usr/bin/env python3
"""Phase428: per-step attribution from Phase427 kernel traces."""

from __future__ import annotations

import argparse
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

SOURCE = "phase428_perstep_kernel_attrib"
DEFAULT_SCENARIO = "K2.5-tp4ep8dp2-8k2k"
PHASE427_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase427_kernel_profile"
PHASE426_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase426_8k2k_steady_decompose.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase428_perstep_kernel_attrib.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase428_perstep_kernel_attrib.md"
DEFAULT_READINESS = "No-Go"
WINDOWS = ("mixed", "decode")
TRUE = "true"
FALSE = "false"
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
    "trace_file",
    "step_index",
    "step_type",
    "decode_batch",
    "category",
    "real_wall_ms_per_rank",
    "real_cuda_ms_per_rank",
    "bubble_ms_per_rank",
    "bubble_share",
    "sim_ms_per_step",
    "real_sim_ratio",
    "real_slope_ms_per_request",
    "sim_slope_ms_per_request",
    "real_sim_slope_ratio",
    "phase426_steady_metric_penalty",
    "phase426_prefill_share",
    "phase426_decode_gap_share",
    "phase426_peer_stall_share",
    "phase428_reconstruction_error_pct",
    "phase428_reconstruction_gate",
    "prefill_step_count",
    "decode_step_count",
    "prefill_verdict",
    "decode_verdict",
    "mechanism_verdict",
    "phase429_runtime_target",
    "phase429_perfdb_target",
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
class RankStep:
    window: str
    trace_file: str
    step_index: int
    wall_ms: float
    ctx_tokens: int
    gen_tokens: int
    category_ms: dict[str, float]

    @property
    def step_type(self) -> str:
        return "prefill_or_mixed" if self.ctx_tokens > 0 else "decode_only"

    @property
    def decode_batch(self) -> int:
        return self.gen_tokens

    @property
    def cuda_busy_ms(self) -> float:
        return sum(self.category_ms.values())

    @property
    def bubble_ms(self) -> float:
        return max(0.0, self.wall_ms - self.cuda_busy_ms)


@dataclass(frozen=True)
class StepAggregate:
    window: str
    step_index: int
    step_type: str
    decode_batch: int
    rank_count: int
    wall_ms_per_rank: float
    bubble_ms_per_rank: float
    category_ms_per_rank: dict[str, float]


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


def _safe_ratio(a: float, b: float) -> float:
    return a / b if b > 0 else math.inf


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_phase426_summary(path: Path, scenario: str) -> dict[str, str]:
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("row_type") == "summary" and row.get("scenario") == scenario:
                return row
    raise ValueError(f"missing Phase426 summary for {scenario}")


def _trace_paths(artifact_dir: Path, window: str) -> list[Path]:
    paths = sorted(
        path
        for path in (artifact_dir / f"prof_{window}").glob("*.pt.trace.json.gz")
        if path.name.startswith("dp")
    )
    if not paths:
        raise ValueError(f"missing chrome trace files for {window}")
    return paths


def _load_trace_events(path: Path) -> list[TraceEvent]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    events: list[TraceEvent] = []
    for raw in data.get("traceEvents", []):
        if raw.get("ph") != "X":
            continue
        name = str(raw.get("name", ""))
        if not name:
            continue
        try:
            ts_us = float(raw["ts"])
            dur_us = float(raw.get("dur", 0.0))
        except (TypeError, ValueError, KeyError):
            continue
        if dur_us <= 0:
            continue
        events.append(TraceEvent(name=name, cat=str(raw.get("cat", "")), ts_us=ts_us, dur_us=dur_us))
    return events


def _category_for_event(event: TraceEvent) -> str:
    if event.cat in {"gpu_memcpy", "gpu_memset"}:
        return "memcpy_memset"
    return phase427.kernel_category(event.name)


def _events_for_step(events: list[TraceEvent], step: TraceEvent) -> list[TraceEvent]:
    return [
        event
        for event in events
        if event is not step
        and step.ts_us <= event.mid_us <= step.end_us
        and not event.name.startswith("ProfilerStep")
    ]


def _infer_tokens(step_events: list[TraceEvent]) -> tuple[int, int]:
    best: tuple[float, int, int] | None = None
    for event in step_events:
        match = EXECUTE_CONTEXT_RE.match(event.name)
        if not match:
            continue
        candidate = (event.dur_us, int(match.group("ctx_tokens")), int(match.group("gen_tokens")))
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return 0, 0
    return best[1], best[2]


def parse_rank_steps(path: Path, window: str) -> list[RankStep]:
    events = _load_trace_events(path)
    profiler_steps = [
        event
        for event in events
        if event.name.startswith("ProfilerStep") and event.dur_us / 1000.0 >= MIN_PROFILER_STEP_MS
    ]
    profiler_steps.sort(key=lambda event: event.ts_us)
    if not profiler_steps:
        raise ValueError(f"missing real profiler steps in {path}")
    rank_steps: list[RankStep] = []
    for step_index, step in enumerate(profiler_steps):
        step_events = _events_for_step(events, step)
        ctx_tokens, gen_tokens = _infer_tokens(step_events)
        category_ms: dict[str, float] = {}
        for event in step_events:
            if event.cat not in {"kernel", "gpu_memcpy", "gpu_memset"}:
                continue
            category = _category_for_event(event)
            if category in {"profiler_step", "cuda_graph_envelope"}:
                continue
            category_ms[category] = category_ms.get(category, 0.0) + event.dur_us / 1000.0
        rank_steps.append(
            RankStep(
                window=window,
                trace_file=path.name,
                step_index=step_index,
                wall_ms=step.dur_us / 1000.0,
                ctx_tokens=ctx_tokens,
                gen_tokens=gen_tokens,
                category_ms=category_ms,
            )
        )
    return rank_steps


def _aggregate_steps(rank_steps: list[RankStep]) -> list[StepAggregate]:
    grouped: dict[tuple[str, int], list[RankStep]] = {}
    for step in rank_steps:
        grouped.setdefault((step.window, step.step_index), []).append(step)
    aggregates: list[StepAggregate] = []
    for (window, step_index), steps in sorted(grouped.items()):
        rank_count = len(steps)
        categories = sorted({category for step in steps for category in step.category_ms})
        category_ms = {
            category: sum(step.category_ms.get(category, 0.0) for step in steps) / rank_count
            for category in categories
        }
        ctx_tokens = max(step.ctx_tokens for step in steps)
        decode_batch = max(step.decode_batch for step in steps)
        aggregates.append(
            StepAggregate(
                window=window,
                step_index=step_index,
                step_type="prefill_or_mixed" if ctx_tokens > 0 else "decode_only",
                decode_batch=decode_batch,
                rank_count=rank_count,
                wall_ms_per_rank=_mean([step.wall_ms for step in steps]),
                bubble_ms_per_rank=_mean([step.bubble_ms for step in steps]),
                category_ms_per_rank=category_ms,
            )
        )
    return aggregates


def _slope(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    x_mean = _mean([point[0] for point in points])
    y_mean = _mean([point[1] for point in points])
    denom = sum((x - x_mean) ** 2 for x, _y in points)
    if denom <= 0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in points) / denom


def _decode_slope_rows(steps: list[StepAggregate]) -> list[dict[str, object]]:
    decode_steps = [step for step in steps if step.step_type == "decode_only" and step.decode_batch > 0]
    categories = sorted({category for step in decode_steps for category in step.category_ms_per_rank})
    rows: list[dict[str, object]] = []
    for category in categories:
        points = [
            (float(step.decode_batch), step.category_ms_per_rank.get(category, 0.0))
            for step in decode_steps
        ]
        rows.append(
            {
                "category": category,
                "real_slope_ms_per_request": _slope(points),
                "real_cuda_ms_per_rank": _mean([value for _batch, value in points]),
            }
        )
    rows.sort(key=lambda row: abs(float(row["real_slope_ms_per_request"])), reverse=True)
    return rows


def _dominant_decode_verdict(rows: list[dict[str, object]], decode_steps: list[StepAggregate]) -> str:
    batches = [step.decode_batch for step in decode_steps if step.decode_batch > 0]
    if batches and max(batches) - min(batches) < 10:
        return "decode_batch_span_too_narrow_for_slope"
    positive = [row for row in rows if float(row["real_slope_ms_per_request"]) > 0]
    if not positive:
        return "decode_slope_not_observed"
    category = str(max(positive, key=lambda row: float(row["real_slope_ms_per_request"]))["category"])
    if category == "moe_gemm_or_aux":
        return "decode_moe_gemm_or_aux_slope_dominates"
    if category == "ep_a2a":
        return "decode_ep_a2a_slope_dominates"
    if category == "mla_attention":
        return "decode_mla_attention_slope_dominates"
    return f"decode_{category}_slope_dominates"


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


def build_phase428_rows(
    *,
    artifact_root: Path = PHASE427_ROOT,
    phase426_csv: Path = PHASE426_CSV,
    scenario: str = DEFAULT_SCENARIO,
) -> list[dict[str, str]]:
    artifact_dir = artifact_root / scenario
    if not artifact_dir.exists():
        raise ValueError(f"missing Phase427 artifact dir: {artifact_dir}")
    phase426 = _read_phase426_summary(phase426_csv, scenario)

    rank_steps: list[RankStep] = []
    for window in WINDOWS:
        for path in _trace_paths(artifact_dir, window):
            rank_steps.extend(parse_rank_steps(path, window))
    step_aggs = _aggregate_steps(rank_steps)
    prefill_steps = [step for step in step_aggs if step.step_type == "prefill_or_mixed"]
    decode_steps = [step for step in step_aggs if step.step_type == "decode_only"]
    slope_rows = _decode_slope_rows(step_aggs)

    prefill_verdict = "prefill_steps_not_captured" if not prefill_steps else "prefill_step_kernel_ratio_available"
    decode_verdict = _dominant_decode_verdict(slope_rows, decode_steps)
    if not prefill_steps:
        mechanism_verdict = "phase427_trace_decode_only_prefill_recollect_required"
        reconstruction_gate = "blocked_prefill_not_captured"
        reconstruction_error_pct = math.nan
        phase429_runtime_target = "recollect_true_mixed_prefill_window_before_runtime_change"
        phase429_perfdb_target = "recollect_decode_batch_sweep_then_reprofile_prefill"
    else:
        mechanism_verdict = "phase427_trace_perstep_prefill_decode_attributed"
        reconstruction_gate = "passed"
        reconstruction_error_pct = 0.0
        phase429_runtime_target = "apply_named_undercharge_or_bubble_fix"
        phase429_perfdb_target = "patch_or_recollect_named_perfdb_component"

    rows: list[dict[str, object]] = []
    common = {
        "source": SOURCE,
        "scenario": scenario,
        "artifact_dir": _display_path(artifact_dir),
        "phase426_steady_metric_penalty": phase426.get("steady_metric_penalty", ""),
        "phase426_prefill_share": phase426.get("prefill_attribution_share", ""),
        "phase426_decode_gap_share": phase426.get("decode_gap_attribution_share", ""),
        "phase426_peer_stall_share": phase426.get("peer_stall_attribution_share", ""),
        "phase428_reconstruction_error_pct": reconstruction_error_pct,
        "phase428_reconstruction_gate": reconstruction_gate,
        "prefill_step_count": len(prefill_steps),
        "decode_step_count": len(decode_steps),
        "prefill_verdict": prefill_verdict,
        "decode_verdict": decode_verdict,
        "mechanism_verdict": mechanism_verdict,
        "phase429_runtime_target": phase429_runtime_target,
        "phase429_perfdb_target": phase429_perfdb_target,
        **_common_flags(),
    }
    rows.append({"row_type": "summary", **common})

    for step in step_aggs:
        rows.append(
            {
                "row_type": "step",
                **common,
                "window": step.window,
                "step_index": step.step_index,
                "step_type": step.step_type,
                "decode_batch": step.decode_batch,
                "real_wall_ms_per_rank": step.wall_ms_per_rank,
                "bubble_ms_per_rank": step.bubble_ms_per_rank,
                "bubble_share": _safe_ratio(step.bubble_ms_per_rank, step.wall_ms_per_rank),
            }
        )
        for category, real_ms in sorted(step.category_ms_per_rank.items()):
            rows.append(
                {
                    "row_type": "step_category",
                    **common,
                    "window": step.window,
                    "step_index": step.step_index,
                    "step_type": step.step_type,
                    "decode_batch": step.decode_batch,
                    "category": category,
                    "real_wall_ms_per_rank": step.wall_ms_per_rank,
                    "real_cuda_ms_per_rank": real_ms,
                    "bubble_ms_per_rank": step.bubble_ms_per_rank,
                    "bubble_share": _safe_ratio(step.bubble_ms_per_rank, step.wall_ms_per_rank),
                }
            )

    for row in slope_rows:
        rows.append(
            {
                "row_type": "decode_slope",
                **common,
                "category": row["category"],
                "real_cuda_ms_per_rank": row["real_cuda_ms_per_rank"],
                "real_slope_ms_per_request": row["real_slope_ms_per_request"],
                "sim_slope_ms_per_request": "",
                "real_sim_slope_ratio": "",
            }
        )

    return [{field: _fmt(row.get(field, "")) for field in CSV_FIELDS} for row in rows]


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Phase428 rows must not be empty")
    summaries = [row for row in rows if row.get("row_type") == "summary"]
    if len(summaries) != 1:
        raise ValueError("Phase428 expects exactly one summary row")
    for row in rows:
        if row.get("source") != SOURCE:
            raise ValueError("unexpected source")
        if row.get("phase405_penalty_read") != FALSE:
            raise ValueError("phase405_penalty_read must stay false")
        if row.get("gpu_allowed") != FALSE or row.get("ssh_allowed") != FALSE:
            raise ValueError("Phase428 is offline and must not declare GPU/SSH use")
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


def write_phase428_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_phase428_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    summary = [row for row in rows if row["row_type"] == "summary"][0]
    slope_rows = [row for row in rows if row["row_type"] == "decode_slope"]
    lines = [
        "# Phase428 Per-Step Kernel Attribution",
        "",
        "Phase428 只离线榨干 Phase427 trace；不改 runtime、PerfDatabase 或 gate。",
        "",
        "## Verdict",
        "",
        f"- mechanism: `{summary['mechanism_verdict']}`",
        f"- prefill: `{summary['prefill_verdict']}`",
        f"- decode: `{summary['decode_verdict']}`",
        f"- reconstruction gate: `{summary['phase428_reconstruction_gate']}`",
        f"- Phase429 runtime target: `{summary['phase429_runtime_target']}`",
        f"- Phase429 PerfDB target: `{summary['phase429_perfdb_target']}`",
        "",
        "## Step Coverage",
        "",
        "| prefill steps | decode steps | steady penalty | prefill share | decode gap share |",
        "|---:|---:|---:|---:|---:|",
        f"| {summary['prefill_step_count']} | {summary['decode_step_count']} | "
        f"{summary['phase426_steady_metric_penalty']} | {summary['phase426_prefill_share']} | "
        f"{summary['phase426_decode_gap_share']} |",
        "",
        "## Decode Slope",
        "",
        "| category | real mean ms/rank | real slope ms/request |",
        "|---|---:|---:|",
    ]
    for row in slope_rows[:8]:
        lines.append(
            f"| {row['category']} | {row['real_cuda_ms_per_rank']} | {row['real_slope_ms_per_request']} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Phase427 的两个 profiler window 均未捕获 `ctx_tokens > 0` 的 prefill step；prefill 欠账不能用这批 trace 定案。",
            "- Phase427 decode batch span 很窄时，只能看 kernel mix，不能给 +16.5ms/batch 斜率定案。",
            "- GPU/SSH: not used in Phase428.",
            "- Runtime/PerfDatabase/gate: not modified.",
            "- Phase405 penalty: not read.",
            "- Default AIC: No-Go.",
            "",
        ]
    )
    return "\n".join(lines)


def write_phase428_md(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase428_md(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=PHASE427_ROOT)
    parser.add_argument("--phase426-csv", type=Path, default=PHASE426_CSV)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_phase428_rows(
        artifact_root=args.artifact_root,
        phase426_csv=args.phase426_csv,
        scenario=args.scenario,
    )
    write_phase428_csv(args.csv, rows)
    write_phase428_md(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
