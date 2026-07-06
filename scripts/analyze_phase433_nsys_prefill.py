#!/usr/bin/env python3
"""Phase433: attribute 32k3k prefill execution overhead from profiler traces."""

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


SOURCE = "phase433_nsys_prefill"
DEFAULT_SCENARIO = "K2.5-tp4ep8dp2-32k3k"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase433_nsys_prefill"
PHASE415_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase415_prefill_charge_components.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase433_nsys_prefill.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase433_nsys_prefill.md"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"
TARGET_CTX_TOKENS = 31_000
TARGET_MISSING_MS = 4718.954849 - 1818.637996
RECONSTRUCTION_TOLERANCE = 0.20
MIN_PROFILER_STEP_MS = 1.0

EXECUTE_CONTEXT_RE = re.compile(
    r"execute_context_(?P<ctx_requests>\d+)\((?P<ctx_tokens>\d+)\)_"
    r"generation_(?P<gen_requests>\d+)\((?P<gen_tokens>\d+)\)"
)
RANK_RE = re.compile(r"rank(?P<rank>\d+)")
DP_RANK_RE = re.compile(r"dp(?P<dp>\d+).*rank(?P<rank>\d+)")

COMPONENT_TO_CATEGORY = {
    "prefill_mla_attention": "mla_attention",
    "moe_compute": "moe_gemm_or_aux",
    "ep_dispatch_combine": "ep_a2a",
    "gemm_other": "dense_gemm",
    "tp_comm": "collective_other",
    "other": "other_cuda",
}

CSV_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "artifact_dir",
    "profiler_mode",
    "nsys_available",
    "trace_precision",
    "window",
    "trace_file",
    "rank",
    "step_index",
    "step_type",
    "ctx_tokens",
    "gen_tokens",
    "step_count",
    "prefill_step_count",
    "wall_ms_per_step",
    "cuda_busy_ms_per_step",
    "bubble_ms_per_step",
    "category",
    "category_real_ms_per_step",
    "category_sim_ms_per_step",
    "category_excess_ms_per_step",
    "category_share_of_real_cuda",
    "per_gpu_busy_ms",
    "per_gpu_busy_ratio_to_mean",
    "max_gpu_busy_ms",
    "min_gpu_busy_ms",
    "max_mean_busy_ratio",
    "collective_wait_or_transfer_ms",
    "serving_slow_kernel_ms",
    "cpu_gap_ms",
    "transfer_or_memcpy_ms",
    "target_missing_ms",
    "reconstructed_gap_ms",
    "reconstruction_error_pct",
    "reconstruction_gate",
    "dominant_mechanism",
    "phase434_target",
    "precision_boundary",
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
    rank: str
    step_index: int
    wall_ms: float
    ctx_tokens: int
    gen_tokens: int
    union_busy_ms: float
    category_ms: dict[str, float]

    @property
    def step_type(self) -> str:
        return "prefill_or_mixed" if self.ctx_tokens > 0 else "decode_only"

    @property
    def bubble_ms(self) -> float:
        return max(0.0, self.wall_ms - self.union_busy_ms)


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
    return value / target


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _trace_paths(artifact_dir: Path, window: str) -> list[Path]:
    prof_dir = artifact_dir / f"prof_{window}"
    paths = sorted(path for path in prof_dir.glob("*.pt.trace.json.gz") if path.name.startswith("dp"))
    if not paths:
        raise ValueError(f"missing chrome trace files for {window} under {prof_dir}")
    return paths


def _read_phase415_baseline(path: Path, scenario: str) -> dict[str, float]:
    baselines: dict[str, float] = {}
    summary = math.nan
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("scenario") != scenario:
                continue
            component = row.get("component", "")
            if row.get("row_type") == "summary" and component == "summary":
                summary = float(row.get("sim_mean_ms") or 0.0)
                continue
            if row.get("row_type") != "mixed_component":
                continue
            category = COMPONENT_TO_CATEGORY.get(component)
            if not category:
                continue
            baselines[category] = baselines.get(category, 0.0) + float(row.get("sim_mean_ms") or 0.0)
    if not baselines:
        raise ValueError(f"missing Phase415 mixed baselines for {scenario}")
    if math.isnan(summary):
        summary = sum(baselines.values())
    baselines["__summary__"] = summary
    return baselines


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
        except (KeyError, TypeError, ValueError):
            continue
        if dur_us <= 0:
            continue
        events.append(TraceEvent(name=name, cat=str(raw.get("cat", "")), ts_us=ts_us, dur_us=dur_us))
    return events


def _category_for_event(event: TraceEvent) -> str:
    if event.cat in {"gpu_memcpy", "gpu_memset"}:
        return "memcpy_memset"
    lower = event.name.lower()
    if "nccl" in lower:
        if (
            "allgather" in lower
            or "all_gather" in lower
            or "reducescatter" in lower
            or "reduce_scatter" in lower
        ):
            return "ep_a2a"
        return "collective_other"
    if "cross_device_reduce" in lower or "custom_ar" in lower or "all_reduce" in lower or "allreduce" in lower:
        return "collective_other"
    return phase427.kernel_category(event.name)


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


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        merged[-1] = (prev_start, max(prev_end, end))
    return merged


def _rank_from_path(path: Path) -> str:
    dp_match = DP_RANK_RE.search(path.name)
    if dp_match:
        return f"dp{dp_match.group('dp')}_rank{dp_match.group('rank')}"
    match = RANK_RE.search(path.name)
    return f"rank{match.group('rank')}" if match else ""


def parse_rank_steps(path: Path, window: str) -> list[RankStep]:
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
    payloads: list[list[TraceEvent]] = [[] for _step in raw_steps]
    intervals: list[list[tuple[float, float]]] = [[] for _step in raw_steps]
    category_payloads: list[dict[str, float]] = [dict() for _step in raw_steps]
    for event in events:
        if event.name.startswith("ProfilerStep"):
            continue
        idx = bisect.bisect_right(step_starts, event.mid_us) - 1
        if idx < 0:
            continue
        step = raw_steps[idx]
        if event.mid_us > step.end_us:
            continue
        payloads[idx].append(event)
        if event.cat not in {"kernel", "gpu_memcpy", "gpu_memset"}:
            continue
        category = _category_for_event(event)
        if category in {"profiler_step", "cuda_graph_envelope"}:
            continue
        clipped_start = max(event.ts_us, step.ts_us)
        clipped_end = min(event.end_us, step.end_us)
        if clipped_end <= clipped_start:
            continue
        intervals[idx].append((clipped_start, clipped_end))
        category_payloads[idx][category] = category_payloads[idx].get(category, 0.0) + (
            clipped_end - clipped_start
        ) / 1000.0

    rank = _rank_from_path(path)
    parsed: list[RankStep] = []
    for idx, step in enumerate(raw_steps):
        ctx_tokens, gen_tokens = _infer_tokens(payloads[idx])
        merged = _merge_intervals(intervals[idx])
        union_busy = sum(end - start for start, end in merged) / 1000.0
        parsed.append(
            RankStep(
                window=window,
                trace_file=path.name,
                rank=rank,
                step_index=idx,
                wall_ms=step.dur_us / 1000.0,
                ctx_tokens=ctx_tokens,
                gen_tokens=gen_tokens,
                union_busy_ms=union_busy,
                category_ms=category_payloads[idx],
            )
        )
    return parsed


def _select_prefill_steps(steps: list[RankStep], min_ctx_tokens: int) -> list[RankStep]:
    return [step for step in steps if step.ctx_tokens >= min_ctx_tokens]


def _dominant_breakdown(values: dict[str, float]) -> str:
    positive = {key: value for key, value in values.items() if value > 0}
    if not positive:
        return "phase433_inconclusive_no_positive_gap_component"
    dominant = max(positive, key=positive.get)
    if dominant == "collective_wait_or_transfer_ms":
        return "collective_or_transfer_dominates_torch_fallback_cannot_split_wait_vs_transfer"
    if dominant == "cpu_gap_ms":
        return "cpu_or_host_gap_dominates"
    if dominant == "serving_slow_kernel_ms":
        return "serving_slow_kernel_excess_dominates"
    if dominant == "transfer_or_memcpy_ms":
        return "explicit_memcpy_transfer_dominates"
    return f"{dominant}_dominates"


def _common_flags() -> dict[str, object]:
    return {
        "phase405_penalty_read": False,
        "gpu_allowed": True,
        "ssh_allowed": True,
        "runtime_modified": False,
        "perf_database": False,
        "valid_for_default": False,
        "diagnostic_only": True,
        "default_readiness": DEFAULT_READINESS,
    }


def build_window_check(
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    scenario: str = DEFAULT_SCENARIO,
    window: str = "prefill",
    phase415_csv: Path = PHASE415_CSV,
    min_ctx_tokens: int = TARGET_CTX_TOKENS,
) -> dict[str, object]:
    # phase415_csv is accepted so the runner can call the same script signature
    # before the full report generation. It is not needed for the hit check.
    _ = phase415_csv
    artifact_dir = artifact_root / scenario
    steps: list[RankStep] = []
    for path in _trace_paths(artifact_dir, window):
        steps.extend(parse_rank_steps(path, window))
    prefill = _select_prefill_steps(steps, min_ctx_tokens)
    max_ctx = max((step.ctx_tokens for step in steps), default=0)
    return {
        "passed": bool(prefill),
        "scenario": scenario,
        "window": window,
        "trace_count": len(_trace_paths(artifact_dir, window)),
        "prefill_step_count": len(prefill),
        "max_ctx_tokens": max_ctx,
        "min_ctx_tokens": min_ctx_tokens,
    }


def build_phase433_rows(
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    phase415_csv: Path = PHASE415_CSV,
    scenario: str = DEFAULT_SCENARIO,
    window: str = "prefill",
    min_ctx_tokens: int = TARGET_CTX_TOKENS,
) -> list[dict[str, str]]:
    artifact_dir = artifact_root / scenario
    if not artifact_dir.exists():
        raise ValueError(f"missing Phase433 artifact dir: {artifact_dir}")
    meta = _load_json(artifact_dir / "meta.json")
    profiler_mode = str(meta.get("profiler_mode", "unknown"))
    nsys_available = bool(meta.get("nsys_available", False))
    precision_boundary = (
        "nsys_wait_transfer_split_available"
        if nsys_available
        else "torch_profiler_fallback_cannot_split_nccl_wait_from_transfer"
    )
    trace_precision = "nsys" if nsys_available else "torch_profiler_fallback"
    baselines = _read_phase415_baseline(phase415_csv, scenario)

    all_steps: list[RankStep] = []
    paths = _trace_paths(artifact_dir, window)
    for path in paths:
        all_steps.extend(parse_rank_steps(path, window))
    target_steps = _select_prefill_steps(all_steps, min_ctx_tokens)
    if not target_steps:
        raise ValueError("no 32k prefill/mixed profiler steps found")

    category_real: dict[str, float] = {}
    for step in target_steps:
        for category, value in step.category_ms.items():
            category_real[category] = category_real.get(category, 0.0) + value
    for category in list(category_real):
        category_real[category] /= len(target_steps)

    wall_ms = _mean([step.wall_ms for step in target_steps])
    cuda_busy_ms = _mean([step.union_busy_ms for step in target_steps])
    bubble_ms = _mean([step.bubble_ms for step in target_steps])
    step_count = len(target_steps)
    rank_busy: dict[str, list[float]] = {}
    for step in target_steps:
        rank_busy.setdefault(step.rank, []).append(step.union_busy_ms)
    rank_busy_mean = {rank: _mean(values) for rank, values in rank_busy.items()}
    busy_values = list(rank_busy_mean.values())
    busy_mean = _mean(busy_values)
    max_busy = max(busy_values) if busy_values else math.nan
    min_busy = min(busy_values) if busy_values else math.nan
    max_mean_ratio = _safe_ratio(max_busy, busy_mean) if busy_values else math.nan

    category_excess: dict[str, float] = {}
    for category, real_value in category_real.items():
        category_excess[category] = max(0.0, real_value - baselines.get(category, 0.0))

    collective = category_excess.get("ep_a2a", 0.0) + category_excess.get("collective_other", 0.0)
    memcpy = category_real.get("memcpy_memset", 0.0)
    slow_categories = {
        category: value
        for category, value in category_excess.items()
        if category not in {"ep_a2a", "collective_other", "memcpy_memset"}
    }
    slow_kernel = sum(slow_categories.values())
    breakdown = {
        "collective_wait_or_transfer_ms": collective,
        "serving_slow_kernel_ms": slow_kernel,
        "cpu_gap_ms": bubble_ms,
        "transfer_or_memcpy_ms": memcpy,
    }
    reconstructed = sum(breakdown.values())
    reconstruction_error = abs(reconstructed - TARGET_MISSING_MS) / TARGET_MISSING_MS * 100.0
    reconstruction_gate = "passed" if reconstruction_error <= RECONSTRUCTION_TOLERANCE * 100.0 else "failed"
    dominant = _dominant_breakdown(breakdown)
    phase434_target = (
        "separate_nccl_wait_vs_transfer_or_build_serving_state_ep_model"
        if "collective_or_transfer" in dominant
        else "gpu_nsys_required_before_runtime_model"
    )

    common = {
        "source": SOURCE,
        "scenario": scenario,
        "artifact_dir": _display_path(artifact_dir),
        "profiler_mode": profiler_mode,
        "nsys_available": nsys_available,
        "trace_precision": trace_precision,
        "window": window,
        "step_count": len(all_steps),
        "prefill_step_count": step_count,
        "wall_ms_per_step": wall_ms,
        "cuda_busy_ms_per_step": cuda_busy_ms,
        "bubble_ms_per_step": bubble_ms,
        "max_gpu_busy_ms": max_busy,
        "min_gpu_busy_ms": min_busy,
        "max_mean_busy_ratio": max_mean_ratio,
        "collective_wait_or_transfer_ms": collective,
        "serving_slow_kernel_ms": slow_kernel,
        "cpu_gap_ms": bubble_ms,
        "transfer_or_memcpy_ms": memcpy,
        "target_missing_ms": TARGET_MISSING_MS,
        "reconstructed_gap_ms": reconstructed,
        "reconstruction_error_pct": reconstruction_error,
        "reconstruction_gate": reconstruction_gate,
        "dominant_mechanism": dominant,
        "phase434_target": phase434_target,
        "precision_boundary": precision_boundary,
        **_common_flags(),
    }

    rows: list[dict[str, object]] = [{"row_type": "summary", **common}]
    for category, real_value in sorted(category_real.items()):
        sim_value = baselines.get(category, 0.0)
        rows.append(
            {
                "row_type": "category",
                **common,
                "category": category,
                "category_real_ms_per_step": real_value,
                "category_sim_ms_per_step": sim_value,
                "category_excess_ms_per_step": max(0.0, real_value - sim_value),
                "category_share_of_real_cuda": _safe_ratio(real_value, sum(category_real.values())),
            }
        )
    for rank, value in sorted(rank_busy_mean.items(), key=lambda item: item[0]):
        rows.append(
            {
                "row_type": "per_gpu_busy",
                **common,
                "rank": rank,
                "per_gpu_busy_ms": value,
                "per_gpu_busy_ratio_to_mean": _safe_ratio(value, busy_mean),
            }
        )
    for step in target_steps:
        rows.append(
            {
                "row_type": "prefill_step",
                **common,
                "trace_file": step.trace_file,
                "rank": step.rank,
                "step_index": step.step_index,
                "step_type": step.step_type,
                "ctx_tokens": step.ctx_tokens,
                "gen_tokens": step.gen_tokens,
                "wall_ms_per_step": step.wall_ms,
                "cuda_busy_ms_per_step": step.union_busy_ms,
                "bubble_ms_per_step": step.bubble_ms,
            }
        )

    return [{field: _fmt(row.get(field, "")) for field in CSV_FIELDS} for row in rows]


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Phase433 rows must not be empty")
    summaries = [row for row in rows if row.get("row_type") == "summary"]
    if len(summaries) != 1:
        raise ValueError("Phase433 expects exactly one summary row")
    for row in rows:
        if row.get("source") != SOURCE:
            raise ValueError("unexpected source")
        if row.get("phase405_penalty_read") != FALSE:
            raise ValueError("phase405_penalty_read must stay false")
        if row.get("gpu_allowed") != TRUE or row.get("ssh_allowed") != TRUE:
            raise ValueError("Phase433 is a GPU measurement phase and must declare GPU/SSH use")
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


def write_phase433_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_phase433_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    summary = [row for row in rows if row["row_type"] == "summary"][0]
    categories = [row for row in rows if row["row_type"] == "category"]
    busy = [row for row in rows if row["row_type"] == "per_gpu_busy"]
    lines = [
        "# Phase433 NSYS Prefill Forensics",
        "",
        "Phase433 is measurement-only. It does not modify runtime, PerfDatabase, or gate.",
        "",
        "## Verdict",
        "",
        f"- profiler mode: `{summary['profiler_mode']}`",
        f"- precision boundary: `{summary['precision_boundary']}`",
        f"- mechanism: `{summary['dominant_mechanism']}`",
        f"- reconstruction gate: `{summary['reconstruction_gate']}` "
        f"({summary['reconstruction_error_pct']}% error)",
        f"- Phase434 target: `{summary['phase434_target']}`",
        "",
        "## Four-Way Decomposition",
        "",
        "| item | ms/step |",
        "|---|---:|",
        f"| collective wait or transfer | {summary['collective_wait_or_transfer_ms']} |",
        f"| serving slow kernel excess | {summary['serving_slow_kernel_ms']} |",
        f"| CPU/host or unobserved gap | {summary['cpu_gap_ms']} |",
        f"| explicit memcpy/memset transfer | {summary['transfer_or_memcpy_ms']} |",
        f"| reconstructed | {summary['reconstructed_gap_ms']} |",
        f"| target missing | {summary['target_missing_ms']} |",
        "",
        "## Per-GPU Busy",
        "",
        "| rank | busy ms | ratio to mean |",
        "|---:|---:|---:|",
    ]
    for row in busy:
        lines.append(f"| {row['rank']} | {row['per_gpu_busy_ms']} | {row['per_gpu_busy_ratio_to_mean']} |")
    lines.extend(
        [
            "",
            "## Kernel Categories",
            "",
            "| category | real ms/step | sim ms/step | excess ms/step |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in categories:
        lines.append(
            f"| {row['category']} | {row['category_real_ms_per_step']} | "
            f"{row['category_sim_ms_per_step']} | {row['category_excess_ms_per_step']} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- `nsys` was not available when `trace_precision=torch_profiler_fallback`; NCCL wait and wire transfer remain a combined bucket.",
            "- The output is diagnostic-only and not valid for Default AIC.",
            "- Runtime/PerfDatabase/gate: not modified.",
            "- Default AIC: No-Go.",
            "",
        ]
    )
    return "\n".join(lines)


def write_phase433_md(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase433_md(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--phase415-csv", type=Path, default=PHASE415_CSV)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--window", default="prefill")
    parser.add_argument("--min-ctx-tokens", type=int, default=TARGET_CTX_TOKENS)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--check-window", action="store_true")
    args = parser.parse_args()

    if args.check_window:
        check = build_window_check(
            artifact_root=args.artifact_root,
            scenario=args.scenario,
            window=args.window,
            phase415_csv=args.phase415_csv,
            min_ctx_tokens=args.min_ctx_tokens,
        )
        print(json.dumps(check, sort_keys=True))
        raise SystemExit(0 if check["passed"] else 1)

    rows = build_phase433_rows(
        artifact_root=args.artifact_root,
        phase415_csv=args.phase415_csv,
        scenario=args.scenario,
        window=args.window,
        min_ctx_tokens=args.min_ctx_tokens,
    )
    write_phase433_csv(args.csv, rows)
    write_phase433_md(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
