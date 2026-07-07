#!/usr/bin/env python3
"""Phase434: split cross-rank collective wait and test expert imbalance."""

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

from scripts import analyze_phase433_nsys_prefill as phase433  # noqa: E402


SOURCE = "phase434_crossrank_wait_split"
DEFAULT_SCENARIO = phase433.DEFAULT_SCENARIO
DEFAULT_ARTIFACT_ROOT = phase433.DEFAULT_ARTIFACT_ROOT
DEFAULT_PHASE433_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase433_nsys_prefill.csv"
DEFAULT_DECODE_ARTIFACT_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase429_reprofile"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase434_crossrank_wait_split.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase434_crossrank_wait_split.md"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"
MIN_CTX_TOKENS = 31_000
CLOCK_P95_GATE_MS = 10.0
DUAL_CHECK_TOLERANCE_PCT = 20.0

CSV_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "artifact_dir",
    "trace_precision",
    "profiler_mode",
    "rank",
    "rank_count",
    "occurrence",
    "category",
    "collective_count",
    "clock_offset_ms",
    "clock_residual_median_abs_ms",
    "clock_residual_p95_abs_ms",
    "clock_residual_iqr_ms",
    "clock_alignment_gate",
    "wait_ms_per_step",
    "transfer_ms_per_step",
    "matched_collective_ms_per_step",
    "matched_coverage_pct",
    "real_collective_ms_per_step",
    "sim_collective_ms_per_step",
    "phase433_collective_bucket_ms",
    "collective_excess_reconstructed_ms",
    "collective_bucket_error_pct",
    "moe_mean_ms_per_step",
    "moe_hot_ms_per_step",
    "moe_hot_mean_ratio",
    "moe_hot_excess_ms_per_step",
    "phase433_moe_excess_ms",
    "phase433_slow_kernel_bucket_ms",
    "wait_vs_hot_excess_error_pct",
    "moe_excess_from_ratio_ms",
    "moe_excess_ratio_error_pct",
    "dual_check_gate",
    "decode_crosscheck_available",
    "decode_wait_ms_per_step",
    "decode_transfer_ms_per_step",
    "dominant_mechanism",
    "verdict",
    "phase435_target",
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
class KernelSlice:
    rank: str
    occurrence: int
    category: str
    name: str
    start_us: float
    end_us: float

    @property
    def duration_ms(self) -> float:
        return (self.end_us - self.start_us) / 1000.0


@dataclass(frozen=True)
class StepSlice:
    rank: str
    occurrence: int
    step_index: int
    ctx_tokens: int
    gen_tokens: int
    category_ms: dict[str, float]


@dataclass(frozen=True)
class SplitResult:
    category: str
    event_count: int
    wait_ms_per_step: float
    transfer_ms_per_step: float
    matched_real_ms_per_step: float


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


def _median(values: list[float]) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return ordered[low]
    return ordered[low] * (high - pos) + ordered[high] * (pos - low)


def _safe_ratio(value: float, target: float) -> float:
    if target == 0:
        return 0.0 if value == 0 else math.inf
    return value / target


def _pct_error(value: float, target: float) -> float:
    if target == 0:
        return 0.0 if value == 0 else math.inf
    return abs(value - target) / abs(target) * 100.0


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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


def _load_phase433_reference(path: Path) -> dict[str, float]:
    defaults = {
        "collective_bucket": math.nan,
        "slow_kernel_bucket": math.nan,
        "sim_ep_a2a": 0.0,
        "sim_collective_other": 0.0,
        "sim_all_collective": 0.0,
        "sim_moe": 0.0,
        "moe_excess": math.nan,
    }
    if not path.exists():
        return defaults
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("scenario") != DEFAULT_SCENARIO:
                continue
            if row.get("row_type") == "summary":
                defaults["collective_bucket"] = float(row.get("collective_wait_or_transfer_ms") or "nan")
                defaults["slow_kernel_bucket"] = float(row.get("serving_slow_kernel_ms") or "nan")
            if row.get("row_type") != "category":
                continue
            category = row.get("category")
            sim = float(row.get("category_sim_ms_per_step") or 0.0)
            excess = float(row.get("category_excess_ms_per_step") or 0.0)
            if category == "ep_a2a":
                defaults["sim_ep_a2a"] = sim
            elif category == "collective_other":
                defaults["sim_collective_other"] = sim
            elif category == "moe_gemm_or_aux":
                defaults["sim_moe"] = sim
                defaults["moe_excess"] = excess
    defaults["sim_all_collective"] = defaults["sim_ep_a2a"] + defaults["sim_collective_other"]
    return defaults


def _profiler_steps(events: list[phase433.TraceEvent]) -> list[phase433.TraceEvent]:
    return sorted(
        [event for event in events if event.name.startswith("ProfilerStep")],
        key=lambda event: event.ts_us,
    )


def _load_rank_data(path: Path, window: str, min_ctx_tokens: int) -> tuple[list[StepSlice], list[KernelSlice]]:
    events = phase433._load_trace_events(path)
    profiler_steps = _profiler_steps(events)
    parsed_steps = phase433.parse_rank_steps(path, window)
    rank = phase433._rank_from_path(path)
    selected = [step for step in parsed_steps if step.ctx_tokens >= min_ctx_tokens]
    step_by_index = {step.step_index: occurrence for occurrence, step in enumerate(selected)}
    steps = [
        StepSlice(
            rank=rank,
            occurrence=occurrence,
            step_index=step.step_index,
            ctx_tokens=step.ctx_tokens,
            gen_tokens=step.gen_tokens,
            category_ms=step.category_ms,
        )
        for occurrence, step in enumerate(selected)
    ]
    kernels: list[KernelSlice] = []
    step_starts = [step.ts_us for step in profiler_steps]
    for event in events:
        if event.name.startswith("ProfilerStep"):
            continue
        if event.cat not in {"kernel", "gpu_memcpy", "gpu_memset"}:
            continue
        idx = _bisect_step(step_starts, profiler_steps, event.mid_us)
        if idx is None:
            continue
        occurrence = step_by_index.get(idx)
        if occurrence is None:
            continue
        category = phase433._category_for_event(event)
        if category in {"ep_a2a", "collective_other", "moe_gemm_or_aux"}:
            kernels.append(
                KernelSlice(
                    rank=rank,
                    occurrence=occurrence,
                    category=category,
                    name=event.name,
                    start_us=max(event.ts_us, profiler_steps[idx].ts_us),
                    end_us=min(event.end_us, profiler_steps[idx].end_us),
                )
            )
    return steps, kernels


def _bisect_step(
    step_starts: list[float],
    profiler_steps: list[phase433.TraceEvent],
    event_mid_us: float,
) -> int | None:
    import bisect

    idx = bisect.bisect_right(step_starts, event_mid_us) - 1
    if idx < 0:
        return None
    if event_mid_us > profiler_steps[idx].end_us:
        return None
    return idx


def _events_by_rank(kernels: list[KernelSlice], *, categories: set[str]) -> dict[str, list[KernelSlice]]:
    by_rank: dict[str, list[KernelSlice]] = {}
    for event in kernels:
        if event.category not in categories:
            continue
        by_rank.setdefault(event.rank, []).append(event)
    for values in by_rank.values():
        values.sort(key=lambda event: (event.occurrence, event.start_us, event.end_us))
    return by_rank


def _clock_offsets(by_rank: dict[str, list[KernelSlice]]) -> tuple[dict[str, float], list[float]]:
    ranks = sorted(by_rank)
    if not ranks:
        return {}, []
    ref = ranks[0]
    offsets = {ref: 0.0}
    residuals_ms: list[float] = []
    for rank in ranks[1:]:
        count = min(len(by_rank[ref]), len(by_rank[rank]))
        if count == 0:
            offsets[rank] = 0.0
            continue
        diffs = [by_rank[rank][idx].end_us - by_rank[ref][idx].end_us for idx in range(count)]
        offset = _median(diffs)
        offsets[rank] = offset
        residuals_ms.extend((diff - offset) / 1000.0 for diff in diffs)
    return offsets, residuals_ms


def _split_collectives(
    kernels: list[KernelSlice],
    offsets: dict[str, float],
    *,
    categories: set[str],
    category_name: str,
    occurrence_count: int,
) -> SplitResult:
    by_rank = _events_by_rank(kernels, categories=categories)
    if not by_rank or occurrence_count <= 0:
        return SplitResult(category_name, 0, math.nan, math.nan, math.nan)
    ranks = sorted(by_rank)
    total_wait_ms = 0.0
    total_transfer_ms = 0.0
    total_real_ms = 0.0
    event_count = 0
    for occurrence in range(occurrence_count):
        per_rank = {
            rank: [event for event in by_rank.get(rank, []) if event.occurrence == occurrence]
            for rank in ranks
        }
        local_count = min((len(events) for events in per_rank.values()), default=0)
        for idx in range(local_count):
            starts = [per_rank[rank][idx].start_us - offsets.get(rank, 0.0) for rank in ranks]
            ends = [per_rank[rank][idx].end_us - offsets.get(rank, 0.0) for rank in ranks]
            max_start = max(starts)
            max_end = max(ends)
            waits = [max(0.0, max_start - start) for start in starts]
            durations = [max(0.0, end - start) for start, end in zip(starts, ends)]
            total_wait_ms += _mean(waits) / 1000.0
            total_transfer_ms += max(0.0, max_end - max_start) / 1000.0
            total_real_ms += _mean(durations) / 1000.0
            event_count += 1
    return SplitResult(
        category=category_name,
        event_count=event_count,
        wait_ms_per_step=total_wait_ms / occurrence_count,
        transfer_ms_per_step=total_transfer_ms / occurrence_count,
        matched_real_ms_per_step=total_real_ms / occurrence_count,
    )


def _step_imbalance(steps: list[StepSlice]) -> dict[str, float]:
    by_occurrence: dict[int, list[float]] = {}
    for step in steps:
        by_occurrence.setdefault(step.occurrence, []).append(step.category_ms.get("moe_gemm_or_aux", 0.0))
    ratios: list[float] = []
    hot_excesses: list[float] = []
    hot_values: list[float] = []
    mean_values: list[float] = []
    for values in by_occurrence.values():
        if not values:
            continue
        mean_value = _mean(values)
        hot_value = max(values)
        ratios.append(_safe_ratio(hot_value, mean_value))
        hot_excesses.append(max(0.0, hot_value - mean_value))
        hot_values.append(hot_value)
        mean_values.append(mean_value)
    return {
        "mean": _mean(mean_values),
        "hot": _mean(hot_values),
        "ratio": _mean(ratios),
        "ratio_p95": _quantile(ratios, 0.95),
        "hot_excess": _mean(hot_excesses),
    }


def _observed_category_ms(steps: list[StepSlice], category: str) -> float:
    if not steps:
        return math.nan
    return _mean([step.category_ms.get(category, 0.0) for step in steps])


def _observed_collective_ms(steps: list[StepSlice], categories: set[str]) -> float:
    if not steps:
        return math.nan
    return _mean([sum(step.category_ms.get(category, 0.0) for category in categories) for step in steps])


def _collect_prefill_data(
    artifact_root: Path,
    scenario: str,
    window: str,
    min_ctx_tokens: int,
) -> tuple[Path, list[StepSlice], list[KernelSlice], int]:
    artifact_dir = artifact_root / scenario
    paths = phase433._trace_paths(artifact_dir, window)
    all_steps: list[StepSlice] = []
    all_kernels: list[KernelSlice] = []
    occurrence_counts: list[int] = []
    for path in paths:
        steps, kernels = _load_rank_data(path, window, min_ctx_tokens)
        all_steps.extend(steps)
        all_kernels.extend(kernels)
        occurrence_counts.append(len(steps))
    common_occurrences = min(occurrence_counts) if occurrence_counts else 0
    all_steps = [step for step in all_steps if step.occurrence < common_occurrences]
    all_kernels = [kernel for kernel in all_kernels if kernel.occurrence < common_occurrences]
    return artifact_dir, all_steps, all_kernels, common_occurrences


def _decode_crosscheck() -> dict[str, float | str | bool]:
    scenario = "K2.5-tp4ep8dp2-8k2k"
    prof_dir = DEFAULT_DECODE_ARTIFACT_ROOT / scenario / "prof_w3_decode_c128"
    paths = sorted(path for path in prof_dir.glob("dp*.pt.trace.json.gz"))
    if not paths:
        return {"available": False}
    all_steps: list[StepSlice] = []
    all_kernels: list[KernelSlice] = []
    occurrence_counts: list[int] = []
    for path in paths:
        events = phase433._load_trace_events(path)
        profiler_steps = _profiler_steps(events)
        parsed_steps = phase433.parse_rank_steps(path, "decode_c128")
        rank = phase433._rank_from_path(path)
        selected_raw = [step for step in parsed_steps if step.ctx_tokens == 0 and step.gen_tokens > 0]
        selected = selected_raw[: min(10, len(selected_raw))]
        step_by_index = {step.step_index: occurrence for occurrence, step in enumerate(selected)}
        occurrence_counts.append(len(selected))
        for occurrence, step in enumerate(selected):
            all_steps.append(
                StepSlice(
                    rank=rank,
                    occurrence=occurrence,
                    step_index=step.step_index,
                    ctx_tokens=step.ctx_tokens,
                    gen_tokens=step.gen_tokens,
                    category_ms=step.category_ms,
                )
            )
        step_starts = [step.ts_us for step in profiler_steps]
        for event in events:
            if event.cat not in {"kernel", "gpu_memcpy", "gpu_memset"}:
                continue
            idx = _bisect_step(step_starts, profiler_steps, event.mid_us)
            if idx is None:
                continue
            occurrence = step_by_index.get(idx)
            if occurrence is None:
                continue
            category = phase433._category_for_event(event)
            if category in {"ep_a2a", "collective_other"}:
                all_kernels.append(
                    KernelSlice(
                        rank=rank,
                        occurrence=occurrence,
                        category=category,
                        name=event.name,
                        start_us=max(event.ts_us, profiler_steps[idx].ts_us),
                        end_us=min(event.end_us, profiler_steps[idx].end_us),
                    )
                )
    occurrence_count = min(occurrence_counts) if occurrence_counts else 0
    all_kernels = [kernel for kernel in all_kernels if kernel.occurrence < occurrence_count]
    by_rank = _events_by_rank(all_kernels, categories={"ep_a2a", "collective_other"})
    offsets, _ = _clock_offsets(by_rank)
    split = _split_collectives(
        all_kernels,
        offsets,
        categories={"ep_a2a", "collective_other"},
        category_name="decode_all_collective",
        occurrence_count=occurrence_count,
    )
    return {
        "available": True,
        "wait_ms": split.wait_ms_per_step,
        "transfer_ms": split.transfer_ms_per_step,
    }


def build_phase434_rows(
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    phase433_csv: Path = DEFAULT_PHASE433_CSV,
    scenario: str = DEFAULT_SCENARIO,
    window: str = "prefill",
    min_ctx_tokens: int = MIN_CTX_TOKENS,
) -> list[dict[str, str]]:
    artifact_dir, steps, kernels, occurrence_count = _collect_prefill_data(
        artifact_root, scenario, window, min_ctx_tokens
    )
    if occurrence_count == 0:
        raise ValueError("missing cross-rank 32k prefill occurrences")
    ranks = sorted({step.rank for step in steps})
    references = _load_phase433_reference(phase433_csv)
    by_rank = _events_by_rank(kernels, categories={"ep_a2a", "collective_other"})
    offsets, residuals = _clock_offsets(by_rank)
    residual_abs = [abs(value) for value in residuals]
    residual_iqr = _quantile(residuals, 0.75) - _quantile(residuals, 0.25) if residuals else math.nan
    residual_median_abs = _median(residual_abs)
    residual_p95_abs = _quantile(residual_abs, 0.95)
    clock_gate = "passed" if residual_p95_abs <= CLOCK_P95_GATE_MS else "low_confidence"

    split_ep = _split_collectives(
        kernels,
        offsets,
        categories={"ep_a2a"},
        category_name="ep_a2a",
        occurrence_count=occurrence_count,
    )
    split_other = _split_collectives(
        kernels,
        offsets,
        categories={"collective_other"},
        category_name="collective_other",
        occurrence_count=occurrence_count,
    )
    split_all = _split_collectives(
        kernels,
        offsets,
        categories={"ep_a2a", "collective_other"},
        category_name="all_collective",
        occurrence_count=occurrence_count,
    )
    observed_ep = _observed_category_ms(steps, "ep_a2a")
    observed_other = _observed_category_ms(steps, "collective_other")
    observed_all = _observed_collective_ms(steps, {"ep_a2a", "collective_other"})
    sim_collective = references["sim_all_collective"]
    collective_excess = max(0.0, observed_all - sim_collective)
    phase433_collective_bucket = references["collective_bucket"]
    collective_error = _pct_error(collective_excess, phase433_collective_bucket)

    imbalance = _step_imbalance(steps)
    wait_vs_hot_error = _pct_error(imbalance["hot_excess"], split_all.wait_ms_per_step)
    moe_excess_from_ratio = max(0.0, references["sim_moe"] * (imbalance["ratio"] - 1.0))
    moe_excess_error = _pct_error(moe_excess_from_ratio, references["moe_excess"])
    dual_gate = (
        "passed"
        if wait_vs_hot_error <= DUAL_CHECK_TOLERANCE_PCT
        and moe_excess_error <= DUAL_CHECK_TOLERANCE_PCT
        else "failed"
    )
    if clock_gate != "passed":
        verdict = "crossrank_split_low_confidence_expert_imbalance_not_confirmed"
    elif dual_gate == "passed":
        verdict = "expert_imbalance_root_cause_confirmed"
    else:
        verdict = "expert_imbalance_not_confirmed_collective_transfer_or_serving_state_more_likely"
    phase435_target = (
        "model_expert_imbalance_wait_and_hot_kernel"
        if verdict == "expert_imbalance_root_cause_confirmed"
        else "separate_nccl_wait_vs_transfer_or_build_serving_state_ep_model"
    )
    if clock_gate != "passed":
        dominant = "crossrank_alignment_low_confidence"
    elif split_all.wait_ms_per_step > split_all.transfer_ms_per_step:
        dominant = "collective_wait_dominates"
    else:
        dominant = "collective_transfer_dominates"
    precision_boundary = (
        "torch_profiler_crossrank_end_alignment_estimate_not_nsys_wire_split"
    )
    decode = _decode_crosscheck()

    common = {
        "source": SOURCE,
        "scenario": scenario,
        "artifact_dir": _display_path(artifact_dir),
        "trace_precision": "torch_profiler_crossrank_fallback",
        "profiler_mode": "torch_profiler_cpu_fallback",
        "rank_count": len(ranks),
        "clock_residual_median_abs_ms": residual_median_abs,
        "clock_residual_p95_abs_ms": residual_p95_abs,
        "clock_residual_iqr_ms": residual_iqr,
        "clock_alignment_gate": clock_gate,
        "phase433_collective_bucket_ms": phase433_collective_bucket,
        "phase433_moe_excess_ms": references["moe_excess"],
        "phase433_slow_kernel_bucket_ms": references["slow_kernel_bucket"],
        "wait_vs_hot_excess_error_pct": wait_vs_hot_error,
        "moe_excess_from_ratio_ms": moe_excess_from_ratio,
        "moe_excess_ratio_error_pct": moe_excess_error,
        "dual_check_gate": dual_gate,
        "decode_crosscheck_available": bool(decode.get("available", False)),
        "decode_wait_ms_per_step": decode.get("wait_ms", math.nan),
        "decode_transfer_ms_per_step": decode.get("transfer_ms", math.nan),
        "dominant_mechanism": dominant,
        "verdict": verdict,
        "phase435_target": phase435_target,
        "precision_boundary": precision_boundary,
        **_common_flags(),
    }

    rows: list[dict[str, object]] = [
        {
            "row_type": "summary",
            **common,
            "wait_ms_per_step": split_all.wait_ms_per_step,
            "transfer_ms_per_step": split_all.transfer_ms_per_step,
            "matched_collective_ms_per_step": split_all.matched_real_ms_per_step,
            "matched_coverage_pct": _safe_ratio(split_all.matched_real_ms_per_step, observed_all) * 100.0,
            "real_collective_ms_per_step": observed_all,
            "sim_collective_ms_per_step": sim_collective,
            "collective_excess_reconstructed_ms": collective_excess,
            "collective_bucket_error_pct": collective_error,
            "moe_mean_ms_per_step": imbalance["mean"],
            "moe_hot_ms_per_step": imbalance["hot"],
            "moe_hot_mean_ratio": imbalance["ratio"],
            "moe_hot_excess_ms_per_step": imbalance["hot_excess"],
        }
    ]
    for split in [split_all, split_ep, split_other]:
        sim_value = sim_collective
        observed_value = observed_all
        if split.category == "ep_a2a":
            sim_value = references["sim_ep_a2a"]
            observed_value = observed_ep
        elif split.category == "collective_other":
            sim_value = references["sim_collective_other"]
            observed_value = observed_other
        excess = max(0.0, observed_value - sim_value)
        rows.append(
            {
                "row_type": "collective_split",
                **common,
                "category": split.category,
                "collective_count": split.event_count,
                "wait_ms_per_step": split.wait_ms_per_step,
                "transfer_ms_per_step": split.transfer_ms_per_step,
                "matched_collective_ms_per_step": split.matched_real_ms_per_step,
                "matched_coverage_pct": _safe_ratio(split.matched_real_ms_per_step, observed_value) * 100.0,
                "real_collective_ms_per_step": observed_value,
                "sim_collective_ms_per_step": sim_value,
                "collective_excess_reconstructed_ms": excess,
                "collective_bucket_error_pct": _pct_error(excess, phase433_collective_bucket)
                if split.category == "all_collective"
                else "",
            }
        )
    rows.append(
        {
            "row_type": "imbalance_summary",
            **common,
            "moe_mean_ms_per_step": imbalance["mean"],
            "moe_hot_ms_per_step": imbalance["hot"],
            "moe_hot_mean_ratio": imbalance["ratio"],
            "moe_hot_excess_ms_per_step": imbalance["hot_excess"],
        }
    )
    for rank, offset in sorted(offsets.items()):
        rows.append(
            {
                "row_type": "clock_offset",
                **common,
                "rank": rank,
                "clock_offset_ms": offset / 1000.0,
            }
        )

    return [{field: _fmt(row.get(field, "")) for field in CSV_FIELDS} for row in rows]


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Phase434 rows must not be empty")
    summaries = [row for row in rows if row.get("row_type") == "summary"]
    if len(summaries) != 1:
        raise ValueError("Phase434 expects exactly one summary row")
    for row in rows:
        if row.get("source") != SOURCE:
            raise ValueError("unexpected source")
        if row.get("phase405_penalty_read") != FALSE:
            raise ValueError("phase405_penalty_read must stay false")
        if row.get("gpu_allowed") != FALSE or row.get("ssh_allowed") != FALSE:
            raise ValueError("Phase434 is offline and must declare no GPU/SSH use")
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


def write_phase434_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_phase434_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    summary = [row for row in rows if row["row_type"] == "summary"][0]
    splits = [row for row in rows if row["row_type"] == "collective_split"]
    offsets = [row for row in rows if row["row_type"] == "clock_offset"]
    lines = [
        "# Phase434 Cross-Rank Wait Split",
        "",
        "Phase434 is report-only. It reuses Phase433 torch-profiler traces and does not modify runtime, PerfDatabase, or gate.",
        "",
        "## Verdict",
        "",
        f"- verdict: `{summary['verdict']}`",
        f"- dominant mechanism: `{summary['dominant_mechanism']}`",
        f"- clock alignment gate: `{summary['clock_alignment_gate']}` "
        f"(p95 abs residual {summary['clock_residual_p95_abs_ms']} ms)",
        f"- dual imbalance gate: `{summary['dual_check_gate']}`",
        f"- precision boundary: `{summary['precision_boundary']}`",
        f"- Phase435 target: `{summary['phase435_target']}`",
        "",
        "## Collective Split",
        "",
        "| category | wait ms/step | transfer ms/step | matched ms/step | coverage pct | observed real ms/step | sim ms/step | excess ms/step |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in splits:
        lines.append(
            f"| {row['category']} | {row['wait_ms_per_step']} | {row['transfer_ms_per_step']} | "
            f"{row['matched_collective_ms_per_step']} | {row['matched_coverage_pct']} | "
            f"{row['real_collective_ms_per_step']} | {row['sim_collective_ms_per_step']} | "
            f"{row['collective_excess_reconstructed_ms']} |"
        )
    lines.extend(
        [
            "",
            "## Expert Imbalance Check",
            "",
            "| item | value |",
            "|---|---:|",
            f"| MoE mean ms/step | {summary['moe_mean_ms_per_step']} |",
            f"| MoE hot ms/step | {summary['moe_hot_ms_per_step']} |",
            f"| hot/mean ratio | {summary['moe_hot_mean_ratio']} |",
            f"| hot excess ms/step | {summary['moe_hot_excess_ms_per_step']} |",
            f"| wait vs hot-excess error pct | {summary['wait_vs_hot_excess_error_pct']} |",
            f"| MoE excess from ratio ms | {summary['moe_excess_from_ratio_ms']} |",
            f"| MoE excess ratio error pct | {summary['moe_excess_ratio_error_pct']} |",
            "",
            "## Clock Offsets",
            "",
            "| rank | offset ms |",
            "|---|---:|",
        ]
    )
    for row in offsets:
        lines.append(f"| {row['rank']} | {row['clock_offset_ms']} |")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- `nsys` was not available in Phase433; this split is based on torch-profiler cross-rank end-time alignment.",
            "- DP replicas are phase-skewed in the captured window, so matched collective rows are a subset and the observed real column is the bucket-safe value.",
            "- NCCL wait and wire transfer remain an estimate, not an nsys-grade split.",
            "- Default AIC: No-Go.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_phase434_md(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase434_md(rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--phase433-csv", type=Path, default=DEFAULT_PHASE433_CSV)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--window", default="prefill")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_phase434_rows(
        artifact_root=args.artifact_root,
        phase433_csv=args.phase433_csv,
        scenario=args.scenario,
        window=args.window,
    )
    write_phase434_csv(args.csv, rows)
    write_phase434_md(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
