#!/usr/bin/env python3
"""Phase413: decompose sustained-arrival DP2 steady penalty."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
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

from scripts import analyze_phase409_iter_trace as phase409  # noqa: E402


SOURCE = "phase413_steady_decompose"
DEFAULT_SCENARIO = "K2.5-tp4ep8dp2-32k3k"
SWEEP_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase412_arrival_sweep"
PHASE412_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase412_arrival_sweep.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase413_steady_decompose.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase413_steady_decompose.md"
DEFAULT_READINESS = "No-Go"

CSV_FIELDS = [
    "source",
    "scenario",
    "artifact_dir",
    "steady_target_penalty",
    "steady_metric_penalty",
    "steady_penalty_error_pct",
    "steady_consistency_gate",
    "steady_window_start_ms",
    "steady_window_end_ms",
    "steady_window_wall_ms",
    "prefill_wall_ms",
    "peer_stall_extra_ms",
    "clean_decode_wall_ms",
    "clean_decode_gap_ms",
    "prefill_attribution_share",
    "peer_stall_attribution_share",
    "decode_gap_attribution_share",
    "intrinsic_decode_ms",
    "decode_only_step_count",
    "clean_decode_step_count",
    "peer_stalled_decode_step_count",
    "decode_batch_mean",
    "decode_batch_p50",
    "decode_batch_p90",
    "real_decode_ms_mean",
    "sim_decode_ms_mean",
    "real_sim_decode_latency_ratio",
    "mechanism_verdict",
    "phase414_target",
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


@dataclass(frozen=True)
class SteadyWindow:
    start_ms: float
    end_ms: float
    output_tok_s_gpu: float

    @property
    def wall_ms(self) -> float:
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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _median(values: list[float]) -> float:
    return _percentile(values, 0.5)


def _read_csv_by_scenario(path: Path, *, row_type: str | None = None) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if row_type is not None:
        rows = [row for row in rows if row.get("row_type") == row_type]
    return {row["scenario"]: row for row in rows}


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            return f.read()
    return path.read_text(encoding="utf-8", errors="replace")


def _serve_log_path(artifact_dir: Path) -> Path:
    for name in ("serve.log", "serve.log.gz"):
        path = artifact_dir / name
        if path.exists():
            return path
    raise ValueError(f"missing serve.log artifact under {artifact_dir}")


def _metrics_path(artifact_dir: Path) -> Path:
    for name in ("metrics.jsonl", "metrics.jsonl.gz"):
        path = artifact_dir / name
        if path.exists():
            return path
    raise ValueError(f"missing metrics artifact under {artifact_dir}")


def parse_iteration_steps(path: Path) -> list[phase409.IterationStep]:
    steps: list[phase409.IterationStep] = []
    for line in _read_text(path).splitlines():
        match = phase409.ITER_RE.search(line)
        if not match:
            continue
        steps.append(
            phase409.IterationStep(
                engine=match.group("engine"),
                iteration=int(match.group("iteration")),
                ctx_requests=int(match.group("ctx_req")),
                ctx_tokens=int(match.group("ctx_tok")),
                generation_requests=int(match.group("gen_req")),
                generation_tokens=int(match.group("gen_tok")),
                elapsed_ms=float(match.group("elapsed")),
            )
        )
    if not steps:
        raise ValueError(f"missing iteration details in {path}")
    return steps


def parse_metric_snapshots(path: Path) -> list[phase409.MetricSnapshot]:
    snapshots: list[phase409.MetricSnapshot] = []
    for raw in _read_text(path).splitlines():
        if not raw.strip():
            continue
        record = json.loads(raw)
        if int(record.get("status", 0)) != 200:
            continue
        prompt_tokens = {"0": 0.0, "1": 0.0}
        generation_tokens = {"0": 0.0, "1": 0.0}
        for metric_line in str(record.get("body", "")).splitlines():
            match = phase409.METRIC_RE.match(metric_line.strip())
            if not match:
                continue
            labels = {
                m.group("key"): m.group("value")
                for m in phase409.LABEL_RE.finditer(match.group("labels"))
            }
            engine = labels.get("engine")
            if engine not in prompt_tokens:
                continue
            value = float(match.group("value"))
            if match.group("name") == "vllm:prompt_tokens_total":
                prompt_tokens[engine] = value
            else:
                generation_tokens[engine] = value
        snapshots.append(
            phase409.MetricSnapshot(
                timestamp_s=phase409.datetime.fromisoformat(record["ts"]).timestamp(),
                prompt_tokens=prompt_tokens,
                generation_tokens=generation_tokens,
            )
        )
    if not snapshots:
        raise ValueError(f"missing metric snapshots in {path}")
    return snapshots


def steady_window_from_metrics(path: Path) -> SteadyWindow:
    snapshots = parse_metric_snapshots(path)
    intervals: list[tuple[float, float]] = []
    for before, after in zip(snapshots, snapshots[1:]):
        wall_s = max(0.0, after.timestamp_s - before.timestamp_s)
        gen_delta = sum(
            max(0.0, after.generation_tokens[engine] - before.generation_tokens[engine])
            for engine in ("0", "1")
        )
        if wall_s > 0 and gen_delta > 0:
            intervals.append((wall_s, gen_delta))
    if not intervals:
        raise ValueError(f"missing active generation intervals in {path}")
    trim = max(1, int(len(intervals) * 0.1)) if len(intervals) >= 5 else 0
    kept = intervals[trim:-trim] if trim else intervals
    if not kept:
        kept = intervals
        trim = 0
    start_s = sum(wall for wall, _ in intervals[:trim])
    wall_s = sum(wall for wall, _ in kept)
    generation_tokens = sum(tokens for _, tokens in kept)
    return SteadyWindow(
        start_ms=start_s * 1000.0,
        end_ms=(start_s + wall_s) * 1000.0,
        output_tok_s_gpu=_safe_ratio(generation_tokens, wall_s) / 8.0,
    )


def build_wallclock_timelines(steps: list[phase409.IterationStep]) -> dict[str, list[TimedStep]]:
    grouped: dict[str, list[phase409.IterationStep]] = {"0": [], "1": []}
    for step in steps:
        if step.engine in grouped:
            grouped[step.engine].append(step)
    timelines: dict[str, list[TimedStep]] = {}
    for engine, engine_steps in grouped.items():
        wall_ms = 0.0
        timeline: list[TimedStep] = []
        for step in sorted(engine_steps, key=lambda item: item.iteration):
            start_ms = wall_ms
            wall_ms += step.elapsed_ms
            timeline.append(TimedStep(step=step, start_ms=start_ms, end_ms=wall_ms))
        if not timeline:
            raise ValueError("Phase413 requires both DP engines")
        timelines[engine] = timeline
    return timelines


def _overlap_ms(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def _step_overlaps_window(step: TimedStep, window: SteadyWindow) -> float:
    overlap = _overlap_ms(step.start_ms, step.end_ms, window.start_ms, window.end_ms)
    return overlap if overlap > 1e-3 else 0.0


def _active_at(timeline: list[TimedStep], midpoint_ms: float) -> TimedStep | None:
    for step in timeline:
        if step.start_ms <= midpoint_ms < step.end_ms:
            return step
    return None


def _prefill_intervals(
    timeline: list[TimedStep],
    window: SteadyWindow,
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for step in timeline:
        if not step.step.has_prefill:
            continue
        start = max(step.start_ms, window.start_ms)
        end = min(step.end_ms, window.end_ms)
        if end - start > 1e-3:
            intervals.append((start, end))
    return intervals


def _has_prefill_overlap(
    step: TimedStep,
    peer_prefill_intervals: list[tuple[float, float]],
    window: SteadyWindow,
) -> bool:
    if not (step.step.ctx_tokens == 0 and step.step.generation_tokens > 0):
        return False
    step_start = max(step.start_ms, window.start_ms)
    step_end = min(step.end_ms, window.end_ms)
    for start, end in peer_prefill_intervals:
        if end <= step_start:
            continue
        if start >= step_end:
            break
        overlap = _overlap_ms(
            step_start,
            step_end,
            start,
            end,
        )
        if overlap > 0:
            return True
    return False


def _clean_decode_steps(
    timelines: dict[str, list[TimedStep]],
    window: SteadyWindow,
) -> list[TimedStep]:
    clean: list[TimedStep] = []
    prefill_intervals = {
        engine: _prefill_intervals(timeline, window)
        for engine, timeline in timelines.items()
    }
    for engine, timeline in timelines.items():
        peer = "1" if engine == "0" else "0"
        for step in timeline:
            if _step_overlaps_window(step, window) <= 0:
                continue
            if step.step.ctx_tokens == 0 and step.step.generation_tokens > 0:
                if not _has_prefill_overlap(step, prefill_intervals[peer], window):
                    clean.append(step)
    return clean


def _nearest_sim_latency(batch: int, sim_decode_ms_by_batch: dict[int, float]) -> float:
    if batch in sim_decode_ms_by_batch:
        return sim_decode_ms_by_batch[batch]
    if not sim_decode_ms_by_batch:
        return math.nan
    nearest = min(sim_decode_ms_by_batch, key=lambda item: (abs(item - batch), item))
    return sim_decode_ms_by_batch[nearest]


def _default_sim_decode_ms_by_batch(
    batches: list[int],
    *,
    decode_avg_kv_len: int = 32000,
) -> dict[int, float]:
    from aiconfigurator.sdk.backends.cb_simulator.iteration_latency import (  # noqa: PLC0415
        IterationLatencyCalculator,
    )
    from scripts import validate_cb_simulator as validate_cb  # noqa: PLC0415

    if not batches:
        return {}
    model, db, backend = validate_cb._load_model_and_db(tp=4, dp=2, moe_tp=1, moe_ep=8)
    calc = IterationLatencyCalculator(
        backend,
        model,
        db,
        overlap_factor=0.0,
        per_iteration_overhead_ms=0.0,
    )
    out: dict[int, float] = {}
    for batch in sorted(set(max(1, int(value)) for value in batches)):
        out[batch] = calc.compute(
            prefill_tokens=0,
            prefill_batch_size=0,
            prefill_seq_len=1,
            decode_batch_size=batch,
            decode_avg_kv_len=decode_avg_kv_len,
        )
    return out


def _wall_segments(
    timelines: dict[str, list[TimedStep]],
    window: SteadyWindow,
) -> tuple[float, float, float]:
    breakpoints = {window.start_ms, window.end_ms}
    for timeline in timelines.values():
        for step in timeline:
            if _step_overlaps_window(step, window) > 0:
                breakpoints.add(max(step.start_ms, window.start_ms))
                breakpoints.add(min(step.end_ms, window.end_ms))
    ordered = sorted(breakpoints)
    indices = {"0": 0, "1": 0}
    prefill_wall = 0.0
    decode_wall = 0.0
    idle_wall = 0.0
    for start, end in zip(ordered, ordered[1:]):
        if end <= start:
            continue
        midpoint = (start + end) / 2.0
        active: list[TimedStep] = []
        for engine in ("0", "1"):
            timeline = timelines[engine]
            idx = indices[engine]
            while idx + 1 < len(timeline) and timeline[idx].end_ms <= midpoint:
                idx += 1
            indices[engine] = idx
            step = timeline[idx]
            if step.start_ms <= midpoint < step.end_ms:
                active.append(step)
        duration = end - start
        if any(step.step.has_prefill for step in active):
            prefill_wall += duration
        elif any(step.step.has_generation for step in active):
            decode_wall += duration
        else:
            idle_wall += duration
    return prefill_wall, decode_wall, idle_wall


def summarize_steady_decompose(
    *,
    steps: list[phase409.IterationStep],
    window: SteadyWindow,
    steady_target_penalty: float,
    steady_metric_penalty: float,
    sim_decode_ms_by_batch: dict[int, float],
) -> dict[str, float | int | str]:
    timelines = build_wallclock_timelines(steps)
    clean_steps = _clean_decode_steps(timelines, window)
    if not clean_steps:
        raise ValueError("Phase413 requires at least one clean decode step")

    clean_by_batch: dict[int, list[float]] = {}
    for step in clean_steps:
        clean_by_batch.setdefault(step.step.generation_requests, []).append(step.step.elapsed_ms)
    clean_median_by_batch = {batch: _median(values) for batch, values in clean_by_batch.items()}
    intrinsic_decode_ms = min(clean_median_by_batch.values())

    decode_only_steps: list[TimedStep] = []
    peer_stalled_steps = 0
    peer_stall_extra_ms = 0.0
    prefill_intervals = {
        engine: _prefill_intervals(timeline, window)
        for engine, timeline in timelines.items()
    }
    for engine, timeline in timelines.items():
        peer = "1" if engine == "0" else "0"
        for step in timeline:
            overlap = _step_overlaps_window(step, window)
            if overlap <= 0:
                continue
            if step.step.ctx_tokens != 0 or step.step.generation_tokens <= 0:
                continue
            decode_only_steps.append(step)
            if _has_prefill_overlap(step, prefill_intervals[peer], window):
                expected = clean_median_by_batch.get(step.step.generation_requests, intrinsic_decode_ms)
                extra = max(0.0, step.step.elapsed_ms - expected) * (overlap / max(step.duration_ms, 1e-9))
                if extra > 0:
                    peer_stalled_steps += 1
                    peer_stall_extra_ms += extra

    prefill_wall_ms, decode_wall_ms, _idle_wall_ms = _wall_segments(timelines, window)

    clean_batches = [step.step.generation_requests for step in clean_steps]
    clean_real_ms = [step.step.elapsed_ms for step in clean_steps]
    clean_sim_ms = [
        _nearest_sim_latency(batch, sim_decode_ms_by_batch)
        for batch in clean_batches
    ]
    valid_pairs = [
        (real, sim)
        for real, sim in zip(clean_real_ms, clean_sim_ms)
        if sim > 0 and not math.isnan(sim)
    ]
    if not valid_pairs:
        real_sim_ratio = math.nan
        sim_decode_mean = math.nan
        clean_decode_gap_ms = 0.0
    else:
        ratios = [real / sim for real, sim in valid_pairs]
        real_sim_ratio = _mean(ratios)
        sim_decode_mean = _mean([sim for _, sim in valid_pairs])
        clean_decode_gap_ms = decode_wall_ms * (1.0 - _safe_ratio(1.0, real_sim_ratio))

    attribution = {
        "prefill_occupancy": max(0.0, prefill_wall_ms),
        "dp_peer_stall": max(0.0, peer_stall_extra_ms),
        "decode_latency_gap": abs(clean_decode_gap_ms),
    }
    attribution_total = sum(attribution.values()) or 1.0
    dominant = max(attribution, key=attribution.get)
    if dominant == "prefill_occupancy":
        verdict = "prefill_occupancy_dominates"
        phase414_target = "audit_chunked_prefill_wall_occupancy"
    elif dominant == "dp_peer_stall":
        verdict = "dp_peer_stall_dominates"
        phase414_target = "model_two_engine_prefill_stall_coupling"
    else:
        verdict = "decode_latency_gap_dominates"
        phase414_target = "audit_decode_iteration_latency_model"

    return {
        "steady_target_penalty": steady_target_penalty,
        "steady_metric_penalty": steady_metric_penalty,
        "steady_penalty_error_pct": _error_pct(steady_metric_penalty, steady_target_penalty),
        "steady_consistency_gate": (
            "passed" if _error_pct(steady_metric_penalty, steady_target_penalty) <= 10.0 else "failed"
        ),
        "steady_window_start_ms": window.start_ms,
        "steady_window_end_ms": window.end_ms,
        "steady_window_wall_ms": window.wall_ms,
        "prefill_wall_ms": prefill_wall_ms,
        "peer_stall_extra_ms": peer_stall_extra_ms,
        "clean_decode_wall_ms": decode_wall_ms,
        "clean_decode_gap_ms": clean_decode_gap_ms,
        "prefill_attribution_share": attribution["prefill_occupancy"] / attribution_total,
        "peer_stall_attribution_share": attribution["dp_peer_stall"] / attribution_total,
        "decode_gap_attribution_share": attribution["decode_latency_gap"] / attribution_total,
        "intrinsic_decode_ms": intrinsic_decode_ms,
        "decode_only_step_count": len(decode_only_steps),
        "clean_decode_step_count": len(clean_steps),
        "peer_stalled_decode_step_count": peer_stalled_steps,
        "decode_batch_mean": _mean([step.step.generation_requests for step in decode_only_steps]),
        "decode_batch_p50": _percentile([step.step.generation_requests for step in decode_only_steps], 0.5),
        "decode_batch_p90": _percentile([step.step.generation_requests for step in decode_only_steps], 0.9),
        "real_decode_ms_mean": _mean(clean_real_ms),
        "sim_decode_ms_mean": sim_decode_mean,
        "real_sim_decode_latency_ratio": real_sim_ratio,
        "mechanism_verdict": verdict,
        "phase414_target": phase414_target,
    }


def _artifact_dir_from_phase412(row: dict[str, str], sweep_root: Path, scenario: str) -> Path:
    raw = row.get("artifact_dir", "")
    if raw:
        candidate = Path(raw)
        if candidate.exists():
            return candidate
        repo_candidate = REPO_ROOT / raw
        if repo_candidate.exists():
            return repo_candidate
    candidate = sweep_root / scenario
    if candidate.exists():
        return candidate
    raise ValueError(f"missing Phase412 artifact for {scenario}")


def build_phase413_rows(
    *,
    sweep_root: Path = SWEEP_ROOT,
    phase412_csv: Path = PHASE412_CSV,
    scenario: str = DEFAULT_SCENARIO,
    sim_decode_ms_by_batch: dict[int, float] | None = None,
) -> list[dict[str, str]]:
    phase412 = _read_csv_by_scenario(phase412_csv, row_type="trend")[scenario]
    artifact_dir = _artifact_dir_from_phase412(phase412, sweep_root, scenario)
    metrics_path = _metrics_path(artifact_dir)
    serve_log = _serve_log_path(artifact_dir)
    window = steady_window_from_metrics(metrics_path)
    steps = parse_iteration_steps(serve_log)
    steady_target_penalty = float(phase412["steady_penalty"])
    uncoupled = float(phase412["uncoupled_output_tok_s_gpu"])
    steady_metric_penalty = _safe_ratio(uncoupled, window.output_tok_s_gpu)

    if sim_decode_ms_by_batch is None:
        batches = [
            step.generation_requests
            for step in steps
            if step.ctx_tokens == 0 and step.generation_tokens > 0
        ]
        sim_decode_ms_by_batch = _default_sim_decode_ms_by_batch(batches)

    summary = summarize_steady_decompose(
        steps=steps,
        window=window,
        steady_target_penalty=steady_target_penalty,
        steady_metric_penalty=steady_metric_penalty,
        sim_decode_ms_by_batch=sim_decode_ms_by_batch,
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
        raise ValueError("Phase413 expects exactly one scenario row")
    for row in rows:
        if row.get("source") != SOURCE:
            raise ValueError("unexpected source")
        if row.get("phase405_penalty_read") != "false":
            raise ValueError("phase405_penalty_read must stay false")
        if row.get("gpu_allowed") != "false" or row.get("ssh_allowed") != "false":
            raise ValueError("Phase413 is offline and must not use GPU/SSH")
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


def write_phase413_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_phase413_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    row = rows[0]
    return "\n".join(
        [
            "# Phase413 Steady Decompose",
            "",
            "Phase413 复用 Phase412 N=512 trace，只做离线 elapsed 分解；不改 runtime、PerfDatabase 或 gate。",
            "",
            "## Verdict",
            "",
            f"- verdict: `{row['mechanism_verdict']}`",
            f"- Phase414 target: `{row['phase414_target']}`",
            f"- steady target penalty: `{row['steady_target_penalty']}`",
            f"- steady metric penalty: `{row['steady_metric_penalty']}`",
            f"- consistency gate: `{row['steady_consistency_gate']}`",
            "",
            "## Decomposition",
            "",
            "| block | ms | share |",
            "|---|---:|---:|",
            f"| prefill wall | {row['prefill_wall_ms']} | {row['prefill_attribution_share']} |",
            f"| DP peer-stall extra | {row['peer_stall_extra_ms']} | {row['peer_stall_attribution_share']} |",
            f"| clean decode latency gap | {row['clean_decode_gap_ms']} | {row['decode_gap_attribution_share']} |",
            "",
            "## Decode Audit",
            "",
            "| clean decode steps | intrinsic ms | real mean ms | sim mean ms | real/sim | batch mean | batch p90 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            f"| {row['clean_decode_step_count']} | {row['intrinsic_decode_ms']} | {row['real_decode_ms_mean']} | {row['sim_decode_ms_mean']} | {row['real_sim_decode_latency_ratio']} | {row['decode_batch_mean']} | {row['decode_batch_p90']} |",
            "",
            "## Interpretation",
            "",
            "- steady penalty 先从 raw metrics 重新计算，再和 Phase412 CSV 交叉核对。",
            "- prefill wall 是同一墙钟窗口内任一 DP engine 正在 prefill 的时间。",
            "- peer-stall extra 是 decode-only step 和 peer prefill 重叠时，相对同 batch clean decode median 多出来的 elapsed。",
            "- clean decode latency gap 用 clean decode-only step 对 cb_sim pure-decode 模型逐 batch 对账。",
            "",
            "## Boundary",
            "",
            "- GPU/SSH: not used in Phase413.",
            "- Runtime/PerfDatabase/gate: not modified.",
            "- Phase405 penalty: not read.",
            "- Default AIC: No-Go.",
            "",
        ]
    )


def write_phase413_md(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase413_md(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", type=Path, default=SWEEP_ROOT)
    parser.add_argument("--phase412-csv", type=Path, default=PHASE412_CSV)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_phase413_rows(
        sweep_root=args.sweep_root,
        phase412_csv=args.phase412_csv,
        scenario=args.scenario,
    )
    write_phase413_csv(args.csv, rows)
    write_phase413_md(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
