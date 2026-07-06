#!/usr/bin/env python3
"""Phase426: decompose DP2 8k2k sustained-arrival steady gap."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import analyze_phase413_steady_decompose as phase413  # noqa: E402


SOURCE = "phase426_8k2k_steady_decompose"
DEFAULT_SCENARIO = "K2.5-tp4ep8dp2-8k2k"
SWEEP_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase425_8k2k_sweep"
PHASE425_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase425_8k2k_sweep.csv"
PHASE413_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase413_steady_decompose.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase426_8k2k_steady_decompose.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase426_8k2k_steady_decompose.md"
DEFAULT_READINESS = "No-Go"

DECODE_AVG_KV_LEN = 8000
MOE_LAYER_COUNT = 60
EP_DIRECTIONS = 2
EP_FLOOR_US_MIN = 50.0
EP_FLOOR_US_MAX = 70.0
EP_SOURCE_NOTE = "vllm_0.19_allgather_reducescatter_AgRsAll2AllManager"

CSV_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "artifact_dir",
    "decode_avg_kv_len",
    "steady_target_penalty",
    "steady_metric_penalty",
    "steady_penalty_error_pct",
    "steady_consistency_gate",
    "validation_steady_penalty",
    "validation_baseline_penalty",
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
    "decode_gap_ms_mean",
    "decode_gap_ms_min",
    "decode_gap_ms_max",
    "decode_gap_ms_span",
    "decode_gap_ms_slope_per_request",
    "decode_gap_shape",
    "ep_floor_us_min",
    "ep_floor_us_max",
    "ep_floor_ms_min",
    "ep_floor_ms_max",
    "ep_floor_covers_decode_gap",
    "ep_floor_source",
    "after_decode_fix_remaining_share",
    "after_prefill_fix_remaining_share",
    "after_prefill_decode_fix_remaining_share",
    "phase413_32k_steady_penalty",
    "phase413_32k_prefill_share",
    "phase413_32k_peer_stall_share",
    "phase413_32k_decode_gap_share",
    "phase413_32k_verdict",
    "decode_batch",
    "decode_batch_step_count",
    "decode_batch_real_ms_mean",
    "decode_batch_sim_ms",
    "decode_batch_gap_ms",
    "mechanism_verdict",
    "phase427_target",
    "phase405_penalty_read",
    "gpu_allowed",
    "ssh_allowed",
    "runtime_modified",
    "perf_database",
    "valid_for_default",
    "diagnostic_only",
    "default_readiness",
]


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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _read_trend_row(path: Path, scenario: str) -> dict[str, str]:
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("row_type") == "trend" and row.get("scenario") == scenario:
                return row
    raise ValueError(f"missing Phase425 trend row for {scenario}")


def _read_phase413_reference(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("scenario") == "K2.5-tp4ep8dp2-32k3k":
                return row
    return {}


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _artifact_dir_from_phase425(row: dict[str, str], sweep_root: Path, scenario: str) -> Path:
    raw = row.get("artifact_dir", "")
    candidates: list[Path] = []
    if raw:
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            candidates.append(Path(item))
            candidates.append(REPO_ROOT / item)
    candidates.append(sweep_root / scenario)
    for candidate in reversed(candidates):
        if candidate.exists():
            return candidate
    raise ValueError(f"missing Phase425 artifact for {scenario}")


def _nearest_sim_latency(batch: int, sim_decode_ms_by_batch: dict[int, float]) -> float:
    return phase413._nearest_sim_latency(batch, sim_decode_ms_by_batch)  # noqa: SLF001


def _clean_decode_steps(
    steps: list[phase413.phase409.IterationStep],
    window: phase413.SteadyWindow,
) -> list[phase413.TimedStep]:
    timelines = phase413.build_wallclock_timelines(steps)
    return phase413._clean_decode_steps(timelines, window)  # noqa: SLF001


def _decode_batch_rows(
    *,
    clean_steps: list[phase413.TimedStep],
    sim_decode_ms_by_batch: dict[int, float],
) -> list[dict[str, float | int]]:
    grouped: dict[int, list[float]] = {}
    for step in clean_steps:
        grouped.setdefault(step.step.generation_requests, []).append(step.step.elapsed_ms)
    rows: list[dict[str, float | int]] = []
    for batch in sorted(grouped):
        real_mean = _mean(grouped[batch])
        sim_ms = _nearest_sim_latency(batch, sim_decode_ms_by_batch)
        rows.append(
            {
                "decode_batch": batch,
                "decode_batch_step_count": len(grouped[batch]),
                "decode_batch_real_ms_mean": real_mean,
                "decode_batch_sim_ms": sim_ms,
                "decode_batch_gap_ms": real_mean - sim_ms,
            }
        )
    return rows


def _weighted_mean(values: list[tuple[float, float]]) -> float:
    weight_sum = sum(weight for _value, weight in values)
    if weight_sum <= 0:
        return math.nan
    return sum(value * weight for value, weight in values) / weight_sum


def _slope(rows: list[dict[str, float | int]]) -> float:
    xs = [float(row["decode_batch"]) for row in rows]
    ys = [float(row["decode_batch_gap_ms"]) for row in rows]
    weights = [float(row["decode_batch_step_count"]) for row in rows]
    if len(xs) < 2:
        return 0.0
    x_mean = _weighted_mean(list(zip(xs, weights)))
    y_mean = _weighted_mean(list(zip(ys, weights)))
    denom = sum(weight * (x - x_mean) ** 2 for x, weight in zip(xs, weights))
    if denom <= 0:
        return 0.0
    return (
        sum(weight * (x - x_mean) * (y - y_mean) for x, y, weight in zip(xs, ys, weights))
        / denom
    )


def _decode_gap_shape(rows: list[dict[str, float | int]]) -> tuple[str, dict[str, float]]:
    gaps = [float(row["decode_batch_gap_ms"]) for row in rows]
    weights = [float(row["decode_batch_step_count"]) for row in rows]
    if not gaps:
        return "no_clean_decode", {
            "mean": math.nan,
            "min": math.nan,
            "max": math.nan,
            "span": math.nan,
            "slope": math.nan,
        }
    mean_gap = _weighted_mean(list(zip(gaps, weights)))
    min_gap = min(gaps)
    max_gap = max(gaps)
    span = max_gap - min_gap
    slope = _slope(rows)
    batch_range = (
        max(float(row["decode_batch"]) for row in rows)
        - min(float(row["decode_batch"]) for row in rows)
        if len(rows) > 1
        else 0.0
    )
    slope_span = abs(slope) * batch_range
    fixed_threshold = max(1.0, abs(mean_gap) * 0.25)
    if span <= fixed_threshold:
        shape = "fixed_latency_floor"
    elif slope_span >= max(1.0, abs(mean_gap) * 0.50):
        shape = "batch_scaled_gap"
    else:
        shape = "mixed_fixed_and_batch_gap"
    return shape, {
        "mean": mean_gap,
        "min": min_gap,
        "max": max_gap,
        "span": span,
        "slope": slope,
    }


def _ep_floor_bounds_ms() -> tuple[float, float]:
    calls = MOE_LAYER_COUNT * EP_DIRECTIONS
    return calls * EP_FLOOR_US_MIN / 1000.0, calls * EP_FLOOR_US_MAX / 1000.0


def _verdict(
    *,
    prefill_share: float,
    peer_share: float,
    decode_share: float,
    decode_shape: str,
    ep_covers: bool,
) -> tuple[str, str]:
    if decode_share >= 0.50 and decode_shape == "fixed_latency_floor" and ep_covers:
        return (
            "decode_fixed_gap_ep_latency_floor",
            "add_decode_ep_a2a_latency_floor_or_profile_if_needed",
        )
    if prefill_share >= 0.50 and decode_share >= 0.25:
        return (
            "prefill_dominant_with_decode_gap_secondary",
            "audit_prefill_occupancy_then_decode_gap",
        )
    if prefill_share >= max(peer_share, decode_share):
        return (
            "prefill_occupancy_dominates",
            "audit_prefill_occupancy_model",
        )
    if decode_share >= max(prefill_share, peer_share):
        if decode_shape == "fixed_latency_floor" and ep_covers:
            return (
                "decode_fixed_gap_ep_latency_floor",
                "add_decode_ep_a2a_latency_floor_or_profile_if_needed",
            )
        if decode_shape == "fixed_latency_floor":
            return (
                "decode_fixed_gap_exceeds_ep_floor",
                "profile_or_model_decode_fixed_gap",
            )
        return (
            "decode_batch_scaled_gap",
            "audit_decode_batch_slope_model",
        )
    return (
        "dp_peer_stall_dominates",
        "model_dp_peer_stall",
    )


def _base_flags() -> dict[str, object]:
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


def build_phase426_rows(
    *,
    sweep_root: Path = SWEEP_ROOT,
    phase425_csv: Path = PHASE425_CSV,
    phase413_csv: Path = PHASE413_CSV,
    scenario: str = DEFAULT_SCENARIO,
    sim_decode_ms_by_batch: dict[int, float] | None = None,
) -> list[dict[str, str]]:
    phase425 = _read_trend_row(phase425_csv, scenario)
    phase413_ref = _read_phase413_reference(phase413_csv)
    artifact_dir = _artifact_dir_from_phase425(phase425, sweep_root, scenario)
    metrics_path = phase413._metrics_path(artifact_dir)  # noqa: SLF001
    serve_log = phase413._serve_log_path(artifact_dir)  # noqa: SLF001
    window = phase413.steady_window_from_metrics(metrics_path)
    steps = phase413.parse_iteration_steps(serve_log)

    if sim_decode_ms_by_batch is None:
        batches = [
            step.generation_requests
            for step in steps
            if step.ctx_tokens == 0 and step.generation_tokens > 0
        ]
        sim_decode_ms_by_batch = phase413._default_sim_decode_ms_by_batch(  # noqa: SLF001
            batches,
            decode_avg_kv_len=DECODE_AVG_KV_LEN,
        )

    steady_target = float(phase425["steady_penalty"])
    uncoupled = float(phase425["uncoupled_output_tok_s_gpu"])
    steady_metric = _safe_ratio(uncoupled, window.output_tok_s_gpu)
    summary = phase413.summarize_steady_decompose(
        steps=steps,
        window=window,
        steady_target_penalty=steady_target,
        steady_metric_penalty=steady_metric,
        sim_decode_ms_by_batch=sim_decode_ms_by_batch,
    )
    clean_steps = _clean_decode_steps(steps, window)
    batch_rows_raw = _decode_batch_rows(
        clean_steps=clean_steps,
        sim_decode_ms_by_batch=sim_decode_ms_by_batch,
    )
    decode_shape, gap_stats = _decode_gap_shape(batch_rows_raw)
    floor_min, floor_max = _ep_floor_bounds_ms()
    ep_covers = floor_min <= gap_stats["mean"] <= floor_max
    prefill_share = float(summary["prefill_attribution_share"])
    peer_share = float(summary["peer_stall_attribution_share"])
    decode_share = float(summary["decode_gap_attribution_share"])
    mechanism, target = _verdict(
        prefill_share=prefill_share,
        peer_share=peer_share,
        decode_share=decode_share,
        decode_shape=decode_shape,
        ep_covers=ep_covers,
    )

    common = {
        "source": SOURCE,
        "scenario": scenario,
        "artifact_dir": _display_path(artifact_dir),
        "decode_avg_kv_len": DECODE_AVG_KV_LEN,
        "steady_target_penalty": summary["steady_target_penalty"],
        "steady_metric_penalty": summary["steady_metric_penalty"],
        "steady_penalty_error_pct": summary["steady_penalty_error_pct"],
        "steady_consistency_gate": summary["steady_consistency_gate"],
        "validation_steady_penalty": phase425.get("validation_last_steady_penalty", ""),
        "validation_baseline_penalty": phase425.get("validation_baseline_penalty", ""),
        "steady_window_start_ms": summary["steady_window_start_ms"],
        "steady_window_end_ms": summary["steady_window_end_ms"],
        "steady_window_wall_ms": summary["steady_window_wall_ms"],
        "prefill_wall_ms": summary["prefill_wall_ms"],
        "peer_stall_extra_ms": summary["peer_stall_extra_ms"],
        "clean_decode_wall_ms": summary["clean_decode_wall_ms"],
        "clean_decode_gap_ms": summary["clean_decode_gap_ms"],
        "prefill_attribution_share": summary["prefill_attribution_share"],
        "peer_stall_attribution_share": summary["peer_stall_attribution_share"],
        "decode_gap_attribution_share": summary["decode_gap_attribution_share"],
        "intrinsic_decode_ms": summary["intrinsic_decode_ms"],
        "decode_only_step_count": summary["decode_only_step_count"],
        "clean_decode_step_count": summary["clean_decode_step_count"],
        "peer_stalled_decode_step_count": summary["peer_stalled_decode_step_count"],
        "decode_batch_mean": summary["decode_batch_mean"],
        "decode_batch_p50": summary["decode_batch_p50"],
        "decode_batch_p90": summary["decode_batch_p90"],
        "real_decode_ms_mean": summary["real_decode_ms_mean"],
        "sim_decode_ms_mean": summary["sim_decode_ms_mean"],
        "real_sim_decode_latency_ratio": summary["real_sim_decode_latency_ratio"],
        "decode_gap_ms_mean": gap_stats["mean"],
        "decode_gap_ms_min": gap_stats["min"],
        "decode_gap_ms_max": gap_stats["max"],
        "decode_gap_ms_span": gap_stats["span"],
        "decode_gap_ms_slope_per_request": gap_stats["slope"],
        "decode_gap_shape": decode_shape,
        "ep_floor_us_min": EP_FLOOR_US_MIN,
        "ep_floor_us_max": EP_FLOOR_US_MAX,
        "ep_floor_ms_min": floor_min,
        "ep_floor_ms_max": floor_max,
        "ep_floor_covers_decode_gap": ep_covers,
        "ep_floor_source": EP_SOURCE_NOTE,
        "after_decode_fix_remaining_share": prefill_share + peer_share,
        "after_prefill_fix_remaining_share": peer_share + decode_share,
        "after_prefill_decode_fix_remaining_share": peer_share,
        "phase413_32k_steady_penalty": phase413_ref.get("steady_target_penalty", ""),
        "phase413_32k_prefill_share": phase413_ref.get("prefill_attribution_share", ""),
        "phase413_32k_peer_stall_share": phase413_ref.get("peer_stall_attribution_share", ""),
        "phase413_32k_decode_gap_share": phase413_ref.get("decode_gap_attribution_share", ""),
        "phase413_32k_verdict": phase413_ref.get("mechanism_verdict", ""),
        "mechanism_verdict": mechanism,
        "phase427_target": target,
        **_base_flags(),
    }
    rows: list[dict[str, object]] = [{"row_type": "summary", **common}]
    for batch_row in batch_rows_raw:
        rows.append(
            {
                "row_type": "decode_batch",
                **common,
                **batch_row,
            }
        )
    return [{field: _fmt(row.get(field, "")) for field in CSV_FIELDS} for row in rows]


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Phase426 rows must not be empty")
    summaries = [row for row in rows if row.get("row_type") == "summary"]
    if len(summaries) != 1:
        raise ValueError("Phase426 expects exactly one summary row")
    for row in rows:
        if row.get("source") != SOURCE:
            raise ValueError("unexpected source")
        if row.get("phase405_penalty_read") != "false":
            raise ValueError("phase405_penalty_read must stay false")
        if row.get("gpu_allowed") != "false" or row.get("ssh_allowed") != "false":
            raise ValueError("Phase426 is offline and must not use GPU/SSH")
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


def write_phase426_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_phase426_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    summary = [row for row in rows if row["row_type"] == "summary"][0]
    batch_rows = [row for row in rows if row["row_type"] == "decode_batch"]
    lines = [
        "# Phase426 8k2k Steady Decompose",
        "",
        "Phase426 复用 Phase425 N=512 trace，只做离线 steady-window 分解；不改 runtime、PerfDatabase 或 gate。",
        "",
        "## Verdict",
        "",
        f"- verdict: `{summary['mechanism_verdict']}`",
        f"- Phase427 target: `{summary['phase427_target']}`",
        f"- trace steady penalty: `{summary['steady_metric_penalty']}`",
        f"- validation steady penalty: `{summary['validation_steady_penalty']}`",
        f"- consistency gate: `{summary['steady_consistency_gate']}`",
        "",
        "## Decomposition",
        "",
        "| block | ms | share |",
        "|---|---:|---:|",
        f"| prefill wall | {summary['prefill_wall_ms']} | {summary['prefill_attribution_share']} |",
        f"| DP peer-stall extra | {summary['peer_stall_extra_ms']} | {summary['peer_stall_attribution_share']} |",
        f"| clean decode latency gap | {summary['clean_decode_gap_ms']} | {summary['decode_gap_attribution_share']} |",
        "",
        "## Decode Audit",
        "",
        "| shape | gap mean ms | gap min | gap max | slope ms/request | real mean | sim mean | real/sim |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| {summary['decode_gap_shape']} | {summary['decode_gap_ms_mean']} | {summary['decode_gap_ms_min']} | "
        f"{summary['decode_gap_ms_max']} | {summary['decode_gap_ms_slope_per_request']} | "
        f"{summary['real_decode_ms_mean']} | {summary['sim_decode_ms_mean']} | {summary['real_sim_decode_latency_ratio']} |",
        "",
        "## EP Latency Floor Estimate",
        "",
        "| layers | directions | us/call range | ms/iter range | covers mean decode gap | source |",
        "|---:|---:|---:|---:|---|---|",
        f"| {MOE_LAYER_COUNT} | {EP_DIRECTIONS} | {summary['ep_floor_us_min']} - {summary['ep_floor_us_max']} | "
        f"{summary['ep_floor_ms_min']} - {summary['ep_floor_ms_max']} | {summary['ep_floor_covers_decode_gap']} | "
        f"{summary['ep_floor_source']} |",
        "",
        "## Expected Residual Direction",
        "",
        "| hypothetical fix | remaining attribution share |",
        "|---|---:|",
        f"| fix decode gap only | {summary['after_decode_fix_remaining_share']} |",
        f"| fix prefill wall only | {summary['after_prefill_fix_remaining_share']} |",
        f"| fix prefill + decode | {summary['after_prefill_decode_fix_remaining_share']} |",
        "",
        "## Batch Curve",
        "",
        "| decode batch | steps | real ms | sim ms | gap ms |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in batch_rows:
        lines.append(
            f"| {row['decode_batch']} | {row['decode_batch_step_count']} | "
            f"{row['decode_batch_real_ms_mean']} | {row['decode_batch_sim_ms']} | "
            f"{row['decode_batch_gap_ms']} |"
        )
    lines.extend(
        [
            "",
            "## 32k3k Reference",
            "",
            "| source | steady penalty | prefill share | peer-stall share | decode gap share | verdict |",
            "|---|---:|---:|---:|---:|---|",
            f"| Phase413 32k3k | {summary['phase413_32k_steady_penalty']} | "
            f"{summary['phase413_32k_prefill_share']} | {summary['phase413_32k_peer_stall_share']} | "
            f"{summary['phase413_32k_decode_gap_share']} | {summary['phase413_32k_verdict']} |",
            "",
            "## Interpretation",
            "",
            "- trace steady penalty 用 Phase425 N=512 raw metrics 重新计算，并与 Phase425 CSV 交叉核对。",
            "- validation steady penalty 只作为现行验收口径参考；它和 trace 口径不同，不混入模型分解。",
            "- EP 延迟底估算按 vLLM 0.19.0 默认 allgather/reducescatter EP a2a 路径，60 层、dispatch/combine 两方向、每次 50-70us。",
            "- 本报告只做 attribution；若离线估算不能覆盖 decode gap，Phase427 需要先 profiler 或继续拆 fixed gap。",
            "",
            "## Boundary",
            "",
            "- GPU/SSH: not used in Phase426.",
            "- Runtime/PerfDatabase/gate: not modified.",
            "- Phase405 penalty: not read.",
            "- Default AIC: No-Go.",
            "",
        ]
    )
    return "\n".join(lines)


def write_phase426_md(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase426_md(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", type=Path, default=SWEEP_ROOT)
    parser.add_argument("--phase425-csv", type=Path, default=PHASE425_CSV)
    parser.add_argument("--phase413-csv", type=Path, default=PHASE413_CSV)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_phase426_rows(
        sweep_root=args.sweep_root,
        phase425_csv=args.phase425_csv,
        phase413_csv=args.phase413_csv,
        scenario=args.scenario,
    )
    write_phase426_csv(args.csv, rows)
    write_phase426_md(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
