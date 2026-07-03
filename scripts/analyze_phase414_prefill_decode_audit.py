#!/usr/bin/env python3
"""Phase414: audit prefill mixed-step charge and decode KV-growth gap."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Callable
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import analyze_phase413_steady_decompose as phase413  # noqa: E402


SOURCE = "phase414_prefill_decode_audit"
DEFAULT_SCENARIO = "K2.5-tp4ep8dp2-32k3k"
SWEEP_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase412_arrival_sweep"
PHASE413_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase413_steady_decompose.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase414_prefill_decode_audit.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase414_prefill_decode_audit.md"
DEFAULT_READINESS = "No-Go"
FULL_CHUNK_TOKENS = 32000
FULL_CHUNK_MIN_TOKENS = 30000
DECODE_KV_BASE = 32000
DECODE_KV_SPAN = 3000
DECODE_BINS = 5

MixedMsFunc = Callable[[int, int, int], float]
DecodeMsFunc = Callable[[int, int], float]

CSV_FIELDS = [
    "source",
    "scenario",
    "artifact_dir",
    "steady_target_penalty",
    "prefill_step_count",
    "prefill_real_mean_ms",
    "prefill_real_p50_ms",
    "prefill_gen_reqs_mean",
    "prefill_sim_mean_ms",
    "prefill_real_sim_ratio",
    "prefill_gap_ms",
    "decode_bin_count",
    "decode_real_first_ms",
    "decode_real_last_ms",
    "decode_real_slope_ms",
    "decode_sim_first_ms",
    "decode_sim_last_ms",
    "decode_sim_slope_ms",
    "decode_slope_ratio",
    "decode_gap_ms",
    "target_missing_ms",
    "reconstructed_missing_ms",
    "reconstructed_penalty",
    "reconstruction_error_pct",
    "consistency_gate",
    "mechanism_verdict",
    "phase415_target",
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


def _read_csv_by_scenario(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        return {row["scenario"]: row for row in csv.DictReader(f)}


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _artifact_dir_from_phase413(row: dict[str, str], sweep_root: Path, scenario: str) -> Path:
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


def _default_latency_funcs() -> tuple[MixedMsFunc, DecodeMsFunc]:
    from aiconfigurator.sdk.backends.cb_simulator.iteration_latency import (  # noqa: PLC0415
        IterationLatencyCalculator,
    )
    from scripts import validate_cb_simulator as validate_cb  # noqa: PLC0415

    model, db, backend = validate_cb._load_model_and_db(tp=4, dp=2, moe_tp=1, moe_ep=8)
    calc = IterationLatencyCalculator(
        backend,
        model,
        db,
        overlap_factor=0.0,
        per_iteration_overhead_ms=0.0,
    )
    mixed_cache: dict[tuple[int, int, int], float] = {}
    decode_cache: dict[tuple[int, int], float] = {}

    def mixed(prefill_tokens: int, gen_reqs: int, kv_len: int) -> float:
        key = (prefill_tokens, gen_reqs, kv_len)
        if key not in mixed_cache:
            mixed_cache[key] = calc.compute(
                prefill_tokens=prefill_tokens,
                prefill_batch_size=max(1, math.ceil(prefill_tokens / FULL_CHUNK_TOKENS)),
                prefill_seq_len=FULL_CHUNK_TOKENS,
                decode_batch_size=gen_reqs,
                decode_avg_kv_len=kv_len,
            )
        return mixed_cache[key]

    def decode(batch: int, kv_len: int) -> float:
        key = (batch, kv_len)
        if key not in decode_cache:
            decode_cache[key] = calc.compute(
                prefill_tokens=0,
                prefill_batch_size=0,
                prefill_seq_len=1,
                decode_batch_size=batch,
                decode_avg_kv_len=kv_len,
            )
        return decode_cache[key]

    return mixed, decode


def _full_prefill_steps(
    timelines: dict[str, list[phase413.TimedStep]],
    window: phase413.SteadyWindow,
) -> list[phase413.TimedStep]:
    out: list[phase413.TimedStep] = []
    for timeline in timelines.values():
        for step in timeline:
            if phase413._step_overlaps_window(step, window) <= 0:
                continue
            if step.step.ctx_tokens >= FULL_CHUNK_MIN_TOKENS:
                out.append(step)
    return out


def _clean_decode_steps(
    timelines: dict[str, list[phase413.TimedStep]],
    window: phase413.SteadyWindow,
) -> list[phase413.TimedStep]:
    return phase413._clean_decode_steps(timelines, window)


def _kv_for_index(index: int, total: int) -> int:
    if total <= 1:
        return DECODE_KV_BASE
    progress = index / (total - 1)
    return DECODE_KV_BASE + int(round(progress * DECODE_KV_SPAN))


def _bin_index(index: int, total: int) -> int:
    if total <= 1:
        return 0
    return min(DECODE_BINS - 1, int(index * DECODE_BINS / total))


def summarize_prefill_decode_audit(
    *,
    steps: list[phase413.phase409.IterationStep],
    window: phase413.SteadyWindow,
    phase413_row: dict[str, str],
    sim_mixed_ms_func: MixedMsFunc,
    sim_decode_ms_func: DecodeMsFunc,
) -> dict[str, float | int | str]:
    timelines = phase413.build_wallclock_timelines(steps)
    prefill_steps = _full_prefill_steps(timelines, window)
    if not prefill_steps:
        raise ValueError("Phase414 requires full chunk prefill steps")

    prefill_real = [step.step.elapsed_ms for step in prefill_steps]
    prefill_sim = [
        sim_mixed_ms_func(step.step.ctx_tokens, step.step.generation_requests, DECODE_KV_BASE)
        for step in prefill_steps
    ]
    prefill_gap = sum(max(0.0, real - sim) for real, sim in zip(prefill_real, prefill_sim))

    clean_decode = _clean_decode_steps(timelines, window)
    if not clean_decode:
        raise ValueError("Phase414 requires clean decode steps")
    bins: dict[int, list[tuple[int, int, float, float]]] = {idx: [] for idx in range(DECODE_BINS)}
    total_decode = len(clean_decode)
    for idx, step in enumerate(clean_decode):
        kv_len = _kv_for_index(idx, total_decode)
        sim_ms = sim_decode_ms_func(step.step.generation_requests, kv_len)
        bins[_bin_index(idx, total_decode)].append(
            (step.step.generation_requests, kv_len, step.step.elapsed_ms, sim_ms)
        )
    nonempty_bins = [items for _, items in sorted(bins.items()) if items]
    real_by_bin = [_mean([item[2] for item in items]) for items in nonempty_bins]
    sim_by_bin = [_mean([item[3] for item in items]) for items in nonempty_bins]
    decode_gap = sum(max(0.0, real - sim) for real, sim in zip(real_by_bin, sim_by_bin))

    target_missing = (
        float(phase413_row["prefill_wall_ms"])
        + float(phase413_row["clean_decode_gap_ms"])
        + float(phase413_row.get("peer_stall_extra_ms") or 0.0)
    )
    reconstructed_missing = prefill_gap + decode_gap
    steady_target_penalty = float(phase413_row["steady_target_penalty"])
    reconstructed_penalty = 1.0 + (steady_target_penalty - 1.0) * _safe_ratio(
        reconstructed_missing,
        target_missing,
    )
    reconstruction_error = _error_pct(reconstructed_penalty, steady_target_penalty)
    consistency_gate = "passed" if reconstruction_error <= 10.0 else "failed"

    prefill_real_mean = _mean(prefill_real)
    prefill_sim_mean = _mean(prefill_sim)
    prefill_ratio = _safe_ratio(prefill_real_mean, prefill_sim_mean)
    decode_real_slope = real_by_bin[-1] - real_by_bin[0]
    decode_sim_slope = sim_by_bin[-1] - sim_by_bin[0]
    decode_slope_ratio = _safe_ratio(decode_real_slope, decode_sim_slope)

    if prefill_gap >= decode_gap:
        verdict = "prefill_charge_gap_dominates"
        phase415_target = "audit_mixed_prefill_charge_components"
    else:
        verdict = "decode_kv_growth_gap_dominates"
        phase415_target = "audit_mla_decode_kv_growth_curve"

    return {
        "steady_target_penalty": steady_target_penalty,
        "prefill_step_count": len(prefill_steps),
        "prefill_real_mean_ms": prefill_real_mean,
        "prefill_real_p50_ms": _percentile(prefill_real, 0.5),
        "prefill_gen_reqs_mean": _mean([step.step.generation_requests for step in prefill_steps]),
        "prefill_sim_mean_ms": prefill_sim_mean,
        "prefill_real_sim_ratio": prefill_ratio,
        "prefill_gap_ms": prefill_gap,
        "decode_bin_count": len(nonempty_bins),
        "decode_real_first_ms": real_by_bin[0],
        "decode_real_last_ms": real_by_bin[-1],
        "decode_real_slope_ms": decode_real_slope,
        "decode_sim_first_ms": sim_by_bin[0],
        "decode_sim_last_ms": sim_by_bin[-1],
        "decode_sim_slope_ms": decode_sim_slope,
        "decode_slope_ratio": decode_slope_ratio,
        "decode_gap_ms": decode_gap,
        "target_missing_ms": target_missing,
        "reconstructed_missing_ms": reconstructed_missing,
        "reconstructed_penalty": reconstructed_penalty,
        "reconstruction_error_pct": reconstruction_error,
        "consistency_gate": consistency_gate,
        "mechanism_verdict": verdict,
        "phase415_target": phase415_target,
    }


def build_phase414_rows(
    *,
    sweep_root: Path = SWEEP_ROOT,
    phase413_csv: Path = PHASE413_CSV,
    scenario: str = DEFAULT_SCENARIO,
    sim_mixed_ms_func: MixedMsFunc | None = None,
    sim_decode_ms_func: DecodeMsFunc | None = None,
) -> list[dict[str, str]]:
    phase413_row = _read_csv_by_scenario(phase413_csv)[scenario]
    artifact_dir = _artifact_dir_from_phase413(phase413_row, sweep_root, scenario)
    window = phase413.steady_window_from_metrics(phase413._metrics_path(artifact_dir))
    steps = phase413.parse_iteration_steps(phase413._serve_log_path(artifact_dir))
    if sim_mixed_ms_func is None or sim_decode_ms_func is None:
        default_mixed, default_decode = _default_latency_funcs()
        sim_mixed_ms_func = sim_mixed_ms_func or default_mixed
        sim_decode_ms_func = sim_decode_ms_func or default_decode
    summary = summarize_prefill_decode_audit(
        steps=steps,
        window=window,
        phase413_row=phase413_row,
        sim_mixed_ms_func=sim_mixed_ms_func,
        sim_decode_ms_func=sim_decode_ms_func,
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
        raise ValueError("Phase414 expects exactly one scenario row")
    for row in rows:
        if row.get("source") != SOURCE:
            raise ValueError("unexpected source")
        if row.get("phase405_penalty_read") != "false":
            raise ValueError("phase405_penalty_read must stay false")
        if row.get("gpu_allowed") != "false" or row.get("ssh_allowed") != "false":
            raise ValueError("Phase414 is offline and must not use GPU/SSH")
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


def write_phase414_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_phase414_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    row = rows[0]
    return "\n".join(
        [
            "# Phase414 Prefill Decode Audit",
            "",
            "Phase414 复用 Phase412 N=512 trace 与 Phase413 steady 窗口，只做离线计费审计；不改 runtime、PerfDatabase 或 gate。",
            "",
            "## Verdict",
            "",
            f"- verdict: `{row['mechanism_verdict']}`",
            f"- Phase415 target: `{row['phase415_target']}`",
            f"- consistency gate: `{row['consistency_gate']}`",
            f"- reconstructed penalty: `{row['reconstructed_penalty']}` vs target `{row['steady_target_penalty']}`",
            "",
            "## Audit A: Mixed Prefill",
            "",
            "| prefill steps | real mean ms | sim mean ms | real/sim | gap ms | gen reqs mean |",
            "|---:|---:|---:|---:|---:|---:|",
            f"| {row['prefill_step_count']} | {row['prefill_real_mean_ms']} | {row['prefill_sim_mean_ms']} | {row['prefill_real_sim_ratio']} | {row['prefill_gap_ms']} | {row['prefill_gen_reqs_mean']} |",
            "",
            "## Audit B: Decode KV Growth",
            "",
            "| bins | real first ms | real last ms | sim first ms | sim last ms | real slope | sim slope | slope ratio | gap ms |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| {row['decode_bin_count']} | {row['decode_real_first_ms']} | {row['decode_real_last_ms']} | {row['decode_sim_first_ms']} | {row['decode_sim_last_ms']} | {row['decode_real_slope_ms']} | {row['decode_sim_slope_ms']} | {row['decode_slope_ratio']} | {row['decode_gap_ms']} |",
            "",
            "## Reconstruction",
            "",
            "| target missing ms | reconstructed missing ms | error pct |",
            "|---:|---:|---:|",
            f"| {row['target_missing_ms']} | {row['reconstructed_missing_ms']} | {row['reconstruction_error_pct']} |",
            "",
            "## Boundary",
            "",
            "- GPU/SSH: not used in Phase414.",
            "- Runtime/PerfDatabase/gate: not modified.",
            "- Phase405 penalty: not read.",
            "- Default AIC: No-Go.",
            "",
        ]
    )


def write_phase414_md(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase414_md(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", type=Path, default=SWEEP_ROOT)
    parser.add_argument("--phase413-csv", type=Path, default=PHASE413_CSV)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_phase414_rows(
        sweep_root=args.sweep_root,
        phase413_csv=args.phase413_csv,
        scenario=args.scenario,
    )
    write_phase414_csv(args.csv, rows)
    write_phase414_md(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
