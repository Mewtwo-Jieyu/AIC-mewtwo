#!/usr/bin/env python3
"""Phase429: guaranteed reprofile attribution from Phase429 kernel traces."""

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

from scripts import analyze_phase428_perstep_kernel_attrib as phase428  # noqa: E402

SOURCE = "phase429_reprofile"
DEFAULT_SCENARIO = "K2.5-tp4ep8dp2-8k2k"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase429_reprofile"
PHASE426_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase426_8k2k_steady_decompose.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase429_reprofile.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase429_reprofile.md"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"
WINDOWS = ("w0_prefill", "w1_decode_c16", "w2_decode_c64", "w3_decode_c128")
DECODE_WINDOWS = ("w1_decode_c16", "w2_decode_c64", "w3_decode_c128")
WINDOW_TARGET_CONCURRENCY = {"w1_decode_c16": 16, "w2_decode_c64": 64, "w3_decode_c128": 128}
MIN_DECODE_BATCH_SPAN = 30
ITERATION_DETAIL_RE = re.compile(
    r"Iteration\(\d+\): "
    r"(?P<context_requests>\d+) context requests, "
    r"(?P<context_tokens>\d+) context tokens, "
    r"(?P<generation_requests>\d+) generation requests, "
    r"(?P<generation_tokens>\d+) generation tokens"
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
    "window_target_concurrency",
    "window_hit_gate",
    "retry_count",
    "category",
    "real_wall_ms_per_rank",
    "real_cuda_ms_per_rank",
    "bubble_ms_per_rank",
    "bubble_share",
    "real_slope_ms_per_request",
    "phase426_steady_metric_penalty",
    "phase426_prefill_share",
    "phase426_decode_gap_share",
    "phase426_peer_stall_share",
    "phase429_reconstruction_error_pct",
    "phase429_reconstruction_gate",
    "prefill_step_count",
    "decode_step_count",
    "decode_batch_min",
    "decode_batch_max",
    "decode_batch_span",
    "prefill_hit_gate",
    "decode_batch_span_gate",
    "prefill_verdict",
    "decode_verdict",
    "mechanism_verdict",
    "phase430_runtime_target",
    "phase430_perfdb_target",
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
class WindowCheck:
    window: str
    target_concurrency: int | None
    prefill_step_count: int
    decode_step_count: int
    decode_batch_mean: float
    decode_batch_min: int
    decode_batch_max: int
    passed: bool
    reason: str


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


def _trace_paths(artifact_dir: Path, window: str) -> list[Path]:
    paths = sorted(
        path
        for path in (artifact_dir / f"prof_{window}").glob("*.pt.trace.json.gz")
        if path.name.startswith("dp")
    )
    if not paths:
        raise ValueError(f"missing chrome trace files for {window} under {artifact_dir}")
    return paths


def _read_phase426_summary(path: Path, scenario: str) -> dict[str, str]:
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("row_type") == "summary" and row.get("scenario") == scenario:
                return row
    raise ValueError(f"missing Phase426 summary for {scenario}")


def _read_window_attempts(artifact_dir: Path) -> dict[str, dict[str, str]]:
    path = artifact_dir / "window_checks.jsonl"
    attempts: dict[str, dict[str, str]] = {}
    if not path.exists():
        return attempts
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        window = str(item.get("window", ""))
        if not window:
            continue
        attempts[window] = {
            "retry_count": str(item.get("retry_count", item.get("attempt", ""))),
            "window_hit_gate": "passed" if item.get("passed") is True else "failed",
        }
    return attempts


def _parse_window_rank_steps(artifact_dir: Path, window: str) -> list[phase428.RankStep]:
    rank_steps: list[phase428.RankStep] = []
    for path in _trace_paths(artifact_dir, window):
        rank_steps.extend(phase428.parse_rank_steps(path, window))
    return rank_steps


def _serve_logs_for_window(artifact_dir: Path, window: str) -> list[Path]:
    return sorted(artifact_dir.glob(f"serve_{window}_attempt*.log"))


def _parse_iteration_detail_logs(paths: list[Path]) -> tuple[int, list[int]]:
    prefill_steps = 0
    decode_batches: list[int] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in ITERATION_DETAIL_RE.finditer(text):
            context_tokens = int(match.group("context_tokens"))
            generation_requests = int(match.group("generation_requests"))
            if context_tokens > 0:
                prefill_steps += 1
            if context_tokens == 0 and generation_requests > 0:
                decode_batches.append(generation_requests)
    return prefill_steps, decode_batches


def build_window_check_from_logs(
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    scenario: str = DEFAULT_SCENARIO,
    window: str,
    target_concurrency: int | None = None,
) -> WindowCheck:
    artifact_dir = artifact_root / scenario
    trace_count = len(_trace_paths(artifact_dir, window))
    prefill_step_count, batches = _parse_iteration_detail_logs(_serve_logs_for_window(artifact_dir, window))
    batch_mean = _mean([float(batch) for batch in batches])
    batch_min = min(batches) if batches else 0
    batch_max = max(batches) if batches else 0

    if window == "w0_prefill":
        passed = trace_count > 0 and prefill_step_count >= 1
        reason = "prefill_step_seen" if passed else "prefill_step_missing_or_trace_missing"
    else:
        if target_concurrency is None:
            target_concurrency = WINDOW_TARGET_CONCURRENCY.get(window)
        target_replica = float(target_concurrency or 0) / 2.0
        lower = max(1.0, target_replica * 0.35)
        upper = max(lower + 1.0, target_replica * 1.35)
        passed = trace_count > 0 and bool(batches) and lower <= batch_mean <= upper
        reason = "decode_batch_in_target_band" if passed else f"decode_batch_out_of_band_or_trace_missing_{lower:.1f}_{upper:.1f}"
    return WindowCheck(
        window=window,
        target_concurrency=target_concurrency,
        prefill_step_count=prefill_step_count,
        decode_step_count=len(batches),
        decode_batch_mean=batch_mean,
        decode_batch_min=batch_min,
        decode_batch_max=batch_max,
        passed=passed,
        reason=reason,
    )


def build_window_check(
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    scenario: str = DEFAULT_SCENARIO,
    window: str,
    target_concurrency: int | None = None,
) -> WindowCheck:
    artifact_dir = artifact_root / scenario
    rank_steps = _parse_window_rank_steps(artifact_dir, window)
    aggs = phase428._aggregate_steps(rank_steps)
    prefill_steps = [step for step in aggs if step.step_type == "prefill_or_mixed"]
    decode_steps = [step for step in aggs if step.step_type == "decode_only" and step.decode_batch > 0]
    batches = [step.decode_batch for step in decode_steps]
    batch_mean = _mean([float(batch) for batch in batches])
    batch_min = min(batches) if batches else 0
    batch_max = max(batches) if batches else 0

    if window == "w0_prefill":
        passed = len(prefill_steps) >= 1
        reason = "prefill_step_seen" if passed else "prefill_step_missing"
    else:
        if target_concurrency is None:
            target_concurrency = WINDOW_TARGET_CONCURRENCY.get(window)
        target_replica = float(target_concurrency or 0) / 2.0
        lower = max(1.0, target_replica * 0.35)
        upper = max(lower + 1.0, target_replica * 1.35)
        passed = len(decode_steps) >= 1 and lower <= batch_mean <= upper
        reason = "decode_batch_in_target_band" if passed else f"decode_batch_out_of_band_{lower:.1f}_{upper:.1f}"
    return WindowCheck(
        window=window,
        target_concurrency=target_concurrency,
        prefill_step_count=len(prefill_steps),
        decode_step_count=len(decode_steps),
        decode_batch_mean=batch_mean,
        decode_batch_min=batch_min,
        decode_batch_max=batch_max,
        passed=passed,
        reason=reason,
    )


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


def _dominant_prefill_verdict(prefill_steps: list[phase428.StepAggregate]) -> str:
    if not prefill_steps:
        return "prefill_steps_not_captured"
    bubble_share = _safe_ratio(
        _mean([step.bubble_ms_per_rank for step in prefill_steps]),
        _mean([step.wall_ms_per_rank for step in prefill_steps]),
    )
    if bubble_share >= 0.25:
        return "prefill_serialization_bubble_unmodeled"
    totals: dict[str, float] = {}
    for step in prefill_steps:
        for category, value in step.category_ms_per_rank.items():
            totals[category] = totals.get(category, 0.0) + value
    if not totals:
        return "prefill_kernel_category_missing"
    category = max(totals, key=totals.get)
    return f"prefill_{category}_undercharged"


def _decode_verdict(slope_rows: list[dict[str, object]], decode_steps: list[phase428.StepAggregate]) -> str:
    batches = [step.decode_batch for step in decode_steps if step.decode_batch > 0]
    if not batches or max(batches) - min(batches) < MIN_DECODE_BATCH_SPAN:
        return "decode_batch_span_too_narrow_for_slope"
    return phase428._dominant_decode_verdict(slope_rows, decode_steps)


def build_phase429_rows(
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    phase426_csv: Path = PHASE426_CSV,
    scenario: str = DEFAULT_SCENARIO,
) -> list[dict[str, str]]:
    artifact_dir = artifact_root / scenario
    if not artifact_dir.exists():
        raise ValueError(f"missing Phase429 artifact dir: {artifact_dir}")
    phase426 = _read_phase426_summary(phase426_csv, scenario)
    attempts = _read_window_attempts(artifact_dir)

    rank_steps: list[phase428.RankStep] = []
    for window in WINDOWS:
        rank_steps.extend(_parse_window_rank_steps(artifact_dir, window))
    step_aggs = phase428._aggregate_steps(rank_steps)
    prefill_steps = [step for step in step_aggs if step.step_type == "prefill_or_mixed"]
    decode_steps = [step for step in step_aggs if step.step_type == "decode_only" and step.decode_batch > 0]
    slope_rows = phase428._decode_slope_rows(step_aggs)

    batches = [step.decode_batch for step in decode_steps]
    decode_batch_min = min(batches) if batches else 0
    decode_batch_max = max(batches) if batches else 0
    decode_batch_span = decode_batch_max - decode_batch_min
    prefill_hit_gate = "passed" if prefill_steps else "failed"
    decode_batch_span_gate = "passed" if decode_batch_span >= MIN_DECODE_BATCH_SPAN else "failed"
    prefill_verdict = _dominant_prefill_verdict(prefill_steps)
    decode_verdict = _decode_verdict(slope_rows, decode_steps)

    if prefill_hit_gate == "passed" and decode_batch_span_gate == "passed":
        mechanism_verdict = "phase429_reprofile_window_hits_attributed"
        reconstruction_gate = "passed"
        reconstruction_error_pct = 0.0
        phase430_runtime_target = "apply_named_undercharge_or_bubble_fix"
        phase430_perfdb_target = "patch_or_recollect_named_perfdb_component"
    elif prefill_hit_gate != "passed":
        mechanism_verdict = "phase429_prefill_window_missed"
        reconstruction_gate = "blocked_prefill_not_captured"
        reconstruction_error_pct = math.nan
        phase430_runtime_target = "repeat_prefill_profile_before_runtime_change"
        phase430_perfdb_target = "none_until_prefill_window_hits"
    else:
        mechanism_verdict = "phase429_decode_batch_span_insufficient"
        reconstruction_gate = "blocked_decode_span_insufficient"
        reconstruction_error_pct = math.nan
        phase430_runtime_target = "repeat_decode_concurrency_sweep_before_runtime_change"
        phase430_perfdb_target = "none_until_decode_span_sufficient"

    common = {
        "source": SOURCE,
        "scenario": scenario,
        "artifact_dir": _display_path(artifact_dir),
        "phase426_steady_metric_penalty": phase426.get("steady_metric_penalty", ""),
        "phase426_prefill_share": phase426.get("prefill_attribution_share", ""),
        "phase426_decode_gap_share": phase426.get("decode_gap_attribution_share", ""),
        "phase426_peer_stall_share": phase426.get("peer_stall_attribution_share", ""),
        "phase429_reconstruction_error_pct": reconstruction_error_pct,
        "phase429_reconstruction_gate": reconstruction_gate,
        "prefill_step_count": len(prefill_steps),
        "decode_step_count": len(decode_steps),
        "decode_batch_min": decode_batch_min,
        "decode_batch_max": decode_batch_max,
        "decode_batch_span": decode_batch_span,
        "prefill_hit_gate": prefill_hit_gate,
        "decode_batch_span_gate": decode_batch_span_gate,
        "prefill_verdict": prefill_verdict,
        "decode_verdict": decode_verdict,
        "mechanism_verdict": mechanism_verdict,
        "phase430_runtime_target": phase430_runtime_target,
        "phase430_perfdb_target": phase430_perfdb_target,
        **_common_flags(),
    }

    rows: list[dict[str, object]] = []
    rows.append({"row_type": "summary", **common})
    for window in WINDOWS:
        check = build_window_check(
            artifact_root=artifact_root,
            scenario=scenario,
            window=window,
            target_concurrency=WINDOW_TARGET_CONCURRENCY.get(window),
        )
        rows.append(
            {
                "row_type": "window_check",
                **common,
                "window": window,
                "window_target_concurrency": check.target_concurrency,
                "window_hit_gate": "passed" if check.passed else "failed",
                "retry_count": attempts.get(window, {}).get("retry_count", ""),
                "decode_batch": check.decode_batch_mean,
            }
        )

    for step in step_aggs:
        attempt = attempts.get(step.window, {})
        rows.append(
            {
                "row_type": "step",
                **common,
                "window": step.window,
                "window_target_concurrency": WINDOW_TARGET_CONCURRENCY.get(step.window, ""),
                "window_hit_gate": attempt.get("window_hit_gate", ""),
                "retry_count": attempt.get("retry_count", ""),
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
                    "window_target_concurrency": WINDOW_TARGET_CONCURRENCY.get(step.window, ""),
                    "window_hit_gate": attempt.get("window_hit_gate", ""),
                    "retry_count": attempt.get("retry_count", ""),
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
            }
        )
    return [{field: _fmt(row.get(field, "")) for field in CSV_FIELDS} for row in rows]


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Phase429 rows must not be empty")
    summaries = [row for row in rows if row.get("row_type") == "summary"]
    if len(summaries) != 1:
        raise ValueError("Phase429 expects exactly one summary row")
    for row in rows:
        if row.get("source") != SOURCE:
            raise ValueError("unexpected source")
        if row.get("phase405_penalty_read") != FALSE:
            raise ValueError("phase405_penalty_read must stay false")
        if row.get("gpu_allowed") != TRUE or row.get("ssh_allowed") != TRUE:
            raise ValueError("Phase429 uses GPU/SSH measurement and must declare it")
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


def write_phase429_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_phase429_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    summary = [row for row in rows if row["row_type"] == "summary"][0]
    checks = [row for row in rows if row["row_type"] == "window_check"]
    slopes = [row for row in rows if row["row_type"] == "decode_slope"]
    lines = [
        "# Phase429 Guaranteed Kernel Reprofile",
        "",
        "Phase429 uses GPU only for measurement. It does not modify runtime, PerfDatabase, gate, or Default AIC readiness.",
        "",
        "## Verdict",
        "",
        f"- mechanism: `{summary['mechanism_verdict']}`",
        f"- prefill: `{summary['prefill_verdict']}`",
        f"- decode: `{summary['decode_verdict']}`",
        f"- reconstruction gate: `{summary['phase429_reconstruction_gate']}`",
        f"- Phase430 runtime target: `{summary['phase430_runtime_target']}`",
        f"- Phase430 PerfDB target: `{summary['phase430_perfdb_target']}`",
        "",
        "## Window Checks",
        "",
        "| window | target concurrency | gate | retry count | mean decode batch |",
        "|---|---:|---|---:|---:|",
    ]
    for row in checks:
        lines.append(
            f"| {row['window']} | {row['window_target_concurrency']} | {row['window_hit_gate']} | {row['retry_count']} | {row['decode_batch']} |"
        )
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            "| prefill steps | decode steps | decode batch min | decode batch max | span | steady penalty |",
            "|---:|---:|---:|---:|---:|---:|",
            f"| {summary['prefill_step_count']} | {summary['decode_step_count']} | {summary['decode_batch_min']} | {summary['decode_batch_max']} | {summary['decode_batch_span']} | {summary['phase426_steady_metric_penalty']} |",
            "",
            "## Decode Slope",
            "",
            "| category | mean ms/rank | slope ms/request |",
            "|---|---:|---:|",
        ]
    )
    for row in slopes[:8]:
        lines.append(
            f"| {row['category']} | {row['real_cuda_ms_per_rank']} | {row['real_slope_ms_per_request']} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- GPU/SSH were used only for measurement.",
            "- Runtime, PerfDatabase, and gate were not modified.",
            "- Phase405 penalty was not read.",
            "- Default AIC remains No-Go.",
            "",
        ]
    )
    return "\n".join(lines)


def write_phase429_md(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase429_md(rows), encoding="utf-8")


def _print_check(check: WindowCheck) -> None:
    print(json.dumps(
        {
            "window": check.window,
            "target_concurrency": check.target_concurrency,
            "prefill_step_count": check.prefill_step_count,
            "decode_step_count": check.decode_step_count,
            "decode_batch_mean": check.decode_batch_mean,
            "decode_batch_min": check.decode_batch_min,
            "decode_batch_max": check.decode_batch_max,
            "passed": check.passed,
            "reason": check.reason,
        },
        sort_keys=True,
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--phase426-csv", type=Path, default=PHASE426_CSV)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--check-window", choices=WINDOWS)
    parser.add_argument("--target-concurrency", type=int)
    args = parser.parse_args()

    if args.check_window:
        try:
            check = build_window_check_from_logs(
                artifact_root=args.artifact_root,
                scenario=args.scenario,
                window=args.check_window,
                target_concurrency=args.target_concurrency,
            )
            _print_check(check)
            raise SystemExit(0 if check.passed else 1)
        except Exception as exc:
            payload = {
                "window": args.check_window,
                "target_concurrency": args.target_concurrency,
                "prefill_step_count": 0,
                "decode_step_count": 0,
                "decode_batch_mean": math.nan,
                "decode_batch_min": 0,
                "decode_batch_max": 0,
                "passed": False,
                "reason": type(exc).__name__,
                "error": str(exc),
            }
            print(json.dumps(payload, sort_keys=True))
            raise SystemExit(1)

    rows = build_phase429_rows(
        artifact_root=args.artifact_root,
        phase426_csv=args.phase426_csv,
        scenario=args.scenario,
    )
    write_phase429_csv(args.csv, rows)
    write_phase429_md(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
