#!/usr/bin/env python3
"""Phase451-I: discriminate EngineCore-side arrival visibility."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

SCENARIO = "K2.5-tp4ep8dp2-8k2k"
RUN_DIR = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase446_b2b_event_timing/"
    / "overhead_gate_20260708_075726/overhead_on"
)
SCENARIO_DIR = RUN_DIR / SCENARIO
DEFAULT_METRICS = SCENARIO_DIR / "metrics.jsonl"
DEFAULT_SERVE_LOG = SCENARIO_DIR / "serve.log"
DEFAULT_EVENT = RUN_DIR / "event_timing.jsonl"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase451i_arrival_visibility.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase451i_arrival_visibility.md"
DEFAULT_READINESS = "No-Go"

CSV_FIELDS = [
    "section",
    "scenario",
    "metric",
    "value",
    "target",
    "status",
    "note",
]

PROM_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+0-9.eE]+)$"
)
LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')
ADDED_RE = re.compile(r"\b(added|add(?:ing)?|new)\s+request\b|\brequest\b.*\badded\b", re.IGNORECASE)


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def _fmt(value: object) -> str:
    if value == "":
        return ""
    if isinstance(value, bool):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "nan"
    if abs(number) >= 1000:
        return f"{number:.0f}"
    if number.is_integer():
        return f"{number:.0f}"
    return f"{number:.6f}".rstrip("0").rstrip(".")


def _row(
    section: str,
    metric: str,
    value: object,
    *,
    target: object = "",
    status: str = "",
    note: str = "",
    scenario: str = SCENARIO,
) -> dict[str, object]:
    return {
        "section": section,
        "scenario": scenario,
        "metric": metric,
        "value": value,
        "target": target,
        "status": status,
        "note": note,
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse_prometheus_line(line: str) -> tuple[str, dict[str, str], float] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    match = PROM_RE.match(stripped)
    if not match:
        return None
    labels = {
        label.group(1): label.group(2)
        for label in LABEL_RE.finditer(match.group("labels") or "")
    }
    return match.group("name"), labels, float(match.group("value"))


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def extract_waiting_running_samples(path: Path) -> list[dict[str, float | str]]:
    snapshots: list[dict[str, object]] = []
    with _open_text(path) as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            ts = _parse_ts(str(payload["ts"]))
            by_engine: dict[str, dict[str, float]] = defaultdict(
                lambda: {"running": math.nan, "waiting": math.nan, "kv": math.nan}
            )
            for metric_line in str(payload.get("body", "")).splitlines():
                parsed = parse_prometheus_line(metric_line)
                if parsed is None:
                    continue
                name, labels, value = parsed
                engine = labels.get("engine")
                if engine is None:
                    continue
                if name == "vllm:num_requests_running":
                    by_engine[engine]["running"] = value
                elif name == "vllm:num_requests_waiting":
                    by_engine[engine]["waiting"] = value
                elif name == "vllm:kv_cache_usage_perc":
                    by_engine[engine]["kv"] = value
            if by_engine:
                snapshots.append({"ts": ts, "by_engine": by_engine})
    if not snapshots:
        return []
    t0 = min(snapshot["ts"] for snapshot in snapshots)
    rows: list[dict[str, float | str]] = []
    for snapshot in snapshots:
        t_s = (snapshot["ts"] - t0).total_seconds()
        for engine, values in dict(snapshot["by_engine"]).items():
            running = values["running"]
            waiting = values["waiting"]
            kv = values["kv"]
            rows.append(
                {
                    "t_s": t_s,
                    "engine": str(engine),
                    "running": 0.0 if math.isnan(running) else running,
                    "waiting": 0.0 if math.isnan(waiting) else waiting,
                    "kv": math.nan if math.isnan(kv) else kv,
                }
            )
    return rows


def classify_engine_visibility(
    samples: Iterable[dict[str, float | str]],
    *,
    active_window_s: float = 60.0,
    burst_waiting_threshold: float = 50.0,
    drizzle_waiting_threshold: float = 5.0,
) -> dict[str, object]:
    rows = list(samples)
    active = [
        row for row in rows
        if float(row.get("running", 0.0)) + float(row.get("waiting", 0.0)) > 0.0
    ]
    if not active:
        return {
            "visibility": "unknown",
            "route": "blocked_no_active_metrics",
            "active_samples": 0,
            "max_waiting_per_engine": 0.0,
            "max_waiting_global": 0.0,
            "max_running_per_engine": 0.0,
            "note": "no nonzero running/waiting samples",
        }

    active_start = min(float(row["t_s"]) for row in active)
    window = [
        row for row in active
        if float(row["t_s"]) <= active_start + active_window_s
    ]
    waiting_by_time: dict[float, float] = defaultdict(float)
    for row in window:
        waiting_by_time[float(row["t_s"])] += float(row.get("waiting", 0.0))
    max_waiting_per_engine = max(float(row.get("waiting", 0.0)) for row in window)
    max_waiting_global = max(waiting_by_time.values()) if waiting_by_time else 0.0
    max_running_per_engine = max(float(row.get("running", 0.0)) for row in window)
    first_active_running = min(float(row.get("running", 0.0)) for row in window)
    running_growth = max_running_per_engine - first_active_running

    if (
        max_waiting_per_engine >= burst_waiting_threshold
        or max_waiting_global >= burst_waiting_threshold * 2
    ):
        return {
            "visibility": "engine_burst",
            "route": "debug_observation_required",
            "active_samples": len(window),
            "active_start_s": active_start,
            "max_waiting_per_engine": max_waiting_per_engine,
            "max_waiting_global": max_waiting_global,
            "max_running_per_engine": max_running_per_engine,
            "running_growth": running_growth,
            "note": "EngineCore-visible waiting queue jumps to burst scale",
        }
    if max_waiting_per_engine <= drizzle_waiting_threshold and running_growth > 0:
        return {
            "visibility": "engine_drizzle",
            "route": "engine_visible_arrival_replay",
            "active_samples": len(window),
            "active_start_s": active_start,
            "max_waiting_per_engine": max_waiting_per_engine,
            "max_waiting_global": max_waiting_global,
            "max_running_per_engine": max_running_per_engine,
            "running_growth": running_growth,
            "note": "waiting stays near empty while running ramps",
        }
    return {
        "visibility": "inconclusive",
        "route": "blocked_pending_debug_or_better_metrics",
        "active_samples": len(window),
        "active_start_s": active_start,
        "max_waiting_per_engine": max_waiting_per_engine,
        "max_waiting_global": max_waiting_global,
        "max_running_per_engine": max_running_per_engine,
        "running_growth": running_growth,
        "note": "waiting/running does not cleanly match burst or drizzle thresholds",
    }


def find_added_request_lines(lines: Iterable[str], *, limit: int = 50) -> list[str]:
    matches: list[str] = []
    for line in lines:
        if ADDED_RE.search(line):
            matches.append(line.rstrip("\n"))
            if len(matches) >= limit:
                break
    return matches


def scan_added_request_lines(path: Path) -> list[str]:
    with _open_text(path) as f:
        return find_added_request_lines(f)


def summarize_samples(samples: list[dict[str, float | str]]) -> dict[str, object]:
    if not samples:
        return {
            "sample_count": 0,
            "max_running": 0.0,
            "max_waiting": 0.0,
            "max_kv": math.nan,
            "running_p50": math.nan,
            "waiting_p50": math.nan,
        }
    running = [float(row["running"]) for row in samples]
    waiting = [float(row["waiting"]) for row in samples]
    kv = [float(row["kv"]) for row in samples if not math.isnan(float(row["kv"]))]
    return {
        "sample_count": len(samples),
        "max_running": max(running),
        "max_waiting": max(waiting),
        "max_kv": max(kv) if kv else math.nan,
        "running_p50": _percentile(running, 50),
        "waiting_p50": _percentile(waiting, 50),
    }


def build_report_rows(
    *,
    metrics_path: Path = DEFAULT_METRICS,
    serve_log: Path = DEFAULT_SERVE_LOG,
) -> list[dict[str, object]]:
    samples = extract_waiting_running_samples(metrics_path)
    summary = summarize_samples(samples)
    verdict = classify_engine_visibility(samples)
    added = scan_added_request_lines(serve_log)
    route = str(verdict["route"])

    rows: list[dict[str, object]] = [
        _row("i1_discriminator", "metrics_sample_count", summary["sample_count"]),
        _row("i1_discriminator", "max_running_per_engine", summary["max_running"]),
        _row("i1_discriminator", "max_waiting_per_engine_all_run", summary["max_waiting"]),
        _row("i1_discriminator", "max_kv_usage", summary["max_kv"]),
        _row("i1_discriminator", "running_p50", summary["running_p50"]),
        _row("i1_discriminator", "waiting_p50", summary["waiting_p50"]),
        _row("i1_discriminator", "active_window_samples", verdict.get("active_samples", 0)),
        _row("i1_discriminator", "active_start_s", verdict.get("active_start_s", "")),
        _row(
            "i1_discriminator",
            "active_window_max_waiting_per_engine",
            verdict.get("max_waiting_per_engine", 0.0),
            target=">=50 burst; <=5 drizzle",
        ),
        _row(
            "i1_discriminator",
            "active_window_max_waiting_global",
            verdict.get("max_waiting_global", 0.0),
            target=">=100 global burst",
        ),
        _row("i1_discriminator", "active_window_max_running_per_engine", verdict.get("max_running_per_engine", 0.0)),
        _row(
            "i1_discriminator",
            "engine_visibility",
            verdict["visibility"],
            target="burst or drizzle",
            status="pass" if verdict["visibility"] in {"engine_burst", "engine_drizzle"} else "blocked",
            note=str(verdict["note"]),
        ),
        _row("i1_discriminator", "next_route", route, status="blocked" if route.startswith("blocked") else "open"),
        _row(
            "i1_added_lines",
            "added_request_line_count",
            len(added),
            target=">0 gives per-request engine-visible arrival",
            status="pass" if added else "blocked",
            note=(added[0][:180] if added else "no Added request lines at INFO-level serve.log"),
        ),
    ]

    if verdict["visibility"] == "engine_burst":
        rows.extend(
            [
                _row(
                    "i3_debug_run",
                    "debug_short_run_needed",
                    True,
                    status="open",
                    note="metrics reject drizzle; need DEBUG/victim/waiting observation before any semantic fix",
                ),
                _row("i2_boundary", "engine_visible_replay", "not_applied", status="blocked", note="drizzle route not supported by existing metrics"),
            ]
        )
    elif verdict["visibility"] == "engine_drizzle":
        rows.extend(
            [
                _row("i2_boundary", "engine_visible_replay", "candidate", status="open", note="derive visible arrivals from Added lines or prefill lower-bound events"),
                _row("i3_debug_run", "debug_short_run_needed", False, status="blocked", note="not needed unless replay fails"),
            ]
        )
    else:
        rows.extend(
            [
                _row("i2_boundary", "engine_visible_replay", "not_applied", status="blocked"),
                _row("i3_debug_run", "debug_short_run_needed", "undecided", status="blocked"),
            ]
        )

    rows.extend(
        [
            _row("i4_accept", "runtime_patch", "not_applied", status="blocked", note="Phase451-I Step1 is report-only until route is proven"),
            _row("i4_accept", "default_aic", DEFAULT_READINESS, status="blocked"),
        ]
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def write_md(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_metric = {str(row["metric"]): row for row in rows}
    visibility = by_metric.get("engine_visibility", {}).get("value", "unknown")
    route = by_metric.get("next_route", {}).get("value", "unknown")
    lines = [
        "# Phase451-I arrival visibility",
        "",
        (
            f"结论: EngineCore 侧判别为 `{visibility}`, 下一步路线 `{route}`。"
            "本报告只做判别,不改 runtime、PerfDB 或 validate gate。"
        ),
        "",
        f"- Default AIC remains `{DEFAULT_READINESS}`.",
        "",
        "## Summary",
        "",
        "| section | metric | value | target | status | note |",
        "|---|---|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['section']} | {row['metric']} | {_fmt(row['value'])} | "
            f"{row.get('target', '')} | {row.get('status', '')} | {row.get('note', '')} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Waiting/running gauges are sampled at metrics poll cadence; they can prove burst-scale visibility, but not victim identity.",
            "- INFO serve.log has no Added request lines in the current raw, so per-request EngineCore visible arrival is unavailable offline.",
            "- If the route is `debug_observation_required`, the next valid step is a short DEBUG ramp run, not a scheduler fix.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--serve-log", type=Path, default=DEFAULT_SERVE_LOG)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_report_rows(metrics_path=args.metrics, serve_log=args.serve_log)
    write_csv(args.csv_out, rows)
    write_md(args.md_out, rows)
    print(f"wrote {args.csv_out}")
    print(f"wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
