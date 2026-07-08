#!/usr/bin/env python3
"""Phase449: real KV watermark profile and validate-capacity audit."""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SCENARIO = "K2.5-tp4ep8dp2-8k2k"
DEFAULT_SERVE_LOG = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase446_b2b_event_timing/"
    / "overhead_gate_20260708_075726/overhead_on/"
    / SCENARIO
    / "serve.log"
)
DEFAULT_METRICS = DEFAULT_SERVE_LOG.with_name("metrics.jsonl")
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase449_kv_watermark_profile.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase449_kv_watermark_profile.md"

BLOCK_SIZE = 16
YEAR = 2026
DEFAULT_READINESS = "No-Go"

CSV_FIELDS = [
    "scenario",
    "section",
    "metric",
    "engine",
    "value",
    "p10",
    "p50",
    "p90",
    "min",
    "max",
    "sample_count",
    "source",
    "passed",
    "note",
]

PROM_RE = re.compile(
    r"^(?P<name>vllm:(?:num_requests_running|num_requests_waiting|"
    r"kv_cache_usage_perc|num_preemptions_total))\{(?P<labels>[^}]*)\}\s+"
    r"(?P<value>[-+0-9.eE]+)$"
)
ENGINE_RE = re.compile(r'engine="(?P<engine>\d+)"')
KV_RE = re.compile(
    r"EngineCore_DP(?P<engine>\d+).*?GPU KV cache size: (?P<tokens>[\d,]+) tokens"
)
ITER_RE = re.compile(
    r"EngineCore_DP(?P<engine>\d+).*?INFO (?P<month>\d{2})-(?P<day>\d{2}) "
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2}).*?"
    r"Iteration\((?P<iteration>\d+)\): (?P<context_requests>\d+) context requests, "
    r"(?P<context_tokens>\d+) context tokens, (?P<generation_requests>\d+) generation requests, "
    r"(?P<generation_tokens>\d+) generation tokens, iteration elapsed time: "
    r"(?P<elapsed_ms>[-+0-9.]+) ms"
)


@dataclass(frozen=True)
class MetricSnapshot:
    timestamp: datetime
    values: dict[str, dict[str, float]]


@dataclass(frozen=True)
class KVCapacityRow:
    engine: str
    kv_cache_tokens: int
    num_gpu_blocks: int
    source: str


@dataclass(frozen=True)
class IterationStep:
    engine: str
    iteration: int
    timestamp: datetime
    context_tokens: int
    generation_requests: int
    elapsed_ms: float

    @property
    def is_mixed(self) -> bool:
        return self.context_tokens > 0 and self.generation_requests > 0


@dataclass(frozen=True)
class MixedMetricSample:
    engine: str
    iteration: int
    context_tokens: int
    generation_requests: int
    elapsed_ms: float
    running: float
    waiting: float
    kv_usage: float
    preemptions: float
    delta_s: float


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100.0
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    frac = index - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    data = list(values)
    if not data:
        return {
            "value": math.nan,
            "p10": math.nan,
            "p50": math.nan,
            "p90": math.nan,
            "min": math.nan,
            "max": math.nan,
            "sample_count": 0,
        }
    return {
        "value": statistics.mean(data),
        "p10": _percentile(data, 10),
        "p50": _percentile(data, 50),
        "p90": _percentile(data, 90),
        "min": min(data),
        "max": max(data),
        "sample_count": len(data),
    }


def _parse_log_timestamp(match: re.Match[str]) -> datetime:
    return datetime(
        YEAR,
        int(match.group("month")),
        int(match.group("day")),
        int(match.group("hour")),
        int(match.group("minute")),
        int(match.group("second")),
        tzinfo=timezone.utc,
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_metric_snapshots(path: Path) -> list[MetricSnapshot]:
    snapshots: list[MetricSnapshot] = []
    with _open_text(path) as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            values: dict[str, dict[str, float]] = {}
            for metric_line in str(payload.get("body", "")).splitlines():
                match = PROM_RE.match(metric_line.strip())
                if not match:
                    continue
                engine_match = ENGINE_RE.search(match.group("labels"))
                if not engine_match:
                    continue
                metric = match.group("name").removeprefix("vllm:")
                values.setdefault(metric, {})[engine_match.group("engine")] = float(match.group("value"))
            if values:
                snapshots.append(
                    MetricSnapshot(
                        timestamp=datetime.fromisoformat(payload["ts"]),
                        values=values,
                    )
                )
    return snapshots


def parse_kv_capacity_rows(path: Path) -> list[KVCapacityRow]:
    rows: list[KVCapacityRow] = []
    with _open_text(path) as f:
        for line_no, line in enumerate(f, start=1):
            match = KV_RE.search(line)
            if not match:
                continue
            tokens = int(match.group("tokens").replace(",", ""))
            rows.append(
                KVCapacityRow(
                    engine=match.group("engine"),
                    kv_cache_tokens=tokens,
                    num_gpu_blocks=tokens // BLOCK_SIZE,
                    source=f"{_display_path(path)}:{line_no}",
                )
            )
    return rows


def parse_iteration_steps(path: Path) -> list[IterationStep]:
    steps: list[IterationStep] = []
    with _open_text(path) as f:
        for line in f:
            match = ITER_RE.search(line)
            if not match:
                continue
            steps.append(
                IterationStep(
                    engine=match.group("engine"),
                    iteration=int(match.group("iteration")),
                    timestamp=_parse_log_timestamp(match),
                    context_tokens=int(match.group("context_tokens")),
                    generation_requests=int(match.group("generation_requests")),
                    elapsed_ms=float(match.group("elapsed_ms")),
                )
            )
    return steps


def _nearest_snapshot(step: IterationStep, snapshots: list[MetricSnapshot]) -> tuple[MetricSnapshot, float] | None:
    candidates = [
        (snapshot, abs((snapshot.timestamp - step.timestamp).total_seconds()))
        for snapshot in snapshots
        if step.engine in snapshot.values.get("num_requests_running", {})
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[1])


def align_mixed_steps_to_metrics(
    steps: Iterable[IterationStep],
    snapshots: list[MetricSnapshot],
    *,
    max_delta_s: float,
) -> list[MixedMetricSample]:
    samples: list[MixedMetricSample] = []
    for step in steps:
        if not step.is_mixed:
            continue
        nearest = _nearest_snapshot(step, snapshots)
        if nearest is None:
            continue
        snapshot, delta_s = nearest
        if delta_s > max_delta_s:
            continue
        samples.append(
            MixedMetricSample(
                engine=step.engine,
                iteration=step.iteration,
                context_tokens=step.context_tokens,
                generation_requests=step.generation_requests,
                elapsed_ms=step.elapsed_ms,
                running=snapshot.values.get("num_requests_running", {}).get(step.engine, math.nan),
                waiting=snapshot.values.get("num_requests_waiting", {}).get(step.engine, math.nan),
                kv_usage=snapshot.values.get("kv_cache_usage_perc", {}).get(step.engine, math.nan),
                preemptions=snapshot.values.get("num_preemptions_total", {}).get(step.engine, math.nan),
                delta_s=delta_s,
            )
        )
    return samples


def _load_validate_module():
    path = REPO_ROOT / "scripts" / "validate_cb_simulator.py"
    spec = importlib.util.spec_from_file_location("validate_cb_simulator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_capacity(scenario: str) -> tuple[int, int, str]:
    validate = _load_validate_module()
    capacity = validate.PHASE397K_KV_CAPACITY_BY_SCENARIO[scenario]
    source_lines = ",".join(str(line) for line in capacity.serve_log_line_numbers)
    return (
        int(capacity.kv_cache_tokens),
        int(capacity.num_gpu_blocks),
        f"{capacity.serve_log}:{source_lines}",
    )


def capacity_diff_row(
    *,
    scenario: str,
    validate_tokens: int,
    validate_blocks: int,
    real_tokens: int,
    real_blocks: int,
    source: str,
) -> dict[str, object]:
    ratio = validate_tokens / real_tokens if real_tokens else math.nan
    return {
        "scenario": scenario,
        "section": "capacity_diff",
        "metric": "validate_tokens_over_real_tokens",
        "engine": "all",
        "value": ratio,
        "p10": "",
        "p50": "",
        "p90": "",
        "min": real_tokens,
        "max": validate_tokens,
        "sample_count": "",
        "source": source,
        "passed": abs(validate_tokens - real_tokens) <= BLOCK_SIZE,
        "note": f"validate_blocks={validate_blocks}; real_blocks={real_blocks}",
    }


def _metric_rows_from_samples(
    scenario: str,
    section: str,
    metric: str,
    by_engine: dict[str, list[float]],
    *,
    source: str,
    note: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for engine, values in sorted(by_engine.items()):
        summary = _summary(values)
        rows.append(
            {
                "scenario": scenario,
                "section": section,
                "metric": metric,
                "engine": engine,
                "value": summary["value"],
                "p10": summary["p10"],
                "p50": summary["p50"],
                "p90": summary["p90"],
                "min": summary["min"],
                "max": summary["max"],
                "sample_count": summary["sample_count"],
                "source": source,
                "passed": "",
                "note": note,
            }
        )
    return rows


def analyze(
    *,
    scenario: str,
    serve_log: Path,
    metrics_jsonl: Path,
    max_delta_s: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    snapshots = parse_metric_snapshots(metrics_jsonl)
    steps = parse_iteration_steps(serve_log)
    kv_rows = parse_kv_capacity_rows(serve_log)
    aligned = align_mixed_steps_to_metrics(steps, snapshots, max_delta_s=max_delta_s)

    rows: list[dict[str, object]] = []
    for kv in kv_rows:
        rows.append(
            {
                "scenario": scenario,
                "section": "real_capacity",
                "metric": "kv_cache_tokens",
                "engine": kv.engine,
                "value": kv.kv_cache_tokens,
                "p10": "",
                "p50": "",
                "p90": "",
                "min": "",
                "max": "",
                "sample_count": 1,
                "source": kv.source,
                "passed": "",
                "note": f"num_gpu_blocks={kv.num_gpu_blocks}; block_size={BLOCK_SIZE}",
            }
        )

    real_tokens = int(statistics.median([row.kv_cache_tokens for row in kv_rows])) if kv_rows else 0
    real_blocks = real_tokens // BLOCK_SIZE if real_tokens else 0
    validate_tokens, validate_blocks, validate_source = _validate_capacity(scenario)
    rows.append(
        capacity_diff_row(
            scenario=scenario,
            validate_tokens=validate_tokens,
            validate_blocks=validate_blocks,
            real_tokens=real_tokens,
            real_blocks=real_blocks,
            source=f"validate={validate_source}; real={kv_rows[0].source if kv_rows else serve_log}",
        )
    )

    for metric_name in [
        "num_requests_running",
        "num_requests_waiting",
        "kv_cache_usage_perc",
        "num_preemptions_total",
    ]:
        by_engine: dict[str, list[float]] = {}
        for snapshot in snapshots:
            for engine, value in snapshot.values.get(metric_name, {}).items():
                by_engine.setdefault(engine, []).append(value)
        rows.extend(
            _metric_rows_from_samples(
                scenario,
                "metrics_full_run",
                metric_name,
                by_engine,
                source=_display_path(metrics_jsonl),
                note="all poll samples",
            )
        )

    mixed_fields = {
        "mixed_context_tokens": "context_tokens",
        "mixed_decode_batch": "generation_requests",
        "mixed_nearest_running": "running",
        "mixed_nearest_waiting": "waiting",
        "mixed_nearest_kv_usage": "kv_usage",
        "mixed_nearest_preemptions": "preemptions",
        "mixed_metric_delta_s": "delta_s",
    }
    for metric, attr in mixed_fields.items():
        by_engine = {}
        for sample in aligned:
            by_engine.setdefault(sample.engine, []).append(float(getattr(sample, attr)))
        rows.extend(
            _metric_rows_from_samples(
                scenario,
                "mixed_aligned",
                metric,
                by_engine,
                source=f"{_display_path(serve_log)} + {_display_path(metrics_jsonl)}",
                note=f"nearest metrics poll within {max_delta_s:.1f}s",
            )
        )

    capacity_ceiling = real_tokens / 10000.0 if real_tokens else math.nan
    rows.append(
        {
            "scenario": scenario,
            "section": "derived",
            "metric": "kv_ceiling_full_8k2k_requests",
            "engine": "all",
            "value": capacity_ceiling,
            "p10": "",
            "p50": "",
            "p90": "",
            "min": "",
            "max": "",
            "sample_count": "",
            "source": "real_capacity/(isl+osl)",
            "passed": "",
            "note": "rough ceiling for full 8000+2000 token requests per engine",
        }
    )

    metadata = {
        "scenario": scenario,
        "real_tokens": real_tokens,
        "validate_tokens": validate_tokens,
        "capacity_ratio": validate_tokens / real_tokens if real_tokens else math.nan,
        "mixed_samples": len(aligned),
        "max_preemptions": max(
            (
                value
                for snapshot in snapshots
                for value in snapshot.values.get("num_preemptions_total", {}).values()
            ),
            default=math.nan,
        ),
    }
    return rows, metadata


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
    if abs(number) >= 1000 or number.is_integer():
        return f"{number:.0f}"
    return f"{number:.4f}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, object]], metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    capacity_ratio = float(metadata["capacity_ratio"])
    conclusion = (
        "validate 8k2k DP2 capacity is stale"
        if capacity_ratio > 1.05
        else "validate 8k2k DP2 capacity matches real log"
    )
    lines = [
        "# Phase449 KV watermark profile",
        "",
        f"结论: `{conclusion}`。real KV={metadata['real_tokens']} tokens, "
        f"validate KV={metadata['validate_tokens']} tokens, ratio={capacity_ratio:.3f}x。",
        "",
        "| section | metric | engine | value | p10 | p50 | p90 | min | max | n | note |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    interesting = {"real_capacity", "capacity_diff", "mixed_aligned", "derived"}
    for row in rows:
        if row["section"] not in interesting:
            continue
        lines.append(
            "| {section} | {metric} | {engine} | {value} | {p10} | {p50} | {p90} | "
            "{min} | {max} | {sample_count} | {note} |".format(
                section=row["section"],
                metric=row["metric"],
                engine=row["engine"],
                value=_fmt(row["value"]),
                p10=_fmt(row["p10"]),
                p50=_fmt(row["p50"]),
                p90=_fmt(row["p90"]),
                min=_fmt(row["min"]),
                max=_fmt(row["max"]),
                sample_count=_fmt(row["sample_count"]),
                note=row["note"],
            )
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- `valid_for_default=false`; this is a waterline audit, not a readiness upgrade.",
            "- `num_preemptions_total` is reported from the same metrics stream; no GPU collection was run.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument("--serve-log", type=Path, default=DEFAULT_SERVE_LOG)
    parser.add_argument("--metrics-jsonl", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--max-delta-s", type=float, default=3.0)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows, metadata = analyze(
        scenario=args.scenario,
        serve_log=args.serve_log,
        metrics_jsonl=args.metrics_jsonl,
        max_delta_s=args.max_delta_s,
    )
    write_csv(args.csv_out, rows)
    write_md(args.md_out, rows, metadata)
    print(f"wrote {args.csv_out}")
    print(f"wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
