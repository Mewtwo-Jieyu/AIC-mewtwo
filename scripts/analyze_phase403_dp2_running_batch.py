#!/usr/bin/env python3
"""Phase403: parse DP2 /metrics recollect and attribute the remaining gap."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = "phase403_dp2_running_batch"
DEFAULT_RAW_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase403_dp2_stats"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase403_dp2_running_batch.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase403_dp2_running_batch.md"
PHASE401_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase401_preempt_admission.csv"
DEFAULT_READINESS = "No-Go"

DP2_SCENARIOS = (
    "K2.5-tp4ep8dp2-8k2k",
    "K2.5-tp4ep8dp2-32k3k",
)

CSV_FIELDS = [
    "source",
    "scenario",
    "tp",
    "dp",
    "ep",
    "isl",
    "osl",
    "max_num_batched_tokens",
    "world_size",
    "ok_requests",
    "failed_requests",
    "phase403_output_tok_s_global",
    "phase403_output_tok_s_gpu",
    "phase401_sim_output_tok_s_gpu",
    "phase401_vs_phase403_error_ratio",
    "phase401_vs_phase403_direction",
    "sim_peak_decode_reqs_per_iter",
    "sim_peak_decode_reqs_global",
    "real_running_global_max",
    "real_running_global_mean",
    "real_running_global_p50",
    "real_running_vs_sim_global_ratio",
    "running_engine0_max",
    "running_engine1_max",
    "running_engine0_mean",
    "running_engine1_mean",
    "waiting_global_max",
    "preemptions_total_last",
    "preemptions_total_delta",
    "gpu_cache_usage_max_pct",
    "metrics_samples",
    "gpu_kv_cache_tokens_per_engine",
    "num_gpu_blocks",
    "kv_cache_line_numbers",
    "prefix_caching",
    "collection_status",
    "attribution_verdict",
    "phase404_target",
    "artifact_dir",
    "gpu_allowed",
    "ssh_allowed",
    "runtime_modified",
    "perf_database",
    "valid_for_default",
    "diagnostic_only",
    "default_readiness",
]


@dataclass(frozen=True)
class MetricsSummary:
    samples: int
    running_global_max: float
    running_global_mean: float
    running_global_p50: float
    running_engine0_max: float | None
    running_engine1_max: float | None
    running_engine0_mean: float | None
    running_engine1_mean: float | None
    waiting_global_max: float
    preemptions_total_last: float
    preemptions_total_delta: float
    gpu_cache_usage_max_pct: float | None


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


def _read_csv_by_scenario(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        return {row["scenario"]: row for row in csv.DictReader(f)}


def _safe_error_ratio(predicted: float, real: float) -> float:
    if predicted <= 0 or real <= 0:
        return math.inf
    return max(predicted, real) / min(predicted, real)


def _direction(predicted: float, real: float) -> str:
    if predicted > real:
        return "sim_over_predicts_throughput"
    if predicted < real:
        return "sim_under_predicts_throughput"
    return "matched"


def _median(values: list[float]) -> float:
    if not values:
        return math.nan
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


_PROM_LINE = re.compile(
    r"^(?P<metric>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)


def _parse_labels(text: str | None) -> dict[str, str]:
    if not text:
        return {}
    labels: dict[str, str] = {}
    for key, value in re.findall(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"', text):
        labels[key] = value
    return labels


def _engine_key(labels: dict[str, str]) -> str:
    for key in ("engine", "data_parallel_rank", "rank"):
        if key in labels:
            value = labels[key]
            return value.split("_")[-1] if value.startswith("EngineCore_") else value
    return "all"


def _parse_metric_body(body: str) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    running: dict[str, float] = {}
    waiting: dict[str, float] = {}
    preemptions: dict[str, float] = {}
    cache_usage: dict[str, float] = {}
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROM_LINE.match(line)
        if not match:
            continue
        metric = match.group("metric")
        labels = _parse_labels(match.group("labels"))
        engine = _engine_key(labels)
        value = float(match.group("value"))
        if metric == "vllm:num_requests_running":
            running[engine] = value
        elif metric == "vllm:num_requests_waiting":
            waiting[engine] = value
        elif metric == "vllm:num_preemptions_total":
            preemptions[engine] = value
        elif "cache_usage" in metric or "gpu_cache" in metric:
            cache_usage[engine] = value
    return running, waiting, preemptions, cache_usage


def parse_metrics_jsonl(path: Path) -> MetricsSummary:
    running_global: list[float] = []
    waiting_global: list[float] = []
    preemption_global: list[float] = []
    cache_global: list[float] = []
    by_engine_running: dict[str, list[float]] = {}

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") != 200:
                continue
            running, waiting, preemptions, cache_usage = _parse_metric_body(
                record.get("body", "")
            )
            if not running and not waiting and not preemptions:
                continue
            running_global.append(sum(running.values()))
            waiting_global.append(sum(waiting.values()))
            preemption_global.append(sum(preemptions.values()))
            if cache_usage:
                cache_global.append(max(cache_usage.values()))
            for engine, value in running.items():
                by_engine_running.setdefault(engine, []).append(value)

    if not running_global:
        raise ValueError(f"missing running metrics in {path}")

    engine0 = by_engine_running.get("0", [])
    engine1 = by_engine_running.get("1", [])
    preempt_delta = (
        preemption_global[-1] - preemption_global[0]
        if len(preemption_global) >= 2
        else 0.0
    )
    return MetricsSummary(
        samples=len(running_global),
        running_global_max=max(running_global),
        running_global_mean=_mean(running_global),
        running_global_p50=_median(running_global),
        running_engine0_max=max(engine0) if engine0 else None,
        running_engine1_max=max(engine1) if engine1 else None,
        running_engine0_mean=_mean(engine0) if engine0 else None,
        running_engine1_mean=_mean(engine1) if engine1 else None,
        waiting_global_max=max(waiting_global) if waiting_global else 0.0,
        preemptions_total_last=preemption_global[-1] if preemption_global else 0.0,
        preemptions_total_delta=preempt_delta,
        gpu_cache_usage_max_pct=max(cache_global) if cache_global else None,
    )


def _parse_kv_cache(path: Path) -> tuple[int | None, str]:
    values: list[int] = []
    line_numbers: list[int] = []
    pattern = re.compile(r"GPU KV cache size:\s*([0-9,]+)\s+tokens")
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        match = pattern.search(line)
        if match:
            values.append(int(match.group(1).replace(",", "")))
            line_numbers.append(line_no)
    if not values:
        return None, ""
    unique_values = sorted(set(values))
    if len(unique_values) != 1:
        raise ValueError(f"mixed GPU KV cache sizes in {path}: {unique_values}")
    return unique_values[0], ";".join(str(value) for value in line_numbers)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _scenario_row(
    *,
    scenario: str,
    raw_root: Path,
    phase401: dict[str, dict[str, str]],
) -> dict[str, str]:
    scenario_dir = raw_root / scenario
    meta = _load_json(scenario_dir / "meta.json")
    bench = _load_json(scenario_dir / "bench_result.json")
    metrics = parse_metrics_jsonl(scenario_dir / "metrics.jsonl")
    kv_tokens, kv_lines = _parse_kv_cache(scenario_dir / "serve.log")
    try:
        artifact_dir = str(scenario_dir.relative_to(REPO_ROOT))
    except ValueError:
        artifact_dir = str(scenario_dir)
    world_size = int(meta["world_size"])
    phase403_output_global = float(bench["output_tok_s"])
    phase403_output_gpu = phase403_output_global / world_size
    phase401_row = phase401[scenario]
    sim_output = float(phase401_row["phase401_sim_output_tok_s_gpu"])
    sim_peak_per_iter = float(phase401_row["sim_peak_decode_reqs_per_iter"])
    sim_peak_global = sim_peak_per_iter * int(meta["dp"])
    running_ratio = (
        metrics.running_global_max / sim_peak_global if sim_peak_global > 0 else math.inf
    )
    direction = _direction(sim_output, phase403_output_gpu)

    if metrics.running_global_max < 0.8 * sim_peak_global:
        verdict = "capacity_or_occupancy_overestimated"
        target = "phase404_dp_capacity_or_occupancy_semantics"
    elif direction == "sim_over_predicts_throughput":
        verdict = "dp_ep_comm_or_sync_under_modeled"
        target = "phase404_dp_ep_comm_sync_modeling"
    elif direction == "sim_under_predicts_throughput":
        verdict = "not_dp2_overprediction_on_recollect"
        target = "phase404_reconcile_recollected_baseline"
    else:
        verdict = "matched"
        target = "phase404_no_dp2_fix_required"

    row = {
        "source": SOURCE,
        "scenario": scenario,
        "tp": int(meta["tp"]),
        "dp": int(meta["dp"]),
        "ep": int(meta["ep"]),
        "isl": int(meta["isl"]),
        "osl": int(meta["osl"]),
        "max_num_batched_tokens": int(meta["max_num_batched_tokens"]),
        "world_size": world_size,
        "ok_requests": int(bench["ok_requests"]),
        "failed_requests": int(bench["failed_requests"]),
        "phase403_output_tok_s_global": phase403_output_global,
        "phase403_output_tok_s_gpu": phase403_output_gpu,
        "phase401_sim_output_tok_s_gpu": sim_output,
        "phase401_vs_phase403_error_ratio": _safe_error_ratio(
            sim_output,
            phase403_output_gpu,
        ),
        "phase401_vs_phase403_direction": direction,
        "sim_peak_decode_reqs_per_iter": sim_peak_per_iter,
        "sim_peak_decode_reqs_global": sim_peak_global,
        "real_running_global_max": metrics.running_global_max,
        "real_running_global_mean": metrics.running_global_mean,
        "real_running_global_p50": metrics.running_global_p50,
        "real_running_vs_sim_global_ratio": running_ratio,
        "running_engine0_max": metrics.running_engine0_max,
        "running_engine1_max": metrics.running_engine1_max,
        "running_engine0_mean": metrics.running_engine0_mean,
        "running_engine1_mean": metrics.running_engine1_mean,
        "waiting_global_max": metrics.waiting_global_max,
        "preemptions_total_last": metrics.preemptions_total_last,
        "preemptions_total_delta": metrics.preemptions_total_delta,
        "gpu_cache_usage_max_pct": metrics.gpu_cache_usage_max_pct,
        "metrics_samples": metrics.samples,
        "gpu_kv_cache_tokens_per_engine": kv_tokens,
        "num_gpu_blocks": kv_tokens // 16 if kv_tokens else None,
        "kv_cache_line_numbers": kv_lines,
        "prefix_caching": bool(meta.get("prefix_caching")),
        "collection_status": "passed_dp2_metrics_recollect",
        "attribution_verdict": verdict,
        "phase404_target": target,
        "artifact_dir": artifact_dir,
        "gpu_allowed": True,
        "ssh_allowed": True,
        "runtime_modified": False,
        "perf_database": False,
        "valid_for_default": False,
        "diagnostic_only": True,
        "default_readiness": DEFAULT_READINESS,
    }
    return {field: _fmt(row.get(field)) for field in CSV_FIELDS}


def build_phase403_rows(
    *,
    raw_root: Path = DEFAULT_RAW_ROOT,
    phase401_csv: Path = PHASE401_CSV,
) -> list[dict[str, str]]:
    phase401 = _read_csv_by_scenario(phase401_csv)
    rows = [
        _scenario_row(scenario=scenario, raw_root=raw_root, phase401=phase401)
        for scenario in DP2_SCENARIOS
    ]
    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != len(DP2_SCENARIOS):
        raise ValueError("Phase403 must contain exactly two DP2 rows")
    if {row["scenario"] for row in rows} != set(DP2_SCENARIOS):
        raise ValueError("Phase403 scenario set mismatch")
    for row in rows:
        if row["dp"] != "2":
            raise ValueError("Phase403 rows must be DP2")
        if row["gpu_allowed"] != "true":
            raise ValueError("gpu_allowed")
        if row["ssh_allowed"] != "true":
            raise ValueError("ssh_allowed")
        for field in ("runtime_modified", "perf_database", "valid_for_default"):
            if row[field] != "false":
                raise ValueError(field)
        if row["diagnostic_only"] != "true":
            raise ValueError("diagnostic_only")
        if row["default_readiness"] != DEFAULT_READINESS:
            raise ValueError("default_readiness")
        if row["failed_requests"] != "0":
            raise ValueError("failed_requests")
        if not row["metrics_samples"]:
            raise ValueError("metrics_samples")


def write_phase403_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_phase403_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    lines = [
        "# Phase403 DP2 running batch",
        "",
        "Phase403 recollects only the two DP2 clean points with prefix caching off and /metrics polling. It does not change scheduler, runtime, PerfDatabase, or gates.",
        "",
        "## Verdict",
        "",
        "| scenario | real tok/s/gpu | sim tok/s/gpu | error | real running max | sim running global | verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {phase403_output_tok_s_gpu} | {phase401_sim_output_tok_s_gpu} | {phase401_vs_phase403_error_ratio} | {real_running_global_max} | {sim_peak_decode_reqs_global} | {attribution_verdict} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- If real running were far below sim running, the next target would be capacity/occupancy.",
            "- When real running is near or reaches the sim DP2 running target while throughput is still slower, the next target is DP/EP communication or synchronization not represented in the current composition.",
            "",
            "Boundary:",
            "",
            "- gpu_allowed=true and ssh_allowed=true because this phase recollects evidence.",
            "- runtime_modified=false, perf_database=false, valid_for_default=false, diagnostic_only=true, default_readiness=No-Go.",
            "",
            "Next: Phase404 should implement only the mechanism selected by this attribution result.",
            "",
        ]
    )
    return "\n".join(lines)


def write_phase403_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase403_md(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--phase401-csv", type=Path, default=PHASE401_CSV)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_phase403_rows(raw_root=args.raw_root, phase401_csv=args.phase401_csv)
    write_phase403_csv(args.csv, rows)
    write_phase403_md(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
