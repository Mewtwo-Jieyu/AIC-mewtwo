#!/usr/bin/env python3
"""Validate and summarize Phase463 fixed-shape streaming artifacts."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any


PROM_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+0-9.eE]+)$"
)
LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')
ITER_RE = re.compile(
    r"EngineCore(?:_DP(?P<engine>\d+))?.*?"
    r"Iteration\((?P<iteration>\d+)\):\s+"
    r"(?P<context_requests>\d+)\s+context requests,\s+"
    r"(?P<context_tokens>\d+)\s+context tokens,\s+"
    r"(?P<generation_requests>\d+)\s+generation requests,\s+"
    r"(?P<generation_tokens>\d+)\s+generation tokens,\s+"
    r"iteration elapsed time:\s+(?P<elapsed_ms>[0-9.]+)\s+ms"
)

SeriesKey = tuple[str, tuple[tuple[str, str], ...]]
PHASE463_DIAGNOSTIC_SCENARIO = "K2.5-tp4ep8dp2-32k3k-c64-diagnostic"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("missing_samples")
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse_prometheus_text(text: str) -> dict[SeriesKey, float]:
    values: dict[SeriesKey, float] = {}
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = PROM_RE.match(stripped)
        if not match:
            continue
        labels = tuple(sorted(LABEL_RE.findall(match.group("labels") or "")))
        key = (match.group("name"), labels)
        if key in values:
            raise ValueError(f"duplicate_metric_series:{key}")
        value = float(match.group("value"))
        if not math.isfinite(value):
            raise ValueError(f"nonfinite_metric:{key}")
        values[key] = value
    return values


def _labels_match(labels: tuple[tuple[str, str], ...], expected: dict[str, str]) -> bool:
    lookup = dict(labels)
    return all(lookup.get(key) == value for key, value in expected.items())


def _series_deltas(
    before: dict[SeriesKey, float],
    after: dict[SeriesKey, float],
    metric_name: str,
    labels: dict[str, str] | None = None,
) -> dict[tuple[tuple[str, str], ...], float]:
    labels = labels or {}
    before_matches = {
        series_labels: value
        for (name, series_labels), value in before.items()
        if name == metric_name and _labels_match(series_labels, labels)
    }
    after_matches = {
        series_labels: value
        for (name, series_labels), value in after.items()
        if name == metric_name and _labels_match(series_labels, labels)
    }
    if not after_matches:
        raise ValueError(f"missing_metric:{metric_name}")
    missing_after = set(before_matches) - set(after_matches)
    if missing_after:
        raise ValueError(f"metric_series_disappeared:{metric_name}:{sorted(missing_after)}")
    deltas: dict[tuple[tuple[str, str], ...], float] = {}
    for series_labels, after_value in after_matches.items():
        delta = after_value - before_matches.get(series_labels, 0.0)
        if delta < 0:
            raise ValueError(f"counter_reset:{metric_name}:{series_labels}")
        deltas[series_labels] = delta
    return deltas


def _counter_delta(
    before: dict[SeriesKey, float],
    after: dict[SeriesKey, float],
    metric_name: str,
    labels: dict[str, str] | None = None,
) -> float:
    return sum(_series_deltas(before, after, metric_name, labels).values())


def _histogram_quantile(cumulative: dict[float, float], quantile: float) -> float:
    if math.inf not in cumulative:
        raise ValueError("histogram_missing_inf_bucket")
    count = cumulative[math.inf]
    if count <= 0:
        raise ValueError("histogram_empty")
    target = count * quantile
    previous_bound = 0.0
    previous_count = 0.0
    finite_buckets = sorted((bound, value) for bound, value in cumulative.items() if math.isfinite(bound))
    for bound, bucket_count in finite_buckets:
        if bucket_count < previous_count:
            raise ValueError("histogram_not_cumulative")
        if bucket_count >= target:
            width = bound - previous_bound
            population = bucket_count - previous_count
            if population <= 0:
                return bound
            return previous_bound + width * (target - previous_count) / population
        previous_bound = bound
        previous_count = bucket_count
    if not finite_buckets:
        raise ValueError("histogram_missing_finite_bucket")
    return finite_buckets[-1][0]


def _histogram_summary(
    before: dict[SeriesKey, float],
    after: dict[SeriesKey, float],
    base_name: str,
) -> dict[str, float]:
    count = _counter_delta(before, after, f"{base_name}_count")
    total = _counter_delta(before, after, f"{base_name}_sum")
    bucket_deltas = _series_deltas(before, after, f"{base_name}_bucket")
    cumulative: dict[float, float] = defaultdict(float)
    for labels, delta in bucket_deltas.items():
        le = dict(labels).get("le")
        if le is None:
            raise ValueError(f"histogram_bucket_missing_le:{base_name}")
        cumulative[math.inf if le == "+Inf" else float(le)] += delta
    if not math.isclose(cumulative.get(math.inf, -1.0), count, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"histogram_count_mismatch:{base_name}")
    return {
        "count": count,
        "mean": total / count if count else math.nan,
        "p50": _histogram_quantile(cumulative, 0.50),
        "p90": _histogram_quantile(cumulative, 0.90),
        "p99": _histogram_quantile(cumulative, 0.99),
    }


def summarize_service_metrics(
    *,
    before_text: str,
    after_text: str,
    expected_requests: int,
    expected_prompt_tokens: int,
    expected_generation_tokens: int,
    output_len: int,
) -> dict[str, float]:
    before = parse_prometheus_text(before_text)
    after = parse_prometheus_text(after_text)
    request_success = _counter_delta(
        before,
        after,
        "vllm:request_success_total",
        {"finished_reason": "length"},
    )
    prompt_tokens = _counter_delta(before, after, "vllm:prompt_tokens_total")
    generation_tokens = _counter_delta(before, after, "vllm:generation_tokens_total")
    expected = {
        "request_success": (request_success, expected_requests),
        "prompt_tokens": (prompt_tokens, expected_prompt_tokens),
        "generation_tokens": (generation_tokens, expected_generation_tokens),
    }
    for name, (actual, wanted) in expected.items():
        if not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"{name}_mismatch:{actual}!={wanted}")

    histograms = {
        "ttft": _histogram_summary(before, after, "vllm:time_to_first_token_seconds"),
        "itl": _histogram_summary(before, after, "vllm:inter_token_latency_seconds"),
        "tpot": _histogram_summary(before, after, "vllm:request_time_per_output_token_seconds"),
        "e2e": _histogram_summary(before, after, "vllm:e2e_request_latency_seconds"),
    }
    expected_counts = {
        "ttft": expected_requests,
        "itl": expected_requests * (output_len - 1),
        "tpot": expected_requests,
        "e2e": expected_requests,
    }
    for name, wanted in expected_counts.items():
        actual = histograms[name]["count"]
        if not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"{name}_count_mismatch:{actual}!={wanted}")

    result: dict[str, float] = {
        "request_success": request_success,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
    }
    for name, summary in histograms.items():
        result[f"{name}_count"] = summary["count"]
        for statistic in ("mean", "p50", "p90", "p99"):
            result[f"{name}_{statistic}_ms"] = summary[statistic] * 1000.0
    return result


def validate_metrics_continuity(path: Path, *, max_gap_s: float) -> dict[str, float]:
    timestamps: list[datetime] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != 200:
            raise ValueError(f"metrics_sampling_failed:{row.get('status')}")
        timestamps.append(datetime.fromisoformat(str(row["ts"])))
    if len(timestamps) < 2:
        raise ValueError("metrics_sampling_too_short")
    gaps = [(right - left).total_seconds() for left, right in zip(timestamps, timestamps[1:])]
    if any(gap <= 0 or gap > max_gap_s for gap in gaps):
        raise ValueError(f"metrics_sampling_gap:{max(gaps)}")
    return {"sample_count": float(len(timestamps)), "max_gap_s": max(gaps)}


def summarize_iterations(path: Path) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ITER_RE.search(line)
        if not match:
            continue
        rows.append(
            {
                "context_requests": float(match.group("context_requests")),
                "context_tokens": float(match.group("context_tokens")),
                "generation_requests": float(match.group("generation_requests")),
                "generation_tokens": float(match.group("generation_tokens")),
                "elapsed_ms": float(match.group("elapsed_ms")),
            }
        )
    if not rows:
        raise ValueError(f"missing_iteration_rows:{path}")
    elapsed = [row["elapsed_ms"] for row in rows]
    return {
        "iteration_count": float(len(rows)),
        "context_requests": sum(row["context_requests"] for row in rows),
        "context_tokens": sum(row["context_tokens"] for row in rows),
        "generation_requests": sum(row["generation_requests"] for row in rows),
        "generation_tokens": sum(row["generation_tokens"] for row in rows),
        "iteration_elapsed_mean_ms": statistics.mean(elapsed),
        "iteration_elapsed_p50_ms": _percentile(elapsed, 50.0),
        "iteration_elapsed_p90_ms": _percentile(elapsed, 90.0),
        "iteration_elapsed_p99_ms": _percentile(elapsed, 99.0),
    }


def summarize_gpu_telemetry(
    path: Path,
    *,
    expected_gpus: int,
    max_gap_s: float,
) -> dict[str, float]:
    rows: list[tuple[datetime, int, tuple[float, float, float]]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if len(row) != 5:
                raise ValueError(f"gpu_telemetry_bad_row:{row}")
            timestamp = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
            gpu_index = int(row[1].strip())
            values = tuple(float(value.strip()) for value in row[2:])
            if not all(math.isfinite(value) for value in values):
                raise ValueError("gpu_telemetry_nonfinite")
            rows.append((timestamp, gpu_index, values))
    if len(rows) % expected_gpus != 0:
        raise ValueError(f"gpu_telemetry_incomplete_round:{len(rows)}%{expected_gpus}")
    rounds = [rows[index : index + expected_gpus] for index in range(0, len(rows), expected_gpus)]
    if len(rounds) < 2:
        raise ValueError("gpu_telemetry_too_short")
    timestamps: list[datetime] = []
    samples: list[tuple[float, float, float]] = []
    expected_indices = set(range(expected_gpus))
    for sample_round in rounds:
        indices = {row[1] for row in sample_round}
        if indices != expected_indices:
            raise ValueError(f"gpu_index_mismatch:{sorted(indices)}!={sorted(expected_indices)}")
        round_timestamps = [row[0] for row in sample_round]
        if (max(round_timestamps) - min(round_timestamps)).total_seconds() > 1.0:
            raise ValueError("gpu_round_timestamp_spread")
        timestamps.append(min(round_timestamps))
        samples.extend(row[2] for row in sample_round)
    gaps = [(right - left).total_seconds() for left, right in zip(timestamps, timestamps[1:])]
    if any(gap <= 0 or gap > max_gap_s for gap in gaps):
        raise ValueError(f"gpu_sampling_gap:{max(gaps)}")
    utilization = [sample[0] for sample in samples]
    power = [sample[1] for sample in samples]
    memory = [sample[2] for sample in samples]
    return {
        "sample_count": float(len(timestamps)),
        "max_gap_s": max(gaps),
        "gpu_util_mean_pct": statistics.mean(utilization),
        "gpu_util_p90_pct": _percentile(utilization, 90.0),
        "power_mean_w": statistics.mean(power),
        "power_p90_w": _percentile(power, 90.0),
        "memory_max_mib": max(memory),
    }


def canary_gate(
    *,
    nonstream_output_tok_s: float,
    stream_output_tok_s: float,
    max_delta_pct: float = 2.0,
) -> dict[str, float | bool]:
    if nonstream_output_tok_s <= 0 or stream_output_tok_s <= 0:
        raise ValueError("canary_nonpositive_throughput")
    delta_pct = abs(stream_output_tok_s / nonstream_output_tok_s - 1.0) * 100.0
    if delta_pct > max_delta_pct:
        raise ValueError(f"stream_canary_throughput_delta:{delta_pct:.6f}>{max_delta_pct}")
    return {"delta_pct": delta_pct, "passed": True}


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        raise ValueError(f"nonpositive_comparison_denominator:{denominator}")
    return numerator / denominator


def build_comparison_row(
    real: dict[str, Any],
    sim: dict[str, float],
    *,
    frozen_real_output_tok_s_gpu: float | None,
) -> dict[str, Any]:
    throughput_ratio = _ratio(
        float(sim["output_tok_s_gpu"]),
        float(real["real_output_tok_s_gpu"]),
    )
    frozen_drift_pct = None
    if frozen_real_output_tok_s_gpu is not None:
        frozen_drift_pct = (
            _ratio(
                float(real["real_output_tok_s_gpu"]),
                frozen_real_output_tok_s_gpu,
            )
            - 1.0
        ) * 100.0
    return {
        **real,
        "sim_output_tok_s_gpu": float(sim["output_tok_s_gpu"]),
        "throughput_sim_over_real": throughput_ratio,
        "throughput_abs_error_pct": abs(throughput_ratio - 1.0) * 100.0,
        "throughput_gate": (
            "diagnostic"
            if real["role"] != "formal"
            else "pass" if abs(throughput_ratio - 1.0) <= 0.15 else "fail"
        ),
        "sim_ttft_ms": float(sim["ttft_ms"]),
        "ttft_sim_over_real": _ratio(
            float(sim["ttft_ms"]),
            float(real["service_ttft_mean_ms"]),
        ),
        "sim_tpot_ms": float(sim["tpot_ms"]),
        "tpot_sim_over_real": _ratio(
            float(sim["tpot_ms"]),
            float(real["service_tpot_mean_ms"]),
        ),
        "sim_e2e_ms": float(sim["e2e_ms"]),
        "e2e_sim_over_real": _ratio(
            float(sim["e2e_ms"]),
            float(real["service_e2e_mean_ms"]),
        ),
        "latency_gate": "baseline_only",
        "frozen_real_output_tok_s_gpu": frozen_real_output_tok_s_gpu,
        "frozen_real_drift_pct": frozen_drift_pct,
        "sim_avg_prefill_reqs_per_iter": float(sim["avg_prefill_reqs_per_iter"]),
        "sim_avg_decode_reqs_per_iter": float(sim["avg_decode_reqs_per_iter"]),
        "real_avg_context_reqs_per_logged_iteration": _ratio(
            float(real["context_requests"]),
            float(real["iteration_count"]),
        ),
        "real_avg_generation_reqs_per_logged_iteration": _ratio(
            float(real["generation_requests"]),
            float(real["iteration_count"]),
        ),
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
        "default_readiness": "No-Go",
    }


def write_comparison_outputs(
    rows: list[dict[str, Any]],
    *,
    csv_path: Path,
    md_path: Path,
    artifact_root: Path,
    canary_delta_pct: float,
) -> None:
    if not rows:
        raise ValueError("missing_comparison_rows")
    fieldnames = list(rows[0])
    if any(set(row) != set(fieldnames) for row in rows):
        raise ValueError("comparison_row_schema_mismatch")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    formal_rows = [row for row in rows if row["role"] == "formal"]
    passed = sum(row["throughput_gate"] == "pass" for row in formal_rows)
    failed = [row["scenario"] for row in formal_rows if row["throughput_gate"] == "fail"]
    max_frozen_drift = max(
        abs(float(row["frozen_real_drift_pct"]))
        for row in formal_rows
        if row["frozen_real_drift_pct"] is not None
    )
    lines = [
        "# Phase463 six-point latency recollect",
        "",
        f"- Remote artifact: `{artifact_root}`",
        f"- Streaming canary throughput delta: `{canary_delta_pct:.4f}%` (limit `2.0000%`)",
        f"- Formal throughput gate: `{passed}/{len(formal_rows)}` passed; failed: `{';'.join(failed)}`.",
        f"- Maximum frozen-real throughput drift: `{max_frozen_drift:.4f}%`.",
        "- Throughput gate: simulator/real absolute error must be at most `15%`.",
        "- Latency truth: vLLM Prometheus histogram `_sum/_count`; streaming client timing is cross-check evidence.",
        "- Latency gate: `baseline_only`; TTFT, TPOT, and E2E are comparison evidence, not readiness gates. Percentiles remain in the CSV.",
        "- Evidence boundary: `diagnostic_only=true`, `valid_for_default=false`, `perf_database=false`.",
        "- Default AIC 仍为 No-Go.",
        "",
        "## Throughput",
        "",
        "| Scenario | Real output tok/s/GPU | Sim output tok/s/GPU | Absolute error | Gate | Frozen drift |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        frozen_drift = row["frozen_real_drift_pct"]
        lines.append(
            "| {scenario} | {real:.4f} | {sim:.4f} | {error:.2f}% | {gate} | {drift} |".format(
                scenario=row["scenario"],
                real=float(row["real_output_tok_s_gpu"]),
                sim=float(row["sim_output_tok_s_gpu"]),
                error=float(row["throughput_abs_error_pct"]),
                gate=row["throughput_gate"],
                drift="n/a" if frozen_drift is None else f"{float(frozen_drift):.3f}%",
            )
        )
    lines.extend(
        [
            "",
            "## Mean latency",
            "",
            "| Scenario | TTFT real/sim/ratio (ms) | TPOT real/sim/ratio (ms) | E2E real/sim/ratio (ms) |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {scenario} | {ttft_real:.2f} / {ttft_sim:.2f} / {ttft_ratio:.3f}x | {tpot_real:.2f} / {tpot_sim:.2f} / {tpot_ratio:.3f}x | {e2e_real:.2f} / {e2e_sim:.2f} / {e2e_ratio:.3f}x |".format(
                scenario=row["scenario"],
                ttft_real=float(row["service_ttft_mean_ms"]),
                ttft_sim=float(row["sim_ttft_ms"]),
                ttft_ratio=float(row["ttft_sim_over_real"]),
                tpot_real=float(row["service_tpot_mean_ms"]),
                tpot_sim=float(row["sim_tpot_ms"]),
                tpot_ratio=float(row["tpot_sim_over_real"]),
                e2e_real=float(row["service_e2e_mean_ms"]),
                e2e_sim=float(row["sim_e2e_ms"]),
                e2e_ratio=float(row["e2e_sim_over_real"]),
            )
        )
    lines.extend(
        [
            "",
            "## Runtime diagnostics",
            "",
            "| Scenario | GPU util mean | Power mean (W) | Real context/gen reqs per logged iter | Sim prefill/decode reqs per iter |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {scenario} | {gpu:.2f}% | {power:.2f} | {real_context:.4f} / {real_generation:.4f} | {sim_prefill:.4f} / {sim_decode:.4f} |".format(
                scenario=row["scenario"],
                gpu=float(row["gpu_util_mean_pct"]),
                power=float(row["power_mean_w"]),
                real_context=float(row["real_avg_context_reqs_per_logged_iteration"]),
                real_generation=float(row["real_avg_generation_reqs_per_logged_iteration"]),
                sim_prefill=float(row["sim_avg_prefill_reqs_per_iter"]),
                sim_decode=float(row["sim_avg_decode_reqs_per_iter"]),
            )
        )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_validation_module() -> Any:
    path = Path(__file__).with_name("validate_cb_simulator.py")
    module_name = "phase463_validate_cb_simulator"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot_load_validation_module:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def collect_simulator_metrics() -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    validate = _load_validation_module()
    formal_points = list(validate.MULTI_CONFIG_DATA)
    diagnostic_source = next(
        point for point in formal_points if point.name == "K2.5-tp4ep8dp2-32k3k"
    )
    diagnostic_point = replace(
        diagnostic_source,
        name=PHASE463_DIAGNOSTIC_SCENARIO,
        batch_size=64,
    )
    point_specs = [(point, point) for point in formal_points]
    point_specs.append((diagnostic_point, diagnostic_source))
    backend = validate.VLLMBackend()
    loaded: dict[tuple[int, int, int, int], tuple[Any, Any]] = {}
    metrics: dict[str, dict[str, float]] = {}
    for point, config_source in point_specs:
        key = (point.tp, point.dp, point.moe_tp, point.moe_ep)
        if key not in loaded:
            model, database, _ = validate._load_model_and_db(
                tp=point.tp,
                dp=point.dp,
                moe_tp=point.moe_tp,
                moe_ep=point.moe_ep,
            )
            loaded[key] = (model, database)
        model, database = loaded[key]
        cb_config = validate._make_official_validation_cb_config(
            config_source,
            overlap_factor=0.0,
            ep8_per_iteration_overhead_ms=validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
        )
        with validate._official_validation_metric_scope(point):
            summary = backend.run_agg(
                model,
                database,
                validate.RuntimeConfig(
                    batch_size=point.batch_size,
                    isl=point.isl,
                    osl=point.osl,
                ),
                ctx_tokens=point.max_num_batched_tokens,
                database_mode=validate.common.DatabaseMode.HYBRID,
                method="cb_sim",
                cb_config=cb_config,
            )
        result = summary.get_result_dict()
        per_ops = summary.get_per_ops_data()
        if result is None or per_ops is None:
            raise ValueError(f"missing_simulator_result:{point.name}")
        scheduling = per_ops.get("cb_sim_scheduling")
        if not isinstance(scheduling, dict):
            raise ValueError(f"missing_simulator_scheduling:{point.name}")
        metrics[point.name] = {
            "output_tok_s_gpu": float(result["tokens/s/gpu"]),
            "ttft_ms": float(result["ttft"]),
            "tpot_ms": float(result["tpot"]),
            "e2e_ms": float(result["request_latency"]),
            "avg_prefill_reqs_per_iter": float(
                scheduling["avg_prefill_reqs_per_iter"]
            ),
            "avg_decode_reqs_per_iter": float(
                scheduling["avg_decode_reqs_per_iter"]
            ),
        }
    frozen = {point.name: point.real_output_tok_s_gpu for point in formal_points}
    return metrics, frozen


def build_artifact_comparison_rows(artifact_root: Path) -> list[dict[str, Any]]:
    simulator, frozen = collect_simulator_metrics()
    real: dict[str, dict[str, Any]] = {}
    for path in artifact_root.glob("*/scenario_analysis.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        scenario = str(row["scenario"])
        if scenario in real:
            raise ValueError(f"duplicate_scenario_analysis:{scenario}")
        real[scenario] = row
    missing = set(simulator) - set(real)
    if missing:
        raise ValueError(f"missing_scenario_analysis:{sorted(missing)}")
    return [
        build_comparison_row(
            real[name],
            simulator[name],
            frozen_real_output_tok_s_gpu=frozen.get(name),
        )
        for name in simulator
    ]


def _require_equal(name: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise ValueError(f"{name}_mismatch:{actual}!={expected}")


def analyze_scenario(path: Path) -> dict[str, Any]:
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    bench = json.loads((path / "bench_result.json").read_text(encoding="utf-8"))
    requests = int(meta["bench_num_prompts"])
    input_len = int(meta["isl"])
    output_len = int(meta["osl"])
    expected_prompt_tokens = requests * input_len
    expected_generation_tokens = requests * output_len
    _require_equal("ok_requests", int(bench["ok_requests"]), requests)
    _require_equal("failed_requests", int(bench["failed_requests"]), 0)
    _require_equal("client_prompt_tokens", int(bench["total_prompt_tokens"]), expected_prompt_tokens)
    _require_equal(
        "client_generation_tokens",
        int(bench["total_completion_tokens"]),
        expected_generation_tokens,
    )
    _require_equal(
        "client_total_tokens",
        int(bench["total_tokens"]),
        expected_prompt_tokens + expected_generation_tokens,
    )
    if bool(bench.get("stream")) != bool(meta["stream"]):
        raise ValueError("stream_mode_mismatch")

    service = summarize_service_metrics(
        before_text=(path / "metrics_before.prom").read_text(encoding="utf-8"),
        after_text=(path / "metrics_after.prom").read_text(encoding="utf-8"),
        expected_requests=requests,
        expected_prompt_tokens=expected_prompt_tokens,
        expected_generation_tokens=expected_generation_tokens,
        output_len=output_len,
    )
    metrics = validate_metrics_continuity(path / "metrics.jsonl", max_gap_s=6.5)
    gpu = summarize_gpu_telemetry(
        path / "gpu.csv",
        expected_gpus=int(meta["world_size"]),
        max_gap_s=6.5,
    )
    iterations = summarize_iterations(path / "serve.log")
    result: dict[str, Any] = {
        "scenario": meta["name"],
        "role": meta["role"],
        "stream": bool(meta["stream"]),
        "tp": int(meta["tp"]),
        "dp": int(meta["dp"]),
        "world_size": int(meta["world_size"]),
        "input_len": input_len,
        "output_len": output_len,
        "max_num_batched_tokens": int(meta["max_num_batched_tokens"]),
        "concurrency": int(meta["batch_size"]),
        "num_requests": requests,
        "real_output_tok_s": float(bench["output_tok_s"]),
        "real_output_tok_s_gpu": float(bench["output_tok_s"]) / int(meta["world_size"]),
        "real_total_tok_s": float(bench["total_tok_s"]),
        "real_total_tok_s_gpu": float(bench["total_tok_s"]) / int(meta["world_size"]),
        "client_e2e_mean_ms": float(bench["mean_latency_ms"]),
        "client_e2e_p50_ms": float(bench["p50_latency_ms"]),
        "client_e2e_p90_ms": float(bench["p90_latency_ms"]),
        "client_e2e_p99_ms": float(bench["p99_latency_ms"]),
        "service_ttft_mean_ms": service["ttft_mean_ms"],
        "service_ttft_p50_ms": service["ttft_p50_ms"],
        "service_ttft_p90_ms": service["ttft_p90_ms"],
        "service_ttft_p99_ms": service["ttft_p99_ms"],
        "service_tpot_mean_ms": service["tpot_mean_ms"],
        "service_tpot_p50_ms": service["tpot_p50_ms"],
        "service_tpot_p90_ms": service["tpot_p90_ms"],
        "service_tpot_p99_ms": service["tpot_p99_ms"],
        "service_itl_mean_ms": service["itl_mean_ms"],
        "service_itl_p50_ms": service["itl_p50_ms"],
        "service_itl_p90_ms": service["itl_p90_ms"],
        "service_itl_p99_ms": service["itl_p99_ms"],
        "service_e2e_mean_ms": service["e2e_mean_ms"],
        "service_e2e_p50_ms": service["e2e_p50_ms"],
        "service_e2e_p90_ms": service["e2e_p90_ms"],
        "service_e2e_p99_ms": service["e2e_p99_ms"],
        "metrics_sample_count": int(metrics["sample_count"]),
        "metrics_max_gap_s": metrics["max_gap_s"],
        "gpu_sample_count": int(gpu["sample_count"]),
        **gpu,
        **iterations,
    }
    if bool(meta["stream"]):
        for metric in ("ttft", "tpot"):
            for statistic in ("mean", "p50", "p90", "p99"):
                result[f"client_{metric}_{statistic}_ms"] = float(
                    bench[f"{statistic}_{metric}_ms"]
                )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scenario-dir", type=Path)
    mode.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--md", type=Path)
    parser.add_argument("--raw-artifact-root", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.scenario_dir is not None:
        result = analyze_scenario(args.scenario_dir)
        rendered = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.output_json is not None or args.csv is None or args.md is None:
        raise ValueError("artifact_report_requires_csv_and_md_only")
    rows = build_artifact_comparison_rows(args.artifact_root)
    canary = json.loads((args.artifact_root / "canary_gate.json").read_text(encoding="utf-8"))
    write_comparison_outputs(
        rows,
        csv_path=args.csv,
        md_path=args.md,
        artifact_root=args.raw_artifact_root or args.artifact_root,
        canary_delta_pct=float(canary["delta_pct"]),
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "csv": str(args.csv),
                "md": str(args.md),
                "default_readiness": "No-Go",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
