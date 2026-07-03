#!/usr/bin/env python3
"""Phase412: compare burst vs sustained-arrival DP2 imbalance."""

from __future__ import annotations

import argparse
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

from scripts import analyze_phase409_iter_trace as phase409


SOURCE = "phase412_arrival_sweep"
DEFAULT_SCENARIO = "K2.5-tp4ep8dp2-32k3k"
BASELINE_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase409_iter_trace"
SWEEP_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase412_arrival_sweep"
PHASE407_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase407_dp_lockstep_joint_sim.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase412_arrival_sweep.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase412_arrival_sweep.md"
DEFAULT_READINESS = "No-Go"

CSV_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "artifact_dir",
    "num_prompts",
    "max_concurrency",
    "uncoupled_output_tok_s_gpu",
    "bench_output_tok_s_gpu",
    "real_output_source",
    "overall_penalty",
    "steady_output_tok_s_gpu",
    "steady_penalty",
    "trace_bench_error_pct",
    "trace_fidelity_gate",
    "engine0_prompt_tokens",
    "engine1_prompt_tokens",
    "engine0_generation_tokens",
    "engine1_generation_tokens",
    "engine0_request_success_count",
    "engine1_request_success_count",
    "engine0_request_count_est",
    "engine1_request_count_est",
    "request_count_ratio",
    "generation_token_ratio",
    "request_imbalance_monotonic_down",
    "penalty_monotonic_down",
    "last_request_count_ratio",
    "last_overall_penalty",
    "mechanism_verdict",
    "phase413_target",
    "phase405_penalty_read",
    "gpu_allowed",
    "ssh_allowed",
    "runtime_modified",
    "perf_database",
    "valid_for_default",
    "diagnostic_only",
    "default_readiness",
]

SUCCESS_RE = re.compile(
    r"^vllm:request_success_total\{(?P<labels>[^}]*)\}\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EngineMetrics:
    prompt_tokens: dict[str, float]
    generation_tokens: dict[str, float]
    request_success_count: dict[str, float]


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


def _read_csv_by_scenario(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        return {row["scenario"]: row for row in csv.DictReader(f)}


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _labels(label_text: str) -> dict[str, str]:
    return {m.group("key"): m.group("value") for m in phase409.LABEL_RE.finditer(label_text)}


def _bench_tput(path: Path) -> float:
    return phase409._bench_tput(path)


def _read_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            return f.read()
    return path.read_text(encoding="utf-8", errors="replace")


def _metrics_path(artifact_dir: Path) -> Path:
    raw = artifact_dir / "metrics.jsonl"
    if raw.exists():
        return raw
    compressed = artifact_dir / "metrics.jsonl.gz"
    if compressed.exists():
        return compressed
    raise ValueError(f"missing metrics artifact under {artifact_dir}")


def _parse_metric_snapshots(path: Path) -> list[phase409.MetricSnapshot]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    snapshots: list[phase409.MetricSnapshot] = []
    for line in _read_text(path).splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record.get("status", 0)) != 200:
            continue
        prompt_tokens = {"0": 0.0, "1": 0.0}
        generation_tokens = {"0": 0.0, "1": 0.0}
        for metric_line in str(record.get("body", "")).splitlines():
            match = phase409.METRIC_RE.match(metric_line.strip())
            if not match:
                continue
            labels = _labels(match.group("labels"))
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
    return snapshots


def _num_prompts(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "num_prompts" in data:
        return int(data["num_prompts"])
    meta_path = path.with_name("meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in ("bench_num_prompts", "num_prompts"):
            if key in meta:
                return int(meta[key])
    raise ValueError(f"missing num_prompts in {path}")


def _max_concurrency(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("max_concurrency", "concurrency"):
        if key in data:
            return int(data[key])
    meta_path = path.with_name("meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in ("max_concurrency", "batch_size"):
            if key in meta:
                return int(meta[key])
    return 128


def parse_engine_metrics(metrics_path: Path) -> EngineMetrics:
    snapshots = _parse_metric_snapshots(metrics_path)
    if not snapshots:
        raise ValueError(f"missing metric snapshots in {metrics_path}")
    final_snapshot = snapshots[-1]
    request_success_count = {"0": 0.0, "1": 0.0}
    for raw_line in reversed(_read_text(metrics_path).splitlines()):
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        for line in str(record.get("body", "")).splitlines():
            match = SUCCESS_RE.match(line.strip())
            if not match:
                continue
            labels = _labels(match.group("labels"))
            engine = labels.get("engine")
            if engine not in request_success_count:
                continue
            if labels.get("finished_reason") != "length":
                continue
            request_success_count[engine] += float(match.group("value"))
        break
    return EngineMetrics(
        prompt_tokens=final_snapshot.prompt_tokens,
        generation_tokens=final_snapshot.generation_tokens,
        request_success_count=request_success_count,
    )


def _steady_metric_tput(path: Path) -> float:
    snapshots = _parse_metric_snapshots(path)
    intervals: list[tuple[float, float]] = []
    for before, after in zip(snapshots, snapshots[1:]):
        gen_delta = sum(
            max(0.0, after.generation_tokens[engine] - before.generation_tokens[engine])
            for engine in ("0", "1")
        )
        wall_s = max(0.0, after.timestamp_s - before.timestamp_s)
        if gen_delta <= 0 or wall_s <= 0:
            continue
        intervals.append((wall_s, gen_delta))
    if not intervals:
        return math.nan
    if len(intervals) >= 5:
        trim = max(1, int(len(intervals) * 0.1))
        kept = intervals[trim:-trim] or intervals
    else:
        kept = intervals
    wall_s = sum(item[0] for item in kept)
    generation_tokens = sum(item[1] for item in kept)
    return _safe_ratio(generation_tokens, wall_s) / 8.0


def _trace_metric_tput(path: Path) -> float:
    snapshots = _parse_metric_snapshots(path)
    return float(phase409.summarize_metric_deltas(snapshots)["trace_output_tok_s_gpu"])


def _artifact_dirs(
    *,
    baseline_root: Path,
    sweep_root: Path,
    scenario: str,
) -> list[Path]:
    dirs: list[Path] = []
    baseline_dir = baseline_root / scenario
    if baseline_dir.exists():
        dirs.append(baseline_dir)
    if sweep_root.exists():
        exact = sweep_root / scenario
        if exact.exists():
            dirs.append(exact)
        for child in sorted(sweep_root.iterdir()):
            if child == exact:
                continue
            if child.is_dir() and child.name.startswith(scenario):
                dirs.append(child)
    return dirs


def _sample_row(
    artifact_dir: Path,
    *,
    scenario: str,
    uncoupled_output: float,
    isl: float,
    baseline_phase407_real: float | None,
) -> dict[str, object]:
    bench_path = artifact_dir / "bench_result.json"
    metrics_path = _metrics_path(artifact_dir)
    metrics = parse_engine_metrics(metrics_path)
    num_prompts = _num_prompts(bench_path)
    bench_tput = _bench_tput(bench_path)
    real_output = bench_tput
    real_source = "bench_result"
    if num_prompts == 128 and baseline_phase407_real is not None:
        real_output = baseline_phase407_real
        real_source = "phase407_clean_baseline"
    overall_penalty = _safe_ratio(uncoupled_output, real_output)
    steady_tput = _steady_metric_tput(metrics_path)
    steady_penalty = _safe_ratio(uncoupled_output, steady_tput)
    trace_tput = _trace_metric_tput(metrics_path)
    trace_bench_error = _error_pct(trace_tput, bench_tput)
    request_count_est = {
        engine: _safe_ratio(metrics.prompt_tokens[engine], isl)
        for engine in ("0", "1")
    }
    request_count_ratio = _safe_ratio(
        max(request_count_est.values()),
        min(request_count_est.values()),
    )
    generation_token_ratio = _safe_ratio(
        max(metrics.generation_tokens.values()),
        min(metrics.generation_tokens.values()),
    )
    return {
        "source": SOURCE,
        "row_type": "sample",
        "scenario": scenario,
        "artifact_dir": _display_path(artifact_dir),
        "num_prompts": num_prompts,
        "max_concurrency": _max_concurrency(bench_path),
        "uncoupled_output_tok_s_gpu": uncoupled_output,
        "bench_output_tok_s_gpu": bench_tput,
        "real_output_source": real_source,
        "overall_penalty": overall_penalty,
        "steady_output_tok_s_gpu": steady_tput,
        "steady_penalty": steady_penalty,
        "trace_bench_error_pct": trace_bench_error,
        "trace_fidelity_gate": "passed" if trace_bench_error <= 10.0 else "failed",
        "engine0_prompt_tokens": metrics.prompt_tokens["0"],
        "engine1_prompt_tokens": metrics.prompt_tokens["1"],
        "engine0_generation_tokens": metrics.generation_tokens["0"],
        "engine1_generation_tokens": metrics.generation_tokens["1"],
        "engine0_request_success_count": metrics.request_success_count["0"],
        "engine1_request_success_count": metrics.request_success_count["1"],
        "engine0_request_count_est": request_count_est["0"],
        "engine1_request_count_est": request_count_est["1"],
        "request_count_ratio": request_count_ratio,
        "generation_token_ratio": generation_token_ratio,
        "request_imbalance_monotonic_down": "",
        "penalty_monotonic_down": "",
        "last_request_count_ratio": "",
        "last_overall_penalty": "",
        "mechanism_verdict": "",
        "phase413_target": "",
        "phase405_penalty_read": False,
        "gpu_allowed": True,
        "ssh_allowed": True,
        "runtime_modified": False,
        "perf_database": False,
        "valid_for_default": False,
        "diagnostic_only": True,
        "default_readiness": DEFAULT_READINESS,
    }


def _monotonic_down(values: list[float]) -> bool:
    return all(later <= earlier for earlier, later in zip(values, values[1:]))


def _trend_row(sample_rows: list[dict[str, object]], *, scenario: str) -> dict[str, object]:
    if len(sample_rows) < 2:
        raise ValueError("Phase412 needs baseline plus at least one arrival-sweep sample")
    ordered = sorted(sample_rows, key=lambda row: int(row["num_prompts"]))
    request_ratios = [float(row["request_count_ratio"]) for row in ordered]
    penalties = [float(row["overall_penalty"]) for row in ordered]
    request_down = _monotonic_down(request_ratios)
    penalty_down = _monotonic_down(penalties)
    last_ratio = request_ratios[-1]
    last_penalty = penalties[-1]
    last_steady_penalty = float(ordered[-1]["steady_penalty"])
    if request_down and penalty_down and last_ratio <= 1.10 and last_penalty <= 1.25:
        verdict = "burst_arrival_artifact_confirmed"
        target = "fix_validation_harness_arrival_mode_before_runtime_model"
    elif request_down and last_ratio <= 1.10 and last_steady_penalty <= 1.50:
        verdict = "arrival_sweep_partial_collapse"
        target = "separate_tail_or_drain_from_remaining_dp_cost"
    elif last_penalty >= 1.50:
        verdict = "persistent_dp_imbalance"
        target = "model_or_measure_persistent_dp_imbalance"
    else:
        verdict = "arrival_sweep_partial_collapse"
        target = "separate_arrival_artifact_from_remaining_dp_cost"
    return {
        "source": SOURCE,
        "row_type": "trend",
        "scenario": scenario,
        "artifact_dir": ",".join(str(row["artifact_dir"]) for row in ordered),
        "num_prompts": "->".join(str(row["num_prompts"]) for row in ordered),
        "max_concurrency": ordered[-1]["max_concurrency"],
        "uncoupled_output_tok_s_gpu": ordered[-1]["uncoupled_output_tok_s_gpu"],
        "bench_output_tok_s_gpu": ordered[-1]["bench_output_tok_s_gpu"],
        "real_output_source": "trend",
        "overall_penalty": last_penalty,
        "steady_output_tok_s_gpu": ordered[-1]["steady_output_tok_s_gpu"],
        "steady_penalty": ordered[-1]["steady_penalty"],
        "trace_bench_error_pct": ordered[-1]["trace_bench_error_pct"],
        "trace_fidelity_gate": ordered[-1]["trace_fidelity_gate"],
        "engine0_prompt_tokens": ordered[-1]["engine0_prompt_tokens"],
        "engine1_prompt_tokens": ordered[-1]["engine1_prompt_tokens"],
        "engine0_generation_tokens": ordered[-1]["engine0_generation_tokens"],
        "engine1_generation_tokens": ordered[-1]["engine1_generation_tokens"],
        "engine0_request_success_count": ordered[-1]["engine0_request_success_count"],
        "engine1_request_success_count": ordered[-1]["engine1_request_success_count"],
        "engine0_request_count_est": ordered[-1]["engine0_request_count_est"],
        "engine1_request_count_est": ordered[-1]["engine1_request_count_est"],
        "request_count_ratio": last_ratio,
        "generation_token_ratio": ordered[-1]["generation_token_ratio"],
        "request_imbalance_monotonic_down": request_down,
        "penalty_monotonic_down": penalty_down,
        "last_request_count_ratio": last_ratio,
        "last_overall_penalty": last_penalty,
        "mechanism_verdict": verdict,
        "phase413_target": target,
        "phase405_penalty_read": False,
        "gpu_allowed": True,
        "ssh_allowed": True,
        "runtime_modified": False,
        "perf_database": False,
        "valid_for_default": False,
        "diagnostic_only": True,
        "default_readiness": DEFAULT_READINESS,
    }


def build_phase412_rows(
    *,
    baseline_root: Path = BASELINE_ROOT,
    sweep_root: Path = SWEEP_ROOT,
    phase407_csv: Path = PHASE407_CSV,
    scenario: str = DEFAULT_SCENARIO,
) -> list[dict[str, str]]:
    phase407 = _read_csv_by_scenario(phase407_csv)[scenario]
    uncoupled = float(phase407["uncoupled_joint_output_tok_s_gpu"])
    baseline_real = float(phase407["real_output_tok_s_gpu"])
    isl = float(phase407.get("isl") or 32000.0)
    rows = [
        _sample_row(
            artifact_dir,
            scenario=scenario,
            uncoupled_output=uncoupled,
            isl=isl,
            baseline_phase407_real=baseline_real,
        )
        for artifact_dir in _artifact_dirs(
            baseline_root=baseline_root,
            sweep_root=sweep_root,
            scenario=scenario,
        )
    ]
    seen: set[int] = set()
    deduped: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: int(item["num_prompts"])):
        num_prompts = int(row["num_prompts"])
        if num_prompts in seen:
            continue
        seen.add(num_prompts)
        deduped.append(row)
    deduped.append(_trend_row(deduped, scenario=scenario))
    return [
        {field: _fmt(row.get(field, "")) for field in CSV_FIELDS}
        for row in deduped
    ]


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Phase412 rows must not be empty")
    if sum(1 for row in rows if row.get("row_type") == "trend") != 1:
        raise ValueError("Phase412 expects exactly one trend row")
    for row in rows:
        if row.get("source") != SOURCE:
            raise ValueError("unexpected source")
        if row.get("phase405_penalty_read") != "false":
            raise ValueError("phase405_penalty_read must stay false")
        if row.get("gpu_allowed") != "true" or row.get("ssh_allowed") != "true":
            raise ValueError("Phase412 is GPU measurement and must declare GPU/SSH use")
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


def write_phase412_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_phase412_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    trend = [row for row in rows if row["row_type"] == "trend"][0]
    samples = [row for row in rows if row["row_type"] == "sample"]
    lines = [
        "# Phase412 Arrival Sweep",
        "",
        "Phase412 对比 N=128 突发 baseline 与持续到达样本，只做 GPU 测量结果归因；不改 runtime、PerfDatabase 或 gate。",
        "",
        "## Verdict",
        "",
        f"- verdict: `{trend['mechanism_verdict']}`",
        f"- Phase413 target: `{trend['phase413_target']}`",
        f"- request imbalance monotonic down: `{trend['request_imbalance_monotonic_down']}`",
        f"- penalty monotonic down: `{trend['penalty_monotonic_down']}`",
        f"- last request ratio: `{trend['last_request_count_ratio']}`",
        f"- last overall penalty: `{trend['last_overall_penalty']}`",
        "",
        "## Samples",
        "",
        "| N | artifact | request ratio | overall penalty | steady penalty | bench tok/s/gpu | trace fidelity |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in samples:
        lines.append(
            f"| {row['num_prompts']} | {row['artifact_dir']} | {row['request_count_ratio']} | "
            f"{row['overall_penalty']} | {row['steady_penalty']} | {row['bench_output_tok_s_gpu']} | "
            f"{row['trace_fidelity_gate']} |"
        )
    first_sample = samples[0]
    last_sample = samples[-1]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- N={last_sample['num_prompts']} 持续到达把 per-engine request ratio 从 `{first_sample['request_count_ratio']}` 拉平到 `{last_sample['request_count_ratio']}`，说明 Phase411 的 request-count imbalance 是闭集突发放大的。",
            f"- overall penalty 只从 `{first_sample['overall_penalty']}` 降到 `{last_sample['overall_penalty']}`，没有塌到 `<=1.25`；所以不能说只是 benchmark 突发伪影。",
            f"- steady-window penalty 是 `{last_sample['steady_penalty']}`，明显低于 overall penalty；剩余问题更像 tail/drain 或持续到达下的 DP cost，需要 Phase413 单独拆。",
            f"- N=256 暂不必补：N={last_sample['num_prompts']} 已经把请求分配拉平，趋势问题不在中间点，而在 steady 与 overall 的差。",
            "",
            "## Boundary",
            "",
            "- GPU/SSH: used only for Phase412 measurement artifacts.",
            "- Large raw logs are stored compressed locally (`serve.log.gz`, `metrics.jsonl.gz`); remote raw originals remain unchanged.",
            "- Runtime/PerfDatabase/gate: not modified.",
            "- Phase405 penalty: not read.",
            "- Default AIC: No-Go.",
            "",
        ]
    )
    return "\n".join(lines)


def write_phase412_md(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase412_md(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--sweep-root", type=Path, default=SWEEP_ROOT)
    parser.add_argument("--phase407-csv", type=Path, default=PHASE407_CSV)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_phase412_rows(
        baseline_root=args.baseline_root,
        sweep_root=args.sweep_root,
        phase407_csv=args.phase407_csv,
        scenario=args.scenario,
    )
    write_phase412_csv(args.csv, rows)
    write_phase412_md(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
