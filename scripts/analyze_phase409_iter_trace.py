#!/usr/bin/env python3
"""Phase409: parse DP2 per-iteration trace and measure lockstep penalty."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = "phase409_iter_trace"
DEFAULT_SCENARIO = "K2.5-tp4ep8dp2-32k3k"
DEFAULT_RAW_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase409_iter_trace"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase409_iter_trace.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase409_iter_trace.md"
PHASE407_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase407_dp_lockstep_joint_sim.csv"
DEFAULT_READINESS = "No-Go"

CSV_FIELDS = [
    "source",
    "scenario",
    "artifact_dir",
    "iteration_detail_rows",
    "aligned_steps",
    "engine0_steps",
    "engine1_steps",
    "total_unpadded_tokens",
    "total_padded_tokens",
    "direct_token_lockstep_penalty",
    "needed_penalty",
    "penalty_error_pct",
    "penalty_gate",
    "trace_output_tok_s_gpu",
    "bench_output_tok_s_gpu",
    "trace_bench_error_pct",
    "trace_fidelity_gate",
    "co_prefill_decode_steps",
    "co_prefill_decode_step_share",
    "mixed_prefill_steps",
    "decode_only_steps",
    "mean_pair_elapsed_ms",
    "p95_pair_elapsed_ms",
    "mean_decode_elapsed_ms",
    "mean_co_prefill_elapsed_ms",
    "elapsed_lift_ms",
    "cudagraph_rows_found",
    "cudagraph_weighted_penalty",
    "real_phase_direct_penalty_source",
    "mechanism_verdict",
    "phase410_target",
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
class IterationStep:
    engine: str
    iteration: int
    ctx_requests: int
    ctx_tokens: int
    generation_requests: int
    generation_tokens: int
    elapsed_ms: float

    @property
    def total_tokens(self) -> int:
        return self.ctx_tokens + self.generation_tokens

    @property
    def has_prefill(self) -> bool:
        return self.ctx_tokens > 0 or self.ctx_requests > 0

    @property
    def has_generation(self) -> bool:
        return self.generation_tokens > 0 or self.generation_requests > 0


@dataclass(frozen=True)
class MetricSnapshot:
    timestamp_s: float
    prompt_tokens: dict[str, float]
    generation_tokens: dict[str, float]


ITER_RE = re.compile(
    r"EngineCore_(?:DP)?(?P<engine>[01]).*?"
    r"Iteration\((?P<iteration>\d+)\):\s+"
    r"(?P<ctx_req>\d+)\s+context requests,\s+"
    r"(?P<ctx_tok>\d+)\s+context tokens,\s+"
    r"(?P<gen_req>\d+)\s+generation requests,\s+"
    r"(?P<gen_tok>\d+)\s+generation tokens,\s+"
    r"iteration elapsed time:\s+"
    r"(?P<elapsed>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\s+ms"
)

METRIC_RE = re.compile(
    r"^(?P<name>vllm:(?:prompt_tokens_total|generation_tokens_total))"
    r"\{(?P<labels>[^}]*)\}\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)$",
    re.IGNORECASE,
)
LABEL_RE = re.compile(r'(?P<key>[A-Za-z_][A-Za-z0-9_]*)="(?P<value>[^"]*)"')

CG_ROW_RE = re.compile(
    r"^\|\s*(?P<unpadded>\d+)\s*\|\s*"
    r"(?P<padded>\d+)\s*\|\s*"
    r"(?P<paddings>\d+)\s*\|\s*"
    r"(?P<mode>[^|]+?)\s*\|\s*"
    r"(?P<count>\d+)\s*\|$"
)


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


def parse_iteration_steps(path: Path) -> list[IterationStep]:
    steps: list[IterationStep] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ITER_RE.search(line)
        if not match:
            continue
        steps.append(
            IterationStep(
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


def parse_metric_snapshots(path: Path) -> list[MetricSnapshot]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    snapshots: list[MetricSnapshot] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record.get("status", 0)) != 200:
            continue
        ts = datetime.fromisoformat(record["ts"]).timestamp()
        prompt_tokens = {"0": 0.0, "1": 0.0}
        generation_tokens = {"0": 0.0, "1": 0.0}
        for metric_line in str(record.get("body", "")).splitlines():
            match = METRIC_RE.match(metric_line.strip())
            if not match:
                continue
            labels = {m.group("key"): m.group("value") for m in LABEL_RE.finditer(match.group("labels"))}
            engine = labels.get("engine")
            if engine not in prompt_tokens:
                continue
            value = float(match.group("value"))
            if match.group("name") == "vllm:prompt_tokens_total":
                prompt_tokens[engine] = value
            else:
                generation_tokens[engine] = value
        snapshots.append(
            MetricSnapshot(
                timestamp_s=ts,
                prompt_tokens=prompt_tokens,
                generation_tokens=generation_tokens,
            )
        )
    return snapshots


def summarize_metric_deltas(snapshots: list[MetricSnapshot]) -> dict[str, float | int]:
    active_intervals = 0
    active_start_s: float | None = None
    active_end_s: float | None = None
    total_prompt_tokens = 0.0
    total_generation_tokens = 0.0
    total_unpadded = 0.0
    total_padded = 0.0
    co_prefill_decode_intervals = 0
    mixed_prefill_intervals = 0
    decode_only_intervals = 0

    for before, after in zip(snapshots, snapshots[1:]):
        prompt_delta = {
            engine: max(0.0, after.prompt_tokens[engine] - before.prompt_tokens[engine])
            for engine in ("0", "1")
        }
        generation_delta = {
            engine: max(0.0, after.generation_tokens[engine] - before.generation_tokens[engine])
            for engine in ("0", "1")
        }
        tokens0 = prompt_delta["0"] + generation_delta["0"]
        tokens1 = prompt_delta["1"] + generation_delta["1"]
        if tokens0 + tokens1 <= 0:
            continue

        active_intervals += 1
        if active_start_s is None:
            active_start_s = before.timestamp_s
        active_end_s = after.timestamp_s
        total_prompt_tokens += prompt_delta["0"] + prompt_delta["1"]
        total_generation_tokens += generation_delta["0"] + generation_delta["1"]
        total_unpadded += tokens0 + tokens1
        total_padded += 2.0 * max(tokens0, tokens1)

        any_prompt = prompt_delta["0"] > 0 or prompt_delta["1"] > 0
        any_generation = generation_delta["0"] > 0 or generation_delta["1"] > 0
        cross_prefill_decode = (
            (prompt_delta["0"] > 0 and generation_delta["1"] > 0)
            or (prompt_delta["1"] > 0 and generation_delta["0"] > 0)
        )
        if cross_prefill_decode:
            co_prefill_decode_intervals += 1
        if any_prompt and any_generation:
            mixed_prefill_intervals += 1
        if not any_prompt and any_generation:
            decode_only_intervals += 1

    if active_intervals == 0 or active_start_s is None or active_end_s is None:
        raise ValueError("metrics do not contain active prompt/generation token deltas")

    wall_s = max(0.0, active_end_s - active_start_s)
    trace_tput_gpu = _safe_ratio(total_generation_tokens, wall_s) / 8.0
    return {
        "active_intervals": active_intervals,
        "total_prompt_tokens": total_prompt_tokens,
        "total_generation_tokens": total_generation_tokens,
        "total_unpadded_tokens": total_unpadded,
        "total_padded_tokens": total_padded,
        "direct_token_lockstep_penalty": _safe_ratio(total_padded, total_unpadded),
        "trace_output_tok_s_gpu": trace_tput_gpu,
        "co_prefill_decode_steps": co_prefill_decode_intervals,
        "co_prefill_decode_step_share": co_prefill_decode_intervals / active_intervals,
        "mixed_prefill_steps": mixed_prefill_intervals,
        "decode_only_steps": decode_only_intervals,
    }


def parse_cudagraph_rows(path: Path) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = CG_ROW_RE.match(line.strip())
        if not match:
            continue
        mode = match.group("mode").strip()
        if mode == "Runtime Mode":
            continue
        rows.append(
            {
                "unpadded_tokens": int(match.group("unpadded")),
                "padded_tokens": int(match.group("padded")),
                "paddings": int(match.group("paddings")),
                "runtime_mode": mode,
                "count": int(match.group("count")),
            }
        )
    return rows


def align_iteration_pairs(steps: list[IterationStep]) -> list[tuple[IterationStep, IterationStep]]:
    by_engine: dict[str, dict[int, IterationStep]] = {"0": {}, "1": {}}
    for step in steps:
        if step.engine in by_engine:
            by_engine[step.engine][step.iteration] = step
    shared = sorted(set(by_engine["0"]) & set(by_engine["1"]))
    return [(by_engine["0"][idx], by_engine["1"][idx]) for idx in shared]


def summarize_iteration_pairs(pairs: list[tuple[IterationStep, IterationStep]]) -> dict[str, float | int]:
    if not pairs:
        raise ValueError("no aligned DP iteration pairs")
    total_unpadded = 0.0
    total_padded = 0.0
    pair_elapsed: list[float] = []
    decode_elapsed: list[float] = []
    co_prefill_elapsed: list[float] = []
    co_prefill_decode_steps = 0
    mixed_prefill_steps = 0
    decode_only_steps = 0
    total_generation_tokens = 0

    for step0, step1 in pairs:
        tokens0 = step0.total_tokens
        tokens1 = step1.total_tokens
        total_unpadded += tokens0 + tokens1
        total_padded += 2 * max(tokens0, tokens1)
        elapsed = max(step0.elapsed_ms, step1.elapsed_ms)
        pair_elapsed.append(elapsed)
        total_generation_tokens += step0.generation_tokens + step1.generation_tokens

        prefill_count = int(step0.has_prefill) + int(step1.has_prefill)
        decode_count = int(step0.has_generation) + int(step1.has_generation)
        if prefill_count == 0:
            decode_only_steps += 1
            decode_elapsed.append(elapsed)
        elif prefill_count == 1 and decode_count >= 1:
            co_prefill_decode_steps += 1
            co_prefill_elapsed.append(elapsed)
        else:
            mixed_prefill_steps += 1

    wall_s = sum(pair_elapsed) / 1000.0
    trace_tput_gpu = _safe_ratio(total_generation_tokens, wall_s) / 8.0
    return {
        "aligned_steps": len(pairs),
        "total_unpadded_tokens": total_unpadded,
        "total_padded_tokens": total_padded,
        "direct_token_lockstep_penalty": _safe_ratio(total_padded, total_unpadded),
        "trace_output_tok_s_gpu": trace_tput_gpu,
        "co_prefill_decode_steps": co_prefill_decode_steps,
        "co_prefill_decode_step_share": co_prefill_decode_steps / len(pairs),
        "mixed_prefill_steps": mixed_prefill_steps,
        "decode_only_steps": decode_only_steps,
        "mean_pair_elapsed_ms": _mean(pair_elapsed),
        "p95_pair_elapsed_ms": _percentile(pair_elapsed, 0.95),
        "mean_decode_elapsed_ms": _mean(decode_elapsed),
        "mean_co_prefill_elapsed_ms": _mean(co_prefill_elapsed),
        "elapsed_lift_ms": _mean(co_prefill_elapsed) - _mean(decode_elapsed)
        if decode_elapsed and co_prefill_elapsed
        else math.nan,
    }


def _bench_tput(path: Path) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "output_tokens_per_second_per_gpu",
        "output_throughput_per_gpu",
        "output_tok_s_per_gpu",
    ):
        if key in data:
            return float(data[key])
    if "output_tokens_per_second" in data:
        return float(data["output_tokens_per_second"]) / 8.0
    if "output_tok_s" in data:
        meta_path = path.with_name("meta.json")
        world_size = 8.0
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            world_size = float(meta.get("world_size", world_size))
        return float(data["output_tok_s"]) / world_size
    raise ValueError(f"missing output tok/s/gpu in {path}")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _cudagraph_weighted_penalty(rows: list[dict[str, int | str]]) -> float:
    if not rows:
        return math.nan
    padded = 0.0
    unpadded = 0.0
    for row in rows:
        count = int(row["count"])
        padded += int(row["padded_tokens"]) * count
        unpadded += int(row["unpadded_tokens"]) * count
    return _safe_ratio(padded, unpadded)


def build_phase409_rows(
    *,
    raw_root: Path = DEFAULT_RAW_ROOT,
    phase407_csv: Path = PHASE407_CSV,
    scenario: str = DEFAULT_SCENARIO,
) -> list[dict[str, str]]:
    artifact_dir = raw_root / scenario
    serve_log = artifact_dir / "serve.log"
    steps = parse_iteration_steps(serve_log)
    pairs = align_iteration_pairs(steps)
    iteration_summary = summarize_iteration_pairs(pairs)
    metric_summary: dict[str, float | int] | None = None
    metric_path = artifact_dir / "metrics.jsonl"
    metric_snapshots = parse_metric_snapshots(metric_path)
    if metric_snapshots:
        metric_summary = summarize_metric_deltas(metric_snapshots)
    cudagraph_rows = parse_cudagraph_rows(serve_log)
    phase407 = _read_csv_by_scenario(phase407_csv)[scenario]
    uncoupled = float(phase407["uncoupled_joint_output_tok_s_gpu"])
    real = float(phase407["real_output_tok_s_gpu"])
    needed_penalty = _safe_ratio(uncoupled, real)
    summary = metric_summary or iteration_summary
    direct_penalty = float(summary["direct_token_lockstep_penalty"])
    penalty_error = _error_pct(direct_penalty, needed_penalty)
    bench_tput = _bench_tput(artifact_dir / "bench_result.json")
    trace_tput = float(summary["trace_output_tok_s_gpu"])
    fidelity_error = _error_pct(trace_tput, bench_tput)
    penalty_gate = "passed" if penalty_error <= 10.0 else "failed"
    fidelity_gate = "passed" if fidelity_error <= 10.0 else "failed"
    direct_penalty_source = (
        "metrics_prompt_generation_delta_lower_bound"
        if metric_summary
        else "serve_log_iteration_details"
    )
    if penalty_gate == "passed" and fidelity_gate == "passed":
        verdict = "real_phase_asymmetry_max_sufficient"
        phase410 = "runtime_two_engine_asymmetry_max_coupling"
    elif (
        direct_penalty_source == "metrics_prompt_generation_delta_lower_bound"
        and fidelity_gate == "passed"
    ):
        verdict = "metrics_lower_bound_below_needed_per_step_padded_absent"
        phase410 = "collect_per_step_padded_tokens_or_recheck_missing_dp_cost"
    elif fidelity_gate == "failed":
        verdict = "trace_fidelity_failed"
        phase410 = "fix_trace_capture_before_runtime_model"
    else:
        verdict = "real_phase_asymmetry_max_insufficient"
        phase410 = "recheck_prefill_occupancy_or_missing_dp_cost"

    row = {
        "source": SOURCE,
        "scenario": scenario,
        "artifact_dir": _display_path(artifact_dir),
        "iteration_detail_rows": len(steps),
        "aligned_steps": summary.get("active_intervals", iteration_summary["aligned_steps"]),
        "engine0_steps": sum(1 for step in steps if step.engine == "0"),
        "engine1_steps": sum(1 for step in steps if step.engine == "1"),
        "total_unpadded_tokens": summary["total_unpadded_tokens"],
        "total_padded_tokens": summary["total_padded_tokens"],
        "direct_token_lockstep_penalty": direct_penalty,
        "needed_penalty": needed_penalty,
        "penalty_error_pct": penalty_error,
        "penalty_gate": penalty_gate,
        "trace_output_tok_s_gpu": trace_tput,
        "bench_output_tok_s_gpu": bench_tput,
        "trace_bench_error_pct": fidelity_error,
        "trace_fidelity_gate": fidelity_gate,
        "co_prefill_decode_steps": summary["co_prefill_decode_steps"],
        "co_prefill_decode_step_share": summary["co_prefill_decode_step_share"],
        "mixed_prefill_steps": summary["mixed_prefill_steps"],
        "decode_only_steps": summary["decode_only_steps"],
        "mean_pair_elapsed_ms": iteration_summary["mean_pair_elapsed_ms"],
        "p95_pair_elapsed_ms": iteration_summary["p95_pair_elapsed_ms"],
        "mean_decode_elapsed_ms": iteration_summary["mean_decode_elapsed_ms"],
        "mean_co_prefill_elapsed_ms": iteration_summary["mean_co_prefill_elapsed_ms"],
        "elapsed_lift_ms": iteration_summary["elapsed_lift_ms"],
        "cudagraph_rows_found": len(cudagraph_rows),
        "cudagraph_weighted_penalty": _cudagraph_weighted_penalty(cudagraph_rows),
        "real_phase_direct_penalty_source": direct_penalty_source,
        "mechanism_verdict": verdict,
        "phase410_target": phase410,
        "phase405_penalty_read": False,
        "gpu_allowed": True,
        "ssh_allowed": True,
        "runtime_modified": False,
        "perf_database": False,
        "valid_for_default": False,
        "diagnostic_only": True,
        "default_readiness": DEFAULT_READINESS,
    }
    return [{field: _fmt(row.get(field, "")) for field in CSV_FIELDS}]


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 1:
        raise ValueError("Phase409 expects exactly one scenario row")
    for row in rows:
        if row.get("source") != SOURCE:
            raise ValueError("unexpected source")
        if row.get("phase405_penalty_read") != "false":
            raise ValueError("phase405_penalty_read must stay false")
        if row.get("gpu_allowed") != "true" or row.get("ssh_allowed") != "true":
            raise ValueError("Phase409 is a GPU/SSH measurement phase")
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


def write_phase409_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_phase409_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    row = rows[0]
    return "\n".join(
        [
            "# Phase409 Iter Trace",
            "",
            "Phase409 是测量-only。它只跑一个 DP2 32k3k GPU 点，开启 vLLM iteration-detail logging 与 cudagraph_metrics，不改 runtime、PerfDatabase 或 gate。",
            "",
            "## Verdict",
            "",
            f"- verdict: `{row['mechanism_verdict']}`",
            f"- direct penalty source: `{row['real_phase_direct_penalty_source']}`",
            f"- measured token penalty: `{row['direct_token_lockstep_penalty']}`",
            f"- needed penalty from Phase407 uncoupled/real: `{row['needed_penalty']}`",
            f"- penalty gate: `{row['penalty_gate']}`",
            f"- trace fidelity gate: `{row['trace_fidelity_gate']}`",
            "",
            "## Trace Summary",
            "",
            "| scenario | active intervals | trace tok/s/gpu | bench tok/s/gpu | fidelity error | co-prefill share | cudagraph rows | Phase410 target |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
            f"| {row['scenario']} | {row['aligned_steps']} | {row['trace_output_tok_s_gpu']} | {row['bench_output_tok_s_gpu']} | {row['trace_bench_error_pct']}% | {row['co_prefill_decode_step_share']} | {row['cudagraph_rows_found']} | {row['phase410_target']} |",
            "",
            "## Interpretation",
            "",
            "- metrics 的 output tok/s/gpu 与 bench 只差上表的 fidelity error，说明这次采集主窗口可信。",
            "- `cudagraph_rows_found=0`，本次 vLLM 没有落出可解析的 per-step padded token 行；因此 measured token penalty 是 metrics 2 秒采样 delta 的真实相位下界，不是逐 step padded 真值。",
            "- 这个下界是 `1.186159`，明显低于 Phase407 所需 `1.887133`；所以当前证据不能支持“asymmetry + max 已精确解释 1.887”。",
            "- Phase410 不能直接落 runtime 耦合；先要拿到 per-step padded token，或转查缺失 DP cost。",
            "",
            "## Boundary",
            "",
            "- GPU/SSH: used for measurement only.",
            "- Runtime/PerfDatabase/gate: not modified.",
            "- Phase405 penalty: not read.",
            "- Default AIC: No-Go.",
            "",
        ]
    )


def write_phase409_md(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase409_md(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--phase407-csv", type=Path, default=PHASE407_CSV)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_phase409_rows(
        raw_root=args.raw_root,
        phase407_csv=args.phase407_csv,
        scenario=args.scenario,
    )
    write_phase409_csv(args.csv, rows)
    write_phase409_md(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
