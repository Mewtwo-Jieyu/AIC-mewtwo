#!/usr/bin/env python3
"""Phase437: extract serving-state grid rows from collected serving traces."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import analyze_phase428_perstep_kernel_attrib as phase428  # noqa: E402


SOURCE = "phase437_serving_state_collect"
KERNEL_SOURCE = "phase437_serving_state_collect"
SCENARIO = "K2.5-tp4ep8dp2-serving-grid"
MODEL = "kimi-k2.5"
TOPOLOGY = "tp4dp2ep8"
QUANT_RUNTIME = "CompressedTensorsWNA16MarlinMoEMethod"
HIDDEN_SIZE = 7168
TOPK = 8
MOE_EP_SIZE = 8
DEFAULT_READINESS = "No-Go"
TARGET_CATEGORIES = ("ep_a2a", "moe_gemm_or_aux", "other_cuda", "collective_other")
CATEGORY_MAP = {
    "ep_a2a": "ep_a2a",
    "moe_gemm_or_aux": "moe_gemm_or_aux",
    "other_cuda": "other_cuda",
    "tp_or_dp_allreduce": "collective_other",
    "collective_other": "collective_other",
}
EXECUTE_CONTEXT_RE = re.compile(
    r"execute_context_\d+\((?P<ctx_tokens>\d+)\)_generation_\d+\((?P<gen_tokens>\d+)\)"
)

DEFAULT_ARTIFACT_DIR = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase437_serving_state_collect"
    / SCENARIO
)
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase437_serving_state_collect.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase437_serving_state_collect.md"
DEFAULT_PERFDB = (
    REPO_ROOT
    / "src/aiconfigurator/systems/data/h200_sxm/vllm/0.19.0/vllm_serving_state_perf.txt"
)

CSV_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "phase",
    "category",
    "bucket_tokens",
    "decode_batch",
    "latency_ms",
    "sample_count",
    "window_list",
    "kernel_source",
    "provenance",
    "coverage_note",
    "mechanism_verdict",
    "phase438_target",
    "gpu_allowed",
    "ssh_allowed",
    "runtime_modified",
    "perf_database",
    "valid_for_default",
    "diagnostic_only",
    "default_readiness",
]

PERFDB_FIELDS = [
    "framework",
    "version",
    "device",
    "model",
    "topology",
    "phase",
    "category",
    "kernel_source",
    "bucket_tokens",
    "decode_batch",
    "hidden_size",
    "topk",
    "moe_ep_size",
    "quant_runtime",
    "latency",
    "provenance",
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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _base_row() -> dict[str, object]:
    return {
        "source": SOURCE,
        "kernel_source": KERNEL_SOURCE,
        "provenance": "phase437_serving_state_collect_step_bucket",
        "mechanism_verdict": "serving_state_grid_measured_no_extrapolation",
        "phase438_target": "hit_audit_validate_ab_and_protocol_decision",
        "gpu_allowed": True,
        "ssh_allowed": True,
        "runtime_modified": False,
        "perf_database": True,
        "valid_for_default": False,
        "diagnostic_only": False,
        "default_readiness": DEFAULT_READINESS,
    }


def _trace_dirs(artifact_dir: Path) -> list[Path]:
    return sorted(path for path in artifact_dir.glob("prof_*") if path.is_dir())


def _window_from_prof_dir(path: Path) -> str:
    name = path.name
    if not name.startswith("prof_"):
        raise ValueError(f"unexpected profile directory name: {path}")
    return name[len("prof_") :]


class _MutableStep:
    def __init__(
        self,
        *,
        window: str,
        trace_file: str,
        step_index: int,
        start_us: float,
        end_us: float,
        wall_ms: float,
    ) -> None:
        self.window = window
        self.trace_file = trace_file
        self.step_index = step_index
        self.start_us = start_us
        self.end_us = end_us
        self.wall_ms = wall_ms
        self.ctx_tokens = 0
        self.gen_tokens = 0
        self.token_event_dur_us = -1.0
        self.category_ms: dict[str, float] = {}


def _steps_from_execute_context_events(
    events: list[phase428.TraceEvent],
    *,
    trace_name: str,
    window: str,
) -> list[phase428.RankStep]:
    step_events = [
        event
        for event in events
        if EXECUTE_CONTEXT_RE.match(event.name)
        and event.dur_us / 1000.0 >= phase428.MIN_PROFILER_STEP_MS
    ]
    step_events.sort(key=lambda event: event.ts_us)
    rank_steps: list[phase428.RankStep] = []
    for step_index, step in enumerate(step_events):
        match = EXECUTE_CONTEXT_RE.match(step.name)
        if match is None:
            continue
        category_ms: dict[str, float] = {}
        for event in events:
            if event is step or not (step.ts_us <= event.mid_us <= step.end_us):
                continue
            if event.cat not in {"kernel", "gpu_memcpy", "gpu_memset"}:
                continue
            category = phase428._category_for_event(event)
            if category in {"profiler_step", "cuda_graph_envelope"}:
                continue
            category_ms[category] = category_ms.get(category, 0.0) + event.dur_us / 1000.0
        rank_steps.append(
            phase428.RankStep(
                window=window,
                trace_file=trace_name,
                step_index=step_index,
                wall_ms=step.dur_us / 1000.0,
                ctx_tokens=int(match.group("ctx_tokens")),
                gen_tokens=int(match.group("gen_tokens")),
                category_ms=category_ms,
            )
        )
    return rank_steps


def _iter_trace_x_events(path: Path) -> object:
    """Yield Chrome trace X events without loading the whole trace file."""
    decoder = json.JSONDecoder()
    buffer = ""
    pos = 0
    in_events = False
    eof = False
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        while True:
            if not in_events:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    return
                buffer += chunk
                key_index = buffer.find('"traceEvents"')
                if key_index < 0:
                    buffer = buffer[-64:]
                    continue
                array_index = buffer.find("[", key_index)
                if array_index < 0:
                    continue
                pos = array_index + 1
                in_events = True

            while True:
                while pos < len(buffer) and buffer[pos] in " \t\r\n,":
                    pos += 1
                if pos < len(buffer) and buffer[pos] == "]":
                    return
                try:
                    raw, end = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError:
                    if eof:
                        return
                    if pos:
                        buffer = buffer[pos:]
                        pos = 0
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        eof = True
                    buffer += chunk
                    continue
                pos = end
                if pos > 1024 * 1024:
                    buffer = buffer[pos:]
                    pos = 0
                if isinstance(raw, dict) and raw.get("ph") == "X":
                    name = str(raw.get("name", ""))
                    if not name:
                        continue
                    try:
                        ts_us = float(raw["ts"])
                        dur_us = float(raw.get("dur", 0.0))
                    except (TypeError, ValueError, KeyError):
                        continue
                    if dur_us <= 0:
                        continue
                    yield phase428.TraceEvent(
                        name=name,
                        cat=str(raw.get("cat", "")),
                        ts_us=ts_us,
                        dur_us=dur_us,
                    )


def _find_steps(trace_path: Path, window: str) -> list[_MutableStep]:
    profiler_events: list[phase428.TraceEvent] = []
    execute_events: list[phase428.TraceEvent] = []
    for event in _iter_trace_x_events(trace_path):
        if event.name.startswith("ProfilerStep") and event.dur_us / 1000.0 >= phase428.MIN_PROFILER_STEP_MS:
            profiler_events.append(event)
        elif EXECUTE_CONTEXT_RE.match(event.name) and event.dur_us / 1000.0 >= phase428.MIN_PROFILER_STEP_MS:
            execute_events.append(event)
    selected = sorted(profiler_events or execute_events, key=lambda event: event.ts_us)
    steps: list[_MutableStep] = []
    for index, event in enumerate(selected):
        step = _MutableStep(
            window=window,
            trace_file=trace_path.name,
            step_index=index,
            start_us=event.ts_us,
            end_us=event.end_us,
            wall_ms=event.dur_us / 1000.0,
        )
        match = EXECUTE_CONTEXT_RE.match(event.name)
        if match is not None:
            step.ctx_tokens = int(match.group("ctx_tokens"))
            step.gen_tokens = int(match.group("gen_tokens"))
            step.token_event_dur_us = event.dur_us
        steps.append(step)
    return steps


def _find_step_for_midpoint(steps: list[_MutableStep], start_index: int, mid_us: float) -> tuple[int, _MutableStep | None]:
    index = start_index
    while index < len(steps) and steps[index].end_us < mid_us:
        index += 1
    if index < len(steps) and steps[index].start_us <= mid_us <= steps[index].end_us:
        return index, steps[index]
    return index, None


def parse_trace_rank_steps(trace_path: Path, window: str) -> list[phase428.RankStep]:
    steps = _find_steps(trace_path, window)
    if not steps:
        raise ValueError(f"missing ProfilerStep and execute_context steps in {trace_path}")
    step_index = 0
    for event in _iter_trace_x_events(trace_path):
        if event.name.startswith("ProfilerStep"):
            continue
        mid_us = event.mid_us
        step_index, step = _find_step_for_midpoint(steps, step_index, mid_us)
        if step is None:
            continue
        match = EXECUTE_CONTEXT_RE.match(event.name)
        if match is not None and event.dur_us > step.token_event_dur_us:
            step.ctx_tokens = int(match.group("ctx_tokens"))
            step.gen_tokens = int(match.group("gen_tokens"))
            step.token_event_dur_us = event.dur_us
        if event.cat not in {"kernel", "gpu_memcpy", "gpu_memset"}:
            continue
        category = phase428._category_for_event(event)
        if category in {"profiler_step", "cuda_graph_envelope"}:
            continue
        step.category_ms[category] = step.category_ms.get(category, 0.0) + event.dur_us / 1000.0
    return [
        phase428.RankStep(
            window=step.window,
            trace_file=step.trace_file,
            step_index=step.step_index,
            wall_ms=step.wall_ms,
            ctx_tokens=step.ctx_tokens,
            gen_tokens=step.gen_tokens,
            category_ms=step.category_ms,
        )
        for step in steps
    ]


def collect_rank_steps(artifact_dir: Path) -> list[phase428.RankStep]:
    if not artifact_dir.exists():
        raise ValueError(f"missing Phase437 artifact dir: {artifact_dir}")
    steps: list[phase428.RankStep] = []
    for prof_dir in _trace_dirs(artifact_dir):
        window = _window_from_prof_dir(prof_dir)
        trace_paths = sorted(path for path in prof_dir.glob("*.pt.trace.json.gz") if path.name.startswith("dp"))
        if not trace_paths:
            continue
        for trace_path in trace_paths:
            steps.extend(parse_trace_rank_steps(trace_path, window))
    if not steps:
        raise ValueError(f"no serving-state profiler steps found under {artifact_dir}")
    return steps


def _step_key(step: object, category: str) -> tuple[str, str, int, int] | None:
    ctx_tokens = int(getattr(step, "ctx_tokens"))
    gen_tokens = int(getattr(step, "gen_tokens"))
    if ctx_tokens > 0:
        # Runtime mixed-prefill query uses non_attn_tokens = prefill + decode.
        return ("mixed_prefill", category, ctx_tokens + gen_tokens, gen_tokens)
    if gen_tokens > 0:
        return ("decode", category, gen_tokens, gen_tokens)
    return None


def build_rows_from_rank_steps(
    rank_steps: list[object],
    *,
    scenario: str = SCENARIO,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, int], list[float]] = defaultdict(list)
    windows: dict[tuple[str, str, int, int], set[str]] = defaultdict(set)
    for step in rank_steps:
        window = str(getattr(step, "window"))
        category_ms = getattr(step, "category_ms")
        for raw_category, latency_ms in category_ms.items():
            category = CATEGORY_MAP.get(raw_category)
            if category not in TARGET_CATEGORIES:
                continue
            key = _step_key(step, category)
            if key is None:
                continue
            grouped[key].append(float(latency_ms))
            windows[key].add(window)

    rows: list[dict[str, object]] = []
    for (phase, category, bucket_tokens, decode_batch), values in sorted(grouped.items()):
        row = _base_row()
        row.update(
            {
                "row_type": "serving_curve",
                "scenario": scenario,
                "phase": phase,
                "category": category,
                "bucket_tokens": bucket_tokens,
                "decode_batch": decode_batch,
                "latency_ms": _mean(values),
                "sample_count": len(values),
                "window_list": ",".join(sorted(windows[(phase, category, bucket_tokens, decode_batch)])),
                "coverage_note": "measured exact step bucket; no clamp or extrapolation",
            }
        )
        rows.append(row)
    if not rows:
        raise ValueError("no target serving-state category rows extracted")
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in CSV_FIELDS})


def write_perfdb(rows: list[dict[str, object]], path: Path) -> None:
    curve_rows = [row for row in rows if row.get("row_type") == "serving_curve"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PERFDB_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in sorted(
            curve_rows,
            key=lambda r: (
                str(r["phase"]),
                str(r["category"]),
                int(r["bucket_tokens"]),
                int(r["decode_batch"]),
            ),
        ):
            writer.writerow(
                {
                    "framework": "VLLM",
                    "version": "0.19.0",
                    "device": "NVIDIA H200",
                    "model": MODEL,
                    "topology": TOPOLOGY,
                    "phase": row["phase"],
                    "category": row["category"],
                    "kernel_source": KERNEL_SOURCE,
                    "bucket_tokens": row["bucket_tokens"],
                    "decode_batch": row["decode_batch"],
                    "hidden_size": HIDDEN_SIZE,
                    "topk": TOPK,
                    "moe_ep_size": MOE_EP_SIZE,
                    "quant_runtime": QUANT_RUNTIME,
                    "latency": _fmt(row["latency_ms"]),
                    "provenance": row["provenance"],
                }
            )


def _coverage(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["phase"]), str(row["category"]))].append(row)
    out: list[dict[str, object]] = []
    for (phase, category), group in sorted(grouped.items()):
        buckets = [int(row["bucket_tokens"]) for row in group]
        batches = [int(row["decode_batch"]) for row in group]
        samples = [int(row["sample_count"]) for row in group]
        out.append(
            {
                "phase": phase,
                "category": category,
                "rows": len(group),
                "bucket_min": min(buckets),
                "bucket_max": max(buckets),
                "decode_batch_min": min(batches),
                "decode_batch_max": max(batches),
                "sample_count": sum(samples),
            }
        )
    return out


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    cov = _coverage(rows)
    verdict = "serving_state_grid_measured_no_extrapolation"
    lines = [
        "# Phase437 Serving-State Grid Collection",
        "",
        "## Verdict",
        "",
        f"- verdict: `{verdict}`.",
        f"- serving curve rows: {sum(1 for row in rows if row.get('row_type') == 'serving_curve')}.",
        "- PerfDB rows use `kernel_source=phase437_serving_state_collect`.",
        "- Query policy stays inner-only; no clamp or extrapolation was added.",
        "- Default AIC remains No-Go until A/B validation and protocol decision are clean.",
        "",
        "## Coverage",
        "",
        "| phase | category | rows | bucket min | bucket max | batch min | batch max | samples |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cov:
        lines.append(
            "| {phase} | {category} | {rows} | {bucket_min} | {bucket_max} | "
            "{decode_batch_min} | {decode_batch_max} | {sample_count} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Scope: K2.5 / tp4dp2ep8 / int4_wo / H200 / vLLM 0.19.0.",
            "- Mechanism attribution remains unresolved; this is an empirical serving-state surface.",
            "- Runtime and gate were not changed by this analyzer.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_rows(artifact_dir: Path, *, scenario: str = SCENARIO) -> list[dict[str, object]]:
    return build_rows_from_rank_steps(collect_rank_steps(artifact_dir), scenario=scenario)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--perfdb-out", type=Path, default=None)
    args = parser.parse_args()

    rows = build_rows(args.artifact_dir, scenario=args.scenario)
    write_csv(rows, args.csv)
    write_markdown(rows, args.md)
    if args.perfdb_out is not None:
        write_perfdb(rows, args.perfdb_out)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")
    if args.perfdb_out is not None:
        print(f"wrote {args.perfdb_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
