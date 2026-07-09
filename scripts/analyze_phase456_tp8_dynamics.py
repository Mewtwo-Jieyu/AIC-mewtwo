#!/usr/bin/env python3
"""Phase456: TP8 capacity and dynamics audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import scripts.validate_cb_simulator as validate
from aiconfigurator.sdk.backends.cb_simulator import CBSimulator
from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend


SCENARIO = "K2.5-tp8ep8-8k2k"
DEFAULT_RAW_ROOT = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase454_gpu_batch_scope/b2b_tp8_8k2k"
)
DEFAULT_EVENT_JSONL = DEFAULT_RAW_ROOT / "overhead_on/event_timing.jsonl"
DEFAULT_SERVE_LOG = DEFAULT_RAW_ROOT / "overhead_off/K2.5-tp8ep8-8k2k/serve.log"
DEFAULT_CLEAN_VALIDATE = (
    REPO_ROOT / "docs/iter_gap_investigation/phase454_clean_reference_validate.csv"
)
DEFAULT_AB_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase456_validate_ab.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase456_tp8_dynamics.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase456_tp8_dynamics.md"

BLOCK_SIZE = validate.KV_CACHE_BLOCK_SIZE
OLD_TP8_8K_TOKENS = 760_160
OLD_TP8_8K_BLOCKS = OLD_TP8_8K_TOKENS // BLOCK_SIZE

CSV_FIELDS = [
    "section",
    "scenario",
    "metric",
    "value",
    "target",
    "status",
    "source",
    "note",
]

KV_RE = re.compile(r"GPU KV cache size: (?P<tokens>[\d,]+) tokens")
LOG_RE = re.compile(
    r"Running: (?P<running>\d+) reqs, Waiting: (?P<waiting>\d+) reqs, "
    r"GPU KV cache usage: (?P<kv>[0-9.]+)%"
)


@dataclass(frozen=True)
class EventStep:
    ctx_tokens: int
    generation_requests: int
    forward_busy_ms: float
    num_tokens_padded: int

    @property
    def is_prefill(self) -> bool:
        return self.ctx_tokens > 0

    @property
    def is_mixed(self) -> bool:
        return self.ctx_tokens > 0 and self.generation_requests > 0

    @property
    def is_decode(self) -> bool:
        return self.ctx_tokens == 0 and self.generation_requests > 0


def _fmt(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def _percentile(values: Iterable[float], pct: float) -> float:
    data = sorted(values)
    if not data:
        return math.nan
    index = (len(data) - 1) * pct / 100.0
    lo = int(index)
    hi = min(lo + 1, len(data) - 1)
    frac = index - lo
    return data[lo] * (1.0 - frac) + data[hi] * frac


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    data = list(values)
    if not data:
        return {"count": 0, "mean": math.nan, "p10": math.nan, "p50": math.nan, "p90": math.nan}
    return {
        "count": len(data),
        "mean": statistics.mean(data),
        "p10": _percentile(data, 10),
        "p50": _percentile(data, 50),
        "p90": _percentile(data, 90),
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_event_steps(path: Path) -> list[EventStep]:
    steps: list[EventStep] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            steps.append(
                EventStep(
                    ctx_tokens=int(row.get("ctx_tokens") or 0),
                    generation_requests=int(row.get("generation_requests") or 0),
                    forward_busy_ms=float(row.get("forward_busy_ms") or 0.0),
                    num_tokens_padded=int(row.get("num_tokens_padded") or 0),
                )
            )
    return steps


def parse_serve_profile(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    kv_matches = [int(m.group("tokens").replace(",", "")) for m in KV_RE.finditer(text)]
    running: list[float] = []
    waiting: list[float] = []
    kv_usage: list[float] = []
    for match in LOG_RE.finditer(text):
        running.append(float(match.group("running")))
        waiting.append(float(match.group("waiting")))
        kv_usage.append(float(match.group("kv")))
    return {
        "kv_cache_tokens": kv_matches[0] if kv_matches else 0,
        "num_gpu_blocks": (kv_matches[0] // BLOCK_SIZE) if kv_matches else 0,
        "running": _summary(running),
        "waiting": _summary(waiting),
        "kv_usage": _summary(kv_usage),
        "source": _display_path(path),
    }


def event_profile(steps: list[EventStep]) -> dict[str, dict[str, float | int]]:
    mixed = [step for step in steps if step.is_mixed]
    prefill = [step for step in steps if step.is_prefill]
    decode = [step for step in steps if step.is_decode]
    total = max(len(steps), 1)
    return {
        "mixed_share": {
            "count": len(mixed),
            "mean": len(mixed) / total,
            "p10": math.nan,
            "p50": math.nan,
            "p90": math.nan,
        },
        "mixed_decode_batch": _summary(step.generation_requests for step in mixed),
        "mixed_ctx_tokens": _summary(step.ctx_tokens for step in mixed),
        "mixed_forward_busy_ms": _summary(step.forward_busy_ms for step in mixed),
        "prefill_forward_busy_ms": _summary(step.forward_busy_ms for step in prefill),
        "decode_batch": _summary(step.generation_requests for step in decode),
        "decode_forward_busy_ms": _summary(step.forward_busy_ms for step in decode),
    }


def _point(name: str = SCENARIO) -> validate.MultiConfigPoint:
    return next(point for point in validate.MULTI_CONFIG_DATA if point.name == name)


def _make_config(point: validate.MultiConfigPoint, num_gpu_blocks: int):
    config = validate._make_multi_config_cb_config(
        point,
        overlap_factor=0.0,
        ep8_per_iteration_overhead_ms=0.0,
    )
    return config.__class__(**{**config.__dict__, "num_gpu_blocks": num_gpu_blocks})


def run_sim_capacity(point: validate.MultiConfigPoint, num_gpu_blocks: int) -> dict[str, float]:
    model, database, _ = validate._load_model_and_db(
        tp=point.tp,
        dp=point.dp,
        moe_tp=point.moe_tp,
        moe_ep=point.moe_ep,
    )
    sim = CBSimulator(VLLMBackend(), model, database, _make_config(point, num_gpu_blocks))
    result = sim.run(
        isl=point.isl,
        osl=point.osl,
        concurrency=point.batch_size,
        prefix=0,
        num_gpus=point.tp,
    )
    sim_output = result.throughput_tok_s_gpu
    real_output = point.real_output_tok_s_gpu
    return {
        "sim_output_tok_s_gpu": sim_output,
        "real_output_tok_s_gpu": real_output,
        "error_ratio": max(sim_output / real_output, real_output / sim_output),
        "peak_decode_reqs": float(result.peak_decode_reqs_per_iter),
        "avg_decode_reqs": float(result.avg_decode_reqs_per_iter),
        "avg_prefill_reqs": float(result.avg_prefill_reqs_per_iter),
    }


def _add(
    rows: list[dict[str, object]],
    section: str,
    metric: str,
    value: object,
    *,
    target: str = "",
    status: str = "",
    source: str = "",
    note: str = "",
    scenario: str = SCENARIO,
) -> None:
    rows.append(
        {
            "section": section,
            "scenario": scenario,
            "metric": metric,
            "value": value,
            "target": target,
            "status": status,
            "source": source,
            "note": note,
        }
    )


def build_rows(
    *,
    serve: dict[str, object],
    events: dict[str, dict[str, float | int]],
    sim_old: dict[str, float],
    sim_current: dict[str, float],
    ab_csv: Path | None = DEFAULT_AB_CSV,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    source = str(serve["source"])
    _add(rows, "real_profile", "kv_cache_tokens", serve["kv_cache_tokens"], source=source)
    _add(rows, "real_profile", "num_gpu_blocks", serve["num_gpu_blocks"], source=source)
    for metric in ("running", "waiting", "kv_usage"):
        summary = serve[metric]
        assert isinstance(summary, dict)
        _add(
            rows,
            "real_profile",
            f"{metric}_p50",
            summary["p50"],
            source=source,
            note=f"p10={_fmt(summary['p10'])}; p90={_fmt(summary['p90'])}; n={summary['count']}",
        )
    for metric, summary in events.items():
        _add(
            rows,
            "real_fingerprint",
            metric,
            summary["mean"] if metric == "mixed_share" else summary["p50"],
            source=str(DEFAULT_EVENT_JSONL.relative_to(REPO_ROOT)),
            note=(
                f"count={summary['count']}; mean={_fmt(summary['mean'])}; "
                f"p10={_fmt(summary['p10'])}; p90={_fmt(summary['p90'])}"
            ),
        )

    current_capacity = validate._multi_config_kv_capacity(_point())
    capacity_matches = int(current_capacity.kv_cache_tokens) == int(serve["kv_cache_tokens"])
    _add(
        rows,
        "wiring",
        "validate_kv_cache_tokens",
        current_capacity.kv_cache_tokens,
        target=str(serve["kv_cache_tokens"]),
        status="pass" if capacity_matches else "fail",
        source=current_capacity.serve_log,
        note="validate must use the Phase454 no-event TP8 capacity",
    )
    _add(
        rows,
        "wiring",
        "validate_num_gpu_blocks",
        current_capacity.num_gpu_blocks,
        target=str(serve["num_gpu_blocks"]),
        status="pass" if capacity_matches else "fail",
        source=current_capacity.serve_log,
    )

    for label, sim in (("old_capacity", sim_old), ("current_capacity", sim_current)):
        status = "pass" if sim["error_ratio"] <= 1.15 else "fail"
        _add(
            rows,
            "capacity_counterfactual",
            f"{label}_error_ratio",
            sim["error_ratio"],
            target="<=1.15",
            status=status,
            note=(
                f"sim={sim['sim_output_tok_s_gpu']:.3f}; real={sim['real_output_tok_s_gpu']:.3f}; "
                f"peak_decode={sim['peak_decode_reqs']:.0f}; avg_decode={sim['avg_decode_reqs']:.3f}"
            ),
        )
    improvement = sim_old["error_ratio"] - sim_current["error_ratio"]
    _add(
        rows,
        "capacity_counterfactual",
        "error_ratio_delta",
        improvement,
        target="positive",
        status="pass" if improvement > 0 else "fail",
        note="capacity wiring fixes most TP8-8k2k overprediction but does not close the 15% gate",
    )
    verdict = (
        "tp8_capacity_wiring_improves_but_residual_remains"
        if sim_current["error_ratio"] > 1.15
        else "tp8_capacity_wiring_closes_gate"
    )
    _add(
        rows,
        "decision",
        "phase456_verdict",
        verdict,
        target="6/6 <=15%",
        status="blocked" if sim_current["error_ratio"] > 1.15 else "pass",
        note="Do not ingest TP8 B2b rows until dynamics residual is resolved.",
    )
    if ab_csv is not None and ab_csv.exists():
        with ab_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                _add(
                    rows,
                    "validate_ab",
                    row["name"],
                    float(row["current_error_ratio"]),
                    target="<=1.15",
                    status=row["classification"],
                    source=str(ab_csv.relative_to(REPO_ROOT)),
                    note=(
                        f"baseline={float(row['baseline_error_ratio']):.3f}; "
                        f"current_sim={float(row['current_sim_output_tok_s_gpu']):.3f}; "
                        f"delta={float(row['delta_error_ratio']):+.3f}"
                    ),
                    scenario=row["name"],
                )
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in CSV_FIELDS})


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    by_section: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_section.setdefault(str(row["section"]), []).append(row)
    verdict = next(row for row in rows if row["metric"] == "phase456_verdict")
    lines = [
        "# Phase456 TP8 dynamics",
        "",
        f"Verdict: `{verdict['value']}`.",
        "",
        "## Real Profile",
        "",
        "| metric | value | note |",
        "|---|---:|---|",
    ]
    for row in by_section.get("real_profile", []):
        lines.append(f"| {row['metric']} | {row['value']} | {row['note']} |")
    lines.extend(["", "## Real Fingerprint", "", "| metric | value | note |", "|---|---:|---|"])
    for row in by_section.get("real_fingerprint", []):
        lines.append(f"| {row['metric']} | {row['value']} | {row['note']} |")
    lines.extend(["", "## Capacity Counterfactual", "", "| metric | value | status | note |", "|---|---:|---|---|"])
    for row in by_section.get("capacity_counterfactual", []):
        lines.append(f"| {row['metric']} | {row['value']} | {row['status']} | {row['note']} |")
    lines.extend(
        [
            "",
            "The TP8 8k2k capacity wiring update moves sim throughput toward the Phase454 real profile, but the remaining error is still above the 15% target.",
            "The withdrawn Phase455 TP8 B2b rows stay out of PerfDB; the residual is a dynamics issue, not a per-step serving-state row issue.",
        ]
    )
    if by_section.get("validate_ab"):
        lines.extend(
            [
                "",
                "## Validate A/B",
                "",
                "| scenario | error | class | note |",
                "|---|---:|---|---|",
            ]
        )
        for row in by_section["validate_ab"]:
            lines.append(
                f"| {row['scenario']} | {float(row['value']):.3f}x | "
                f"{row['status']} | {row['note']} |"
            )
    lines.extend(
        [
            "",
            "`diagnostic_only=true valid_for_default=false perf_database=false`; Default AIC stays No-Go.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    csv_out: Path = DEFAULT_CSV,
    md_out: Path = DEFAULT_MD,
    *,
    event_jsonl: Path = DEFAULT_EVENT_JSONL,
    serve_log: Path = DEFAULT_SERVE_LOG,
) -> list[dict[str, object]]:
    serve = parse_serve_profile(serve_log)
    events = event_profile(parse_event_steps(event_jsonl))
    point = _point()
    sim_old = run_sim_capacity(point, OLD_TP8_8K_BLOCKS)
    sim_current = run_sim_capacity(point, int(validate._multi_config_num_gpu_blocks(point)))
    rows = build_rows(serve=serve, events=events, sim_old=sim_old, sim_current=sim_current)
    write_csv(csv_out, rows)
    write_markdown(md_out, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    rows = run_analysis(args.csv_out, args.md_out)
    verdict = next(row["value"] for row in rows if row["metric"] == "phase456_verdict")
    print(f"wrote {args.csv_out}")
    print(f"wrote {args.md_out}")
    print(f"verdict={verdict}")


if __name__ == "__main__":
    main()
