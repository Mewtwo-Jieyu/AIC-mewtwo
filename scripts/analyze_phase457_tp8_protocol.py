#!/usr/bin/env python3
"""Phase457: quantify the remaining TP8 8k2k residual and protocol delta."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import analyze_phase456_tp8_dynamics as phase456  # noqa: E402

SCENARIO = phase456.SCENARIO
DEFAULT_RAW_ROOT = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase454_gpu_batch_scope/b2b_tp8_8k2k"
)
DEFAULT_OFF_BENCH = (
    DEFAULT_RAW_ROOT / "overhead_off/K2.5-tp8ep8-8k2k/bench_result.json"
)
DEFAULT_ON_BENCH = (
    DEFAULT_RAW_ROOT / "overhead_on/K2.5-tp8ep8-8k2k/bench_result.json"
)
DEFAULT_OFF_METRICS = (
    DEFAULT_RAW_ROOT / "overhead_off/K2.5-tp8ep8-8k2k/metrics.jsonl"
)
DEFAULT_AB_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase456_validate_ab.csv"
DEFAULT_PHASE456_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase456_tp8_dynamics.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase457_tp8_protocol.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase457_tp8_protocol.md"

NUM_GPUS = 8
NUM_REQUESTS = 512
TARGET_RATIO = 1.15
FINGERPRINT_TOLERANCE = 0.15
PREEMPTION_RESIDUAL_RATIO = 1.50
PREEMPTION_LINE_RE = re.compile(
    r"^vllm:num_preemptions_total\{(?P<labels>[^}]*)\}\s+(?P<value>[0-9.eE+-]+)$",
    re.MULTILINE,
)
ENGINE_RE = re.compile(r'engine="(?P<engine>[^"]+)"')

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


def _fmt(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _ratio(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return math.inf
    return max(a / b, b / a)


def load_bench_output_tok_s_gpu(path: Path, *, num_gpus: int = NUM_GPUS) -> dict[str, float]:
    bench = json.loads(path.read_text(encoding="utf-8"))
    output_tok_s = float(bench["output_tok_s"])
    return {
        "num_prompts": float(bench["num_prompts"]),
        "max_concurrency": float(bench["max_concurrency"]),
        "wall_s": float(bench["wall_s"]),
        "output_tok_s": output_tok_s,
        "output_tok_s_gpu": output_tok_s / num_gpus,
    }


def load_validate_row(path: Path, scenario: str = SCENARIO) -> dict[str, float | str]:
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["name"] == scenario:
                return {
                    "name": row["name"],
                    "real_output_tok_s_gpu": float(row["current_real_output_tok_s_gpu"]),
                    "sim_output_tok_s_gpu": float(row["current_sim_output_tok_s_gpu"]),
                    "error_ratio": float(row["current_error_ratio"]),
                }
    raise ValueError(f"scenario not found in {path}: {scenario}")


def load_phase456_metric(path: Path, metric: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["scenario"] == SCENARIO and row["metric"] == metric:
                return row
    raise ValueError(f"metric not found in {path}: {metric}")


def parse_preemption_delta(path: Path) -> dict[str, float]:
    first: dict[str, float] = {}
    last: dict[str, float] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            body = json.loads(line).get("body", "")
            for match in PREEMPTION_LINE_RE.finditer(body):
                engine_match = ENGINE_RE.search(match.group("labels"))
                engine = engine_match.group("engine") if engine_match else "unknown"
                value = float(match.group("value"))
                first.setdefault(engine, value)
                last[engine] = value
    return {engine: last[engine] - first[engine] for engine in sorted(last)}


def run_sim_preemption_count(scenario: str = SCENARIO, num_requests: int = NUM_REQUESTS) -> dict[str, float]:
    from scripts.analyze_phase451_preemption_ledger import run_sim_preemption_ledger

    ledger = run_sim_preemption_ledger(scenario=scenario, num_requests=num_requests)
    return {
        "preemptions": float(ledger["preemptions"]),
        "recompute_tokens": float(ledger["recompute_tokens"]),
        "throughput_tok_s_gpu": float(ledger["throughput_tok_s_gpu"]),
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
    validate_row: dict[str, float | str],
    off_bench: dict[str, float],
    on_bench: dict[str, float],
    preemption_delta: dict[str, float],
    sim_preemption: dict[str, float],
    phase456_csv: Path = DEFAULT_PHASE456_CSV,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    real_n128 = float(validate_row["real_output_tok_s_gpu"])
    sim_current = float(validate_row["sim_output_tok_s_gpu"])
    n512_off = float(off_bench["output_tok_s_gpu"])
    n512_on = float(on_bench["output_tok_s_gpu"])
    n128_error = _ratio(sim_current, real_n128)
    n512_off_error = _ratio(sim_current, n512_off)
    n512_on_error = _ratio(sim_current, n512_on)
    protocol_gain = n512_off / real_n128

    _add(
        rows,
        "protocol_projection",
        "n128_reference_real_output_tok_s_gpu",
        real_n128,
        source=_display_path(DEFAULT_AB_CSV),
        note="current validate reference; N=128 closed-set",
    )
    _add(
        rows,
        "protocol_projection",
        "current_sim_output_tok_s_gpu",
        sim_current,
        source=_display_path(DEFAULT_AB_CSV),
        note="Phase456 current sim after TP8 capacity wiring",
    )
    _add(
        rows,
        "protocol_projection",
        "n512_proxy_overhead_off_output_tok_s_gpu",
        n512_off,
        source=_display_path(DEFAULT_OFF_BENCH),
        note=(
            f"N={off_bench['num_prompts']:.0f}; C={off_bench['max_concurrency']:.0f}; "
            "B2b overhead-off proxy, not a replacement reference"
        ),
    )
    _add(
        rows,
        "protocol_projection",
        "n512_proxy_overhead_on_output_tok_s_gpu",
        n512_on,
        source=_display_path(DEFAULT_ON_BENCH),
        note="used only as patch-overhead cross-check",
    )
    _add(
        rows,
        "protocol_projection",
        "error_vs_n128_reference",
        n128_error,
        target=f"<={TARGET_RATIO}",
        status="pass" if n128_error <= TARGET_RATIO else "fail",
        note="existing mixed-protocol validation point",
    )
    _add(
        rows,
        "protocol_projection",
        "error_vs_n512_proxy_overhead_off",
        n512_off_error,
        target=f"<={TARGET_RATIO}",
        status="pass" if n512_off_error <= TARGET_RATIO else "fail",
        note="predicts the vanilla N=512 recollect payoff",
    )
    _add(
        rows,
        "protocol_projection",
        "error_vs_n512_proxy_overhead_on",
        n512_on_error,
        target=f"<={TARGET_RATIO}",
        status="pass" if n512_on_error <= TARGET_RATIO else "fail",
        note="patch on/off agrees if close to overhead_off",
    )
    _add(
        rows,
        "protocol_projection",
        "n512_protocol_gain_vs_n128",
        protocol_gain,
        target=">1",
        status="pass" if protocol_gain > 1 else "fail",
        note="real reference throughput gain from N=128 to N=512 proxy",
    )

    mixed_batch = load_phase456_metric(phase456_csv, "mixed_decode_batch")
    decode_batch = load_phase456_metric(phase456_csv, "decode_batch")
    running = load_phase456_metric(phase456_csv, "running_p50")
    sim_note = load_phase456_metric(phase456_csv, "current_capacity_error_ratio")["note"]
    sim_avg_match = re.search(r"avg_decode=(?P<value>[0-9.]+)", sim_note)
    sim_avg_decode = float(sim_avg_match.group("value")) if sim_avg_match else math.nan
    real_mixed_batch = float(mixed_batch["value"])
    batch_delta = abs(sim_avg_decode - real_mixed_batch) / real_mixed_batch
    _add(
        rows,
        "dynamics_check",
        "real_mixed_decode_batch_p50",
        real_mixed_batch,
        source=mixed_batch["source"],
        note=mixed_batch["note"],
    )
    _add(
        rows,
        "dynamics_check",
        "sim_avg_decode_batch",
        sim_avg_decode,
        target=f"within {FINGERPRINT_TOLERANCE:.0%} of real mixed p50",
        status="pass" if batch_delta <= FINGERPRINT_TOLERANCE else "fail",
        source=_display_path(phase456_csv),
        note=f"relative_delta={batch_delta:.3f}",
    )
    _add(
        rows,
        "dynamics_check",
        "real_decode_batch_p50",
        float(decode_batch["value"]),
        source=decode_batch["source"],
        note=decode_batch["note"],
    )
    _add(
        rows,
        "dynamics_check",
        "real_running_p50",
        float(running["value"]),
        source=running["source"],
        note=running["note"],
    )
    total_preemptions = sum(preemption_delta.values())
    sim_preemptions = float(sim_preemption["preemptions"])
    if total_preemptions > 0:
        preemption_ratio = sim_preemptions / total_preemptions
    elif sim_preemptions == 0:
        preemption_ratio = 1.0
    else:
        preemption_ratio = math.inf
    _add(
        rows,
        "dynamics_check",
        "real_preemption_delta",
        total_preemptions,
        target="compare with sim",
        status="informational",
        source=_display_path(DEFAULT_OFF_METRICS),
        note="; ".join(f"engine {k}: {v:.0f}" for k, v in preemption_delta.items()),
    )
    _add(
        rows,
        "dynamics_check",
        "sim_preemptions",
        sim_preemptions,
        target=f"real×{PREEMPTION_RESIDUAL_RATIO:.2f}",
        status="pass" if preemption_ratio <= PREEMPTION_RESIDUAL_RATIO else "warn",
        source="current CBSimulator trace",
        note=(
            f"real_preemptions={total_preemptions:.0f}; ratio={preemption_ratio:.3f}; "
            f"sim_recompute_tokens={sim_preemption['recompute_tokens']:.0f}; "
            f"sim_trace_tput={sim_preemption['throughput_tok_s_gpu']:.3f}"
        ),
    )

    model_residual_visible = preemption_ratio > PREEMPTION_RESIDUAL_RATIO
    protocol_closes_gate = n512_off_error <= TARGET_RATIO and batch_delta <= FINGERPRINT_TOLERANCE
    verdict = (
        "tp8_protocol_unification_closes_gate_with_secondary_preemption_residual"
        if protocol_closes_gate and model_residual_visible
        else "tp8_protocol_unification_dominates_recollect_vanilla_n512"
        if protocol_closes_gate
        else "tp8_model_residual_requires_step2_before_recollect"
    )
    if protocol_closes_gate and model_residual_visible:
        verdict_note = (
            "N=512 protocol unification is enough to reach <=15% by proxy; "
            "sim preemption remains higher than real and is stored as a secondary residual."
        )
    elif protocol_closes_gate:
        verdict_note = (
            "Step2 model fix is not triggered; Phase454 N=512 proxy predicts <=15%, "
            "but vanilla Step3 recollect is still required before reference replacement."
        )
    else:
        verdict_note = "Do not recollect as a reference replacement until the model residual is localized."
    _add(
        rows,
        "decision",
        "phase457_verdict",
        verdict,
        target="6/6 <=15%",
        status="pass" if protocol_closes_gate else "blocked",
        note=verdict_note,
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
    verdict = next(row for row in rows if row["metric"] == "phase457_verdict")
    lines = [
        "# Phase457 TP8 protocol residual",
        "",
        f"Verdict: `{verdict['value']}`.",
        "",
        "## Protocol Projection",
        "",
        "| metric | value | status | note |",
        "|---|---:|---|---|",
    ]
    for row in by_section.get("protocol_projection", []):
        lines.append(f"| {row['metric']} | {row['value']} | {row['status']} | {row['note']} |")
    lines.extend(["", "## Dynamics Check", "", "| metric | value | status | note |", "|---|---:|---|---|"])
    for row in by_section.get("dynamics_check", []):
        lines.append(f"| {row['metric']} | {row['value']} | {row['status']} | {row['note']} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "| metric | value | status | note |",
            "|---|---|---|---|",
            f"| {verdict['metric']} | {verdict['value']} | {verdict['status']} | {verdict['note']} |",
            "",
            "The Phase454 N=512 overhead-off run is a proxy, not a clean reference. It is only used to decide whether Phase457 Step2 is needed before vanilla recollection.",
            "`diagnostic_only=true valid_for_default=false perf_database=false`; Default AIC stays No-Go until the vanilla N=512 TP8 reference is collected and validate is rerun.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    csv_out: Path = DEFAULT_CSV,
    md_out: Path = DEFAULT_MD,
    *,
    off_bench_path: Path = DEFAULT_OFF_BENCH,
    on_bench_path: Path = DEFAULT_ON_BENCH,
    metrics_path: Path = DEFAULT_OFF_METRICS,
    ab_csv: Path = DEFAULT_AB_CSV,
) -> list[dict[str, object]]:
    rows = build_rows(
        validate_row=load_validate_row(ab_csv),
        off_bench=load_bench_output_tok_s_gpu(off_bench_path),
        on_bench=load_bench_output_tok_s_gpu(on_bench_path),
        preemption_delta=parse_preemption_delta(metrics_path),
        sim_preemption=run_sim_preemption_count(),
    )
    write_csv(csv_out, rows)
    write_markdown(md_out, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    rows = run_analysis(args.csv_out, args.md_out)
    verdict = next(row["value"] for row in rows if row["metric"] == "phase457_verdict")
    print(f"wrote {args.csv_out}")
    print(f"wrote {args.md_out}")
    print(f"verdict={verdict}")


if __name__ == "__main__":
    main()
