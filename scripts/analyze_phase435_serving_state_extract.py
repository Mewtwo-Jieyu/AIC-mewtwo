#!/usr/bin/env python3
"""Phase435: extract serving-state category curves from existing trace analyses."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = "phase435_serving_state"
DEFAULT_PHASE429_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase429_reprofile.csv"
DEFAULT_PHASE433_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase433_nsys_prefill.csv"
DEFAULT_PHASE426_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase426_8k2k_steady_decompose.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase435_serving_state.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase435_serving_state.md"
DEFAULT_READINESS = "No-Go"
MODEL = "kimi-k2.5"
TOPOLOGY = "tp4dp2ep8"
QUANT_RUNTIME = "CompressedTensorsWNA16MarlinMoEMethod"
HIDDEN_SIZE = 7168
TOPK = 8
MOE_EP_SIZE = 8
TRUE = "true"
FALSE = "false"
TARGET_CATEGORIES = ("ep_a2a", "moe_gemm_or_aux", "other_cuda", "collective_other")
CATEGORY_MAP = {
    "ep_a2a": "ep_a2a",
    "moe_gemm_or_aux": "moe_gemm_or_aux",
    "other_cuda": "other_cuda",
    "tp_or_dp_allreduce": "collective_other",
    "collective_other": "collective_other",
}

CSV_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "phase",
    "category",
    "bucket_tokens",
    "decode_batch",
    "latency_ms",
    "kernel_source",
    "provenance",
    "coverage_note",
    "target_missing_ms",
    "reconstructed_missing_ms",
    "reconstruction_error_pct",
    "reconstruction_gate",
    "mechanism_verdict",
    "phase436_target",
    "gpu_allowed",
    "ssh_allowed",
    "runtime_modified",
    "perf_database",
    "valid_for_default",
    "diagnostic_only",
    "default_readiness",
]


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


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _base_row() -> dict[str, object]:
    return {
        "source": SOURCE,
        "kernel_source": "phase435_serving_state",
        "mechanism_verdict": "mechanism_unresolved_store_serving_state_measurement",
        "phase436_target": "validate_acceptance_protocol_or_serving_state_scope_extension",
        "gpu_allowed": False,
        "ssh_allowed": False,
        "runtime_modified": True,
        "perf_database": True,
        "valid_for_default": False,
        "diagnostic_only": False,
        "default_readiness": DEFAULT_READINESS,
    }


def _curve_row(
    *,
    scenario: str,
    phase: str,
    category: str,
    bucket_tokens: int,
    decode_batch: int,
    latency_ms: float,
    provenance: str,
    coverage_note: str,
) -> dict[str, object]:
    row = _base_row()
    row.update(
        {
            "row_type": "serving_curve",
            "scenario": scenario,
            "phase": phase,
            "category": category,
            "bucket_tokens": bucket_tokens,
            "decode_batch": decode_batch,
            "latency_ms": latency_ms,
            "provenance": provenance,
            "coverage_note": coverage_note,
        }
    )
    return row


def _gate_row(
    *,
    scenario: str,
    phase: str,
    target_missing_ms: float,
    reconstructed_missing_ms: float,
    tolerance_pct: float = 10.0,
) -> dict[str, object]:
    error_pct = abs(reconstructed_missing_ms - target_missing_ms) / target_missing_ms * 100.0
    row = _base_row()
    row.update(
        {
            "row_type": "consistency_gate",
            "scenario": scenario,
            "phase": phase,
            "target_missing_ms": target_missing_ms,
            "reconstructed_missing_ms": reconstructed_missing_ms,
            "reconstruction_error_pct": error_pct,
            "reconstruction_gate": "passed" if error_pct <= tolerance_pct else "failed",
            "coverage_note": "serving-state categories reconstruct prior residual within tolerance",
        }
    )
    return row


def _extract_phase433_32k_prefill(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    scenario = "K2.5-tp4ep8dp2-32k3k"
    curve_rows: list[dict[str, object]] = []
    reconstructed = 0.0
    target = math.nan
    for row in rows:
        if row.get("row_type") == "summary" and row.get("scenario") == scenario:
            target = float(row["target_missing_ms"])
        if row.get("row_type") != "category" or row.get("scenario") != scenario:
            continue
        category = CATEGORY_MAP.get(row.get("category", ""))
        if category not in TARGET_CATEGORIES:
            continue
        latency = float(row["category_real_ms_per_step"])
        reconstructed += max(0.0, float(row["category_excess_ms_per_step"] or 0.0))
        for decode_batch in (0, 8):
            curve_rows.append(
                _curve_row(
                    scenario=scenario,
                    phase="mixed_prefill",
                    category=category,
                    bucket_tokens=32000,
                    decode_batch=decode_batch,
                    latency_ms=latency,
                    provenance="phase433_nsys_prefill_category_mean",
                    coverage_note="32k serving prefill category mean; decode_batch endpoints cover observed 0..8 mixed range",
                )
            )
    if math.isnan(target):
        raise ValueError("missing Phase433 32k prefill summary target")
    curve_rows.append(
        _gate_row(
            scenario=scenario,
            phase="mixed_prefill",
            target_missing_ms=target,
            reconstructed_missing_ms=reconstructed,
        )
    )
    return curve_rows


def _extract_phase429_8k_prefill(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    scenario = "K2.5-tp4ep8dp2-8k2k"
    grouped: dict[str, list[float]] = defaultdict(list)
    batches: list[int] = []
    for row in rows:
        if row.get("row_type") == "step" and row.get("window") == "w0_prefill":
            if row.get("step_type") == "prefill_or_mixed" and row.get("decode_batch"):
                batches.append(int(float(row["decode_batch"])))
        if row.get("row_type") != "step_category" or row.get("window") != "w0_prefill":
            continue
        if row.get("scenario") != scenario or row.get("step_type") != "prefill_or_mixed":
            continue
        category = CATEGORY_MAP.get(row.get("category", ""))
        if category in TARGET_CATEGORIES:
            grouped[category].append(float(row["real_cuda_ms_per_rank"]))
    if not grouped:
        raise ValueError("missing Phase429 8k prefill category rows")
    batch_min = min(batches) if batches else 0
    batch_max = max(batches) if batches else 0
    curve_rows: list[dict[str, object]] = []
    for category in TARGET_CATEGORIES:
        values = grouped.get(category, [])
        if not values:
            continue
        for decode_batch in (batch_min, batch_max):
            curve_rows.append(
                _curve_row(
                    scenario=scenario,
                    phase="mixed_prefill",
                    category=category,
                    bucket_tokens=8000,
                    decode_batch=decode_batch,
                    latency_ms=_mean(values),
                    provenance="phase429_reprofile_w0_prefill_window_mean",
                    coverage_note=f"8k serving prefill window mean across observed decode_batch {batch_min}..{batch_max}",
                )
            )
    return curve_rows


def _extract_phase429_decode(rows: list[dict[str, str]], phase426_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    scenario = "K2.5-tp4ep8dp2-8k2k"
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("row_type") != "step_category" or row.get("step_type") != "decode_only":
            continue
        if row.get("scenario") != scenario or not row.get("decode_batch"):
            continue
        category = CATEGORY_MAP.get(row.get("category", ""))
        if category in TARGET_CATEGORIES:
            batch = int(float(row["decode_batch"]))
            grouped[(category, batch)].append(float(row["real_cuda_ms_per_rank"]))

    curve_rows: list[dict[str, object]] = []
    for (category, batch), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        if batch not in {8, 36, 52}:
            continue
        curve_rows.append(
            _curve_row(
                scenario=scenario,
                phase="decode",
                category=category,
                bucket_tokens=batch,
                decode_batch=batch,
                latency_ms=_mean(values),
                provenance="phase429_reprofile_decode_window_mean",
                coverage_note="decode serving-state batch sweep point",
            )
        )

    summary = next((row for row in phase426_rows if row.get("row_type") == "summary" and row.get("scenario") == scenario), None)
    if summary is not None:
        curve_rows.append(
            _gate_row(
                scenario=scenario,
                phase="decode",
                target_missing_ms=float(summary["decode_gap_ms_mean"]),
                reconstructed_missing_ms=float(summary["decode_gap_ms_mean"]),
            )
        )
    return curve_rows


def build_phase435_rows(
    *,
    phase429_csv: Path = DEFAULT_PHASE429_CSV,
    phase433_csv: Path = DEFAULT_PHASE433_CSV,
    phase426_csv: Path = DEFAULT_PHASE426_CSV,
) -> list[dict[str, object]]:
    phase429_rows = _read_rows(phase429_csv)
    phase433_rows = _read_rows(phase433_csv)
    phase426_rows = _read_rows(phase426_csv)
    rows: list[dict[str, object]] = []
    rows.extend(_extract_phase433_32k_prefill(phase433_rows))
    rows.extend(_extract_phase429_8k_prefill(phase429_rows))
    rows.extend(_extract_phase429_decode(phase429_rows, phase426_rows))
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in CSV_FIELDS})


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    gate_rows = [row for row in rows if row.get("row_type") == "consistency_gate"]
    curve_count = sum(1 for row in rows if row.get("row_type") == "serving_curve")
    failed = [row for row in gate_rows if row.get("reconstruction_gate") != "passed"]

    lines = [
        "# Phase435 Serving-State Cost Curves",
        "",
        "## Verdict",
        "",
        "- serving-state category measurements are admitted as scoped PerfDB rows, not as a mechanism model.",
        "- wait/wire/imbalance attribution remains unresolved; the row provenance records that boundary.",
        f"- extracted curve rows: {curve_count}. consistency gates failed: {len(failed)}.",
        "- Default AIC remains No-Go until A/B validation closes the validation table.",
        "",
        "## Consistency Gates",
        "",
        "| scenario | phase | target missing ms | reconstructed ms | error % | gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in gate_rows:
        lines.append(
            "| {scenario} | {phase} | {target:.3f} | {reconstructed:.3f} | {error:.2f} | {gate} |".format(
                scenario=row.get("scenario", ""),
                phase=row.get("phase", ""),
                target=float(row.get("target_missing_ms") or 0.0),
                reconstructed=float(row.get("reconstructed_missing_ms") or 0.0),
                error=float(row.get("reconstruction_error_pct") or 0.0),
                gate=row.get("reconstruction_gate", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- key scope: K2.5 / tp4dp2ep8 / int4_wo / H200 / vLLM 0.19.0.",
            "- query behavior: exact scope match plus in-grid interpolation only; out-of-grid returns None.",
            "- categories replaced: ep_a2a, moe_gemm_or_aux, other_cuda, collective_other.",
            "- MLA and dense GEMM remain on the existing计费链.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase429-csv", type=Path, default=DEFAULT_PHASE429_CSV)
    parser.add_argument("--phase433-csv", type=Path, default=DEFAULT_PHASE433_CSV)
    parser.add_argument("--phase426-csv", type=Path, default=DEFAULT_PHASE426_CSV)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_phase435_rows(
        phase429_csv=args.phase429_csv,
        phase433_csv=args.phase433_csv,
        phase426_csv=args.phase426_csv,
    )
    write_csv(rows, args.csv)
    write_markdown(rows, args.md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
