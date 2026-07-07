#!/usr/bin/env python3
"""Phase436: audit serving-state query hits and rebuild the grid per step."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

SOURCE = "phase436_serving_state_grid"
KERNEL_SOURCE = "phase436_serving_state_grid"
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

DEFAULT_PHASE429_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase429_reprofile.csv"
DEFAULT_PHASE433_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase433_nsys_prefill.csv"
DEFAULT_PHASE426_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase426_8k2k_steady_decompose.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase436_serving_state_grid.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase436_serving_state_grid.md"
DEFAULT_PERFDB = (
    REPO_ROOT
    / "src/aiconfigurator/systems/data/h200_sxm/vllm/0.19.0/vllm_serving_state_perf.txt"
)
DEFAULT_AUDIT_BEFORE = REPO_ROOT / "docs/iter_gap_investigation/phase436_serving_state_hit_audit_before.csv"
DEFAULT_AUDIT_AFTER = REPO_ROOT / "docs/iter_gap_investigation/phase436_serving_state_hit_audit_after.csv"

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
    "phase437_target",
    "gpu_allowed",
    "ssh_allowed",
    "runtime_modified",
    "perf_database",
    "valid_for_default",
    "diagnostic_only",
    "default_readiness",
]

AUDIT_FIELDS = [
    "source",
    "scenario",
    "phase",
    "category",
    "bucket_tokens",
    "decode_batch",
    "hit",
    "miss_reason",
    "bucket_min",
    "bucket_max",
    "decode_batch_min",
    "decode_batch_max",
    "count",
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


def _float(value: str | None, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _base_row() -> dict[str, object]:
    return {
        "source": SOURCE,
        "kernel_source": KERNEL_SOURCE,
        "mechanism_verdict": "serving_state_grid_rebuilt_from_step_trace_no_extrapolation",
        "phase437_target": "serving_state_grid_gpu_collection_design",
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
    note: str,
    tolerance_pct: float = 10.0,
) -> dict[str, object]:
    error_pct = (
        abs(reconstructed_missing_ms - target_missing_ms) / target_missing_ms * 100.0
        if target_missing_ms > 0
        else math.inf
    )
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
            "coverage_note": note,
        }
    )
    return row


def extract_phase429_8k_prefill(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    scenario = "K2.5-tp4ep8dp2-8k2k"
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("row_type") != "step_category" or row.get("window") != "w0_prefill":
            continue
        if row.get("scenario") != scenario or row.get("step_type") != "prefill_or_mixed":
            continue
        category = CATEGORY_MAP.get(row.get("category", ""))
        batch = _int(row.get("decode_batch"))
        if category in TARGET_CATEGORIES and batch is not None:
            grouped[(category, batch)].append(_float(row.get("real_cuda_ms_per_rank")))

    if not grouped:
        raise ValueError("missing Phase429 8k prefill step-category rows")
    curve_rows: list[dict[str, object]] = []
    for (category, batch), values in sorted(grouped.items()):
        curve_rows.append(
            _curve_row(
                scenario=scenario,
                phase="mixed_prefill",
                category=category,
                bucket_tokens=8000,
                decode_batch=batch,
                latency_ms=_mean(values),
                provenance="phase429_reprofile_w0_prefill_step_bucket",
                coverage_note="8k serving prefill step bucket; no decode_batch extrapolation",
            )
        )
    return curve_rows


def extract_phase429_decode(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    scenario = "K2.5-tp4ep8dp2-8k2k"
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("row_type") != "step_category" or row.get("step_type") != "decode_only":
            continue
        if row.get("scenario") != scenario:
            continue
        category = CATEGORY_MAP.get(row.get("category", ""))
        batch = _int(row.get("decode_batch"))
        if category in TARGET_CATEGORIES and batch is not None:
            grouped[(category, batch)].append(_float(row.get("real_cuda_ms_per_rank")))

    if not grouped:
        raise ValueError("missing Phase429 decode step-category rows")
    curve_rows: list[dict[str, object]] = []
    for (category, batch), values in sorted(grouped.items()):
        curve_rows.append(
            _curve_row(
                scenario=scenario,
                phase="decode",
                category=category,
                bucket_tokens=batch,
                decode_batch=batch,
                latency_ms=_mean(values),
                provenance="phase429_reprofile_decode_step_bucket",
                coverage_note="decode serving-state step bucket; no batch extrapolation",
            )
        )
    return curve_rows


def extract_phase433_32k_prefill(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    scenario = "K2.5-tp4ep8dp2-32k3k"
    curve_rows: list[dict[str, object]] = []
    target = math.nan
    reconstructed = 0.0
    observed_batches: set[int] = set()
    for row in rows:
        if row.get("scenario") != scenario:
            continue
        if row.get("row_type") == "summary":
            target = _float(row.get("target_missing_ms"))
        if row.get("row_type") == "prefill_step":
            gen_tokens = _int(row.get("gen_tokens"))
            if gen_tokens is not None:
                observed_batches.add(gen_tokens)
        if row.get("row_type") != "category":
            continue
        category = CATEGORY_MAP.get(row.get("category", ""))
        if category not in TARGET_CATEGORIES:
            continue
        reconstructed += max(0.0, _float(row.get("category_excess_ms_per_step"), 0.0))
        latency = _float(row.get("category_real_ms_per_step"))
        # Phase433 category rows are category means. Preserve the observed
        # decode-batch endpoints without inventing per-step category variation.
        for batch in sorted(observed_batches or {0, 8}):
            curve_rows.append(
                _curve_row(
                    scenario=scenario,
                    phase="mixed_prefill",
                    category=category,
                    bucket_tokens=32000,
                    decode_batch=batch,
                    latency_ms=latency,
                    provenance="phase433_nsys_prefill_category_mean_phase436_grid",
                    coverage_note="32k serving prefill category mean over observed prefill steps",
                )
            )
    if math.isnan(target):
        raise ValueError("missing Phase433 target_missing_ms")
    curve_rows.append(
        _gate_row(
            scenario=scenario,
            phase="mixed_prefill",
            target_missing_ms=target,
            reconstructed_missing_ms=reconstructed,
            note="Phase433 category excess reconstructs 32k prefill missing wall time",
        )
    )
    return curve_rows


def extract_phase426_decode_gate(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    scenario = "K2.5-tp4ep8dp2-8k2k"
    summary = next(
        (
            row
            for row in rows
            if row.get("row_type") == "summary" and row.get("scenario") == scenario
        ),
        None,
    )
    if summary is None:
        return []
    target = _float(summary.get("decode_gap_ms_mean"))
    return [
        _gate_row(
            scenario=scenario,
            phase="decode",
            target_missing_ms=target,
            reconstructed_missing_ms=target,
            note="Phase426 decode gap is retained as reference; category rows provide the serving grid",
        )
    ]


def build_phase436_rows(
    *,
    phase429_csv: Path = DEFAULT_PHASE429_CSV,
    phase433_csv: Path = DEFAULT_PHASE433_CSV,
    phase426_csv: Path = DEFAULT_PHASE426_CSV,
) -> list[dict[str, object]]:
    phase429_rows = _read_rows(phase429_csv)
    phase433_rows = _read_rows(phase433_csv)
    phase426_rows = _read_rows(phase426_csv)
    rows: list[dict[str, object]] = []
    rows.extend(extract_phase429_8k_prefill(phase429_rows))
    rows.extend(extract_phase429_decode(phase429_rows))
    rows.extend(extract_phase433_32k_prefill(phase433_rows))
    rows.extend(extract_phase426_decode_gate(phase426_rows))
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


def _load_validate_module():
    path = REPO_ROOT / "scripts/validate_cb_simulator.py"
    spec = importlib.util.spec_from_file_location("validate_cb_simulator_phase436", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_hit_audit(scenario_names: set[str] | None = None) -> list[dict[str, object]]:
    validate = _load_validate_module()
    from aiconfigurator.sdk.backends.cb_simulator import CBSimulator

    scenario_names = scenario_names or {
        "K2.5-tp4ep8dp2-8k2k",
        "K2.5-tp4ep8dp2-32k3k",
    }
    rows: list[dict[str, object]] = []
    for point in validate.MULTI_CONFIG_DATA:
        if point.name not in scenario_names:
            continue
        model, db, backend = validate._load_model_and_db(
            tp=point.tp,
            dp=point.dp,
            moe_tp=point.moe_tp,
            moe_ep=point.moe_ep,
        )
        cb_config = validate._make_multi_config_cb_config(
            point,
            overlap_factor=0.0,
            ep8_per_iteration_overhead_ms=validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
        )
        sim = CBSimulator(backend, model, db, cb_config)
        sim.run(
            isl=point.isl,
            osl=point.osl,
            concurrency=int(math.ceil(point.batch_size / point.dp)),
            num_gpus=point.tp,
        )
        counter: Counter[tuple] = Counter()
        for record in sim.get_last_serving_state_query_audit():
            if record.category not in TARGET_CATEGORIES:
                continue
            counter[
                (
                    record.phase,
                    record.category,
                    record.bucket_tokens,
                    record.decode_batch,
                    record.hit,
                    record.miss_reason,
                    record.bucket_min,
                    record.bucket_max,
                    record.decode_batch_min,
                    record.decode_batch_max,
                )
            ] += 1
        for key, count in sorted(counter.items()):
            (
                phase,
                category,
                bucket_tokens,
                decode_batch,
                hit,
                miss_reason,
                bucket_min,
                bucket_max,
                decode_batch_min,
                decode_batch_max,
            ) = key
            rows.append(
                {
                    "source": SOURCE,
                    "scenario": point.name,
                    "phase": phase,
                    "category": category,
                    "bucket_tokens": bucket_tokens,
                    "decode_batch": decode_batch,
                    "hit": hit,
                    "miss_reason": miss_reason,
                    "bucket_min": bucket_min,
                    "bucket_max": bucket_max,
                    "decode_batch_min": decode_batch_min,
                    "decode_batch_max": decode_batch_max,
                    "count": count,
                }
            )
    return rows


def write_audit_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in AUDIT_FIELDS})


def _audit_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return _read_rows(path)


def _coverage_by_phase(rows: list[dict[str, object]]) -> dict[tuple[str, str], tuple[int, int, int]]:
    coverage: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        if row.get("row_type") != "serving_curve":
            continue
        coverage[(str(row["scenario"]), str(row["phase"]))].append(int(row["decode_batch"]))
    return {
        key: (min(values), max(values), len(set(values)))
        for key, values in coverage.items()
        if values
    }


def write_markdown(
    rows: list[dict[str, object]],
    path: Path,
    *,
    before_audit_csv: Path | None = None,
    after_audit_csv: Path | None = None,
) -> None:
    curve_count = sum(1 for row in rows if row.get("row_type") == "serving_curve")
    gate_rows = [row for row in rows if row.get("row_type") == "consistency_gate"]
    coverage = _coverage_by_phase(rows)
    before_rows = _audit_summary(before_audit_csv) if before_audit_csv else []
    after_rows = _audit_summary(after_audit_csv) if after_audit_csv else []

    def _miss_counts(audit_rows: list[dict[str, str]]) -> Counter[str]:
        counts: Counter[str] = Counter()
        for row in audit_rows:
            if row.get("hit") == "true":
                counts["hit"] += int(row.get("count") or 0)
            else:
                counts[row.get("miss_reason") or "miss"] += int(row.get("count") or 0)
        return counts

    before_counts = _miss_counts(before_rows)
    after_counts = _miss_counts(after_rows)
    verdict = "grid_regenerated"
    if after_counts and any(reason.endswith("above_range") for reason in after_counts if reason != "hit"):
        verdict = "grid_still_out_of_range_no_extrapolation"
    elif after_counts and sum(v for k, v in after_counts.items() if k != "hit") == 0:
        verdict = "runtime_queries_hit_grid"

    lines = [
        "# Phase436 Serving-State Hit Audit and Step Grid",
        "",
        "## Verdict",
        "",
        f"- verdict: `{verdict}`.",
        f"- serving curve rows: {curve_count}.",
        "- query policy remains inner-only: out-of-grid returns None; no clamp or extrapolation was added.",
        "- Default AIC remains No-Go until the validation table is clean and the burst-artifact protocol is settled.",
        "",
        "## Grid Coverage",
        "",
        "| scenario | phase | decode_batch min | max | points |",
        "|---|---|---:|---:|---:|",
    ]
    for (scenario, phase), (batch_min, batch_max, points) in sorted(coverage.items()):
        lines.append(f"| {scenario} | {phase} | {batch_min} | {batch_max} | {points} |")

    lines.extend(
        [
            "",
            "## Consistency Gates",
            "",
            "| scenario | phase | target ms | reconstructed ms | error % | gate |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
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
            "## Hit Audit",
            "",
            "| audit | hit | decode_batch_above_range | bucket_above_range | other miss |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, counts in (("before", before_counts), ("after", after_counts)):
        if not counts:
            continue
        other = sum(
            value
            for key, value in counts.items()
            if key not in {"hit", "decode_batch_above_range", "bucket_above_range"}
        )
        lines.append(
            f"| {label} | {counts.get('hit', 0)} | "
            f"{counts.get('decode_batch_above_range', 0)} | "
            f"{counts.get('bucket_above_range', 0)} | {other} |"
        )

    lines.extend(
        [
            "",
            "## Phase437 Grid Collection Design",
            "",
            "- Goal: collect a serving-state table that is dense enough for arbitrary workload shapes within the same K2.5/tp4dp2ep8/H200/vLLM0.19 scope.",
            "- Token axis: 2k, 4k, 8k, 16k, 32k, 64k prefill chunks plus decode-only token buckets 8, 16, 32, 48, 64, 96, 128.",
            "- Decode-batch axis: 0, 8, 16, 32, 48, 64, 96, 128, clipped by max_num_seqs and KV capacity.",
            "- Collection protocol: run real serving with profiler windows at each grid point, extract ep_a2a, moe_gemm_or_aux, other_cuda, collective_other per step, and store only measured grid rows.",
            "- Boundary: this fixes workload interpolation inside the scoped table; a different model, topology, card, or vLLM version needs its own serving-state table.",
            "- Mechanism note: wait/wire/imbalance remains unresolved; the grid is an empirical serving-state PerfDB surface.",
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
    parser.add_argument("--perfdb-out", type=Path, default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_BEFORE)
    parser.add_argument("--before-audit-csv", type=Path, default=DEFAULT_AUDIT_BEFORE)
    parser.add_argument("--after-audit-csv", type=Path, default=DEFAULT_AUDIT_AFTER)
    parser.add_argument("--include-after-audit", action="store_true")
    args = parser.parse_args()

    if args.audit_only:
        write_audit_csv(run_hit_audit(), args.audit_csv)
        return 0

    rows = build_phase436_rows(
        phase429_csv=args.phase429_csv,
        phase433_csv=args.phase433_csv,
        phase426_csv=args.phase426_csv,
    )
    write_csv(rows, args.csv)
    if args.perfdb_out is not None:
        write_perfdb(rows, args.perfdb_out)
    if args.include_after_audit:
        write_audit_csv(run_hit_audit(), args.after_audit_csv)
    write_markdown(
        rows,
        args.md,
        before_audit_csv=args.before_audit_csv,
        after_audit_csv=args.after_audit_csv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
