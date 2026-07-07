#!/usr/bin/env python3
"""Phase438: dry-run serving-state union merge and axis sensitivity."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

SOURCE = "phase438_serving_state_merge"
DEFAULT_PERFDB = (
    REPO_ROOT
    / "src/aiconfigurator/systems/data/h200_sxm/vllm/0.19.0/vllm_serving_state_perf.txt"
)
DEFAULT_PHASE437_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase437_serving_state_collect.csv"
DEFAULT_PHASE437_AB = REPO_ROOT / "docs/iter_gap_investigation/phase437_serving_state_validate_ab.csv"
DEFAULT_AXIS_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase438_axis_sensitivity.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase438_serving_state_merge.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase438_serving_state_merge.md"
DEFAULT_TOLERANCE_PCT = 15.0

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

REPORT_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "phase",
    "category",
    "bucket_tokens",
    "decode_batch",
    "metric",
    "value",
    "baseline_value",
    "candidate_value",
    "ratio",
    "classification",
    "window",
    "reason",
    "count",
    "note",
]

AXIS_FIELDS = [
    "source",
    "phase",
    "category",
    "axis",
    "fixed_axis",
    "fixed_value",
    "point_count",
    "min_latency_ms",
    "max_latency_ms",
    "mean_latency_ms",
    "spread_pct",
    "tolerance_pct",
    "decision",
]


class ServingRow(NamedTuple):
    source: str
    phase: str
    category: str
    bucket_tokens: int
    decode_batch: int
    latency_ms: float
    provenance: str

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (self.phase, self.category, self.bucket_tokens, self.decode_batch)


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


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_perfdb_rows(path: Path, *, source: str = "current_perfdb") -> list[ServingRow]:
    rows: list[ServingRow] = []
    for row in _read_rows(path):
        rows.append(
            ServingRow(
                source=source,
                phase=row["phase"],
                category=row["category"],
                bucket_tokens=int(row["bucket_tokens"]),
                decode_batch=int(row["decode_batch"]),
                latency_ms=float(row["latency"]),
                provenance=row["provenance"],
            )
        )
    return rows


def read_phase437_rows(path: Path, *, source: str = "phase437_candidate") -> list[ServingRow]:
    rows: list[ServingRow] = []
    for row in _read_rows(path):
        if row.get("row_type") != "serving_curve":
            continue
        rows.append(
            ServingRow(
                source=source,
                phase=row["phase"],
                category=row["category"],
                bucket_tokens=int(float(row["bucket_tokens"])),
                decode_batch=int(float(row["decode_batch"])),
                latency_ms=float(row["latency_ms"]),
                provenance=row["provenance"],
            )
        )
    return rows


def read_candidate_rows(paths: list[Path]) -> list[ServingRow]:
    rows: list[ServingRow] = []
    for index, path in enumerate(paths):
        rows.extend(read_phase437_rows(path, source=f"{path.stem}_candidate_{index}"))
    return rows


def _spread_pct(values: list[float]) -> float:
    if not values:
        return math.nan
    mean = sum(values) / len(values)
    return math.inf if mean == 0 else (max(values) - min(values)) / mean * 100.0


def merge_rows(
    baseline_rows: list[ServingRow],
    candidate_rows: list[ServingRow],
    *,
    conflict_threshold_pct: float = DEFAULT_TOLERANCE_PCT,
) -> tuple[list[ServingRow], list[dict[str, object]]]:
    by_key: dict[tuple[str, str, int, int], ServingRow] = {row.key: row for row in baseline_rows}
    conflicts: list[dict[str, object]] = []
    for row in candidate_rows:
        old = by_key.get(row.key)
        if old is None:
            by_key[row.key] = row
            continue
        spread = _spread_pct([old.latency_ms, row.latency_ms])
        if spread > conflict_threshold_pct:
            conflicts.append(
                {
                    "source": SOURCE,
                    "row_type": "merge_conflict",
                    "phase": row.phase,
                    "category": row.category,
                    "bucket_tokens": row.bucket_tokens,
                    "decode_batch": row.decode_batch,
                    "baseline_value": old.latency_ms,
                    "candidate_value": row.latency_ms,
                    "value": spread,
                    "decision": "keep_existing_report_conflict",
                    "note": f"{old.source} vs {row.source}",
                }
            )
        # Same-key candidate never silently overwrites the existing accepted row.
    return sorted(by_key.values(), key=lambda row: row.key), conflicts


def axis_sensitivity_rows(
    rows: list[ServingRow],
    *,
    tolerance_pct: float = DEFAULT_TOLERANCE_PCT,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for (phase, category, bucket), group in sorted(
        _groups(rows, key=lambda row: (row.phase, row.category, row.bucket_tokens)).items()
    ):
        if len(group) < 2:
            continue
        latencies = [row.latency_ms for row in group]
        spread = _spread_pct(latencies)
        out.append(
            {
                "source": SOURCE,
                "phase": phase,
                "category": category,
                "axis": "decode_batch",
                "fixed_axis": "bucket_tokens",
                "fixed_value": bucket,
                "point_count": len(group),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "mean_latency_ms": sum(latencies) / len(latencies),
                "spread_pct": spread,
                "tolerance_pct": tolerance_pct,
                "decision": "secondary_axis_flat" if spread <= tolerance_pct else "secondary_axis_sensitive",
            }
        )
    for (phase, category, batch), group in sorted(
        _groups(rows, key=lambda row: (row.phase, row.category, row.decode_batch)).items()
    ):
        if len(group) < 2:
            continue
        latencies = [row.latency_ms for row in group]
        spread = _spread_pct(latencies)
        out.append(
            {
                "source": SOURCE,
                "phase": phase,
                "category": category,
                "axis": "bucket_tokens",
                "fixed_axis": "decode_batch",
                "fixed_value": batch,
                "point_count": len(group),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "mean_latency_ms": sum(latencies) / len(latencies),
                "spread_pct": spread,
                "tolerance_pct": tolerance_pct,
                "decision": "secondary_axis_flat" if spread <= tolerance_pct else "secondary_axis_sensitive",
            }
        )
    return out


def _groups(rows: list[ServingRow], *, key) -> dict[object, list[ServingRow]]:
    grouped: dict[object, list[ServingRow]] = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    return grouped


def axis_decisions(
    rows: list[ServingRow],
    *,
    tolerance_pct: float = DEFAULT_TOLERANCE_PCT,
) -> dict[tuple[str, str], dict[str, object]]:
    sensitivities = axis_sensitivity_rows(rows, tolerance_pct=tolerance_pct)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in sensitivities:
        grouped[(str(row["phase"]), str(row["category"]))].append(row)

    decisions: dict[tuple[str, str], dict[str, object]] = {}
    for phase, category in sorted({(row.phase, row.category) for row in rows}):
        items = grouped.get((phase, category), [])
        decode_spreads = [float(row["spread_pct"]) for row in items if row["axis"] == "decode_batch"]
        bucket_spreads = [float(row["spread_pct"]) for row in items if row["axis"] == "bucket_tokens"]
        max_decode = max(decode_spreads) if decode_spreads else math.nan
        max_bucket = max(bucket_spreads) if bucket_spreads else math.nan
        if phase == "mixed_prefill" and decode_spreads and max_decode <= tolerance_pct:
            decision = "reduce_decode_batch_axis"
        elif phase == "decode" and bucket_spreads and max_bucket <= tolerance_pct:
            decision = "reduce_bucket_axis"
        else:
            decision = "keep_2d"
        decisions[(phase, category)] = {
            "phase": phase,
            "category": category,
            "decode_batch_spread_pct": max_decode,
            "bucket_tokens_spread_pct": max_bucket,
            "tolerance_pct": tolerance_pct,
            "decision": decision,
        }
    return decisions


def write_axis_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AXIS_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in AXIS_FIELDS})


def write_candidate_perfdb(rows: list[ServingRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PERFDB_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item.key):
            writer.writerow(
                {
                    "framework": "VLLM",
                    "version": "0.19.0",
                    "device": "NVIDIA H200",
                    "model": "kimi-k2.5",
                    "topology": "tp4dp2ep8",
                    "phase": row.phase,
                    "category": row.category,
                    "kernel_source": row.source,
                    "bucket_tokens": row.bucket_tokens,
                    "decode_batch": row.decode_batch,
                    "hidden_size": 7168,
                    "topk": 8,
                    "moe_ep_size": 8,
                    "quant_runtime": "CompressedTensorsWNA16MarlinMoEMethod",
                    "latency": _fmt(row.latency_ms),
                    "provenance": row.provenance,
                }
            )


def _load_validate_module():
    path = REPO_ROOT / "scripts/validate_cb_simulator.py"
    spec = importlib.util.spec_from_file_location("validate_cb_simulator_phase438", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _inject_serving_state(db, candidate_perfdb: Path) -> None:
    from aiconfigurator.sdk.perf_database import load_vllm_serving_state_data

    db._vllm_serving_state_data = load_vllm_serving_state_data(str(candidate_perfdb))
    clear_cache = getattr(db.query_vllm_serving_state, "cache_clear", None)
    if clear_cache is not None:
        clear_cache()


def run_validate_ab_with_candidate(
    candidate_perfdb: Path,
    *,
    baseline_ab_csv: Path = DEFAULT_PHASE437_AB,
) -> list[dict[str, object]]:
    validate = _load_validate_module()
    from aiconfigurator.sdk import common
    from aiconfigurator.sdk.config import RuntimeConfig

    baseline: dict[str, dict[str, str]] = {}
    if baseline_ab_csv.exists():
        for row in _read_rows(baseline_ab_csv):
            baseline[row["scenario"]] = row

    loaded: dict[tuple[int, int, int, int], tuple] = {}
    out: list[dict[str, object]] = []
    for point in validate.MULTI_CONFIG_DATA:
        key = (point.tp, point.dp, point.moe_tp, point.moe_ep)
        if key not in loaded:
            model, db, backend = validate._load_model_and_db(
                tp=point.tp,
                dp=point.dp,
                moe_tp=point.moe_tp,
                moe_ep=point.moe_ep,
            )
            _inject_serving_state(db, candidate_perfdb)
            loaded[key] = (model, db, backend)
        model, db, backend = loaded[key]
        _inject_serving_state(db, candidate_perfdb)
        cb_config = validate._make_multi_config_cb_config(
            point,
            overlap_factor=0.0,
            ep8_per_iteration_overhead_ms=validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
        )
        cb_summary = backend.run_agg(
            model,
            db,
            RuntimeConfig(batch_size=point.batch_size, isl=point.isl, osl=point.osl),
            ctx_tokens=point.max_num_batched_tokens,
            database_mode=common.DatabaseMode.HYBRID,
            method="cb_sim",
            cb_config=cb_config,
        )
        sim = float(cb_summary.get_result_dict()["tokens/s/gpu"])
        real = float(point.real_output_tok_s_gpu)
        ratio = max(sim / real, real / sim)
        before = baseline.get(point.name, {})
        before_ratio = float(before.get("phase436_ratio") or 0.0)
        if before_ratio == 0:
            classification = "no_baseline"
        elif ratio > before_ratio + 0.02:
            classification = "regressed"
        elif ratio < before_ratio - 0.02:
            classification = "improved"
        else:
            classification = "unchanged"
        out.append(
            {
                "source": SOURCE,
                "row_type": "validate_ab",
                "scenario": point.name,
                "metric": "error_ratio",
                "baseline_value": before_ratio,
                "candidate_value": ratio,
                "value": sim,
                "ratio": ratio,
                "classification": classification,
                "note": f"real={real:.3f}; sim={sim:.3f}",
            }
        )
    return out


def run_hit_audit_with_candidate(candidate_perfdb: Path) -> list[dict[str, object]]:
    validate = _load_validate_module()
    from aiconfigurator.sdk.backends.cb_simulator import CBSimulator

    scenario_names = {
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
        _inject_serving_state(db, candidate_perfdb)
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
            counter[
                (
                    record.phase,
                    record.category,
                    record.bucket_tokens,
                    record.decode_batch,
                    record.hit,
                    record.miss_reason,
                )
            ] += 1
        for key, count in sorted(counter.items()):
            phase, category, bucket_tokens, decode_batch, hit, reason = key
            rows.append(
                {
                    "source": SOURCE,
                    "row_type": "hit_audit",
                    "scenario": point.name,
                    "phase": phase,
                    "category": category,
                    "bucket_tokens": bucket_tokens,
                    "decode_batch": decode_batch,
                    "metric": "hit" if hit else reason,
                    "value": count,
                    "count": count,
                    "classification": "hit" if hit else "miss",
                    "note": reason,
                }
            )
    return rows


def recommend_gpu_windows(audit_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in audit_rows:
        scenario = str(row.get("scenario", ""))
        phase = str(row.get("phase", ""))
        reason = str(row.get("miss_reason") or row.get("metric") or "")
        count = int(float(row.get("count") or row.get("value") or 0))
        counts[(scenario, phase, reason)] += count

    windows: list[dict[str, object]] = []
    miss8 = sum(
        count
        for (scenario, phase, reason), count in counts.items()
        if scenario == "K2.5-tp4ep8dp2-8k2k"
        and phase == "mixed_prefill"
        and reason in {"decode_batch_above_range", "interpolation_gap", "bucket_below_range", "table_missing"}
    )
    if miss8:
        windows.append(
            {
                "source": SOURCE,
                "row_type": "window_recommendation",
                "window": "W-A",
                "scenario": "K2.5-tp4ep8dp2-8k2k",
                "reason": "8k mixed prefill in-scope coverage miss",
                "count": miss8,
                "note": "steady trigger required; extract all observed steps, not only high decode_batch rows",
            }
        )
    miss32 = sum(
        count
        for (scenario, phase, reason), count in counts.items()
        if scenario == "K2.5-tp4ep8dp2-32k3k"
        and phase == "mixed_prefill"
        and reason in {"bucket_above_range", "table_missing"}
    )
    if miss32:
        windows.append(
            {
                "source": SOURCE,
                "row_type": "window_recommendation",
                "window": "W-B",
                "scenario": "K2.5-tp4ep8dp2-32k3k",
                "reason": "32k mixed prefill bucket coverage missing",
                "count": miss32,
                "note": "only needed if union merge does not preserve Phase433/436 32k rows",
            }
        )
    return windows


def write_report_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in REPORT_FIELDS})


def write_markdown(
    *,
    path: Path,
    axis_rows: list[dict[str, object]],
    decisions: dict[tuple[str, str], dict[str, object]],
    conflicts: list[dict[str, object]],
    audit_rows: list[dict[str, object]],
    validate_rows: list[dict[str, object]],
    windows: list[dict[str, object]],
) -> None:
    audit_counts: Counter[str] = Counter()
    for row in audit_rows:
        metric = str(row.get("metric", ""))
        audit_counts[metric] += int(float(row.get("count") or row.get("value") or 0))
    verdict = "offline_union_merge_requires_gpu_window_confirmation"
    regressions = [row for row in validate_rows if row.get("classification") == "regressed"]
    if regressions:
        verdict = "offline_union_merge_regresses_do_not_ingest"
    elif not windows:
        verdict = "offline_union_merge_candidate_covers_in_scope"

    lines = [
        "# Phase438 Serving-State Merge Dry Run",
        "",
        "## Verdict",
        "",
        f"`{verdict}`.",
        "",
        "Step 0 only: no GPU was used and `vllm_serving_state_perf.txt` was not modified.",
        "",
        "## Axis Decisions",
        "",
        "| phase | category | decode-batch spread % | bucket spread % | decision |",
        "|---|---|---:|---:|---|",
    ]
    for key in sorted(decisions):
        row = decisions[key]
        lines.append(
            "| {phase} | {category} | {decode:.2f} | {bucket:.2f} | {decision} |".format(
                phase=row["phase"],
                category=row["category"],
                decode=float(row["decode_batch_spread_pct"])
                if not math.isnan(float(row["decode_batch_spread_pct"]))
                else math.nan,
                bucket=float(row["bucket_tokens_spread_pct"])
                if not math.isnan(float(row["bucket_tokens_spread_pct"]))
                else math.nan,
                decision=row["decision"],
            )
        )
    lines.extend(
        [
            "",
            "## Merge Conflicts",
            "",
            "| count | action |",
            "|---:|---|",
            f"| {len(conflicts)} | exact-key candidate conflicts are reported; existing accepted row is kept |",
            "",
            "## Hit Audit Dry Run",
            "",
            "| metric | count |",
            "|---|---:|",
        ]
    )
    for metric, count in sorted(audit_counts.items()):
        lines.append(f"| {metric} | {count} |")

    lines.extend(
        [
            "",
            "## Validate Dry Run",
            "",
            "| scenario | baseline ratio | candidate ratio | classification |",
            "|---|---:|---:|---|",
        ]
    )
    for row in validate_rows:
        lines.append(
            "| {scenario} | {baseline:.3f} | {candidate:.3f} | {classification} |".format(
                scenario=row.get("scenario", ""),
                baseline=float(row.get("baseline_value") or 0.0),
                candidate=float(row.get("candidate_value") or 0.0),
                classification=row.get("classification", ""),
            )
        )

    lines.extend(
        [
            "",
            "## GPU Window Confirmation List",
            "",
            "| window | scenario | reason | count | note |",
            "|---|---|---|---:|---|",
        ]
    )
    if windows:
        for row in windows:
            lines.append(
                f"| {row['window']} | {row['scenario']} | {row['reason']} | {row['count']} | {row['note']} |"
            )
    else:
        lines.append("| none | - | dry run covers in-scope queries | 0 | no GPU window proposed |")

    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Inner-only query semantics stay unchanged.",
            "- No clamp, extrapolation, or silent overwrite is introduced.",
            "- GPU collection must wait for explicit window confirmation.",
            "- Default AIC remains No-Go.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> int:
    baseline = read_perfdb_rows(args.perfdb)
    candidate_paths = [args.phase437_csv] + list(args.extra_candidate_csv or [])
    phase437 = read_candidate_rows(candidate_paths)
    merged, conflicts = merge_rows(
        baseline,
        phase437,
        conflict_threshold_pct=args.conflict_threshold_pct,
    )
    axis_rows = axis_sensitivity_rows(merged, tolerance_pct=args.tolerance_pct)
    decisions = axis_decisions(merged, tolerance_pct=args.tolerance_pct)
    write_axis_csv(axis_rows, args.axis_csv)

    with tempfile.TemporaryDirectory() as tmp:
        candidate_perfdb = Path(tmp) / "vllm_serving_state_perf.txt"
        write_candidate_perfdb(merged, candidate_perfdb)
        audit_rows = run_hit_audit_with_candidate(candidate_perfdb)
        validate_rows = run_validate_ab_with_candidate(candidate_perfdb, baseline_ab_csv=args.phase437_ab)

    windows = recommend_gpu_windows(audit_rows)
    report_rows: list[dict[str, object]] = []
    report_rows.extend(conflicts)
    report_rows.extend(audit_rows)
    report_rows.extend(validate_rows)
    report_rows.extend(windows)
    write_report_csv(report_rows, args.csv)
    write_markdown(
        path=args.md,
        axis_rows=axis_rows,
        decisions=decisions,
        conflicts=conflicts,
        audit_rows=audit_rows,
        validate_rows=validate_rows,
        windows=windows,
    )
    if args.perfdb_out is not None:
        regressions = [row for row in validate_rows if row.get("classification") == "regressed"]
        if regressions:
            names = ", ".join(str(row.get("scenario")) for row in regressions)
            raise RuntimeError(f"refusing to write perfdb candidate because validate regressed: {names}")
        write_candidate_perfdb(merged, args.perfdb_out)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perfdb", type=Path, default=DEFAULT_PERFDB)
    parser.add_argument("--phase437-csv", type=Path, default=DEFAULT_PHASE437_CSV)
    parser.add_argument("--extra-candidate-csv", type=Path, action="append", default=[])
    parser.add_argument("--phase437-ab", type=Path, default=DEFAULT_PHASE437_AB)
    parser.add_argument("--axis-csv", type=Path, default=DEFAULT_AXIS_CSV)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--perfdb-out", type=Path, default=None)
    parser.add_argument("--tolerance-pct", type=float, default=DEFAULT_TOLERANCE_PCT)
    parser.add_argument("--conflict-threshold-pct", type=float, default=DEFAULT_TOLERANCE_PCT)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
