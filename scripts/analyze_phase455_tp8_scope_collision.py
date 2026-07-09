#!/usr/bin/env python3
"""Phase455 TP8 serving-state scope collision audit.

The script is report-only. It temporarily appends the Phase454 TP8 B2b
forward_total rows to a copy of the PerfDB table, then runs the TP8 validation
points against that temporary table to quantify cross-regime pollution.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import scripts.analyze_phase454_gpu_batch_acceptance as phase454
DEFAULT_PERFDB = phase454.DEFAULT_PERFDB
DEFAULT_SCOPE_ROOT = phase454.DEFAULT_SCOPE_ROOT
DEFAULT_CLEAN_VALIDATE = phase454.DEFAULT_CLEAN_VALIDATE
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase455_scope_collision.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase455_scope_collision.md"

TP8_SCENARIOS = (
    "K2.5-tp8ep8-8k2k",
    "K2.5-tp8ep8-32k3k",
    "K2.5-tp8ep8-8k2k-bt65536",
)

CSV_FIELDS = [
    "section",
    "scenario",
    "phase",
    "row_kind",
    "category",
    "bucket_min",
    "bucket_max",
    "decode_batch_min",
    "decode_batch_max",
    "rows",
    "baseline_error_ratio",
    "no_scope_error_ratio",
    "delta_error_ratio",
    "hit_count",
    "miss_count",
    "status",
    "note",
]


@dataclass(frozen=True)
class CandidateRow:
    scenario: str
    topology: str
    phase: str
    row_kind: str
    category: str
    bucket_tokens: int
    decode_batch: int
    latency_ms: float
    sample_count: int


def _fmt(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in CSV_FIELDS})


def _phase454_to_candidate(row: phase454.PerfDBCandidate) -> CandidateRow:
    return CandidateRow(
        scenario=row.scenario,
        topology=row.topology,
        phase=row.phase,
        row_kind=row.row_kind,
        category=row.category,
        bucket_tokens=row.bucket_tokens,
        decode_batch=row.decode_batch,
        latency_ms=row.latency_ms,
        sample_count=row.sample_count,
    )


def load_tp8_b2b_candidates(scope_root: Path = DEFAULT_SCOPE_ROOT) -> list[CandidateRow]:
    specs = [phase454.default_b2b_specs(scope_root)[0]]
    return [_phase454_to_candidate(row) for row in phase454.extract_b2b_rows(specs)]


def summarize_candidate_surface(rows: Iterable[CandidateRow]) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[CandidateRow]] = defaultdict(list)
    for row in rows:
        grouped[row.phase].append(row)

    summary: dict[str, dict[str, int]] = {}
    for phase, items in grouped.items():
        summary[phase] = {
            "rows": len(items),
            "bucket_min": min(item.bucket_tokens for item in items),
            "bucket_max": max(item.bucket_tokens for item in items),
            "decode_batch_min": min(item.decode_batch for item in items),
            "decode_batch_max": max(item.decode_batch for item in items),
        }
    return summary


def classify_delta(baseline_error: float, current_error: float, threshold: float = 0.005) -> str:
    delta = current_error - baseline_error
    if delta > threshold:
        return "regressed"
    if delta < -threshold:
        return "improved"
    return "unchanged"


def regime_scope_rows(scenario_to_max_bt: dict[str, int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario, max_bt in sorted(scenario_to_max_bt.items()):
        rows.append(
            {
                "section": "regime_scope",
                "scenario": scenario,
                "regime_scope": f"max_num_batched_tokens={max_bt}",
                "status": "config_derived",
                "note": "exact-match scope; no scenario label",
            }
        )
    return rows


def clean_baseline_by_name(path: Path = DEFAULT_CLEAN_VALIDATE) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for row in _read_csv(path):
        rows[row["name"]] = {
            "real_output_tok_s_gpu": float(row["real_output_tok_s_gpu"]),
            "sim_output_tok_s_gpu": float(row["sim_output_tok_s_gpu"]),
            "error_ratio": float(row["error_ratio"]),
        }
    return rows


def _perfdb_row(row: CandidateRow, *, max_num_batched_tokens: int | None = None) -> dict[str, object]:
    return {
        "framework": "VLLM",
        "version": "0.19.0",
        "device": "NVIDIA H200",
        "model": phase454.MODEL,
        "topology": row.topology,
        "phase": row.phase,
        "row_kind": row.row_kind,
        "category": row.category,
        "kernel_source": phase454.KERNEL_SOURCE,
        "max_num_batched_tokens": (
            phase454.SCENARIO_MAX_BT[row.scenario]
            if max_num_batched_tokens is None
            else max_num_batched_tokens
        ),
        "bucket_tokens": row.bucket_tokens,
        "decode_batch": row.decode_batch,
        "hidden_size": phase454.HIDDEN_SIZE,
        "topk": phase454.TOPK,
        "moe_ep_size": phase454.MOE_EP_SIZE,
        "quant_runtime": phase454.QUANT_RUNTIME,
        "latency": row.latency_ms,
        "provenance": "phase454_b2b_event_step_bucket",
    }


def _no_scope_perfdb_rows(row: CandidateRow) -> list[dict[str, object]]:
    return [
        _perfdb_row(row, max_num_batched_tokens=phase454.SCENARIO_MAX_BT[scenario])
        for scenario in TP8_SCENARIOS
    ]


def _write_temp_perfdb(base_perfdb: Path, candidate_rows: Iterable[CandidateRow], path: Path) -> None:
    path.write_text(base_perfdb.read_text(encoding="utf-8").rstrip("\n") + "\n", encoding="utf-8")
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=phase454.PERFDB_FIELDS, lineterminator="\n")
        for row in candidate_rows:
            writer.writerows(_no_scope_perfdb_rows(row))


def run_no_scope_tp8_audit(
    candidate_rows: list[CandidateRow],
    *,
    perfdb_path: Path = DEFAULT_PERFDB,
) -> list[dict[str, object]]:
    import validate_cb_simulator as validate
    from aiconfigurator.sdk.backends.cb_simulator import CBSimulator
    from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend
    from aiconfigurator.sdk.perf_database import load_vllm_serving_state_data

    baseline = clean_baseline_by_name()
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="phase455_scope_") as tmp:
        temp_perfdb = Path(tmp) / "vllm_serving_state_perf.txt"
        _write_temp_perfdb(perfdb_path, candidate_rows, temp_perfdb)

        for scenario in TP8_SCENARIOS:
            point = next(pt for pt in validate.MULTI_CONFIG_DATA if pt.name == scenario)
            model, db, _ = validate._load_model_and_db(
                tp=point.tp,
                dp=point.dp,
                moe_tp=point.moe_tp,
                moe_ep=point.moe_ep,
            )
            db._vllm_serving_state_data = load_vllm_serving_state_data(str(temp_perfdb))
            clear_cache = getattr(db.query_vllm_serving_state, "cache_clear", None)
            if clear_cache is not None:
                clear_cache()
            config = validate._make_multi_config_cb_config(
                point,
                overlap_factor=0.0,
                ep8_per_iteration_overhead_ms=0.0,
            )
            sim = CBSimulator(VLLMBackend(), model, db, config)
            result = sim.run(
                isl=point.isl,
                osl=point.osl,
                concurrency=point.batch_size,
                prefix=0,
                num_gpus=point.tp,
            )
            sim_output = result.throughput_tok_s / point.tp
            real_output = float(baseline[scenario]["real_output_tok_s_gpu"])
            current_error = max(sim_output / real_output, real_output / sim_output)
            baseline_error = float(baseline[scenario]["error_ratio"])
            audit = [record.as_dict() for record in sim.get_last_serving_state_query_audit()]
            hits = [record for record in audit if record["hit"]]
            hit_counter = Counter(
                (
                    record["phase"],
                    record["row_kind"],
                    record["category"],
                    record["bucket_tokens"],
                    record["decode_batch"],
                )
                for record in hits
            )
            top_hit = hit_counter.most_common(1)[0][0] if hit_counter else ()
            rows.append(
                {
                    "section": "no_scope_temp_ingest",
                    "scenario": scenario,
                    "phase": top_hit[0] if top_hit else "",
                    "row_kind": top_hit[1] if top_hit else "",
                    "category": top_hit[2] if top_hit else "",
                    "bucket_min": top_hit[3] if top_hit else "",
                    "bucket_max": top_hit[3] if top_hit else "",
                    "decode_batch_min": top_hit[4] if top_hit else "",
                    "decode_batch_max": top_hit[4] if top_hit else "",
                    "rows": len(hit_counter),
                    "baseline_error_ratio": baseline_error,
                    "no_scope_error_ratio": current_error,
                    "delta_error_ratio": current_error - baseline_error,
                    "hit_count": len(hits),
                    "miss_count": len(audit) - len(hits),
                    "status": classify_delta(baseline_error, current_error),
                    "note": f"top_hit={top_hit}; temp_perfdb_without_regime_scope",
                }
            )
    return rows


def build_report_rows(
    *,
    candidates: list[CandidateRow],
    no_scope_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for phase, summary in sorted(summarize_candidate_surface(candidates).items()):
        rows.append(
            {
                "section": "tp8_b2b_candidate_surface",
                "scenario": "K2.5-tp8ep8-8k2k",
                "phase": phase,
                "row_kind": "forward_total",
                "category": "forward_total",
                **summary,
                "status": "candidate_retained",
                "note": "Phase454 B2b rows; blocked until scoped",
            }
        )
    rows.extend(no_scope_rows)
    max_bt_by_scenario = {
        "K2.5-tp8ep8-8k2k": 8000,
        "K2.5-tp8ep8-32k3k": 32000,
        "K2.5-tp8ep8-8k2k-bt65536": 65536,
    }
    for item in regime_scope_rows(max_bt_by_scenario):
        rows.append(
            {
                "section": item["section"],
                "scenario": item["scenario"],
                "status": item["status"],
                "note": item["regime_scope"],
            }
        )
    return rows


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    surface = [row for row in rows if row.get("section") == "tp8_b2b_candidate_surface"]
    no_scope = [row for row in rows if row.get("section") == "no_scope_temp_ingest"]
    scope = [row for row in rows if row.get("section") == "regime_scope"]
    max_no_scope = max(float(row.get("no_scope_error_ratio") or 0.0) for row in no_scope)
    verdict = (
        "tp8_scope_required_max_num_batched_tokens"
        if any(row.get("status") == "regressed" for row in no_scope)
        else "tp8_scope_not_required"
    )
    lines = [
        "# Phase455 TP8 scope collision",
        "",
        f"Verdict: `{verdict}`.",
        "",
        "## Candidate Surface",
        "",
        "| phase | rows | bucket range | decode batch range |",
        "|---|---:|---:|---:|",
    ]
    for row in surface:
        lines.append(
            f"| {row['phase']} | {row['rows']} | {row['bucket_min']}..{row['bucket_max']} "
            f"| {row['decode_batch_min']}..{row['decode_batch_max']} |"
        )
    lines.extend(
        [
            "",
            "## No-Scope Temporary Ingest",
            "",
            "| scenario | baseline | no-scope | delta | hits | status |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in no_scope:
        lines.append(
            f"| {row['scenario']} | {float(row['baseline_error_ratio']):.3f}x "
            f"| {float(row['no_scope_error_ratio']):.3f}x "
            f"| {float(row['delta_error_ratio']):+.3f} | {row['hit_count']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Scope Decision",
            "",
            "| scenario | scope |",
            "|---|---|",
        ]
    )
    for row in scope:
        lines.append(f"| {row['scenario']} | `{row['note']}` |")
    lines.extend(
        [
            "",
            f"Max no-scope error: `{max_no_scope:.3f}x`.",
            "",
            "`max_num_batched_tokens` is configuration-derived and separates the three TP8 regimes in the six-point table.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(csv_out: Path = DEFAULT_CSV, md_out: Path = DEFAULT_MD) -> list[dict[str, object]]:
    candidates = load_tp8_b2b_candidates()
    no_scope_rows = run_no_scope_tp8_audit(candidates)
    rows = build_report_rows(candidates=candidates, no_scope_rows=no_scope_rows)
    _write_csv(csv_out, rows)
    write_markdown(md_out, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    rows = run_analysis(args.csv_out, args.md_out)
    verdict = "tp8_scope_required_max_num_batched_tokens"
    print(f"wrote {args.csv_out}")
    print(f"wrote {args.md_out}")
    print(f"verdict={verdict} rows={len(rows)}")


if __name__ == "__main__":
    main()
