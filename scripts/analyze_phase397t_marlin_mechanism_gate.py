#!/usr/bin/env python3
"""Phase397t: H200 marlin mechanism gate result.

The gate tries structured no-k collector variants against the Phase397l serve
anchor. It is report-only unless a route lands inside the anchor band.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "docs/iter_gap_investigation/phase397t_marlin_mechanism_gate"
OUTPUT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase397t_marlin_mechanism_gate.csv"
OUTPUT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase397t_marlin_mechanism_gate.md"

SOURCE = "phase397t_marlin_mechanism_gate"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"
ANCHOR_MS_PER_LAYER = 8.0201 / 60.0
GATE_LOWER_MS = ANCHOR_MS_PER_LAYER * 0.85
GATE_UPPER_MS = ANCHOR_MS_PER_LAYER * 1.15

ROUTE_FILES = {
    "power_law": "power_law_ep8_b128",
    "power_law_rank0_compact": "power_law_rank0_compact_ep8_b128",
    "power_law_eplb": "power_law_eplb_ep8_b128",
    "balanced": "balanced_ep8_b128",
    "power_law_batched_rank0": "power_law_batched_rank0_ep8_b128",
}

FIELDNAMES = [
    "source",
    "row_type",
    "verdict",
    "route",
    "distribution",
    "latency_ms",
    "anchor_ms_per_layer",
    "gate_lower_ms",
    "gate_upper_ms",
    "ratio_vs_anchor",
    "gate_passed",
    "diag_effective_m_mean",
    "diag_rank0_tokens_mean",
    "diag_rank0_selections_mean",
    "diag_valid_blocks_mean",
    "diag_block_size_m",
    "diag_num_tokens_post_padded_mean",
    "moe_dtype",
    "num_tokens",
    "hidden_size",
    "inter_size",
    "topk",
    "num_experts",
    "moe_tp_size",
    "moe_ep_size",
    "gpu_process_residue",
    "process_residue",
    "write_moe_perf",
    "validate_repoint",
    "runtime_modified",
    "gate_modified",
    "gpu_used",
    "ssh_used",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
    "default_readiness",
    "next_allowed_phase",
]


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _common(row_type: str, verdict: str = "") -> dict[str, str]:
    return {
        "source": SOURCE,
        "row_type": row_type,
        "verdict": verdict,
        "route": "",
        "distribution": "",
        "latency_ms": "",
        "anchor_ms_per_layer": _fmt(ANCHOR_MS_PER_LAYER),
        "gate_lower_ms": _fmt(GATE_LOWER_MS),
        "gate_upper_ms": _fmt(GATE_UPPER_MS),
        "ratio_vs_anchor": "",
        "gate_passed": FALSE,
        "diag_effective_m_mean": "",
        "diag_rank0_tokens_mean": "",
        "diag_rank0_selections_mean": "",
        "diag_valid_blocks_mean": "",
        "diag_block_size_m": "",
        "diag_num_tokens_post_padded_mean": "",
        "moe_dtype": "int4_wo",
        "num_tokens": "",
        "hidden_size": "",
        "inter_size": "",
        "topk": "",
        "num_experts": "",
        "moe_tp_size": "",
        "moe_ep_size": "",
        "gpu_process_residue": FALSE,
        "process_residue": FALSE,
        "write_moe_perf": FALSE,
        "validate_repoint": FALSE,
        "runtime_modified": FALSE,
        "gate_modified": FALSE,
        "gpu_used": TRUE,
        "ssh_used": TRUE,
        "diagnostic_only": TRUE,
        "valid_for_default": FALSE,
        "perf_database": FALSE,
        "default_readiness": DEFAULT_READINESS,
        "next_allowed_phase": "phase397t_replan_or_keep_phase397m_profcal",
    }


def _load_single_perf_row(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one data row")
    return rows[0]


def _parse_diag(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, str]] = []
    for line in lines:
        if not (line.startswith("[marlin_diag]") or line.startswith("[marlin_batched_diag]")):
            continue
        pairs = dict(re.findall(r"([A-Za-z0-9_]+)=([^ ]+)", line))
        records.append(pairs)
    if not records:
        return {}

    def mean_field(name: str) -> str:
        values = [float(record[name]) for record in records if name in record]
        return _fmt(statistics.mean(values)) if values else ""

    block_sizes = {record["block_size_m"] for record in records if "block_size_m" in record}
    return {
        "diag_effective_m_mean": mean_field("effective_m") or mean_field("max_tokens_per_expert"),
        "diag_rank0_tokens_mean": mean_field("rank0_num_tokens"),
        "diag_rank0_selections_mean": mean_field("rank0_total_selections"),
        "diag_valid_blocks_mean": mean_field("valid_blocks"),
        "diag_block_size_m": "/".join(sorted(block_sizes, key=int)),
        "diag_num_tokens_post_padded_mean": mean_field("num_tokens_post_padded"),
    }


def _route_row(route: str, stem: str, raw_dir: Path) -> dict[str, str]:
    perf = _load_single_perf_row(raw_dir / f"{stem}.txt")
    diag = _parse_diag(raw_dir / f"{stem}.log")
    latency = float(perf["latency"])
    gate_passed = GATE_LOWER_MS <= latency <= GATE_UPPER_MS

    row = _common(f"route_gate_{route}", "route_gate_passed" if gate_passed else "route_gate_failed")
    row.update(
        {
            "route": route,
            "distribution": perf["distribution"],
            "latency_ms": _fmt(latency),
            "ratio_vs_anchor": _fmt(latency / ANCHOR_MS_PER_LAYER),
            "gate_passed": TRUE if gate_passed else FALSE,
            "moe_dtype": perf["moe_dtype"],
            "num_tokens": perf["num_tokens"],
            "hidden_size": perf["hidden_size"],
            "inter_size": perf["inter_size"],
            "topk": perf["topk"],
            "num_experts": perf["num_experts"],
            "moe_tp_size": perf["moe_tp_size"],
            "moe_ep_size": perf["moe_ep_size"],
        }
    )
    row.update(diag)
    return row


def build_rows(raw_dir: Path = RAW_DIR) -> list[dict[str, str]]:
    rows = [_route_row(route, stem, raw_dir) for route, stem in ROUTE_FILES.items()]
    gpu_residue = (raw_dir / "gpu_compute_apps_after.txt").read_text(encoding="utf-8") != ""
    process_residue = (raw_dir / "process_residual_after.txt").read_text(encoding="utf-8") != ""

    decision = _common("decision", "mechanism_gate_failed_report_only")
    decision.update(
        {
            "gpu_process_residue": TRUE if gpu_residue else FALSE,
            "process_residue": TRUE if process_residue else FALSE,
        }
    )
    rows.append(decision)
    validate_rows(rows)
    return rows


def validate_rows(rows: list[dict[str, str]]) -> None:
    if [row["row_type"] for row in rows] != [
        "route_gate_power_law",
        "route_gate_power_law_rank0_compact",
        "route_gate_power_law_eplb",
        "route_gate_balanced",
        "route_gate_power_law_batched_rank0",
        "decision",
    ]:
        raise ValueError("Phase397t gate rows must have fixed shape")
    for row in rows:
        if row["write_moe_perf"] != FALSE:
            raise ValueError("Phase397t failed gate requires write_moe_perf=false")
        if row["validate_repoint"] != FALSE:
            raise ValueError("Phase397t failed gate requires validate_repoint=false")
        if row["valid_for_default"] != FALSE:
            raise ValueError("Phase397t gate requires valid_for_default=false")
        if row["perf_database"] != FALSE:
            raise ValueError("Phase397t gate requires perf_database=false")
        if row["default_readiness"] != DEFAULT_READINESS:
            raise ValueError("Phase397t gate requires default_readiness=No-Go")
        if row["row_type"].startswith("route_gate_") and row["gate_passed"] != FALSE:
            raise ValueError("Phase397t gate report expects all gate_passed=false")
    if rows[-1]["gpu_process_residue"] != FALSE or rows[-1]["process_residue"] != FALSE:
        raise ValueError("Phase397t cleanup requires empty GPU/process residue")


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict[str, str]], path: Path) -> None:
    validate_rows(rows)
    route_rows = [row for row in rows if row["row_type"].startswith("route_gate_")]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase397t Marlin Mechanism Gate",
        "",
        "Phase397t H200 mechanism gate failed. No route reached the no-k serve anchor band.",
        "",
        "| Route | Latency ms/layer | Ratio vs anchor | Gate |",
        "|---|---:|---:|---|",
    ]
    for row in route_rows:
        lines.append(
            f"| {row['route']} | {row['latency_ms']} | {row['ratio_vs_anchor']}x | {row['verdict']} |"
        )
    lines.extend(
        [
            "",
            "Decision: `mechanism_gate_failed_report_only`.",
            "",
            "Do not append `moe_perf.txt`, do not retire `k=0.312`, and do not persist validate 0.19.0 repoint.",
            "",
            "Default AIC remains No-Go.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--out-md", type=Path, default=OUTPUT_MD)
    args = parser.parse_args(argv)

    rows = build_rows(args.raw_dir)
    write_csv(rows, args.out_csv)
    write_md(rows, args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
