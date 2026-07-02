#!/usr/bin/env python3
"""Phase397s: int4_wo MoE route gate report.

This analyzer is report-only. It records that moving validate to vLLM 0.19.0
exposes the missing Kimi-K2.5 int4_wo moe_tp=16/moe_ep=1 table family, then
checks the H200 ep8 route gate used to decide whether the Phase397m calibrated
factor can be retired. The route gate failed, so no moe_perf rows are written.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = "phase397s_int4_wo_route_gate"
RAW_DIR = REPO_ROOT / "docs/iter_gap_investigation/phase397s_moe_route_gate"
MOE_PERF = REPO_ROOT / "src/aiconfigurator/systems/data/h200_sxm/vllm/0.19.0/moe_perf.txt"
OUTPUT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase397s_int4_wo_route_gate.csv"
OUTPUT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase397s_int4_wo_route_gate.md"

ANCHOR_MS_PER_LAYER = 9.529 / 60.0
GATE_TOLERANCE = 0.15
GATE_LOWER_MS = ANCHOR_MS_PER_LAYER * (1.0 - GATE_TOLERANCE)
GATE_UPPER_MS = ANCHOR_MS_PER_LAYER * (1.0 + GATE_TOLERANCE)
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"

ROUTE_FILES = {
    "power_law": "power_law_ep8_b128.txt",
    "balanced": "balanced_ep8_b128.txt",
    "power_law_eplb": "power_law_eplb_ep8_b128.txt",
}

FIELDNAMES = [
    "source",
    "row_type",
    "verdict",
    "evidence",
    "route",
    "distribution",
    "latency_ms",
    "anchor_ms_per_layer",
    "gate_lower_ms",
    "gate_upper_ms",
    "ratio_vs_anchor",
    "gate_passed",
    "version",
    "worker",
    "moe_dtype",
    "num_tokens",
    "hidden_size",
    "inter_size",
    "topk",
    "num_experts",
    "moe_tp_size",
    "moe_ep_size",
    "missing_family",
    "existing_ep8_int4_wo_rows",
    "existing_tp16_ep1_int4_wo_rows",
    "gpu_process_residue",
    "process_residue",
    "write_moe_perf",
    "validate_repoint_persisted",
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


def _common(row_type: str, verdict: str, evidence: str) -> dict[str, str]:
    return {
        "source": SOURCE,
        "row_type": row_type,
        "verdict": verdict,
        "evidence": evidence,
        "route": "",
        "distribution": "",
        "latency_ms": "",
        "anchor_ms_per_layer": _fmt(ANCHOR_MS_PER_LAYER),
        "gate_lower_ms": _fmt(GATE_LOWER_MS),
        "gate_upper_ms": _fmt(GATE_UPPER_MS),
        "ratio_vs_anchor": "",
        "gate_passed": FALSE,
        "version": "0.19.0",
        "worker": "h200-16-clone20260701110651-13503750-dccd5-c4ca4",
        "moe_dtype": "int4_wo",
        "num_tokens": "",
        "hidden_size": "",
        "inter_size": "",
        "topk": "",
        "num_experts": "",
        "moe_tp_size": "",
        "moe_ep_size": "",
        "missing_family": "",
        "existing_ep8_int4_wo_rows": "",
        "existing_tp16_ep1_int4_wo_rows": "",
        "gpu_process_residue": FALSE,
        "process_residue": FALSE,
        "write_moe_perf": FALSE,
        "validate_repoint_persisted": FALSE,
        "runtime_modified": FALSE,
        "gate_modified": FALSE,
        "gpu_used": TRUE,
        "ssh_used": TRUE,
        "diagnostic_only": TRUE,
        "valid_for_default": FALSE,
        "perf_database": FALSE,
        "default_readiness": DEFAULT_READINESS,
        "next_allowed_phase": "phase397s_replan_or_keep_calibrated_moe_path",
    }


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _count_existing_rows(path: Path = MOE_PERF) -> tuple[int, int]:
    ep8_rows = 0
    tp16_ep1_rows = 0
    for row in _load_csv(path):
        is_kimi_k25_shape = (
            row["hidden_size"] == "7168"
            and row["inter_size"] == "2048"
            and row["topk"] == "8"
            and row["num_experts"] == "384"
        )
        if not is_kimi_k25_shape or row["moe_dtype"] != "int4_wo":
            continue
        if row["moe_tp_size"] == "1" and row["moe_ep_size"] == "8":
            ep8_rows += 1
        if row["moe_tp_size"] == "16" and row["moe_ep_size"] == "1":
            tp16_ep1_rows += 1
    return ep8_rows, tp16_ep1_rows


def _route_row(route: str, path: Path) -> dict[str, str]:
    rows = _load_csv(path)
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one data row")
    raw = rows[0]
    latency = float(raw["latency"])
    gate_passed = GATE_LOWER_MS <= latency <= GATE_UPPER_MS
    verdict = "route_gate_passed" if gate_passed else "route_gate_failed"
    row = _common(
        f"route_gate_{route}",
        verdict,
        f"H200 ep8 b128 route gate for {route}; must be within +/-15% of 0.1588 ms/layer to retire k.",
    )
    row.update(
        {
            "route": route,
            "distribution": raw["distribution"],
            "latency_ms": _fmt(latency),
            "ratio_vs_anchor": _fmt(latency / ANCHOR_MS_PER_LAYER),
            "gate_passed": TRUE if gate_passed else FALSE,
            "moe_dtype": raw["moe_dtype"],
            "num_tokens": raw["num_tokens"],
            "hidden_size": raw["hidden_size"],
            "inter_size": raw["inter_size"],
            "topk": raw["topk"],
            "num_experts": raw["num_experts"],
            "moe_tp_size": raw["moe_tp_size"],
            "moe_ep_size": raw["moe_ep_size"],
        }
    )
    return row


def build_rows(raw_dir: Path = RAW_DIR, moe_perf: Path = MOE_PERF) -> list[dict[str, str]]:
    ep8_rows, tp16_ep1_rows = _count_existing_rows(moe_perf)

    rows: list[dict[str, str]] = []

    row = _common(
        "validate_repoint_probe",
        "blocked_on_missing_int4_wo_tp16_ep1",
        "Repointing validate to vLLM 0.19.0 reaches Kimi int4_wo moe_tp=16/moe_ep=1 and fails because that exact family is absent.",
    )
    row.update(
        {
            "missing_family": "moonshotai/Kimi-K2.5:int4_wo:moe_tp16:moe_ep1:power_law_1.01",
            "existing_ep8_int4_wo_rows": str(ep8_rows),
            "existing_tp16_ep1_int4_wo_rows": str(tp16_ep1_rows),
            "gpu_used": FALSE,
            "ssh_used": FALSE,
        }
    )
    rows.append(row)

    row = _common(
        "coverage_check",
        "unique_missing_family_confirmed",
        "vLLM 0.19.0 moe_perf has Kimi int4_wo ep8 rows but zero tp16/ep1 rows.",
    )
    row.update(
        {
            "missing_family": "moonshotai/Kimi-K2.5:int4_wo:moe_tp16:moe_ep1",
            "existing_ep8_int4_wo_rows": str(ep8_rows),
            "existing_tp16_ep1_int4_wo_rows": str(tp16_ep1_rows),
            "gpu_used": FALSE,
            "ssh_used": FALSE,
        }
    )
    rows.append(row)

    route_rows = [_route_row(route, raw_dir / filename) for route, filename in ROUTE_FILES.items()]
    rows.extend(route_rows)

    gpu_residue = (raw_dir / "gpu_compute_apps_after.txt").read_text(encoding="utf-8") != ""
    process_residue = (raw_dir / "process_residual_after.txt").read_text(encoding="utf-8") != ""

    row = _common(
        "decision",
        "route_gate_failed_report_only",
        "All structured routes miss the 0.1588 ms/layer anchor by more than the allowed 15%, so Phase397s must not collect tp16/ep1 rows or rewrite moe_perf.",
    )
    row.update(
        {
            "missing_family": "moonshotai/Kimi-K2.5:int4_wo:moe_tp16:moe_ep1",
            "existing_ep8_int4_wo_rows": str(ep8_rows),
            "existing_tp16_ep1_int4_wo_rows": str(tp16_ep1_rows),
            "gpu_process_residue": TRUE if gpu_residue else FALSE,
            "process_residue": TRUE if process_residue else FALSE,
        }
    )
    rows.append(row)

    validate_rows(rows)
    return rows


def validate_rows(rows: list[dict[str, str]]) -> None:
    if [row["row_type"] for row in rows] != [
        "validate_repoint_probe",
        "coverage_check",
        "route_gate_power_law",
        "route_gate_balanced",
        "route_gate_power_law_eplb",
        "decision",
    ]:
        raise ValueError("Phase397s rows must have the fixed report-only shape")
    for row in rows:
        if row["write_moe_perf"] != FALSE:
            raise ValueError("Phase397s report-only rows require write_moe_perf=false")
        if row["validate_repoint_persisted"] != FALSE:
            raise ValueError("Phase397s failed gate requires validate_repoint_persisted=false")
        if row["valid_for_default"] != FALSE:
            raise ValueError("Phase397s rows require valid_for_default=false")
        if row["perf_database"] != FALSE:
            raise ValueError("Phase397s rows require perf_database=false")
        if row["default_readiness"] != DEFAULT_READINESS:
            raise ValueError("Phase397s rows require default_readiness=No-Go")
        if row["row_type"].startswith("route_gate_") and row["gate_passed"] != FALSE:
            raise ValueError("Phase397s route gate report expects gate_passed=false")
    decision = rows[-1]
    if decision["gpu_process_residue"] != FALSE or decision["process_residue"] != FALSE:
        raise ValueError("Phase397s cleanup requires empty GPU/process residue")


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
        "# Phase397s Int4 WO Route Gate",
        "",
        "Phase397s stopped report-only. The route gate did not justify retiring the Phase397m calibrated MoE path.",
        "",
        "| Route | Latency ms/layer | Ratio vs 0.1588 ms anchor | Gate |",
        "|---|---:|---:|---|",
    ]
    for row in route_rows:
        lines.append(
            f"| {row['route']} | {row['latency_ms']} | {row['ratio_vs_anchor']}x | {row['verdict']} |"
        )
    decision = rows[-1]
    lines.extend(
        [
            "",
            f"Decision: `{decision['verdict']}`.",
            "",
            "Do not append `moe_perf.txt` rows, do not persist the validate 0.19.0 repoint as a passing state, and do not relax thresholds.",
            "The missing family remains `moonshotai/Kimi-K2.5:int4_wo:moe_tp16:moe_ep1`.",
            "",
            "Default AIC remains No-Go.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--moe-perf", type=Path, default=MOE_PERF)
    parser.add_argument("--out-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--out-md", type=Path, default=OUTPUT_MD)
    args = parser.parse_args(argv)

    rows = build_rows(args.raw_dir, args.moe_perf)
    write_csv(rows, args.out_csv)
    write_md(rows, args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
