#!/usr/bin/env python3
"""Phase417: summarize the MoE/EP charge runtime fix and remaining gate."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SOURCE = "phase417_moe_ep_charge_fix"
DEFAULT_SCENARIO = "K2.5-tp4ep8dp2-32k3k"
PHASE415_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase415_prefill_charge_components.csv"
PHASE416_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase416_moe_charge_trace.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase417_moe_ep_charge_fix.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase417_moe_ep_charge_fix.md"
DEFAULT_READINESS = "No-Go"

BEFORE_MOE_SCALED_TOKENS = 16002
BEFORE_MOE_PATH = "phase397v_int4_wo_calibrated_sol"
BEFORE_MOE_RETURNED_MS = 52.706517
BEFORE_EP_SCALED_TOKENS = 8001
BEFORE_EP_PATH = "fallback_tp_dp_collectives"
BEFORE_EP_RETURNED_MS = 136.511523

CSV_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "component",
    "before_scaled_tokens",
    "after_scaled_tokens",
    "expected_tokens",
    "before_query_path",
    "after_query_path",
    "before_returned_ms",
    "after_returned_ms",
    "roofline_lower_bound_ms",
    "remaining_undercharge_ms",
    "status",
    "mechanism_verdict",
    "next_phase_target",
    "anchor_revalidation_status",
    "phase405_penalty_read",
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


def _phase416_trace(rows: list[dict[str, str]], component: str, phase: str) -> dict[str, str]:
    for row in rows:
        if row.get("row_type") == "query_trace" and row.get("component") == component and row.get("phase") == phase:
            return row
    raise ValueError(f"missing Phase416 trace row for {component}:{phase}")


def _phase415_summary(rows: list[dict[str, str]]) -> dict[str, str]:
    for row in rows:
        if row.get("row_type") == "summary":
            return row
    raise ValueError("missing Phase415 summary row")


def _per_step(row: dict[str, str], field: str) -> float:
    count = float(row.get("prefill_step_count") or 1.0)
    if count <= 0.0:
        raise ValueError("prefill_step_count must be positive")
    return float(row[field]) / count


def _status(after: float, roofline: float) -> tuple[str, float]:
    remaining = max(0.0, roofline - after)
    return ("fixed" if remaining <= 0.0 else "remaining_under_roofline"), remaining


def build_phase417_rows(
    *,
    phase415_csv: Path = PHASE415_CSV,
    phase416_csv: Path = PHASE416_CSV,
    scenario: str = DEFAULT_SCENARIO,
) -> list[dict[str, str]]:
    phase415_rows = [row for row in _read_rows(phase415_csv) if row.get("scenario") == scenario]
    phase416_rows = [row for row in _read_rows(phase416_csv) if row.get("scenario") == scenario]
    if not phase415_rows or not phase416_rows:
        raise ValueError(f"missing Phase415/416 rows for {scenario}")

    moe = _phase416_trace(phase416_rows, "moe_compute", "mixed_prefill")
    ep = _phase416_trace(phase416_rows, "ep_dispatch_combine", "mixed_prefill")
    phase415_summary = _phase415_summary(phase415_rows)

    moe_status, moe_remaining = _status(float(moe["returned_ms"]), float(moe["roofline_lower_bound_ms"]))
    ep_status, ep_remaining = _status(float(ep["returned_ms"]), float(ep["roofline_lower_bound_ms"]))
    if moe_status == "fixed" and ep_status == "fixed":
        verdict = "phase417_moe_ep_charge_fixed"
        next_phase = "anchor_revalidate_phase400_phase403"
        anchor_status = "required_before_default"
    else:
        verdict = "phase417_partial_fix_moe_table_still_under_roofline"
        next_phase = "phase418_moe_prefill_table_or_roofline_reconciliation"
        anchor_status = "blocked_by_component_gate"

    rows: list[dict[str, object]] = [
        {
            "source": SOURCE,
            "row_type": "component",
            "scenario": scenario,
            "component": "moe_compute",
            "before_scaled_tokens": BEFORE_MOE_SCALED_TOKENS,
            "after_scaled_tokens": int(moe["scaled_tokens"]),
            "expected_tokens": int(moe["expected_tokens"]),
            "before_query_path": BEFORE_MOE_PATH,
            "after_query_path": moe["query_path"],
            "before_returned_ms": BEFORE_MOE_RETURNED_MS,
            "after_returned_ms": float(moe["returned_ms"]),
            "roofline_lower_bound_ms": float(moe["roofline_lower_bound_ms"]),
            "remaining_undercharge_ms": moe_remaining,
            "status": moe_status,
        },
        {
            "source": SOURCE,
            "row_type": "component",
            "scenario": scenario,
            "component": "ep_dispatch_combine",
            "before_scaled_tokens": BEFORE_EP_SCALED_TOKENS,
            "after_scaled_tokens": int(ep["scaled_tokens"]),
            "expected_tokens": int(ep["expected_tokens"]),
            "before_query_path": BEFORE_EP_PATH,
            "after_query_path": ep["query_path"],
            "before_returned_ms": BEFORE_EP_RETURNED_MS,
            "after_returned_ms": float(ep["returned_ms"]),
            "roofline_lower_bound_ms": float(ep["roofline_lower_bound_ms"]),
            "remaining_undercharge_ms": ep_remaining,
            "status": ep_status,
        },
        {
            "source": SOURCE,
            "row_type": "summary",
            "scenario": scenario,
            "component": "summary",
            "after_returned_ms": float(phase415_summary["sim_mean_ms"]),
            "roofline_lower_bound_ms": _per_step(phase415_summary, "roofline_lower_bound_ms"),
            "remaining_undercharge_ms": _per_step(phase415_summary, "roofline_gap_ms"),
            "status": "component_gate_failed" if verdict != "phase417_moe_ep_charge_fixed" else "component_gate_passed",
        },
    ]
    for row in rows:
        row.update(
            {
                "mechanism_verdict": verdict,
                "next_phase_target": next_phase,
                "anchor_revalidation_status": anchor_status,
                "phase405_penalty_read": False,
                "gpu_allowed": False,
                "ssh_allowed": False,
                "runtime_modified": True,
                "perf_database": False,
                "valid_for_default": False,
                "diagnostic_only": True,
                "default_readiness": DEFAULT_READINESS,
            }
        )
    return [{field: _fmt(row.get(field, "")) for field in CSV_FIELDS} for row in rows]


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Phase417 rows must not be empty")
    if sum(1 for row in rows if row.get("row_type") == "summary") != 1:
        raise ValueError("Phase417 expects exactly one summary row")
    for row in rows:
        if row.get("source") != SOURCE:
            raise ValueError("unexpected source")
        if row.get("phase405_penalty_read") != "false":
            raise ValueError("phase405_penalty_read must stay false")
        if row.get("gpu_allowed") != "false" or row.get("ssh_allowed") != "false":
            raise ValueError("Phase417 is offline and must not use GPU/SSH")
        if row.get("runtime_modified") != "true":
            raise ValueError("runtime_modified must be true for Phase417")
        if row.get("perf_database") != "false":
            raise ValueError("perf_database must stay false")
        if row.get("valid_for_default") != "false":
            raise ValueError("valid_for_default must stay false")
        if row.get("diagnostic_only") != "true":
            raise ValueError("diagnostic_only must stay true")
        if row.get("default_readiness") != DEFAULT_READINESS:
            raise ValueError("default_readiness must stay No-Go")


def write_phase417_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_phase417_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    summary = [row for row in rows if row["row_type"] == "summary"][0]
    components = [row for row in rows if row["row_type"] == "component"]
    if summary["mechanism_verdict"] == "phase417_moe_ep_charge_fixed":
        summary_notes = [
            "- EP dispatch/combine is fixed in this phase.",
            "- MoE now uses the measured large-context table and clears the corrected Phase415 physical lower-bound check.",
            "- Anchor revalidation is required before any Default AIC readiness claim.",
        ]
    else:
        summary_notes = [
            "- EP dispatch/combine is fixed in this phase.",
            "- MoE now uses the measured large-context table, but that table value is still below the Phase415 physical lower-bound check.",
            "- Phase417 therefore stops before anchor revalidation and keeps Default AIC as No-Go.",
        ]
    lines = [
        "# Phase417 MoE/EP Charge Fix",
        "",
        "Phase417 修改 runtime 计费路径，但不改 PerfDatabase 数据、不改 gate、不使用 GPU。",
        "",
        "## Verdict",
        "",
        f"- verdict: `{summary['mechanism_verdict']}`",
        f"- next phase: `{summary['next_phase_target']}`",
        f"- anchor revalidation: `{summary['anchor_revalidation_status']}`",
        f"- Default AIC: `{summary['default_readiness']}`",
        "",
        "## Component Result",
        "",
        "| component | before path | after path | before scaled | after scaled | expected | before ms | after ms | roofline ms | remaining ms | status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in components:
        lines.append(
            f"| {row['component']} | {row['before_query_path']} | {row['after_query_path']} | "
            f"{row['before_scaled_tokens']} | {row['after_scaled_tokens']} | {row['expected_tokens']} | "
            f"{row['before_returned_ms']} | {row['after_returned_ms']} | "
            f"{row['roofline_lower_bound_ms']} | {row['remaining_undercharge_ms']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- mixed step current charge: `{summary['after_returned_ms']}` ms",
            f"- mixed step roofline lower-bound per-step: `{summary['roofline_lower_bound_ms']}` ms",
            f"- remaining below-roofline gap: `{summary['remaining_undercharge_ms']}` ms",
            *summary_notes,
            "",
            "## Boundary",
            "",
            "- GPU/SSH: not used.",
            "- PerfDatabase data file: not modified.",
            "- Gate/default readiness: not modified.",
            "- Phase405 penalty: not read.",
            "",
        ]
    )
    return "\n".join(lines)


def write_phase417_md(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase417_md(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase415-csv", type=Path, default=PHASE415_CSV)
    parser.add_argument("--phase416-csv", type=Path, default=PHASE416_CSV)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_phase417_rows(
        phase415_csv=args.phase415_csv,
        phase416_csv=args.phase416_csv,
        scenario=args.scenario,
    )
    write_phase417_csv(args.csv, rows)
    write_phase417_md(args.md, rows)
    summary = [row for row in rows if row["row_type"] == "summary"][0]
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")
    print(f"verdict={summary['mechanism_verdict']}")


if __name__ == "__main__":
    main()
