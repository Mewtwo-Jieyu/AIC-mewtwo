#!/usr/bin/env python3
"""Phase418: reconcile MoE roofline gate with moe_perf provenance."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import analyze_phase414_prefill_decode_audit as phase414  # noqa: E402
from scripts import analyze_phase415_prefill_charge_components as phase415  # noqa: E402


SOURCE = "phase418_roofline_reconcile"
DEFAULT_SCENARIO = "K2.5-tp4ep8dp2-32k3k"
PHASE416_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase416_moe_charge_trace.csv"
PHASE414_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase414_prefill_decode_audit.csv"
PHASE417_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase417_moe_ep_charge_fix.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase418_roofline_reconcile.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase418_roofline_reconcile.md"
DEFAULT_READINESS = "No-Go"

OLD_MOE_BOUND_MS = 455.757015
FP8_TFLOPS = 1979.0
BF16_TFLOPS = 990.0
REASONABLE_MFU_MIN = 0.20
REASONABLE_MFU_MAX = 0.70

CSV_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "component",
    "old_bound_ms",
    "corrected_fp8_bound_ms",
    "corrected_bf16_bound_ms",
    "selected_bound_ms",
    "measured_ms",
    "fixed_mixed_step_ms",
    "real_prefill_step_ms",
    "remaining_real_over_fixed_ratio",
    "remaining_gap_statement",
    "old_gate_status",
    "corrected_gate_status",
    "implicit_mfu",
    "mfu_status",
    "collector_committed_semantics",
    "collector_dirty_drift",
    "provenance_status",
    "mechanism_verdict",
    "next_phase_target",
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


def _row_by_type(rows: list[dict[str, str]], *, scenario: str, row_type: str) -> dict[str, str]:
    for row in rows:
        if row.get("scenario") == scenario and row.get("row_type", "") == row_type:
            return row
    raise ValueError(f"missing {row_type} row for {scenario}")


def _run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True)


def committed_collect_moe_text() -> str:
    return _run_git(["show", "HEAD:collector/vllm/collect_moe.py"])


def dirty_collect_moe_diff() -> str:
    return _run_git(["diff", "--", "collector/vllm/collect_moe.py"])


def audit_collector_provenance(committed_text: str, dirty_diff: str) -> tuple[str, str, str]:
    required = [
        "hidden_states = torch.randn([num_tokens, hidden_size]",
        "topk_weights_list.append",
        "fused_marlin_moe(",
        "global_num_experts=num_experts",
        "expert_map=expert_map",
    ]
    missing = [needle for needle in required if needle not in committed_text]
    if missing:
        return "collector_semantics_missing_required_markers", "dirty_not_evaluated", "mismatch"
    semantics = "num_tokens_token_rows_single_layer_ep_expert_map_no_dispatch"
    dirty_status = "dirty_collect_moe_diff_present" if dirty_diff.strip() else "dirty_collect_moe_diff_absent"
    return semantics, dirty_status, "aligned"


def corrected_moe_bound_ms(*, peak_tflops: float) -> float:
    token_expert_rows_per_gpu = phase414.FULL_CHUNK_TOKENS * phase415.TOPK / phase415.MOE_EP_SIZE
    flops = (
        phase415.MOE_LAYERS
        * token_expert_rows_per_gpu
        * 6.0
        * phase415.HIDDEN_SIZE
        * phase415.INTER_SIZE
    )
    return flops / (peak_tflops * 1e12) * 1000.0


def build_phase418_rows(
    *,
    phase416_csv: Path = PHASE416_CSV,
    phase414_csv: Path = PHASE414_CSV,
    phase417_csv: Path = PHASE417_CSV,
    scenario: str = DEFAULT_SCENARIO,
    committed_text: str | None = None,
    dirty_diff: str | None = None,
) -> list[dict[str, str]]:
    phase416_rows = [row for row in _read_rows(phase416_csv) if row.get("scenario") == scenario]
    if not phase416_rows:
        raise ValueError(f"missing Phase416 rows for {scenario}")
    moe = _phase416_trace(phase416_rows, "moe_compute", "mixed_prefill")
    ep = _phase416_trace(phase416_rows, "ep_dispatch_combine", "mixed_prefill")
    phase414_summary = _row_by_type(_read_rows(phase414_csv), scenario=scenario, row_type="")
    phase417_summary = _row_by_type(_read_rows(phase417_csv), scenario=scenario, row_type="summary")
    if committed_text is None:
        committed_text = committed_collect_moe_text()
    if dirty_diff is None:
        dirty_diff = dirty_collect_moe_diff()
    collector_semantics, dirty_status, provenance_status = audit_collector_provenance(committed_text, dirty_diff)

    measured = float(moe["returned_ms"])
    fp8_bound = corrected_moe_bound_ms(peak_tflops=FP8_TFLOPS)
    bf16_bound = corrected_moe_bound_ms(peak_tflops=BF16_TFLOPS)
    selected_bound = bf16_bound
    implicit_mfu = selected_bound / measured if measured > 0.0 else math.inf
    corrected_passes = measured >= selected_bound
    mfu_ok = REASONABLE_MFU_MIN <= implicit_mfu <= REASONABLE_MFU_MAX
    real_prefill_step = float(phase414_summary["prefill_real_mean_ms"])
    fixed_mixed_step = float(phase417_summary["after_returned_ms"])
    remaining_ratio = real_prefill_step / fixed_mixed_step if fixed_mixed_step > 0.0 else math.inf
    remaining_gap_statement = "mixed_prefill_still_approximately_2_6x_below_real_after_gate_unlock"
    if corrected_passes and provenance_status == "aligned" and mfu_ok:
        verdict = "phase415_roofline_gate_wrong_not_moe_perf_table"
        next_phase = "regenerate_phase415_416_417_and_anchor_revalidate"
    else:
        verdict = "moe_perf_semantics_or_efficiency_still_unresolved"
        next_phase = "phase419_moe_prefill_data_recollection_or_profiler"

    rows: list[dict[str, object]] = [
        {
            "source": SOURCE,
            "row_type": "roofline",
            "scenario": scenario,
            "component": "moe_compute",
            "old_bound_ms": OLD_MOE_BOUND_MS,
            "corrected_fp8_bound_ms": fp8_bound,
            "corrected_bf16_bound_ms": bf16_bound,
            "selected_bound_ms": selected_bound,
            "measured_ms": measured,
            "fixed_mixed_step_ms": fixed_mixed_step,
            "real_prefill_step_ms": real_prefill_step,
            "remaining_real_over_fixed_ratio": remaining_ratio,
            "remaining_gap_statement": remaining_gap_statement,
            "old_gate_status": "failed",
            "corrected_gate_status": "passed" if corrected_passes else "failed",
            "implicit_mfu": implicit_mfu,
            "mfu_status": "reasonable" if mfu_ok else "out_of_band",
            "collector_committed_semantics": collector_semantics,
            "collector_dirty_drift": dirty_status,
            "provenance_status": provenance_status,
        },
        {
            "source": SOURCE,
            "row_type": "roofline",
            "scenario": scenario,
            "component": "ep_dispatch_combine",
            "old_bound_ms": float(ep["roofline_lower_bound_ms"]),
            "selected_bound_ms": float(ep["roofline_lower_bound_ms"]),
            "measured_ms": float(ep["returned_ms"]),
            "fixed_mixed_step_ms": fixed_mixed_step,
            "real_prefill_step_ms": real_prefill_step,
            "remaining_real_over_fixed_ratio": remaining_ratio,
            "remaining_gap_statement": remaining_gap_statement,
            "old_gate_status": "passed",
            "corrected_gate_status": "passed" if float(ep["returned_ms"]) >= float(ep["roofline_lower_bound_ms"]) else "failed",
            "collector_committed_semantics": collector_semantics,
            "collector_dirty_drift": dirty_status,
            "provenance_status": provenance_status,
        },
        {
            "source": SOURCE,
            "row_type": "summary",
            "scenario": scenario,
            "component": "summary",
            "old_bound_ms": OLD_MOE_BOUND_MS,
            "selected_bound_ms": selected_bound,
            "measured_ms": measured,
            "fixed_mixed_step_ms": fixed_mixed_step,
            "real_prefill_step_ms": real_prefill_step,
            "remaining_real_over_fixed_ratio": remaining_ratio,
            "remaining_gap_statement": remaining_gap_statement,
            "old_gate_status": "failed",
            "corrected_gate_status": "passed" if corrected_passes else "failed",
            "implicit_mfu": implicit_mfu,
            "mfu_status": "reasonable" if mfu_ok else "out_of_band",
            "collector_committed_semantics": collector_semantics,
            "collector_dirty_drift": dirty_status,
            "provenance_status": provenance_status,
        },
    ]
    for row in rows:
        row.update(
            {
                "mechanism_verdict": verdict,
                "next_phase_target": next_phase,
                "phase405_penalty_read": False,
                "gpu_allowed": False,
                "ssh_allowed": False,
                "runtime_modified": False,
                "perf_database": False,
                "valid_for_default": False,
                "diagnostic_only": True,
                "default_readiness": DEFAULT_READINESS,
            }
        )
    return [{field: _fmt(row.get(field, "")) for field in CSV_FIELDS} for row in rows]


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Phase418 rows must not be empty")
    if sum(1 for row in rows if row.get("row_type") == "summary") != 1:
        raise ValueError("Phase418 expects exactly one summary row")
    for row in rows:
        if row.get("source") != SOURCE:
            raise ValueError("unexpected source")
        if row.get("phase405_penalty_read") != "false":
            raise ValueError("phase405_penalty_read must stay false")
        if row.get("gpu_allowed") != "false" or row.get("ssh_allowed") != "false":
            raise ValueError("Phase418 is offline and must not use GPU/SSH")
        if row.get("runtime_modified") != "false":
            raise ValueError("runtime_modified must stay false")
        if row.get("perf_database") != "false":
            raise ValueError("perf_database must stay false")
        if row.get("valid_for_default") != "false":
            raise ValueError("valid_for_default must stay false")
        if row.get("diagnostic_only") != "true":
            raise ValueError("diagnostic_only must stay true")
        if row.get("default_readiness") != DEFAULT_READINESS:
            raise ValueError("default_readiness must stay No-Go")


def write_phase418_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_phase418_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    summary = [row for row in rows if row["row_type"] == "summary"][0]
    roofline = [row for row in rows if row["row_type"] == "roofline"]
    lines = [
        "# Phase418 Roofline Reconcile",
        "",
        "Phase418 只复核报告门和 provenance；不加 runtime clamp、不改 PerfDatabase 数据。",
        "",
        "## Verdict",
        "",
        f"- verdict: `{summary['mechanism_verdict']}`",
        f"- next phase: `{summary['next_phase_target']}`",
        f"- remaining mixed-prefill gap: real `{summary['real_prefill_step_ms']}` ms vs fixed sim `{summary['fixed_mixed_step_ms']}` ms = `{summary['remaining_real_over_fixed_ratio']}`x",
        f"- Default AIC: `{summary['default_readiness']}`",
        "",
        "## Roofline Replay",
        "",
        "| component | old bound | fp8 bound | bf16 bound | selected bound | measured | old gate | corrected gate | MFU |",
        "|---|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in roofline:
        lines.append(
            f"| {row['component']} | {row['old_bound_ms']} | {row['corrected_fp8_bound_ms']} | "
            f"{row['corrected_bf16_bound_ms']} | {row['selected_bound_ms']} | {row['measured_ms']} | "
            f"{row['old_gate_status']} | {row['corrected_gate_status']} | {row['implicit_mfu']} |"
        )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- committed collector semantics: `{summary['collector_committed_semantics']}`",
            f"- dirty collector drift: `{summary['collector_dirty_drift']}`",
            f"- provenance: `{summary['provenance_status']}`",
            "",
            "## Boundary",
            "",
            "- GPU/SSH: not used.",
            "- Runtime/PerfDatabase/gate: not modified.",
            "- Phase405 penalty: not read.",
            "- Default AIC: No-Go.",
            "",
        ]
    )
    return "\n".join(lines)


def write_phase418_md(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase418_md(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase414-csv", type=Path, default=PHASE414_CSV)
    parser.add_argument("--phase416-csv", type=Path, default=PHASE416_CSV)
    parser.add_argument("--phase417-csv", type=Path, default=PHASE417_CSV)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_phase418_rows(
        phase414_csv=args.phase414_csv,
        phase416_csv=args.phase416_csv,
        phase417_csv=args.phase417_csv,
        scenario=args.scenario,
    )
    write_phase418_csv(args.csv, rows)
    write_phase418_md(args.md, rows)
    summary = [row for row in rows if row["row_type"] == "summary"][0]
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")
    print(f"verdict={summary['mechanism_verdict']}")


if __name__ == "__main__":
    main()
