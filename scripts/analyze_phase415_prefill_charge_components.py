#!/usr/bin/env python3
"""Phase415: decompose mixed-prefill charge into sim components."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Mapping
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aiconfigurator.sdk.config import RuntimeConfig  # noqa: E402
from scripts import analyze_phase413_steady_decompose as phase413  # noqa: E402
from scripts import analyze_phase414_prefill_decode_audit as phase414  # noqa: E402


SOURCE = "phase415_prefill_charge_components"
DEFAULT_SCENARIO = "K2.5-tp4ep8dp2-32k3k"
PHASE414_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase414_prefill_decode_audit.csv"
SWEEP_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase412_arrival_sweep"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase415_prefill_charge_components.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase415_prefill_charge_components.md"
DEFAULT_READINESS = "No-Go"

HIDDEN_SIZE = 7168
INTER_SIZE = 2048
TOPK = 8
MOE_LAYERS = 60
MOE_EP_SIZE = 8
ATTENTION_LAYERS = 61
TP4_LOCAL_HEADS = 16
MLA_QK_DIM = 192
MLA_V_DIM = 128
H200_TENSOR_TFLOPS = 1979.0
MOE_WNA16_TENSOR_TFLOPS = 990.0
NVLINK_GBPS = 900.0
BYTES_FP16 = 2.0

CSV_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "artifact_dir",
    "component",
    "sim_ms",
    "sim_mean_ms",
    "roofline_lower_bound_ms",
    "roofline_gap_ms",
    "roofline_status",
    "component_gap_share",
    "phase414_prefill_gap_ms",
    "phase414_decode_gap_ms",
    "target_missing_ms",
    "reconstructed_missing_ms",
    "reconstruction_error_pct",
    "consistency_gate",
    "prefill_step_count",
    "mechanism_verdict",
    "phase416_target",
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


def _safe_ratio(a: float, b: float) -> float:
    if b <= 0:
        return math.inf
    return a / b


def _error_pct(predicted: float, target: float) -> float:
    return abs(_safe_ratio(predicted, target) - 1.0) * 100.0


def _read_csv_by_scenario(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        return {row["scenario"]: row for row in csv.DictReader(f)}


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _component_for_op(name: str, *, decode: bool = False) -> str:
    prefix = "decode_" if decode else ""
    if name in {"context_attention", "generation_attention"}:
        return f"{prefix}prefill_mla_attention" if not decode else "decode_mla_attention"
    if name in {"context_moe", "generation_moe"}:
        return f"{prefix}moe_compute" if decode else "moe_compute"
    if "moe_pre_dispatch" in name or "moe_post_dispatch" in name:
        return f"{prefix}ep_dispatch_combine" if decode else "ep_dispatch_combine"
    if "_ar_" in name or name.endswith("_p2p"):
        return f"{prefix}tp_comm" if decode else "tp_comm"
    if "gemm" in name and "moe" not in name:
        return f"{prefix}gemm_other" if decode else "gemm_other"
    return f"{prefix}other" if decode else "other"


def _add_component(out: dict[str, float], component: str, value: float) -> None:
    out[component] = out.get(component, 0.0) + float(value)


def _mixed_components_for_gen_reqs(gen_reqs: int) -> dict[str, float]:
    from scripts import validate_cb_simulator as validate_cb  # noqa: PLC0415

    model, db, backend = validate_cb._load_model_and_db(tp=4, dp=2, moe_tp=1, moe_ep=8)
    total_tokens = phase414.FULL_CHUNK_TOKENS + gen_reqs
    pass1 = backend.run_static(
        model,
        db,
        RuntimeConfig(batch_size=1, beam_width=1, isl=total_tokens, osl=1, prefix=0),
        mode="static_ctx",
        op_query_overrides={"context_prefill_tokens": phase414.FULL_CHUNK_TOKENS},
    ).get_context_latency_dict()
    pass2 = backend.run_static(
        model,
        db,
        RuntimeConfig(batch_size=1, beam_width=1, isl=phase414.FULL_CHUNK_TOKENS, osl=1, prefix=0),
        mode="static_ctx",
    ).get_context_latency_dict()
    pass3 = backend.run_static(
        model,
        db,
        RuntimeConfig(batch_size=max(gen_reqs, 1), beam_width=1, isl=phase414.DECODE_KV_BASE, osl=2, prefix=0),
        mode="static_gen",
    ).get_generation_latency_dict()

    out: dict[str, float] = {}
    for name, value in pass1.items():
        if name == "context_attention":
            continue
        _add_component(out, _component_for_op(name), value)
    _add_component(out, "prefill_mla_attention", pass2.get("context_attention", 0.0))
    _add_component(out, "decode_mla_attention", pass3.get("generation_attention", 0.0))
    return out


def _decode_components(batch: int = 9, kv_len: int = phase414.DECODE_KV_BASE) -> dict[str, float]:
    from scripts import validate_cb_simulator as validate_cb  # noqa: PLC0415

    model, db, backend = validate_cb._load_model_and_db(tp=4, dp=2, moe_tp=1, moe_ep=8)
    gen = backend.run_static(
        model,
        db,
        RuntimeConfig(batch_size=batch, beam_width=1, isl=kv_len, osl=2, prefix=0),
        mode="static_gen",
    ).get_generation_latency_dict()
    out: dict[str, float] = {}
    for name, value in gen.items():
        _add_component(out, _component_for_op(name, decode=True), value)
    return out


def _full_prefill_gen_reqs(artifact_dir: Path) -> list[int]:
    window = phase413.steady_window_from_metrics(phase413._metrics_path(artifact_dir))
    steps = phase413.parse_iteration_steps(phase413._serve_log_path(artifact_dir))
    timelines = phase413.build_wallclock_timelines(steps)
    return [
        timed.step.generation_requests
        for timeline in timelines.values()
        for timed in timeline
        if phase413._step_overlaps_window(timed, window) > 0
        and timed.step.ctx_tokens >= phase414.FULL_CHUNK_MIN_TOKENS
    ]


def _mixed_component_totals(gen_reqs_values: list[int]) -> dict[str, float]:
    cache: dict[int, dict[str, float]] = {}
    totals: dict[str, float] = {}
    for gen_reqs in gen_reqs_values:
        if gen_reqs not in cache:
            cache[gen_reqs] = _mixed_components_for_gen_reqs(gen_reqs)
        for component, value in cache[gen_reqs].items():
            _add_component(totals, component, value)
    return totals


def _roofline_lower_bounds(prefill_step_count: int) -> dict[str, float]:
    tokens = phase414.FULL_CHUNK_TOKENS
    attention_flops = (
        ATTENTION_LAYERS
        * 2.0
        * tokens
        * tokens
        * TP4_LOCAL_HEADS
        * (MLA_QK_DIM + MLA_V_DIM)
    )
    moe_flops = (
        MOE_LAYERS
        * (tokens * TOPK / MOE_EP_SIZE)
        * 6.0
        * HIDDEN_SIZE
        * INTER_SIZE
    )
    ep_bytes = (
        MOE_LAYERS
        * tokens
        * HIDDEN_SIZE
        * BYTES_FP16
        * TOPK
        * 2.0
    )
    tp_bytes = (
        ATTENTION_LAYERS
        * tokens
        * HIDDEN_SIZE
        * BYTES_FP16
        * 2.0
    )
    per_step = {
        "prefill_mla_attention": attention_flops / (H200_TENSOR_TFLOPS * 1e12) * 1000.0,
        "moe_compute": moe_flops / (MOE_WNA16_TENSOR_TFLOPS * 1e12) * 1000.0,
        "ep_dispatch_combine": ep_bytes / (NVLINK_GBPS * 1e9) * 1000.0,
        "tp_comm": tp_bytes / (NVLINK_GBPS * 1e9) * 1000.0,
    }
    return {name: value * prefill_step_count for name, value in per_step.items()}


def _artifact_dir_from_phase414(row: Mapping[str, str], sweep_root: Path, scenario: str) -> Path:
    raw = row.get("artifact_dir", "")
    if raw:
        candidate = Path(raw)
        if candidate.exists():
            return candidate
        repo_candidate = REPO_ROOT / raw
        if repo_candidate.exists():
            return repo_candidate
    candidate = sweep_root / scenario
    if candidate.exists():
        return candidate
    raise ValueError(f"missing Phase412 artifact for {scenario}")


def build_rows_from_component_totals(
    *,
    scenario: str,
    artifact_dir: str,
    phase414_row: Mapping[str, str],
    mixed_components: Mapping[str, float],
    decode_components: Mapping[str, float],
    roofline_lower_bounds: Mapping[str, float],
    prefill_step_count: int,
) -> list[dict[str, str]]:
    phase414_prefill_gap = float(phase414_row["prefill_gap_ms"])
    phase414_decode_gap = float(phase414_row["decode_gap_ms"])
    target_missing = float(phase414_row["target_missing_ms"])
    reconstructed = phase414_prefill_gap + phase414_decode_gap
    reconstruction_error = _error_pct(reconstructed, target_missing)
    consistency_gate = "passed" if reconstruction_error <= 10.0 else "failed"

    roofline_gaps = {
        component: max(0.0, float(roofline_lower_bounds.get(component, 0.0)) - float(value))
        for component, value in mixed_components.items()
    }
    total_roofline_gap = sum(roofline_gaps.values())
    if total_roofline_gap > 0:
        verdict_component = max(roofline_gaps, key=roofline_gaps.get)
        mechanism_verdict = f"{verdict_component}_undercharged"
        phase416_target = f"audit_{verdict_component}_prefill_charge"
    else:
        mechanism_verdict = "offline_roofline_inconclusive"
        phase416_target = "gpu_profile_single_prefill_iteration"

    rows: list[dict[str, object]] = []
    for component, value in sorted(mixed_components.items()):
        lower = float(roofline_lower_bounds.get(component, 0.0))
        gap = max(0.0, lower - float(value))
        rows.append(
            {
                "source": SOURCE,
                "row_type": "mixed_component",
                "scenario": scenario,
                "artifact_dir": artifact_dir,
                "component": component,
                "sim_ms": float(value),
                "sim_mean_ms": _safe_ratio(float(value), prefill_step_count),
                "roofline_lower_bound_ms": lower,
                "roofline_gap_ms": gap,
                "roofline_status": "below_roofline" if gap > 0 else "ok",
                "component_gap_share": _safe_ratio(gap, total_roofline_gap),
                "phase414_prefill_gap_ms": phase414_prefill_gap,
                "phase414_decode_gap_ms": phase414_decode_gap,
                "target_missing_ms": target_missing,
                "reconstructed_missing_ms": reconstructed,
                "reconstruction_error_pct": reconstruction_error,
                "consistency_gate": consistency_gate,
                "prefill_step_count": prefill_step_count,
                "mechanism_verdict": mechanism_verdict,
                "phase416_target": phase416_target,
            }
        )
    for component, value in sorted(decode_components.items()):
        rows.append(
            {
                "source": SOURCE,
                "row_type": "decode_component",
                "scenario": scenario,
                "artifact_dir": artifact_dir,
                "component": component,
                "sim_ms": float(value),
                "sim_mean_ms": float(value),
                "roofline_lower_bound_ms": 0.0,
                "roofline_gap_ms": 0.0,
                "roofline_status": "not_checked_fixed_gap_candidate",
                "component_gap_share": 0.0,
                "phase414_prefill_gap_ms": phase414_prefill_gap,
                "phase414_decode_gap_ms": phase414_decode_gap,
                "target_missing_ms": target_missing,
                "reconstructed_missing_ms": reconstructed,
                "reconstruction_error_pct": reconstruction_error,
                "consistency_gate": consistency_gate,
                "prefill_step_count": prefill_step_count,
                "mechanism_verdict": mechanism_verdict,
                "phase416_target": phase416_target,
            }
        )
    rows.append(
        {
            "source": SOURCE,
            "row_type": "summary",
            "scenario": scenario,
            "artifact_dir": artifact_dir,
            "component": "summary",
            "sim_ms": sum(mixed_components.values()) + sum(decode_components.values()),
            "sim_mean_ms": _safe_ratio(sum(mixed_components.values()), prefill_step_count),
            "roofline_lower_bound_ms": sum(float(value) for value in roofline_lower_bounds.values()),
            "roofline_gap_ms": total_roofline_gap,
            "roofline_status": "below_roofline" if total_roofline_gap > 0 else "ok",
            "component_gap_share": 1.0,
            "phase414_prefill_gap_ms": phase414_prefill_gap,
            "phase414_decode_gap_ms": phase414_decode_gap,
            "target_missing_ms": target_missing,
            "reconstructed_missing_ms": reconstructed,
            "reconstruction_error_pct": reconstruction_error,
            "consistency_gate": consistency_gate,
            "prefill_step_count": prefill_step_count,
            "mechanism_verdict": mechanism_verdict,
            "phase416_target": phase416_target,
        }
    )
    for row in rows:
        row.update(
            {
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


def build_phase415_rows(
    *,
    phase414_csv: Path = PHASE414_CSV,
    sweep_root: Path = SWEEP_ROOT,
    scenario: str = DEFAULT_SCENARIO,
) -> list[dict[str, str]]:
    phase414_row = _read_csv_by_scenario(phase414_csv)[scenario]
    artifact_dir = _artifact_dir_from_phase414(phase414_row, sweep_root, scenario)
    gen_reqs_values = _full_prefill_gen_reqs(artifact_dir)
    mixed_components = _mixed_component_totals(gen_reqs_values)
    decode_components = _decode_components(batch=9, kv_len=phase414.DECODE_KV_BASE)
    roofline = _roofline_lower_bounds(len(gen_reqs_values))
    return build_rows_from_component_totals(
        scenario=scenario,
        artifact_dir=_display_path(artifact_dir),
        phase414_row=phase414_row,
        mixed_components=mixed_components,
        decode_components=decode_components,
        roofline_lower_bounds=roofline,
        prefill_step_count=len(gen_reqs_values),
    )


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Phase415 rows must not be empty")
    if sum(1 for row in rows if row.get("row_type") == "summary") != 1:
        raise ValueError("Phase415 expects exactly one summary row")
    for row in rows:
        if row.get("source") != SOURCE:
            raise ValueError("unexpected source")
        if row.get("phase405_penalty_read") != "false":
            raise ValueError("phase405_penalty_read must stay false")
        if row.get("gpu_allowed") != "false" or row.get("ssh_allowed") != "false":
            raise ValueError("Phase415 is offline and must not use GPU/SSH")
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


def write_phase415_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_phase415_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    summary = [row for row in rows if row["row_type"] == "summary"][0]
    mixed = [row for row in rows if row["row_type"] == "mixed_component"]
    decode = [row for row in rows if row["row_type"] == "decode_component"]
    lines = [
        "# Phase415 Prefill Charge Components",
        "",
        "Phase415 只做离线组件审计；不改 runtime、PerfDatabase 或 gate。",
        "",
        "## Verdict",
        "",
        f"- verdict: `{summary['mechanism_verdict']}`",
        f"- Phase416 target: `{summary['phase416_target']}`",
        f"- consistency gate: `{summary['consistency_gate']}`",
        f"- reconstructed missing ms: `{summary['reconstructed_missing_ms']}` / target `{summary['target_missing_ms']}`",
        "",
        "## Mixed Prefill Components",
        "",
        "| component | sim ms | sim mean ms | roofline lower ms | roofline gap ms | status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in mixed:
        lines.append(
            f"| {row['component']} | {row['sim_ms']} | {row['sim_mean_ms']} | "
            f"{row['roofline_lower_bound_ms']} | {row['roofline_gap_ms']} | {row['roofline_status']} |"
        )
    lines.extend(
        [
            "",
            "## Decode Components",
            "",
            "| component | sim ms | note |",
            "|---|---:|---|",
        ]
    )
    for row in decode:
        lines.append(f"| {row['component']} | {row['sim_ms']} | {row['roofline_status']} |")
    lines.extend(
        [
            "",
            "## Roofline Assumptions",
            "",
            f"- hidden={HIDDEN_SIZE}, inter={INTER_SIZE}, topk={TOPK}, moe_layers={MOE_LAYERS}, moe_ep={MOE_EP_SIZE}, attention_layers={ATTENTION_LAYERS}.",
            f"- attention roofline={H200_TENSOR_TFLOPS} TFLOP/s, MoE WNA16 conservative roofline={MOE_WNA16_TENSOR_TFLOPS} TFLOP/s, NVLink budget={NVLINK_GBPS} GB/s.",
            "- MoE lower bound uses EP-local token-expert rows and 6hi gated MLP work.",
            "- lower_bound 是物理下界，只用于找明显不可能的 undercharge，不代表真实耗时。",
            "",
            "## Boundary",
            "",
            "- GPU/SSH: not used in Phase415.",
            "- Runtime/PerfDatabase/gate: not modified.",
            "- Phase405 penalty: not read.",
            "- Default AIC: No-Go.",
            "",
        ]
    )
    return "\n".join(lines)


def write_phase415_md(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase415_md(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase414-csv", type=Path, default=PHASE414_CSV)
    parser.add_argument("--sweep-root", type=Path, default=SWEEP_ROOT)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_phase415_rows(
        phase414_csv=args.phase414_csv,
        sweep_root=args.sweep_root,
        scenario=args.scenario,
    )
    write_phase415_csv(args.csv, rows)
    write_phase415_md(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
