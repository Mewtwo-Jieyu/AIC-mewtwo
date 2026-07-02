#!/usr/bin/env python3
"""Phase401: diagnose non-preemptive waiting admission against Phase400 clean data."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aiconfigurator.sdk import common  # noqa: E402
from aiconfigurator.sdk.config import RuntimeConfig  # noqa: E402
from scripts import validate_cb_simulator as validate_cb  # noqa: E402


SOURCE = "phase401_preempt_admission"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase401_preempt_admission.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase401_preempt_admission.md"
PHASE400_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase400_clean_recollect.csv"
DEFAULT_READINESS = "No-Go"

CLEAN_SCENARIOS = (
    "K2.5-tp8ep8-8k2k",
    "K2.5-tp8ep8-32k3k",
    "K2.5-tp4ep8dp2-8k2k",
    "K2.5-tp4ep8dp2-32k3k",
)

CSV_FIELDS = [
    "source",
    "scenario",
    "tp",
    "dp",
    "ep",
    "isl",
    "osl",
    "max_num_batched_tokens",
    "num_gpu_blocks",
    "phase400_clean_output_tok_s_gpu",
    "phase401_sim_output_tok_s_gpu",
    "phase401_error_ratio",
    "phase401_direction",
    "phase400_running_reqs_max",
    "phase400_running_reqs_mean",
    "sim_avg_prefill_reqs_per_iter",
    "sim_avg_decode_reqs_per_iter",
    "sim_peak_decode_reqs_per_iter",
    "sim_peak_decode_vs_phase400_running_max_ratio",
    "sim_avg_tokens_per_iter",
    "sim_peak_tokens_per_iter",
    "steady_state_iterations",
    "steady_state_time_ms",
    "vllm_scheduler_source_url",
    "vllm_running_preempt_line_range",
    "vllm_waiting_no_preempt_line_range",
    "scheduler_change",
    "runtime_modified",
    "capacity_formula_modified",
    "perf_database",
    "valid_for_default",
    "diagnostic_only",
    "default_readiness",
]

VLLM_SCHEDULER_URL = (
    "https://raw.githubusercontent.com/vllm-project/vllm/v0.19.0/"
    "vllm/v1/core/sched/scheduler.py"
)


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isinf(value) or math.isnan(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def _read_csv_by_scenario(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        return {row["scenario"]: row for row in csv.DictReader(f)}


def _point_by_name() -> dict[str, validate_cb.MultiConfigPoint]:
    return {point.name: point for point in validate_cb.MULTI_CONFIG_DATA}


def _safe_error_ratio(predicted: float, real: float) -> float:
    if predicted <= 0 or real <= 0:
        return math.inf
    return max(predicted, real) / min(predicted, real)


def _direction(predicted: float, real: float) -> str:
    if predicted > real:
        return "sim_over_predicts_throughput"
    if predicted < real:
        return "sim_under_predicts_throughput"
    return "matched"


def _required_metric(mapping: dict, key: str, scenario: str) -> float:
    try:
        return float(mapping[key])
    except KeyError as exc:
        raise ValueError(f"missing cb_sim_scheduling.{key} for {scenario}") from exc


def run_phase401_budget_rows() -> list[SimpleNamespace]:
    """Run the 4 clean scenarios with Phase400 log-derived KV capacity."""
    phase400 = _read_csv_by_scenario(PHASE400_CSV)
    points = _point_by_name()
    backend = validate_cb.VLLMBackend()
    loaded: dict[tuple[int, int, int, int], tuple] = {}
    rows: list[SimpleNamespace] = []

    for scenario in CLEAN_SCENARIOS:
        point = points[scenario]
        key = (point.tp, point.dp, point.moe_tp, point.moe_ep)
        if key not in loaded:
            loaded[key] = validate_cb._load_model_and_db(
                tp=point.tp,
                dp=point.dp,
                moe_tp=point.moe_tp,
                moe_ep=point.moe_ep,
            )[:2]
        model, db = loaded[key]
        cb_config = validate_cb._make_cb_config(
            point.isl,
            point.batch_size,
            max_num_batched_tokens=point.max_num_batched_tokens,
            overlap_factor=0.0,
            per_iteration_overhead_ms=0.0,
            num_gpu_blocks=int(phase400[scenario]["num_gpu_blocks"]),
        )
        summary = backend.run_agg(
            model,
            db,
            RuntimeConfig(batch_size=point.batch_size, isl=point.isl, osl=point.osl),
            ctx_tokens=point.max_num_batched_tokens,
            database_mode=common.DatabaseMode.HYBRID,
            method="cb_sim",
            cb_config=cb_config,
        )
        result = summary.get_result_dict()
        if not result or "tokens/s/gpu" not in result:
            raise ValueError(f"missing cb_sim throughput for {scenario}")
        scheduling = summary.get_per_ops_data().get("cb_sim_scheduling")
        if not isinstance(scheduling, dict):
            raise ValueError(f"missing cb_sim_scheduling for {scenario}")
        rows.append(
            SimpleNamespace(
                name=scenario,
                sim_output_tok_s_gpu=float(result["tokens/s/gpu"]),
                avg_prefill_reqs_per_iter=_required_metric(
                    scheduling,
                    "avg_prefill_reqs_per_iter",
                    scenario,
                ),
                avg_decode_reqs_per_iter=_required_metric(
                    scheduling,
                    "avg_decode_reqs_per_iter",
                    scenario,
                ),
                avg_tokens_per_iter=_required_metric(
                    scheduling,
                    "avg_tokens_per_iter",
                    scenario,
                ),
                peak_prefill_reqs_per_iter=_required_metric(
                    scheduling,
                    "peak_prefill_reqs_per_iter",
                    scenario,
                ),
                peak_decode_reqs_per_iter=_required_metric(
                    scheduling,
                    "peak_decode_reqs_per_iter",
                    scenario,
                ),
                peak_tokens_per_iter=_required_metric(
                    scheduling,
                    "peak_tokens_per_iter",
                    scenario,
                ),
                steady_state_iterations=int(
                    _required_metric(scheduling, "steady_state_iterations", scenario)
                ),
                steady_state_time_ms=_required_metric(
                    scheduling,
                    "steady_state_time_ms",
                    scenario,
                ),
            )
        )
    return rows


def _base_row() -> dict[str, object]:
    return {
        "source": SOURCE,
        "vllm_scheduler_source_url": VLLM_SCHEDULER_URL,
        "vllm_running_preempt_line_range": "385-516",
        "vllm_waiting_no_preempt_line_range": "564-791",
        "scheduler_change": "waiting_admission_no_running_preempt",
        "runtime_modified": True,
        "capacity_formula_modified": False,
        "perf_database": False,
        "valid_for_default": False,
        "diagnostic_only": True,
        "default_readiness": DEFAULT_READINESS,
    }


def build_phase401_rows(
    *,
    budget_rows: Iterable[object] | None = None,
    phase400_csv: Path = PHASE400_CSV,
) -> list[dict[str, str]]:
    phase400 = _read_csv_by_scenario(phase400_csv)
    points = _point_by_name()
    budget_by_name = {
        row.name: row for row in (run_phase401_budget_rows() if budget_rows is None else budget_rows)
    }
    rows: list[dict[str, str]] = []

    for scenario in CLEAN_SCENARIOS:
        point = points[scenario]
        clean = phase400[scenario]
        budget = budget_by_name[scenario]
        real = float(clean["output_tok_s_gpu"])
        sim = float(budget.sim_output_tok_s_gpu)
        running_max = (
            int(float(clean["serve_running_reqs_max"]))
            if clean["serve_running_reqs_max"]
            else None
        )
        peak_decode = float(budget.peak_decode_reqs_per_iter)

        row = _base_row()
        row.update(
            {
                "scenario": scenario,
                "tp": point.tp,
                "dp": point.dp,
                "ep": point.moe_ep,
                "isl": point.isl,
                "osl": point.osl,
                "max_num_batched_tokens": point.max_num_batched_tokens,
                "num_gpu_blocks": int(clean["num_gpu_blocks"]),
                "phase400_clean_output_tok_s_gpu": real,
                "phase401_sim_output_tok_s_gpu": sim,
                "phase401_error_ratio": _safe_error_ratio(sim, real),
                "phase401_direction": _direction(sim, real),
                "phase400_running_reqs_max": running_max,
                "phase400_running_reqs_mean": (
                    float(clean["serve_running_reqs_mean"])
                    if clean["serve_running_reqs_mean"]
                    else None
                ),
                "sim_avg_prefill_reqs_per_iter": float(
                    budget.avg_prefill_reqs_per_iter
                ),
                "sim_avg_decode_reqs_per_iter": float(budget.avg_decode_reqs_per_iter),
                "sim_peak_decode_reqs_per_iter": peak_decode,
                "sim_peak_decode_vs_phase400_running_max_ratio": (
                    peak_decode / running_max if running_max else None
                ),
                "sim_avg_tokens_per_iter": float(budget.avg_tokens_per_iter),
                "sim_peak_tokens_per_iter": float(budget.peak_tokens_per_iter),
                "steady_state_iterations": int(budget.steady_state_iterations),
                "steady_state_time_ms": float(budget.steady_state_time_ms),
            }
        )
        rows.append({field: _fmt(row.get(field, "")) for field in CSV_FIELDS})

    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 4:
        raise ValueError("Phase401 must contain exactly 4 clean scenario rows")
    if {row["scenario"] for row in rows} != set(CLEAN_SCENARIOS):
        raise ValueError("Phase401 scenario set mismatch")
    for row in rows:
        if row["runtime_modified"] != "true":
            raise ValueError("runtime_modified")
        if row["capacity_formula_modified"] != "false":
            raise ValueError("capacity_formula_modified")
        for field in ("perf_database", "valid_for_default"):
            if row[field] != "false":
                raise ValueError(field)
        if row["diagnostic_only"] != "true":
            raise ValueError("diagnostic_only")
        if row["default_readiness"] != DEFAULT_READINESS:
            raise ValueError("default_readiness")


def write_phase401_csv(
    path: Path = DEFAULT_CSV,
    rows: list[dict[str, str]] | None = None,
) -> None:
    rows = build_phase401_rows() if rows is None else rows
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_phase401_md(rows: list[dict[str, str]]) -> str:
    lines = [
        "# Phase401 preempt admission",
        "",
        "Phase401 changes only the CB scheduler waiting-admission rule: a waiting "
        "request may use free capacity, but it must not preempt an already-running "
        "request. Capacity formulas, PerfDatabase rows, and acceptance gates are unchanged.",
        "",
        "| scenario | clean tok/s/gpu | sim tok/s/gpu | error | direction | running max | sim peak decode |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {phase400_clean_output_tok_s_gpu} | "
            "{phase401_sim_output_tok_s_gpu} | {phase401_error_ratio} | "
            "{phase401_direction} | {phase400_running_reqs_max} | "
            "{sim_peak_decode_reqs_per_iter} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Source check:",
            "",
            f"- vLLM v0.19.0 scheduler source: `{VLLM_SCHEDULER_URL}`.",
            "- Lines 385-516 schedule running requests and preempt only when running KV allocation fails.",
            "- Lines 564-791 schedule waiting requests; if `allocate_slots` returns `None`, waiting admission stops instead of preempting running requests.",
            "",
            "Boundary:",
            "",
            "- `runtime_modified=true`, but only for scheduler admission semantics.",
            "- `capacity_formula_modified=false`, `perf_database=false`, `valid_for_default=false`, `default_readiness=No-Go`.",
            "",
            "Next: Phase402 should decide how to generalize capacity formulas from model/system memory, not from serve logs.",
            "",
        ]
    )
    return "\n".join(lines)


def write_phase401_md(
    path: Path = DEFAULT_MD,
    rows: list[dict[str, str]] | None = None,
) -> None:
    rows = build_phase401_rows() if rows is None else rows
    path.write_text(render_phase401_md(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    rows = build_phase401_rows()
    write_phase401_csv(args.csv, rows)
    write_phase401_md(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
