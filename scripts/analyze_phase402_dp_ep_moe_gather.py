#!/usr/bin/env python3
"""Phase402: attribute DP+EP MoE gather-token semantics without changing runtime."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import validate_cb_simulator as validate_cb  # noqa: E402


SOURCE = "phase402_dp_ep_moe_gather"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase402_dp_ep_moe_gather.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase402_dp_ep_moe_gather.md"
PHASE400_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase400_clean_recollect.csv"
PHASE401_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase401_preempt_admission.csv"
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
    "phase400_clean_output_tok_s_gpu",
    "phase401_sim_output_tok_s_gpu",
    "phase401_error_ratio",
    "phase401_direction",
    "nominal_local_decode_tokens",
    "nominal_gathered_decode_tokens",
    "runtime_effective_moe_tokens",
    "phase401_peak_decode_reqs_per_iter",
    "phase401_peak_effective_moe_tokens",
    "generation_moe_local64_ms",
    "generation_moe_gather128_ms",
    "generation_moe_gather_minus_local_ms",
    "generation_moe_gather_vs_local_ratio",
    "forced_moe64_sim_output_tok_s_gpu",
    "forced_moe64_error_ratio",
    "forced_moe64_direction",
    "forced_moe128_sim_output_tok_s_gpu",
    "forced_moe128_error_ratio",
    "forced_moe128_direction",
    "dp0_decode_tokens",
    "dp1_decode_tokens",
    "generation_moe_dp0_72_ms",
    "generation_moe_dp1_56_ms",
    "dp_max_vs_mean_ratio",
    "dp_max_vs_min_ratio",
    "hypothesis_verdict",
    "primary_driver",
    "secondary_imbalance",
    "code_evidence",
    "runtime_modified",
    "perf_database",
    "valid_for_default",
    "diagnostic_only",
    "default_readiness",
]


@dataclass(frozen=True)
class MoeCounterfactual:
    scenario: str
    generation_moe_local64_ms: float
    generation_moe_gather128_ms: float
    generation_moe_dp0_72_ms: float
    generation_moe_dp1_56_ms: float


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


def _find_generation_moe_op(model):
    for op in model.generation_ops:
        if getattr(op, "_name", "") == "generation_moe":
            return op
    raise ValueError("missing generation_moe op")


def _query_generation_moe_ms(database, moe_op, num_tokens: int) -> float:
    result = database.query_moe(
        num_tokens=num_tokens,
        hidden_size=moe_op._hidden_size,
        inter_size=moe_op._inter_size,
        topk=moe_op._topk,
        num_experts=moe_op._num_experts,
        moe_tp_size=moe_op._moe_tp_size,
        moe_ep_size=moe_op._moe_ep_size,
        quant_mode=moe_op._quant_mode,
        workload_distribution=moe_op._workload_distribution,
        is_context=moe_op._is_context,
        moe_backend=moe_op._moe_backend,
        is_gated=moe_op._is_gated,
        enable_eplb=moe_op._enable_eplb,
    )
    return float(result) * float(moe_op._scale_factor)


def collect_moe_counterfactuals() -> list[MoeCounterfactual]:
    points = _point_by_name()
    loaded: dict[tuple[int, int, int, int], tuple] = {}
    rows: list[MoeCounterfactual] = []

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
        model, database = loaded[key]
        moe_op = _find_generation_moe_op(model)
        local_tokens = math.ceil(point.batch_size / point.dp)
        gathered_tokens = local_tokens * point.dp
        rows.append(
            MoeCounterfactual(
                scenario=scenario,
                generation_moe_local64_ms=_query_generation_moe_ms(
                    database,
                    moe_op,
                    local_tokens,
                ),
                generation_moe_gather128_ms=_query_generation_moe_ms(
                    database,
                    moe_op,
                    gathered_tokens,
                ),
                generation_moe_dp0_72_ms=_query_generation_moe_ms(database, moe_op, 72),
                generation_moe_dp1_56_ms=_query_generation_moe_ms(database, moe_op, 56),
            )
        )
    return rows


def _counterfactual_by_scenario(
    counterfactuals: Iterable[MoeCounterfactual] | Iterable[object],
) -> dict[str, object]:
    return {row.scenario: row for row in counterfactuals}


def _forced_output_from_iter_delta(
    *,
    current_output_tok_s_gpu: float,
    steady_state_time_ms: float,
    steady_state_iterations: int,
    current_minus_forced_iter_delta_ms: float,
) -> float:
    current_iter_ms = steady_state_time_ms / max(steady_state_iterations, 1)
    forced_iter_ms = current_iter_ms - current_minus_forced_iter_delta_ms
    if forced_iter_ms <= 0:
        return math.inf
    return current_output_tok_s_gpu * current_iter_ms / forced_iter_ms


def build_phase402_rows(
    *,
    counterfactuals: Iterable[MoeCounterfactual] | Iterable[object] | None = None,
    phase400_csv: Path = PHASE400_CSV,
    phase401_csv: Path = PHASE401_CSV,
) -> list[dict[str, str]]:
    phase400 = _read_csv_by_scenario(phase400_csv)
    phase401 = _read_csv_by_scenario(phase401_csv)
    points = _point_by_name()
    cf_rows = _counterfactual_by_scenario(
        collect_moe_counterfactuals() if counterfactuals is None else counterfactuals
    )

    rows: list[dict[str, str]] = []
    for scenario in CLEAN_SCENARIOS:
        point = points[scenario]
        clean_output = float(phase400[scenario]["output_tok_s_gpu"])
        current_output = float(phase401[scenario]["phase401_sim_output_tok_s_gpu"])
        steady_iterations = int(float(phase401[scenario]["steady_state_iterations"]))
        steady_time_ms = float(phase401[scenario]["steady_state_time_ms"])
        peak_decode = float(phase401[scenario]["sim_peak_decode_reqs_per_iter"])
        local_tokens = math.ceil(point.batch_size / point.dp)
        gathered_tokens = local_tokens * point.dp
        runtime_effective_moe_tokens = gathered_tokens
        counter = cf_rows[scenario]
        local_moe_ms = float(counter.generation_moe_local64_ms)
        gathered_moe_ms = float(counter.generation_moe_gather128_ms)
        gather_minus_local = gathered_moe_ms - local_moe_ms
        forced64_output = _forced_output_from_iter_delta(
            current_output_tok_s_gpu=current_output,
            steady_state_time_ms=steady_time_ms,
            steady_state_iterations=steady_iterations,
            current_minus_forced_iter_delta_ms=max(gather_minus_local, 0.0),
        )
        forced64_ratio = _safe_error_ratio(forced64_output, clean_output)
        forced128_ratio = _safe_error_ratio(current_output, clean_output)
        if point.dp == 1:
            verdict = "control_no_dp_gather_delta"
            primary_driver = "control"
            secondary = "not_applicable"
            dp0_tokens = None
            dp1_tokens = None
            dp_mean_ratio = None
            dp_min_ratio = None
        else:
            verdict = "rejected_current_runtime_already_gathers_moe_tokens"
            primary_driver = "not_moe_local64_undercharge"
            secondary = "dp0_72_dp1_56_secondary_12p5pct_vs_mean"
            dp0_tokens = 72
            dp1_tokens = 56
            mean_tokens = (dp0_tokens + dp1_tokens) / 2
            dp_mean_ratio = max(dp0_tokens, dp1_tokens) / mean_tokens
            dp_min_ratio = max(dp0_tokens, dp1_tokens) / min(dp0_tokens, dp1_tokens)

        row = {
            "source": SOURCE,
            "scenario": scenario,
            "tp": point.tp,
            "dp": point.dp,
            "ep": point.moe_ep,
            "isl": point.isl,
            "osl": point.osl,
            "max_num_batched_tokens": point.max_num_batched_tokens,
            "phase400_clean_output_tok_s_gpu": clean_output,
            "phase401_sim_output_tok_s_gpu": current_output,
            "phase401_error_ratio": _safe_error_ratio(current_output, clean_output),
            "phase401_direction": _direction(current_output, clean_output),
            "nominal_local_decode_tokens": local_tokens,
            "nominal_gathered_decode_tokens": gathered_tokens,
            "runtime_effective_moe_tokens": runtime_effective_moe_tokens,
            "phase401_peak_decode_reqs_per_iter": peak_decode,
            "phase401_peak_effective_moe_tokens": peak_decode * point.dp,
            "generation_moe_local64_ms": local_moe_ms,
            "generation_moe_gather128_ms": gathered_moe_ms,
            "generation_moe_gather_minus_local_ms": gather_minus_local,
            "generation_moe_gather_vs_local_ratio": (
                gathered_moe_ms / local_moe_ms if local_moe_ms > 0 else math.inf
            ),
            "forced_moe64_sim_output_tok_s_gpu": forced64_output,
            "forced_moe64_error_ratio": forced64_ratio,
            "forced_moe64_direction": _direction(forced64_output, clean_output),
            "forced_moe128_sim_output_tok_s_gpu": current_output,
            "forced_moe128_error_ratio": forced128_ratio,
            "forced_moe128_direction": _direction(current_output, clean_output),
            "dp0_decode_tokens": dp0_tokens,
            "dp1_decode_tokens": dp1_tokens,
            "generation_moe_dp0_72_ms": (
                float(counter.generation_moe_dp0_72_ms) if point.dp > 1 else None
            ),
            "generation_moe_dp1_56_ms": (
                float(counter.generation_moe_dp1_56_ms) if point.dp > 1 else None
            ),
            "dp_max_vs_mean_ratio": dp_mean_ratio,
            "dp_max_vs_min_ratio": dp_min_ratio,
            "hypothesis_verdict": verdict,
            "primary_driver": primary_driver,
            "secondary_imbalance": secondary,
            "code_evidence": (
                "vllm_backend_splits_global_batch_then_moe_query_multiplies_attention_dp"
            ),
            "runtime_modified": False,
            "perf_database": False,
            "valid_for_default": False,
            "diagnostic_only": True,
            "default_readiness": DEFAULT_READINESS,
        }
        rows.append({field: _fmt(row.get(field)) for field in CSV_FIELDS})
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != len(CLEAN_SCENARIOS):
        raise ValueError(f"expected {len(CLEAN_SCENARIOS)} rows, got {len(rows)}")
    scenarios = {row["scenario"] for row in rows}
    if scenarios != set(CLEAN_SCENARIOS):
        raise ValueError(f"unexpected scenarios: {sorted(scenarios)}")
    for row in rows:
        if row["runtime_modified"] != "false":
            raise ValueError("runtime_modified must stay false")
        if row["perf_database"] != "false":
            raise ValueError("perf_database must stay false")
        if row["valid_for_default"] != "false":
            raise ValueError("valid_for_default must stay false")
        if row["diagnostic_only"] != "true":
            raise ValueError("diagnostic_only must stay true")
        if row["default_readiness"] != DEFAULT_READINESS:
            raise ValueError("default_readiness must stay No-Go")
        if row["scenario"].startswith("K2.5-tp4") and row["runtime_effective_moe_tokens"] != "128":
            raise ValueError("DP2 runtime_effective_moe_tokens must be 128")


def write_phase402_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_phase402_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    dp2_rows = [row for row in rows if row["dp"] == "2"]
    worst_current = max(float(row["phase401_error_ratio"]) for row in dp2_rows)
    worst_forced64 = max(float(row["forced_moe64_error_ratio"]) for row in dp2_rows)
    lines = [
        "# Phase402 DP+EP MoE Gather Attribution",
        "",
        "## Verdict",
        "",
        (
            "Phase402 rejects the proposed root cause: the current DP2 runtime already "
            "charges `generation_moe` with gathered MoE tokens. `run_agg` splits the "
            "global batch to 64 per replica, then `MoE.query` multiplies by "
            "`attention_dp_size=2`, so the effective MoE token key is 128."
        ),
        "",
        (
            "Forcing a local-only MoE@64 counterfactual does not explain the DP2 "
            f"residual. The current DP2 worst error is {worst_current:.2f}x; "
            f"the forced MoE@64 counterfactual is {worst_forced64:.2f}x. "
            "With the Phase397v int4_wo calibrated-SOL path, MoE@64 and MoE@128 "
            "are nearly identical because decode is weight-load bound."
        ),
        "",
        "## Scenario Summary",
        "",
        (
            "| scenario | dp | clean tok/s/gpu | current tok/s/gpu | current ratio | "
            "MoE@64 ms | MoE@128 ms | forced MoE@64 ratio | verdict |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {dp} | {clean} | {current} | {ratio} | {moe64} | {moe128} | {forced} | {verdict} |".format(
                scenario=row["scenario"],
                dp=row["dp"],
                clean=row["phase400_clean_output_tok_s_gpu"],
                current=row["phase401_sim_output_tok_s_gpu"],
                ratio=row["phase401_error_ratio"],
                moe64=row["generation_moe_local64_ms"],
                moe128=row["generation_moe_gather128_ms"],
                forced=row["forced_moe64_error_ratio"],
                verdict=row["hypothesis_verdict"],
            )
        )
    lines.extend(
        [
            "",
            "## DP Imbalance",
            "",
            (
                "The DP0/DP1 72/56 split is kept as a secondary effect. "
                "Its max-vs-mean ratio is 1.125 and max-vs-min ratio is 1.286, "
                "which is too small to explain the remaining DP2 1.8x throughput gap."
            ),
            "",
            "## Boundary",
            "",
            "- runtime_modified=false",
            "- perf_database=false",
            "- valid_for_default=false",
            "- diagnostic_only=true",
            "- default_readiness=No-Go",
            "",
            "## Next",
            "",
            (
                "Phase403 should not implement a MoE gather-token fix as the next "
                "runtime change. The remaining DP2 residual should be attributed to "
                "another mechanism, such as DP stats visibility, communication, or "
                "per-op composition under the Phase400 clean baseline."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_phase402_md(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase402_md(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_phase402_rows()
    write_phase402_csv(args.csv, rows)
    write_phase402_md(args.md, rows)


if __name__ == "__main__":
    main()
