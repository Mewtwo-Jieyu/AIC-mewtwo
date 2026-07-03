#!/usr/bin/env python3
"""Phase406: validate DP pad-to-max lockstep as the DP2 duty-loss model."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = "phase406_dp_lockstep_model"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase406_dp_lockstep_model.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase406_dp_lockstep_model.md"
PHASE405_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase405_dp2_duty_cycle.csv"
PHASE403_RAW_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase403_dp2_stats"
DEFAULT_READINESS = "No-Go"

DP2_SCENARIOS = (
    "K2.5-tp4ep8dp2-8k2k",
    "K2.5-tp4ep8dp2-32k3k",
)
TP8_CONTROL = "K2.5-tp8ep8-32k3k"
ALL_SCENARIOS = DP2_SCENARIOS + (TP8_CONTROL,)

PAD_RULE_SOURCE = (
    "vllm/v1/worker/dp_utils.py:_post_process_dp_padding:L78-L90;"
    "vllm/v1/worker/dp_utils.py:_synchronize_dp_ranks:L153;"
    "vllm/v1/worker/gpu_model_runner.py:L3615-L3640"
)
PAPER_MODEL_RULE = "two_replica_step_tokens_max"

CSV_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "tp",
    "dp",
    "ep",
    "isl",
    "osl",
    "max_num_batched_tokens",
    "uncoupled_sim_output_tok_s_gpu",
    "real_output_tok_s_gpu",
    "uncoupled_ratio",
    "uncoupled_direction",
    "paper_model_rule",
    "lockstep_penalty",
    "duty_cycle_ratio",
    "active_iter_ratio",
    "coupled_sim_output_tok_s_gpu",
    "coupled_ratio",
    "coupled_error_pct",
    "sim_avg_decode_reqs_global",
    "sim_peak_decode_reqs_global",
    "real_running_global_mean",
    "real_running_global_max",
    "raw_batch_occupancy_ratio",
    "real_active_decode_ms_per_iter",
    "sim_decode_ms_per_iter",
    "pad_rule_source",
    "serve_prereq_status",
    "serve_prereq_evidence",
    "mechanism_verdict",
    "phase407_target",
    "gpu_allowed",
    "ssh_allowed",
    "runtime_modified",
    "perf_database",
    "valid_for_default",
    "diagnostic_only",
    "default_readiness",
]


@dataclass(frozen=True)
class ServePrereq:
    status: str
    evidence: str


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


def _safe_float(value: str) -> float | None:
    return float(value) if value else None


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return math.inf
    return numerator / denominator


def _direction(predicted: float, real: float) -> str:
    if predicted > real:
        return "sim_over_predicts_throughput"
    if predicted < real:
        return "sim_under_predicts_throughput"
    return "matched"


def _ratio_error_pct(ratio: float) -> float:
    return abs(ratio - 1.0) * 100.0


def _line_refs_for_patterns(path: Path, patterns: tuple[str, ...]) -> tuple[bool, str]:
    if not path.exists():
        return False, f"{path}:missing"
    refs: list[str] = []
    found_all = True
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for pattern in patterns:
        hit = next((idx for idx, line in enumerate(lines, start=1) if pattern in line), None)
        if hit is None:
            found_all = False
            refs.append(f"{path.name}:{pattern}:missing")
        else:
            refs.append(f"{path.name}:{hit}:{pattern}")
    return found_all, ";".join(refs)


def _serve_prereq_for_scenario(raw_root: Path, scenario: str) -> ServePrereq:
    ok, evidence = _line_refs_for_patterns(
        raw_root / scenario / "serve.log",
        (
            "data_parallel_size': 2",
            "enforce_eager=False",
            "enable_chunked_prefill=True",
            "CUDAGraphMode.FULL_AND_PIECEWISE",
            "Worker_DP0",
            "Worker_DP1",
        ),
    )
    return ServePrereq("passed" if ok else "failed", evidence)


def _row_type(phase405_row: dict[str, str]) -> str:
    if phase405_row["row_type"] == "tp8_control":
        return "tp8_early_exit_control"
    return "dp2_lockstep_paper_model"


def _lockstep_penalty(phase405_row: dict[str, str]) -> float:
    if phase405_row["row_type"] == "tp8_control":
        return 1.0
    return float(phase405_row["reconstructed_tput_ratio"])


def _mechanism_verdict(row_type: str, coupled_error_pct: float, serve_status: str) -> tuple[str, str]:
    if row_type == "tp8_early_exit_control":
        return "tp8_dp1_no_lockstep_change", "no_runtime_change"
    if serve_status != "passed":
        return "serve_prereq_missing_recheck_before_phase407", "recheck_phase403_prereqs"
    if coupled_error_pct <= 10.0:
        return "dp_pad_to_max_lockstep_model_matches_real", "runtime_dp_lockstep_coupling"
    return "dp_pad_to_max_lockstep_model_incomplete", "recheck_dp_padding_or_additional_bubble"


def _build_row(
    *,
    scenario: str,
    phase405: dict[str, str],
    raw_root: Path,
) -> dict[str, str]:
    row_type = _row_type(phase405)
    sim_output = float(phase405["phase401_sim_output_tok_s_gpu"])
    real_output = float(phase405["real_output_tok_s_gpu"])
    penalty = _lockstep_penalty(phase405)
    coupled_output = sim_output / penalty
    uncoupled_ratio = _safe_ratio(sim_output, real_output)
    coupled_ratio = _safe_ratio(coupled_output, real_output)
    prereq = (
        _serve_prereq_for_scenario(raw_root, scenario)
        if row_type == "dp2_lockstep_paper_model"
        else ServePrereq("not_applicable_dp1", "dp=1 early exit")
    )
    verdict, phase407_target = _mechanism_verdict(
        row_type,
        _ratio_error_pct(coupled_ratio),
        prereq.status,
    )
    row = {
        "source": SOURCE,
        "row_type": row_type,
        "scenario": scenario,
        "tp": int(phase405["tp"]),
        "dp": int(phase405["dp"]),
        "ep": int(phase405["ep"]),
        "isl": int(phase405["isl"]),
        "osl": int(phase405["osl"]),
        "max_num_batched_tokens": int(phase405["max_num_batched_tokens"]),
        "uncoupled_sim_output_tok_s_gpu": sim_output,
        "real_output_tok_s_gpu": real_output,
        "uncoupled_ratio": uncoupled_ratio,
        "uncoupled_direction": _direction(sim_output, real_output),
        "paper_model_rule": PAPER_MODEL_RULE,
        "lockstep_penalty": penalty,
        "duty_cycle_ratio": _safe_float(phase405.get("duty_cycle_ratio", "")),
        "active_iter_ratio": _safe_float(phase405.get("active_iter_ratio", "")),
        "coupled_sim_output_tok_s_gpu": coupled_output,
        "coupled_ratio": coupled_ratio,
        "coupled_error_pct": _ratio_error_pct(coupled_ratio),
        "sim_avg_decode_reqs_global": float(phase405["sim_avg_decode_reqs_global"]),
        "sim_peak_decode_reqs_global": float(phase405["sim_peak_decode_reqs_global"]),
        "real_running_global_mean": _safe_float(phase405.get("real_running_global_mean", "")),
        "real_running_global_max": _safe_float(phase405.get("real_running_global_max", "")),
        "raw_batch_occupancy_ratio": _safe_float(phase405.get("raw_batch_occupancy_ratio", "")),
        "real_active_decode_ms_per_iter": _safe_float(phase405.get("real_active_decode_ms_per_iter", "")),
        "sim_decode_ms_per_iter": _safe_float(phase405.get("sim_decode_ms_per_iter", "")),
        "pad_rule_source": PAD_RULE_SOURCE if row_type == "dp2_lockstep_paper_model" else "dp=1 no DP padding",
        "serve_prereq_status": prereq.status,
        "serve_prereq_evidence": prereq.evidence,
        "mechanism_verdict": verdict,
        "phase407_target": phase407_target,
        "gpu_allowed": False,
        "ssh_allowed": False,
        "runtime_modified": False,
        "perf_database": False,
        "valid_for_default": False,
        "diagnostic_only": True,
        "default_readiness": DEFAULT_READINESS,
    }
    return {field: _fmt(row.get(field)) for field in CSV_FIELDS}


def build_phase406_rows(
    *,
    phase405_csv: Path = PHASE405_CSV,
    phase403_raw_root: Path = PHASE403_RAW_ROOT,
) -> list[dict[str, str]]:
    phase405 = _read_csv_by_scenario(phase405_csv)
    rows = [
        _build_row(
            scenario=scenario,
            phase405=phase405[scenario],
            raw_root=phase403_raw_root,
        )
        for scenario in DP2_SCENARIOS
    ]
    rows.append(
        _build_row(
            scenario=TP8_CONTROL,
            phase405=phase405[TP8_CONTROL],
            raw_root=phase403_raw_root,
        )
    )
    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    for row in rows:
        if row.get("source") != SOURCE:
            raise ValueError("source")
        if row.get("gpu_allowed") != "false":
            raise ValueError("gpu_allowed")
        if row.get("ssh_allowed") != "false":
            raise ValueError("ssh_allowed")
        for field in ("runtime_modified", "perf_database", "valid_for_default"):
            if row.get(field) != "false":
                raise ValueError(field)
        if row.get("diagnostic_only") != "true":
            raise ValueError("diagnostic_only")
        if row.get("default_readiness") != DEFAULT_READINESS:
            raise ValueError("default_readiness")
        if row["row_type"] == "dp2_lockstep_paper_model":
            if row["paper_model_rule"] != PAPER_MODEL_RULE:
                raise ValueError("paper_model_rule")
            if row["serve_prereq_status"] != "passed":
                raise ValueError("serve_prereq_status")
    if len(rows) != 3:
        raise ValueError("Phase406 must contain two DP2 rows plus one TP8 control")
    if {row["scenario"] for row in rows} != set(ALL_SCENARIOS):
        raise ValueError("Phase406 scenario set mismatch")


def write_phase406_csv(path: Path, rows: list[dict[str, str]] | None = None) -> None:
    rows = build_phase406_rows() if rows is None else rows
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_phase406_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    dp_rows = [row for row in rows if row["row_type"] == "dp2_lockstep_paper_model"]
    matched = all(float(row["coupled_error_pct"]) <= 10.0 for row in dp_rows)
    summary = (
        "Phase406 validates DP pad-to-max lockstep as the structural model for "
        "the DP2 duty loss. Coupling the two replicas by max step tokens brings "
        "both DP2 scenarios back to the clean real baseline within 10%."
        if matched
        else "Phase406 does not fully explain the DP2 duty loss; inspect the residual rows before Phase407."
    )
    lines = [
        "# Phase406 DP lockstep structural model",
        "",
        summary,
        "",
        "## Mechanism",
        "",
        f"- pad_rule_source: `{PAD_RULE_SOURCE}`.",
        "- serve prerequisites checked from Phase403 logs: data_parallel_size=2, enforce_eager=false, chunked prefill enabled, FULL_AND_PIECEWISE CUDA graph mode, and both DP workers present.",
        "- paper model: consume the Phase405 scheduler/metrics decomposition and charge the uncoupled DP2 output by the deterministic `step_tokens = max(tokens_dp0, tokens_dp1)` lockstep penalty. This is not a runtime implementation.",
        "",
        "## Result",
        "",
        "| scenario | row_type | uncoupled ratio | penalty | coupled ratio | error pct | verdict |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {row_type} | {uncoupled_ratio} | {lockstep_penalty} | {coupled_ratio} | {coupled_error_pct} | {mechanism_verdict} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- report-only; runtime_modified=false, perf_database=false.",
            "- gpu_allowed=false, ssh_allowed=false.",
            "- valid_for_default=false, diagnostic_only=true, default_readiness=No-Go.",
            "- Phase407 target: implement DP lockstep coupling in runtime only if this report is accepted.",
            "",
        ]
    )
    return "\n".join(lines)


def write_phase406_md(path: Path, rows: list[dict[str, str]] | None = None) -> None:
    rows = build_phase406_rows() if rows is None else rows
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase406_md(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase405-csv", type=Path, default=PHASE405_CSV)
    parser.add_argument("--phase403-raw-root", type=Path, default=PHASE403_RAW_ROOT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_phase406_rows(
        phase405_csv=args.phase405_csv,
        phase403_raw_root=args.phase403_raw_root,
    )
    write_phase406_csv(args.csv, rows)
    write_phase406_md(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
