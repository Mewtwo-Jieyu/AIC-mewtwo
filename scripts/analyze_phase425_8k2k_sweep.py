#!/usr/bin/env python3
"""Phase425: DP2 8k2k burst-vs-arrival sweep attribution."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import analyze_phase412_arrival_sweep as phase412


SOURCE = "phase425_8k2k_sweep"
DEFAULT_SCENARIO = "K2.5-tp4ep8dp2-8k2k"
BASELINE_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase403_dp2_stats"
SWEEP_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase425_8k2k_sweep"
PHASE407_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase407_dp_lockstep_joint_sim.csv"
PHASE412_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase412_arrival_sweep.csv"
VALIDATION_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase424_bt65536_recollect.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase425_8k2k_sweep.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase425_8k2k_sweep.md"
DEFAULT_READINESS = "No-Go"

CSV_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "artifact_dir",
    "num_prompts",
    "max_concurrency",
    "uncoupled_output_tok_s_gpu",
    "bench_output_tok_s_gpu",
    "real_output_source",
    "overall_penalty",
    "steady_output_tok_s_gpu",
    "steady_penalty",
    "trace_bench_error_pct",
    "trace_fidelity_gate",
    "engine0_prompt_tokens",
    "engine1_prompt_tokens",
    "engine0_generation_tokens",
    "engine1_generation_tokens",
    "engine0_request_success_count",
    "engine1_request_success_count",
    "engine0_request_count_est",
    "engine1_request_count_est",
    "request_count_ratio",
    "generation_token_ratio",
    "request_imbalance_monotonic_down",
    "penalty_monotonic_down",
    "last_request_count_ratio",
    "baseline_overall_penalty",
    "last_overall_penalty",
    "last_steady_penalty",
    "burst_artifact_multiplier",
    "validation_real_output_tok_s_gpu",
    "validation_sim_output_tok_s_gpu",
    "validation_baseline_penalty",
    "validation_last_overall_penalty",
    "validation_last_steady_penalty",
    "validation_burst_artifact_multiplier",
    "phase412_32k_request_count_ratio",
    "phase412_32k_overall_penalty",
    "phase412_32k_steady_penalty",
    "phase412_32k_verdict",
    "mechanism_verdict",
    "phase426_target",
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


def _float(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    return float(raw) if raw else math.nan


def _safe_ratio(a: float, b: float) -> float:
    if b <= 0:
        return math.inf
    return a / b


def _read_phase412_trend(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if (
                row.get("row_type") == "trend"
                and row.get("scenario") == "K2.5-tp4ep8dp2-32k3k"
            ):
                return row
    return {}


def _read_validation_baseline(path: Path, scenario: str) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("row_type") != "config" or row.get("scenario") != scenario:
                continue
            real = float(row["real_output_tok_s_gpu"])
            sim = float(row.get("phase424_sim_tok_s_gpu") or row.get("phase422_sim_tok_s_gpu"))
            return {
                "real": real,
                "sim": sim,
                "penalty": _safe_ratio(sim, real),
            }
    return {}


def _phase425_verdict(
    *,
    request_down: bool,
    last_ratio: float,
    baseline_overall: float,
    last_steady: float,
) -> tuple[str, str]:
    artifact_multiplier = _safe_ratio(baseline_overall, last_steady)
    if request_down and last_ratio <= 1.10 and artifact_multiplier >= 1.25:
        return (
            "8k2k_burst_artifact_significant",
            "decide_validation_arrival_mode_or_steady_gap_modeling",
        )
    if last_steady >= 2.0:
        return (
            "8k2k_steady_model_gap_dominates",
            "profile_or_model_8k2k_steady_gap",
        )
    return (
        "8k2k_mixed_artifact_and_steady_gap",
        "separate_arrival_artifact_from_remaining_8k2k_gap",
    )


def _adapt_phase412_row(row: dict[str, str]) -> dict[str, str]:
    adapted = {field: "" for field in CSV_FIELDS}
    for field in CSV_FIELDS:
        if field in row:
            adapted[field] = row[field]
    adapted["source"] = SOURCE
    adapted["phase426_target"] = row.get("phase413_target", "")
    return adapted


def build_phase425_rows(
    *,
    baseline_root: Path = BASELINE_ROOT,
    sweep_root: Path = SWEEP_ROOT,
    phase407_csv: Path = PHASE407_CSV,
    phase412_csv: Path = PHASE412_CSV,
    validation_csv: Path = VALIDATION_CSV,
    scenario: str = DEFAULT_SCENARIO,
) -> list[dict[str, str]]:
    base_rows = phase412.build_phase412_rows(
        baseline_root=baseline_root,
        sweep_root=sweep_root,
        phase407_csv=phase407_csv,
        scenario=scenario,
    )
    rows = [_adapt_phase412_row(row) for row in base_rows]
    samples = sorted(
        [row for row in rows if row["row_type"] == "sample"],
        key=lambda item: int(item["num_prompts"]),
    )
    trend = [row for row in rows if row["row_type"] == "trend"][0]
    baseline = samples[0]
    last = samples[-1]

    baseline_overall = _float(baseline, "overall_penalty")
    last_overall = _float(last, "overall_penalty")
    last_steady = _float(last, "steady_penalty")
    artifact_multiplier = _safe_ratio(baseline_overall, last_steady)
    request_down = trend["request_imbalance_monotonic_down"] == "true"
    last_ratio = _float(trend, "last_request_count_ratio")
    verdict, target = _phase425_verdict(
        request_down=request_down,
        last_ratio=last_ratio,
        baseline_overall=baseline_overall,
        last_steady=last_steady,
    )

    phase412_trend = _read_phase412_trend(phase412_csv)
    validation = _read_validation_baseline(validation_csv, scenario)
    validation_real = validation.get("real", math.nan)
    validation_sim = validation.get("sim", math.nan)
    validation_baseline = validation.get("penalty", math.nan)
    validation_last_overall = _safe_ratio(validation_sim, _float(last, "bench_output_tok_s_gpu"))
    validation_last_steady = _safe_ratio(validation_sim, _float(last, "steady_output_tok_s_gpu"))
    validation_artifact = _safe_ratio(validation_baseline, validation_last_steady)
    trend.update(
        {
            "baseline_overall_penalty": _fmt(baseline_overall),
            "last_overall_penalty": _fmt(last_overall),
            "last_steady_penalty": _fmt(last_steady),
            "burst_artifact_multiplier": _fmt(artifact_multiplier),
            "validation_real_output_tok_s_gpu": _fmt(validation_real),
            "validation_sim_output_tok_s_gpu": _fmt(validation_sim),
            "validation_baseline_penalty": _fmt(validation_baseline),
            "validation_last_overall_penalty": _fmt(validation_last_overall),
            "validation_last_steady_penalty": _fmt(validation_last_steady),
            "validation_burst_artifact_multiplier": _fmt(validation_artifact),
            "phase412_32k_request_count_ratio": phase412_trend.get("request_count_ratio", ""),
            "phase412_32k_overall_penalty": phase412_trend.get("overall_penalty", ""),
            "phase412_32k_steady_penalty": phase412_trend.get("steady_penalty", ""),
            "phase412_32k_verdict": phase412_trend.get("mechanism_verdict", ""),
            "mechanism_verdict": verdict,
            "phase426_target": target,
        }
    )
    return [{field: row.get(field, "") for field in CSV_FIELDS} for row in rows]


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Phase425 rows must not be empty")
    if sum(1 for row in rows if row.get("row_type") == "trend") != 1:
        raise ValueError("Phase425 expects exactly one trend row")
    for row in rows:
        if row.get("source") != SOURCE:
            raise ValueError("unexpected source")
        if row.get("phase405_penalty_read") != "false":
            raise ValueError("phase405_penalty_read must stay false")
        if row.get("gpu_allowed") != "true" or row.get("ssh_allowed") != "true":
            raise ValueError("Phase425 is GPU measurement and must declare GPU/SSH use")
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


def write_phase425_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_phase425_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    trend = [row for row in rows if row["row_type"] == "trend"][0]
    samples = [row for row in rows if row["row_type"] == "sample"]
    lines = [
        "# Phase425 8k2k Arrival Sweep",
        "",
        "Phase425 镜像 Phase412，只对 DP2 8k2k 做 N=128 突发与 N=512 持续到达对照；不改 runtime、PerfDatabase 或 gate。",
        "",
        "## Verdict",
        "",
        f"- verdict: `{trend['mechanism_verdict']}`",
        f"- Phase426 target: `{trend['phase426_target']}`",
        f"- current validation baseline penalty: `{trend['validation_baseline_penalty']}`",
        f"- current validation steady penalty: `{trend['validation_last_steady_penalty']}`",
        f"- current validation burst artifact multiplier: `{trend['validation_burst_artifact_multiplier']}`",
        f"- trace baseline penalty: `{trend['baseline_overall_penalty']}`",
        "",
        "## Samples",
        "",
        "| N | artifact | request ratio | overall penalty | steady penalty | bench tok/s/gpu | trace fidelity |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in samples:
        lines.append(
            f"| {row['num_prompts']} | {row['artifact_dir']} | {row['request_count_ratio']} | "
            f"{row['overall_penalty']} | {row['steady_penalty']} | {row['bench_output_tok_s_gpu']} | "
            f"{row['trace_fidelity_gate']} |"
        )
    lines.extend(
        [
            "",
            "## 32k3k Reference",
            "",
            "| source | request ratio | overall penalty | steady penalty | verdict |",
            "|---|---:|---:|---:|---|",
            f"| Phase412 32k3k | {trend['phase412_32k_request_count_ratio']} | "
            f"{trend['phase412_32k_overall_penalty']} | {trend['phase412_32k_steady_penalty']} | "
            f"{trend['phase412_32k_verdict']} |",
            "",
            "## Validation Baseline",
            "",
            "| source | real tok/s/gpu | sim tok/s/gpu | baseline penalty | N=512 overall penalty | N=512 steady penalty | artifact multiplier |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| Phase424 current validation | {trend['validation_real_output_tok_s_gpu']} | "
            f"{trend['validation_sim_output_tok_s_gpu']} | {trend['validation_baseline_penalty']} | "
            f"{trend['validation_last_overall_penalty']} | {trend['validation_last_steady_penalty']} | "
            f"{trend['validation_burst_artifact_multiplier']} |",
            f"| Phase403/407 trace baseline |  |  | {trend['baseline_overall_penalty']} | "
            f"{trend['last_overall_penalty']} | {trend['last_steady_penalty']} | "
            f"{trend['burst_artifact_multiplier']} |",
            "",
            "## Interpretation",
            "",
            f"- 当前 validate 的 8k2k `2.553139x` 口径可拆成 artifact multiplier `{trend['validation_burst_artifact_multiplier']}` 和 steady gap `{trend['validation_last_steady_penalty']}`。",
            f"- Phase403/407 trace-baseline 口径是 `{trend['baseline_overall_penalty']}`，用于和 Phase412 方法保持可比；它不是当前 validate 的最大残差口径。",
            "- 若 N=512 把 request ratio 拉平但 steady penalty 仍高，Phase426 需要在验收到达模式和稳态模型残差之间做口径决策。",
            "- 本报告只提供测量证据；不替换验收参考，不打开 Default AIC。",
            "",
            "## Boundary",
            "",
            "- GPU/SSH: used only for Phase425 measurement artifacts.",
            "- Runtime/PerfDatabase/gate: not modified.",
            "- Phase405 penalty: not read.",
            "- Default AIC: No-Go.",
            "",
        ]
    )
    return "\n".join(lines)


def write_phase425_md(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase425_md(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--sweep-root", type=Path, default=SWEEP_ROOT)
    parser.add_argument("--phase407-csv", type=Path, default=PHASE407_CSV)
    parser.add_argument("--phase412-csv", type=Path, default=PHASE412_CSV)
    parser.add_argument("--validation-csv", type=Path, default=VALIDATION_CSV)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_phase425_rows(
        baseline_root=args.baseline_root,
        sweep_root=args.sweep_root,
        phase407_csv=args.phase407_csv,
        phase412_csv=args.phase412_csv,
        validation_csv=args.validation_csv,
        scenario=args.scenario,
    )
    write_phase425_csv(args.csv, rows)
    write_phase425_md(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
