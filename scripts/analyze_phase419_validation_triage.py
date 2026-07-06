#!/usr/bin/env python3
"""Phase419: A/B triage for Phase417 validation behavior."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BEFORE_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase419_validation_triage_before.csv"
DEFAULT_AFTER_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase419_validation_triage_after.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase419_validation_triage.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase419_validation_triage.md"
SOURCE = "phase419_validation_triage"
DEFAULT_READINESS = "No-Go"

INPUT_FIELDS = [
    "scenario",
    "tp",
    "dp",
    "ep",
    "max_bt",
    "real_output_tok_s_gpu",
    "sim_output_tok_s_gpu",
    "sim_real_ratio",
    "error_ratio",
    "throughput_source",
]

OUTPUT_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "tp",
    "dp",
    "ep",
    "max_bt",
    "real_output_tok_s_gpu",
    "sim_before_tok_s_gpu",
    "sim_after_tok_s_gpu",
    "sim_real_ratio_before",
    "sim_real_ratio_after",
    "error_ratio_before",
    "error_ratio_after",
    "error_ratio_delta",
    "classification",
    "is_max_after_error",
    "throughput_source_before",
    "throughput_source_after",
    "mechanism_hint",
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


def _abs_error(sim: float, real: float) -> float:
    if sim <= 0.0 or real <= 0.0:
        return float("inf")
    ratio = sim / real
    return max(ratio, 1.0 / ratio)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_validate_module(repo_root: Path) -> ModuleType:
    validate_path = repo_root / "scripts/validate_cb_simulator.py"
    if not validate_path.exists():
        raise FileNotFoundError(validate_path)
    src_root = repo_root / "src"
    for key in list(sys.modules):
        if key == "aiconfigurator" or key.startswith("aiconfigurator."):
            del sys.modules[key]
    sys.path.insert(0, str(src_root))
    sys.path.insert(0, str(repo_root))
    spec = importlib.util.spec_from_file_location("phase419_validate_cb_simulator", validate_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {validate_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def collect_validation_rows(repo_root: Path = REPO_ROOT) -> list[dict[str, str]]:
    """Run validate_cb_simulator's official MULTI_CONFIG path and return per-config rows."""
    validate = _load_validate_module(repo_root)
    backend = validate.VLLMBackend()
    loaded: dict[tuple[int, int, int, int], tuple[Any, Any]] = {}
    rows: list[dict[str, str]] = []
    for pt in validate.MULTI_CONFIG_DATA:
        key = (pt.tp, pt.dp, pt.moe_tp, pt.moe_ep)
        if key not in loaded:
            model, db, _ = validate._load_model_and_db(
                tp=pt.tp,
                dp=pt.dp,
                moe_tp=pt.moe_tp,
                moe_ep=pt.moe_ep,
            )
            loaded[key] = (model, db)
        model, db = loaded[key]
        cb_config = validate._make_cb_config(
            pt.isl,
            pt.batch_size,
            overlap_factor=0.0,
            per_iteration_overhead_ms=validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
            num_gpu_blocks=validate._multi_config_num_gpu_blocks(pt),
        )
        cb_summary = backend.run_agg(
            model,
            db,
            validate.RuntimeConfig(batch_size=pt.batch_size, isl=pt.isl, osl=pt.osl),
            ctx_tokens=pt.isl,
            database_mode=validate.common.DatabaseMode.HYBRID,
            method="cb_sim",
            cb_config=cb_config,
        )
        cb_dict = cb_summary.get_result_dict()
        per_ops = cb_summary.get_per_ops_data()
        sim = float(cb_dict["tokens/s/gpu"]) if cb_dict else 0.0
        real = float(pt.real_output_tok_s_gpu)
        source = per_ops.get("cb_sim_boundary", {}).get("throughput_source", "unknown")
        rows.append(
            {
                "scenario": pt.name,
                "tp": str(pt.tp),
                "dp": str(pt.dp),
                "ep": str(pt.moe_ep),
                "max_bt": str(pt.max_num_batched_tokens),
                "real_output_tok_s_gpu": _fmt(real),
                "sim_output_tok_s_gpu": _fmt(sim),
                "sim_real_ratio": _fmt(sim / real if real > 0.0 else 0.0),
                "error_ratio": _fmt(_abs_error(sim, real)),
                "throughput_source": str(source),
            }
        )
    return rows


def _classify(before_error: float, after_error: float) -> str:
    if after_error < before_error:
        return "improved"
    if after_error > before_error:
        return "regressed"
    return "unchanged"


def _direction(sim_real_ratio: float) -> str:
    if sim_real_ratio > 1.0:
        return "overprediction"
    if sim_real_ratio < 1.0:
        return "underprediction"
    return "matched"


def _mechanism_hint(row: dict[str, object]) -> str:
    classification = str(row["classification"])
    if classification != "regressed":
        return "not_regressed"
    scenario = str(row["scenario"])
    max_bt = int(row["max_bt"])
    direction = _direction(float(row["sim_real_ratio_after"]))
    if "bt65536" in scenario and direction == "underprediction":
        return "bt65536_underprediction_regression"
    if int(row["dp"]) > 1 and direction == "overprediction":
        return "dp_ep_overprediction_regression"
    if max_bt > 32768:
        return "large_budget_regression"
    return f"{direction}_regression"


def _verdict(config_rows: list[dict[str, str]]) -> tuple[str, str]:
    regressed = [row for row in config_rows if row["classification"] == "regressed"]
    if not regressed:
        return "no_regression", "phase420_gpu_prefill_profiler_or_existing_fail_line"
    hints = {row["mechanism_hint"] for row in regressed}
    if "bt65536_underprediction_regression" in hints:
        return "fix_regression_bt65536_underprediction", "phase420_trace_bt65536_capacity_or_ep_fallback_branch"
    return "fix_regression_phase417_branch", "phase420_replay_regressed_config_query_chain"


def build_phase419_rows(
    *,
    before_csv: Path = DEFAULT_BEFORE_CSV,
    after_csv: Path = DEFAULT_AFTER_CSV,
) -> list[dict[str, str]]:
    before_by_name = {row["scenario"]: row for row in _read_csv(before_csv)}
    after_by_name = {row["scenario"]: row for row in _read_csv(after_csv)}
    if set(before_by_name) != set(after_by_name):
        missing_before = sorted(set(after_by_name) - set(before_by_name))
        missing_after = sorted(set(before_by_name) - set(after_by_name))
        raise ValueError(f"mismatched scenarios before_missing={missing_before} after_missing={missing_after}")

    rows: list[dict[str, object]] = []
    max_after_error = -1.0
    for scenario in sorted(after_by_name):
        before = before_by_name[scenario]
        after = after_by_name[scenario]
        before_error = float(before["error_ratio"])
        after_error = float(after["error_ratio"])
        max_after_error = max(max_after_error, after_error)
        row: dict[str, object] = {
            "source": SOURCE,
            "row_type": "config",
            "scenario": scenario,
            "tp": int(after["tp"]),
            "dp": int(after["dp"]),
            "ep": int(after["ep"]),
            "max_bt": int(after["max_bt"]),
            "real_output_tok_s_gpu": float(after["real_output_tok_s_gpu"]),
            "sim_before_tok_s_gpu": float(before["sim_output_tok_s_gpu"]),
            "sim_after_tok_s_gpu": float(after["sim_output_tok_s_gpu"]),
            "sim_real_ratio_before": float(before["sim_real_ratio"]),
            "sim_real_ratio_after": float(after["sim_real_ratio"]),
            "error_ratio_before": before_error,
            "error_ratio_after": after_error,
            "error_ratio_delta": after_error - before_error,
            "classification": _classify(before_error, after_error),
            "throughput_source_before": before.get("throughput_source", "unknown"),
            "throughput_source_after": after.get("throughput_source", "unknown"),
        }
        row["mechanism_hint"] = _mechanism_hint(row)
        rows.append(row)

    for row in rows:
        row["is_max_after_error"] = float(row["error_ratio_after"]) == max_after_error

    config_rows = [{field: _fmt(row.get(field, "")) for field in OUTPUT_FIELDS} for row in rows]
    verdict, next_phase = _verdict(config_rows)
    summary = {
        "source": SOURCE,
        "row_type": "summary",
        "scenario": "all_multi_config",
        "error_ratio_before": max(float(row["error_ratio_before"]) for row in config_rows),
        "error_ratio_after": max(float(row["error_ratio_after"]) for row in config_rows),
        "error_ratio_delta": max(float(row["error_ratio_after"]) for row in config_rows)
        - max(float(row["error_ratio_before"]) for row in config_rows),
        "classification": "has_regression"
        if any(row["classification"] == "regressed" for row in config_rows)
        else "no_regression",
        "mechanism_hint": "max_after_error="
        + [row["scenario"] for row in config_rows if row["is_max_after_error"] == "true"][0],
        "mechanism_verdict": verdict,
        "next_phase_target": next_phase,
    }
    all_rows = config_rows + [{field: _fmt(summary.get(field, "")) for field in OUTPUT_FIELDS}]
    for row in all_rows:
        row.update(
            {
                "phase405_penalty_read": "false",
                "gpu_allowed": "false",
                "ssh_allowed": "false",
                "runtime_modified": "false",
                "perf_database": "false",
                "valid_for_default": "false",
                "diagnostic_only": "true",
                "default_readiness": DEFAULT_READINESS,
            }
        )
    return all_rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Phase419 rows must not be empty")
    if sum(1 for row in rows if row.get("row_type") == "summary") != 1:
        raise ValueError("Phase419 expects exactly one summary row")
    for row in rows:
        if row.get("source") != SOURCE:
            raise ValueError("unexpected source")
        if row.get("phase405_penalty_read") != "false":
            raise ValueError("phase405_penalty_read must stay false")
        if row.get("gpu_allowed") != "false" or row.get("ssh_allowed") != "false":
            raise ValueError("Phase419 must not use GPU/SSH")
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


def write_phase419_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    _write_csv(path, OUTPUT_FIELDS, rows)


def render_phase419_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    summary = [row for row in rows if row["row_type"] == "summary"][0]
    configs = [row for row in rows if row["row_type"] == "config"]
    regressed = [row for row in configs if row["classification"] == "regressed"]
    improved = [row for row in configs if row["classification"] == "improved"]
    max_row = [row for row in configs if row["is_max_after_error"] == "true"][0]
    lines = [
        "# Phase419 Validation Triage",
        "",
        f"Verdict: `{summary['mechanism_verdict']}`.",
        "",
        f"Default AIC: `{DEFAULT_READINESS}`.",
        "",
        "## Summary",
        "",
        f"- Max before error: `{summary['error_ratio_before']}x`.",
        f"- Max after error: `{summary['error_ratio_after']}x`.",
        f"- Improved configs: `{len(improved)}`.",
        f"- Regressed configs: `{len(regressed)}`.",
        f"- Max after-error config: `{max_row['scenario']}` (`{max_row['error_ratio_after']}x`, `{max_row['mechanism_hint']}`).",
        f"- Max config before/after: `{max_row['error_ratio_before']}x` -> `{max_row['error_ratio_after']}x`.",
        f"- Next phase target: `{summary['next_phase_target']}`.",
        "",
        "## Config Table",
        "",
        "| Scenario | Before sim/real | After sim/real | Before error | After error | Class | Mechanism |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in configs:
        lines.append(
            f"| {row['scenario']} | {row['sim_real_ratio_before']} | {row['sim_real_ratio_after']} | "
            f"{row['error_ratio_before']} | {row['error_ratio_after']} | {row['classification']} | {row['mechanism_hint']} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- `diagnostic_only=true`.",
            "- `valid_for_default=false`.",
            "- `perf_database=false`.",
            "- `runtime_modified=false`.",
            "- `gpu_allowed=false`, `ssh_allowed=false`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")
    collect = sub.add_parser("collect")
    collect.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    collect.add_argument("--out", type=Path, required=True)
    build = sub.add_parser("build")
    build.add_argument("--before-csv", type=Path, default=DEFAULT_BEFORE_CSV)
    build.add_argument("--after-csv", type=Path, default=DEFAULT_AFTER_CSV)
    build.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    build.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    if args.cmd == "collect":
        _write_csv(args.out, INPUT_FIELDS, collect_validation_rows(args.repo_root))
        return 0
    if args.cmd in (None, "build"):
        rows = build_phase419_rows(before_csv=args.before_csv, after_csv=args.after_csv)
        write_phase419_csv(args.csv_out, rows)
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_phase419_md(rows))
        return 0
    raise ValueError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
