#!/usr/bin/env python3
"""Phase421: summarize bt65536 validation wiring and post-wiring trace."""

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
DEFAULT_BEFORE_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase419_validation_triage_after.csv"
DEFAULT_AFTER_CSV = Path("/private/tmp/phase421_validation_after.csv")
DEFAULT_TRACE_CSV = Path("/private/tmp/phase421_bt65536_trace_after.csv")
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase421_wire_bt65536.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase421_wire_bt65536.md"
SOURCE = "phase421_wire_bt65536"
DEFAULT_READINESS = "No-Go"

TP8_BT_SCENARIO = "K2.5-tp8ep8-8k2k-bt65536"
DP2_BT_SCENARIO = "K2.5-tp4ep8dp2-8k2k-bt65536"

VALIDATION_FIELDS = [
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

TRACE_FIELDS = [
    "source",
    "row_type",
    "label",
    "scenario",
    "tp",
    "dp",
    "ep",
    "max_bt",
    "real_output_tok_s_gpu",
    "sim_output_tok_s_gpu",
    "error_ratio",
    "num_gpu_blocks",
    "full_sequence_capacity_per_engine",
    "per_replica_concurrency",
    "configured_max_num_batched_tokens",
    "budget_full_prefill_reqs",
    "initial_prefill_reqs",
    "initial_prefill_tokens",
    "representative_step_ms",
    "context_non_attention_ms",
    "context_attention_ms",
    "moe_query_path",
    "ep_query_path",
    "capacity_classification",
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
    "num_gpu_blocks",
    "full_sequence_capacity_per_engine",
    "configured_max_num_batched_tokens",
    "budget_full_prefill_reqs",
    "initial_prefill_reqs",
    "initial_prefill_tokens",
    "representative_step_ms",
    "moe_query_path",
    "ep_query_path",
    "capacity_classification",
    "trace_status",
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
        return math.inf
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
    spec = importlib.util.spec_from_file_location("phase421_validate_cb_simulator", validate_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {validate_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def collect_validation_rows(repo_root: Path = REPO_ROOT) -> list[dict[str, str]]:
    validate = _load_validate_module(repo_root)
    backend = validate.VLLMBackend()
    loaded: dict[tuple[int, int, int, int], tuple[Any, Any]] = {}
    rows: list[dict[str, str]] = []
    for pt in validate.MULTI_CONFIG_DATA:
        key = (pt.tp, pt.dp, pt.moe_tp, pt.moe_ep)
        if key not in loaded:
            model, db, _ = validate._load_model_and_db(tp=pt.tp, dp=pt.dp, moe_tp=pt.moe_tp, moe_ep=pt.moe_ep)
            loaded[key] = (model, db)
        model, db = loaded[key]
        cb_config = validate._make_multi_config_cb_config(
            pt,
            overlap_factor=0.0,
            ep8_per_iteration_overhead_ms=validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
        )
        cb_summary = backend.run_agg(
            model,
            db,
            validate.RuntimeConfig(batch_size=pt.batch_size, isl=pt.isl, osl=pt.osl),
            ctx_tokens=pt.max_num_batched_tokens,
            database_mode=validate.common.DatabaseMode.HYBRID,
            method="cb_sim",
            cb_config=cb_config,
        )
        cb_dict = cb_summary.get_result_dict()
        per_ops = cb_summary.get_per_ops_data()
        sim = float(cb_dict["tokens/s/gpu"]) if cb_dict else 0.0
        real = float(pt.real_output_tok_s_gpu)
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
                "throughput_source": str(per_ops.get("cb_sim_boundary", {}).get("throughput_source", "unknown")),
            }
        )
    return rows


def _classify(before: float, after: float) -> str:
    if after < before:
        return "improved"
    if after > before:
        return "regressed"
    return "unchanged"


def _trace_status(row: dict[str, str]) -> str:
    configured = int(row["configured_max_num_batched_tokens"])
    max_bt = int(row["max_bt"])
    tokens = int(row["initial_prefill_tokens"])
    if configured != max_bt:
        return "budget_not_wired"
    if row["scenario"] == TP8_BT_SCENARIO and tokens >= max_bt:
        return "wired_full_budget"
    if row["scenario"] == DP2_BT_SCENARIO and row["capacity_classification"] == "dirty_capacity_conflict" and tokens < max_bt:
        return "wired_but_capacity_limited"
    return "wired_partial_budget"


def _summary_verdict(trace_rows: list[dict[str, str]]) -> tuple[str, str]:
    status_by_scenario = {row["scenario"]: row["trace_status"] for row in trace_rows}
    if (
        status_by_scenario.get(TP8_BT_SCENARIO) == "wired_full_budget"
        and status_by_scenario.get(DP2_BT_SCENARIO) == "wired_but_capacity_limited"
    ):
        return (
            "bt65536_wired_tp8_recovers_dp2_dirty_capacity_remains",
            "phase422_resolve_dirty_dp2_bt65536_capacity_before_large_step_cost",
        )
    if set(status_by_scenario.values()) == {"wired_full_budget"}:
        return "bt65536_wired_charges_sane", "phase422_return_to_main_residual_line"
    return "bt65536_wiring_incomplete", "phase422_recheck_bt65536_wiring"


def build_phase421_rows(
    *,
    before_csv: Path = DEFAULT_BEFORE_CSV,
    after_csv: Path = DEFAULT_AFTER_CSV,
    trace_csv: Path = DEFAULT_TRACE_CSV,
) -> list[dict[str, str]]:
    before_by_name = {row["scenario"]: row for row in _read_csv(before_csv)}
    after_by_name = {row["scenario"]: row for row in _read_csv(after_csv)}
    if set(before_by_name) != set(after_by_name):
        raise ValueError("before/after validation scenarios do not match")

    rows: list[dict[str, str]] = []
    for scenario in sorted(after_by_name):
        before = before_by_name[scenario]
        after = after_by_name[scenario]
        before_error = float(before["error_ratio"])
        after_error = float(after["error_ratio"])
        data = {
            "source": SOURCE,
            "row_type": "config",
            "scenario": scenario,
            "tp": after["tp"],
            "dp": after["dp"],
            "ep": after["ep"],
            "max_bt": after["max_bt"],
            "real_output_tok_s_gpu": after["real_output_tok_s_gpu"],
            "sim_before_tok_s_gpu": before["sim_output_tok_s_gpu"],
            "sim_after_tok_s_gpu": after["sim_output_tok_s_gpu"],
            "sim_real_ratio_before": before["sim_real_ratio"],
            "sim_real_ratio_after": after["sim_real_ratio"],
            "error_ratio_before": before["error_ratio"],
            "error_ratio_after": after["error_ratio"],
            "error_ratio_delta": after_error - before_error,
            "classification": _classify(before_error, after_error),
        }
        rows.append({field: _fmt(data.get(field, "")) for field in OUTPUT_FIELDS})

    trace_rows: list[dict[str, str]] = []
    for row in _read_csv(trace_csv):
        if row.get("row_type") != "trace" or row.get("label") != "after":
            continue
        if row["scenario"] not in {TP8_BT_SCENARIO, DP2_BT_SCENARIO}:
            continue
        data = {
            "source": SOURCE,
            "row_type": "trace",
            "scenario": row["scenario"],
            "tp": row["tp"],
            "dp": row["dp"],
            "ep": row["ep"],
            "max_bt": row["max_bt"],
            "real_output_tok_s_gpu": row["real_output_tok_s_gpu"],
            "sim_after_tok_s_gpu": row["sim_output_tok_s_gpu"],
            "error_ratio_after": row["error_ratio"],
            "num_gpu_blocks": row["num_gpu_blocks"],
            "full_sequence_capacity_per_engine": row["full_sequence_capacity_per_engine"],
            "configured_max_num_batched_tokens": row["configured_max_num_batched_tokens"],
            "budget_full_prefill_reqs": row["budget_full_prefill_reqs"],
            "initial_prefill_reqs": row["initial_prefill_reqs"],
            "initial_prefill_tokens": row["initial_prefill_tokens"],
            "representative_step_ms": row["representative_step_ms"],
            "moe_query_path": row["moe_query_path"],
            "ep_query_path": row["ep_query_path"],
            "capacity_classification": row["capacity_classification"],
            "trace_status": _trace_status(row),
        }
        formatted = {field: _fmt(data.get(field, "")) for field in OUTPUT_FIELDS}
        trace_rows.append(formatted)
        rows.append(formatted)

    verdict, next_phase = _summary_verdict(trace_rows)
    summary = {
        "source": SOURCE,
        "row_type": "summary",
        "scenario": "all_multi_config",
        "error_ratio_before": max(float(row["error_ratio_before"]) for row in rows if row["row_type"] == "config"),
        "error_ratio_after": max(float(row["error_ratio_after"]) for row in rows if row["row_type"] == "config"),
        "mechanism_verdict": verdict,
        "next_phase_target": next_phase,
    }
    rows.append({field: _fmt(summary.get(field, "")) for field in OUTPUT_FIELDS})
    for row in rows:
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
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Phase421 rows must not be empty")
    if sum(1 for row in rows if row.get("row_type") == "summary") != 1:
        raise ValueError("Phase421 expects exactly one summary row")
    for row in rows:
        if row.get("source") != SOURCE:
            raise ValueError("unexpected source")
        if row.get("phase405_penalty_read") != "false":
            raise ValueError("phase405_penalty_read must stay false")
        if row.get("gpu_allowed") != "false" or row.get("ssh_allowed") != "false":
            raise ValueError("Phase421 must not use GPU/SSH")
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


def write_phase421_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    _write_csv(path, OUTPUT_FIELDS, rows)


def render_phase421_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    summary = [row for row in rows if row["row_type"] == "summary"][0]
    configs = [row for row in rows if row["row_type"] == "config"]
    traces = [row for row in rows if row["row_type"] == "trace"]
    lines = [
        "# Phase421 Wire BT65536",
        "",
        f"Verdict: `{summary['mechanism_verdict']}`.",
        "",
        f"Default AIC: `{DEFAULT_READINESS}`.",
        "",
        "## A/B Validation",
        "",
        "| Scenario | Before error | After error | Class | Before sim | After sim |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for row in configs:
        lines.append(
            f"| {row['scenario']} | {row['error_ratio_before']} | {row['error_ratio_after']} | "
            f"{row['classification']} | {row['sim_before_tok_s_gpu']} | {row['sim_after_tok_s_gpu']} |"
        )
    lines.extend(
        [
            "",
            "## Post-Wiring Trace",
            "",
            "`Initial reqs` can exceed `full-fit` by one when the final request is a partial chunk.",
            "",
            "| Scenario | Configured bt | Initial tokens | Initial reqs | Capacity | MoE path | EP path | Status |",
            "|---|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in traces:
        lines.append(
            f"| {row['scenario']} | {row['configured_max_num_batched_tokens']} | {row['initial_prefill_tokens']} | "
            f"{row['initial_prefill_reqs']}/{row['budget_full_prefill_reqs']} | {row['full_sequence_capacity_per_engine']} | "
            f"{row['moe_query_path']} | {row['ep_query_path']} | {row['trace_status']} |"
        )
    lines.extend(
        [
            "",
            "## Phase422 Target",
            "",
            f"`{summary['next_phase_target']}`.",
            "",
            "The TP8 bt65536 path now reaches the 64k step. The DP2 bt65536 path is still limited by the dirty capacity row before it can exercise the 64k step.",
            "",
            "## Guardrails",
            "",
            "- `diagnostic_only=true`.",
            "- `valid_for_default=false`.",
            "- `runtime_modified=false`.",
            "- `perf_database=false`.",
            "- `gpu_allowed=false`, `ssh_allowed=false`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")
    collect = sub.add_parser("collect-validation")
    collect.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    collect.add_argument("--out", type=Path, required=True)
    build = sub.add_parser("build")
    build.add_argument("--before-csv", type=Path, default=DEFAULT_BEFORE_CSV)
    build.add_argument("--after-csv", type=Path, default=DEFAULT_AFTER_CSV)
    build.add_argument("--trace-csv", type=Path, default=DEFAULT_TRACE_CSV)
    build.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    build.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    if args.cmd == "collect-validation":
        _write_csv(args.out, VALIDATION_FIELDS, collect_validation_rows(args.repo_root))
        return 0
    if args.cmd in (None, "build"):
        rows = build_phase421_rows(before_csv=args.before_csv, after_csv=args.after_csv, trace_csv=args.trace_csv)
        write_phase421_csv(args.csv_out, rows)
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_phase421_md(rows))
        return 0
    raise ValueError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
