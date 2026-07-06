#!/usr/bin/env python3
"""Phase422: account for the DP2 bt65536 KV-capacity truth."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE421_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase421_wire_bt65536.csv"
DEFAULT_AFTER_CSV = Path("/private/tmp/phase422_validation_after.csv")
DEFAULT_TRACE_CSV = Path("/private/tmp/phase422_bt65536_trace_after.csv")
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase422_dp2_capacity_fix.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase422_dp2_capacity_fix.md"

SOURCE = "phase422_dp2_capacity_fix"
DEFAULT_READINESS = "No-Go"

TP8_BT_SCENARIO = "K2.5-tp8ep8-8k2k-bt65536"
DP2_BT_SCENARIO = "K2.5-tp4ep8dp2-8k2k-bt65536"

OUTPUT_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "tp",
    "dp",
    "ep",
    "max_bt",
    "real_output_tok_s_gpu",
    "phase421_sim_tok_s_gpu",
    "phase422_sim_tok_s_gpu",
    "phase421_error_ratio",
    "phase422_error_ratio",
    "error_ratio_delta",
    "classification",
    "kv_cache_tokens",
    "block_size",
    "num_gpu_blocks",
    "max_num_seqs",
    "full_sequence_capacity_per_engine",
    "serve_log",
    "serve_log_lines",
    "override_num_gpu_blocks",
    "override_lines",
    "configured_max_num_batched_tokens",
    "budget_full_prefill_reqs",
    "initial_prefill_reqs",
    "initial_prefill_tokens",
    "representative_step_ms",
    "moe_query_path",
    "ep_query_path",
    "capacity_classification",
    "trace_status",
    "capacity_verdict",
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
    if isinstance(value, (tuple, list)):
        return ";".join(str(v) for v in value)
    return str(value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_validate_module(repo_root: Path) -> ModuleType:
    validate_path = repo_root / "scripts/validate_cb_simulator.py"
    if not validate_path.exists():
        raise FileNotFoundError(validate_path)
    for key in list(sys.modules):
        if key == "aiconfigurator" or key.startswith("aiconfigurator."):
            del sys.modules[key]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))
    spec = importlib.util.spec_from_file_location("phase422_validate_cb_simulator", validate_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {validate_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _guard_row(row: dict[str, str]) -> dict[str, str]:
    out = {field: _fmt(row.get(field, "")) for field in OUTPUT_FIELDS}
    out.update(
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
    return out


def _phase421_config_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["scenario"]: row for row in rows if row.get("row_type") == "config"}


def _validation_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["scenario"]: row for row in rows}


def _classify(before_error: float, after_error: float) -> str:
    if after_error < before_error:
        return "improved"
    if after_error > before_error:
        return "regressed"
    return "unchanged"


def collect_capacity_rows(repo_root: Path = REPO_ROOT) -> list[dict[str, str]]:
    validate = _load_validate_module(repo_root)
    rows: list[dict[str, str]] = []
    for point in validate.MULTI_CONFIG_DATA:
        if point.name not in {TP8_BT_SCENARIO, DP2_BT_SCENARIO}:
            continue
        capacity = validate._multi_config_kv_capacity(point)
        full_sequence_capacity = capacity.kv_cache_tokens / float(point.isl + point.osl)
        verdict = (
            "retry5e_dirty_capacity_truth"
            if point.name == DP2_BT_SCENARIO
            else "bt65536_reference_capacity_truth"
        )
        rows.append(
            _guard_row(
                {
                    "source": SOURCE,
                    "row_type": "capacity",
                    "scenario": point.name,
                    "tp": point.tp,
                    "dp": point.dp,
                    "ep": point.moe_ep,
                    "max_bt": point.max_num_batched_tokens,
                    "kv_cache_tokens": capacity.kv_cache_tokens,
                    "block_size": capacity.block_size,
                    "num_gpu_blocks": capacity.num_gpu_blocks,
                    "max_num_seqs": capacity.max_num_seqs,
                    "full_sequence_capacity_per_engine": full_sequence_capacity,
                    "serve_log": capacity.serve_log,
                    "serve_log_lines": capacity.serve_log_line_numbers,
                    "override_num_gpu_blocks": capacity.override_num_gpu_blocks,
                    "override_lines": capacity.override_line_numbers,
                    "capacity_verdict": verdict,
                }
            )
        )
    return rows


def _trace_status(row: dict[str, str]) -> str:
    if row["scenario"] == TP8_BT_SCENARIO and int(row["initial_prefill_tokens"]) >= int(row["max_bt"]):
        return "tp8_reaches_64k_step"
    if row["scenario"] == DP2_BT_SCENARIO and int(row["initial_prefill_tokens"]) < int(row["max_bt"]):
        return "real_capacity_blocks_64k_step"
    return "unexpected_trace_shape"


def build_phase422_rows(
    *,
    phase421_csv: Path = DEFAULT_PHASE421_CSV,
    after_csv: Path = DEFAULT_AFTER_CSV,
    trace_csv: Path = DEFAULT_TRACE_CSV,
    repo_root: Path = REPO_ROOT,
) -> list[dict[str, str]]:
    phase421_by_name = _phase421_config_rows(_read_csv(phase421_csv))
    after_by_name = _validation_rows(_read_csv(after_csv))
    if set(phase421_by_name) != set(after_by_name):
        raise ValueError("Phase421 and Phase422 validation scenarios do not match")

    rows: list[dict[str, str]] = []
    for scenario in sorted(after_by_name):
        before = phase421_by_name[scenario]
        after = after_by_name[scenario]
        before_error = float(before["error_ratio_after"])
        after_error = float(after["error_ratio"])
        rows.append(
            _guard_row(
                {
                    "source": SOURCE,
                    "row_type": "config",
                    "scenario": scenario,
                    "tp": after["tp"],
                    "dp": after["dp"],
                    "ep": after["ep"],
                    "max_bt": after["max_bt"],
                    "real_output_tok_s_gpu": after["real_output_tok_s_gpu"],
                    "phase421_sim_tok_s_gpu": before["sim_after_tok_s_gpu"],
                    "phase422_sim_tok_s_gpu": after["sim_output_tok_s_gpu"],
                    "phase421_error_ratio": before["error_ratio_after"],
                    "phase422_error_ratio": after["error_ratio"],
                    "error_ratio_delta": after_error - before_error,
                    "classification": _classify(before_error, after_error),
                }
            )
        )

    rows.extend(collect_capacity_rows(repo_root))

    for trace in _read_csv(trace_csv):
        if trace.get("row_type") != "trace" or trace.get("label") != "after":
            continue
        if trace["scenario"] not in {TP8_BT_SCENARIO, DP2_BT_SCENARIO}:
            continue
        rows.append(
            _guard_row(
                {
                    "source": SOURCE,
                    "row_type": "trace",
                    "scenario": trace["scenario"],
                    "tp": trace["tp"],
                    "dp": trace["dp"],
                    "ep": trace["ep"],
                    "max_bt": trace["max_bt"],
                    "real_output_tok_s_gpu": trace["real_output_tok_s_gpu"],
                    "phase422_sim_tok_s_gpu": trace["sim_output_tok_s_gpu"],
                    "phase422_error_ratio": trace["error_ratio"],
                    "num_gpu_blocks": trace["num_gpu_blocks"],
                    "full_sequence_capacity_per_engine": trace["full_sequence_capacity_per_engine"],
                    "configured_max_num_batched_tokens": trace["configured_max_num_batched_tokens"],
                    "budget_full_prefill_reqs": trace["budget_full_prefill_reqs"],
                    "initial_prefill_reqs": trace["initial_prefill_reqs"],
                    "initial_prefill_tokens": trace["initial_prefill_tokens"],
                    "representative_step_ms": trace["representative_step_ms"],
                    "moe_query_path": trace["moe_query_path"],
                    "ep_query_path": trace["ep_query_path"],
                    "capacity_classification": trace["capacity_classification"],
                    "trace_status": _trace_status(trace),
                }
            )
        )

    summary = {
        "source": SOURCE,
        "row_type": "summary",
        "scenario": "all_multi_config",
        "phase421_error_ratio": max(float(row["phase421_error_ratio"]) for row in rows if row["row_type"] == "config"),
        "phase422_error_ratio": max(float(row["phase422_error_ratio"]) for row in rows if row["row_type"] == "config"),
        "mechanism_verdict": "dp2_bt65536_capacity_truth_is_dirty_reference_not_fixable_by_wiring",
        "next_phase_target": "phase423_recollect_or_remove_dp2_bt65536_dirty_reference_before_large_step_cost",
    }
    rows.append(_guard_row(summary))
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Phase422 rows must not be empty")
    if sum(1 for row in rows if row.get("row_type") == "summary") != 1:
        raise ValueError("Phase422 expects exactly one summary row")
    dp2_capacity = [
        row for row in rows
        if row.get("row_type") == "capacity" and row.get("scenario") == DP2_BT_SCENARIO
    ]
    if len(dp2_capacity) != 1:
        raise ValueError("Phase422 expects one DP2 bt65536 capacity row")
    dp2 = dp2_capacity[0]
    if dp2.get("kv_cache_tokens") != "25744" or dp2.get("num_gpu_blocks") != "1609":
        raise ValueError("DP2 bt65536 capacity must stay tied to retry5e log truth")
    if dp2.get("max_num_seqs") != "128":
        raise ValueError("DP2 bt65536 max_num_seqs must stay tied to retry5e log truth")
    for row in rows:
        if row.get("source") != SOURCE:
            raise ValueError("unexpected source")
        if row.get("phase405_penalty_read") != "false":
            raise ValueError("phase405_penalty_read must stay false")
        if row.get("gpu_allowed") != "false" or row.get("ssh_allowed") != "false":
            raise ValueError("Phase422 must not use GPU/SSH")
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


def write_phase422_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    _write_csv(path, rows)


def render_phase422_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    summary = [row for row in rows if row["row_type"] == "summary"][0]
    configs = [row for row in rows if row["row_type"] == "config"]
    capacities = [row for row in rows if row["row_type"] == "capacity"]
    traces = [row for row in rows if row["row_type"] == "trace"]
    lines = [
        "# Phase422 DP2 Capacity Fix",
        "",
        f"Verdict: `{summary['mechanism_verdict']}`.",
        "",
        f"Default AIC: `{DEFAULT_READINESS}`.",
        "",
        "Phase422 fixes the validation harness wiring for `max_num_seqs`, then accounts for the DP2 bt65536 capacity truth. The logged retry5e capacity is only `25,744` KV tokens (`1,609` blocks) with `max_num_seqs=128`; that truth blocks an independent 64k prefill step. There is no log-backed larger capacity to wire in.",
        "",
        "## Validation A/B",
        "",
        "| Scenario | Phase421 error | Phase422 error | Class | Phase422 sim |",
        "|---|---:|---:|---|---:|",
    ]
    for row in configs:
        lines.append(
            f"| {row['scenario']} | {row['phase421_error_ratio']} | {row['phase422_error_ratio']} | "
            f"{row['classification']} | {row['phase422_sim_tok_s_gpu']} |"
        )
    lines.extend(
        [
            "",
            "## Capacity Truth",
            "",
            "| Scenario | KV tokens | Blocks | max_num_seqs | Full sequence capacity | Source lines |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in capacities:
        lines.append(
            f"| {row['scenario']} | {row['kv_cache_tokens']} | {row['num_gpu_blocks']} | "
            f"{row['max_num_seqs']} | {row['full_sequence_capacity_per_engine']} | "
            f"{row['serve_log_lines']} |"
        )
    lines.extend(
        [
            "",
            "## Trace",
            "",
            "| Scenario | Configured bt | Initial tokens | Initial reqs | Trace status | MoE path | EP path |",
            "|---|---:|---:|---:|---|---|---|",
        ]
    )
    for row in traces:
        lines.append(
            f"| {row['scenario']} | {row['configured_max_num_batched_tokens']} | "
            f"{row['initial_prefill_tokens']} | {row['initial_prefill_reqs']}/{row['budget_full_prefill_reqs']} | "
            f"{row['trace_status']} | {row['moe_query_path']} | {row['ep_query_path']} |"
        )
    lines.extend(
        [
            "",
            "## Phase423 Target",
            "",
            f"`{summary['next_phase_target']}`.",
            "",
            "The next step is not to invent capacity. Either recollect a clean DP2 bt65536 reference with consistent KV capacity, or remove this dirty cached reference from the acceptance surface before judging large-step cost.",
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
    parser.add_argument("--phase421-csv", type=Path, default=DEFAULT_PHASE421_CSV)
    parser.add_argument("--after-csv", type=Path, default=DEFAULT_AFTER_CSV)
    parser.add_argument("--trace-csv", type=Path, default=DEFAULT_TRACE_CSV)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_phase422_rows(
        phase421_csv=args.phase421_csv,
        after_csv=args.after_csv,
        trace_csv=args.trace_csv,
    )
    write_phase422_csv(args.csv_out, rows)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_phase422_md(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
