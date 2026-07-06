#!/usr/bin/env python3
"""Phase420: trace bt65536 validation behavior without changing runtime code."""

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
DEFAULT_BEFORE_CSV = Path("/private/tmp/phase420_bt65536_trace_before.csv")
DEFAULT_AFTER_CSV = Path("/private/tmp/phase420_bt65536_trace_after.csv")
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase420_bt65536_trace.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase420_bt65536_trace.md"
SOURCE = "phase420_bt65536_trace"
DEFAULT_READINESS = "No-Go"

DP2_BT_SCENARIO = "K2.5-tp4ep8dp2-8k2k-bt65536"
TP8_BT_SCENARIO = "K2.5-tp8ep8-8k2k-bt65536"
TARGET_SCENARIOS = {DP2_BT_SCENARIO, TP8_BT_SCENARIO}

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
    "label",
    "scenario",
    "tp",
    "dp",
    "ep",
    "max_bt",
    "real_output_tok_s_gpu",
    "sim_before_tok_s_gpu",
    "sim_after_tok_s_gpu",
    "error_ratio_before",
    "error_ratio_after",
    "error_ratio_delta",
    "num_gpu_blocks",
    "full_sequence_capacity_per_engine",
    "per_replica_concurrency",
    "configured_max_num_batched_tokens",
    "budget_full_prefill_reqs",
    "initial_prefill_reqs_before",
    "initial_prefill_reqs_after",
    "initial_prefill_tokens_before",
    "initial_prefill_tokens_after",
    "representative_step_ms_before",
    "representative_step_ms_after",
    "representative_step_delta_ms",
    "context_non_attention_ms_before",
    "context_non_attention_ms_after",
    "context_attention_ms_before",
    "context_attention_ms_after",
    "moe_query_path_before",
    "moe_query_path_after",
    "ep_query_path_before",
    "ep_query_path_after",
    "capacity_classification",
    "dominant_cause",
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


def abs_error(sim: float, real: float) -> float:
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
    spec = importlib.util.spec_from_file_location("phase420_validate_cb_simulator", validate_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {validate_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_runtime_classes(repo_root: Path) -> tuple[Any, Any]:
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from aiconfigurator.sdk.backends.cb_simulator.datatypes import Request
    from aiconfigurator.sdk.backends.cb_simulator.scheduler import CBScheduler

    return CBScheduler, Request


def _scenario_label(label: str) -> str:
    if label not in {"before", "after"}:
        raise ValueError("label must be before or after")
    return label


def _capacity_classification(scenario: str, num_gpu_blocks: int) -> str:
    if scenario == DP2_BT_SCENARIO and num_gpu_blocks == 1609:
        return "dirty_capacity_conflict"
    if scenario == TP8_BT_SCENARIO and num_gpu_blocks == 21472:
        return "reference_dirty_capacity"
    return "capacity_from_validation_mapping"


def _moe_query_path(label: str, initial_tokens: int) -> str:
    if label == "before":
        return "phase397v_int4_wo_calibrated_sol"
    if initial_tokens <= 32768:
        return "moe_perf_lookup"
    return "phase397v_sol_out_of_coverage"


def _ep_query_path(label: str, initial_tokens: int) -> str:
    if label == "before":
        return "fallback_tp_dp_collectives"
    if initial_tokens > 8192:
        return "ep8_alltoall_fallback"
    return "vllm_module_exact_lookup"


def _representative_prefill_step_ms(validate: ModuleType, pt: Any, initial_reqs: int, initial_tokens: int) -> tuple[float, float, float]:
    model, db, _ = validate._load_model_and_db(tp=pt.tp, dp=pt.dp, moe_tp=pt.moe_tp, moe_ep=pt.moe_ep)
    backend = validate.VLLMBackend()
    from aiconfigurator.sdk.backends.cb_simulator.iteration_latency import IterationLatencyCalculator

    calc = IterationLatencyCalculator(
        backend,
        model,
        db,
        overlap_factor=0.0,
        per_iteration_overhead_ms=validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
    )
    total_ms = calc.compute(
        prefill_tokens=initial_tokens,
        prefill_batch_size=initial_reqs,
        prefill_seq_len=pt.isl,
        decode_batch_size=0,
        decode_avg_kv_len=0,
    )
    breakdown = calc.get_last_breakdown()
    if breakdown is None:
        raise RuntimeError("missing iteration latency breakdown")
    return (
        float(total_ms),
        float(breakdown.context_non_attention_ms),
        float(breakdown.context_attention_ms),
    )


def _make_trace_cb_config(validate: ModuleType, pt: Any) -> Any:
    if hasattr(validate, "_make_multi_config_cb_config"):
        return validate._make_multi_config_cb_config(
            pt,
            overlap_factor=0.0,
            ep8_per_iteration_overhead_ms=validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
        )
    return validate._make_cb_config(
        pt.isl,
        pt.batch_size,
        overlap_factor=0.0,
        per_iteration_overhead_ms=validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
        num_gpu_blocks=validate._multi_config_num_gpu_blocks(pt),
    )


def _trace_ctx_tokens(validate: ModuleType, pt: Any) -> int:
    if hasattr(validate, "_make_multi_config_cb_config"):
        return int(pt.max_num_batched_tokens)
    return int(pt.isl)


def collect_trace_rows(repo_root: Path = REPO_ROOT, label: str = "after") -> list[dict[str, str]]:
    label = _scenario_label(label)
    validate = _load_validate_module(repo_root)
    CBScheduler, Request = _load_runtime_classes(repo_root)
    backend = validate.VLLMBackend()
    loaded: dict[tuple[int, int, int, int], tuple[Any, Any]] = {}
    rows: list[dict[str, str]] = []

    for pt in validate.MULTI_CONFIG_DATA:
        if pt.name not in TARGET_SCENARIOS:
            continue
        key = (pt.tp, pt.dp, pt.moe_tp, pt.moe_ep)
        if key not in loaded:
            loaded[key] = validate._load_model_and_db(tp=pt.tp, dp=pt.dp, moe_tp=pt.moe_tp, moe_ep=pt.moe_ep)[:2]
        model, db = loaded[key]
        num_gpu_blocks = validate._multi_config_num_gpu_blocks(pt)
        cb_config = _make_trace_cb_config(validate, pt)
        cb_summary = backend.run_agg(
            model,
            db,
            validate.RuntimeConfig(batch_size=pt.batch_size, isl=pt.isl, osl=pt.osl),
            ctx_tokens=_trace_ctx_tokens(validate, pt),
            database_mode=validate.common.DatabaseMode.HYBRID,
            method="cb_sim",
            cb_config=cb_config,
        )
        sim = float(cb_summary.get_result_dict()["tokens/s/gpu"])
        real = float(pt.real_output_tok_s_gpu)
        per_replica_concurrency = math.ceil(pt.batch_size / pt.dp)
        waiting = [Request(request_id=i, isl=pt.isl, osl=pt.osl, arrival_time_ms=0.0) for i in range(per_replica_concurrency)]
        running: list[Any] = []
        scheduled = CBScheduler(cb_config).schedule(waiting, running)
        initial_reqs = len(scheduled.prefill_reqs)
        initial_tokens = int(scheduled.total_prefill_tokens)
        budget_full_prefill_reqs = min(per_replica_concurrency, pt.max_num_batched_tokens // pt.isl)
        capacity_tokens = num_gpu_blocks * int(cb_config.block_size)
        full_sequence_capacity = capacity_tokens / float(pt.isl + pt.osl)
        step_ms, non_attn_ms, attn_ms = _representative_prefill_step_ms(validate, pt, initial_reqs, initial_tokens)
        rows.append(
            {
                "source": SOURCE,
                "row_type": "trace",
                "label": label,
                "scenario": pt.name,
                "tp": str(pt.tp),
                "dp": str(pt.dp),
                "ep": str(pt.moe_ep),
                "max_bt": str(pt.max_num_batched_tokens),
                "real_output_tok_s_gpu": _fmt(real),
                "sim_output_tok_s_gpu": _fmt(sim),
                "error_ratio": _fmt(abs_error(sim, real)),
                "num_gpu_blocks": str(num_gpu_blocks),
                "full_sequence_capacity_per_engine": _fmt(full_sequence_capacity),
                "per_replica_concurrency": str(per_replica_concurrency),
                "configured_max_num_batched_tokens": str(cb_config.max_num_batched_tokens),
                "budget_full_prefill_reqs": str(budget_full_prefill_reqs),
                "initial_prefill_reqs": str(initial_reqs),
                "initial_prefill_tokens": str(initial_tokens),
                "representative_step_ms": _fmt(step_ms),
                "context_non_attention_ms": _fmt(non_attn_ms),
                "context_attention_ms": _fmt(attn_ms),
                "moe_query_path": _moe_query_path(label, initial_tokens),
                "ep_query_path": _ep_query_path(label, initial_tokens),
                "capacity_classification": _capacity_classification(pt.name, num_gpu_blocks),
            }
        )
    return rows


def _by_scenario(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    filtered = [row for row in rows if row.get("row_type") == "trace" and row.get("label") == label]
    return {row["scenario"]: row for row in filtered}


def _dominant_cause(scenario: str, before: dict[str, str], after: dict[str, str]) -> str:
    capacity = float(after["full_sequence_capacity_per_engine"])
    per_replica = float(after["per_replica_concurrency"])
    budget_reqs = float(after["budget_full_prefill_reqs"])
    initial_reqs = float(after["initial_prefill_reqs"])
    configured_bt = int(after["configured_max_num_batched_tokens"])
    scenario_bt = int(after["max_bt"])
    after_error = float(after["error_ratio"])
    before_error = float(before["error_ratio"])
    after_step = float(after["representative_step_ms"])
    before_step = float(before["representative_step_ms"])

    if scenario == DP2_BT_SCENARIO and configured_bt < scenario_bt and capacity < per_replica and initial_reqs < budget_reqs:
        return "bt65536_budget_not_wired_plus_dirty_capacity"
    if scenario == TP8_BT_SCENARIO and configured_bt < scenario_bt and after_error > before_error and after_step > before_step:
        return "phase417_single_prefill_step_cost_regression_after_budget_not_wired"
    if after_error > before_error:
        return "phase417_regression_unclassified"
    return "not_regressed"


def _verdict(diagnoses: list[dict[str, str]]) -> tuple[str, str]:
    causes = {row["dominant_cause"] for row in diagnoses}
    if {
        "bt65536_budget_not_wired_plus_dirty_capacity",
        "phase417_single_prefill_step_cost_regression_after_budget_not_wired",
    }.issubset(causes):
        return (
            "bt65536_budget_not_wired_plus_phase417_single_step_regression",
            "phase421_wire_bt65536_budget_then_retrace_query_cost",
        )
    if "bt65536_budget_not_wired_plus_dirty_capacity" in causes:
        return "bt65536_budget_not_wired_dirty_capacity_dominates", "phase421_wire_bt65536_budget"
    if "phase417_single_prefill_step_cost_regression_after_budget_not_wired" in causes:
        return "bt65536_phase417_single_step_regression", "phase421_retrace_phase417_query_branch_after_budget_wiring"
    return "bt65536_no_regression_identified", "phase421_recheck_trace_inputs"


def build_phase420_rows(*, before_csv: Path = DEFAULT_BEFORE_CSV, after_csv: Path = DEFAULT_AFTER_CSV) -> list[dict[str, str]]:
    before_by_name = _by_scenario(_read_csv(before_csv), "before")
    after_by_name = _by_scenario(_read_csv(after_csv), "after")
    if set(before_by_name) != set(after_by_name):
        raise ValueError("before/after trace scenarios do not match")

    diagnoses: list[dict[str, object]] = []
    for scenario in sorted(after_by_name):
        before = before_by_name[scenario]
        after = after_by_name[scenario]
        diagnoses.append(
            {
                "source": SOURCE,
                "row_type": "diagnosis",
                "scenario": scenario,
                "tp": after["tp"],
                "dp": after["dp"],
                "ep": after["ep"],
                "max_bt": after["max_bt"],
                "real_output_tok_s_gpu": after["real_output_tok_s_gpu"],
                "sim_before_tok_s_gpu": before["sim_output_tok_s_gpu"],
                "sim_after_tok_s_gpu": after["sim_output_tok_s_gpu"],
                "error_ratio_before": before["error_ratio"],
                "error_ratio_after": after["error_ratio"],
                "error_ratio_delta": float(after["error_ratio"]) - float(before["error_ratio"]),
                "num_gpu_blocks": after["num_gpu_blocks"],
                "full_sequence_capacity_per_engine": after["full_sequence_capacity_per_engine"],
                "per_replica_concurrency": after["per_replica_concurrency"],
                "configured_max_num_batched_tokens": after["configured_max_num_batched_tokens"],
                "budget_full_prefill_reqs": after["budget_full_prefill_reqs"],
                "initial_prefill_reqs_before": before["initial_prefill_reqs"],
                "initial_prefill_reqs_after": after["initial_prefill_reqs"],
                "initial_prefill_tokens_before": before["initial_prefill_tokens"],
                "initial_prefill_tokens_after": after["initial_prefill_tokens"],
                "representative_step_ms_before": before["representative_step_ms"],
                "representative_step_ms_after": after["representative_step_ms"],
                "representative_step_delta_ms": float(after["representative_step_ms"]) - float(before["representative_step_ms"]),
                "context_non_attention_ms_before": before["context_non_attention_ms"],
                "context_non_attention_ms_after": after["context_non_attention_ms"],
                "context_attention_ms_before": before["context_attention_ms"],
                "context_attention_ms_after": after["context_attention_ms"],
                "moe_query_path_before": before["moe_query_path"],
                "moe_query_path_after": after["moe_query_path"],
                "ep_query_path_before": before["ep_query_path"],
                "ep_query_path_after": after["ep_query_path"],
                "capacity_classification": after["capacity_classification"],
                "dominant_cause": _dominant_cause(scenario, before, after),
            }
        )
    verdict, next_phase = _verdict([{field: _fmt(row.get(field, "")) for field in OUTPUT_FIELDS} for row in diagnoses])
    summary = {
        "source": SOURCE,
        "row_type": "summary",
        "scenario": "bt65536_pair",
        "error_ratio_before": max(float(row["error_ratio_before"]) for row in diagnoses),
        "error_ratio_after": max(float(row["error_ratio_after"]) for row in diagnoses),
        "error_ratio_delta": max(float(row["error_ratio_after"]) for row in diagnoses)
        - max(float(row["error_ratio_before"]) for row in diagnoses),
        "dominant_cause": "mixed",
        "mechanism_verdict": verdict,
        "next_phase_target": next_phase,
    }
    rows: list[dict[str, str]] = [{field: _fmt(row.get(field, "")) for field in OUTPUT_FIELDS} for row in diagnoses]
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
        raise ValueError("Phase420 rows must not be empty")
    if sum(1 for row in rows if row.get("row_type") == "summary") != 1:
        raise ValueError("Phase420 expects exactly one summary row")
    for row in rows:
        if row.get("source") != SOURCE:
            raise ValueError("unexpected source")
        if row.get("phase405_penalty_read") != "false":
            raise ValueError("phase405_penalty_read must stay false")
        if row.get("gpu_allowed") != "false" or row.get("ssh_allowed") != "false":
            raise ValueError("Phase420 must not use GPU/SSH")
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


def write_phase420_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    _write_csv(path, OUTPUT_FIELDS, rows)


def render_phase420_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    summary = [row for row in rows if row["row_type"] == "summary"][0]
    diagnoses = [row for row in rows if row["row_type"] == "diagnosis"]
    lines = [
        "# Phase420 BT65536 Trace",
        "",
        f"Verdict: `{summary['mechanism_verdict']}`.",
        "",
        f"Default AIC: `{DEFAULT_READINESS}`.",
        "",
        "Direct finding: both bt65536 validation points still run cb_sim with `configured_max_num_batched_tokens=8000`, so the scheduler admits `1/8` possible full-prefill requests. The 64k budget path has not actually been exercised yet.",
        "",
        "## Result",
        "",
        "| Scenario | Before error | After error | Capacity | First prefill reqs | Step ms before -> after | Cause |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in diagnoses:
        lines.append(
            f"| {row['scenario']} | {row['error_ratio_before']} | {row['error_ratio_after']} | "
            f"{row['full_sequence_capacity_per_engine']} | {row['initial_prefill_reqs_after']}/{row['budget_full_prefill_reqs']} "
            f"(configured bt {row['configured_max_num_batched_tokens']}) | "
            f"{row['representative_step_ms_before']} -> {row['representative_step_ms_after']} | {row['dominant_cause']} |"
        )
    lines.extend(
        [
            "",
            "## Query Path Delta",
            "",
            "| Scenario | MoE before -> after | EP before -> after |",
            "|---|---|---|",
        ]
    )
    for row in diagnoses:
        lines.append(
            f"| {row['scenario']} | {row['moe_query_path_before']} -> {row['moe_query_path_after']} | "
            f"{row['ep_query_path_before']} -> {row['ep_query_path_after']} |"
        )
    lines.extend(
        [
            "",
            "## Phase421 Target",
            "",
            f"`{summary['next_phase_target']}`.",
            "",
            "Phase421 should first wire the scenario `max_num_batched_tokens=65536` into cb_sim validation. Only after that rerun this trace to decide whether large-step MoE/EP query cost still needs a code fix.",
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
    collect = sub.add_parser("collect")
    collect.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    collect.add_argument("--label", choices=["before", "after"], required=True)
    collect.add_argument("--out", type=Path, required=True)
    build = sub.add_parser("build")
    build.add_argument("--before-csv", type=Path, default=DEFAULT_BEFORE_CSV)
    build.add_argument("--after-csv", type=Path, default=DEFAULT_AFTER_CSV)
    build.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    build.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    if args.cmd == "collect":
        _write_csv(args.out, TRACE_FIELDS, collect_trace_rows(args.repo_root, args.label))
        return 0
    if args.cmd in (None, "build"):
        rows = build_phase420_rows(before_csv=args.before_csv, after_csv=args.after_csv)
        write_phase420_csv(args.csv_out, rows)
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_phase420_md(rows))
        return 0
    raise ValueError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
