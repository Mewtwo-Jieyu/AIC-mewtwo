#!/usr/bin/env python3
"""Phase416: trace MoE and EP-dispatch charge paths for the 32k mixed-prefill step."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MethodType
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aiconfigurator.sdk import common  # noqa: E402
from aiconfigurator.sdk.config import RuntimeConfig  # noqa: E402
from scripts import analyze_phase414_prefill_decode_audit as phase414  # noqa: E402
from scripts import analyze_phase415_prefill_charge_components as phase415  # noqa: E402


SOURCE = "phase416_moe_charge_trace"
DEFAULT_SCENARIO = "K2.5-tp4ep8dp2-32k3k"
PHASE414_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase414_prefill_decode_audit.csv"
PHASE415_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase415_prefill_charge_components.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase416_moe_charge_trace.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase416_moe_charge_trace.md"
MOE_PERF = REPO_ROOT / "src/aiconfigurator/systems/data/h200_sxm/vllm/0.19.0/moe_perf.txt"
VLLM_MODULE_PERF = REPO_ROOT / "src/aiconfigurator/systems/data/h200_sxm/vllm/0.19.0/vllm_module_perf.txt"
DEFAULT_READINESS = "No-Go"

HIDDEN_SIZE = 7168
INTER_SIZE = 2048
TOPK = 8
NUM_EXPERTS = 384
MOE_TP_SIZE = 1
MOE_EP_SIZE = 8
ATTENTION_DP_SIZE = 2
SCALE_NUM_TOKENS = 4
MIXED_GEN_REQS = 6
DECODE_BATCH = 9
DECODE_KV_LEN = phase414.DECODE_KV_BASE

CSV_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "artifact_dir",
    "component",
    "phase",
    "op_input_tokens",
    "scale_num_tokens",
    "attention_dp_size",
    "scaled_tokens",
    "expected_tokens",
    "token_ratio_actual_expected",
    "query_path",
    "perfdb_table",
    "coverage_min_tokens",
    "coverage_max_tokens",
    "coverage_contains_scaled",
    "coverage_contains_expected",
    "coverage_tokens",
    "db_call_count",
    "db_return_ms",
    "returned_ms",
    "roofline_lower_bound_ms",
    "undercharge_ms",
    "phase414_prefill_gap_ms",
    "phase415_component_gap_ms",
    "corrected_mixed_step_ms",
    "mechanism_verdict",
    "phase417_target",
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
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    return str(value)


def _safe_ratio(a: float, b: float) -> float:
    if b <= 0:
        return math.inf
    return a / b


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _row_by_component(rows: Iterable[Mapping[str, str]], row_type: str, component: str) -> Mapping[str, str]:
    for row in rows:
        if row.get("row_type") == row_type and row.get("component") == component:
            return row
    raise ValueError(f"missing {row_type}:{component}")


def _phase414_row(path: Path, scenario: str) -> Mapping[str, str]:
    for row in _read_csv_rows(path):
        if row["scenario"] == scenario:
            return row
    raise ValueError(f"missing Phase414 row for {scenario}")


def _phase415_rows(path: Path, scenario: str) -> list[dict[str, str]]:
    return [row for row in _read_csv_rows(path) if row["scenario"] == scenario]


def _contains_range(min_token: int, max_token: int, token: int) -> bool:
    return min_token <= token <= max_token


def _coverage_from_tokens(
    *,
    component: str,
    perfdb_table: str,
    tokens: list[int],
    scaled_tokens: int,
    expected_tokens: int,
    exact: bool,
) -> dict[str, Any]:
    if not tokens:
        min_token = 0
        max_token = 0
    else:
        min_token = min(tokens)
        max_token = max(tokens)
    contains_scaled = scaled_tokens in tokens if exact else _contains_range(min_token, max_token, scaled_tokens)
    contains_expected = expected_tokens in tokens if exact else _contains_range(min_token, max_token, expected_tokens)
    return {
        "component": component,
        "perfdb_table": perfdb_table,
        "coverage_min_tokens": min_token,
        "coverage_max_tokens": max_token,
        "coverage_contains_scaled": contains_scaled,
        "coverage_contains_expected": contains_expected,
        "coverage_tokens": tokens,
    }


def read_moe_int4_wo_coverage(
    path: Path = MOE_PERF,
    *,
    scaled_tokens: int,
    expected_tokens: int,
) -> dict[str, Any]:
    tokens: set[int] = set()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if (
                row.get("moe_dtype") == "int4_wo"
                and row.get("hidden_size") == str(HIDDEN_SIZE)
                and row.get("inter_size") == str(INTER_SIZE)
                and row.get("topk") == str(TOPK)
                and row.get("num_experts") == str(NUM_EXPERTS)
                and row.get("moe_tp_size") == str(MOE_TP_SIZE)
                and row.get("moe_ep_size") == str(MOE_EP_SIZE)
            ):
                tokens.add(int(row["num_tokens"]))
    return _coverage_from_tokens(
        component="moe_compute",
        perfdb_table="moe_perf",
        tokens=sorted(tokens),
        scaled_tokens=scaled_tokens,
        expected_tokens=expected_tokens,
        exact=False,
    )


def read_ep8_module_coverage(
    path: Path = VLLM_MODULE_PERF,
    *,
    scaled_tokens: int,
    expected_tokens: int,
) -> dict[str, Any]:
    tokens: set[int] = set()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("module_boundary") == "ep8_comm_dispatch_combine":
                tokens.add(int(row["bucket_tokens"]))
    return _coverage_from_tokens(
        component="ep_dispatch_combine",
        perfdb_table="vllm_module_perf",
        tokens=sorted(tokens),
        scaled_tokens=scaled_tokens,
        expected_tokens=expected_tokens,
        exact=True,
    )


def _moe_query_path(database: Any, quant_mode: Any) -> str:
    if (
        getattr(database, "system", None) == "h200_sxm"
        and getattr(database, "backend", None) == common.BackendName.vllm.value
        and getattr(database, "version", None) == "0.19.0"
        and quant_mode == common.MoEQuantMode.int4_wo
    ):
        return "phase397v_int4_wo_calibrated_sol"
    return "moe_perf_lookup"


def _expected_moe_tokens(prefill_tokens: int) -> int:
    # Unit is token-expert rows local to one EP rank. For EP8, 32k tokens * topk8 / ep8 = 32k rows.
    return prefill_tokens * TOPK // MOE_EP_SIZE


def _expected_dispatch_tokens(prefill_tokens: int) -> int:
    return prefill_tokens * TOPK // MOE_EP_SIZE


def _mixed_trace_records(
    *,
    gen_reqs: int,
    component_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    from scripts import validate_cb_simulator as validate_cb  # noqa: PLC0415

    model, db, backend = validate_cb._load_model_and_db(tp=4, dp=2, moe_tp=1, moe_ep=8)
    query_moe_original = db.query_moe
    query_custom_original = db.query_custom_allreduce
    query_nccl_original = db.query_nccl
    moe_calls: list[dict[str, Any]] = []
    custom_calls: list[dict[str, Any]] = []
    nccl_calls: list[dict[str, Any]] = []

    def traced_query_moe(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = query_moe_original(*args, **kwargs)
        moe_calls.append(
            {
                "kwargs": dict(kwargs),
                "query_path": _moe_query_path(self, kwargs.get("quant_mode")),
                "result_ms": float(result),
            }
        )
        return result

    def traced_custom(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = query_custom_original(*args, **kwargs)
        custom_calls.append({"args": args, "kwargs": dict(kwargs), "result_ms": float(result)})
        return result

    def traced_nccl(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = query_nccl_original(*args, **kwargs)
        nccl_calls.append({"args": args, "kwargs": dict(kwargs), "result_ms": float(result)})
        return result

    try:
        db.query_moe = MethodType(traced_query_moe, db)
        db.query_custom_allreduce = MethodType(traced_custom, db)
        db.query_nccl = MethodType(traced_nccl, db)
        total_tokens = phase414.FULL_CHUNK_TOKENS + gen_reqs
        backend.run_static(
            model,
            db,
            RuntimeConfig(batch_size=1, beam_width=1, isl=total_tokens, osl=1, prefix=0),
            mode="static_ctx",
        ).get_context_latency_dict()
    finally:
        db.query_moe = query_moe_original
        db.query_custom_allreduce = query_custom_original
        db.query_nccl = query_nccl_original

    if not moe_calls:
        raise ValueError("mixed trace did not capture query_moe")
    moe_call = moe_calls[0]
    moe_kwargs = moe_call["kwargs"]
    op_input_tokens = phase414.FULL_CHUNK_TOKENS + gen_reqs
    moe_scaled = int(moe_kwargs["num_tokens"])
    dispatch_scaled = max(1, op_input_tokens // SCALE_NUM_TOKENS)
    dispatch_volume = dispatch_scaled * HIDDEN_SIZE
    dispatch_custom = [
        call for call in custom_calls if call["args"] and int(call["args"][-1]) == dispatch_volume
    ]
    dispatch_nccl = [
        call for call in nccl_calls if call["args"] and int(call["args"][-1]) == dispatch_volume * ATTENTION_DP_SIZE
    ]
    dispatch_db_return = sum(call["result_ms"] for call in dispatch_custom + dispatch_nccl)

    moe_component = _row_by_component(component_rows, "mixed_component", "moe_compute")
    dispatch_component = _row_by_component(component_rows, "mixed_component", "ep_dispatch_combine")
    return [
        {
            "component": "moe_compute",
            "phase": "mixed_prefill",
            "op_input_tokens": op_input_tokens,
            "scaled_tokens": moe_scaled,
            "expected_tokens": _expected_moe_tokens(phase414.FULL_CHUNK_TOKENS),
            "query_path": moe_call["query_path"],
            "perfdb_table": "moe_perf_not_used"
            if moe_call["query_path"] == "phase397v_int4_wo_calibrated_sol"
            else "moe_perf",
            "db_call_count": len(moe_calls),
            "db_return_ms": moe_call["result_ms"],
            "returned_ms": float(moe_component["sim_mean_ms"]),
            "roofline_lower_bound_ms": float(moe_component["roofline_lower_bound_ms"]) / float(
                moe_component["prefill_step_count"]
            ),
            "phase415_component_gap_ms": float(moe_component["roofline_gap_ms"]) / float(
                moe_component["prefill_step_count"]
            ),
        },
        {
            "component": "ep_dispatch_combine",
            "phase": "mixed_prefill",
            "op_input_tokens": op_input_tokens,
            "scaled_tokens": dispatch_scaled,
            "expected_tokens": _expected_dispatch_tokens(phase414.FULL_CHUNK_TOKENS),
            "query_path": "fallback_tp_dp_collectives",
            "perfdb_table": "custom_allreduce+nccl",
            "db_call_count": len(dispatch_custom) + len(dispatch_nccl),
            "db_return_ms": dispatch_db_return,
            "returned_ms": float(dispatch_component["sim_mean_ms"]),
            "roofline_lower_bound_ms": float(dispatch_component["roofline_lower_bound_ms"]) / float(
                dispatch_component["prefill_step_count"]
            ),
            "phase415_component_gap_ms": float(dispatch_component["roofline_gap_ms"]) / float(
                dispatch_component["prefill_step_count"]
            ),
        },
    ]


def _decode_trace_records(component_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    from scripts import validate_cb_simulator as validate_cb  # noqa: PLC0415

    model, db, backend = validate_cb._load_model_and_db(tp=4, dp=2, moe_tp=1, moe_ep=8)
    query_moe_original = db.query_moe
    query_custom_original = db.query_custom_allreduce
    query_nccl_original = db.query_nccl
    moe_calls: list[dict[str, Any]] = []
    custom_calls: list[dict[str, Any]] = []
    nccl_calls: list[dict[str, Any]] = []

    def traced_query_moe(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = query_moe_original(*args, **kwargs)
        moe_calls.append(
            {
                "kwargs": dict(kwargs),
                "query_path": _moe_query_path(self, kwargs.get("quant_mode")),
                "result_ms": float(result),
            }
        )
        return result

    def traced_custom(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = query_custom_original(*args, **kwargs)
        custom_calls.append({"args": args, "kwargs": dict(kwargs), "result_ms": float(result)})
        return result

    def traced_nccl(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = query_nccl_original(*args, **kwargs)
        nccl_calls.append({"args": args, "kwargs": dict(kwargs), "result_ms": float(result)})
        return result

    try:
        db.query_moe = MethodType(traced_query_moe, db)
        db.query_custom_allreduce = MethodType(traced_custom, db)
        db.query_nccl = MethodType(traced_nccl, db)
        backend.run_static(
            model,
            db,
            RuntimeConfig(batch_size=DECODE_BATCH, beam_width=1, isl=DECODE_KV_LEN, osl=2, prefix=0),
            mode="static_gen",
        ).get_generation_latency_dict()
    finally:
        db.query_moe = query_moe_original
        db.query_custom_allreduce = query_custom_original
        db.query_nccl = query_nccl_original

    if not moe_calls:
        raise ValueError("decode trace did not capture query_moe")
    moe_call = moe_calls[0]
    moe_kwargs = moe_call["kwargs"]
    dispatch_scaled = max(1, DECODE_BATCH // SCALE_NUM_TOKENS)
    dispatch_volume = dispatch_scaled * HIDDEN_SIZE
    dispatch_custom = [
        call for call in custom_calls if call["args"] and int(call["args"][-1]) == dispatch_volume
    ]
    dispatch_nccl = [
        call for call in nccl_calls if call["args"] and int(call["args"][-1]) == dispatch_volume * ATTENTION_DP_SIZE
    ]
    decode_moe = _row_by_component(component_rows, "decode_component", "decode_moe_compute")
    decode_dispatch = _row_by_component(component_rows, "decode_component", "decode_ep_dispatch_combine")
    return [
        {
            "component": "moe_compute",
            "phase": "decode",
            "op_input_tokens": DECODE_BATCH,
            "scaled_tokens": int(moe_kwargs["num_tokens"]),
            "expected_tokens": DECODE_BATCH * ATTENTION_DP_SIZE,
            "query_path": moe_call["query_path"],
            "perfdb_table": "moe_perf_not_used"
            if moe_call["query_path"] == "phase397v_int4_wo_calibrated_sol"
            else "moe_perf",
            "db_call_count": len(moe_calls),
            "db_return_ms": moe_call["result_ms"],
            "returned_ms": float(decode_moe["sim_ms"]),
            "roofline_lower_bound_ms": 0.0,
            "phase415_component_gap_ms": 0.0,
        },
        {
            "component": "ep_dispatch_combine",
            "phase": "decode",
            "op_input_tokens": DECODE_BATCH,
            "scaled_tokens": dispatch_scaled,
            "expected_tokens": DECODE_BATCH * TOPK // MOE_EP_SIZE,
            "query_path": "fallback_tp_dp_collectives",
            "perfdb_table": "custom_allreduce+nccl",
            "db_call_count": len(dispatch_custom) + len(dispatch_nccl),
            "db_return_ms": sum(call["result_ms"] for call in dispatch_custom + dispatch_nccl),
            "returned_ms": float(decode_dispatch["sim_ms"]),
            "roofline_lower_bound_ms": 0.0,
            "phase415_component_gap_ms": 0.0,
        },
    ]


def _coverage_by_component(coverage_rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row["component"]): row for row in coverage_rows}


def _mechanism_verdict(trace_records: Iterable[Mapping[str, Any]]) -> tuple[str, str]:
    paths = {(row.get("component"), row.get("phase")): row.get("query_path") for row in trace_records}
    if paths.get(("moe_compute", "mixed_prefill")) == "phase397v_int4_wo_calibrated_sol" and paths.get(
        ("ep_dispatch_combine", "mixed_prefill")
    ) == "fallback_tp_dp_collectives":
        return (
            "moe_sol_prefill_misapplied_plus_ep_dispatch_fallback_undercharge",
            "split_prefill_moe_charge_and_restore_ep8_alltoall_charge",
        )
    return ("moe_charge_trace_inconclusive", "audit_prefill_query_chain_manually")


def build_rows_from_trace(
    *,
    scenario: str,
    artifact_dir: str,
    trace_records: list[Mapping[str, Any]],
    coverage_rows: list[Mapping[str, Any]],
    phase414_prefill_gap_ms: float,
    phase415_mixed_step_ms: float | None = None,
) -> list[dict[str, str]]:
    coverage = _coverage_by_component(coverage_rows)
    verdict, target = _mechanism_verdict(trace_records)
    mixed_moe = next(
        (row for row in trace_records if row.get("component") == "moe_compute" and row.get("phase") == "mixed_prefill"),
        None,
    )
    mixed_dispatch = next(
        (
            row
            for row in trace_records
            if row.get("component") == "ep_dispatch_combine" and row.get("phase") == "mixed_prefill"
        ),
        None,
    )
    corrected_mixed_step = 0.0
    if mixed_moe and mixed_dispatch:
        moe_returned = float(mixed_moe.get("returned_ms", 0.0))
        dispatch_returned = float(mixed_dispatch.get("returned_ms", 0.0))
        moe_roofline = float(mixed_moe.get("roofline_lower_bound_ms", 0.0))
        dispatch_roofline = float(mixed_dispatch.get("roofline_lower_bound_ms", 0.0))
        if phase415_mixed_step_ms is None:
            corrected_mixed_step = moe_roofline + dispatch_roofline
        else:
            # Lift only the two below-roofline terms; all other Phase415 mixed-step charges stay unchanged.
            corrected_mixed_step = phase415_mixed_step_ms - moe_returned - dispatch_returned + moe_roofline + dispatch_roofline
    rows: list[dict[str, Any]] = []
    for trace in trace_records:
        component = str(trace["component"])
        cov = coverage.get(component, {})
        scaled = int(trace.get("scaled_tokens", 0))
        expected = int(trace.get("expected_tokens", 0))
        roofline = float(trace.get("roofline_lower_bound_ms", 0.0))
        returned = float(trace.get("returned_ms", 0.0))
        rows.append(
            {
                "source": SOURCE,
                "row_type": "query_trace",
                "scenario": scenario,
                "artifact_dir": artifact_dir,
                "component": component,
                "phase": trace.get("phase", ""),
                "op_input_tokens": int(trace.get("op_input_tokens", 0)),
                "scale_num_tokens": SCALE_NUM_TOKENS,
                "attention_dp_size": ATTENTION_DP_SIZE,
                "scaled_tokens": scaled,
                "expected_tokens": expected,
                "token_ratio_actual_expected": _safe_ratio(scaled, expected),
                "query_path": trace.get("query_path", ""),
                "perfdb_table": trace.get("perfdb_table", ""),
                "coverage_min_tokens": cov.get("coverage_min_tokens", ""),
                "coverage_max_tokens": cov.get("coverage_max_tokens", ""),
                "coverage_contains_scaled": cov.get("coverage_contains_scaled", ""),
                "coverage_contains_expected": cov.get("coverage_contains_expected", ""),
                "coverage_tokens": cov.get("coverage_tokens", ""),
                "db_call_count": trace.get("db_call_count", ""),
                "db_return_ms": float(trace.get("db_return_ms", 0.0)),
                "returned_ms": returned,
                "roofline_lower_bound_ms": roofline,
                "undercharge_ms": max(0.0, roofline - returned),
                "phase414_prefill_gap_ms": phase414_prefill_gap_ms,
                "phase415_component_gap_ms": float(trace.get("phase415_component_gap_ms", 0.0)),
                "corrected_mixed_step_ms": "",
                "mechanism_verdict": verdict,
                "phase417_target": target,
            }
        )
    for cov in coverage_rows:
        rows.append(
            {
                "source": SOURCE,
                "row_type": "perfdb_coverage",
                "scenario": scenario,
                "artifact_dir": artifact_dir,
                "component": cov.get("component", ""),
                "phase": "coverage",
                "perfdb_table": cov.get("perfdb_table", ""),
                "coverage_min_tokens": cov.get("coverage_min_tokens", ""),
                "coverage_max_tokens": cov.get("coverage_max_tokens", ""),
                "coverage_contains_scaled": cov.get("coverage_contains_scaled", ""),
                "coverage_contains_expected": cov.get("coverage_contains_expected", ""),
                "coverage_tokens": cov.get("coverage_tokens", ""),
                "mechanism_verdict": verdict,
                "phase417_target": target,
            }
        )
    rows.append(
        {
            "source": SOURCE,
            "row_type": "summary",
            "scenario": scenario,
            "artifact_dir": artifact_dir,
            "component": "summary",
            "phase": "summary",
            "phase414_prefill_gap_ms": phase414_prefill_gap_ms,
            "corrected_mixed_step_ms": corrected_mixed_step,
            "mechanism_verdict": verdict,
            "phase417_target": target,
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


def build_phase416_rows(
    *,
    phase414_csv: Path = PHASE414_CSV,
    phase415_csv: Path = PHASE415_CSV,
    scenario: str = DEFAULT_SCENARIO,
) -> list[dict[str, str]]:
    phase414_row = _phase414_row(phase414_csv, scenario)
    phase415_rows = _phase415_rows(phase415_csv, scenario)
    artifact_dir = phase414_row["artifact_dir"]
    trace_records = _mixed_trace_records(gen_reqs=MIXED_GEN_REQS, component_rows=phase415_rows)
    trace_records.extend(_decode_trace_records(phase415_rows))
    mixed_moe = next(row for row in trace_records if row["component"] == "moe_compute" and row["phase"] == "mixed_prefill")
    mixed_dispatch = next(
        row for row in trace_records if row["component"] == "ep_dispatch_combine" and row["phase"] == "mixed_prefill"
    )
    coverage_rows = [
        read_moe_int4_wo_coverage(
            scaled_tokens=int(mixed_moe["scaled_tokens"]),
            expected_tokens=int(mixed_moe["expected_tokens"]),
        ),
        read_ep8_module_coverage(
            scaled_tokens=int(mixed_dispatch["scaled_tokens"]),
            expected_tokens=int(mixed_dispatch["expected_tokens"]),
        ),
    ]
    return build_rows_from_trace(
        scenario=scenario,
        artifact_dir=artifact_dir,
        trace_records=trace_records,
        coverage_rows=coverage_rows,
        phase414_prefill_gap_ms=float(phase414_row["prefill_gap_ms"]),
        phase415_mixed_step_ms=float(_row_by_component(phase415_rows, "summary", "summary")["sim_mean_ms"]),
    )


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Phase416 rows must not be empty")
    if sum(1 for row in rows if row.get("row_type") == "summary") != 1:
        raise ValueError("Phase416 expects exactly one summary row")
    for row in rows:
        if row.get("source") != SOURCE:
            raise ValueError("unexpected source")
        if row.get("phase405_penalty_read") != "false":
            raise ValueError("phase405_penalty_read must stay false")
        if row.get("gpu_allowed") != "false" or row.get("ssh_allowed") != "false":
            raise ValueError("Phase416 is offline and must not use GPU/SSH")
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


def write_phase416_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_phase416_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    summary = [row for row in rows if row["row_type"] == "summary"][0]
    traces = [row for row in rows if row["row_type"] == "query_trace"]
    coverage = [row for row in rows if row["row_type"] == "perfdb_coverage"]
    lines = [
        "# Phase416 MoE Charge Trace",
        "",
        "Phase416 只做离线查询链追踪；不改 runtime、PerfDatabase 或 gate。",
        "",
        "## Verdict",
        "",
        f"- verdict: `{summary['mechanism_verdict']}`",
        f"- Phase417 target: `{summary['phase417_target']}`",
        f"- roofline-lifted mixed step: `{summary['corrected_mixed_step_ms']}` ms",
        f"- Default AIC: `{summary['default_readiness']}`",
        "",
        "## Query Trace",
        "",
        "| component | phase | input tokens | scaled tokens | expected tokens | path | table | returned ms | roofline ms |",
        "|---|---|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in traces:
        lines.append(
            f"| {row['component']} | {row['phase']} | {row['op_input_tokens']} | {row['scaled_tokens']} | "
            f"{row['expected_tokens']} | {row['query_path']} | {row['perfdb_table']} | "
            f"{row['returned_ms']} | {row['roofline_lower_bound_ms']} |"
        )
    lines.extend(
        [
            "",
            "## PerfDB Coverage",
            "",
            "| component | table | min | max | contains scaled | contains expected |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for row in coverage:
        lines.append(
            f"| {row['component']} | {row['perfdb_table']} | {row['coverage_min_tokens']} | "
            f"{row['coverage_max_tokens']} | {row['coverage_contains_scaled']} | "
            f"{row['coverage_contains_expected']} |"
        )
    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            "- MoE: int4_wo 在 h200_sxm/vLLM 0.19.0 下提前走 Phase397v decode 锚定 calibrated-SOL；`moe_perf.txt` 有 32k 覆盖，但这次没有被查。",
            "- EP dispatch/combine: 32k mixed prefill 的 exact module bucket 不存在，fallback 只按 TP/DP collectives 的 scaled token 计费，不是 EP8 all-to-all token-expert 字节口径。",
            "- decode 侧也走同类路径，但 token 很小，只表现为小的固定缺口，不解释 32k prefill 主 gap。",
            "",
            "## Boundary",
            "",
            "- GPU/SSH: not used in Phase416.",
            "- Runtime/PerfDatabase/gate: not modified.",
            "- Phase405 penalty: not read.",
            "- Default AIC: No-Go.",
            "",
        ]
    )
    return "\n".join(lines)


def write_phase416_md(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase416_md(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase414-csv", type=Path, default=PHASE414_CSV)
    parser.add_argument("--phase415-csv", type=Path, default=PHASE415_CSV)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_phase416_rows(
        phase414_csv=args.phase414_csv,
        phase415_csv=args.phase415_csv,
        scenario=args.scenario,
    )
    write_phase416_csv(args.csv, rows)
    write_phase416_md(args.md, rows)
    summary = [row for row in rows if row["row_type"] == "summary"][0]
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")
    print(f"verdict={summary['mechanism_verdict']}")


if __name__ == "__main__":
    main()
