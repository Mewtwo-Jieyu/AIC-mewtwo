from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

from aiconfigurator.sdk import common
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.operations import MoE, MoEDispatch
from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk.performance_result import PerformanceResult


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE380_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase380_vllm_module_runtime_binding_sufficiency_gate.csv"
)
DEFAULT_SYSTEMS_ROOT = REPO_ROOT / "src/aiconfigurator/systems"
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase381_local_end_to_end_exact_bucket_probe.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase381_local_end_to_end_exact_bucket_probe.md"
)

SOURCE = "phase381_local_end_to_end_exact_bucket_probe"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"
RUNTIME_MODEL = "moonshotai/Kimi-K2.5"
NON_KIMI_MODEL = "meta-llama/Llama-3.1-8B"
PERFDB_MODEL = "kimi-k2.5"
HARDWARE = "h200_sxm"
BACKEND = "vllm"
VLLM_VERSION = "0.19.0"
TOPOLOGY = "tp4dp2ep8"
NON_TOPOLOGY = "tp4dp1ep8"
QUANT_RUNTIME = "CompressedTensorsWNA16MarlinMoEMethod"
FUSED_MODULE = "fusedmoe_runner_compute"
EP8_MODULE = "ep8_comm_dispatch_combine"
REAL_BUCKETS = ["1", "15", "16", "241", "1808", "2048", "8192"]
REAL_BUCKETS_TEXT = "/".join(REAL_BUCKETS)
NEXT_PHASE = "phase382_runtime_binding_probe_sufficiency_gate"
PHASE380_NEXT = "phase381_local_end_to_end_exact_bucket_probe"

FIELDNAMES = [
    "source",
    "row_type",
    "verdict",
    "probe_level",
    "run_static_used",
    "runtime_model",
    "perfdb_model_key",
    "hardware",
    "backend",
    "vllm_version",
    "topology",
    "raw_tokens",
    "bucket_tokens",
    "measurement_boundary",
    "runner_boundary_api",
    "module_boundary",
    "quant_runtime",
    "query_vllm_module_call_count",
    "query_vllm_module_key",
    "latency_ms",
    "post_dispatch_policy",
    "post_dispatch_latency_ms",
    "fallback_query",
    "non_scope_reason",
    "fail_fast",
    "error",
    "combined_full_model_status",
    "next_allowed_phase",
    "new_perfdb_data",
    "gpu_allowed",
    "ssh_allowed",
    "default_aic_allowed",
    "interpolation_allowed",
    "extrapolation_allowed",
    "nearest_bucket_allowed",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

COMMON_FIELDS = {
    "source": SOURCE,
    "verdict": "",
    "probe_level": "local_end_to_end",
    "run_static_used": FALSE,
    "runtime_model": RUNTIME_MODEL,
    "perfdb_model_key": PERFDB_MODEL,
    "hardware": HARDWARE,
    "backend": BACKEND,
    "vllm_version": VLLM_VERSION,
    "topology": TOPOLOGY,
    "raw_tokens": "",
    "bucket_tokens": "",
    "measurement_boundary": "",
    "runner_boundary_api": "",
    "module_boundary": "",
    "quant_runtime": QUANT_RUNTIME,
    "query_vllm_module_call_count": "0",
    "query_vllm_module_key": "",
    "latency_ms": "",
    "post_dispatch_policy": "",
    "post_dispatch_latency_ms": "",
    "fallback_query": "",
    "non_scope_reason": "",
    "fail_fast": FALSE,
    "error": "",
    "combined_full_model_status": "",
    "next_allowed_phase": "",
    "new_perfdb_data": FALSE,
    "gpu_allowed": FALSE,
    "ssh_allowed": FALSE,
    "default_aic_allowed": FALSE,
    "interpolation_allowed": FALSE,
    "extrapolation_allowed": FALSE,
    "nearest_bucket_allowed": FALSE,
    "default_readiness": DEFAULT_READINESS,
    "diagnostic_only": TRUE,
    "valid_for_default": FALSE,
    "perf_database": FALSE,
}


class _ProbeBackend(BaseBackend):
    def run_agg(self, model, database, runtime_config, **kwargs):
        raise NotImplementedError("Phase381 local probe only uses run_static")

    def find_best_agg_result_under_constraints(self, model, database, runtime_config, **kwargs):
        raise NotImplementedError("Phase381 local probe only uses run_static")

    def _get_memory_usage(
        self,
        model,
        database,
        batch_size: int,
        beam_width: int,
        isl: int,
        osl: int,
        num_tokens: int = 0,
    ) -> dict[str, float]:
        return {"total": 0.0}


class _RecordingPerfDatabase(PerfDatabase):
    def __init__(self, systems_root: Path) -> None:
        super().__init__(HARDWARE, BACKEND, VLLM_VERSION, str(systems_root))
        self.vllm_module_calls: list[tuple[str, str, str, str, int, str, str]] = []
        self.fallback_queries: list[str] = []

    def query_vllm_module(
        self,
        model: str,
        hardware: str,
        vllm_version: str,
        topology: str,
        bucket_tokens: int,
        module_boundary: str,
        quant_runtime: str,
    ) -> PerformanceResult:
        self.vllm_module_calls.append(
            (
                model,
                hardware,
                vllm_version,
                topology,
                bucket_tokens,
                module_boundary,
                quant_runtime,
            )
        )
        return super().query_vllm_module(
            model,
            hardware,
            vllm_version,
            topology,
            bucket_tokens,
            module_boundary,
            quant_runtime,
        )

    def query_moe(self, *args, **kwargs) -> PerformanceResult:
        self.fallback_queries.append("query_moe")
        return PerformanceResult(9.0, energy=0.0)

    def query_custom_allreduce(self, *args, **kwargs) -> PerformanceResult:
        self.fallback_queries.append("query_custom_allreduce")
        return PerformanceResult(2.0, energy=0.0)

    def query_nccl(self, *args, **kwargs) -> PerformanceResult:
        self.fallback_queries.append("query_nccl")
        return PerformanceResult(3.0, energy=0.0)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return rows


def _require(row: dict[str, str], field: str, expected: str, label: str) -> None:
    actual = row.get(field)
    if actual != expected:
        raise ValueError(f"{label} {field} expected {expected!r}, got {actual!r}")


def _require_phase380(path: Path) -> None:
    rows = _read_csv(path)
    next_row = next((row for row in rows if row.get("row_type") == "next_phase"), None)
    if next_row is None:
        raise ValueError("Phase380 next_phase row missing")
    _require(next_row, "next_allowed_phase", PHASE380_NEXT, "phase380 next_phase")
    default_row = next((row for row in rows if row.get("row_type") == "default_aic_blocked"), None)
    if default_row is None:
        raise ValueError("Phase380 default_aic_blocked row missing")
    _require(default_row, "default_aic_allowed", FALSE, "phase380 default_aic_blocked")

    for row in rows:
        _require(row, "default_readiness", DEFAULT_READINESS, row["row_type"])
        _require(row, "diagnostic_only", TRUE, row["row_type"])
        _require(row, "valid_for_default", FALSE, row["row_type"])
        _require(row, "perf_database", FALSE, row["row_type"])


def _runtime_config(bucket_tokens: int) -> RuntimeConfig:
    config = RuntimeConfig()
    config.batch_size = bucket_tokens
    config.beam_width = 1
    config.isl = 1
    config.osl = 1
    config.prefix = 0
    return config


def _model(
    ops: list,
    *,
    model_name: str = RUNTIME_MODEL,
    topology: str = TOPOLOGY,
):
    if topology == TOPOLOGY:
        tp_size = 4
        attention_dp_size = 2
        moe_ep_size = 8
    elif topology == NON_TOPOLOGY:
        tp_size = 4
        attention_dp_size = 1
        moe_ep_size = 8
    else:
        raise ValueError(f"unknown probe topology {topology!r}")

    return SimpleNamespace(
        model_path=model_name,
        model_name=model_name,
        context_ops=ops,
        generation_ops=[],
        _nextn=0,
        config=SimpleNamespace(
            tp_size=tp_size,
            pp_size=1,
            attention_dp_size=attention_dp_size,
            moe_tp_size=1,
            moe_ep_size=moe_ep_size,
            gemm_quant_mode=common.GEMMQuantMode.float16,
            kvcache_quant_mode=common.KVCacheQuantMode.float16,
            fmha_quant_mode=common.FMHAQuantMode.float16,
            moe_quant_mode=common.MoEQuantMode.int4_wo,
            comm_quant_mode=common.CommQuantMode.half,
        ),
    )


def _moe_op(name: str = "phase381_moe") -> MoE:
    return MoE(
        name,
        1.0,
        hidden_size=7168,
        inter_size=2048,
        topk=8,
        num_experts=384,
        moe_tp_size=1,
        moe_ep_size=8,
        quant_mode=common.MoEQuantMode.int4_wo,
        workload_distribution="balanced",
        attention_dp_size=2,
        is_context=True,
        is_gated=True,
        scale_num_tokens=4,
    )


def _dispatch_op(name: str, *, pre_dispatch: bool) -> MoEDispatch:
    return MoEDispatch(
        name,
        1.0,
        hidden_size=7168,
        topk=8,
        num_experts=384,
        moe_tp_size=1,
        moe_ep_size=8,
        attention_dp_size=2,
        pre_dispatch=pre_dispatch,
        scale_num_tokens=4,
        is_context=True,
    )


def _run_static(ops: list, database: _RecordingPerfDatabase, *, bucket_tokens: int, model_name: str, topology: str):
    backend = _ProbeBackend()
    return backend.run_static(
        _model(ops, model_name=model_name, topology=topology),
        database,
        _runtime_config(bucket_tokens),
        mode="static_ctx",
    )


def _format_latency(value: float) -> str:
    return f"{value:.6f}"


def _call_key(call: tuple[str, str, str, str, int, str, str]) -> str:
    return "|".join(str(part) for part in call)


def _row(row_type: str, **overrides: str) -> dict[str, str]:
    row = dict(COMMON_FIELDS)
    row["row_type"] = row_type
    row.update(overrides)
    return row


def _probe_fusedmoe(systems_root: Path) -> dict[str, str]:
    database = _RecordingPerfDatabase(systems_root)
    summary = _run_static(
        [_moe_op()],
        database,
        bucket_tokens=32,
        model_name=RUNTIME_MODEL,
        topology=TOPOLOGY,
    )
    latency = float(summary.get_context_latency_dict()["phase381_moe"])
    if len(database.vllm_module_calls) != 1:
        raise ValueError("fusedmoe probe must issue exactly one query_vllm_module call")
    return _row(
        "fusedmoe_runner_compute_run_static_exact_bucket_probe",
        verdict="pass",
        probe_level="basebackend_run_static",
        run_static_used=TRUE,
        raw_tokens="32",
        bucket_tokens="16",
        measurement_boundary="fusedmoe_forward_runner_level",
        runner_boundary_api="FusedMoE.forward()",
        module_boundary=FUSED_MODULE,
        query_vllm_module_call_count=str(len(database.vllm_module_calls)),
        query_vllm_module_key=_call_key(database.vllm_module_calls[0]),
        latency_ms=_format_latency(latency),
    )


def _probe_ep8_dispatch(systems_root: Path) -> dict[str, str]:
    database = _RecordingPerfDatabase(systems_root)
    summary = _run_static(
        [
            _dispatch_op("phase381_ep8_pre", pre_dispatch=True),
            _dispatch_op("phase381_ep8_post", pre_dispatch=False),
        ],
        database,
        bucket_tokens=64,
        model_name=RUNTIME_MODEL,
        topology=TOPOLOGY,
    )
    latencies = summary.get_context_latency_dict()
    pre_latency = float(latencies["phase381_ep8_pre"])
    post_latency = float(latencies["phase381_ep8_post"])
    if len(database.vllm_module_calls) != 1:
        raise ValueError("ep8 dispatch probe must issue exactly one query_vllm_module call")
    if post_latency != 0.0:
        raise ValueError("post dispatch must remain zero after bucket check")
    return _row(
        "ep8_comm_dispatch_combine_run_static_pre_post_probe",
        verdict="pass",
        probe_level="basebackend_run_static",
        run_static_used=TRUE,
        raw_tokens="64",
        bucket_tokens="16",
        measurement_boundary="vllm_ep_group_dispatch_router_logits_plus_combine",
        module_boundary=EP8_MODULE,
        query_vllm_module_call_count=str(len(database.vllm_module_calls)),
        query_vllm_module_key=_call_key(database.vllm_module_calls[0]),
        latency_ms=_format_latency(pre_latency),
        post_dispatch_policy="zero_after_bucket_check",
        post_dispatch_latency_ms=_format_latency(post_latency),
    )


def _probe_bucket_128_fail_fast(systems_root: Path) -> dict[str, str]:
    database = _RecordingPerfDatabase(systems_root)
    error = ""
    try:
        _run_static(
            [_dispatch_op("phase381_ep8_bucket_128", pre_dispatch=True)],
            database,
            bucket_tokens=512,
            model_name=RUNTIME_MODEL,
            topology=TOPOLOGY,
        )
    except ValueError as exc:
        error = str(exc)
    if "bucket_tokens must be one of" not in error:
        raise ValueError(f"bucket 128 probe did not fail fast with bucket guard: {error}")
    return _row(
        "bucket_128_fail_fast_probe",
        verdict="pass",
        probe_level="basebackend_run_static",
        run_static_used=TRUE,
        raw_tokens="512",
        bucket_tokens="128",
        fail_fast=TRUE,
        error=error,
        query_vllm_module_call_count=str(len(database.vllm_module_calls)),
    )


def _probe_non_scope(systems_root: Path, *, model_name: str, topology: str, reason: str, row_type: str) -> dict[str, str]:
    database = _RecordingPerfDatabase(systems_root)
    summary = _run_static(
        [_moe_op()],
        database,
        bucket_tokens=32,
        model_name=model_name,
        topology=topology,
    )
    latency = float(summary.get_context_latency_dict()["phase381_moe"])
    if database.vllm_module_calls:
        raise ValueError(f"{row_type} unexpectedly used query_vllm_module")
    if database.fallback_queries != ["query_moe"]:
        raise ValueError(f"{row_type} expected query_moe fallback, got {database.fallback_queries}")
    return _row(
        row_type,
        verdict="pass",
        probe_level="basebackend_run_static",
        run_static_used=TRUE,
        runtime_model=model_name,
        topology=topology,
        raw_tokens="32",
        bucket_tokens="16",
        query_vllm_module_call_count="0",
        latency_ms=_format_latency(latency),
        fallback_query="query_moe",
        non_scope_reason=reason,
    )


def analyze_phase381_local_end_to_end_exact_bucket_probe(
    phase380_csv: Path = DEFAULT_PHASE380_CSV,
    systems_root: Path = DEFAULT_SYSTEMS_ROOT,
) -> list[dict[str, str]]:
    _require_phase380(phase380_csv)
    rows = [
        _row(
            "phase380_prerequisite",
            verdict="phase380_runtime_binding_sufficiency_gate_passed",
            next_allowed_phase=PHASE380_NEXT,
        ),
        _probe_fusedmoe(systems_root),
        _probe_ep8_dispatch(systems_root),
        _probe_bucket_128_fail_fast(systems_root),
        _probe_non_scope(
            systems_root,
            model_name=NON_KIMI_MODEL,
            topology=TOPOLOGY,
            reason="model_name_not_kimi",
            row_type="non_kimi_scope_probe",
        ),
        _probe_non_scope(
            systems_root,
            model_name=RUNTIME_MODEL,
            topology=NON_TOPOLOGY,
            reason="topology_not_tp4dp2ep8",
            row_type="non_topology_scope_probe",
        ),
        _row(
            "combined_full_model_exact_bucket_probe",
            verdict="blocked_by_bucket_mapping",
            run_static_used=FALSE,
            combined_full_model_status="blocked_by_bucket_mapping",
            fallback_query="not_bypassed",
            error=(
                "same_raw_tokens_maps_to_ep8_bucket_raw_div_4_and_"
                "fusedmoe_bucket_raw_div_4_times_2"
            ),
        ),
        _row(
            "default_aic_blocked",
            verdict="blocked_unit_local_probe_only",
        ),
        _row(
            "next_phase",
            verdict="phase382_runtime_binding_probe_sufficiency_gate_only",
            next_allowed_phase=NEXT_PHASE,
        ),
    ]
    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    expected = [
        "phase380_prerequisite",
        "fusedmoe_runner_compute_run_static_exact_bucket_probe",
        "ep8_comm_dispatch_combine_run_static_pre_post_probe",
        "bucket_128_fail_fast_probe",
        "non_kimi_scope_probe",
        "non_topology_scope_probe",
        "combined_full_model_exact_bucket_probe",
        "default_aic_blocked",
        "next_phase",
    ]
    if [row.get("row_type") for row in rows] != expected:
        raise ValueError("Phase381 row order or row_type set changed")
    for index, row in enumerate(rows, start=1):
        for field in FIELDNAMES:
            if field not in row:
                raise ValueError(f"row {index} missing field {field}")
        for field in (
            "new_perfdb_data",
            "gpu_allowed",
            "ssh_allowed",
            "default_aic_allowed",
            "interpolation_allowed",
            "extrapolation_allowed",
            "nearest_bucket_allowed",
            "valid_for_default",
            "perf_database",
        ):
            if row[field] != FALSE:
                raise ValueError(f"row {index} {field} must remain false")
        _require(row, "default_readiness", DEFAULT_READINESS, f"row {index}")
        _require(row, "diagnostic_only", TRUE, f"row {index}")


def write_phase381_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase381_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    fused = next(row for row in rows if row["row_type"].startswith("fusedmoe_runner"))
    ep8 = next(row for row in rows if row["row_type"].startswith("ep8_comm"))
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = f"""# Phase381 Local End-to-End Exact-Bucket Probe

Phase381 runs a local BaseBackend.run_static probe through the real MoE.query and MoEDispatch.query vLLM module binding.
It proves simulator runtime binding local exact lookup, not vLLM GPU runtime evidence.
It is local evidence only: no GPU, no SSH, no new PerfDatabase rows, and no Default AIC enablement.

| Gate | Verdict |
|---|---|
| FusedMoE runner probe | pass: raw_tokens {fused["raw_tokens"]} maps to bucket_tokens {fused["bucket_tokens"]}; exact key `{fused["query_vllm_module_key"]}`, latency {fused["latency_ms"]} ms |
| EP8 dispatch/combine probe | pass: raw_tokens {ep8["raw_tokens"]} maps to bucket_tokens {ep8["bucket_tokens"]}; exact key `{ep8["query_vllm_module_key"]}`, pre latency {ep8["latency_ms"]} ms, post latency {ep8["post_dispatch_latency_ms"]} ms |
| bucket 128 | fail-fast inside Kimi h200_sxm vLLM 0.19.0 tp4dp2ep8 scope |
| non-scope | non-Kimi and non-tp4dp2ep8 do not use vLLM module table |
| combined full model | blocked-by-bucket-mapping: same raw_tokens maps to ep8_bucket=raw//4 and fusedmoe_bucket=raw//4*2; not treated as failure and not bypassed |
| PerfDatabase | no new rows |
| GPU / SSH | not allowed |
| Default AIC | No-Go |

Phase382 may decide whether this local probe evidence is sufficient for a larger exact-bucket runtime gate.
It must not turn this probe into default AIC readiness.
"""
    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase380-csv", type=Path, default=DEFAULT_PHASE380_CSV)
    parser.add_argument("--systems-root", type=Path, default=DEFAULT_SYSTEMS_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase381_local_end_to_end_exact_bucket_probe(
        phase380_csv=args.phase380_csv,
        systems_root=args.systems_root,
    )
    write_phase381_csv(args.output_csv, rows)
    write_phase381_md(args.output_md, rows)


if __name__ == "__main__":
    main()
