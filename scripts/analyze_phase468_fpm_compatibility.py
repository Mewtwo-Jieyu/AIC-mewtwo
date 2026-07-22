#!/usr/bin/env python3
"""Build Phase468 exact forward workloads and local FPM compatibility evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import analyze_phase467_perf_sources as phase467
import validate_cb_simulator as validate

from aiconfigurator.sdk import common
from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend
from aiconfigurator.sdk.backends.cb_simulator.forward_descriptor import (
    make_forward_workload_topology_key,
)
from aiconfigurator.sdk.config import RuntimeConfig


SCHEMA = "phase468_fpm_runtime_compatibility_v1"
SOURCE_LOCK_SCHEMA = "phase468_fpm_source_lock_v1"
BASELINE_COMMIT = "21d6d705c12c19b30e417c713589f703b54c06ef"
DEFAULT_OUT_DIR = (
    REPO_ROOT / "docs" / "iter_gap_investigation" / "phase468_fpm_compatibility"
)
NUMERIC_TOLERANCE = 1e-9
EXPECTED_MODEL_PATH = "moonshotai/Kimi-K2.5"
EXPECTED_ARCHITECTURE = "KimiK25ForConditionalGeneration"
EXPECTED_MODEL_CONFIG_SHA256 = (
    "58944cf9cf456619768616f38acd75b3d7ccc26d8b6a64d9e312aba4c8f0b927"
)
EXPECTED_HARDWARE = "h200_sxm"
EXPECTED_BACKEND = "vllm"
EXPECTED_BACKEND_VERSION = "0.19.0"
PHASE466_EXECUTION_MANIFEST_SHA256 = (
    "bf562c5a649106acb12f1f981cb8d60a637ee079d02c7f3666507b77ba6c7609"
)
EXPECTED_QUANT_RUNTIME = "CompressedTensorsWNA16MarlinMoEMethod"
EXPECTED_MODEL_QUANT_MODE = "int4_wo"
ALLOWED_TOPOLOGIES = {
    "tp8pp1dp1moetp1ep8cp1",
    "tp4pp1dp2moetp1ep8cp1",
}
DESCRIPTOR_MANIFEST_SCHEMA = "phase468_forward_workload_manifest_v1"
FPM_WORKLOAD_FIELDS = (
    "num_prefill_requests",
    "sum_prefill_tokens",
    "sum_prefill_kv_tokens",
    "num_decode_requests",
    "sum_decode_kv_tokens",
)
EXACT_SUPPORT_CONTRACT = {
    "architecture": EXPECTED_ARCHITECTURE,
    "model_path": EXPECTED_MODEL_PATH,
    "attention_source": "mla_module",
    "selected_ops": ["mla_context_module", "mla_generation_module"],
    "backend": EXPECTED_BACKEND,
    "quant_mode": EXPECTED_MODEL_QUANT_MODE,
}

_PINNED_SOURCES = (
    {
        "repository": "ai-dynamo/aiconfigurator",
        "commit": "cf1b3cbbc00e0a822891abd29310d68a259e79ed",
        "path": "collector/fpm_forward/types.py",
        "sha256": "d32e8b55794f447957f5c287e3e07b33d3616f03afcd3c06e34e5c862087e91f",
        "borrowed_fields": ["FPMPoint", "local_dp_rank_point_identity"],
    },
    {
        "repository": "ai-dynamo/aiconfigurator",
        "commit": "cf1b3cbbc00e0a822891abd29310d68a259e79ed",
        "path": "collector/fpm_forward/capabilities.py",
        "sha256": "83ce8d8036408b5f06a5132e86ac3eb9f38b74e4ee4c2ef6509b82d3e905383b",
        "borrowed_fields": [
            "support_level=exact",
            "support_level=family_template",
            "support_level=bootstrap_template",
        ],
    },
    {
        "repository": "ai-dynamo/aiconfigurator",
        "commit": "cf1b3cbbc00e0a822891abd29310d68a259e79ed",
        "path": "collector/fpm_forward/planner.py",
        "sha256": "6fbab3c49429f92ee11c3873ea18bada7926812f36485cb69a03b89d5f4f0ddf",
        "borrowed_fields": [
            "build_collection_plan",
            "has_model_cases=True",
            "resolve_model_capability",
        ],
    },
    {
        "repository": "ai-dynamo/aiconfigurator",
        "commit": "cf1b3cbbc00e0a822891abd29310d68a259e79ed",
        "path": "collector/fpm_forward/model_capability.py",
        "sha256": "bb4f67bc94568ac32db158d6afe35e9e2c526a7b31f67640c6e2fd38c8209337",
        "borrowed_fields": [
            "ResolvedModelConfig.sha256",
            "resolve_attention_source",
            "mla_module",
        ],
    },
    {
        "repository": "ai-dynamo/aiconfigurator",
        "commit": "cf1b3cbbc00e0a822891abd29310d68a259e79ed",
        "path": "collector/model_cases.py",
        "sha256": "45c2b38a9b494629ae81cdaaf015911374a9961625e203bb96063bd8d9120f64",
        "borrowed_fields": [
            "model_path",
            "model_architecture",
            "selected_ops",
            "framework_specific_op_cases",
        ],
    },
    {
        "repository": "ai-dynamo/aiconfigurator",
        "commit": "cf1b3cbbc00e0a822891abd29310d68a259e79ed",
        "path": "collector/cases/models/KimiK25ForConditionalGeneration_cases.yaml",
        "sha256": "5e6619e0845a371e52aff408b7bb9302372a1458ebe0595055529a36661f3e34",
        "borrowed_fields": [
            "architecture=KimiK25ForConditionalGeneration",
            "model_path=moonshotai/Kimi-K2.5",
            "vllm.mla_context_module",
            "vllm.mla_generation_module",
            "vllm.quant_mode=int4_wo",
        ],
    },
    {
        "repository": "ai-dynamo/aiconfigurator",
        "commit": "cf1b3cbbc00e0a822891abd29310d68a259e79ed",
        "path": "collector/fpm_forward/native_artifact.py",
        "sha256": "13d52d344c76b2287d2a9404d353faf044e2885d9ca0cd830093f06980cb0113",
        "borrowed_fields": list(FPM_WORKLOAD_FIELDS),
    },
    {
        "repository": "ai-dynamo/dynamo",
        "commit": "41882ae9b07232eed4850fb1daf8c958abb2556a",
        "path": "components/src/dynamo/common/forward_pass_metrics.py",
        "sha256": "523dfc4e7ccbc5a37dbcf03b331a41b628227ad8a5d9c64fda8abc96dd31b188",
        "borrowed_fields": list(FPM_WORKLOAD_FIELDS),
    },
    {
        "repository": "ai-dynamo/dynamo",
        "commit": "41882ae9b07232eed4850fb1daf8c958abb2556a",
        "path": "components/src/dynamo/vllm/instrumented_scheduler.py",
        "sha256": "443255313134f32331c6b40cfc90fe72064f207e089c8d95867f8963c22e20a4",
        "borrowed_fields": [
            "SchedulerOutput",
            "data_parallel_index",
            "num_computed_tokens_before_update",
            *FPM_WORKLOAD_FIELDS,
        ],
    },
)


class SourceContractError(ValueError):
    """Pinned upstream source contract is missing or changed."""


class RuntimeContractError(ValueError):
    """Local Phase468 runtime/descriptor contract is not exact."""


def expected_source_lock_manifest() -> dict:
    return {
        "schema": SOURCE_LOCK_SCHEMA,
        "sources": deepcopy(list(_PINNED_SOURCES)),
        "exact_support": deepcopy(EXACT_SUPPORT_CONTRACT),
        "policy": {
            "cherry_pick": False,
            "follow_pr_head": False,
            "borrow_contract_only": True,
        },
    }


def _source_key(row: dict) -> tuple[str, str]:
    return str(row.get("repository", "")), str(row.get("path", ""))


def validate_source_lock_manifest(manifest: dict) -> dict:
    expected = expected_source_lock_manifest()
    if manifest.get("schema") != SOURCE_LOCK_SCHEMA:
        raise SourceContractError("source lock schema mismatch")
    rows = manifest.get("sources")
    if not isinstance(rows, list):
        raise SourceContractError("source set is missing")
    actual_by_key = {_source_key(row): row for row in rows if isinstance(row, dict)}
    expected_by_key = {_source_key(row): row for row in expected["sources"]}
    if len(actual_by_key) != len(rows) or set(actual_by_key) != set(expected_by_key):
        raise SourceContractError("source set does not match pinned contract")
    for key, expected_row in expected_by_key.items():
        actual = actual_by_key[key]
        if actual.get("commit") != expected_row["commit"]:
            raise SourceContractError(f"source commit mismatch:{key}")
        if actual.get("sha256") != expected_row["sha256"]:
            raise SourceContractError(f"source hash mismatch:{key}")
        if actual.get("borrowed_fields") != expected_row["borrowed_fields"]:
            raise SourceContractError(f"borrowed fields mismatch:{key}")
    if manifest.get("policy") != expected["policy"]:
        raise SourceContractError("source lock policy mismatch")
    if manifest.get("exact_support") != expected["exact_support"]:
        raise SourceContractError("exact support contract mismatch")
    return manifest


def validate_runtime_contract(contract: dict) -> dict:
    topology_values = {
        name: contract.get(name)
        for name in ("tp", "pp", "dp", "moe_tp", "moe_ep", "cp")
    }
    topology_complete = all(
        isinstance(value, int) and value > 0 for value in topology_values.values()
    )
    expected_topology = None
    if topology_complete:
        expected_topology = make_forward_workload_topology_key(**topology_values)
    gpu_count = contract.get("gpu_count")
    topology_gpu_count = (
        topology_values["tp"] * topology_values["pp"] * topology_values["dp"]
        if topology_complete
        else None
    )
    moe_world_matches = topology_complete and (
        topology_values["moe_tp"] * topology_values["moe_ep"]
        == topology_values["tp"] * topology_values["dp"]
    )
    checks = (
        (contract.get("model_path") == EXPECTED_MODEL_PATH, "model path"),
        (
            contract.get("model_config_sha256") == EXPECTED_MODEL_CONFIG_SHA256,
            "model config SHA256",
        ),
        (contract.get("architecture") == EXPECTED_ARCHITECTURE, "model architecture"),
        (contract.get("support_level") == "exact", "support_level must be exact"),
        (contract.get("hardware") == EXPECTED_HARDWARE, "hardware"),
        (contract.get("backend") == EXPECTED_BACKEND, "backend"),
        (
            contract.get("backend_version") == EXPECTED_BACKEND_VERSION,
            "backend version",
        ),
        (topology_complete, "topology dimensions"),
        (topology_values["pp"] == 1, "PP must be 1"),
        (topology_values["cp"] == 1, "CP must be 1"),
        (gpu_count == topology_gpu_count == 8, "GPU count"),
        (moe_world_matches, "MoE world"),
        (
            contract.get("topology") == expected_topology
            and expected_topology in ALLOWED_TOPOLOGIES,
            "topology",
        ),
        (
            contract.get("quant_runtime") == EXPECTED_QUANT_RUNTIME,
            "quant runtime",
        ),
        (contract.get("engine_loop_enabled") is False, "engine loop is unsupported"),
    )
    failures = [label for passed, label in checks if not passed]
    if failures:
        raise RuntimeContractError("runtime contract mismatch: " + ", ".join(failures))
    return {"status": "PASS", **contract}


def validate_descriptor_coverage(execution_mode: str, rows: list[dict]) -> dict:
    if not rows:
        raise RuntimeContractError("descriptor coverage is empty")
    if {row.get("execution_mode", execution_mode) for row in rows} != {
        execution_mode
    }:
        raise RuntimeContractError("descriptor execution mode mismatch")
    by_step: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_step[int(row["engine_step_id"])].append(row)
    if any(step <= 0 for step in by_step):
        raise RuntimeContractError("descriptor step id must be positive")
    ranks = sorted({int(row["dp_rank"]) for row in rows})
    rank1_status = "present" if 1 in ranks else "not_applicable"
    if execution_mode == "dp_lockstep":
        for step_rows in by_step.values():
            if sorted(int(row["dp_rank"]) for row in step_rows) != [0, 1]:
                raise RuntimeContractError("lockstep rank coverage is incomplete")
    elif execution_mode == "dp_legacy":
        if ranks != [0, 1] or any(not row["active"] for row in rows):
            raise RuntimeContractError("legacy DP rank coverage is incomplete")
    elif execution_mode == "dp_legacy_representative":
        if ranks != [0] or any(not row["active"] for row in rows):
            raise RuntimeContractError("legacy representative coverage is invalid")
        rank1_status = "not_executed_by_official_legacy_path"
    elif execution_mode == "tp_single_replica":
        if ranks != [0] or any(not row["active"] for row in rows):
            raise RuntimeContractError("single-replica coverage is invalid")
    else:
        raise RuntimeContractError(f"unsupported execution mode:{execution_mode}")
    return {
        "execution_mode": execution_mode,
        "step_count": len(by_step),
        "row_count": len(rows),
        "active_row_count": sum(bool(row["active"]) for row in rows),
        "idle_row_count": sum(not bool(row["active"]) for row in rows),
        "ranks": ranks,
        "rank1_status": rank1_status,
        "status": "PASS",
    }


def _point_topology(point) -> str:
    return make_forward_workload_topology_key(
        tp=point.tp,
        pp=1,
        dp=point.dp,
        moe_tp=point.moe_tp,
        moe_ep=point.moe_ep,
        cp=1,
    )


def _expected_execution_mode(point) -> str:
    if point.dp == 1:
        return "tp_single_replica"
    if point.max_num_batched_tokens == point.isl:
        return "dp_lockstep"
    return "dp_legacy_representative"


def _validate_local_model_contract(model) -> dict:
    config_path = (
        REPO_ROOT
        / "src"
        / "aiconfigurator"
        / "model_configs"
        / "moonshotai--Kimi-K2.5_config.json"
    )
    payload_bytes = config_path.read_bytes()
    payload = json.loads(payload_bytes)
    canonical_payload = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    config_sha256 = hashlib.sha256(canonical_payload).hexdigest()
    if config_sha256 != EXPECTED_MODEL_CONFIG_SHA256:
        raise RuntimeContractError("model config hash mismatch")
    architectures = payload.get("architectures")
    if architectures != [EXPECTED_ARCHITECTURE]:
        raise RuntimeContractError("model architecture mismatch")
    actual_ops = {
        type(operation).__name__
        for operation in (*model.context_ops, *model.generation_ops)
    }
    required_ops = {"ContextMLA", "GenerationMLA"}
    if not required_ops.issubset(actual_ops):
        raise RuntimeContractError("exact MLA operation set is missing")
    return {
        "architecture": architectures[0],
        "model_config_sha256": config_sha256,
        "exact_attention_source": "mla_module",
        "actual_attention_operations": sorted(required_ops),
        "support_level": "exact",
    }


def _runtime_contract(point, model, database, cb_config) -> dict:
    local_model = _validate_local_model_contract(model)
    quant_mode = getattr(model.config.moe_quant_mode, "name", "")
    if quant_mode != EXPECTED_MODEL_QUANT_MODE:
        raise RuntimeContractError(
            f"model quant mode mismatch:{quant_mode!r}"
        )
    return {
        "model_path": model.model_path,
        "architecture": local_model["architecture"],
        "support_level": local_model["support_level"],
        "support_evidence": {
            "collector_entry": "build_collection_plan(has_model_cases=True)",
            "model_capability_source": "collector/fpm_forward/model_capability.py",
            "model_cases_source": "collector/model_cases.py",
            "kimi_case_source": (
                "collector/cases/models/"
                "KimiK25ForConditionalGeneration_cases.yaml"
            ),
            "attention_source": local_model["exact_attention_source"],
            "selected_ops": EXACT_SUPPORT_CONTRACT["selected_ops"],
            "actual_operations": local_model["actual_attention_operations"],
            "model_config_sha256": local_model["model_config_sha256"],
            "model_quant_mode": quant_mode,
        },
        "hardware": database.system,
        "backend": database.backend,
        "backend_version": database.version,
        "gpu_count": point.tp * point.dp,
        "tp": point.tp,
        "pp": 1,
        "dp": point.dp,
        "moe_tp": point.moe_tp,
        "moe_ep": point.moe_ep,
        "cp": 1,
        "topology": _point_topology(point),
        "model_config_sha256": local_model["model_config_sha256"],
        "quant_runtime": cb_config.forward_workload_quant_runtime,
        "model_quant_mode": quant_mode,
        "engine_loop_enabled": cb_config.engine_loop_enabled,
    }


def _serialize_rows(rows: list[dict]) -> bytes:
    return (
        json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _run_point(point, model, database) -> tuple[dict, list[dict], dict, dict]:
    official_config = validate._make_official_validation_cb_config(
        point,
        overlap_factor=0.0,
        ep8_per_iteration_overhead_ms=(
            validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS
        ),
    )
    cb_config = replace(
        official_config,
        capture_forward_workloads=True,
        forward_workload_scenario=point.name,
        forward_workload_model_config_sha256=EXPECTED_MODEL_CONFIG_SHA256,
        forward_workload_cp_size=1,
        forward_workload_quant_runtime=EXPECTED_QUANT_RUNTIME,
    )
    contract = _runtime_contract(point, model, database, cb_config)
    validated_contract = validate_runtime_contract(contract)
    results: list[tuple[dict, list[dict]]] = []
    for _ in range(2):
        backend = VLLMBackend()
        with validate._official_validation_metric_scope(point):
            summary = backend.run_agg(
                model,
                database,
                RuntimeConfig(
                    batch_size=point.batch_size,
                    isl=point.isl,
                    osl=point.osl,
                ),
                ctx_tokens=point.max_num_batched_tokens,
                database_mode=common.DatabaseMode.HYBRID,
                method="cb_sim",
                cb_config=cb_config,
            )
        result = summary.get_result_dict()
        rows = [
            asdict(descriptor)
            for descriptor in summary.get_forward_workload_descriptors()
        ]
        results.append((result, rows))
    first_result, first_rows = results[0]
    second_result, second_rows = results[1]
    if _serialize_rows(first_rows) != _serialize_rows(second_rows):
        raise RuntimeContractError(
            f"descriptor determinism mismatch:{point.name}"
        )
    if not math.isclose(
        float(first_result["tokens/s/gpu"]),
        float(second_result["tokens/s/gpu"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeContractError(f"numeric repeat mismatch:{point.name}")
    expected_mode = _expected_execution_mode(point)
    coverage = validate_descriptor_coverage(expected_mode, first_rows)
    for row in first_rows:
        row_contract = {
            "model_path": row["model_path"],
            "model_config_sha256": row["model_config_sha256"],
            "architecture": EXPECTED_ARCHITECTURE,
            "support_level": "exact",
            "hardware": row["hardware"],
            "backend": row["backend"],
            "backend_version": row["backend_version"],
            "gpu_count": row["tp"] * row["pp"] * row["dp"],
            "tp": row["tp"],
            "pp": row["pp"],
            "dp": row["dp"],
            "moe_tp": row["moe_tp"],
            "moe_ep": row["moe_ep"],
            "cp": row["cp"],
            "topology": row["topology"],
            "quant_runtime": row["quant_runtime"],
            "engine_loop_enabled": cb_config.engine_loop_enabled,
        }
        validate_runtime_contract(row_contract)
        if row["scenario"] != point.name:
            raise RuntimeContractError(
                f"descriptor scenario mismatch:{point.name}"
            )
    expected_numeric = phase467.EXPECTED_SIM_OUTPUT_TOK_S_GPU[point.name]
    actual_numeric = float(first_result["tokens/s/gpu"])
    delta = abs(actual_numeric - expected_numeric)
    if delta > NUMERIC_TOLERANCE:
        raise RuntimeContractError(
            f"numeric invariance mismatch:{point.name}:{delta!r}"
        )
    numeric = {
        "scenario": point.name,
        "phase467_sim_output_tok_s_gpu": expected_numeric,
        "phase468_sim_output_tok_s_gpu": actual_numeric,
        "absolute_delta": delta,
        "tolerance": NUMERIC_TOLERANCE,
        "status": "PASS",
    }
    return (
        numeric,
        first_rows,
        {"scenario": point.name, **coverage},
        {"scenario": point.name, **validated_contract},
    )


def _phase_query_point(row: dict, phase: str) -> tuple:
    if phase == "prefill":
        return (
            row["topology"],
            int(row["num_prefill_requests"]),
            int(row["sum_prefill_tokens"]),
            int(row["sum_prefill_kv_tokens"]),
        )
    if phase == "decode":
        return (
            row["topology"],
            int(row["num_decode_requests"]),
            int(row["sum_decode_kv_tokens"]),
        )
    raise ValueError(f"unsupported workload phase:{phase}")


def build_workload_domains(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    active = [row for row in rows if bool(row["active"])]
    if not active:
        raise RuntimeContractError("descriptor set has no active workloads")
    by_topology: dict[str, list[dict]] = defaultdict(list)
    for row in active:
        by_topology[str(row["topology"])].append(row)

    domains: list[dict] = []
    topology_rows: list[dict] = []
    for topology in sorted(by_topology):
        topology_workloads = by_topology[topology]
        mixed_row_count = sum(
            int(row["num_prefill_requests"]) > 0
            and int(row["num_decode_requests"]) > 0
            for row in topology_workloads
        )
        phase_points: dict[str, set[tuple]] = {}
        for phase, request_field, token_field, kv_field in (
            (
                "prefill",
                "num_prefill_requests",
                "sum_prefill_tokens",
                "sum_prefill_kv_tokens",
            ),
            (
                "decode",
                "num_decode_requests",
                None,
                "sum_decode_kv_tokens",
            ),
        ):
            phase_rows = [
                row for row in topology_workloads if int(row[request_field]) > 0
            ]
            if not phase_rows:
                raise RuntimeContractError(f"{phase} domain is empty:{topology}")
            request_values = [int(row[request_field]) for row in phase_rows]
            kv_values = [int(row[kv_field]) for row in phase_rows]
            token_values = (
                [int(row[token_field]) for row in phase_rows]
                if token_field is not None
                else None
            )
            points = {_phase_query_point(row, phase) for row in phase_rows}
            phase_points[phase] = points
            domains.append(
                {
                    "topology": topology,
                    "phase": phase,
                    "row_count": len(phase_rows),
                    "num_requests_min": min(request_values),
                    "num_requests_max": max(request_values),
                    "sum_tokens_min": min(token_values) if token_values else None,
                    "sum_tokens_max": max(token_values) if token_values else None,
                    "sum_kv_tokens_min": min(kv_values),
                    "sum_kv_tokens_max": max(kv_values),
                    "unique_query_point_count": len(points),
                    "mixed_row_count": mixed_row_count,
                }
            )
        topology_rows.append(
            {
                "topology": topology,
                "active_row_count": len(topology_workloads),
                "scenario_count": len(
                    {str(row["scenario"]) for row in topology_workloads}
                ),
                "mixed_row_count": mixed_row_count,
                "unique_prefill_query_point_count": len(phase_points["prefill"]),
                "unique_decode_query_point_count": len(phase_points["decode"]),
                "unique_query_point_count": sum(
                    len(points) for points in phase_points.values()
                ),
            }
        )
    return domains, topology_rows


class _HashingTextSink:
    def __init__(self) -> None:
        self.digest = hashlib.sha256()

    def write(self, text: str) -> int:
        encoded = text.encode("utf-8")
        self.digest.update(encoded)
        return len(text)


def _write_csv_stream(stream, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeContractError("refusing to serialize empty descriptor rows")
    fieldnames = list(rows[0])
    expected_fields = set(fieldnames)
    if any(set(row) != expected_fields for row in rows):
        raise RuntimeContractError("descriptor rows have inconsistent fields")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def build_descriptor_manifest(rows: list[dict]) -> dict:
    sink = _HashingTextSink()
    _write_csv_stream(sink, rows)
    query_points = set()
    for row in rows:
        if not bool(row["active"]):
            continue
        if int(row["num_prefill_requests"]) > 0:
            query_points.add(("prefill", *_phase_query_point(row, "prefill")))
        if int(row["num_decode_requests"]) > 0:
            query_points.add(("decode", *_phase_query_point(row, "decode")))
    return {
        "schema": DESCRIPTOR_MANIFEST_SCHEMA,
        "row_count": len(rows),
        "unique_query_point_count": len(query_points),
        "scenario_row_counts": dict(
            sorted(Counter(str(row["scenario"]) for row in rows).items())
        ),
        "topology_row_counts": dict(
            sorted(Counter(str(row["topology"]) for row in rows).items())
        ),
        "streaming_csv_sha256": sink.digest.hexdigest(),
        "csv_wire_format": "utf8_rfc4180_header_and_rows_lf",
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
        "default_aic": "No-Go",
    }


def write_full_descriptor_csv(path: Path, rows: list[dict]) -> None:
    resolved = path.expanduser().resolve()
    repo = REPO_ROOT.resolve()
    if resolved == repo or repo in resolved.parents:
        raise RuntimeContractError(
            "full descriptor CSV output must be outside repository"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", newline="", encoding="utf-8") as stream:
        _write_csv_stream(stream, rows)


def collect() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    loaded: dict[tuple[int, int, int, int], tuple] = {}
    numeric_rows: list[dict] = []
    descriptor_rows: list[dict] = []
    coverage_rows: list[dict] = []
    runtime_contracts: list[dict] = []
    for point in validate.MULTI_CONFIG_DATA:
        key = (point.tp, point.dp, point.moe_tp, point.moe_ep)
        if key not in loaded:
            loaded[key] = validate._load_model_and_db(
                tp=point.tp,
                dp=point.dp,
                moe_tp=point.moe_tp,
                moe_ep=point.moe_ep,
            )[:2]
        model, database = loaded[key]
        numeric, rows, coverage, runtime_contract = _run_point(
            point,
            model,
            database,
        )
        numeric_rows.append(numeric)
        descriptor_rows.extend(rows)
        coverage_rows.append(coverage)
        runtime_contracts.append(runtime_contract)
    return (
        numeric_rows,
        descriptor_rows,
        coverage_rows,
        runtime_contracts,
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeContractError(f"refusing to write empty CSV:{path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _base_report(status: str) -> dict:
    return {
        "schema": SCHEMA,
        "status": status,
        "phase467_baseline_commit": BASELINE_COMMIT,
        "workload_fields": list(FPM_WORKLOAD_FIELDS),
        "model_path": EXPECTED_MODEL_PATH,
        "model_config_sha256": EXPECTED_MODEL_CONFIG_SHA256,
        "phase466_execution_manifest_sha256": PHASE466_EXECUTION_MANIFEST_SHA256,
        "phase466_execution_manifest_scope": "historical_evidence_reference_only",
        "remote_runtime_method_check": "deferred_to_phase469_no_model_preflight",
        "phase466_csv_input": False,
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
        "default_aic": "No-Go",
    }


def _write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# Phase468 FPM Input and Runtime Compatibility",
        "",
        f"Status: `{report['status']}`.",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Source contract | {report.get('source_contract', 'BLOCKED')} |",
        f"| Scenarios | {report.get('scenario_count', 0)}/6 |",
        f"| Descriptor rows | {report.get('descriptor_row_count', 0)} |",
        f"| Numeric max delta | {report.get('max_numeric_delta', 'n/a')} |",
        "| Worker/Dynamo method check | Deferred to Phase469 preflight |",
        "| Default AIC | No-Go |",
    ]
    coverage = report.get("coverage", [])
    if coverage:
        lines.extend(
            [
                "",
                "## Scenario Coverage",
                "",
                "| Scenario | Execution mode | Rows | Steps | Ranks | Rank 1 |",
                "|---|---|---:|---:|---|---|",
            ]
        )
        lines.extend(
            "| {scenario} | {execution_mode} | {row_count} | {step_count} | "
            "{ranks} | {rank1_status} |".format(
                **{
                    **row,
                    "ranks": ",".join(str(rank) for rank in row["ranks"]),
                },
            )
            for row in coverage
        )
    runtime_contracts = report.get("runtime_contracts", [])
    if runtime_contracts:
        support = runtime_contracts[0]["support_evidence"]
        lines.extend(
            [
                "",
                "## Exact Support Evidence",
                "",
                f"- Collector entry: `{support['collector_entry']}`",
                f"- Attention source: `{support['attention_source']}`",
                "- Actual operations: "
                + ", ".join(f"`{name}`" for name in support["actual_operations"]),
                f"- Model config SHA256: `{support['model_config_sha256']}`",
            ]
        )
    if report.get("block_reason"):
        lines.extend(["", "## Block Reason", "", str(report["block_reason"])])
    lines.extend(
        [
            "",
            "The Phase466 execution-manifest SHA is retained only as historical evidence. "
            "Phase469 must attest the flat model identity separately before model load.",
            "",
            "`diagnostic_only=true`, `valid_for_default=false`, `perf_database=false`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_report(out_dir: Path, report: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase468_runtime_compatibility.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(out_dir / "phase468_fpm_compatibility.md", report)


def run(
    out_dir: Path,
    source_lock_path: Path | None = None,
    descriptor_csv_out: Path | None = None,
) -> dict:
    if source_lock_path is None:
        source_manifest = expected_source_lock_manifest()
    else:
        try:
            source_manifest = json.loads(
                source_lock_path.read_text(encoding="utf-8")
            )
        except OSError as error:
            raise SourceContractError(
                f"source lock is missing:{source_lock_path}"
            ) from error
        except json.JSONDecodeError as error:
            raise SourceContractError(
                f"source lock is invalid JSON:{source_lock_path}"
            ) from error
    validate_source_lock_manifest(source_manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase468_pinned_source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    numeric, descriptors, coverage, runtime_contracts = collect()
    domains, topology_domains = build_workload_domains(descriptors)
    descriptor_manifest = build_descriptor_manifest(descriptors)
    descriptor_manifest["topology_query_summaries"] = topology_domains
    (out_dir / "phase468_descriptor_manifest.json").write_text(
        json.dumps(descriptor_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if descriptor_csv_out is not None:
        write_full_descriptor_csv(descriptor_csv_out, descriptors)
    _write_csv(out_dir / "phase468_workload_domain.csv", domains)
    _write_csv(out_dir / "phase468_numeric_invariance.csv", numeric)
    coverage_report = {
        "schema": "phase468_rank_step_coverage_v1",
        "scenarios": coverage,
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
        "default_aic": "No-Go",
    }
    (out_dir / "phase468_rank_step_coverage.json").write_text(
        json.dumps(coverage_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        **_base_report("READY_FOR_RUNTIME_PREFLIGHT"),
        "source_contract": "PASS",
        "scenario_count": len(numeric),
        "descriptor_row_count": len(descriptors),
        "descriptor_unique_query_point_count": descriptor_manifest[
            "unique_query_point_count"
        ],
        "descriptor_streaming_csv_sha256": descriptor_manifest[
            "streaming_csv_sha256"
        ],
        "descriptor_csv_materialized": descriptor_csv_out is not None,
        "topology_query_summaries": topology_domains,
        "runtime_contracts": runtime_contracts,
        "coverage": coverage,
        "max_numeric_delta": max(row["absolute_delta"] for row in numeric),
    }
    _write_report(out_dir, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--source-lock", type=Path, default=None)
    parser.add_argument("--descriptor-csv-out", type=Path, default=None)
    args = parser.parse_args()
    try:
        report = run(args.out_dir, args.source_lock, args.descriptor_csv_out)
    except SourceContractError as error:
        report = {
            **_base_report("BLOCKED_SOURCE_CONTRACT"),
            "source_contract": "BLOCKED",
            "block_reason": str(error),
        }
        _write_report(args.out_dir, report)
    except RuntimeContractError as error:
        report = {
            **_base_report("BLOCKED_RUNTIME_CONTRACT"),
            "source_contract": "PASS",
            "block_reason": str(error),
        }
        _write_report(args.out_dir, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "READY_FOR_RUNTIME_PREFLIGHT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
