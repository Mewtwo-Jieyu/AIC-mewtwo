#!/usr/bin/env python3
"""Validate Phase466 evidence and apply scenario-local exit-review semantics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


CANDIDATES = (
    "schedule_merged_batch_composition",
    "iteration_cost_serving_state_coverage",
    "dp_rank_synchronization_asymmetry",
)
FORMAL_SCENARIOS = (
    "K2.5-tp4ep8dp2-32k3k",
    "K2.5-tp8ep8-8k2k-bt65536",
    "K2.5-tp4ep8dp2-8k2k-bt65536",
)
SCENARIO_DP = {
    "K2.5-tp4ep8dp2-32k3k": 2,
    "K2.5-tp8ep8-8k2k-bt65536": 1,
    "K2.5-tp4ep8dp2-8k2k-bt65536": 2,
}
BASE_FIELDS = {
    "run_id",
    "source",
    "rank_id",
    "rank_scope",
    "workload_cohort_digest",
    "iteration_seq",
    "iteration_start_offset_ms",
    "iteration_end_offset_ms",
    "cumulative_scheduled_tokens",
    "progress_start_tokens",
    "progress_end_tokens",
    "scheduled_prefill_tokens",
    "scheduled_decode_tokens",
    "prefill_request_count",
    "decode_request_count",
    "progress_window_id",
    "iteration_elapsed_ms",
}
SCHEDULE_FIELDS = {
    "prefill_request_count",
    "decode_request_count",
    "prefill_chunk_token_histogram",
    "fresh_prefill_tokens",
    "recompute_prefill_tokens",
    "resume_prefill_tokens",
    "decode_kv_token_sum",
    "running_count",
    "waiting_count",
    "cudagraph_mode",
}
STATE_FIELDS = {
    "scheduled_prefill_tokens",
    "scheduled_decode_tokens",
    "running_count",
    "waiting_count",
    "completed_request_count",
}
REAL_CANDIDATE_FIELDS = {
    "schedule_merged_batch_composition": BASE_FIELDS | SCHEDULE_FIELDS,
    "iteration_cost_serving_state_coverage": BASE_FIELDS
    | {
        "scheduled_prefill_tokens",
        "scheduled_decode_tokens",
        "prefill_chunk_token_histogram",
        "fresh_prefill_tokens",
        "recompute_prefill_tokens",
        "resume_prefill_tokens",
        "decode_kv_token_sum",
        "cudagraph_mode",
    },
    "dp_rank_synchronization_asymmetry": BASE_FIELDS | STATE_FIELDS,
}
SIM_CANDIDATE_FIELDS = {
    "schedule_merged_batch_composition": BASE_FIELDS | SCHEDULE_FIELDS,
    "iteration_cost_serving_state_coverage": BASE_FIELDS
    | {
        "scheduled_prefill_tokens",
        "scheduled_decode_tokens",
        "prefill_chunk_token_histogram",
        "fresh_prefill_tokens",
        "recompute_prefill_tokens",
        "resume_prefill_tokens",
        "decode_kv_token_sum",
        "cudagraph_mode",
        "sim_serving_state_key",
        "sim_predicted_iteration_ms",
        "sim_component_cost_ms",
    },
    "dp_rank_synchronization_asymmetry": BASE_FIELDS | STATE_FIELDS,
}
VALID_JUDGEMENTS = {"PASS", "DISPROVED", "INCONCLUSIVE"}
EXIT_REVIEW_SCHEMA = "phase466_exit_review_v2"


def execution_manifest_digest(manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def iteration_csv_identity(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty_iteration_csv:{path}")
    run_ids = {row.get("run_id", "") for row in rows}
    sources = {row.get("source", "") for row in rows}
    digests = {row.get("workload_cohort_digest", "") for row in rows}
    if len(run_ids) != 1 or "" in run_ids:
        raise ValueError(f"iteration_csv_run_id_mismatch:{path}")
    if len(sources) != 1 or "" in sources:
        raise ValueError(f"iteration_csv_source_mismatch:{path}")
    if len(digests) != 1 or not re.fullmatch(r"[0-9a-f]{64}", next(iter(digests))):
        raise ValueError(f"iteration_csv_workload_digest_mismatch:{path}")
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(int(row["rank_id"]), []).append(row)
    rank_bounds: dict[str, dict[str, Any]] = {}
    for rank, rank_rows in sorted(grouped.items()):
        ordered = sorted(rank_rows, key=lambda row: int(row["iteration_seq"]))
        first = ordered[0]
        last = ordered[-1]
        rank_bounds[str(rank)] = {
            "row_count": len(ordered),
            "first_iteration_seq": int(first["iteration_seq"]),
            "last_iteration_seq": int(last["iteration_seq"]),
            "first_progress_start_tokens": int(first["progress_start_tokens"]),
            "last_progress_end_tokens": int(last["progress_end_tokens"]),
            "first_start_offset_ms": float(first["iteration_start_offset_ms"]),
            "last_end_offset_ms": float(last["iteration_end_offset_ms"]),
        }
    return {
        "sha256": hashlib.sha256(content).hexdigest(),
        "row_count": len(rows),
        "run_id": next(iter(run_ids)),
        "source": next(iter(sources)),
        "workload_cohort_digest": next(iter(digests)),
        "rank_bounds": rank_bounds,
    }


def _load_hash_file(path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if separator != "  " or not re.fullmatch(r"[0-9a-f]{64}", digest) or not name:
            raise ValueError(f"invalid_hash_file:{path}")
        if name in hashes:
            raise ValueError(f"duplicate_hash_entry:{path}:{name}")
        hashes[name] = digest
    if not hashes:
        raise ValueError(f"empty_hash_file:{path}")
    return hashes


def _present_fields(rows: list[dict[str, Any]]) -> set[str]:
    if not rows:
        return set()
    keys = set.intersection(*(set(row) for row in rows))
    return {
        key
        for key in keys
        if all(
            row[key] is not None
            and (not isinstance(row[key], str) or bool(row[key].strip()))
            for row in rows
        )
    }


def _validate_alignment_rows(rows: list[dict[str, Any]], *, source: str) -> None:
    seen: set[tuple[int, int]] = set()
    previous: dict[int, tuple[int, float, int]] = {}
    for row in rows:
        missing = BASE_FIELDS - set(row)
        if missing:
            raise ValueError(f"missing_alignment_fields:{source}:{sorted(missing)}")
        if row.get("source") != source:
            raise ValueError(f"{source}_source_mismatch")
        if not str(row.get("run_id", "")).strip():
            raise ValueError(f"missing_run_id:{source}")
        if not re.fullmatch(
            r"[0-9a-f]{64}", str(row.get("workload_cohort_digest", ""))
        ):
            raise ValueError(f"invalid_workload_cohort_digest:{source}")
        rank = int(row["rank_id"])
        iteration = int(row["iteration_seq"])
        start_ms = float(row["iteration_start_offset_ms"])
        end_ms = float(row["iteration_end_offset_ms"])
        elapsed_ms = float(row["iteration_elapsed_ms"])
        progress_start = int(row["progress_start_tokens"])
        progress_end = int(row["progress_end_tokens"])
        cumulative = int(row["cumulative_scheduled_tokens"])
        scheduled = int(row["scheduled_prefill_tokens"]) + int(
            row["scheduled_decode_tokens"]
        )
        prefill_requests = int(row["prefill_request_count"])
        decode_requests = int(row["decode_request_count"])
        key = (rank, iteration)
        if key in seen:
            raise ValueError(f"duplicate_rank_iteration:{key}")
        if (
            rank < 0
            or iteration < 0
            or not all(math.isfinite(value) for value in (start_ms, end_ms, elapsed_ms))
            or elapsed_ms <= 0
            or scheduled < 0
            or prefill_requests < 0
            or decode_requests < 0
        ):
            raise ValueError(f"invalid_alignment_value:{key}")
        if scheduled == 0 and (prefill_requests != 0 or decode_requests != 0):
            raise ValueError(f"zero_token_request_mismatch:{key}")
        if abs((end_ms - start_ms) - elapsed_ms) > 1e-6:
            raise ValueError(f"iteration_offset_mismatch:{key}")
        if progress_end - progress_start != scheduled or cumulative != progress_end:
            raise ValueError(f"progress_token_mismatch:{key}")
        expected_window = max(0, (cumulative - 1) // 65536)
        if int(row["progress_window_id"]) != expected_window:
            raise ValueError(f"progress_window_mismatch:{key}")
        if rank in previous:
            previous_iteration, previous_end_ms, previous_tokens = previous[rank]
            if iteration <= previous_iteration:
                raise ValueError(f"nonmonotonic_iteration:{key}")
            if abs(start_ms - previous_end_ms) > 1e-6:
                raise ValueError(f"noncontiguous_rank_clock:{key}")
            if progress_start != previous_tokens:
                raise ValueError(f"noncontiguous_rank_progress:{key}")
        elif abs(start_ms) > 1e-6 or progress_start != 0:
            raise ValueError(f"rank_alignment_must_start_at_zero:{key}")
        seen.add(key)
        previous[rank] = (iteration, end_ms, progress_end)


def evaluate_evidence_coverage(
    real_rows: list[dict[str, Any]],
    sim_rows: list[dict[str, Any]],
    *,
    expected_dp: int | None = None,
) -> dict[str, Any]:
    if not real_rows or not sim_rows:
        raise ValueError("missing_real_or_sim_rows")
    _validate_alignment_rows(real_rows, source="real")
    _validate_alignment_rows(sim_rows, source="sim")
    if any(row.get("source") != "real" for row in real_rows):
        raise ValueError("real_source_mismatch")
    if any(row.get("rank_scope") != "dp_rank" for row in real_rows):
        raise ValueError("real_rank_scope_mismatch")
    if any(row.get("source") != "sim" for row in sim_rows):
        raise ValueError("sim_source_mismatch")
    if any(row.get("rank_scope") != "dp_rank" for row in sim_rows):
        raise ValueError("sim_rank_scope_mismatch")

    real_digests = {str(row.get("workload_cohort_digest", "")) for row in real_rows}
    sim_digests = {str(row.get("workload_cohort_digest", "")) for row in sim_rows}
    if len(real_digests) != 1 or real_digests != sim_digests:
        raise ValueError("workload_cohort_digest_mismatch")
    real_rank_ids = sorted({int(row["rank_id"]) for row in real_rows})
    sim_rank_ids = sorted({int(row["rank_id"]) for row in sim_rows})
    if real_rank_ids != sim_rank_ids:
        raise ValueError("real_sim_rank_set_mismatch")
    if expected_dp is not None and real_rank_ids != list(range(expected_dp)):
        raise ValueError("scenario_rank_set_mismatch")
    real_windows = {
        (int(row["rank_id"]), int(row["progress_window_id"]))
        for row in real_rows
    }
    sim_windows = {
        (int(row["rank_id"]), int(row["progress_window_id"]))
        for row in sim_rows
    }
    joined_windows = real_windows & sim_windows
    if not joined_windows:
        raise ValueError("missing_joined_progress_window")
    joined_rank_ids = {rank for rank, unused_window in joined_windows}
    missing_joined_rank_ids = sorted(set(real_rank_ids) - joined_rank_ids)

    real_fields = _present_fields(real_rows)
    sim_fields = _present_fields(sim_rows)
    candidate_coverage: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATES:
        missing_real = sorted(REAL_CANDIDATE_FIELDS[candidate] - real_fields)
        missing_sim = sorted(SIM_CANDIDATE_FIELDS[candidate] - sim_fields)
        missing = sorted(set(missing_real) | set(missing_sim))
        if missing_joined_rank_ids:
            status = "INCONCLUSIVE_MISSING_RANK_WINDOWS"
        elif missing:
            status = "INCONCLUSIVE_MISSING_FIELDS"
        else:
            status = "EVALUABLE"
        candidate_coverage[candidate] = {
            "status": status,
            "missing_fields": missing,
            "missing_real_fields": missing_real,
            "missing_sim_fields": missing_sim,
        }
    return {
        "workload_cohort_digest": next(iter(real_digests)),
        "joined_progress_windows": len(joined_windows),
        "missing_joined_rank_ids": missing_joined_rank_ids,
        "real_rank_ids": real_rank_ids,
        "real_rank_scope": "dp_rank",
        "sim_rank_scope": "dp_rank",
        "candidate_coverage": candidate_coverage,
    }


def select_route(
    judgements: dict[str, str],
    *,
    coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if set(judgements) != set(CANDIDATES):
        raise ValueError("candidate_judgement_set_mismatch")
    invalid = {value for value in judgements.values() if value not in VALID_JUDGEMENTS}
    if invalid:
        raise ValueError(f"invalid_candidate_judgement:{sorted(invalid)}")
    if coverage is not None:
        candidate_coverage = coverage.get("candidate_coverage")
        if not isinstance(candidate_coverage, dict) or set(candidate_coverage) != set(
            CANDIDATES
        ):
            raise ValueError("coverage_candidate_set_mismatch")
        for candidate, judgement in judgements.items():
            if candidate_coverage[candidate].get("status") == "EVALUABLE":
                continue
            if judgement == "PASS":
                raise ValueError(f"pass_without_evaluable_coverage:{candidate}")
            if judgement == "DISPROVED":
                raise ValueError(
                    f"disproved_without_evaluable_coverage:{candidate}"
                )
    passed = [candidate for candidate in CANDIDATES if judgements[candidate] == "PASS"]
    disproved = [
        candidate for candidate in CANDIDATES if judgements[candidate] == "DISPROVED"
    ]
    if len(passed) == 1 and len(disproved) == len(CANDIDATES) - 1:
        return {
            "status": "SELECTED",
            "selected_route": passed[0],
            "pass_count": 1,
        }
    return {
        "status": "INCONCLUSIVE",
        "selected_route": None,
        "pass_count": len(passed),
    }


def select_scenario_routes(
    judgements: dict[str, dict[str, str]],
    *,
    coverage: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if set(judgements) != set(FORMAL_SCENARIOS):
        raise ValueError("judgement_scenario_set_mismatch")
    if set(coverage) != set(FORMAL_SCENARIOS):
        raise ValueError("coverage_scenario_set_mismatch")
    scenario_routes = {
        scenario: select_route(
            judgements[scenario], coverage=coverage[scenario]
        )
        for scenario in FORMAL_SCENARIOS
    }
    selected = [
        result["selected_route"]
        for result in scenario_routes.values()
        if result["status"] == "SELECTED"
    ]
    shared_route = (
        selected[0]
        if len(selected) == len(FORMAL_SCENARIOS) and len(set(selected)) == 1
        else None
    )
    return {
        "status": (
            "SCENARIO_ROUTES_SELECTED"
            if len(selected) == len(FORMAL_SCENARIOS)
            else "INCONCLUSIVE"
        ),
        "scenario_routes": scenario_routes,
        "shared_route": shared_route,
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def load_scenario_judgements(path: Path) -> dict[str, dict[str, str]]:
    raw = _load_json(path)
    if raw.get("schema") != EXIT_REVIEW_SCHEMA:
        raise ValueError("exit_review_judgement_schema_mismatch")
    scenario_judgements = raw.get("scenario_judgements")
    if not isinstance(scenario_judgements, dict):
        raise ValueError("scenario_judgements_must_be_object")
    result: dict[str, dict[str, str]] = {}
    for scenario, candidate_judgements in scenario_judgements.items():
        if not isinstance(candidate_judgements, dict):
            raise ValueError(f"candidate_judgements_must_be_object:{scenario}")
        result[str(scenario)] = {
            str(candidate): str(judgement)
            for candidate, judgement in candidate_judgements.items()
        }
    return result


def validate_real_artifact_root(root: Path) -> dict[str, Path]:
    result = _load_json(root / "phase466_result.json")
    gate = _load_json(root / "overhead" / "overhead_gate.json")
    manifest = _load_json(root / "expected_execution_manifest.json")
    digest = execution_manifest_digest(manifest)
    if result.get("execution_manifest_sha256") != digest:
        raise ValueError("execution_manifest_digest_mismatch")
    if (
        result.get("status") != "DIAGNOSTIC_COMPLETE"
        or result.get("gate_status") != "PASS"
    ):
        raise ValueError("phase466_supervisor_not_passed")
    if set(result.get("formal_scenarios", [])) != set(FORMAL_SCENARIOS):
        raise ValueError("formal_scenario_set_mismatch")
    if result.get("route_selection_executed") is not False:
        raise ValueError("phase466_route_selection_must_not_run")
    if result.get("simulator_rank_rows_generated") is not False:
        raise ValueError("phase466_simulator_rank_rows_must_not_exist")
    if result.get("rank_timing_report") != "rank_timing_report.json":
        raise ValueError("rank_timing_report_missing")
    rank_timing_path = root / "rank_timing_report.json"
    if not rank_timing_path.is_file():
        raise ValueError("rank_timing_report_missing")
    if result.get("rank_timing_report_sha256") != hashlib.sha256(
        rank_timing_path.read_bytes()
    ).hexdigest():
        raise ValueError("rank_timing_report_digest_mismatch")
    rank_timing = _load_json(rank_timing_path)
    if (
        rank_timing.get("schema") != "phase466_rank_timing_report_v1"
        or rank_timing.get("status") != "DIAGNOSTIC_COMPLETE"
        or rank_timing.get("route_selection_executed") is not False
        or rank_timing.get("simulator_rank_rows_generated") is not False
    ):
        raise ValueError("rank_timing_report_contract_mismatch")
    if gate.get("status") != "PASS" or gate.get("pair_count") != 6:
        raise ValueError("phase466_overhead_gate_not_passed")
    pairs = gate.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 6:
        raise ValueError("phase466_overhead_gate_pairs_missing")
    gate_digest = execution_manifest_digest(gate)
    if result.get("overhead_gate_sha256") != gate_digest:
        raise ValueError("overhead_gate_digest_mismatch")
    if manifest.get("schema") != "phase466_execution_manifest_v4":
        raise ValueError("execution_manifest_schema_mismatch")
    coordinator = manifest.get("coordinator_manifest")
    attestation = manifest.get("worker_attestation")
    if not isinstance(coordinator, dict) or not isinstance(attestation, dict):
        raise ValueError("execution_manifest_identity_layers_missing")
    coordinator_digest = execution_manifest_digest(coordinator)
    attestation_digest = execution_manifest_digest(attestation)
    if (
        coordinator.get("schema") != "phase466_coordinator_manifest_v2"
        or manifest.get("coordinator_manifest_sha256") != coordinator_digest
    ):
        raise ValueError("coordinator_manifest_digest_mismatch")
    if (
        attestation.get("schema") != "phase466_worker_attestation_v2"
        or manifest.get("worker_attestation_sha256") != attestation_digest
        or attestation.get("coordinator_manifest_sha256") != coordinator_digest
    ):
        raise ValueError("worker_attestation_digest_mismatch")
    expected_tools = attestation.get("tool_sha256")
    if not isinstance(expected_tools, dict) or set(expected_tools) != {
        "contract",
        "supervisor",
        "benchmark",
        "rank_analyzer",
        "rank_logging_handler",
        "rank_logging_config",
    }:
        raise ValueError("execution_manifest_tool_hashes_missing")
    transport = attestation.get("rank_logging_transport")
    if (
        not isinstance(transport, dict)
        or transport.get("schema") != "phase466_rank_logging_transport_v1"
        or transport.get("handler_module") != "phase466_rank_local_logging"
        or transport.get("rank_log_dir_env") != "PHASE466_RANK_LOG_DIR"
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(attestation.get("rank_local_canary_prompt_cohort_sha256", "")),
        )
    ):
        raise ValueError("rank_logging_transport_identity_missing")
    coordinator_tools = coordinator.get("tools")
    if not isinstance(coordinator_tools, dict) or {
        name: item.get("sha256") if isinstance(item, dict) else None
        for name, item in coordinator_tools.items()
    } != expected_tools:
        raise ValueError("coordinator_worker_tool_hash_mismatch")
    model_identity = attestation.get("model_identity")
    if not isinstance(model_identity, dict):
        raise ValueError("execution_manifest_model_identity_missing")
    model_schema = model_identity.get("schema")
    if model_schema == "phase466_flat_model_fingerprint_v1":
        model_digest = str(model_identity.get("fingerprint_sha256", ""))
        model_payload = dict(model_identity)
        model_payload.pop("fingerprint_sha256", None)
        metadata = model_identity.get("metadata_files")
        runtime_files = model_identity.get("runtime_files")
        shards = model_identity.get("shards")
        if (
            not re.fullmatch(r"[0-9a-f]{64}", model_digest)
            or execution_manifest_digest(model_payload) != model_digest
            or model_identity.get("identity_scope") != "same_flat_mirror_instance"
            or model_identity.get("official_immutable_revision") is not False
            or not isinstance(metadata, list)
            or [item.get("path") for item in metadata if isinstance(item, dict)]
            != [".msc", ".mv"]
            or not isinstance(runtime_files, list)
            or not runtime_files
            or "model.safetensors.index.json"
            not in {
                item.get("path")
                for item in runtime_files
                if isinstance(item, dict)
            }
            or not isinstance(shards, list)
            or not shards
        ):
            raise ValueError("execution_manifest_model_identity_missing")
    elif model_schema == "phase466_snapshot_model_identity_v1":
        revision = str(model_identity.get("revision", ""))
        model_files = model_identity.get("files_sha256")
        if (
            not re.fullmatch(r"[0-9a-f]{40,64}", revision)
            or not isinstance(model_files, dict)
            or any(
                not re.fullmatch(r"[0-9a-f]{64}", str(value))
                for value in model_files.values()
            )
        ):
            raise ValueError("execution_manifest_model_identity_missing")
        model_digest = execution_manifest_digest(model_identity)
    else:
        raise ValueError("execution_manifest_model_identity_missing")
    manifest_prompts = attestation.get("prompt_cohort_sha256")
    if not isinstance(manifest_prompts, dict):
        raise ValueError("execution_manifest_prompt_identity_missing")
    formal_artifacts = result.get("formal_artifacts")
    if not isinstance(formal_artifacts, dict) or set(formal_artifacts) != set(
        FORMAL_SCENARIOS
    ):
        raise ValueError("formal_artifact_identity_set_mismatch")
    paths: dict[str, Path] = {}
    for scenario in FORMAL_SCENARIOS:
        run_dir = root / "formal" / scenario
        summary = _load_json(run_dir / "probe_summary.json")
        meta = _load_json(run_dir / "meta.json")
        if summary.get("status") != "ARTIFACT_VALID":
            raise ValueError(f"formal_probe_not_passed:{scenario}")
        if meta.get("execution_manifest_sha256") != digest:
            raise ValueError(f"formal_execution_manifest_mismatch:{scenario}")
        if meta.get("overhead_gate_sha256") != gate_digest:
            raise ValueError(f"formal_overhead_gate_mismatch:{scenario}")
        if meta.get("execution_tool_sha256") != expected_tools:
            raise ValueError(f"formal_execution_tool_mismatch:{scenario}")
        if (
            meta.get("model_identity_schema") != model_schema
            or meta.get("model_identity_sha256") != model_digest
        ):
            raise ValueError(f"formal_model_identity_mismatch:{scenario}")
        prompt_identity = _load_json(run_dir / "prompt_identity.json")
        expected_prompt_identity = manifest_prompts.get(meta.get("id"))
        if prompt_identity != expected_prompt_identity:
            raise ValueError(f"formal_manifest_prompt_identity_mismatch:{scenario}")
        if (
            prompt_identity.get("measurement")
            != meta.get("prompt_cohort_sha256")
            or prompt_identity.get("warmup")
            != meta.get("warmup_prompt_cohort_sha256")
        ):
            raise ValueError(f"formal_prompt_identity_mismatch:{scenario}")
        benchmark = _load_json(run_dir / "bench_result.json")
        if benchmark.get("prompt_cohort_sha256") != prompt_identity.get(
            "measurement"
        ):
            raise ValueError(f"formal_prompt_digest_mismatch:{scenario}")
        warmup_benchmark = _load_json(run_dir / "warmup" / "bench_result.json")
        if warmup_benchmark.get("prompt_cohort_sha256") != prompt_identity.get(
            "warmup"
        ):
            raise ValueError(f"formal_warmup_prompt_digest_mismatch:{scenario}")
        if _load_hash_file(run_dir / "tooling.sha256") != expected_tools:
            raise ValueError(f"formal_tooling_file_mismatch:{scenario}")
        for name in ("gpu_compute_apps_after.txt", "process_residue_after.txt"):
            if (run_dir / name).read_text(encoding="utf-8").strip():
                raise ValueError(f"formal_residue:{scenario}:{name}")
        path = run_dir / "iteration_rows.csv"
        if not path.is_file():
            raise ValueError(f"missing_formal_iteration_rows:{scenario}")
        actual_identity = iteration_csv_identity(path)
        if (
            summary.get("iteration_rows_identity") != actual_identity
            or formal_artifacts.get(scenario) != actual_identity
        ):
            raise ValueError(f"formal_iteration_identity_mismatch:{scenario}")
        paths[scenario] = path
    return paths


def validate_sim_artifact_root(root: Path) -> dict[str, Path]:
    manifest = _load_json(root / "manifest.json")
    if (
        manifest.get("source") != "sim"
        or manifest.get("rank_scope") != "dp_rank"
        or manifest.get("backend") != "vllm"
        or manifest.get("system") != "h200_sxm"
        or manifest.get("database_version") != "0.19.0"
    ):
        raise ValueError("simulator_manifest_contract_mismatch")
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("simulator_manifest_scenarios_missing")
    names = [str(item.get("scenario")) for item in scenarios if isinstance(item, dict)]
    if set(names) != set(FORMAL_SCENARIOS) or len(names) != len(FORMAL_SCENARIOS):
        raise ValueError("simulator_scenario_set_mismatch")
    paths: dict[str, Path] = {}
    entries = {
        str(item["scenario"]): item
        for item in scenarios
        if isinstance(item, dict) and "scenario" in item
    }
    if len(entries) != len(FORMAL_SCENARIOS):
        raise ValueError("simulator_scenario_identity_mismatch")
    for scenario in FORMAL_SCENARIOS:
        path = root / scenario / "iteration_rows.csv"
        if not path.is_file():
            raise ValueError(f"missing_simulator_iteration_rows:{scenario}")
        if entries[scenario].get("iteration_rows_identity") != iteration_csv_identity(
            path
        ):
            raise ValueError(f"simulator_iteration_identity_mismatch:{scenario}")
        paths[scenario] = path
    return paths


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--judgements-json", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    real_paths_by_scenario = validate_real_artifact_root(args.real_root)
    sim_paths_by_scenario = validate_sim_artifact_root(args.sim_root)
    real_by_scenario = {
        scenario: _read_csv(path)
        for scenario, path in real_paths_by_scenario.items()
    }
    sim_by_scenario = {
        scenario: _read_csv(path)
        for scenario, path in sim_paths_by_scenario.items()
    }
    if set(real_by_scenario) != set(FORMAL_SCENARIOS):
        raise ValueError("real_formal_scenario_set_mismatch")
    if set(sim_by_scenario) != set(FORMAL_SCENARIOS):
        raise ValueError("real_sim_scenario_set_mismatch")
    coverage = {
        scenario: evaluate_evidence_coverage(
            real_by_scenario[scenario],
            sim_by_scenario[scenario],
            expected_dp=SCENARIO_DP[scenario],
        )
        for scenario in FORMAL_SCENARIOS
    }
    judgements = {
        scenario: {candidate: "INCONCLUSIVE" for candidate in CANDIDATES}
        for scenario in FORMAL_SCENARIOS
    }
    if args.judgements_json is not None:
        judgements = load_scenario_judgements(args.judgements_json)
    result = {
        "schema": EXIT_REVIEW_SCHEMA,
        "coverage": coverage,
        "human_reviewed_judgements": judgements,
        "route_selection": select_scenario_routes(judgements, coverage=coverage),
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
        "default_readiness": "No-Go",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
