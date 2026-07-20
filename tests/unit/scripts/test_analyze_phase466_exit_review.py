from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase466_exit_review.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_phase466_exit_review", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _real_row(rank: int) -> dict[str, object]:
    return {
        "run_id": "real-test",
        "source": "real",
        "rank_id": rank,
        "rank_scope": "dp_rank",
        "workload_cohort_digest": "a" * 64,
        "iteration_seq": 1,
        "iteration_start_offset_ms": 0.0,
        "iteration_end_offset_ms": 10.0 + rank,
        "cumulative_scheduled_tokens": 64004,
        "progress_start_tokens": 0,
        "progress_end_tokens": 64004,
        "progress_window_id": 0,
        "iteration_elapsed_ms": 10.0 + rank,
        "scheduled_prefill_tokens": 64000,
        "scheduled_decode_tokens": 4,
        "prefill_request_count": 2,
        "decode_request_count": 4,
        "running_count": 6,
        "waiting_count": 122,
        "completed_request_count": 0,
    }


def _sim_row(rank: int = 0) -> dict[str, object]:
    return {
        "run_id": "sim-test",
        "source": "sim",
        "rank_id": rank,
        "rank_scope": "dp_rank",
        "workload_cohort_digest": "a" * 64,
        "iteration_seq": 1,
        "iteration_start_offset_ms": 0.0,
        "iteration_end_offset_ms": 8.0,
        "cumulative_scheduled_tokens": 64004,
        "progress_start_tokens": 0,
        "progress_end_tokens": 64004,
        "progress_window_id": 0,
        "iteration_elapsed_ms": 8.0,
        "scheduled_prefill_tokens": 64000,
        "scheduled_decode_tokens": 4,
        "prefill_request_count": 2,
        "decode_request_count": 4,
        "running_count": 6,
        "waiting_count": 122,
        "completed_request_count": 0,
        "sim_predicted_iteration_ms": 8.0,
        "sim_component_cost_ms": 8.0,
    }


def test_evidence_coverage_keeps_stock_probe_missing_fields_explicit() -> None:
    analysis = _load_module()

    coverage = analysis.evaluate_evidence_coverage(
        [_real_row(0), _real_row(1)], [_sim_row(0), _sim_row(1)]
    )

    assert coverage["joined_progress_windows"] == 2
    assert coverage["real_rank_ids"] == [0, 1]
    assert coverage["candidate_coverage"]["dp_rank_synchronization_asymmetry"]["status"] == "EVALUABLE"
    schedule = coverage["candidate_coverage"]["schedule_merged_batch_composition"]
    assert schedule["status"] == "INCONCLUSIVE_MISSING_FIELDS"
    assert "prefill_chunk_token_histogram" in schedule["missing_fields"]
    cost = coverage["candidate_coverage"]["iteration_cost_serving_state_coverage"]
    assert "sim_serving_state_key" in cost["missing_fields"]


def test_route_selection_requires_exactly_one_human_reviewed_pass() -> None:
    analysis = _load_module()
    candidates = list(analysis.CANDIDATES)

    none = analysis.select_route({candidate: "INCONCLUSIVE" for candidate in candidates})
    assert none == {"status": "INCONCLUSIVE", "selected_route": None, "pass_count": 0}

    one = {candidate: "DISPROVED" for candidate in candidates}
    one["dp_rank_synchronization_asymmetry"] = "PASS"
    assert analysis.select_route(one) == {
        "status": "SELECTED",
        "selected_route": "dp_rank_synchronization_asymmetry",
        "pass_count": 1,
    }

    two = dict(one)
    two["iteration_cost_serving_state_coverage"] = "PASS"
    assert analysis.select_route(two) == {
        "status": "INCONCLUSIVE",
        "selected_route": None,
        "pass_count": 2,
    }

    unresolved = {candidate: "INCONCLUSIVE" for candidate in candidates}
    unresolved["dp_rank_synchronization_asymmetry"] = "PASS"
    assert analysis.select_route(unresolved)["status"] == "INCONCLUSIVE"


def test_scenario_routes_may_differ_without_shared_route() -> None:
    analysis = _load_module()
    judgements = {}
    coverage = {}
    for index, scenario in enumerate(analysis.FORMAL_SCENARIOS):
        selected = analysis.CANDIDATES[index]
        judgements[scenario] = {
            candidate: "PASS" if candidate == selected else "DISPROVED"
            for candidate in analysis.CANDIDATES
        }
        coverage[scenario] = {
            "candidate_coverage": {
                candidate: {"status": "EVALUABLE"}
                for candidate in analysis.CANDIDATES
            }
        }

    result = analysis.select_scenario_routes(judgements, coverage=coverage)

    assert result["status"] == "SCENARIO_ROUTES_SELECTED"
    assert result["shared_route"] is None
    assert {
        scenario: item["selected_route"]
        for scenario, item in result["scenario_routes"].items()
    } == {
        scenario: analysis.CANDIDATES[index]
        for index, scenario in enumerate(analysis.FORMAL_SCENARIOS)
    }


def test_old_judgement_schema_fails_closed(tmp_path: Path) -> None:
    analysis = _load_module()
    path = tmp_path / "judgements.json"
    path.write_text(json.dumps({candidate: "INCONCLUSIVE" for candidate in analysis.CANDIDATES}))

    with pytest.raises(ValueError, match="exit_review_judgement_schema_mismatch"):
        analysis.load_scenario_judgements(path)


def test_evidence_coverage_rejects_wrong_scope_or_unjoinable_digest() -> None:
    analysis = _load_module()
    wrong_scope = _real_row(0)
    wrong_scope["rank_scope"] = "global_simulator"
    with pytest.raises(ValueError, match="real_rank_scope_mismatch"):
        analysis.evaluate_evidence_coverage([wrong_scope], [_sim_row()])

    sim = _sim_row()
    sim["workload_cohort_digest"] = "b" * 64
    with pytest.raises(ValueError, match="workload_cohort_digest_mismatch"):
        analysis.evaluate_evidence_coverage([_real_row(0)], [sim])


def test_rank_window_coverage_requires_every_dp_rank() -> None:
    analysis = _load_module()
    rank1_sim = _sim_row(1)
    rank1_sim["progress_window_id"] = 1
    rank1_sim["cumulative_scheduled_tokens"] = 65537
    rank1_sim["progress_end_tokens"] = 65537
    rank1_sim["scheduled_prefill_tokens"] = 65533

    coverage = analysis.evaluate_evidence_coverage(
        [_real_row(0), _real_row(1)],
        [_sim_row(0), rank1_sim],
        expected_dp=2,
    )

    assert coverage["missing_joined_rank_ids"] == [1]
    assert all(
        item["status"] == "INCONCLUSIVE_MISSING_RANK_WINDOWS"
        for item in coverage["candidate_coverage"].values()
    )


def test_alignment_retains_zero_token_iteration_wall_time() -> None:
    analysis = _load_module()
    zero_real = _real_row(0)
    zero_real.update(
        {
            "iteration_seq": 1,
            "iteration_start_offset_ms": 0.0,
            "iteration_end_offset_ms": 7.0,
            "iteration_elapsed_ms": 7.0,
            "scheduled_prefill_tokens": 0,
            "scheduled_decode_tokens": 0,
            "prefill_request_count": 0,
            "decode_request_count": 0,
            "progress_end_tokens": 0,
            "cumulative_scheduled_tokens": 0,
        }
    )
    real = _real_row(0)
    real.update(
        {
            "iteration_seq": 2,
            "iteration_start_offset_ms": 7.0,
            "iteration_end_offset_ms": 17.0,
            "iteration_elapsed_ms": 10.0,
        }
    )
    sim = _sim_row(0)

    coverage = analysis.evaluate_evidence_coverage([zero_real, real], [sim])

    assert coverage["joined_progress_windows"] == 1


def test_candidate_fields_must_exist_on_the_required_source() -> None:
    analysis = _load_module()
    real = _real_row(0)
    real["completed_request_count"] = ""
    sim = _sim_row()
    sim.update(
        {
            "prefill_chunk_token_histogram": "64000",
            "fresh_prefill_tokens": 64000,
            "recompute_prefill_tokens": 0,
            "resume_prefill_tokens": 0,
            "decode_kv_token_sum": 4,
            "cudagraph_mode": "full",
        }
    )

    coverage = analysis.evaluate_evidence_coverage([real], [sim])

    schedule = coverage["candidate_coverage"]["schedule_merged_batch_composition"]
    assert "prefill_chunk_token_histogram" in schedule["missing_real_fields"]
    dp = coverage["candidate_coverage"]["dp_rank_synchronization_asymmetry"]
    assert "completed_request_count" in dp["missing_real_fields"]
    assert dp["status"] == "INCONCLUSIVE_MISSING_FIELDS"


def test_route_selection_rejects_pass_without_evaluable_coverage() -> None:
    analysis = _load_module()
    judgements = {candidate: "DISPROVED" for candidate in analysis.CANDIDATES}
    judgements["schedule_merged_batch_composition"] = "PASS"
    coverage = {
        "candidate_coverage": {
            candidate: {
                "status": (
                    "INCONCLUSIVE_MISSING_FIELDS"
                    if candidate == "schedule_merged_batch_composition"
                    else "EVALUABLE"
                )
            }
            for candidate in analysis.CANDIDATES
        }
    }

    with pytest.raises(ValueError, match="pass_without_evaluable_coverage"):
        analysis.select_route(judgements, coverage=coverage)


def test_route_selection_rejects_disproved_without_evaluable_coverage() -> None:
    analysis = _load_module()
    selected = "dp_rank_synchronization_asymmetry"
    judgements = {
        candidate: "PASS" if candidate == selected else "DISPROVED"
        for candidate in analysis.CANDIDATES
    }
    coverage = {
        "candidate_coverage": {
            candidate: {
                "status": (
                    "EVALUABLE"
                    if candidate == selected
                    else "INCONCLUSIVE_MISSING_FIELDS"
                )
            }
            for candidate in analysis.CANDIDATES
        }
    }

    with pytest.raises(ValueError, match="disproved_without_evaluable_coverage"):
        analysis.select_route(judgements, coverage=coverage)


def test_real_artifact_root_requires_passed_gate_and_exact_three_scenarios(
    tmp_path: Path,
) -> None:
    analysis = _load_module()
    (tmp_path / "overhead").mkdir()
    (tmp_path / "phase466_result.json").write_text(
        json.dumps(
            {
                "status": "DIAGNOSTIC_COMPLETE",
                "gate_status": "PASS",
                "formal_scenarios": list(analysis.FORMAL_SCENARIOS),
                "execution_manifest_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "overhead" / "overhead_gate.json").write_text(
        json.dumps({"status": "PASS", "pair_count": 6}),
        encoding="utf-8",
    )
    (tmp_path / "expected_execution_manifest.json").write_text(
        json.dumps({"schema": "phase466_execution_manifest_v1"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="execution_manifest_digest_mismatch"):
        analysis.validate_real_artifact_root(tmp_path)


def test_real_artifact_root_rejects_incomplete_formal_set(tmp_path: Path) -> None:
    analysis = _load_module()
    manifest = {"schema": "phase466_execution_manifest_v1"}
    digest = analysis.execution_manifest_digest(manifest)
    gate = {
        "status": "PASS",
        "pair_count": 6,
        "pairs": [{"pair_id": f"pair-{index:02d}"} for index in range(1, 7)],
    }
    gate_digest = analysis.execution_manifest_digest(gate)
    (tmp_path / "overhead").mkdir()
    (tmp_path / "phase466_result.json").write_text(
        json.dumps(
            {
                "status": "DIAGNOSTIC_COMPLETE",
                "gate_status": "PASS",
                "formal_scenarios": [analysis.FORMAL_SCENARIOS[0]],
                "execution_manifest_sha256": digest,
                "overhead_gate_sha256": gate_digest,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "overhead" / "overhead_gate.json").write_text(
        json.dumps(gate), encoding="utf-8"
    )
    (tmp_path / "expected_execution_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="formal_scenario_set_mismatch"):
        analysis.validate_real_artifact_root(tmp_path)


def _write_iteration_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _execution_manifest_v3(analysis) -> dict[str, object]:
    tool_hashes = {
        "contract": "1" * 64,
        "supervisor": "2" * 64,
        "benchmark": "3" * 64,
        "rank_analyzer": "4" * 64,
    }
    coordinator = {
        "schema": "phase466_coordinator_manifest_v1",
        "contract_schema": "phase466_rank_timing_v3",
        "source_commit": "a" * 40,
        "tools": {
            name: {"path": f"scripts/{name}.py", "sha256": digest}
            for name, digest in tool_hashes.items()
        },
    }
    coordinator_digest = analysis.execution_manifest_digest(coordinator)
    model_payload = {
        "schema": "phase466_flat_model_fingerprint_v1",
        "identity_scope": "same_flat_mirror_instance",
        "official_immutable_revision": False,
        "metadata_files": [
            {"path": ".msc", "sha256": "5" * 64},
            {"path": ".mv", "sha256": "6" * 64},
        ],
        "runtime_files": [
            {"path": "model.safetensors.index.json", "sha256": "7" * 64}
        ],
        "shards": [
            {
                "path": "model-00001-of-00001.safetensors",
                "size": 10,
                "mtime_ns": 1,
                "header_sha256": "8" * 64,
            }
        ],
    }
    model_identity = dict(model_payload)
    model_identity["fingerprint_sha256"] = analysis.execution_manifest_digest(
        model_payload
    )
    attestation = {
        "schema": "phase466_worker_attestation_v1",
        "contract_schema": "phase466_rank_timing_v3",
        "coordinator_manifest_sha256": coordinator_digest,
        "tool_sha256": tool_hashes,
        "vllm_version": "0.19.0",
        "vllm_source_sha256": {},
        "vllm_import_paths": [],
        "gpu_identity": {"rows": ["NVIDIA H200, 143771, 575.57.08"]},
        "model_identity": model_identity,
        "prompt_cohort_sha256": {
            f"formal-{scenario}": {
                "warmup": "8" * 64,
                "measurement": "9" * 64,
            }
            for scenario in analysis.FORMAL_SCENARIOS
        },
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
    }
    attestation_digest = analysis.execution_manifest_digest(attestation)
    return {
        "schema": "phase466_execution_manifest_v3",
        "contract_schema": "phase466_rank_timing_v3",
        "coordinator_manifest_sha256": coordinator_digest,
        "worker_attestation_sha256": attestation_digest,
        "coordinator_manifest": coordinator,
        "worker_attestation": attestation,
    }


def test_real_artifact_root_rejects_iteration_csv_tampering(tmp_path: Path) -> None:
    analysis = _load_module()
    manifest = _execution_manifest_v3(analysis)
    attestation = manifest["worker_attestation"]
    model_identity = attestation["model_identity"]
    tool_hashes = attestation["tool_sha256"]
    digest = analysis.execution_manifest_digest(manifest)
    gate = {
        "status": "PASS",
        "pair_count": 6,
        "pairs": [{"pair_id": f"pair-{index:02d}"} for index in range(1, 7)],
    }
    gate_digest = analysis.execution_manifest_digest(gate)
    (tmp_path / "overhead").mkdir()
    (tmp_path / "overhead" / "overhead_gate.json").write_text(json.dumps(gate))
    (tmp_path / "expected_execution_manifest.json").write_text(json.dumps(manifest))
    identities = {}
    for scenario in analysis.FORMAL_SCENARIOS:
        run_dir = tmp_path / "formal" / scenario
        csv_path = run_dir / "iteration_rows.csv"
        _write_iteration_csv(csv_path, _real_row(0))
        identity = analysis.iteration_csv_identity(csv_path)
        identities[scenario] = identity
        (run_dir / "probe_summary.json").write_text(
            json.dumps({"status": "ARTIFACT_VALID", "iteration_rows_identity": identity})
        )
        (run_dir / "meta.json").write_text(
            json.dumps(
                {
                    "id": f"formal-{scenario}",
                    "execution_manifest_sha256": digest,
                    "overhead_gate_sha256": gate_digest,
                    "execution_tool_sha256": tool_hashes,
                    "model_identity_schema": model_identity["schema"],
                    "model_identity_sha256": model_identity["fingerprint_sha256"],
                    "warmup_prompt_cohort_sha256": "8" * 64,
                    "prompt_cohort_sha256": "9" * 64,
                }
            )
        )
        (run_dir / "tooling.sha256").write_text(
            "".join(
                f"{value}  {key}\n"
                for key, value in tool_hashes.items()
            )
        )
        (run_dir / "prompt_identity.json").write_text(
            json.dumps({"warmup": "8" * 64, "measurement": "9" * 64})
        )
        (run_dir / "bench_result.json").write_text(
            json.dumps({"prompt_cohort_sha256": "9" * 64})
        )
        (run_dir / "warmup").mkdir()
        (run_dir / "warmup" / "bench_result.json").write_text(
            json.dumps({"prompt_cohort_sha256": "8" * 64})
        )
        (run_dir / "gpu_compute_apps_after.txt").write_text("")
        (run_dir / "process_residue_after.txt").write_text("")
    rank_timing_report = {
        "schema": "phase466_rank_timing_report_v1",
        "status": "DIAGNOSTIC_COMPLETE",
        "route_selection_executed": False,
        "simulator_rank_rows_generated": False,
    }
    rank_timing_path = tmp_path / "rank_timing_report.json"
    rank_timing_path.write_text(json.dumps(rank_timing_report))
    rank_timing_digest = analysis.hashlib.sha256(rank_timing_path.read_bytes()).hexdigest()
    (tmp_path / "phase466_result.json").write_text(
        json.dumps(
            {
                "status": "DIAGNOSTIC_COMPLETE",
                "gate_status": "PASS",
                "formal_scenarios": list(analysis.FORMAL_SCENARIOS),
                "execution_manifest_sha256": digest,
                "overhead_gate_sha256": gate_digest,
                "formal_artifacts": identities,
                "rank_timing_report": "rank_timing_report.json",
                "rank_timing_report_sha256": rank_timing_digest,
                "route_selection_executed": False,
                "simulator_rank_rows_generated": False,
            }
        )
    )

    analysis.validate_real_artifact_root(tmp_path)
    result_path = tmp_path / "phase466_result.json"
    result = json.loads(result_path.read_text())
    result["route_selection_executed"] = True
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="phase466_route_selection_must_not_run"):
        analysis.validate_real_artifact_root(tmp_path)
    result["route_selection_executed"] = False
    result_path.write_text(json.dumps(result))

    rank_timing_path.write_text(json.dumps({**rank_timing_report, "status": "tampered"}))
    with pytest.raises(ValueError, match="rank_timing_report_digest_mismatch"):
        analysis.validate_real_artifact_root(tmp_path)
    rank_timing_path.write_text(json.dumps(rank_timing_report))

    target = tmp_path / "formal" / analysis.FORMAL_SCENARIOS[0] / "iteration_rows.csv"
    target.write_text(target.read_text() + "\n")

    with pytest.raises(ValueError, match="formal_iteration_identity_mismatch"):
        analysis.validate_real_artifact_root(tmp_path)


def test_sim_artifact_root_rejects_iteration_csv_tampering(tmp_path: Path) -> None:
    analysis = _load_module()
    scenarios = []
    for scenario in analysis.FORMAL_SCENARIOS:
        path = tmp_path / scenario / "iteration_rows.csv"
        _write_iteration_csv(path, _sim_row())
        scenarios.append(
            {
                "scenario": scenario,
                "iteration_rows_identity": analysis.iteration_csv_identity(path),
            }
        )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "source": "sim",
                "rank_scope": "dp_rank",
                "backend": "vllm",
                "system": "h200_sxm",
                "database_version": "0.19.0",
                "scenarios": scenarios,
            }
        )
    )

    analysis.validate_sim_artifact_root(tmp_path)
    target = tmp_path / analysis.FORMAL_SCENARIOS[-1] / "iteration_rows.csv"
    target.write_text(target.read_text().replace("sim-test", "sim-other"))

    with pytest.raises(ValueError, match="simulator_iteration_identity_mismatch"):
        analysis.validate_sim_artifact_root(tmp_path)
