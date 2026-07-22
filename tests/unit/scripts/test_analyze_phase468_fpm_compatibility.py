"""Tests for the Phase468 FPM workload/runtime compatibility audit."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "analyze_phase468_fpm_compatibility.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "analyze_phase468_fpm_compatibility_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _runtime_contract(module, **overrides):
    contract = {
        "model_path": module.EXPECTED_MODEL_PATH,
        "model_config_sha256": module.EXPECTED_MODEL_CONFIG_SHA256,
        "architecture": module.EXPECTED_ARCHITECTURE,
        "support_level": "exact",
        "hardware": module.EXPECTED_HARDWARE,
        "backend": module.EXPECTED_BACKEND,
        "backend_version": module.EXPECTED_BACKEND_VERSION,
        "gpu_count": 8,
        "tp": 8,
        "pp": 1,
        "dp": 1,
        "moe_tp": 1,
        "moe_ep": 8,
        "cp": 1,
        "topology": "tp8pp1dp1moetp1ep8cp1",
        "quant_runtime": module.EXPECTED_QUANT_RUNTIME,
        "engine_loop_enabled": False,
    }
    contract.update(overrides)
    return contract


def test_model_config_sha_uses_pinned_canonical_json_semantics() -> None:
    module = _load()
    config_path = (
        module.REPO_ROOT
        / "src"
        / "aiconfigurator"
        / "model_configs"
        / "moonshotai--Kimi-K2.5_config.json"
    )
    raw = config_path.read_bytes()
    canonical = json.dumps(
        json.loads(raw),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    assert module.EXPECTED_MODEL_CONFIG_SHA256 == hashlib.sha256(canonical).hexdigest()
    assert module.EXPECTED_MODEL_CONFIG_SHA256 != hashlib.sha256(raw).hexdigest()


def test_source_lock_requires_exact_entry_set_and_hashes() -> None:
    module = _load()
    manifest = module.expected_source_lock_manifest()

    assert module.validate_source_lock_manifest(manifest) == manifest

    missing = deepcopy(manifest)
    missing["sources"].pop()
    with pytest.raises(module.SourceContractError, match="source set"):
        module.validate_source_lock_manifest(missing)

    wrong_hash = deepcopy(manifest)
    wrong_hash["sources"][0]["sha256"] = "0" * 64
    with pytest.raises(module.SourceContractError, match="source hash"):
        module.validate_source_lock_manifest(wrong_hash)


def test_source_lock_rejects_borrowed_field_drift() -> None:
    module = _load()
    manifest = module.expected_source_lock_manifest()
    manifest["sources"][-1]["borrowed_fields"] = ["num_prefill_requests"]

    with pytest.raises(module.SourceContractError, match="borrowed fields"):
        module.validate_source_lock_manifest(manifest)


def test_source_lock_requires_kimi_exact_support_sources() -> None:
    module = _load()
    manifest = module.expected_source_lock_manifest()
    paths = {row["path"] for row in manifest["sources"]}

    assert "collector/fpm_forward/model_capability.py" in paths
    assert "collector/model_cases.py" in paths
    assert (
        "collector/cases/models/KimiK25ForConditionalGeneration_cases.yaml"
        in paths
    )
    assert manifest["exact_support"] == {
        "architecture": "KimiK25ForConditionalGeneration",
        "model_path": "moonshotai/Kimi-K2.5",
        "attention_source": "mla_module",
        "selected_ops": ["mla_context_module", "mla_generation_module"],
        "backend": "vllm",
        "quant_mode": "int4_wo",
    }

    missing = deepcopy(manifest)
    missing["sources"] = [
        row for row in missing["sources"] if not row["path"].endswith("_cases.yaml")
    ]
    with pytest.raises(module.SourceContractError, match="source set"):
        module.validate_source_lock_manifest(missing)

    wrong_case = deepcopy(manifest)
    wrong_case["exact_support"]["quant_mode"] = "fp8_block"
    with pytest.raises(module.SourceContractError, match="exact support"):
        module.validate_source_lock_manifest(wrong_case)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"support_level": "family_template"}, "support_level"),
        ({"support_level": "bootstrap_template"}, "support_level"),
        ({"model_path": "moonshotai/Kimi-K2-Instruct"}, "model path"),
        ({"model_config_sha256": "0" * 64}, "model config"),
        ({"quant_runtime": ""}, "quant runtime"),
        ({"topology": "tp2pp1dp4moetp1ep8cp1"}, "topology"),
        ({"pp": None}, "topology dimensions"),
        ({"pp": 2}, "PP"),
        ({"cp": 2}, "CP"),
        ({"gpu_count": 4}, "GPU count"),
        (
            {
                "moe_tp": 2,
                "topology": "tp8pp1dp1moetp2ep8cp1",
            },
            "MoE world",
        ),
        ({"backend_version": "0.18.0"}, "backend version"),
        ({"engine_loop_enabled": True}, "engine loop"),
    ],
)
def test_runtime_contract_blocks_nonexact_or_wrong_identity(override, message) -> None:
    module = _load()

    with pytest.raises(module.RuntimeContractError, match=message):
        module.validate_runtime_contract(_runtime_contract(module, **override))


def test_runtime_contract_accepts_only_the_two_phase468_topologies() -> None:
    module = _load()

    tp = module.validate_runtime_contract(_runtime_contract(module))
    dp = module.validate_runtime_contract(
        _runtime_contract(
            module,
            tp=4,
            dp=2,
            topology="tp4pp1dp2moetp1ep8cp1",
        )
    )

    assert tp["status"] == "PASS"
    assert dp["status"] == "PASS"


def _workload_row(**overrides) -> dict:
    row = {
        "scenario": "scenario-a",
        "active": True,
        "topology": "tp8pp1dp1moetp1ep8cp1",
        "num_prefill_requests": 0,
        "sum_prefill_tokens": 0,
        "sum_prefill_kv_tokens": 0,
        "num_decode_requests": 0,
        "sum_decode_kv_tokens": 0,
    }
    row.update(overrides)
    return row


def test_workload_domains_split_prefill_and_decode_without_zero_padding() -> None:
    module = _load()
    rows = [
        _workload_row(
            num_prefill_requests=2,
            sum_prefill_tokens=128,
            sum_prefill_kv_tokens=64,
        ),
        _workload_row(
            scenario="scenario-b",
            num_decode_requests=4,
            sum_decode_kv_tokens=512,
        ),
        _workload_row(
            scenario="scenario-c",
            num_prefill_requests=1,
            sum_prefill_tokens=32,
            sum_prefill_kv_tokens=96,
            num_decode_requests=3,
            sum_decode_kv_tokens=300,
        ),
    ]

    domains, topology_rows = module.build_workload_domains(rows)

    prefill = next(row for row in domains if row["phase"] == "prefill")
    decode = next(row for row in domains if row["phase"] == "decode")
    assert prefill["row_count"] == 2
    assert prefill["num_requests_min"] == 1
    assert prefill["sum_tokens_min"] == 32
    assert prefill["sum_kv_tokens_min"] == 64
    assert decode["row_count"] == 2
    assert decode["num_requests_min"] == 3
    assert decode["sum_tokens_min"] is None
    assert decode["sum_kv_tokens_min"] == 300
    assert topology_rows[0]["mixed_row_count"] == 1
    assert topology_rows[0]["unique_query_point_count"] == 4


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_workload_domains_reject_missing_phase_domain(phase) -> None:
    module = _load()
    if phase == "prefill":
        rows = [_workload_row(num_decode_requests=1, sum_decode_kv_tokens=16)]
    else:
        rows = [
            _workload_row(
                num_prefill_requests=1,
                sum_prefill_tokens=16,
                sum_prefill_kv_tokens=0,
            )
        ]

    with pytest.raises(module.RuntimeContractError, match=f"{phase} domain"):
        module.build_workload_domains(rows)


def test_descriptor_manifest_streams_csv_digest_without_default_large_file(
    tmp_path,
) -> None:
    module = _load()
    rows = [
        _workload_row(
            num_prefill_requests=1,
            sum_prefill_tokens=16,
            sum_prefill_kv_tokens=0,
        ),
        _workload_row(num_decode_requests=1, sum_decode_kv_tokens=16),
    ]

    manifest = module.build_descriptor_manifest(rows)

    assert manifest["row_count"] == 2
    assert manifest["scenario_row_counts"] == {"scenario-a": 2}
    assert len(manifest["streaming_csv_sha256"]) == 64
    assert not (tmp_path / "phase468_forward_workload_descriptors.csv").exists()


def test_run_default_does_not_materialize_full_descriptor_csv(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load()
    descriptors = [
        _workload_row(
            num_prefill_requests=1,
            sum_prefill_tokens=16,
            sum_prefill_kv_tokens=0,
        ),
        _workload_row(num_decode_requests=1, sum_decode_kv_tokens=16),
    ]
    monkeypatch.setattr(
        module,
        "collect",
        lambda: (
            [{"scenario": "scenario-a", "absolute_delta": 0.0}],
            descriptors,
            [],
            [],
        ),
    )
    out_dir = tmp_path / "out"

    report = module.run(out_dir)

    assert report["descriptor_csv_materialized"] is False
    assert (out_dir / "phase468_descriptor_manifest.json").is_file()
    assert not (out_dir / "phase468_forward_workload_descriptors.csv").exists()


def test_full_descriptor_csv_requires_explicit_path_outside_repository(tmp_path) -> None:
    module = _load()
    rows = [_workload_row(num_decode_requests=1, sum_decode_kv_tokens=16)]

    with pytest.raises(module.RuntimeContractError, match="outside repository"):
        module.write_full_descriptor_csv(
            module.REPO_ROOT / "phase468_forward_workload_descriptors.csv",
            rows,
        )

    output = tmp_path / "phase468_forward_workload_descriptors.csv"
    module.write_full_descriptor_csv(output, rows)
    assert output.is_file()


def test_lockstep_coverage_requires_two_rows_per_step() -> None:
    module = _load()
    rows = [
        {"engine_step_id": 1, "dp_rank": 0, "active": True},
        {"engine_step_id": 1, "dp_rank": 1, "active": False},
        {"engine_step_id": 2, "dp_rank": 0, "active": True},
    ]

    with pytest.raises(module.RuntimeContractError, match="lockstep rank coverage"):
        module.validate_descriptor_coverage("dp_lockstep", rows)


def test_legacy_representative_does_not_invent_rank_one() -> None:
    module = _load()
    rows = [
        {"engine_step_id": 1, "dp_rank": 0, "active": True},
        {"engine_step_id": 2, "dp_rank": 0, "active": True},
    ]

    coverage = module.validate_descriptor_coverage(
        "dp_legacy_representative",
        rows,
    )

    assert coverage["ranks"] == [0]
    assert coverage["rank1_status"] == "not_executed_by_official_legacy_path"


def test_blocked_source_contract_writes_artifact_and_returns_nonzero(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load()
    bad_manifest = module.expected_source_lock_manifest()
    bad_manifest["sources"][0]["sha256"] = "0" * 64
    source_lock = tmp_path / "source-lock.json"
    source_lock.write_text(json.dumps(bad_manifest), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--out-dir",
            str(tmp_path / "out"),
            "--source-lock",
            str(source_lock),
        ],
    )

    assert module.main() == 1
    report = json.loads(
        (tmp_path / "out" / "phase468_runtime_compatibility.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "BLOCKED_SOURCE_CONTRACT"
    assert report["default_aic"] == "No-Go"


def test_markdown_preserves_scenario_and_exact_support_boundaries(tmp_path) -> None:
    module = _load()
    report = {
        **module._base_report("READY_FOR_RUNTIME_PREFLIGHT"),
        "source_contract": "PASS",
        "scenario_count": 6,
        "descriptor_row_count": 2,
        "max_numeric_delta": 0.0,
        "coverage": [
            {
                "scenario": "dp-bt65536",
                "execution_mode": "dp_legacy_representative",
                "row_count": 2,
                "step_count": 2,
                "ranks": [0],
                "rank1_status": "not_executed_by_official_legacy_path",
            }
        ],
        "runtime_contracts": [
            {
                "support_evidence": {
                    "collector_entry": "build_collection_plan(has_model_cases=True)",
                    "attention_source": "mla_module",
                    "actual_operations": ["ContextMLA", "GenerationMLA"],
                    "model_config_sha256": module.EXPECTED_MODEL_CONFIG_SHA256,
                }
            }
        ],
    }
    output = tmp_path / "report.md"

    module._write_markdown(output, report)
    text = output.read_text(encoding="utf-8")

    assert "not_executed_by_official_legacy_path" in text
    assert "build_collection_plan(has_model_cases=True)" in text


@pytest.mark.parametrize("payload", [None, "not-json"])
def test_missing_or_invalid_source_lock_is_blocked(tmp_path, monkeypatch, payload) -> None:
    module = _load()
    source_lock = tmp_path / "source-lock.json"
    if payload is not None:
        source_lock.write_text(payload, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--out-dir",
            str(tmp_path / "out"),
            "--source-lock",
            str(source_lock),
        ],
    )

    assert module.main() == 1
    report = json.loads(
        (tmp_path / "out" / "phase468_runtime_compatibility.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "BLOCKED_SOURCE_CONTRACT"
