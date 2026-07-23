"""Tests for the Phase469A exact-site FPM support contract."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "analyze_phase469_fpm_exact_site_support.py"
TP8 = "tp8pp1dp1moetp1ep8cp1"
DP2 = "tp4pp1dp2moetp1ep8cp1"


def _load():
    assert SCRIPT.is_file(), "Phase469A analyzer is missing"
    spec = importlib.util.spec_from_file_location(
        "analyze_phase469_fpm_exact_site_support_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _query(
    module,
    *,
    phase: str = "prefill",
    topology: str = TP8,
    batch_size: int = 2,
    kv_tokens: int = 64,
    prefill_tokens: int = 128,
):
    return module.ExactSiteQuery(
        phase=phase,
        topology=topology,
        batch_size=batch_size,
        total_kv_read_tokens=kv_tokens,
        total_prefill_tokens=prefill_tokens if phase == "prefill" else 0,
    )


def _native_point(query, *, latency_ms: float = 1.0) -> dict:
    return {
        "batch_size": query.batch_size,
        "total_prefill_tokens": query.total_prefill_tokens,
        "total_kv_read_tokens": query.total_kv_read_tokens,
        "latency_ms": latency_ms,
    }


def _runtime_identity(module, **overrides) -> dict:
    identity = {
        "model_id": module.phase468.EXPECTED_MODEL_PATH,
        "model_load_path": "/models/Kimi-K2.5",
        "model_config_sha256": module.phase468.EXPECTED_MODEL_CONFIG_SHA256,
        "checkpoint_fingerprint": "b" * 64,
        "hardware": module.phase468.EXPECTED_HARDWARE,
        "backend": module.phase468.EXPECTED_BACKEND,
        "backend_version": module.phase468.EXPECTED_BACKEND_VERSION,
        "expected_quant_runtime": module.phase468.EXPECTED_QUANT_RUNTIME,
        "actual_quant_runtime": module.phase468.EXPECTED_QUANT_RUNTIME,
        "execution_manifest_sha256": "c" * 64,
        "phase469b_preflight_manifest_sha256": "d" * 64,
        "phase469c_canary_manifest_sha256": "e" * 64,
    }
    identity.update(overrides)
    return identity


def _topology_dimensions(topology: str) -> dict:
    if topology == TP8:
        return {"tp": 8, "pp": 1, "dp": 1, "moe_tp": 1, "moe_ep": 8, "cp": 1}
    if topology == DP2:
        return {"tp": 4, "pp": 1, "dp": 2, "moe_tp": 1, "moe_ep": 8, "cp": 1}
    raise AssertionError(f"unsupported test topology:{topology}")


def _cell(
    module,
    *,
    workload_kind: str,
    topology: str = TP8,
    points: list[dict] | None = None,
    suffix: str = "a",
    **overrides,
) -> dict:
    if points is None:
        query = _query(
            module,
            phase=workload_kind,
            topology=topology,
            batch_size=1,
            kv_tokens=64,
            prefill_tokens=64,
        )
        points = [_native_point(query)]
    cell = {
        "cell_id": f"{topology}:{workload_kind}:{suffix}",
        "workload_kind": workload_kind,
        **_topology_dimensions(topology),
        "topology": topology,
        "runtime_run_id": f"run-{topology}-{workload_kind}-{suffix}",
        "runtime_grid_digest": hashlib.sha256(
            f"native:{topology}:{workload_kind}:{suffix}".encode("utf-8")
        ).hexdigest(),
        "coordinate_grid_sha256": "",
        "measurement_sha256": "",
        "points": points,
    }
    cell.update(overrides)
    if not cell["coordinate_grid_sha256"]:
        cell["coordinate_grid_sha256"] = module.compute_coordinate_grid_sha256(cell)
    if not cell["measurement_sha256"]:
        cell["measurement_sha256"] = module.compute_measurement_sha256(cell)
    return cell


def _grid(
    module,
    points: list[dict],
    *,
    workload_kind: str = "prefill",
    runtime_identity: dict | None = None,
) -> dict:
    other_kind = "decode" if workload_kind == "prefill" else "prefill"
    return {
        "schema": module.NATIVE_GRID_SCHEMA,
        "source_contract": deepcopy(module.EXPECTED_NATIVE_SOURCE_CONTRACT),
        "runtime_identity": runtime_identity or _runtime_identity(module),
        "cells": [
            _cell(
                module,
                workload_kind=workload_kind,
                points=points,
                suffix="selected",
            ),
            _cell(module, workload_kind=other_kind, suffix="required"),
        ],
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
    }


def _normalize(module, payload: dict, *, required_cell_set: str = "tp8_pilot"):
    return module.normalize_native_grid(
        payload,
        expected_runtime_identity=_runtime_identity(module),
        required_cell_set=required_cell_set,
    )


def _final_grid(module) -> dict:
    cells = []
    for topology in (TP8, DP2):
        for workload_kind in ("prefill", "decode"):
            cells.append(
                _cell(
                    module,
                    topology=topology,
                    workload_kind=workload_kind,
                    suffix="final",
                )
            )
    return {
        "schema": module.NATIVE_GRID_SCHEMA,
        "source_contract": deepcopy(module.EXPECTED_NATIVE_SOURCE_CONTRACT),
        "runtime_identity": _runtime_identity(module),
        "cells": cells,
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
    }


def test_exact_hit_is_selected_before_interpolation() -> None:
    module = _load()
    query = _query(module, prefill_tokens=128)
    lower = _query(module, prefill_tokens=64)
    upper = _query(module, prefill_tokens=192)
    index = _normalize(
        module,
        _grid(
            module,
            [_native_point(lower), _native_point(query), _native_point(upper)],
        )
    )

    judgement = module.judge_exact_site_query(query, index)

    assert judgement.outcome == "EXACT"
    assert judgement.lower_axis == 128
    assert judgement.upper_axis == 128


def test_same_site_bracket_is_allowed_without_computing_interpolation() -> None:
    module = _load()
    query = _query(module, prefill_tokens=128)
    lower = _query(module, prefill_tokens=64)
    upper = _query(module, prefill_tokens=192)
    index = _normalize(
        module,
        _grid(module, [_native_point(lower), _native_point(upper)]),
    )

    judgement = module.judge_exact_site_query(query, index)

    assert judgement.outcome == "SAME_SITE_INTERPOLATION"
    assert judgement.lower_axis == 64
    assert judgement.upper_axis == 192


def test_missing_site_fails_without_cross_site_transfer() -> None:
    module = _load()
    query = _query(module, topology=DP2)
    native = _query(module, topology=TP8)
    index = _normalize(module, _grid(module, [_native_point(native)]))

    judgement = module.judge_exact_site_query(query, index)

    assert judgement.outcome == "MISSING_SITE"
    assert judgement.lower_axis is None
    assert judgement.upper_axis is None


@pytest.mark.parametrize(
    ("prefill_tokens", "outcome"),
    [(32, "CURVE_UNDERFLOW"), (256, "CURVE_OVERFLOW")],
)
def test_curve_range_is_strict(prefill_tokens, outcome) -> None:
    module = _load()
    query = _query(module, prefill_tokens=prefill_tokens)
    lower = _query(module, prefill_tokens=64)
    upper = _query(module, prefill_tokens=192)
    index = _normalize(
        module,
        _grid(module, [_native_point(lower), _native_point(upper)]),
    )

    assert module.judge_exact_site_query(query, index).outcome == outcome


def test_singleton_site_allows_only_exact_hit() -> None:
    module = _load()
    native = _query(module, prefill_tokens=128)
    index = _normalize(module, _grid(module, [_native_point(native)]))

    exact = module.judge_exact_site_query(native, index)
    miss = module.judge_exact_site_query(
        _query(module, prefill_tokens=129),
        index,
    )

    assert exact.outcome == "EXACT"
    assert miss.outcome == "SINGLETON_MISS"


def test_candidate_grid_audit_visits_every_unique_query_and_use() -> None:
    module = _load()
    exact = _query(module, prefill_tokens=64)
    interpolated = _query(module, prefill_tokens=128)
    missing = _query(module, topology=DP2, prefill_tokens=128)
    underflow = _query(module, prefill_tokens=32)
    upper = _query(module, prefill_tokens=192)
    inventory = Counter(
        {
            exact: 2,
            interpolated: 3,
            missing: 5,
            underflow: 7,
        }
    )
    index = _normalize(
        module,
        _grid(module, [_native_point(exact), _native_point(upper)]),
    )

    audit = module.audit_query_inventory(inventory, index)

    assert audit == {
        "unique_query_count": 4,
        "query_use_count": 17,
        "queryable_unique_count": 2,
        "queryable_use_count": 5,
        "all_queries_queryable": False,
        "outcome_unique_counts": {
            "CURVE_UNDERFLOW": 1,
            "EXACT": 1,
            "MISSING_SITE": 1,
            "SAME_SITE_INTERPOLATION": 1,
        },
        "outcome_use_counts": {
            "CURVE_UNDERFLOW": 7,
            "EXACT": 2,
            "MISSING_SITE": 5,
            "SAME_SITE_INTERPOLATION": 3,
        },
    }


def test_duplicate_native_point_is_rejected() -> None:
    module = _load()
    query = _query(module)
    point = _native_point(query)

    with pytest.raises(module.NativeGridContractError, match="duplicate native point"):
        _normalize(module, _grid(module, [point, dict(point)]))


def test_decode_topology_isolation_is_exact() -> None:
    module = _load()
    query = _query(module, phase="decode", topology=DP2, batch_size=8, kv_tokens=9000)
    native = _query(module, phase="decode", topology=TP8, batch_size=8, kv_tokens=9000)
    index = _normalize(
        module,
        _grid(
            module,
            [_native_point(native)],
            workload_kind="decode",
        ),
    )

    assert module.judge_exact_site_query(query, index).outcome == "MISSING_SITE"


@pytest.mark.parametrize(
    ("workload_kind", "point_override"),
    [
        ("prefill", {"total_prefill_tokens": 0}),
        ("decode", {"total_prefill_tokens": 1}),
    ],
)
def test_prefill_and_decode_coordinates_cannot_be_mixed(
    workload_kind,
    point_override,
) -> None:
    module = _load()
    point = _native_point(_query(module, phase=workload_kind))
    point.update(point_override)

    with pytest.raises(module.NativeGridContractError, match="point coordinates"):
        _normalize(
            module,
            _grid(module, [point], workload_kind=workload_kind),
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("checkpoint_fingerprint", "d" * 64),
        ("model_config_sha256", "e" * 64),
        ("backend_version", "0.18.0"),
        ("expected_quant_runtime", "WrongQuantRuntime"),
        ("actual_quant_runtime", "WrongQuantRuntime"),
        ("phase469b_preflight_manifest_sha256", "f" * 64),
        ("phase469c_canary_manifest_sha256", "a" * 64),
    ],
)
def test_runtime_identity_must_match_expected_identity_exactly(field, bad_value) -> None:
    module = _load()
    payload = _grid(module, [_native_point(_query(module))])
    payload["runtime_identity"][field] = bad_value

    with pytest.raises(module.NativeGridContractError, match="runtime identity"):
        _normalize(module, payload)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("aic_collector_commit", "0" * 40),
        ("dynamo_commit", "1" * 40),
        ("native_schema_version", 4),
        ("coordinate_system", "iteration_averages_v1"),
        ("partition_policy", "nearest_site"),
        ("latency_aggregation", "mean_dp_rank"),
    ],
)
def test_source_contract_must_match_pinned_semantics(field, bad_value) -> None:
    module = _load()
    payload = _grid(module, [_native_point(_query(module))])
    payload["source_contract"][field] = bad_value

    with pytest.raises(module.NativeGridContractError, match="source contract"):
        _normalize(module, payload)


@pytest.mark.parametrize(
    "field",
    [
        "execution_manifest_sha256",
        "phase469b_preflight_manifest_sha256",
        "phase469c_canary_manifest_sha256",
    ],
)
def test_runtime_manifest_chain_is_required(field) -> None:
    module = _load()
    payload = _grid(module, [_native_point(_query(module))])
    del payload["runtime_identity"][field]

    with pytest.raises(module.NativeGridContractError, match="runtime identity fields"):
        _normalize(module, payload)


def test_deferred_actual_quant_runtime_is_rejected() -> None:
    module = _load()
    payload = _grid(module, [_native_point(_query(module))])
    payload["runtime_identity"]["actual_quant_runtime"] = (
        "DEFERRED_TO_MODEL_CANARY"
    )

    with pytest.raises(module.NativeGridContractError, match="actual_quant_runtime"):
        _normalize(module, payload)


@pytest.mark.parametrize(
    "field",
    [
        "runtime_run_id",
        "runtime_grid_digest",
        "coordinate_grid_sha256",
        "measurement_sha256",
    ],
)
def test_cell_run_identity_is_required(field) -> None:
    module = _load()
    payload = _grid(module, [_native_point(_query(module))])
    payload["cells"][0][field] = ""

    with pytest.raises(module.NativeGridContractError, match=field):
        _normalize(module, payload)


@pytest.mark.parametrize(
    "field",
    ["runtime_grid_digest", "coordinate_grid_sha256", "measurement_sha256"],
)
def test_digest_field_is_required(field) -> None:
    module = _load()
    payload = _grid(module, [_native_point(_query(module))])
    del payload["cells"][0][field]

    with pytest.raises(module.NativeGridContractError, match="cell fields"):
        _normalize(module, payload)


def test_native_grid_index_preserves_all_digest_types() -> None:
    module = _load()
    payload = _grid(module, [_native_point(_query(module))])

    index = _normalize(module, payload)

    assert set(index.runtime_grid_digests) == {
        cell["runtime_grid_digest"] for cell in payload["cells"]
    }
    assert set(index.coordinate_grid_sha256s) == {
        cell["coordinate_grid_sha256"] for cell in payload["cells"]
    }
    assert set(index.measurement_sha256s) == {
        cell["measurement_sha256"] for cell in payload["cells"]
    }


def test_duplicate_topology_phase_cell_is_rejected() -> None:
    module = _load()
    payload = _grid(module, [_native_point(_query(module))])
    duplicate = deepcopy(payload["cells"][0])
    duplicate["cell_id"] = "duplicate-cell"
    duplicate["runtime_run_id"] = "duplicate-run"
    duplicate["runtime_grid_digest"] = "f" * 64
    duplicate["points"][0]["total_prefill_tokens"] += 1
    duplicate["coordinate_grid_sha256"] = (
        module.compute_coordinate_grid_sha256(duplicate)
    )
    duplicate["measurement_sha256"] = module.compute_measurement_sha256(duplicate)
    payload["cells"].append(duplicate)

    with pytest.raises(module.NativeGridContractError, match="duplicate topology phase"):
        _normalize(module, payload)


def test_duplicate_cell_id_is_rejected() -> None:
    module = _load()
    payload = _grid(module, [_native_point(_query(module))])
    payload["cells"][1]["cell_id"] = payload["cells"][0]["cell_id"]

    with pytest.raises(module.NativeGridContractError, match="duplicate cell_id"):
        _normalize(module, payload)


def test_duplicate_runtime_run_identity_is_rejected() -> None:
    module = _load()
    payload = _grid(module, [_native_point(_query(module))])
    payload["cells"][1]["runtime_run_id"] = payload["cells"][0]["runtime_run_id"]

    with pytest.raises(module.NativeGridContractError, match="duplicate runtime_run_id"):
        _normalize(module, payload)


def test_opaque_runtime_grid_digest_is_only_format_checked() -> None:
    module = _load()
    payload = _grid(module, [_native_point(_query(module))])
    shared_digest = payload["cells"][0]["runtime_grid_digest"]
    payload["cells"][1]["runtime_grid_digest"] = shared_digest

    index = _normalize(module, payload)

    assert index.runtime_grid_digests == (shared_digest, shared_digest)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("runtime_grid_digest", "not-a-sha", "lowercase SHA256"),
        ("coordinate_grid_sha256", "f" * 64, "coordinate_grid_sha256 mismatch"),
        ("measurement_sha256", "f" * 64, "measurement_sha256 mismatch"),
    ],
)
def test_forged_digest_is_rejected(field, bad_value, message) -> None:
    module = _load()
    payload = _grid(module, [_native_point(_query(module))])
    payload["cells"][0][field] = bad_value

    with pytest.raises(module.NativeGridContractError, match=message):
        _normalize(module, payload)


def test_same_coordinates_different_latency_split_grid_and_measurement_digests() -> None:
    module = _load()
    first = _cell(module, workload_kind="prefill", suffix="first")
    second = deepcopy(first)
    second["points"][0]["latency_ms"] = 2.0

    assert module.compute_coordinate_grid_sha256(first) == (
        module.compute_coordinate_grid_sha256(second)
    )
    assert module.compute_measurement_sha256(first) != (
        module.compute_measurement_sha256(second)
    )


def test_native_runtime_digest_can_differ_for_same_normalized_coordinates() -> None:
    module = _load()
    first = _cell(
        module,
        workload_kind="prefill",
        suffix="first",
        runtime_grid_digest="a" * 64,
    )
    second = deepcopy(first)
    second["runtime_grid_digest"] = "b" * 64

    assert first["runtime_grid_digest"] != second["runtime_grid_digest"]
    assert module.compute_coordinate_grid_sha256(first) == (
        module.compute_coordinate_grid_sha256(second)
    )
    assert module.compute_measurement_sha256(first) == (
        module.compute_measurement_sha256(second)
    )


def test_point_order_does_not_change_coordinate_or_measurement_digest() -> None:
    module = _load()
    points = [
        _native_point(_query(module, prefill_tokens=64), latency_ms=1.0),
        _native_point(_query(module, prefill_tokens=128), latency_ms=2.0),
    ]
    first = _cell(module, workload_kind="prefill", points=points, suffix="first")
    second = deepcopy(first)
    second["points"].reverse()

    assert module.compute_coordinate_grid_sha256(first) == (
        module.compute_coordinate_grid_sha256(second)
    )
    assert module.compute_measurement_sha256(first) == (
        module.compute_measurement_sha256(second)
    )


def test_coordinate_change_updates_coordinate_and_measurement_digests() -> None:
    module = _load()
    first = _cell(module, workload_kind="prefill", suffix="first")
    second = deepcopy(first)
    second["points"][0]["total_prefill_tokens"] += 1

    assert module.compute_coordinate_grid_sha256(first) != (
        module.compute_coordinate_grid_sha256(second)
    )
    assert module.compute_measurement_sha256(first) != (
        module.compute_measurement_sha256(second)
    )


def test_latency_tamper_requires_measurement_digest_update() -> None:
    module = _load()
    payload = _grid(module, [_native_point(_query(module))])
    payload["cells"][0]["points"][0]["latency_ms"] = 2.0

    with pytest.raises(module.NativeGridContractError, match="measurement_sha256"):
        _normalize(module, payload)


def test_cell_moe_world_must_match_attention_world() -> None:
    module = _load()
    payload = _grid(module, [_native_point(_query(module))])
    payload["cells"][0]["moe_tp"] = 2

    with pytest.raises(module.NativeGridContractError, match="MoE world"):
        _normalize(module, payload)


def test_tp8_cell_cannot_claim_dp2_dimensions() -> None:
    module = _load()
    payload = _grid(module, [_native_point(_query(module))])
    payload["cells"][0].update(_topology_dimensions(DP2))

    with pytest.raises(module.NativeGridContractError, match="topology mismatch"):
        _normalize(module, payload)


def test_pilot_and_final_required_cell_sets_are_distinct() -> None:
    module = _load()
    pilot = _grid(module, [_native_point(_query(module))])

    pilot_index = _normalize(module, pilot, required_cell_set="tp8_pilot")
    assert set(pilot_index.cell_ids) == {
        f"{TP8}:prefill:selected",
        f"{TP8}:decode:required",
    }
    with pytest.raises(module.NativeGridContractError, match="required cell set"):
        _normalize(module, pilot, required_cell_set="final")

    final = _final_grid(module)
    final_index = _normalize(module, final, required_cell_set="final")
    assert len(final_index.cell_ids) == 4


def test_native_grid_schema_document_exposes_the_full_identity_contract() -> None:
    module = _load()

    document = module._native_grid_schema_document()

    assert document["schema"] == "phase469_fpm_native_grid_v2"
    assert document["source_contract"] == module.EXPECTED_NATIVE_SOURCE_CONTRACT
    assert set(document["runtime_identity_fields"]) == module.RUNTIME_IDENTITY_FIELDS
    assert set(document["cell_fields"]) == module.CELL_FIELDS
    assert set(document["point_fields"]) == module.POINT_FIELDS
    assert document["runtime_grid_digest"] == (
        "opaque Dynamo native digest extracted by the pinned native reader;"
        "Phase469A validates lowercase SHA256 only"
    )
    assert document["coordinate_grid_sha256"] == (
        "sha256(canonical topology, workload kind, and order-independent coordinates)"
    )
    assert document["measurement_sha256"] == (
        "sha256(canonical topology, workload kind, coordinates, and latency_ms)"
    )
    assert document["quant_runtime_contract"] == {
        "phase469b_expected": module.phase468.EXPECTED_QUANT_RUNTIME,
        "phase469b_actual": "DEFERRED_TO_MODEL_CANARY",
        "native_grid_expected": module.phase468.EXPECTED_QUANT_RUNTIME,
        "native_grid_actual": module.phase468.EXPECTED_QUANT_RUNTIME,
    }
    assert document["required_cell_sets"] == {
        "tp8_pilot": [
            {"topology": TP8, "workload_kind": "decode"},
            {"topology": TP8, "workload_kind": "prefill"},
        ],
        "final": [
            {"topology": DP2, "workload_kind": "decode"},
            {"topology": DP2, "workload_kind": "prefill"},
            {"topology": TP8, "workload_kind": "decode"},
            {"topology": TP8, "workload_kind": "prefill"},
        ],
    }


def _descriptor_row(**overrides) -> dict:
    row = {
        "scenario": "scenario-a",
        "engine_step_id": 1,
        "dp_rank": 0,
        "execution_mode": "tp_single_replica",
        "active": True,
        "num_prefill_requests": 1,
        "sum_prefill_tokens": 128,
        "sum_prefill_kv_tokens": 64,
        "num_decode_requests": 0,
        "sum_decode_kv_tokens": 0,
        "model_path": "moonshotai/Kimi-K2.5",
        "model_config_sha256": "a" * 64,
        "hardware": "h200_sxm",
        "backend": "vllm",
        "backend_version": "0.19.0",
        "tp": 8,
        "pp": 1,
        "dp": 1,
        "moe_tp": 1,
        "moe_ep": 8,
        "cp": 1,
        "topology": TP8,
        "quant_runtime": "CompressedTensorsWNA16MarlinMoEMethod",
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
    }
    row.update(overrides)
    return row


def _patch_expected_contract(module, monkeypatch, rows: list[dict]) -> None:
    inventory = module.build_query_inventory(rows)
    summaries, _, _ = module.summarize_query_inventory(inventory)
    manifest = module.phase468.build_descriptor_manifest(rows)
    expected_groups = {
        (row["topology"], row["phase"]): {
            "unique_query_count": row["unique_query_count"],
            "site_count": row["site_count"],
            "singleton_site_count": row["singleton_site_count"],
        }
        for row in summaries
    }
    monkeypatch.setattr(module, "EXPECTED_DESCRIPTOR_ROW_COUNT", len(rows))
    monkeypatch.setattr(
        module,
        "EXPECTED_UNIQUE_QUERY_COUNT",
        len(inventory),
    )
    monkeypatch.setattr(
        module,
        "EXPECTED_DESCRIPTOR_SHA256",
        manifest["streaming_csv_sha256"],
    )
    monkeypatch.setattr(module, "EXPECTED_GROUP_CONTRACT", expected_groups)


def test_phase468_descriptor_sha_drift_is_blocked(monkeypatch) -> None:
    module = _load()
    rows = [_descriptor_row()]
    _patch_expected_contract(module, monkeypatch, rows)
    monkeypatch.setattr(module, "EXPECTED_DESCRIPTOR_SHA256", "0" * 64)

    with pytest.raises(module.DescriptorContractError, match="descriptor SHA256"):
        module.validate_descriptor_contract(rows)


def test_phase468_source_failure_is_a_descriptor_contract_block(monkeypatch) -> None:
    module = _load()

    def fail_collect():
        raise module.phase468.SourceContractError("source lock schema mismatch")

    monkeypatch.setattr(module.phase468, "collect", fail_collect)

    with pytest.raises(
        module.DescriptorContractError,
        match="Phase468 compatibility contract blocked:source lock schema mismatch",
    ):
        module.collect_phase468_descriptor_rows()


def test_expected_query_structure_is_frozen() -> None:
    module = _load()

    assert module.EXPECTED_GROUP_CONTRACT == {
        (DP2, "prefill"): {
            "unique_query_count": 498,
            "site_count": 324,
            "singleton_site_count": 298,
        },
        (DP2, "decode"): {
            "unique_query_count": 57916,
            "site_count": 57,
            "singleton_site_count": 1,
        },
        (TP8, "prefill"): {
            "unique_query_count": 802,
            "site_count": 575,
            "singleton_site_count": 541,
        },
        (TP8, "decode"): {
            "unique_query_count": 58833,
            "site_count": 67,
            "singleton_site_count": 0,
        },
    }


def test_two_runs_are_byte_identical_without_native_grid(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load()
    rows = [
        _descriptor_row(),
        _descriptor_row(
            scenario="scenario-b",
            engine_step_id=2,
            num_prefill_requests=0,
            sum_prefill_tokens=0,
            sum_prefill_kv_tokens=0,
            num_decode_requests=2,
            sum_decode_kv_tokens=256,
        ),
    ]
    _patch_expected_contract(module, monkeypatch, rows)
    monkeypatch.setattr(module, "collect_phase468_descriptor_rows", lambda: rows)
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_report = module.run(first)
    second_report = module.run(second)

    assert first_report["status"] == "READY_FOR_NATIVE_GRID_INPUT"
    assert first_report["native_grid_coverage"] == "NOT_EVALUATED_NO_NATIVE_GRID"
    assert first_report["native_grid_source_contract"] == (
        module.EXPECTED_NATIVE_SOURCE_CONTRACT
    )
    assert first_report["runtime_identity_source"] == (
        "phase469b_preflight_manifest+phase469c_canary_manifest"
    )
    assert second_report == first_report
    first_files = {path.name: path.read_bytes() for path in first.iterdir()}
    second_files = {path.name: path.read_bytes() for path in second.iterdir()}
    assert first_files == second_files
    markdown = first_files["phase469_fpm_exact_site_support.md"].decode("utf-8")
    assert "Phase469C must attest the loaded quant runtime" in markdown
    assert "approximately 17 KiB" in markdown
    assert not any("descriptor" in name and name.endswith(".csv") for name in first_files)
    query_manifest = json.loads(first_files["phase469_query_manifest.json"])
    assert query_manifest["native_grid_source_contract"] == (
        module.EXPECTED_NATIVE_SOURCE_CONTRACT
    )
    assert set(query_manifest["runtime_identity_fields"]) == (
        module.RUNTIME_IDENTITY_FIELDS
    )


def test_blocked_descriptor_contract_writes_report_and_returns_nonzero(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load()
    rows = [_descriptor_row()]
    _patch_expected_contract(module, monkeypatch, rows)
    monkeypatch.setattr(module, "EXPECTED_DESCRIPTOR_SHA256", "0" * 64)
    monkeypatch.setattr(module, "collect_phase468_descriptor_rows", lambda: rows)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--out-dir", str(tmp_path / "out")],
    )

    assert module.main() == 1
    report = json.loads(
        (tmp_path / "out" / "phase469_fpm_exact_site_support.json").read_text()
    )
    assert report["status"] == "BLOCKED_DESCRIPTOR_CONTRACT"
    assert report["default_aic"] == "No-Go"
