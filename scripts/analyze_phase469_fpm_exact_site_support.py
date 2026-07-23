#!/usr/bin/env python3
"""Build the Phase469A exact-site FPM consumption contract."""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import analyze_phase468_fpm_compatibility as phase468


SCHEMA = "phase469_fpm_exact_site_support_v1"
QUERY_MANIFEST_SCHEMA = "phase469_fpm_query_manifest_v1"
NATIVE_GRID_SCHEMA = "phase469_fpm_native_grid_v2"
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "docs"
    / "iter_gap_investigation"
    / "phase469_fpm_exact_site_support"
)
EXPECTED_DESCRIPTOR_SHA256 = (
    "7da3f3ab4d08926e1c4fe918a68a2a2e1248478073621d764b2258987792374d"
)
EXPECTED_DESCRIPTOR_ROW_COUNT = 347_984
EXPECTED_UNIQUE_QUERY_COUNT = 118_049
TP8 = "tp8pp1dp1moetp1ep8cp1"
DP2 = "tp4pp1dp2moetp1ep8cp1"
EXPECTED_GROUP_CONTRACT = {
    (DP2, "prefill"): {
        "unique_query_count": 498,
        "site_count": 324,
        "singleton_site_count": 298,
    },
    (DP2, "decode"): {
        "unique_query_count": 57_916,
        "site_count": 57,
        "singleton_site_count": 1,
    },
    (TP8, "prefill"): {
        "unique_query_count": 802,
        "site_count": 575,
        "singleton_site_count": 541,
    },
    (TP8, "decode"): {
        "unique_query_count": 58_833,
        "site_count": 67,
        "singleton_site_count": 0,
    },
}
EXPECTED_NATIVE_SOURCE_CONTRACT = {
    "aic_collector_commit": "cf1b3cbbc00e0a822891abd29310d68a259e79ed",
    "dynamo_commit": "41882ae9b07232eed4850fb1daf8c958abb2556a",
    "native_schema_version": 5,
    "coordinate_system": "iteration_totals_balanced_v1",
    "partition_policy": "balanced_v1",
    "latency_aggregation": "max_dp_rank",
}
RUNTIME_IDENTITY_FIELDS = {
    "model_id",
    "model_load_path",
    "model_config_sha256",
    "checkpoint_fingerprint",
    "hardware",
    "backend",
    "backend_version",
    "expected_quant_runtime",
    "actual_quant_runtime",
    "execution_manifest_sha256",
    "phase469b_preflight_manifest_sha256",
    "phase469c_canary_manifest_sha256",
}
CELL_FIELDS = {
    "cell_id",
    "workload_kind",
    "tp",
    "pp",
    "dp",
    "moe_tp",
    "moe_ep",
    "cp",
    "topology",
    "runtime_run_id",
    "runtime_grid_digest",
    "coordinate_grid_sha256",
    "measurement_sha256",
    "points",
}
POINT_FIELDS = {
    "batch_size",
    "total_prefill_tokens",
    "total_kv_read_tokens",
    "latency_ms",
}
TOP_LEVEL_FIELDS = {
    "schema",
    "source_contract",
    "runtime_identity",
    "cells",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
}
REQUIRED_CELL_SETS = {
    "tp8_pilot": frozenset({(TP8, "prefill"), (TP8, "decode")}),
    "final": frozenset(
        {
            (TP8, "prefill"),
            (TP8, "decode"),
            (DP2, "prefill"),
            (DP2, "decode"),
        }
    ),
}


class DescriptorContractError(ValueError):
    """Phase468 descriptor identity or query structure changed."""


class NativeGridContractError(ValueError):
    """A candidate native FPM grid violates the normalized input schema."""


@dataclass(frozen=True, order=True)
class ExactSiteQuery:
    phase: str
    topology: str
    batch_size: int
    total_kv_read_tokens: int
    total_prefill_tokens: int

    def __post_init__(self) -> None:
        if self.phase not in {"prefill", "decode"}:
            raise ValueError(f"unsupported query phase:{self.phase}")
        if self.topology not in phase468.ALLOWED_TOPOLOGIES:
            raise ValueError(f"unsupported query topology:{self.topology}")
        values = (
            self.batch_size,
            self.total_kv_read_tokens,
            self.total_prefill_tokens,
        )
        if any(not isinstance(value, int) or value < 0 for value in values):
            raise ValueError("query coordinates must be non-negative integers")
        if self.batch_size == 0:
            raise ValueError("query batch_size must be positive")
        if self.phase == "prefill" and self.total_prefill_tokens == 0:
            raise ValueError("prefill query requires positive total_prefill_tokens")
        if self.phase == "decode" and self.total_prefill_tokens != 0:
            raise ValueError("decode query cannot carry prefill tokens")

    @property
    def axis(self) -> int:
        if self.phase == "prefill":
            return self.total_prefill_tokens
        return self.total_kv_read_tokens

    @property
    def site_key(self) -> tuple:
        if self.phase == "prefill":
            return (
                self.phase,
                self.topology,
                self.batch_size,
                self.total_kv_read_tokens,
            )
        return self.phase, self.topology, self.batch_size

    @property
    def coordinate(self) -> tuple:
        return (*self.site_key, self.axis)


@dataclass(frozen=True)
class QueryJudgement:
    outcome: str
    lower_axis: int | None
    upper_axis: int | None


@dataclass(frozen=True)
class NativeGridIndex:
    axes_by_site: dict[tuple, tuple[int, ...]]
    point_count: int
    cell_ids: tuple[str, ...]
    runtime_grid_digests: tuple[str, ...]
    coordinate_grid_sha256s: tuple[str, ...]
    measurement_sha256s: tuple[str, ...]
    required_cell_set: str
    source_contract: dict
    runtime_identity: dict


def _require_int(point: dict, field: str) -> int:
    value = point.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise NativeGridContractError(
            f"native point field must be a non-negative integer:{field}"
        )
    return value


def _is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _require_non_empty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise NativeGridContractError(f"native grid field must be non-empty:{field}")
    return value


def _validate_runtime_identity(
    actual: object,
    expected: object,
) -> dict:
    if not isinstance(actual, dict) or set(actual) != RUNTIME_IDENTITY_FIELDS:
        raise NativeGridContractError("runtime identity fields do not match schema")
    if not isinstance(expected, dict) or set(expected) != RUNTIME_IDENTITY_FIELDS:
        raise NativeGridContractError("expected runtime identity fields do not match schema")

    fixed = {
        "model_id": phase468.EXPECTED_MODEL_PATH,
        "model_config_sha256": phase468.EXPECTED_MODEL_CONFIG_SHA256,
        "hardware": phase468.EXPECTED_HARDWARE,
        "backend": phase468.EXPECTED_BACKEND,
        "backend_version": phase468.EXPECTED_BACKEND_VERSION,
        "expected_quant_runtime": phase468.EXPECTED_QUANT_RUNTIME,
        "actual_quant_runtime": phase468.EXPECTED_QUANT_RUNTIME,
    }
    for role, identity in (("runtime", actual), ("expected", expected)):
        for field in RUNTIME_IDENTITY_FIELDS:
            _require_non_empty_string(identity[field], f"{role}_identity.{field}")
        for field in (
            "model_config_sha256",
            "checkpoint_fingerprint",
            "execution_manifest_sha256",
            "phase469b_preflight_manifest_sha256",
            "phase469c_canary_manifest_sha256",
        ):
            if not _is_lower_sha256(identity[field]):
                raise NativeGridContractError(
                    f"{role} runtime identity requires lowercase SHA256:{field}"
                )
        for field, value in fixed.items():
            if identity[field] != value:
                raise NativeGridContractError(
                    f"{role} runtime identity contract mismatch:{field}"
                )
    for field in sorted(RUNTIME_IDENTITY_FIELDS):
        if actual[field] != expected[field]:
            raise NativeGridContractError(f"runtime identity mismatch:{field}")
    return dict(actual)


def _canonical_cell_digest_payload(cell: dict, *, include_latency: bool) -> dict:
    points = cell.get("points")
    if not isinstance(points, list):
        raise NativeGridContractError("native cell points are missing")
    point_fields = [
        "batch_size",
        "total_prefill_tokens",
        "total_kv_read_tokens",
    ]
    if include_latency:
        point_fields.append("latency_ms")
    canonical_points = [
        {field: point.get(field) for field in point_fields}
        for point in points
        if isinstance(point, dict)
    ]
    if len(canonical_points) != len(points):
        raise NativeGridContractError("native grid point must be an object")
    return {
        "workload_kind": cell.get("workload_kind"),
        "topology": cell.get("topology"),
        "points": sorted(
            canonical_points,
            key=lambda point: json.dumps(
                point,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ),
    }


def compute_coordinate_grid_sha256(cell: dict) -> str:
    payload = _canonical_cell_digest_payload(cell, include_latency=False)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def compute_measurement_sha256(cell: dict) -> str:
    payload = _canonical_cell_digest_payload(cell, include_latency=True)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _validate_cell_topology(cell: dict) -> tuple[str, str]:
    workload_kind = cell.get("workload_kind")
    if workload_kind not in {"prefill", "decode"}:
        raise NativeGridContractError("native cell workload_kind is unsupported")
    dimensions = {}
    for field in ("tp", "pp", "dp", "moe_tp", "moe_ep", "cp"):
        value = cell.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise NativeGridContractError(
                f"native cell topology dimension must be positive:{field}"
            )
        dimensions[field] = value
    if dimensions["pp"] != 1:
        raise NativeGridContractError("native cell requires PP=1")
    if dimensions["cp"] != 1:
        raise NativeGridContractError("native cell requires CP=1")
    if dimensions["tp"] * dimensions["pp"] * dimensions["dp"] != 8:
        raise NativeGridContractError("native cell requires exactly 8 GPUs")
    if dimensions["moe_tp"] * dimensions["moe_ep"] != (
        dimensions["tp"] * dimensions["dp"]
    ):
        raise NativeGridContractError("native cell MoE world must match attention world")
    expected_topology = phase468.make_forward_workload_topology_key(**dimensions)
    topology = cell.get("topology")
    if topology != expected_topology:
        raise NativeGridContractError(
            f"native cell topology mismatch:{topology!r}!={expected_topology!r}"
        )
    if topology not in phase468.ALLOWED_TOPOLOGIES:
        raise NativeGridContractError(f"native cell topology is unsupported:{topology}")
    return topology, workload_kind


def _query_from_native_point(
    point: dict,
    *,
    workload_kind: str,
    topology: str,
) -> ExactSiteQuery:
    if set(point) != POINT_FIELDS:
        raise NativeGridContractError("native point fields do not match schema")
    batch_size = _require_int(point, "batch_size")
    prefill_tokens = _require_int(point, "total_prefill_tokens")
    kv_tokens = _require_int(point, "total_kv_read_tokens")
    latency_ms = point.get("latency_ms")
    if (
        not isinstance(latency_ms, (int, float))
        or isinstance(latency_ms, bool)
        or not math.isfinite(float(latency_ms))
        or float(latency_ms) <= 0.0
    ):
        raise NativeGridContractError("native point latency_ms must be finite and positive")
    if batch_size == 0:
        raise NativeGridContractError("native point batch_size must be positive")
    if workload_kind == "prefill":
        if prefill_tokens == 0:
            raise NativeGridContractError("prefill point coordinates are invalid")
        return ExactSiteQuery(
            phase="prefill",
            topology=topology,
            batch_size=batch_size,
            total_kv_read_tokens=kv_tokens,
            total_prefill_tokens=prefill_tokens,
        )
    if workload_kind == "decode":
        if prefill_tokens != 0:
            raise NativeGridContractError("decode point coordinates are invalid")
        return ExactSiteQuery(
            phase="decode",
            topology=topology,
            batch_size=batch_size,
            total_kv_read_tokens=kv_tokens,
            total_prefill_tokens=0,
        )
    raise NativeGridContractError(f"unsupported native point phase:{workload_kind}")


def normalize_native_grid(
    payload: dict,
    *,
    expected_runtime_identity: dict,
    required_cell_set: str,
) -> NativeGridIndex:
    if not isinstance(payload, dict) or set(payload) != TOP_LEVEL_FIELDS:
        raise NativeGridContractError("native grid top-level fields do not match schema")
    if payload.get("schema") != NATIVE_GRID_SCHEMA:
        raise NativeGridContractError("native grid schema mismatch")
    if payload.get("source_contract") != EXPECTED_NATIVE_SOURCE_CONTRACT:
        raise NativeGridContractError("native grid source contract mismatch")
    runtime_identity = _validate_runtime_identity(
        payload.get("runtime_identity"),
        expected_runtime_identity,
    )
    if (
        payload.get("diagnostic_only") is not True
        or payload.get("valid_for_default") is not False
        or payload.get("perf_database") is not False
    ):
        raise NativeGridContractError("native grid must remain diagnostic-only")
    if required_cell_set not in REQUIRED_CELL_SETS:
        raise NativeGridContractError(f"unknown required cell set:{required_cell_set}")
    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells:
        raise NativeGridContractError("native grid cells are missing")

    axes: dict[tuple, set[int]] = defaultdict(set)
    coordinates: set[tuple] = set()
    cell_ids: set[str] = set()
    run_ids: set[str] = set()
    grid_digests: list[str] = []
    coordinate_digests: set[str] = set()
    measurement_digests: set[str] = set()
    cell_keys: set[tuple[str, str]] = set()
    for cell in cells:
        if not isinstance(cell, dict) or set(cell) != CELL_FIELDS:
            raise NativeGridContractError("native cell fields do not match schema")
        cell_id = _require_non_empty_string(cell.get("cell_id"), "cell_id")
        run_id = _require_non_empty_string(
            cell.get("runtime_run_id"),
            "runtime_run_id",
        )
        grid_digest = cell.get("runtime_grid_digest")
        coordinate_digest = cell.get("coordinate_grid_sha256")
        measurement_digest = cell.get("measurement_sha256")
        if not _is_lower_sha256(grid_digest):
            raise NativeGridContractError(
                "runtime_grid_digest must be a lowercase SHA256"
            )
        if not _is_lower_sha256(coordinate_digest):
            raise NativeGridContractError(
                "coordinate_grid_sha256 must be a lowercase SHA256"
            )
        if not _is_lower_sha256(measurement_digest):
            raise NativeGridContractError(
                "measurement_sha256 must be a lowercase SHA256"
            )
        if cell_id in cell_ids:
            raise NativeGridContractError(f"duplicate cell_id:{cell_id}")
        if run_id in run_ids:
            raise NativeGridContractError(f"duplicate runtime_run_id:{run_id}")
        cell_ids.add(cell_id)
        run_ids.add(run_id)
        grid_digests.append(grid_digest)
        coordinate_digests.add(coordinate_digest)
        measurement_digests.add(measurement_digest)

        topology, workload_kind = _validate_cell_topology(cell)
        cell_key = (topology, workload_kind)
        if cell_key in cell_keys:
            raise NativeGridContractError(
                f"duplicate topology phase cell:{cell_key!r}"
            )
        cell_keys.add(cell_key)
        expected_coordinate_digest = compute_coordinate_grid_sha256(cell)
        if coordinate_digest != expected_coordinate_digest:
            raise NativeGridContractError(
                "coordinate_grid_sha256 mismatch:"
                f"{coordinate_digest}!={expected_coordinate_digest}"
            )
        expected_measurement_digest = compute_measurement_sha256(cell)
        if measurement_digest != expected_measurement_digest:
            raise NativeGridContractError(
                "measurement_sha256 mismatch:"
                f"{measurement_digest}!={expected_measurement_digest}"
            )

        points = cell.get("points")
        if not isinstance(points, list) or not points:
            raise NativeGridContractError(f"native cell points are missing:{cell_id}")
        for point in points:
            if not isinstance(point, dict):
                raise NativeGridContractError("native grid point must be an object")
            query = _query_from_native_point(
                point,
                workload_kind=workload_kind,
                topology=topology,
            )
            if query.coordinate in coordinates:
                raise NativeGridContractError(
                    f"duplicate native point:{query.coordinate!r}"
                )
            coordinates.add(query.coordinate)
            axes[query.site_key].add(query.axis)

    required_cells = REQUIRED_CELL_SETS[required_cell_set]
    if cell_keys != required_cells:
        raise NativeGridContractError(
            f"native grid required cell set mismatch:{sorted(cell_keys)!r}"
            f"!={sorted(required_cells)!r}"
        )
    return NativeGridIndex(
        axes_by_site={key: tuple(sorted(values)) for key, values in axes.items()},
        point_count=len(coordinates),
        cell_ids=tuple(sorted(cell_ids)),
        runtime_grid_digests=tuple(sorted(grid_digests)),
        coordinate_grid_sha256s=tuple(sorted(coordinate_digests)),
        measurement_sha256s=tuple(sorted(measurement_digests)),
        required_cell_set=required_cell_set,
        source_contract=dict(EXPECTED_NATIVE_SOURCE_CONTRACT),
        runtime_identity=runtime_identity,
    )


def judge_exact_site_query(
    query: ExactSiteQuery,
    index: NativeGridIndex,
) -> QueryJudgement:
    axes = index.axes_by_site.get(query.site_key)
    if axes is None:
        return QueryJudgement("MISSING_SITE", None, None)
    position = bisect.bisect_left(axes, query.axis)
    if position < len(axes) and axes[position] == query.axis:
        return QueryJudgement("EXACT", query.axis, query.axis)
    if len(axes) == 1:
        return QueryJudgement("SINGLETON_MISS", axes[0], axes[0])
    if position == 0:
        return QueryJudgement("CURVE_UNDERFLOW", None, axes[0])
    if position == len(axes):
        return QueryJudgement("CURVE_OVERFLOW", axes[-1], None)
    return QueryJudgement(
        "SAME_SITE_INTERPOLATION",
        axes[position - 1],
        axes[position],
    )


def audit_query_inventory(
    inventory: Counter[ExactSiteQuery],
    index: NativeGridIndex,
) -> dict:
    unique_counts: Counter[str] = Counter()
    use_counts: Counter[str] = Counter()
    for query, use_count in inventory.items():
        outcome = judge_exact_site_query(query, index).outcome
        unique_counts[outcome] += 1
        use_counts[outcome] += use_count

    queryable = {"EXACT", "SAME_SITE_INTERPOLATION"}
    queryable_unique_count = sum(unique_counts[name] for name in queryable)
    queryable_use_count = sum(use_counts[name] for name in queryable)
    return {
        "unique_query_count": sum(unique_counts.values()),
        "query_use_count": sum(use_counts.values()),
        "queryable_unique_count": queryable_unique_count,
        "queryable_use_count": queryable_use_count,
        "all_queries_queryable": queryable_unique_count == len(inventory),
        "outcome_unique_counts": dict(sorted(unique_counts.items())),
        "outcome_use_counts": dict(sorted(use_counts.items())),
    }


def _query_from_descriptor(row: dict, phase: str) -> ExactSiteQuery:
    if phase == "prefill":
        return ExactSiteQuery(
            phase=phase,
            topology=str(row["topology"]),
            batch_size=int(row["num_prefill_requests"]),
            total_kv_read_tokens=int(row["sum_prefill_kv_tokens"]),
            total_prefill_tokens=int(row["sum_prefill_tokens"]),
        )
    if phase == "decode":
        return ExactSiteQuery(
            phase=phase,
            topology=str(row["topology"]),
            batch_size=int(row["num_decode_requests"]),
            total_kv_read_tokens=int(row["sum_decode_kv_tokens"]),
            total_prefill_tokens=0,
        )
    raise ValueError(f"unsupported descriptor phase:{phase}")


def build_query_inventory(rows: list[dict]) -> Counter[ExactSiteQuery]:
    inventory: Counter[ExactSiteQuery] = Counter()
    for row in rows:
        if not bool(row["active"]):
            continue
        if int(row["num_prefill_requests"]) > 0:
            inventory[_query_from_descriptor(row, "prefill")] += 1
        if int(row["num_decode_requests"]) > 0:
            inventory[_query_from_descriptor(row, "decode")] += 1
    if not inventory:
        raise DescriptorContractError("Phase468 query inventory is empty")
    return inventory


def summarize_query_inventory(
    inventory: Counter[ExactSiteQuery],
) -> tuple[list[dict], list[dict], list[dict]]:
    grouped: dict[tuple[str, str], list[ExactSiteQuery]] = defaultdict(list)
    for query in inventory:
        grouped[(query.topology, query.phase)].append(query)

    summaries: list[dict] = []
    curve_distributions: list[dict] = []
    usage_distributions: list[dict] = []
    for (topology, phase), queries in sorted(grouped.items()):
        sites: dict[tuple, set[int]] = defaultdict(set)
        for query in queries:
            sites[query.site_key].add(query.axis)
        curve_counts = Counter(len(axes) for axes in sites.values())
        usage_counts = Counter(inventory[query] for query in queries)
        summaries.append(
            {
                "topology": topology,
                "phase": phase,
                "unique_query_count": len(queries),
                "query_use_count": sum(inventory[query] for query in queries),
                "site_count": len(sites),
                "singleton_site_count": curve_counts.get(1, 0),
                "curve_point_count_min": min(curve_counts),
                "curve_point_count_max": max(curve_counts),
            }
        )
        curve_distributions.extend(
            {
                "topology": topology,
                "phase": phase,
                "curve_point_count": point_count,
                "site_count": site_count,
            }
            for point_count, site_count in sorted(curve_counts.items())
        )
        usage_distributions.extend(
            {
                "topology": topology,
                "phase": phase,
                "query_usage_count": usage_count,
                "unique_query_count": query_count,
                "total_query_uses": usage_count * query_count,
            }
            for usage_count, query_count in sorted(usage_counts.items())
        )
    return summaries, curve_distributions, usage_distributions


def _group_contract(summaries: list[dict]) -> dict:
    return {
        (row["topology"], row["phase"]): {
            "unique_query_count": row["unique_query_count"],
            "site_count": row["site_count"],
            "singleton_site_count": row["singleton_site_count"],
        }
        for row in summaries
    }


def validate_descriptor_contract(
    rows: list[dict],
) -> tuple[Counter[ExactSiteQuery], dict, list[dict], list[dict], list[dict]]:
    manifest = phase468.build_descriptor_manifest(rows)
    if manifest["row_count"] != EXPECTED_DESCRIPTOR_ROW_COUNT:
        raise DescriptorContractError(
            "Phase468 descriptor row count drift:"
            f"{manifest['row_count']}!={EXPECTED_DESCRIPTOR_ROW_COUNT}"
        )
    if manifest["streaming_csv_sha256"] != EXPECTED_DESCRIPTOR_SHA256:
        raise DescriptorContractError(
            "Phase468 descriptor SHA256 drift:"
            f"{manifest['streaming_csv_sha256']}!={EXPECTED_DESCRIPTOR_SHA256}"
        )
    inventory = build_query_inventory(rows)
    if len(inventory) != EXPECTED_UNIQUE_QUERY_COUNT:
        raise DescriptorContractError(
            "Phase468 unique query count drift:"
            f"{len(inventory)}!={EXPECTED_UNIQUE_QUERY_COUNT}"
        )
    summaries, curve_distributions, usage_distributions = (
        summarize_query_inventory(inventory)
    )
    actual_groups = _group_contract(summaries)
    if actual_groups != EXPECTED_GROUP_CONTRACT:
        raise DescriptorContractError(
            "Phase468 query/site structure drift:"
            f"{actual_groups!r}!={EXPECTED_GROUP_CONTRACT!r}"
        )
    return (
        inventory,
        manifest,
        summaries,
        curve_distributions,
        usage_distributions,
    )


def collect_phase468_descriptor_rows() -> list[dict]:
    try:
        _numeric, rows, _coverage, _runtime_contracts = phase468.collect()
    except (phase468.SourceContractError, phase468.RuntimeContractError) as error:
        raise DescriptorContractError(
            f"Phase468 compatibility contract blocked:{error}"
        ) from error
    return rows


def _query_inventory_sha256(inventory: Counter[ExactSiteQuery]) -> str:
    digest = hashlib.sha256()
    for query in sorted(inventory):
        row = {
            "phase": query.phase,
            "topology": query.topology,
            "batch_size": query.batch_size,
            "total_kv_read_tokens": query.total_kv_read_tokens,
            "total_prefill_tokens": query.total_prefill_tokens,
            "usage_count": inventory[query],
        }
        digest.update(
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise DescriptorContractError(f"refusing to write empty CSV:{path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _required_cell_sets_document() -> dict:
    return {
        name: [
            {"topology": topology, "workload_kind": workload_kind}
            for topology, workload_kind in sorted(cells)
        ]
        for name, cells in sorted(REQUIRED_CELL_SETS.items())
    }


def _native_grid_schema_document() -> dict:
    return {
        "schema": NATIVE_GRID_SCHEMA,
        "top_level_fields": sorted(TOP_LEVEL_FIELDS),
        "source_contract": dict(EXPECTED_NATIVE_SOURCE_CONTRACT),
        "runtime_identity_fields": sorted(RUNTIME_IDENTITY_FIELDS),
        "cell_fields": sorted(CELL_FIELDS),
        "point_fields": sorted(POINT_FIELDS),
        "runtime_grid_digest": (
            "opaque Dynamo native digest extracted by the pinned native reader;"
            "Phase469A validates lowercase SHA256 only"
        ),
        "coordinate_grid_sha256": (
            "sha256(canonical topology, workload kind, and order-independent coordinates)"
        ),
        "measurement_sha256": (
            "sha256(canonical topology, workload kind, coordinates, and latency_ms)"
        ),
        "required_cell_sets": _required_cell_sets_document(),
        "phase_contracts": {
            "prefill": {
                "site_key": [
                    "topology",
                    "batch_size",
                    "total_kv_read_tokens",
                ],
                "curve_axis": "total_prefill_tokens",
            },
            "decode": {
                "site_key": ["topology", "batch_size"],
                "curve_axis": "total_kv_read_tokens",
            },
        },
        "identity_comparison": (
            "runtime_identity_exactly_equals_phase469b_preflight_plus_"
            "phase469c_canary_identity"
        ),
        "quant_runtime_contract": {
            "phase469b_expected": phase468.EXPECTED_QUANT_RUNTIME,
            "phase469b_actual": "DEFERRED_TO_MODEL_CANARY",
            "native_grid_expected": phase468.EXPECTED_QUANT_RUNTIME,
            "native_grid_actual": phase468.EXPECTED_QUANT_RUNTIME,
        },
        "topology_contract": {
            "gpu_count": 8,
            "pp": 1,
            "cp": 1,
            "moe_world": "moe_tp*moe_ep == tp*dp",
        },
        "lookup_order": [
            "exact_coordinate",
            "same_site_strict_bracket",
            "singleton_exact_only",
            "fail_closed",
        ],
        "forbidden": [
            "cross_site_transfer",
            "nearest",
            "clamp",
            "util_hold",
            "extrapolation",
            "cross_topology",
        ],
        "interpolation_execution": (
            "guard_only_then_pinned_upstream_interpolator;"
            "this_phase_does_not_compute_latency"
        ),
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
    }


def _base_report(status: str) -> dict:
    return {
        "schema": SCHEMA,
        "status": status,
        "phase468_commit": "ed91cf45560abd90330c3ee8d144aac8cafb8b3a",
        "expected_descriptor_sha256": EXPECTED_DESCRIPTOR_SHA256,
        "native_grid_schema": NATIVE_GRID_SCHEMA,
        "native_grid_coverage": "NOT_EVALUATED_NO_NATIVE_GRID",
        "native_grid_source_contract": dict(EXPECTED_NATIVE_SOURCE_CONTRACT),
        "native_grid_required_cell_sets": _required_cell_sets_document(),
        "runtime_identity_source": (
            "phase469b_preflight_manifest+phase469c_canary_manifest"
        ),
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
        "default_aic": "No-Go",
    }


def _write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# Phase469A Exact Site Support Contract",
        "",
        f"Status: `{report['status']}`.",
        "",
        "| Topology | Phase | Queries | Uses | Sites | Singleton sites | Curve points |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in report.get("query_groups", []):
        lines.append(
            "| {topology} | {phase} | {unique_query_count} | {query_use_count} | "
            "{site_count} | {singleton_site_count} | {curve_point_count_min}-{curve_point_count_max} |".format(
                **row
            )
        )
    if report.get("block_reason"):
        lines.extend(["", "## Block Reason", "", str(report["block_reason"])])
    lines.extend(
        [
            "",
            "No native grid was supplied, so collection coverage is not evaluated.",
            "Native-grid identity must bind both the Phase469B preflight and Phase469C canary manifests.",
            "Phase469B records only the expected quant runtime; Phase469C must attest the loaded quant runtime.",
            "The runtime grid digest is the opaque Dynamo value extracted by the pinned native reader.",
            "Coordinate and measurement digests are separate order-independent integrity checks.",
            "The guard permits only exact coordinates or a strict bracket inside the same site.",
            "It does not calculate latency; interpolation remains delegated to the pinned upstream implementation.",
            "The seven repository artifacts total approximately 17 KiB; no full descriptor CSV is stored.",
            "",
            "`diagnostic_only=true`, `valid_for_default=false`, `perf_database=false`, `Default AIC=No-Go`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_report(out_dir: Path, report: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase469_fpm_exact_site_support.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(out_dir / "phase469_fpm_exact_site_support.md", report)


def run(out_dir: Path) -> dict:
    rows = collect_phase468_descriptor_rows()
    (
        inventory,
        descriptor_manifest,
        summaries,
        curve_distributions,
        usage_distributions,
    ) = validate_descriptor_contract(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    query_manifest = {
        "schema": QUERY_MANIFEST_SCHEMA,
        "phase468_descriptor_sha256": descriptor_manifest["streaming_csv_sha256"],
        "phase468_descriptor_row_count": descriptor_manifest["row_count"],
        "unique_query_count": len(inventory),
        "query_inventory_sha256": _query_inventory_sha256(inventory),
        "query_groups": summaries,
        "native_grid_source_contract": dict(EXPECTED_NATIVE_SOURCE_CONTRACT),
        "runtime_identity_fields": sorted(RUNTIME_IDENTITY_FIELDS),
        "runtime_identity_source": (
            "phase469b_preflight_manifest+phase469c_canary_manifest"
        ),
        "native_grid_required_cell_sets": _required_cell_sets_document(),
        "full_query_rows_materialized": False,
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
        "default_aic": "No-Go",
    }
    (out_dir / "phase469_query_manifest.json").write_text(
        json.dumps(query_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "phase469_native_grid_schema.json").write_text(
        json.dumps(_native_grid_schema_document(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "phase469_query_site_summary.csv", summaries)
    _write_csv(
        out_dir / "phase469_curve_point_distribution.csv",
        curve_distributions,
    )
    _write_csv(
        out_dir / "phase469_query_usage_frequency.csv",
        usage_distributions,
    )
    report = {
        **_base_report("READY_FOR_NATIVE_GRID_INPUT"),
        "descriptor_contract": "PASS",
        "descriptor_row_count": descriptor_manifest["row_count"],
        "unique_query_count": len(inventory),
        "query_inventory_sha256": query_manifest["query_inventory_sha256"],
        "query_groups": summaries,
        "full_descriptor_csv_generated": False,
        "full_query_rows_materialized": False,
    }
    _write_report(out_dir, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    try:
        report = run(args.out_dir)
    except DescriptorContractError as error:
        report = {
            **_base_report("BLOCKED_DESCRIPTOR_CONTRACT"),
            "descriptor_contract": "BLOCKED",
            "block_reason": str(error),
        }
        _write_report(args.out_dir, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "READY_FOR_NATIVE_GRID_INPUT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
