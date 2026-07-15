"""Exact backend-version semantics for the CB simulator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files


_RESOURCE_PACKAGE = "aiconfigurator.sdk.backends.cb_simulator"
_RESOURCE_PATH = "resources/backend_semantic_profiles.json"
_SUPPORTED_PROGRESS = {"sampled_computed_placeholder"}
_SUPPORTED_VICTIM_POLICIES = {"running_tail"}
_SUPPORTED_ADMISSION_ORDERS = {"running_then_waiting"}
_SUPPORTED_COMPLETION_ORDERS = {"future_then_release_then_submit"}


@dataclass(frozen=True)
class BackendSemanticProfile:
    backend: str
    version: str
    null_blocks_per_pool: int
    batch_queue_depth: int
    async_scheduling: bool
    pipeline_parallel_sizes: tuple[int, ...]
    engine_loop_default_enabled: bool
    engine_loop_diagnostic_available: bool
    request_progress_semantics: str
    preemption_victim_policy: str
    admission_queue_order: str
    stop_admission_after_preemption: bool
    stop_after_partial_prefill: bool
    completion_release_order: str
    data_directory_contract: str


def _load_profiles() -> tuple[BackendSemanticProfile, ...]:
    resource = files(_RESOURCE_PACKAGE).joinpath(_RESOURCE_PATH)
    data = json.loads(resource.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("unsupported backend semantic profile schema")
    directory_contract = str(data["data_directory_contract"])

    profiles = []
    for row in data.get("profiles", []):
        profile = BackendSemanticProfile(
            backend=str(row["backend"]),
            version=str(row["version"]),
            null_blocks_per_pool=int(row["null_blocks_per_pool"]),
            batch_queue_depth=int(row["batch_queue_depth"]),
            async_scheduling=bool(row["async_scheduling"]),
            pipeline_parallel_sizes=tuple(
                int(value) for value in row["pipeline_parallel_sizes"]
            ),
            engine_loop_default_enabled=bool(
                row["engine_loop_default_enabled"]
            ),
            engine_loop_diagnostic_available=bool(
                row["engine_loop_diagnostic_available"]
            ),
            request_progress_semantics=str(row["request_progress_semantics"]),
            preemption_victim_policy=str(row["preemption_victim_policy"]),
            admission_queue_order=str(row["admission_queue_order"]),
            stop_admission_after_preemption=bool(
                row["stop_admission_after_preemption"]
            ),
            stop_after_partial_prefill=bool(row["stop_after_partial_prefill"]),
            completion_release_order=str(row["completion_release_order"]),
            data_directory_contract=directory_contract,
        )
        _validate_profile(profile)
        profiles.append(profile)
    if not profiles:
        raise ValueError("backend semantic profile resource is empty")
    keys = [(profile.backend, profile.version) for profile in profiles]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate backend semantic profile key")
    return tuple(profiles)


def _validate_profile(profile: BackendSemanticProfile) -> None:
    if profile.null_blocks_per_pool < 0:
        raise ValueError("null block reserve must be non-negative")
    if profile.batch_queue_depth < 1:
        raise ValueError("batch queue depth must be positive")
    if not profile.pipeline_parallel_sizes:
        raise ValueError("pipeline parallel applicability must not be empty")
    if profile.request_progress_semantics not in _SUPPORTED_PROGRESS:
        raise ValueError("unsupported request progress semantics")
    if profile.preemption_victim_policy not in _SUPPORTED_VICTIM_POLICIES:
        raise ValueError("unsupported preemption victim policy")
    if profile.admission_queue_order not in _SUPPORTED_ADMISSION_ORDERS:
        raise ValueError("unsupported admission queue order")
    if profile.completion_release_order not in _SUPPORTED_COMPLETION_ORDERS:
        raise ValueError("unsupported completion release order")


def resolve_backend_semantic_profile(
    *,
    backend: str,
    version: str,
) -> BackendSemanticProfile:
    matches = [
        profile
        for profile in _load_profiles()
        if profile.backend == backend and profile.version == version
    ]
    if len(matches) != 1:
        raise ValueError(
            "no exact backend semantic profile for "
            f"backend={backend!r}, version={version!r}"
        )
    return matches[0]
