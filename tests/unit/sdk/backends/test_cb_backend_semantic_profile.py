"""Tests for exact backend-version semantic profiles."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aiconfigurator.sdk.backends.cb_simulator.arrival import (
    resolve_tokenizer_primitive,
)
from aiconfigurator.sdk.backends.cb_simulator.backend_semantic_profile import (
    resolve_backend_semantic_profile,
)
from aiconfigurator.sdk.backends.cb_simulator.datatypes import CBSimConfig
from aiconfigurator.sdk.backends.cb_simulator.simulator import CBSimulator


def test_vllm_019_profile_is_exact_and_complete() -> None:
    profile = resolve_backend_semantic_profile(
        backend="vllm",
        version="0.19.0",
    )

    assert profile.backend == "vllm"
    assert profile.version == "0.19.0"
    assert profile.null_blocks_per_pool == 1
    assert profile.batch_queue_depth == 2
    assert profile.async_scheduling is True
    assert profile.pipeline_parallel_sizes == (1,)
    assert profile.engine_loop_default_enabled is False
    assert profile.engine_loop_diagnostic_available is True
    assert profile.request_progress_semantics == "sampled_computed_placeholder"
    assert profile.preemption_victim_policy == "running_tail"
    assert profile.admission_queue_order == "running_then_waiting"
    assert profile.stop_admission_after_preemption is True
    assert profile.stop_after_partial_prefill is True
    assert profile.completion_release_order == "future_then_release_then_submit"
    assert profile.data_directory_contract == (
        "systems/data/<hardware>/<backend>/<version>/"
    )


def test_unknown_backend_version_fails_closed() -> None:
    with pytest.raises(ValueError, match="no exact backend semantic profile"):
        resolve_backend_semantic_profile(
            backend="vllm",
            version="0.20.0",
        )


def test_finite_capacity_requires_an_injected_profile() -> None:
    config = CBSimConfig(num_gpu_blocks=28_825)

    with pytest.raises(ValueError, match="backend semantic profile"):
        _ = config.num_allocatable_gpu_blocks


def test_simulator_injects_database_version_profile() -> None:
    sim = CBSimulator(
        backend=MagicMock(),
        model=MagicMock(),
        database=SimpleNamespace(backend="vllm", version="0.19.0"),
        config=CBSimConfig(num_gpu_blocks=28_825),
    )

    assert sim._config.semantic_profile is not None
    assert sim._config.semantic_profile.version == "0.19.0"
    assert sim._config.num_allocatable_gpu_blocks == 28_824


def test_simulator_rejects_a_mismatched_explicit_profile() -> None:
    profile = resolve_backend_semantic_profile(
        backend="vllm",
        version="0.19.0",
    )
    mismatched = replace(profile, version="0.18.0")

    with pytest.raises(ValueError, match="does not match database"):
        CBSimulator(
            backend=MagicMock(),
            model=MagicMock(),
            database=SimpleNamespace(backend="vllm", version="0.19.0"),
            config=CBSimConfig(semantic_profile=mismatched),
        )


def test_tokenizer_measurement_does_not_own_engine_loop_semantics() -> None:
    primitive = resolve_tokenizer_primitive(
        model_path="moonshotai/Kimi-K2.5",
        system="h200_sxm",
        backend="vllm",
        version="0.19.0",
        prompt_tokens=8_000,
    )

    assert primitive is not None
    assert not hasattr(primitive, "queue_depth")
