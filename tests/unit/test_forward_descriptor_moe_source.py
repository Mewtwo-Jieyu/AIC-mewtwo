from __future__ import annotations

from dataclasses import asdict

import pytest

from aiconfigurator.sdk.backends.cb_simulator.forward_descriptor import (
    VLLMMoESourceRuntimeKey,
    moe_source_runtime_key_from_loaded_weight_boundary_row,
)
from aiconfigurator.sdk.backends.cb_simulator import (
    VLLMMoESourceRuntimeKey as ExportedVLLMMoESourceRuntimeKey,
)


def _base_row() -> dict[str, object]:
    return {
        "source": "phase82_loaded_weight_boundary",
        "scenario": "kimi_k25_h200_tp4dp2ep8",
        "runtime_backend": "vllm",
        "vllm_version": "0.19.0",
        "model_family": "kimi_k25",
        "module_class": "DeepseekV2MoE",
        "experts_class": "SharedFusedMoE",
        "hidden_size": 7168,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 384,
        "local_experts": 48,
        "global_experts": 384,
        "topk": 8,
        "n_shared_experts": 1,
        "moe_method": "CompressedTensorsWNA16MarlinMoEMethod",
        "kernel_backend": "wna16_marlin",
        "group_size": 32,
        "num_bits": 4,
        "dtype": "bfloat16",
        "tp_size": 4,
        "dp_size": 2,
        "ep_size": 8,
        "world_size": 8,
        "rank": 0,
        "device": "cuda:0",
        "tuning_config_loaded": False,
        "moe_config_fallback": True,
        "moe_tuning_config_file": "",
        "loaded_weight": True,
        "random_weight": False,
        "timing": False,
        "valid_for_default": False,
        "perf_database": False,
        "diagnostic_only": True,
    }


def test_moe_source_key_accepts_loaded_weight_fallback_descriptor() -> None:
    key = moe_source_runtime_key_from_loaded_weight_boundary_row(_base_row())

    values = asdict(key)
    assert key.loaded_weight is True
    assert key.random_weight is False
    assert key.tuning_config_loaded is False
    assert key.moe_config_fallback is True
    assert key.valid_for_default is False
    assert key.perf_database is False
    assert key.diagnostic_only is True
    assert key.module_class == "DeepseekV2MoE"
    assert key.experts_class == "SharedFusedMoE"
    assert key.hidden_size == 7168
    assert key.moe_intermediate_size == 2048
    assert key.local_experts == 48
    assert key.global_experts == 384
    for forbidden in (
        "latency_ms",
        "duration_ms",
        "residual_ms",
        "profiled_cuda",
        "profiler",
        "trace",
        "sync",
        "throughput",
    ):
        assert forbidden not in values


def test_moe_source_key_is_exported() -> None:
    assert ExportedVLLMMoESourceRuntimeKey is VLLMMoESourceRuntimeKey


def test_moe_source_key_rejects_random_or_unloaded_weight() -> None:
    row = _base_row()
    row["loaded_weight"] = False
    with pytest.raises(ValueError, match="loaded_weight must be true"):
        moe_source_runtime_key_from_loaded_weight_boundary_row(row)

    row = _base_row()
    row["random_weight"] = True
    with pytest.raises(ValueError, match="random_weight must be false"):
        moe_source_runtime_key_from_loaded_weight_boundary_row(row)


def test_moe_source_key_rejects_missing_class_or_shape() -> None:
    row = _base_row()
    row["module_class"] = ""
    with pytest.raises(ValueError, match="missing module_class"):
        moe_source_runtime_key_from_loaded_weight_boundary_row(row)

    row = _base_row()
    row["hidden_size"] = 0
    with pytest.raises(ValueError, match="hidden_size must be positive"):
        moe_source_runtime_key_from_loaded_weight_boundary_row(row)


def test_moe_source_key_rejects_default_perf_timing_or_forbidden_fields() -> None:
    row = _base_row()
    row["valid_for_default"] = True
    with pytest.raises(ValueError, match="valid_for_default must be false"):
        moe_source_runtime_key_from_loaded_weight_boundary_row(row)

    row = _base_row()
    row["perf_database"] = True
    with pytest.raises(ValueError, match="perf_database must be false"):
        moe_source_runtime_key_from_loaded_weight_boundary_row(row)

    row = _base_row()
    row["timing"] = True
    with pytest.raises(ValueError, match="timing must be false"):
        moe_source_runtime_key_from_loaded_weight_boundary_row(row)

    row = _base_row()
    row["latency_ms"] = 1.0
    with pytest.raises(ValueError, match="forbidden field"):
        moe_source_runtime_key_from_loaded_weight_boundary_row(row)


def test_moe_source_key_rejects_topology_mismatch() -> None:
    row = _base_row()
    row["world_size"] = 16
    with pytest.raises(ValueError, match="world_size must equal tp_size \\* dp_size"):
        moe_source_runtime_key_from_loaded_weight_boundary_row(row)
