from __future__ import annotations

from dataclasses import asdict

import pytest

from aiconfigurator.sdk.backends.cb_simulator.forward_descriptor import (
    compiled_body_runtime_key_from_nccl_summary_rows,
)


def _base_kwargs() -> dict[str, object]:
    return {
        "source": "phase39_nccl_trace",
        "scenario": "10k2k_b32_bt8192",
        "phase": "mixed",
        "topology_key": "tp4dp2moetp1ep8",
        "tp": 4,
        "dp": 2,
        "ep": 8,
        "world_size": 8,
        "forward_regime": "NONE:248",
        "tokens_padded": 248,
        "tokens_actual": 241,
        "cudagraph_runtime_mode": "NONE",
        "moe_module": "DeepseekV2MoE",
        "moe_kernel": "wna16",
        "moe_hidden": 7168,
        "moe_intermediate": 2048,
        "moe_experts": 384,
        "moe_topk": 8,
        "moe_dtype": "bfloat16",
        "tuning_config_loaded": False,
        "fallback": True,
    }


def test_compiled_body_key_uses_comm_candidates_without_latency_fields() -> None:
    key = compiled_body_runtime_key_from_nccl_summary_rows(
        [
            {
                "comm_candidate": "tp_comm_candidate",
                "valid_for_default": "False",
                "perf_database": "False",
            },
            {
                "comm_candidate": "ep_or_global_comm_candidate",
                "valid_for_default": "False",
                "perf_database": "False",
            },
            {
                "comm_candidate": "compiled_comm_envelope_unknown",
                "valid_for_default": "False",
                "perf_database": "False",
            },
        ],
        **_base_kwargs(),
    )

    values = asdict(key)
    assert key.tp_comm_candidate is True
    assert key.ep_or_global_comm_candidate is True
    assert key.unknown_comm_present is True
    assert key.valid_for_default is False
    assert key.perf_database is False
    assert key.diagnostic_only is True
    assert key.compiled_body is True
    for forbidden in (
        "profiled_cuda_time_ms",
        "nccl_trace_line_count",
        "pre_slot_sync_ms",
        "residual_ms",
        "throughput_ratio",
    ):
        assert forbidden not in values


def test_compiled_body_key_rejects_default_or_perf_rows() -> None:
    with pytest.raises(ValueError, match="valid_for_default must be false"):
        compiled_body_runtime_key_from_nccl_summary_rows(
            [
                {
                    "comm_candidate": "tp_comm_candidate",
                    "valid_for_default": "true",
                    "perf_database": "False",
                }
            ],
            **_base_kwargs(),
        )

    with pytest.raises(ValueError, match="perf_database must be false"):
        compiled_body_runtime_key_from_nccl_summary_rows(
            [
                {
                    "comm_candidate": "tp_comm_candidate",
                    "valid_for_default": "False",
                    "perf_database": "1",
                }
            ],
            **_base_kwargs(),
        )


def test_compiled_body_key_rejects_missing_or_unknown_candidate() -> None:
    with pytest.raises(ValueError, match="missing comm_candidate"):
        compiled_body_runtime_key_from_nccl_summary_rows(
            [{"valid_for_default": "False", "perf_database": "False"}],
            **_base_kwargs(),
        )

    with pytest.raises(ValueError, match="unknown comm_candidate"):
        compiled_body_runtime_key_from_nccl_summary_rows(
            [
                {
                    "comm_candidate": "tp_guess",
                    "valid_for_default": "False",
                    "perf_database": "False",
                }
            ],
            **_base_kwargs(),
        )


def test_compiled_body_key_rejects_empty_summary_rows() -> None:
    with pytest.raises(ValueError, match="no NCCL summary rows"):
        compiled_body_runtime_key_from_nccl_summary_rows([], **_base_kwargs())
