from __future__ import annotations

from dataclasses import dataclass

import pytest

import scripts.analyze_phase462_dual_replica_metric_consistency as analysis


@dataclass
class _Request:
    osl: int
    finish_ms: float


def test_full_window_metric_matches_bench_numerator_and_wall_definition() -> None:
    requests = [_Request(osl=10, finish_ms=100.0), _Request(osl=10, finish_ms=200.0)]

    metric = analysis.full_window_metric(requests, num_gpus=2)

    assert metric.output_tokens == 20
    assert metric.wall_ms == 200.0
    assert metric.throughput_tok_s_gpu == pytest.approx(50.0)


def test_first_token_count_difference_is_isolated() -> None:
    requests = [_Request(osl=10, finish_ms=200.0), _Request(osl=10, finish_ms=200.0)]

    full = analysis.full_window_metric(requests, num_gpus=2)
    decode_only = analysis.full_window_metric(
        requests, num_gpus=2, include_prefill_sample=False
    )

    assert full.output_tokens - decode_only.output_tokens == 2
    assert full.throughput_tok_s_gpu / decode_only.throughput_tok_s_gpu == pytest.approx(
        10 / 9
    )


def test_legacy_dp_assembly_preserves_per_gpu_throughput() -> None:
    result = analysis.legacy_dp_assembly(
        tp_group_throughput=400.0,
        tensor_parallel_size=4,
        data_parallel_size=2,
    )

    assert result.global_throughput == 800.0
    assert result.total_gpus == 8
    assert result.throughput_tok_s_gpu == 100.0


def test_definition_audit_rejects_full_wall_vs_trimmed_steady_window() -> None:
    verdict = analysis.definition_verdict(
        real_window="full_client_wall",
        sim_window="warmup_trimmed_replacement_plateau",
        real_token_definition="completion_tokens_including_first_sample",
        sim_token_definition="decode_steps_excluding_prefill_sample",
    )

    assert verdict == "measurement_fidelity_bug"


def test_order_dependent_attribution_is_not_reported_as_unique_fraction() -> None:
    note = analysis.order_dependency_note(
        forward=(0.0718359725, 0.0386542911),
        reverse=(0.086, 0.0244902636),
    )

    assert "path_dependent" in note
    assert "not_unique_causal_fraction" in note
