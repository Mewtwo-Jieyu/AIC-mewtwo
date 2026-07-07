from __future__ import annotations

from pathlib import Path

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import _query_vllm_ep8_alltoall_fallback
from aiconfigurator.sdk.perf_database import PerfDatabase


REAL_SYSTEMS_ROOT = Path(__file__).resolve().parents[4] / "src/aiconfigurator/systems"
PHASE397L_EP8_ANCHOR_MS_PER_LAYER = 8.0201 / 60.0


def _database() -> PerfDatabase:
    return PerfDatabase("h200_sxm", "vllm", "0.19.0", str(REAL_SYSTEMS_ROOT))


def _query_int4_moe(db: PerfDatabase, *, moe_tp: int, moe_ep: int, mode=None):
    return db.query_moe(
        num_tokens=128,
        hidden_size=7168,
        inter_size=2048,
        topk=8,
        num_experts=384,
        moe_tp_size=moe_tp,
        moe_ep_size=moe_ep,
        quant_mode=common.MoEQuantMode.int4_wo,
        workload_distribution="power_law_1.01",
        database_mode=mode,
    )


def test_phase397v_int4_wo_calibrated_sol_reproduces_ep8_anchor() -> None:
    db = _database()

    result = _query_int4_moe(db, moe_tp=1, moe_ep=8)

    assert float(result) == pytest.approx(PHASE397L_EP8_ANCHOR_MS_PER_LAYER, rel=1e-9)


def test_phase417_int4_wo_large_context_uses_measured_moe_perf_when_covered() -> None:
    db = _database()
    table = db._moe_data[common.MoEQuantMode.int4_wo]["power_law_1.01"][8][384][7168][2048][1][8]
    expected = table[32768]["latency"] if isinstance(table[32768], dict) else table[32768]

    result = db.query_moe(
        num_tokens=32768,
        hidden_size=7168,
        inter_size=2048,
        topk=8,
        num_experts=384,
        moe_tp_size=1,
        moe_ep_size=8,
        quant_mode=common.MoEQuantMode.int4_wo,
        workload_distribution="power_law_1.01",
        is_context=True,
    )

    assert float(result) == pytest.approx(float(expected), rel=1e-9)


def test_phase417_int4_wo_decode_anchor_keeps_calibrated_sol() -> None:
    db = _database()

    result = db.query_moe(
        num_tokens=128,
        hidden_size=7168,
        inter_size=2048,
        topk=8,
        num_experts=384,
        moe_tp_size=1,
        moe_ep_size=8,
        quant_mode=common.MoEQuantMode.int4_wo,
        workload_distribution="power_law_1.2",
        is_context=False,
    )

    assert float(result) == pytest.approx(PHASE397L_EP8_ANCHOR_MS_PER_LAYER, rel=1e-9)


def test_phase431_int4_wo_decode_uses_phase431_measured_distribution_when_present() -> None:
    db = _database()
    workload_distribution = "unit_test_power_law"
    phase431_distribution = f"phase431_decode_{workload_distribution}"
    measured = db._moe_data[common.MoEQuantMode.int4_wo][phase431_distribution][8][384][7168][2048][1][8]
    measured[8] = {"latency": 0.04, "power": 0.0, "energy": 0.0, "kernel_source": "phase431_int4_wo_decode"}
    measured[128] = {"latency": 0.28, "power": 0.0, "energy": 0.0, "kernel_source": "phase431_int4_wo_decode"}

    result = db.query_moe(
        num_tokens=64,
        hidden_size=7168,
        inter_size=2048,
        topk=8,
        num_experts=384,
        moe_tp_size=1,
        moe_ep_size=8,
        quant_mode=common.MoEQuantMode.int4_wo,
        workload_distribution=workload_distribution,
        is_context=False,
    )

    assert float(result) == pytest.approx(0.152, rel=1e-9)
    assert float(result) != pytest.approx(PHASE397L_EP8_ANCHOR_MS_PER_LAYER, rel=1e-9)


def test_phase431_int4_wo_decode_uses_committed_phase431_rows() -> None:
    db = _database()

    result = db.query_moe(
        num_tokens=64,
        hidden_size=7168,
        inter_size=2048,
        topk=8,
        num_experts=384,
        moe_tp_size=1,
        moe_ep_size=8,
        quant_mode=common.MoEQuantMode.int4_wo,
        workload_distribution="power_law_1.01",
        is_context=False,
    )

    assert float(result) == pytest.approx(0.38183471679687503, rel=1e-9)
    assert float(result) > PHASE397L_EP8_ANCHOR_MS_PER_LAYER


def test_phase431_vllm_ep8_a2a_decode_uses_measured_curve_inside_coverage() -> None:
    db = _database()

    result = db.query_vllm_ep8_a2a_decode(
        bucket_tokens=64,
        hidden_size=7168,
        topk=8,
        moe_ep_size=8,
    )

    assert float(result) == pytest.approx(0.0852, rel=1e-9)


def test_phase431_ep8_alltoall_fallback_uses_measured_curve_before_byte_model() -> None:
    db = _database()

    measured = _query_vllm_ep8_alltoall_fallback(
        db,
        bucket_tokens=64,
        hidden_size=7168,
        topk=8,
        scale_factor=60,
    )

    assert float(measured) == pytest.approx(0.0852 * 60, rel=1e-9)


def test_phase435_serving_state_query_matches_exact_scope_inside_coverage() -> None:
    db = _database()

    result = db.query_vllm_serving_state(
        model="kimi-k2.5",
        topology="tp4dp2ep8",
        phase="mixed_prefill",
        category="ep_a2a",
        bucket_tokens=32000,
        decode_batch=8,
        hidden_size=7168,
        topk=8,
        moe_ep_size=8,
        quant_runtime="CompressedTensorsWNA16MarlinMoEMethod",
    )

    assert result is not None
    assert float(result) == pytest.approx(1917.743178, rel=1e-6)


def test_phase435_serving_state_query_returns_none_for_other_topology_or_out_of_range() -> None:
    db = _database()

    wrong_topology = db.query_vllm_serving_state(
        model="kimi-k2.5",
        topology="tp8ep8",
        phase="mixed_prefill",
        category="ep_a2a",
        bucket_tokens=32000,
        decode_batch=8,
        hidden_size=7168,
        topk=8,
        moe_ep_size=8,
        quant_runtime="CompressedTensorsWNA16MarlinMoEMethod",
    )
    out_of_range = db.query_vllm_serving_state(
        model="kimi-k2.5",
        topology="tp4dp2ep8",
        phase="decode",
        category="ep_a2a",
        bucket_tokens=999,
        decode_batch=999,
        hidden_size=7168,
        topk=8,
        moe_ep_size=8,
        quant_runtime="CompressedTensorsWNA16MarlinMoEMethod",
    )

    assert wrong_topology is None
    assert out_of_range is None


def test_phase397v_int4_wo_calibrated_sol_scales_tp16_ep1_from_roofline() -> None:
    db = _database()
    sol_ep8 = _query_int4_moe(db, moe_tp=1, moe_ep=8, mode=common.DatabaseMode.SOL_FULL)[0]
    sol_tp16 = _query_int4_moe(db, moe_tp=16, moe_ep=1, mode=common.DatabaseMode.SOL_FULL)[0]
    expected = PHASE397L_EP8_ANCHOR_MS_PER_LAYER * sol_tp16 / sol_ep8

    result = _query_int4_moe(db, moe_tp=16, moe_ep=1)

    assert float(result) == pytest.approx(expected, rel=1e-9)


def test_phase417_int4_wo_large_context_missing_family_uses_calibrated_sol() -> None:
    db = _database()
    sol = db.query_moe(
        num_tokens=32768,
        hidden_size=7168,
        inter_size=2048,
        topk=8,
        num_experts=384,
        moe_tp_size=16,
        moe_ep_size=1,
        quant_mode=common.MoEQuantMode.int4_wo,
        workload_distribution="power_law_1.01",
        is_context=True,
        database_mode=common.DatabaseMode.SOL_FULL,
    )[0]
    sol_ep8 = _query_int4_moe(db, moe_tp=1, moe_ep=8, mode=common.DatabaseMode.SOL_FULL)[0]
    expected = PHASE397L_EP8_ANCHOR_MS_PER_LAYER * sol / sol_ep8

    result = db.query_moe(
        num_tokens=32768,
        hidden_size=7168,
        inter_size=2048,
        topk=8,
        num_experts=384,
        moe_tp_size=16,
        moe_ep_size=1,
        quant_mode=common.MoEQuantMode.int4_wo,
        workload_distribution="power_law_1.01",
        is_context=True,
    )

    assert float(result) == pytest.approx(expected, rel=1e-9)


def test_phase397v_sol_full_still_returns_uncalibrated_roofline_components() -> None:
    db = _database()

    sol_time, sol_math, sol_mem = _query_int4_moe(
        db,
        moe_tp=1,
        moe_ep=8,
        mode=common.DatabaseMode.SOL_FULL,
    )

    assert sol_time == pytest.approx(sol_mem)
    assert sol_time > PHASE397L_EP8_ANCHOR_MS_PER_LAYER
    assert sol_mem > sol_math
