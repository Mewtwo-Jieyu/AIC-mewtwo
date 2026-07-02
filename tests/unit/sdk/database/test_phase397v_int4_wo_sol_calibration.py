from __future__ import annotations

from pathlib import Path

import pytest

from aiconfigurator.sdk import common
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


def test_phase397v_int4_wo_calibrated_sol_scales_tp16_ep1_from_roofline() -> None:
    db = _database()
    sol_ep8 = _query_int4_moe(db, moe_tp=1, moe_ep=8, mode=common.DatabaseMode.SOL_FULL)[0]
    sol_tp16 = _query_int4_moe(db, moe_tp=16, moe_ep=1, mode=common.DatabaseMode.SOL_FULL)[0]
    expected = PHASE397L_EP8_ANCHOR_MS_PER_LAYER * sol_tp16 / sol_ep8

    result = _query_int4_moe(db, moe_tp=16, moe_ep=1)

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
