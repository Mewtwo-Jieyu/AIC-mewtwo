from __future__ import annotations

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk.perf_source import (
    PerfSourceRecord,
    validate_performance_result_sources,
)
from aiconfigurator.sdk.performance_result import PerformanceResult


SHA256 = "a" * 64


def _exact_source() -> PerfSourceRecord:
    return PerfSourceRecord.measured_exact(
        source_id="gemm_perf",
        data_file_sha256=SHA256,
        query_key={"m": 16, "n": 32, "k": 64},
        row_key={"m": 16, "n": 32, "k": 64},
    )


def test_performance_result_arithmetic_preserves_and_transforms_sources() -> None:
    left = PerformanceResult(2.0, energy=4.0, sources=(_exact_source(),))
    right = PerformanceResult(
        3.0,
        energy=6.0,
        sources=(
            PerfSourceRecord.structural(
                formula_id="unit_formula",
                formula_inputs={"tokens": 8},
            ),
        ),
    )

    combined = (left + right) * 2 / 4

    assert float(combined) == pytest.approx(2.5)
    assert combined.energy == pytest.approx(5.0)
    assert [source.source_type for source in combined.sources] == [
        "measured_exact",
        "structural",
    ]
    assert [transform.operation for transform in combined.sources[0].transforms] == [
        "multiply",
        "divide",
    ]
    assert [transform.factor for transform in combined.sources[0].transforms] == [2.0, 4.0]


def test_performance_result_subtraction_preserves_both_sources() -> None:
    left = PerformanceResult(5.0, energy=10.0, sources=(_exact_source(),))
    right_source = PerfSourceRecord.structural(
        formula_id="right_term_v1",
        formula_inputs={"tokens": 4},
    )
    right = PerformanceResult(2.0, energy=4.0, sources=(right_source,))

    result = left - right

    assert isinstance(result, PerformanceResult)
    assert float(result) == pytest.approx(3.0)
    assert result.energy == pytest.approx(6.0)
    assert result.sources[0] == _exact_source()
    assert result.sources[1].source_id == right_source.source_id
    assert result.sources[1].transforms[-1].operation == "multiply"
    assert result.sources[1].transforms[-1].factor == -1.0


def test_performance_result_reverse_subtraction_preserves_sources() -> None:
    result = 10.0 - PerformanceResult(2.0, energy=4.0, sources=(_exact_source(),))

    assert isinstance(result, PerformanceResult)
    assert float(result) == pytest.approx(8.0)
    assert result.energy == pytest.approx(-4.0)
    assert result.sources[0].source_id == _exact_source().source_id
    assert result.sources[0].transforms[-1].operation == "multiply"
    assert result.sources[0].transforms[-1].factor == -1.0
    assert result.sources[-1].formula_id == "unapproved_numeric_offset"
    assert result.sources[-1].approved is False


def test_performance_result_negation_preserves_sources() -> None:
    result = -PerformanceResult(2.0, energy=4.0, sources=(_exact_source(),))

    assert isinstance(result, PerformanceResult)
    assert float(result) == pytest.approx(-2.0)
    assert result.energy == pytest.approx(-4.0)
    assert result.sources[0].source_id == _exact_source().source_id
    assert result.sources[0].transforms[-1].operation == "multiply"
    assert result.sources[0].transforms[-1].factor == -1.0


def test_nonzero_naked_offset_is_unapproved_structural_source() -> None:
    result = PerformanceResult(2.0, sources=(_exact_source(),)) + 0.25

    assert result.sources[-1].formula_id == "unapproved_numeric_offset"
    assert result.sources[-1].approved is False
    with pytest.raises(ValueError, match="unapproved structural formula"):
        validate_performance_result_sources(result)


def test_zero_offset_does_not_create_a_source() -> None:
    result = PerformanceResult(2.0, sources=(_exact_source(),)) + 0

    assert result.sources == (_exact_source(),)


def test_nonzero_result_without_sources_fails_strict_validation() -> None:
    with pytest.raises(ValueError, match="non-zero performance result has no sources"):
        validate_performance_result_sources(PerformanceResult(1.0))


def test_unknown_source_type_fails_fast() -> None:
    with pytest.raises(ValueError, match="unknown performance source type"):
        PerfSourceRecord(source_type="guessed", source_id="bad")


def test_measured_interp_rejects_query_outside_finite_domain() -> None:
    with pytest.raises(ValueError, match="outside measured interpolation domain"):
        PerfSourceRecord.measured_interp(
            source_id="mla_perf",
            data_file_sha256=SHA256,
            query_key={"batch": 33},
            domain={"batch": (1, 32)},
            support_keys=({"batch": 16}, {"batch": 32}),
        )


def test_measured_interp_rejects_support_outside_finite_domain() -> None:
    with pytest.raises(ValueError, match="support field batch=64.*outside"):
        PerfSourceRecord.measured_interp(
            source_id="mla_perf",
            data_file_sha256=SHA256,
            query_key={"batch": 16},
            domain={"batch": (1, 32)},
            support_keys=({"batch": 16}, {"batch": 64}),
        )


def test_calibrated_source_rejects_scope_mismatch() -> None:
    source = PerfSourceRecord.calibrated(
        source_id="phase397v_int4_wo_moe",
        original_source="moe_roofline_v1",
        anchor_id="phase397l_tp8ep8_decode_moe_expert_gemm",
        scale=0.6063,
        scope={"model": "moonshotai/Kimi-K2.5", "moe_ep_size": 8},
    )

    with pytest.raises(ValueError, match="calibration scope mismatch"):
        source.validate_scope({"model": "other/model", "moe_ep_size": 8})


def test_corrected_runtime_grid_is_structural_not_measured(monkeypatch) -> None:
    database = PerfDatabase.__new__(PerfDatabase)
    monkeypatch.setattr(database, "_perf_data_sha256", lambda _filename: SHA256)
    table = {
        1: {
            1: {1: {"latency": 1.0}, 2: {"latency": 2.0}},
            2: {1: {"latency": 3.0}, 2: {"latency": 4.0}},
        },
        2: {
            1: {1: {"latency": 5.0}, 2: {"latency": 6.0}},
            2: {1: {"latency": 7.0}, 2: {"latency": 8.0}},
        },
    }

    source = database._runtime_grid_interp_3d_source(
        source_id="runtime_grid",
        filename=common.PerfDataFilename.gemm,
        query_key={"m": 1, "n": 1, "k": 1},
        coordinates=("m", "n", "k"),
        values=(1, 1, 1),
        table=table,
        fixed_support={"quant_mode": "float16"},
        method="cubic",
    )

    assert source is not None
    assert source.source_type == "structural"
    assert source.formula_id == "corrected_runtime_grid_interp_3d_v1"
    inputs = source.to_dict()["formula_inputs"]
    assert inputs["data_file_sha256"] == SHA256
    assert inputs["method"] == "cubic"
    assert inputs["runtime_support_points"]
    assert inputs["runtime_support_points"][0]["key"]["quant_mode"] == "float16"
    assert inputs["runtime_support_points"][0]["value"]["latency"] == 1.0


def test_one_dimensional_table_distinguishes_exact_interp_and_extrapolation(monkeypatch) -> None:
    database = PerfDatabase.__new__(PerfDatabase)
    monkeypatch.setattr(database, "_perf_data_sha256", lambda _filename: SHA256)
    table = {1: 1.0, 3: 3.0}
    common_kwargs = {
        "source_id": "moe_perf",
        "filename": common.PerfDataFilename.moe,
        "coordinate": "num_tokens",
        "table": table,
        "fixed_support": {"quant_mode": "int4_wo"},
        "inner_only": False,
    }

    exact = database._measured_interp_1d_source(
        query_key={"num_tokens": 1},
        value=1,
        **common_kwargs,
    )
    interpolated = database._measured_interp_1d_source(
        query_key={"num_tokens": 2},
        value=2,
        **common_kwargs,
    )
    extrapolated = database._measured_interp_1d_source(
        query_key={"num_tokens": 4},
        value=4,
        **common_kwargs,
    )

    assert exact is not None and exact.source_type == "measured_exact"
    assert interpolated is not None and interpolated.source_type == "measured_interp"
    assert extrapolated is not None and extrapolated.source_type == "structural"
    assert extrapolated.formula_id == "measured_grid_extrapolation_1d_v1"
