from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.inference_summary import InferenceSummary
from aiconfigurator.sdk.perf_source import PerfSourceRecord


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_module():
    path = REPO_ROOT / "scripts" / "analyze_phase467_perf_sources.py"
    spec = importlib.util.spec_from_file_location("analyze_phase467_perf_sources", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _point(**overrides):
    values = {
        "name": "scenario",
        "moe_tp": 1,
        "moe_ep": 8,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_nonzero_operator_without_source_is_materialized() -> None:
    phase467 = _load_module()
    summary = InferenceSummary(SimpleNamespace())
    summary.set_context_source_map({"context_moe": ()})

    rows = phase467.audit_summary_sources(_point(), summary)

    assert rows[0]["source_type"] == "missing"
    assert rows[0]["operation"] == "context_moe"


def test_unknown_operator_source_is_materialized() -> None:
    phase467 = _load_module()
    summary = InferenceSummary(SimpleNamespace())
    summary.set_context_source_map({"context_moe": (object(),)})

    rows = phase467.audit_summary_sources(_point(), summary)

    assert rows[0]["source_type"] == "unknown"
    assert rows[0]["source_id"] == "object"


def test_calibration_scope_mismatch_fails_fast() -> None:
    phase467 = _load_module()
    summary = InferenceSummary(SimpleNamespace())
    source = PerfSourceRecord.calibrated(
        source_id="phase397v",
        original_source="moe_roofline_v1",
        anchor_id="anchor",
        scale=0.6063,
        scope={
            "system": "h200_sxm",
            "backend": "vllm",
            "version": "0.19.0",
            "model": "moonshotai/Kimi-K2.5",
            "moe_tp_size": 16,
            "moe_ep_size": 1,
        },
    )
    summary.set_context_source_map({"context_moe": (source,)})

    with pytest.raises(ValueError, match="calibration scope mismatch"):
        phase467.audit_summary_sources(_point(moe_tp=1, moe_ep=8), summary)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("hidden_size", 4096),
        ("inter_size", 1024),
        ("topk", 4),
        ("num_experts", 256),
        ("quant_mode", "fp8"),
    ],
)
def test_calibration_scope_uses_actual_model_values(field, bad_value) -> None:
    phase467 = _load_module()
    summary = InferenceSummary(SimpleNamespace())
    scope = {
        "system": "h200_sxm",
        "backend": "vllm",
        "version": "0.19.0",
        "model": "moonshotai/Kimi-K2.5",
        "hidden_size": 7168,
        "inter_size": 2048,
        "topk": 8,
        "num_experts": 384,
        "moe_tp_size": 1,
        "moe_ep_size": 8,
        "quant_mode": "int4_wo",
    }
    scope[field] = bad_value
    source = PerfSourceRecord.calibrated(
        source_id="phase397v",
        original_source="moe_roofline_v1",
        anchor_id="anchor",
        scale=0.6063,
        scope=scope,
    )
    summary.set_context_source_map({"context_moe": (source,)})
    model = SimpleNamespace(
        model_path="moonshotai/Kimi-K2.5",
        _hidden_size=7168,
        _moe_inter_size=2048,
        _topk=8,
        _num_experts=384,
        config=SimpleNamespace(
            moe_tp_size=1,
            moe_ep_size=8,
            moe_quant_mode=common.MoEQuantMode.int4_wo,
        ),
    )
    database = SimpleNamespace(system="h200_sxm", backend="vllm", version="0.19.0")

    with pytest.raises(ValueError, match="calibration scope mismatch"):
        phase467.audit_summary_sources(
            _point(),
            summary,
            model=model,
            database=database,
        )


def test_report_keeps_no_go_boundaries() -> None:
    phase467 = _load_module()
    numeric_rows = [{"scenario": name} for name in phase467.EXPECTED_SIM_OUTPUT_TOK_S_GPU]
    source_rows = [
        {
            "scenario": "scenario",
            "phase": "context",
            "operation": "context_moe",
            "source_type": "structural",
            "source_id": "source",
            "anchor_id": "",
            "scale": "",
            "scope": "{}",
        }
    ]

    charge_rows = [
        {
            "scenario": "scenario",
            "entry_index": 0,
            "total_ms": 1.0,
            "registered_charge_ms": 1.0,
            "valid_charge_ms": 1.0,
            "missing_source_charge_ms": 0.0,
            "unknown_source_charge_ms": 0.0,
            "unapproved_source_charge_ms": 0.0,
            "reconciliation_delta_ms": 0.0,
            "status": "PASS",
        }
    ]

    report = phase467.build_report(numeric_rows, source_rows, charge_rows)

    assert report["diagnostic_only"] is True
    assert report["valid_for_default"] is False
    assert report["perf_database"] is False
    assert report["default_aic"] == "No-Go"
    assert report["source_coverage"] == 1.0
    assert report["unknown_sources"] == []


def test_unregistered_cost_cannot_report_full_coverage() -> None:
    phase467 = _load_module()
    numeric_rows = [{"scenario": "scenario"}]
    source_rows = [
        {
            "scenario": "scenario",
            "phase": "context",
            "operation": "registered",
            "source_type": "structural",
            "source_id": "source",
            "anchor_id": "",
            "scale": "",
            "scope": "{}",
        }
    ]
    charge_rows = [
        {
            "scenario": "scenario",
            "entry_index": 0,
            "total_ms": 2.0,
            "registered_charge_ms": 1.0,
            "valid_charge_ms": 1.0,
            "missing_source_charge_ms": 0.0,
            "unknown_source_charge_ms": 0.0,
            "unapproved_source_charge_ms": 0.0,
            "reconciliation_delta_ms": 1.0,
            "status": "RECONCILIATION_FAILURE",
        }
    ]

    report = phase467.build_report(numeric_rows, source_rows, charge_rows)

    assert report["source_coverage"] == pytest.approx(0.5)
    assert report["reconciliation_failures"] == charge_rows
    assert report["status"] == "FAIL"


def test_missing_and_unknown_operators_do_not_count_as_sourced() -> None:
    phase467 = _load_module()
    numeric_rows = [{"scenario": "scenario"}]
    source_rows = [
        {
            "scenario": "scenario",
            "phase": "context",
            "operation": "valid",
            "source_type": "structural",
            "source_id": "source",
            "anchor_id": "",
            "scale": "",
            "scope": "{}",
        },
        {
            "scenario": "scenario",
            "phase": "context",
            "operation": "missing",
            "source_type": "missing",
            "source_id": "",
            "anchor_id": "",
            "scale": "",
            "scope": "{}",
        },
        {
            "scenario": "scenario",
            "phase": "generation",
            "operation": "unknown",
            "source_type": "unknown",
            "source_id": "object",
            "anchor_id": "",
            "scale": "",
            "scope": "{}",
        },
    ]

    report = phase467.build_report(numeric_rows, source_rows, [])

    assert report["charged_operator_count"] == 3
    assert report["charged_operator_with_source_count"] == 1


def test_operator_with_mixed_valid_and_unknown_sources_is_not_sourced() -> None:
    phase467 = _load_module()
    source_rows = [
        {
            "scenario": "scenario",
            "phase": "context",
            "operation": "mixed",
            "source_type": "structural",
            "source_id": "source",
            "anchor_id": "",
            "scale": "",
            "scope": "{}",
        },
        {
            "scenario": "scenario",
            "phase": "context",
            "operation": "mixed",
            "source_type": "unknown",
            "source_id": "object",
            "anchor_id": "",
            "scale": "",
            "scope": "{}",
        },
    ]

    report = phase467.build_report([{"scenario": "scenario"}], source_rows, [])

    assert report["charged_operator_count"] == 1
    assert report["charged_operator_with_source_count"] == 0


def test_main_returns_nonzero_after_writing_failed_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    phase467 = _load_module()

    def fake_run(out_dir: Path) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "phase467_source_audit.json").write_text(
            '{"status":"FAIL"}\n',
            encoding="utf-8",
        )
        return {"status": "FAIL"}

    monkeypatch.setattr(phase467, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyze_phase467_perf_sources.py", "--out-dir", str(tmp_path)],
    )

    assert phase467.main() == 1
    assert (tmp_path / "phase467_source_audit.json").is_file()
