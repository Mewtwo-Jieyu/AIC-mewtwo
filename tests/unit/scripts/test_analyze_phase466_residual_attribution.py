from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_CSV = (
    REPO_ROOT
    / "docs"
    / "iter_gap_investigation"
    / "phase463_six_point_latency_recollect.csv"
)
ATTRIBUTION_CSV = (
    REPO_ROOT
    / "docs"
    / "iter_gap_investigation"
    / "phase466_residual_attribution.csv"
)
REPORT_MD = (
    REPO_ROOT
    / "docs"
    / "iter_gap_investigation"
    / "phase466_residual_attribution.md"
)


def _load_module():
    path = REPO_ROOT / "scripts" / "analyze_phase466_residual_attribution.py"
    spec = importlib.util.spec_from_file_location(
        "analyze_phase466_residual_attribution", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_loads_only_formal_failed_phase463_rows() -> None:
    phase466 = _load_module()

    rows = phase466.load_failed_scenarios(SOURCE_CSV)

    assert [row["scenario"] for row in rows] == list(phase466.FAILED_SCENARIOS)
    assert all(row["role"] == "formal" for row in rows)
    assert all(row["throughput_gate"] == "fail" for row in rows)
    assert all(row["default_readiness"] == "No-Go" for row in rows)


def test_builds_traceable_numeric_evidence() -> None:
    phase466 = _load_module()
    rows = phase466.load_failed_scenarios(SOURCE_CSV)

    evidence = {
        row["scenario"]: row for row in phase466.build_scenario_evidence(rows)
    }

    dp2_32k = evidence["K2.5-tp4ep8dp2-32k3k"]
    assert dp2_32k["throughput_abs_error_pct"] == pytest.approx(21.6845308575)
    assert dp2_32k["prefill_reqs_sim_over_real_delta_pct"] == pytest.approx(
        20.3944274315
    )
    assert dp2_32k["decode_reqs_sim_over_real_delta_pct"] == pytest.approx(
        0.0313963872
    )

    dp2_bt = evidence["K2.5-tp4ep8dp2-8k2k-bt65536"]
    assert dp2_bt["tpot_sim_over_real"] == pytest.approx(0.9982526785)
    assert dp2_bt["prefill_reqs_sim_over_real_delta_pct"] == pytest.approx(
        53.8636218131
    )
    assert dp2_bt["decode_reqs_sim_over_real_delta_pct"] == pytest.approx(
        4.9778501356
    )


def test_attribution_matrix_is_falsifiable_and_inconclusive() -> None:
    phase466 = _load_module()
    evidence = phase466.build_scenario_evidence(
        phase466.load_failed_scenarios(SOURCE_CSV)
    )

    rows = phase466.load_attribution_rows(ATTRIBUTION_CSV)
    phase466.validate_attribution_rows(rows, evidence)

    assert len(rows) == 9
    assert {row["candidate"] for row in rows} == set(phase466.CANDIDATES)
    assert {row["outcome"] for row in rows} == {"INCONCLUSIVE"}
    assert all(row["existing_support"] for row in rows)
    assert all(row["existing_counterevidence"] for row in rows)
    assert all(row["missing_observation"] for row in rows)
    assert all(row["phase466b_required_fields"] for row in rows)
    assert all(row["expected_signal"] for row in rows)
    assert all(row["disproof_condition"] for row in rows)
    assert all(row["diagnostic_only"] == "true" for row in rows)
    assert all(row["valid_for_default"] == "false" for row in rows)
    assert all(row["perf_database"] == "false" for row in rows)
    assert all(row["default_readiness"] == "No-Go" for row in rows)


def test_checked_in_outputs_are_consistent_and_lf_only() -> None:
    phase466 = _load_module()
    phase466.validate_outputs(SOURCE_CSV, ATTRIBUTION_CSV, REPORT_MD)

    assert b"\r" not in ATTRIBUTION_CSV.read_bytes()
    with ATTRIBUTION_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    markdown = REPORT_MD.read_text(encoding="utf-8")

    assert len(rows) == 9
    assert "结论：`INCONCLUSIVE`" in markdown
    assert "Phase462 composition logging v2/v3/v4" in markdown
    assert "13.8486% / 8.7915% / 8.2043%" in markdown
    assert "DP2-bt65536 off/on" in markdown
    for scenario in phase466.FAILED_SCENARIOS:
        assert scenario in markdown
    for candidate in phase466.CANDIDATES:
        assert candidate in markdown


def test_validator_rejects_numeric_drift(tmp_path: Path) -> None:
    phase466 = _load_module()
    with ATTRIBUTION_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0])
    rows[0]["throughput_abs_error_pct"] = "999"
    drifted = tmp_path / "drifted.csv"
    with drifted.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="numeric_mismatch"):
        phase466.validate_outputs(SOURCE_CSV, drifted, REPORT_MD)


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("evidence_sources", "unknown evidence", "evidence_sources"),
        ("expected_signal", "unrelated text", "hypothesis_contract"),
        ("existing_counterevidence", "unrelated text", "hypothesis_contract"),
        ("disproof_condition", "unrelated text", "hypothesis_contract"),
    ],
)
def test_validator_rejects_unstructured_hypothesis_mutations(
    tmp_path: Path,
    field: str,
    replacement: str,
    error: str,
) -> None:
    phase466 = _load_module()
    with ATTRIBUTION_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0])
    rows[0][field] = replacement
    mutated = tmp_path / "mutated.csv"
    with mutated.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match=error):
        phase466.validate_outputs(SOURCE_CSV, mutated, REPORT_MD)
