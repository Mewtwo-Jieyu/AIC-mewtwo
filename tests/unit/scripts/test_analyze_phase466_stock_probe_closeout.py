from __future__ import annotations

import csv
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_DIR = (
    REPO_ROOT
    / "docs"
    / "iter_gap_investigation"
    / "phase466_stock_probe_gate_closeout"
)
OVERHEAD_GATE = EVIDENCE_DIR / "overhead_gate.json"
PHASE_RESULT = EVIDENCE_DIR / "phase466_result.json"
CSV_PATH = EVIDENCE_DIR / "phase466_stock_probe_gate_closeout.csv"
REPORT_PATH = EVIDENCE_DIR / "phase466_stock_probe_gate_closeout.md"


def _load_module():
    path = REPO_ROOT / "scripts" / "analyze_phase466_stock_probe_closeout.py"
    assert path.is_file(), path
    spec = importlib.util.spec_from_file_location(
        "analyze_phase466_stock_probe_closeout", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _raw_evidence():
    return (
        json.loads(OVERHEAD_GATE.read_text(encoding="utf-8")),
        json.loads(PHASE_RESULT.read_text(encoding="utf-8")),
    )


def test_checked_in_closeout_is_valid() -> None:
    phase466 = _load_module()

    summary = phase466.validate_outputs()

    assert summary["off_mean"] == pytest.approx(843.0636334251724)
    assert summary["off_cv_pct"] == pytest.approx(4.958372701756978)
    assert summary["on_mean"] == pytest.approx(851.1261125176809)
    assert summary["on_cv_pct"] == pytest.approx(3.0107890672781927)
    assert summary["geometric_mean_ratio"] == pytest.approx(1.0102413271460076)
    assert summary["confidence_interval_ratio"] == pytest.approx(
        [0.9460820776552333, 1.0787515831640606]
    )
    assert summary["second_run_faster_count"] == 5


@pytest.mark.parametrize("pair_index", range(6))
def test_analyzer_rejects_any_pair_throughput_change(pair_index: int) -> None:
    phase466 = _load_module()
    gate, result = _raw_evidence()
    gate["pairs"][pair_index]["off_output_tok_s"] += 1.0

    with pytest.raises(ValueError, match="pair_ratio"):
        phase466.validate_evidence(gate, result)


def test_analyzer_rejects_modified_run_order(tmp_path: Path) -> None:
    phase466 = _load_module()
    rows = list(csv.DictReader(CSV_PATH.read_text(encoding="utf-8").splitlines()))
    fieldnames = list(rows[0])
    rows[0]["run_order"] = "ON/OFF"
    mutated = tmp_path / "mutated.csv"
    with mutated.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="csv_mismatch"):
        phase466.validate_outputs(csv_path=mutated)


def test_analyzer_rejects_modified_confidence_interval() -> None:
    phase466 = _load_module()
    gate, result = _raw_evidence()
    gate["confidence_interval_ratio"][0] = 0.98

    with pytest.raises(ValueError, match="confidence_interval_ratio"):
        phase466.validate_evidence(gate, result)


def test_analyzer_rejects_non_inconclusive_gate() -> None:
    phase466 = _load_module()
    gate, result = _raw_evidence()
    gate["status"] = "PASS"

    with pytest.raises(ValueError, match="gate_status"):
        phase466.validate_evidence(gate, result)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "DIAGNOSTIC_COMPLETE"),
        ("gate_status", "PASS"),
        ("diagnostic_only", False),
        ("valid_for_default", True),
        ("perf_database", True),
        ("default_readiness", "Go"),
    ],
)
def test_analyzer_rejects_modified_terminal_flags(field: str, value: object) -> None:
    phase466 = _load_module()
    gate, result = _raw_evidence()
    result[field] = value

    with pytest.raises(ValueError, match="result_contract"):
        phase466.validate_evidence(gate, result)


def test_analyzer_rejects_any_formal_scenario() -> None:
    phase466 = _load_module()
    gate, result = _raw_evidence()
    result["formal_scenarios"] = ["K2.5-tp4ep8dp2-32k3k"]

    with pytest.raises(ValueError, match="formal_scenarios"):
        phase466.validate_evidence(gate, result)


def test_analyzer_rejects_declared_formal_directory(tmp_path: Path) -> None:
    phase466 = _load_module()
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "formal").mkdir()
    (evidence / "overhead_gate.json").write_bytes(OVERHEAD_GATE.read_bytes())
    (evidence / "phase466_result.json").write_bytes(PHASE_RESULT.read_bytes())

    with pytest.raises(ValueError, match="formal_directory_present"):
        phase466.validate_outputs(evidence_dir=evidence)


def test_analyzer_rejects_report_drift(tmp_path: Path) -> None:
    phase466 = _load_module()
    mutated = tmp_path / "mutated.md"
    mutated.write_text(
        REPORT_PATH.read_text(encoding="utf-8").replace(
            "不能解释为 probe 开销约 1%", "probe 开销约 1%"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="report_mismatch"):
        phase466.validate_outputs(report_path=mutated)


def test_raw_evidence_hash_drift_fails_closed(tmp_path: Path) -> None:
    phase466 = _load_module()
    gate = deepcopy(json.loads(OVERHEAD_GATE.read_text(encoding="utf-8")))
    gate["pairs"][0]["on_output_tok_s"] += 1.0
    mutated = tmp_path / "overhead_gate.json"
    mutated.write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="raw_sha256"):
        phase466.validate_outputs(overhead_gate_path=mutated)
