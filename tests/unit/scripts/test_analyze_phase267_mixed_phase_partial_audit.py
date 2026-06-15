from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "analyze_phase267_mixed_phase_partial_audit.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase267_mixed_phase_partial_audit",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


EXPECTED_PAIR_KEYS = [
    "tp8_dp1_ep8:isl4000_osl2000_batch128:control_bt4000:holdout_bt65536",
    "tp4_dp2_ep8:isl4000_osl2000_batch128:control_bt4000:holdout_bt65536",
    "tp8_dp1_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536",
    "tp4_dp2_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536",
]

EXPECTED_VERDICTS = {
    EXPECTED_PAIR_KEYS[0]: "partial",
    EXPECTED_PAIR_KEYS[1]: "explains",
    EXPECTED_PAIR_KEYS[2]: "partial",
    EXPECTED_PAIR_KEYS[3]: "does_not_explain",
}


def _actual_phase264_csv() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "iter_gap_investigation"
        / "phase264_phase_mix_decode_batch.csv"
    )


def test_mixed_phase_partial_audit_outputs_four_pair_rows(tmp_path: Path) -> None:
    rows = analyzer.analyze_mixed_phase_partial_audit(_actual_phase264_csv())
    out_csv = tmp_path / "phase267.csv"
    out_doc = tmp_path / "phase267.md"

    analyzer.write_mixed_phase_partial_audit_csv(out_csv, rows)
    analyzer.write_mixed_phase_partial_audit_doc(out_doc, rows)

    assert [row["pair_key"] for row in rows] == EXPECTED_PAIR_KEYS
    assert {row["source"] for row in rows} == {"phase267_mixed_phase_partial_audit"}
    assert {row["next_mechanism"] for row in rows} == {"boundary_timeline_diagnostic"}
    assert {row["default_readiness"] for row in rows} == {"No-Go"}
    assert {row["diagnostic_only"] for row in rows} == {"true"}
    assert {row["valid_for_default"] for row in rows} == {"false"}
    assert {row["perf_database"] for row in rows} == {"false"}
    assert {row["pure_decode_p99_max_invariant"] for row in rows} == {"true"}
    assert {row["budget_ceiling_not_linear"] for row in rows} == {"true"}

    by_key = {row["pair_key"]: row for row in rows}
    assert {key: row["verdict"] for key, row in by_key.items()} == EXPECTED_VERDICTS

    tp8_4k = by_key[EXPECTED_PAIR_KEYS[0]]
    assert tp8_4k["output_ratio"] == "0.989048"
    assert tp8_4k["mixed_count_delta"] == "1"
    assert tp8_4k["mixed_p99_delta"] == "-480.000000"
    assert tp8_4k["mixed_max_delta"] == "-480"

    tp4_12k = by_key[EXPECTED_PAIR_KEYS[3]]
    assert tp4_12k["output_ratio"] == "1.339504"
    assert tp4_12k["mixed_count_delta"] == "0"
    assert tp4_12k["mixed_p99_delta"] == "570.000000"
    assert tp4_12k["mixed_max_delta"] == "570"
    assert tp4_12k["verdict"] == "does_not_explain"

    persisted = list(csv.DictReader(out_csv.open(newline="", encoding="utf-8")))
    assert len(persisted) == 4
    doc = out_doc.read_text(encoding="utf-8")
    assert "Mixed phase is partial-only" in doc
    assert "Default AIC | No-Go" in doc


def test_verdict_distribution_is_fixed() -> None:
    rows = analyzer.analyze_mixed_phase_partial_audit(_actual_phase264_csv())

    verdicts = [row["verdict"] for row in rows]

    assert verdicts.count("partial") == 2
    assert verdicts.count("explains") == 1
    assert verdicts.count("does_not_explain") == 1


def test_input_must_have_exact_phase264_rows(tmp_path: Path) -> None:
    src = _actual_phase264_csv()
    rows = list(csv.DictReader(src.open(newline="", encoding="utf-8")))
    truncated = tmp_path / "phase264_truncated.csv"
    with truncated.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows[:-1])

    with pytest.raises(ValueError, match="phase264 csv must contain 8 rows"):
        analyzer.analyze_mixed_phase_partial_audit(truncated)


def test_input_delta_change_fails_fast(tmp_path: Path) -> None:
    src = _actual_phase264_csv()
    rows = list(csv.DictReader(src.open(newline="", encoding="utf-8")))
    for row in rows:
        if row["scenario"] == "tp8ep8-4k2k-bt65536":
            row["mixed_iterations"] = "1"
    mutated = tmp_path / "phase264_mutated.csv"
    with mutated.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="mixed_count_delta mismatch"):
        analyzer.analyze_mixed_phase_partial_audit(mutated)


def test_pure_decode_invariant_fails_fast(tmp_path: Path) -> None:
    src = _actual_phase264_csv()
    rows = list(csv.DictReader(src.open(newline="", encoding="utf-8")))
    rows[0]["pure_decode_rank_sum_p99_decode_tokens"] = "127.000000"
    mutated = tmp_path / "phase264_bad_decode.csv"
    with mutated.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="pure decode p99/max must stay invariant"):
        analyzer.analyze_mixed_phase_partial_audit(mutated)


def test_flags_fail_fast(tmp_path: Path) -> None:
    src = _actual_phase264_csv()
    rows = list(csv.DictReader(src.open(newline="", encoding="utf-8")))
    rows[0]["valid_for_default"] = "true"
    mutated = tmp_path / "phase264_bad_flags.csv"
    with mutated.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="phase264 flag mismatch"):
        analyzer.analyze_mixed_phase_partial_audit(mutated)
