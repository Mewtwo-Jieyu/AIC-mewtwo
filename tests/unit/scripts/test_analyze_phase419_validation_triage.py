import csv

import pytest

from scripts import analyze_phase419_validation_triage as phase419


def _write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=phase419.INPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _row(name, sim, real=100.0, source="cb_sim"):
    return {
        "scenario": name,
        "tp": "4",
        "dp": "2",
        "ep": "8",
        "max_bt": "8000",
        "real_output_tok_s_gpu": f"{real:.6f}",
        "sim_output_tok_s_gpu": f"{sim:.6f}",
        "sim_real_ratio": f"{sim / real:.6f}",
        "error_ratio": f"{max(sim / real, real / sim):.6f}",
        "throughput_source": source,
    }


def test_phase419_join_classifies_and_flags_max_regression(tmp_path):
    before = tmp_path / "before.csv"
    after = tmp_path / "after.csv"
    _write_csv(
        before,
        [
            _row("improved", 250.0),
            _row("regressed", 120.0),
            _row("K2.5-tp4ep8dp2-8k2k-bt65536", 90.0),
        ],
    )
    _write_csv(
        after,
        [
            _row("improved", 125.0),
            _row("regressed", 200.0),
            _row("K2.5-tp4ep8dp2-8k2k-bt65536", 13.297872),
        ],
    )

    rows = phase419.build_phase419_rows(before_csv=before, after_csv=after)
    by_name = {row["scenario"]: row for row in rows if row["row_type"] == "config"}
    summary = [row for row in rows if row["row_type"] == "summary"][0]

    assert by_name["improved"]["classification"] == "improved"
    assert by_name["regressed"]["classification"] == "regressed"
    assert by_name["K2.5-tp4ep8dp2-8k2k-bt65536"]["classification"] == "regressed"
    assert by_name["K2.5-tp4ep8dp2-8k2k-bt65536"]["is_max_after_error"] == "true"
    assert by_name["K2.5-tp4ep8dp2-8k2k-bt65536"]["mechanism_hint"] == "bt65536_underprediction_regression"
    assert summary["mechanism_verdict"] == "fix_regression_bt65536_underprediction"
    assert summary["default_readiness"] == "No-Go"
    assert summary["valid_for_default"] == "false"


def test_phase419_writer_rejects_default_claim(tmp_path):
    rows = [{field: "" for field in phase419.OUTPUT_FIELDS}]
    rows[0].update(
        {
            "source": phase419.SOURCE,
            "row_type": "summary",
            "phase405_penalty_read": "false",
            "gpu_allowed": "false",
            "ssh_allowed": "false",
            "runtime_modified": "false",
            "perf_database": "false",
            "valid_for_default": "true",
            "diagnostic_only": "true",
            "default_readiness": "No-Go",
        }
    )

    with pytest.raises(ValueError, match="valid_for_default"):
        phase419.write_phase419_csv(tmp_path / "bad.csv", rows)


def test_phase419_markdown_reports_verdict(tmp_path):
    before = tmp_path / "before.csv"
    after = tmp_path / "after.csv"
    _write_csv(before, [_row("same", 100.0)])
    _write_csv(after, [_row("same", 100.0)])

    rows = phase419.build_phase419_rows(before_csv=before, after_csv=after)
    md = phase419.render_phase419_md(rows)

    assert "Phase419" in md
    assert "Default AIC: `No-Go`" in md
    assert "no_regression" in md
