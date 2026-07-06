import csv

import pytest

from scripts import analyze_phase415_prefill_charge_components as phase415


def test_phase415_flags_largest_roofline_undercharged_component():
    phase414_row = {
        "scenario": phase415.DEFAULT_SCENARIO,
        "steady_target_penalty": "1.35",
        "prefill_gap_ms": "900.0",
        "decode_gap_ms": "100.0",
        "target_missing_ms": "1000.0",
    }
    rows = phase415.build_rows_from_component_totals(
        scenario=phase415.DEFAULT_SCENARIO,
        artifact_dir="fake/artifact",
        phase414_row=phase414_row,
        mixed_components={
            "prefill_mla_attention": 400.0,
            "moe_compute": 50.0,
            "ep_dispatch_combine": 100.0,
            "tp_comm": 150.0,
            "other": 200.0,
        },
        decode_components={
            "decode_mla_attention": 6.0,
            "decode_ep_dispatch_combine": 9.0,
            "decode_other": 4.0,
        },
        roofline_lower_bounds={
            "prefill_mla_attention": 300.0,
            "moe_compute": 500.0,
            "ep_dispatch_combine": 350.0,
            "tp_comm": 80.0,
        },
        prefill_step_count=10,
    )

    summary = [row for row in rows if row["row_type"] == "summary"][0]
    moe = [
        row
        for row in rows
        if row["row_type"] == "mixed_component" and row["component"] == "moe_compute"
    ][0]
    ep = [
        row
        for row in rows
        if row["row_type"] == "mixed_component" and row["component"] == "ep_dispatch_combine"
    ][0]

    assert moe["roofline_status"] == "below_roofline"
    assert moe["roofline_gap_ms"] == "450.000000"
    assert ep["roofline_gap_ms"] == "250.000000"
    assert summary["mechanism_verdict"] == "moe_compute_undercharged"
    assert summary["reconstructed_missing_ms"] == "1000.000000"
    assert summary["consistency_gate"] == "passed"
    assert summary["phase405_penalty_read"] == "false"
    assert summary["valid_for_default"] == "false"
    assert summary["default_readiness"] == "No-Go"


def test_phase415_writer_rejects_runtime_or_default_claim(tmp_path):
    rows = [{field: "" for field in phase415.CSV_FIELDS}]
    rows[0].update(
        {
            "source": phase415.SOURCE,
            "row_type": "summary",
            "scenario": phase415.DEFAULT_SCENARIO,
            "phase405_penalty_read": "false",
            "gpu_allowed": "false",
            "ssh_allowed": "false",
            "runtime_modified": "true",
            "perf_database": "false",
            "valid_for_default": "false",
            "diagnostic_only": "true",
            "default_readiness": "No-Go",
        }
    )
    with pytest.raises(ValueError, match="runtime_modified"):
        phase415.write_phase415_csv(tmp_path / "bad.csv", rows)

    rows[0]["runtime_modified"] = "false"
    rows[0]["valid_for_default"] = "true"
    with pytest.raises(ValueError, match="valid_for_default"):
        phase415.write_phase415_csv(tmp_path / "bad.csv", rows)


def test_phase415_csv_roundtrip_keeps_component_rows(tmp_path):
    phase414_row = {
        "scenario": phase415.DEFAULT_SCENARIO,
        "steady_target_penalty": "1.35",
        "prefill_gap_ms": "900.0",
        "decode_gap_ms": "100.0",
        "target_missing_ms": "1000.0",
    }
    rows = phase415.build_rows_from_component_totals(
        scenario=phase415.DEFAULT_SCENARIO,
        artifact_dir="fake/artifact",
        phase414_row=phase414_row,
        mixed_components={"moe_compute": 50.0},
        decode_components={"decode_other": 10.0},
        roofline_lower_bounds={"moe_compute": 500.0},
        prefill_step_count=10,
    )
    out = tmp_path / "phase415.csv"
    phase415.write_phase415_csv(out, rows)
    with out.open(newline="") as f:
        loaded = list(csv.DictReader(f))
    assert [row["row_type"] for row in loaded] == [
        "mixed_component",
        "decode_component",
        "summary",
    ]


def test_phase415_moe_roofline_uses_ep_local_wna16_work() -> None:
    per_step = phase415._roofline_lower_bounds(1)

    assert per_step["moe_compute"] == pytest.approx(170.822563, rel=1e-6)
    assert per_step["ep_dispatch_combine"] == pytest.approx(489.335467, rel=1e-6)
