import csv

import pytest

from scripts import analyze_phase417_moe_ep_charge_fix as phase417


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_phase417_reports_component_gate_fixed_after_roofline_reconcile(tmp_path):
    phase415 = tmp_path / "phase415.csv"
    phase416 = tmp_path / "phase416.csv"
    _write_csv(
        phase415,
        [
            "row_type",
            "scenario",
            "component",
            "sim_mean_ms",
            "roofline_lower_bound_ms",
            "roofline_gap_ms",
            "prefill_step_count",
        ],
        [
            {
                "row_type": "summary",
                "scenario": phase417.DEFAULT_SCENARIO,
                "component": "summary",
                "sim_mean_ms": "1818.637996",
                "roofline_lower_bound_ms": "347123.858370",
                "roofline_gap_ms": "0.000000",
                "prefill_step_count": "332",
            }
        ],
    )
    _write_csv(
        phase416,
        [
            "row_type",
            "scenario",
            "component",
            "phase",
            "scaled_tokens",
            "expected_tokens",
            "query_path",
            "returned_ms",
            "roofline_lower_bound_ms",
        ],
        [
            {
                "row_type": "query_trace",
                "scenario": phase417.DEFAULT_SCENARIO,
                "component": "moe_compute",
                "phase": "mixed_prefill",
                "scaled_tokens": "32000",
                "expected_tokens": "32000",
                "query_path": "moe_perf_lookup",
                "returned_ms": "352.717448",
                "roofline_lower_bound_ms": "170.822563",
            },
            {
                "row_type": "query_trace",
                "scenario": phase417.DEFAULT_SCENARIO,
                "component": "ep_dispatch_combine",
                "phase": "mixed_prefill",
                "scaled_tokens": "32000",
                "expected_tokens": "32000",
                "query_path": "ep8_alltoall_roofline",
                "returned_ms": "497.491058",
                "roofline_lower_bound_ms": "489.335467",
            },
        ],
    )

    rows = phase417.build_phase417_rows(phase415_csv=phase415, phase416_csv=phase416)
    summary = [row for row in rows if row["row_type"] == "summary"][0]
    moe = [row for row in rows if row["component"] == "moe_compute"][0]
    ep = [row for row in rows if row["component"] == "ep_dispatch_combine"][0]

    assert moe["status"] == "fixed"
    assert moe["remaining_undercharge_ms"] == "0.000000"
    assert ep["status"] == "fixed"
    assert summary["mechanism_verdict"] == "phase417_moe_ep_charge_fixed"
    assert summary["anchor_revalidation_status"] == "required_before_default"
    assert summary["roofline_lower_bound_ms"] == "1045.553790"
    assert summary["remaining_undercharge_ms"] == "0.000000"
    assert summary["valid_for_default"] == "false"
    assert summary["default_readiness"] == "No-Go"


def test_phase417_writer_rejects_default_claim(tmp_path):
    rows = [{field: "" for field in phase417.CSV_FIELDS}]
    rows[0].update(
        {
            "source": phase417.SOURCE,
            "row_type": "summary",
            "scenario": phase417.DEFAULT_SCENARIO,
            "phase405_penalty_read": "false",
            "gpu_allowed": "false",
            "ssh_allowed": "false",
            "runtime_modified": "true",
            "perf_database": "false",
            "valid_for_default": "true",
            "diagnostic_only": "true",
            "default_readiness": "No-Go",
        }
    )

    with pytest.raises(ValueError, match="valid_for_default"):
        phase417.write_phase417_csv(tmp_path / "bad.csv", rows)


def test_phase417_md_states_no_anchor_revalidation():
    rows = [
        {field: "" for field in phase417.CSV_FIELDS},
        {field: "" for field in phase417.CSV_FIELDS},
        {field: "" for field in phase417.CSV_FIELDS},
    ]
    rows[0].update(
        {
            "source": phase417.SOURCE,
            "row_type": "component",
            "scenario": phase417.DEFAULT_SCENARIO,
            "component": "moe_compute",
            "status": "remaining_under_roofline",
            "mechanism_verdict": "phase417_partial_fix_moe_table_still_under_roofline",
            "next_phase_target": "phase418_moe_prefill_table_or_roofline_reconciliation",
            "anchor_revalidation_status": "blocked_by_component_gate",
        }
    )
    rows[1].update(
        {
            "source": phase417.SOURCE,
            "row_type": "component",
            "scenario": phase417.DEFAULT_SCENARIO,
            "component": "ep_dispatch_combine",
            "status": "fixed",
            "mechanism_verdict": "phase417_partial_fix_moe_table_still_under_roofline",
            "next_phase_target": "phase418_moe_prefill_table_or_roofline_reconciliation",
            "anchor_revalidation_status": "blocked_by_component_gate",
        }
    )
    rows[2].update(
        {
            "source": phase417.SOURCE,
            "row_type": "summary",
            "scenario": phase417.DEFAULT_SCENARIO,
            "component": "summary",
            "after_returned_ms": "1818.637996",
            "roofline_lower_bound_ms": "1784.428860",
            "remaining_undercharge_ms": "34209.136259",
            "mechanism_verdict": "phase417_partial_fix_moe_table_still_under_roofline",
            "next_phase_target": "phase418_moe_prefill_table_or_roofline_reconciliation",
            "anchor_revalidation_status": "blocked_by_component_gate",
        }
    )
    for row in rows:
        row.update(
            {
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

    md = phase417.render_phase417_md(rows)

    assert "blocked_by_component_gate" in md
    assert "Default AIC: `No-Go`" in md
    assert "EP dispatch/combine is fixed" in md
    assert "mixed step roofline lower-bound per-step" in md


def test_phase417_md_states_anchor_revalidation_required_after_component_gate():
    rows = [
        {field: "" for field in phase417.CSV_FIELDS},
        {field: "" for field in phase417.CSV_FIELDS},
        {field: "" for field in phase417.CSV_FIELDS},
    ]
    rows[0].update(
        {
            "source": phase417.SOURCE,
            "row_type": "component",
            "scenario": phase417.DEFAULT_SCENARIO,
            "component": "moe_compute",
            "status": "fixed",
            "mechanism_verdict": "phase417_moe_ep_charge_fixed",
            "next_phase_target": "anchor_revalidate_phase400_phase403",
            "anchor_revalidation_status": "required_before_default",
        }
    )
    rows[1].update(
        {
            "source": phase417.SOURCE,
            "row_type": "component",
            "scenario": phase417.DEFAULT_SCENARIO,
            "component": "ep_dispatch_combine",
            "status": "fixed",
            "mechanism_verdict": "phase417_moe_ep_charge_fixed",
            "next_phase_target": "anchor_revalidate_phase400_phase403",
            "anchor_revalidation_status": "required_before_default",
        }
    )
    rows[2].update(
        {
            "source": phase417.SOURCE,
            "row_type": "summary",
            "scenario": phase417.DEFAULT_SCENARIO,
            "component": "summary",
            "after_returned_ms": "1818.637996",
            "roofline_lower_bound_ms": "1045.553790",
            "remaining_undercharge_ms": "0.000000",
            "mechanism_verdict": "phase417_moe_ep_charge_fixed",
            "next_phase_target": "anchor_revalidate_phase400_phase403",
            "anchor_revalidation_status": "required_before_default",
        }
    )
    for row in rows:
        row.update(
            {
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

    md = phase417.render_phase417_md(rows)

    assert "required_before_default" in md
    assert "clears the corrected Phase415 physical lower-bound check" in md
    assert "still below the Phase415 physical lower-bound check" not in md
