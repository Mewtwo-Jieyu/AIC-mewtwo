import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import analyze_phase430_bubble_mla_fix as phase430


def test_phase430_records_fix_a_and_decode_recollect_need():
    rows = phase430.build_phase430_rows()
    summary = next(row for row in rows if row["row_type"] == "summary")
    fix_a = next(row for row in rows if row["row_type"] == "fix_a" and row["category"] == "pure_prefill")
    moe = next(row for row in rows if row["row_type"] == "decode_slope" and row["category"] == "moe_gemm_or_aux")
    mla = next(row for row in rows if row["row_type"] == "decode_slope" and row["category"] == "mla_attention")
    validate = next(
        row
        for row in rows
        if row["row_type"] == "validate_after"
        and row["metric"] == "scenario_ratio"
        and row["scenario"] == "K2.5-tp4ep8dp2-8k2k"
    )

    assert summary["verdict"] == "phase430_fix_a_applied_fix_b_requires_gpu_recollect"
    assert fix_a["verdict"] == "fixed_pure_prefill_no_overlap_lane_drop"
    assert float(fix_a["sim_value"]) == 23.0
    assert float(fix_a["counterfactual_value"].split("_")[0]) == 15.0
    assert moe["verdict"] == "moe_decode_sol_slope_wrong_sign_recollect_required"
    assert float(moe["sim_value"]) < 0.0
    assert "moe_perf_table_slope" in moe["counterfactual_value"]
    assert mla["verdict"] == "mla_slope_near_real_but_joint_recollect_recommended"
    assert validate["verdict"] == "fail"
    assert validate["gap_value"] == "2.55"
    assert summary["valid_for_default"] == "false"
    assert summary["default_readiness"] == "No-Go"


def test_phase430_writer_rejects_default_claim():
    row = {field: "" for field in phase430.CSV_FIELDS}
    row.update(
        {
            "source": phase430.SOURCE,
            "row_type": "summary",
            "scenario": phase430.SCENARIO,
            "runtime_modified": "true",
            "perf_database": "false",
            "valid_for_default": "true",
            "diagnostic_only": "false",
            "default_readiness": "No-Go",
        }
    )
    with tempfile.TemporaryDirectory() as raw:
        try:
            phase430.write_phase430_csv(Path(raw) / "bad.csv", [row])
        except ValueError as exc:
            assert "valid_for_default" in str(exc)
        else:
            raise AssertionError("writer accepted valid_for_default=true")


if __name__ == "__main__":
    test_phase430_records_fix_a_and_decode_recollect_need()
    test_phase430_writer_rejects_default_claim()
