import csv

import pytest

from scripts import analyze_phase418_roofline_reconcile as phase418


COMMITTED_COLLECT_MOE = """
hidden_states = torch.randn([num_tokens, hidden_size], device=device)
topk_weights_list.append(topk_weights)
fused_marlin_moe(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    global_num_experts=num_experts,
    expert_map=expert_map,
)
"""


def _write_phase416_csv(path):
    fields = [
        "row_type",
        "scenario",
        "component",
        "phase",
        "returned_ms",
        "roofline_lower_bound_ms",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            [
                {
                    "row_type": "query_trace",
                    "scenario": phase418.DEFAULT_SCENARIO,
                    "component": "moe_compute",
                    "phase": "mixed_prefill",
                    "returned_ms": "352.717448",
                    "roofline_lower_bound_ms": "455.757015",
                },
                {
                    "row_type": "query_trace",
                    "scenario": phase418.DEFAULT_SCENARIO,
                    "component": "ep_dispatch_combine",
                    "phase": "mixed_prefill",
                    "returned_ms": "497.491058",
                    "roofline_lower_bound_ms": "489.335467",
                },
            ]
        )


def _write_phase414_csv(path):
    fields = ["source", "scenario", "prefill_real_mean_ms"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "source": "phase414_prefill_decode_audit",
                "scenario": phase418.DEFAULT_SCENARIO,
                "prefill_real_mean_ms": "4718.954849",
            }
        )


def _write_phase417_csv(path):
    fields = ["row_type", "scenario", "after_returned_ms"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "row_type": "summary",
                "scenario": phase418.DEFAULT_SCENARIO,
                "after_returned_ms": "1818.637996",
            }
        )


def test_phase418_reconciles_moe_roofline_gate_with_collector_provenance(tmp_path):
    phase414_csv = tmp_path / "phase414.csv"
    phase416_csv = tmp_path / "phase416.csv"
    phase417_csv = tmp_path / "phase417.csv"
    _write_phase414_csv(phase414_csv)
    _write_phase416_csv(phase416_csv)
    _write_phase417_csv(phase417_csv)

    rows = phase418.build_phase418_rows(
        phase414_csv=phase414_csv,
        phase416_csv=phase416_csv,
        phase417_csv=phase417_csv,
        committed_text=COMMITTED_COLLECT_MOE,
        dirty_diff="diff --git a/collector/vllm/collect_moe.py b/collector/vllm/collect_moe.py\n",
    )

    summary = [row for row in rows if row["row_type"] == "summary"][0]
    moe = [row for row in rows if row["row_type"] == "roofline" and row["component"] == "moe_compute"][0]

    assert moe["old_gate_status"] == "failed"
    assert moe["corrected_fp8_bound_ms"] == "85.454440"
    assert moe["corrected_bf16_bound_ms"] == "170.822563"
    assert moe["selected_bound_ms"] == "170.822563"
    assert moe["corrected_gate_status"] == "passed"
    assert moe["mfu_status"] == "reasonable"
    assert moe["fixed_mixed_step_ms"] == "1818.637996"
    assert moe["real_prefill_step_ms"] == "4718.954849"
    assert moe["remaining_real_over_fixed_ratio"] == "2.594774"
    assert moe["remaining_gap_statement"] == "mixed_prefill_still_approximately_2_6x_below_real_after_gate_unlock"
    assert moe["collector_committed_semantics"] == "num_tokens_token_rows_single_layer_ep_expert_map_no_dispatch"
    assert moe["collector_dirty_drift"] == "dirty_collect_moe_diff_present"
    assert moe["provenance_status"] == "aligned"
    assert summary["mechanism_verdict"] == "phase415_roofline_gate_wrong_not_moe_perf_table"
    assert summary["next_phase_target"] == "regenerate_phase415_416_417_and_anchor_revalidate"
    assert summary["valid_for_default"] == "false"
    assert summary["default_readiness"] == "No-Go"


def test_phase418_writer_rejects_runtime_or_default_claim(tmp_path):
    rows = [{field: "" for field in phase418.CSV_FIELDS}]
    rows[0].update(
        {
            "source": phase418.SOURCE,
            "row_type": "summary",
            "scenario": phase418.DEFAULT_SCENARIO,
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
        phase418.write_phase418_csv(tmp_path / "bad.csv", rows)

    rows[0]["runtime_modified"] = "false"
    rows[0]["valid_for_default"] = "true"
    with pytest.raises(ValueError, match="valid_for_default"):
        phase418.write_phase418_csv(tmp_path / "bad.csv", rows)
