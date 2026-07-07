import csv

import pytest

from scripts.analyze_phase435_serving_state_extract import build_phase435_rows


def _write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_phase435_extracts_scoped_curves_and_prefill_gate(tmp_path):
    phase433 = tmp_path / "phase433.csv"
    _write_csv(
        phase433,
        [
            {
                "row_type": "summary",
                "scenario": "K2.5-tp4ep8dp2-32k3k",
                "category": "",
                "category_real_ms_per_step": "",
                "category_excess_ms_per_step": "",
                "target_missing_ms": "100.0",
            },
            {
                "row_type": "category",
                "scenario": "K2.5-tp4ep8dp2-32k3k",
                "category": "ep_a2a",
                "category_real_ms_per_step": "80.0",
                "category_excess_ms_per_step": "55.0",
                "target_missing_ms": "100.0",
            },
            {
                "row_type": "category",
                "scenario": "K2.5-tp4ep8dp2-32k3k",
                "category": "moe_gemm_or_aux",
                "category_real_ms_per_step": "70.0",
                "category_excess_ms_per_step": "40.0",
                "target_missing_ms": "100.0",
            },
        ],
    )
    phase429 = tmp_path / "phase429.csv"
    _write_csv(
        phase429,
        [
            {
                "row_type": "summary",
                "scenario": "K2.5-tp4ep8dp2-8k2k",
                "window": "",
                "step_type": "",
                "decode_batch": "",
                "category": "",
                "real_cuda_ms_per_rank": "",
                "phase426_decode_gap_share": "0.35",
            },
            {
                "row_type": "step",
                "scenario": "K2.5-tp4ep8dp2-8k2k",
                "window": "w0_prefill",
                "step_type": "prefill_or_mixed",
                "decode_batch": "1",
                "category": "",
                "real_cuda_ms_per_rank": "",
                "phase426_decode_gap_share": "0.35",
            },
            {
                "row_type": "step",
                "scenario": "K2.5-tp4ep8dp2-8k2k",
                "window": "w0_prefill",
                "step_type": "prefill_or_mixed",
                "decode_batch": "4",
                "category": "",
                "real_cuda_ms_per_rank": "",
                "phase426_decode_gap_share": "0.35",
            },
            {
                "row_type": "step_category",
                "scenario": "K2.5-tp4ep8dp2-8k2k",
                "window": "w0_prefill",
                "step_type": "prefill_or_mixed",
                "decode_batch": "1",
                "category": "tp_or_dp_allreduce",
                "real_cuda_ms_per_rank": "6.0",
                "phase426_decode_gap_share": "0.35",
            },
            {
                "row_type": "step_category",
                "scenario": "K2.5-tp4ep8dp2-8k2k",
                "window": "w1_decode_c16",
                "step_type": "decode_only",
                "decode_batch": "8",
                "category": "ep_a2a",
                "real_cuda_ms_per_rank": "9.0",
                "phase426_decode_gap_share": "0.35",
            },
        ],
    )
    phase426 = tmp_path / "phase426.csv"
    _write_csv(
        phase426,
        [
            {
                "row_type": "summary",
                "scenario": "K2.5-tp4ep8dp2-8k2k",
                "decode_gap_ms_mean": "16.5",
            }
        ],
    )

    rows = build_phase435_rows(phase429_csv=phase429, phase433_csv=phase433, phase426_csv=phase426)

    curve_rows = [row for row in rows if row["row_type"] == "serving_curve"]
    assert any(
        row["phase"] == "mixed_prefill"
        and row["bucket_tokens"] == 32000
        and row["decode_batch"] == 8
        and row["category"] == "ep_a2a"
        and row["latency_ms"] == pytest.approx(80.0)
        for row in curve_rows
    )
    assert any(
        row["phase"] == "mixed_prefill"
        and row["bucket_tokens"] == 8000
        and row["decode_batch"] == 4
        and row["category"] == "collective_other"
        and row["latency_ms"] == pytest.approx(6.0)
        for row in curve_rows
    )
    gate = next(row for row in rows if row["row_type"] == "consistency_gate" and row["phase"] == "mixed_prefill")
    assert gate["reconstruction_gate"] == "passed"
    assert gate["reconstruction_error_pct"] == pytest.approx(5.0)
    decode_gate = next(row for row in rows if row["row_type"] == "consistency_gate" and row["phase"] == "decode")
    assert decode_gate["target_missing_ms"] == pytest.approx(16.5)
