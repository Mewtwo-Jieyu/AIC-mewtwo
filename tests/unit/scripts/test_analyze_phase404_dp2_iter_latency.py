import csv
from types import SimpleNamespace

import pytest

from scripts import analyze_phase404_dp2_iter_latency as phase404


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sim_row(scenario: str, total_ms: float, dispatch_ms: float = 2.0):
    return SimpleNamespace(
        scenario=scenario,
        sim_running_global=20 if "32k3k" in scenario else 104,
        per_replica_batch=10 if "32k3k" in scenario else 52,
        mid_kv_len=33500 if "32k3k" in scenario else 9000,
        bucketed_kv_len=33792 if "32k3k" in scenario else 9216,
        sim_decode_ms_per_iter=total_ms,
        gen_attention_ms=10.0,
        gen_moe_compute_ms=8.0,
        gen_dispatch_ms=dispatch_ms,
        gen_gemm_ms=5.0,
        gen_other_non_attention_ms=max(total_ms - 23.0 - dispatch_ms, 0.0),
        overhead_ms=0.0,
    )


def test_phase404_rejects_gen_op_gap_when_active_decode_matches_sim(tmp_path):
    phase403_csv = tmp_path / "phase403.csv"
    _write_csv(
        phase403_csv,
        [
            {
                "scenario": "K2.5-tp4ep8dp2-8k2k",
                "tp": "4",
                "dp": "2",
                "ep": "8",
                "isl": "8000",
                "osl": "2000",
                "max_num_batched_tokens": "8000",
                "phase403_output_tok_s_global": "1112.0",
                "phase403_output_tok_s_gpu": "139.0",
                "real_running_global_max": "103",
                "real_running_global_mean": "62",
                "real_running_global_p50": "66",
                "artifact_dir": "fake/8k",
            },
            {
                "scenario": "K2.5-tp4ep8dp2-32k3k",
                "tp": "4",
                "dp": "2",
                "ep": "8",
                "isl": "32000",
                "osl": "3000",
                "max_num_batched_tokens": "32000",
                "phase403_output_tok_s_global": "392.5",
                "phase403_output_tok_s_gpu": "49.06",
                "real_running_global_max": "20",
                "real_running_global_mean": "15.5",
                "real_running_global_p50": "18",
                "artifact_dir": "fake/32k",
            },
        ],
    )
    phase400_csv = tmp_path / "phase400.csv"
    _write_csv(
        phase400_csv,
        [
            {
                "scenario": "K2.5-tp8ep8-32k3k",
                "tp": "8",
                "dp": "1",
                "ep": "8",
                "isl": "32000",
                "osl": "3000",
                "max_num_batched_tokens": "32000",
                "output_tok_s_global": "400.0",
                "output_tok_s_gpu": "50.0",
                "serve_running_reqs_max": "14",
                "serve_running_reqs_mean": "12.6",
            }
        ],
    )
    active_stats = {
        "K2.5-tp4ep8dp2-8k2k": phase404.ActiveDecodeStats(
            intervals=4,
            ms_per_iter_median=40.0,
            ms_per_iter_p10=38.0,
            ms_per_iter_p90=42.0,
        ),
        "K2.5-tp4ep8dp2-32k3k": phase404.ActiveDecodeStats(
            intervals=5,
            ms_per_iter_median=26.0,
            ms_per_iter_p10=25.0,
            ms_per_iter_p90=27.0,
        ),
    }

    rows = phase404.build_phase404_rows(
        sim_rows=[
            _sim_row("K2.5-tp4ep8dp2-8k2k", 39.0),
            _sim_row("K2.5-tp4ep8dp2-32k3k", 27.0),
            _sim_row("K2.5-tp8ep8-32k3k", 35.0),
        ],
        phase403_csv=phase403_csv,
        phase400_csv=phase400_csv,
        active_stats=active_stats,
    )
    by_scenario = {row["scenario"]: row for row in rows}

    dp32 = by_scenario["K2.5-tp4ep8dp2-32k3k"]
    assert dp32["row_type"] == "dp2_attribution"
    assert dp32["real_active_decode_ms_per_iter"] == "26.000000"
    assert dp32["active_decode_residual_ms"] == "-1.000000"
    assert float(dp32["aggregate_implied_residual_ms"]) > 20.0
    assert dp32["attribution_verdict"] == "reject_gen_op_latency_gap"
    assert dp32["phase405_target"] == "dp_decode_duty_cycle_or_queueing"

    tp8 = by_scenario["K2.5-tp8ep8-32k3k"]
    assert tp8["row_type"] == "tp8_crosscheck"
    assert tp8["attribution_verdict"] == "tp8_control_no_dp_only_iter_gap"


def test_phase404_writer_rejects_default_upgrade(tmp_path):
    rows = []
    for scenario, row_type in (
        ("K2.5-tp4ep8dp2-8k2k", "dp2_attribution"),
        ("K2.5-tp4ep8dp2-32k3k", "dp2_attribution"),
        ("K2.5-tp8ep8-32k3k", "tp8_crosscheck"),
    ):
        row = {field: "" for field in phase404.CSV_FIELDS}
        row.update(
            {
                "source": phase404.SOURCE,
                "scenario": scenario,
                "row_type": row_type,
                "gpu_allowed": "false",
                "ssh_allowed": "false",
                "runtime_modified": "false",
                "perf_database": "false",
                "valid_for_default": "false",
                "diagnostic_only": "true",
                "default_readiness": "No-Go",
                "real_active_decode_ms_per_iter": "25.0" if row_type == "dp2_attribution" else "",
            }
        )
        rows.append(row)
    rows[0]["valid_for_default"] = "true"

    with pytest.raises(ValueError, match="valid_for_default"):
        phase404.write_phase404_csv(tmp_path / "bad.csv", rows)
