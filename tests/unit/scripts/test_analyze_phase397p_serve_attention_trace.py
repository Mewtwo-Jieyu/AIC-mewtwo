from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts import analyze_phase397p_serve_attention_trace as phase397p


SAMPLE = """-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
void cutlass::device_kernel<flash::enable_sm90_or_la...         0.00%       0.000us         0.00%       0.000us       0.000us     411.282ms        42.48%     411.282ms     259.812us          1583
void cutlass::device_kernel<flash::FlashAttnFwdCombi...         0.00%       0.000us         0.00%       0.000us       0.000us       7.020ms         0.73%       7.020ms       4.435us          1583
                    _vllm_fa3_C::get_scheduler_metadata         0.01%     247.921us         0.03%     566.446us      22.658us      51.585us         0.01%      51.585us       2.063us            25
void flash::prepare_varlen_num_blocks_kernel<8, fals...         0.00%       0.000us         0.00%       0.000us       0.000us      51.585us         0.01%      51.585us       2.063us            25
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
"""


def _find(rows: list[dict[str, str]], **match: str) -> dict[str, str]:
    for row in rows:
        if all(row[key] == value for key, value in match.items()):
            return row
    raise AssertionError(match)


def test_parse_profiler_table_and_select_roles(tmp_path: Path) -> None:
    path = tmp_path / "profiler_out_0.txt"
    path.write_text(SAMPLE, encoding="utf-8")
    rows = phase397p.parse_profiler_rows(path)
    assert len(rows) == 4
    main = phase397p.select_role(rows, "fa3_main")
    assert main.cuda_time_avg_us == pytest.approx(259.812)
    assert main.cuda_total_us == pytest.approx(411_282.0)
    assert main.calls == 1583
    assert phase397p.select_role(rows, "fa3_combine").cuda_time_avg_us == pytest.approx(4.435)
    assert phase397p.select_role(rows, "scheduler_metadata").calls == 25
    assert phase397p.select_role(rows, "prepare_varlen_num_blocks").calls == 25


def test_phase397p_real_trace_confirms_split_kv_scheduler_gap() -> None:
    rows = phase397p.build_rows()
    rank_rows = [row for row in rows if row["row_type"] == "rank_kernel"]
    assert len(rank_rows) == 64
    assert len([row for row in rows if row["row_type"] == "config_summary"]) == 2

    tp8 = _find(rows, row_type="config_summary", config="tp8ep8-8k2k")
    assert tp8["verdict"] == "fa3_split_kv_scheduler_trace_confirmed"
    assert 250.0 <= float(tp8["mean_cuda_time_avg_us"]) <= 270.0
    assert float(tp8["phase397o_over_serve"]) > 1.8
    assert tp8["fa3_main_present"] == "true"
    assert tp8["fa3_combine_present"] == "true"
    assert tp8["scheduler_metadata_present"] == "true"
    assert tp8["prepare_varlen_present"] == "true"

    decision = _find(rows, row_type="decision")
    assert decision["verdict"] == "confirmed_fa3_split_kv_scheduler_harness_gap"
    assert decision["default_readiness"] == "No-Go"
    assert decision["perf_database"] == "false"
    assert decision["runtime_modified"] == "false"


def test_phase397p_main_writes_csv_and_md(tmp_path: Path) -> None:
    out_csv = tmp_path / "phase397p.csv"
    out_md = tmp_path / "phase397p.md"
    assert phase397p.main(["--out-csv", str(out_csv), "--out-md", str(out_md)]) == 0
    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 67
    assert _find(rows, row_type="decision")["valid_for_default"] == "false"
    text = out_md.read_text(encoding="utf-8")
    assert "confirmed_fa3_split_kv_scheduler_harness_gap" in text
    assert "Default AIC remains No-Go" in text
