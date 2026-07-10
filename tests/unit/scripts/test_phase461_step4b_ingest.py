#!/usr/bin/env python3
"""Data-integrity tests for Phase461 Step4b PerfDB ingestion."""

from __future__ import annotations

import csv
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PERFDB_ROOT = REPO_ROOT / "src/aiconfigurator/systems/data/h200_sxm/vllm/0.19.0"


def test_mla_bad_row_is_replaced_only_by_phase461_measurement() -> None:
    path = PERFDB_ROOT / "generation_mla_perf.txt"
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    matches = [
        row
        for row in rows
        if row["kernel_source"] == "vllm_flash_attn_mla"
        and row["mla_dtype"] == "float16"
        and row["kv_cache_dtype"] == "float16"
        and row["num_heads"] == "8"
        and row["batch_size"] == "8"
        and row["isl"] == "1"
        and row["tp_size"] == "16"
        and row["step"] == "32767"
    ]

    assert len(matches) == 1
    assert math.isclose(
        float(matches[0]["latency"]),
        0.1346773306528727,
        rel_tol=0.0,
        abs_tol=1e-15,
    )


def _phase461_tp8_rows() -> list[dict[str, str]]:
    path = PERFDB_ROOT / "vllm_serving_state_perf.txt"
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    return [
        row
        for row in rows
        if row["kernel_source"] == "phase461_b2b_event_timing"
        and row["provenance"]
        == "phase461_tp8_32k_event_step_bucket_85400fe5_anchor_1ccfc363"
    ]


def test_tp8_32k_rows_are_scoped_and_exclude_rejected_cell() -> None:
    rows = _phase461_tp8_rows()

    assert len(rows) == 54
    assert {
        (row["topology"], row["phase"], row["row_kind"], row["max_num_batched_tokens"])
        for row in rows
    } == {("tp8ep8", "mixed_prefill", "forward_total", "32000")}
    assert ("92", "13") not in {
        (row["bucket_tokens"], row["decode_batch"]) for row in rows
    }


def test_tp8_32k_rows_preserve_measured_anchor_values() -> None:
    rows = {
        (int(row["bucket_tokens"]), int(row["decode_batch"])): float(row["latency"])
        for row in _phase461_tp8_rows()
    }

    assert math.isclose(rows[(26, 13)], 25.138463974, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(rows[(32000, 10)], 2427.190185547, rel_tol=0.0, abs_tol=1e-9)


def test_phase461_does_not_ingest_tp8_bt65536_rows() -> None:
    assert all(
        row["max_num_batched_tokens"] != "65536" for row in _phase461_tp8_rows()
    )
