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
