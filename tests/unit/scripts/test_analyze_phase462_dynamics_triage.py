#!/usr/bin/env python3
"""Tests for Phase462 Step1 report-only dynamics triage."""

from __future__ import annotations

import importlib.util
import csv
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase462_dynamics_triage.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_phase462_dynamics_triage", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_recompute_gate_uses_sim_only_wall_association() -> None:
    mod = _load_module()
    steps = [
        mod.AttributedStep(64_000, 30, 80.0, 63_970, 48_000),
        mod.AttributedStep(60_000, 30, 20.0, 59_970, 0),
        mod.AttributedStep(8_000, 40, 100.0, 7_960, 0),
    ]

    summary = mod.summarize_recompute_correlation(
        steps,
        sim_only_cells={(64_000, 30), (60_000, 30)},
        common_root_threshold=0.70,
    )

    assert summary["sim_only_wall_share"] == 0.5
    assert summary["recompute_associated_sim_only_wall_share"] == 0.8
    assert math.isclose(summary["recompute_token_share_in_sim_only"], 48_000 / 123_940)
    assert summary["verdict"] == "preemption_recompute_common_root_supported"


def test_interpolation_stencil_rejects_unreachable_mla_candidate() -> None:
    mod = _load_module()
    data = {
        8: {8: {16_384: 1.0, 32_768: 1.0, 65_536: 1.0}},
        16: {
            8: {16_384: 1.0, 32_768: 1.0, 65_536: 1.0},
            16: {16_384: 1.0, 32_768: 1.0, 65_536: 1.0},
            32: {16_384: 1.0, 32_768: 1.0, 65_536: 1.0},
        },
        32: {
            8: {16_384: 1.0, 32_768: 1.0, 65_536: 1.0},
            16: {16_384: 1.0, 32_768: 1.0, 65_536: 1.0},
        },
    }

    stencil = mod.interpolation_stencil(data, x=16, y=13, z=33_793)

    assert (16, 8, 32_768) in stencil
    assert (16, 16, 65_536) in stencil
    assert (16, 32, 32_768) not in stencil


def test_preemption_panorama_uses_one_ratio_definition() -> None:
    mod = _load_module()

    row = mod.preemption_comparison(
        scenario="scenario-a",
        real_preemptions=42,
        sim_preemptions=194,
        requests=512,
    )

    assert math.isclose(row["real_preemptions_per_request"], 42 / 512)
    assert math.isclose(row["sim_preemptions_per_request"], 194 / 512)
    assert math.isclose(row["sim_over_real"], 194 / 42)
    assert row["status"] == "sim_over_preempts"


def test_candidate_excess_bound_is_compared_with_observed_residual() -> None:
    mod = _load_module()

    bound = mod.candidate_excess_bound(
        measured_ms=0.766,
        expected_ms=0.145,
        layers=61,
        influence_count=1,
        reference_wall_ms=132_959.0,
        observed_residual_fraction=0.235,
    )

    assert math.isclose(bound["excess_wall_ms"], (0.766 - 0.145) * 61)
    assert bound["wall_share"] < 0.001
    assert bound["can_explain_residual"] is False


def test_heads16_candidate_coordinates_come_from_original_perfdb_row(tmp_path: Path) -> None:
    mod = _load_module()
    perfdb = tmp_path / "perfdb"
    perfdb.mkdir()
    (perfdb / "context_mla_perf.txt").write_text(
        "framework,version,device,op_name,kernel_source,mla_dtype,kv_cache_dtype,"
        "num_heads,batch_size,isl,tp_size,step,latency\n"
        "VLLM,0.19.0,NVIDIA H200,context_mla,vllm_flash_attn_mla,float16,float16,"
        "16,4,64,8,0,0.967\n",
        encoding="utf-8",
    )
    anomaly = tmp_path / "anomaly.csv"
    semantic = json.dumps(
        {
            "num_heads": "16",
            "mla_dtype": "float16",
            "kv_cache_dtype": "float16",
        }
    )
    with anomaly.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "active_six_scope",
                "file",
                "line",
                "semantic_key",
                "deviation_ratio",
                "expected_latency_ms",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "active_six_scope": "true",
                "file": "context_mla_perf.txt",
                "line": "2",
                "semantic_key": semantic,
                "deviation_ratio": "16.6",
                "expected_latency_ms": "0.058",
            }
        )
        writer.writerow(
            {
                "active_six_scope": "true",
                "file": "context_mla_perf.txt",
                "line": "2",
                "semantic_key": semantic,
                "deviation_ratio": "15.1",
                "expected_latency_ms": "0.064",
            }
        )

    candidates = mod.load_active_mla_candidates(anomaly, perfdb_root=perfdb)

    assert candidates == [
        mod.MlaCandidate("context_mla_perf.txt", 2, 16, 4, 64, 16.6, 0.967, 0.058)
    ]
