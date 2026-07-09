#!/usr/bin/env python3
"""Tests for Phase454 GPU batch acceptance analysis."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase454_gpu_batch_acceptance.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase454_gpu", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_event_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payloads = []
    for row in rows:
        base = {
            "schema": "phase446_graph_outer_event_v2",
            "measurement": "model_forward_cuda_event_outer_delayed",
            "dp_rank": 0,
            "tp_rank": None,
            "attention_busy_ms": None,
            "non_attn_busy_ms": None,
            "generation_tokens": row.get("generation_requests", 0),
            "num_tokens_padded": row.get("num_tokens_unpadded", row.get("ctx_tokens", 0)),
        }
        base.update(row)
        payloads.append(json.dumps(base))
    path.write_text("\n".join(payloads) + "\n", encoding="utf-8")


def test_clean_reference_rows_recompute_output_ratio(tmp_path: Path) -> None:
    mod = _load_module()
    validate_csv = tmp_path / "validate.csv"
    validate_csv.write_text(
        "\n".join(
            [
                "name,tp,dp,ep,max_bt,real_output_tok_s_gpu,sim_output_tok_s_gpu,error_ratio",
                "K2.5-tp4ep8dp2-8k2k,4,2,8,8000,137.716,156.48378768131056,1.1362789195250411",
            ]
        ),
        encoding="utf-8",
    )
    bench = tmp_path / "bench.json"
    _write_json(
        bench,
        {
            "input_len": 8000,
            "output_len": 2000,
            "total_tok_s": 6517.940648829252,
        },
    )

    rows = mod.clean_reference_rows(
        [mod.CleanReferenceSpec("K2.5-tp4ep8dp2-8k2k", bench, gpu_count=8)],
        validate_csv,
    )

    assert rows[0]["clean_real_output_tok_s_gpu"] == mod.approx(162.94851622073128)
    assert rows[0]["clean_error_ratio"] == mod.approx(1.0413124492652654)
    assert rows[0]["status"] == "clean_reference_collected"

    final_rows = mod.clean_reference_validate_rows(validate_csv, rows)

    assert final_rows[0]["real_output_tok_s_gpu"] == mod.approx(162.94851622073128)
    assert final_rows[0]["sim_output_tok_s_gpu"] == mod.approx(156.48378768131056)


def test_tp8_b2b_rows_use_tp8_topology_and_dp1_decode(tmp_path: Path) -> None:
    mod = _load_module()
    event = tmp_path / "tp8.jsonl"
    _write_event_rows(
        event,
        [
            {
                "ctx_tokens": 7990,
                "generation_requests": 10,
                "forward_busy_ms": 600.0,
                "num_tokens_unpadded": 8000,
                "cudagraph_mode": "NONE",
            },
            {
                "ctx_tokens": 0,
                "generation_requests": 64,
                "forward_busy_ms": 42.0,
                "num_tokens_unpadded": 64,
                "cudagraph_mode": "FULL",
            },
        ],
    )

    candidate_rows = mod.extract_b2b_rows(
        [
            mod.B2BSpec(
                scenario="K2.5-tp8ep8-8k2k",
                event_path=event,
                topology="tp8ep8",
                dp_size=1,
            )
        ]
    )

    assert {(row.topology, row.phase, row.bucket_tokens, row.decode_batch) for row in candidate_rows} == {
        ("tp8ep8", "mixed_prefill", 8000, 10),
        ("tp8ep8", "decode", 64, 64),
    }


def test_dp2_bt_decode_duplicate_is_blocked_but_large_mixed_is_accepted(tmp_path: Path) -> None:
    mod = _load_module()
    perfdb = tmp_path / "vllm_serving_state_perf.txt"
    perfdb.write_text(
        "\n".join(
            [
                ",".join(mod.PERFDB_FIELDS),
                "VLLM,0.19.0,NVIDIA H200,kimi-k2.5,tp4dp2ep8,decode,forward_total,forward_total,phase446,65536,16,16,7168,8,8,CompressedTensorsWNA16MarlinMoEMethod,20.0,old",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = [
        mod.PerfDBCandidate(
            scenario="K2.5-tp4ep8dp2-8k2k-bt65536",
            topology="tp4dp2ep8",
            phase="decode",
            row_kind="forward_total",
            category="forward_total",
            bucket_tokens=16,
            decode_batch=16,
            latency_ms=18.0,
            sample_count=10,
            provenance="new",
        ),
        mod.PerfDBCandidate(
            scenario="K2.5-tp4ep8dp2-8k2k-bt65536",
            topology="tp4dp2ep8",
            phase="mixed_prefill",
            row_kind="forward_total",
            category="forward_total",
            bucket_tokens=65536,
            decode_batch=7,
            latency_ms=9000.0,
            sample_count=10,
            provenance="new",
        ),
    ]

    decisions = mod.classify_perfdb_candidates(rows, perfdb)

    by_key = {(row["phase"], row["bucket_tokens"], row["decode_batch"]): row for row in decisions}
    assert by_key[("decode", 16, 16)]["status"] == "blocked_duplicate_key"
    assert by_key[("mixed_prefill", 65536, 7)]["status"] == "accepted"


def test_blocked_topology_is_reported_without_perfdb_acceptance(tmp_path: Path) -> None:
    mod = _load_module()
    perfdb = tmp_path / "vllm_serving_state_perf.txt"
    perfdb.write_text(",".join(mod.PERFDB_FIELDS) + "\n", encoding="utf-8")
    rows = [
        mod.PerfDBCandidate(
            scenario="K2.5-tp8ep8-8k2k",
            topology="tp8ep8",
            phase="decode",
            row_kind="forward_total",
            category="forward_total",
            bucket_tokens=64,
            decode_batch=64,
            latency_ms=42.0,
            sample_count=10,
            provenance="new",
        )
    ]

    decisions = mod.classify_perfdb_candidates(
        rows,
        perfdb,
        blocked_topology_reasons={"tp8ep8": "blocked_ab_regression_missing_kv_axis"},
    )

    assert decisions[0]["status"] == "blocked_ab_regression_missing_kv_axis"


def test_phase454_rows_already_in_perfdb_are_idempotent(tmp_path: Path) -> None:
    mod = _load_module()
    perfdb = tmp_path / "vllm_serving_state_perf.txt"
    perfdb.write_text(
        "\n".join(
            [
                ",".join(mod.PERFDB_FIELDS),
                "VLLM,0.19.0,NVIDIA H200,kimi-k2.5,tp4dp2ep8,mixed_prefill,forward_total,forward_total,phase454_b2b_event_timing,65536,65536,7,7168,8,8,CompressedTensorsWNA16MarlinMoEMethod,9000.0,phase454_b2b_event_step_bucket",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = [
        mod.PerfDBCandidate(
            scenario="K2.5-tp4ep8dp2-8k2k-bt65536",
            topology="tp4dp2ep8",
            phase="mixed_prefill",
            row_kind="forward_total",
            category="forward_total",
            bucket_tokens=65536,
            decode_batch=7,
            latency_ms=9000.0,
            sample_count=10,
            provenance="phase454_b2b_event_step_bucket",
        )
    ]

    decisions = mod.classify_perfdb_candidates(rows, perfdb)

    assert decisions[0]["status"] == "already_ingested"


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        test_clean_reference_rows_recompute_output_ratio(base)
        test_tp8_b2b_rows_use_tp8_topology_and_dp1_decode(base)
        test_dp2_bt_decode_duplicate_is_blocked_but_large_mixed_is_accepted(base)
        test_blocked_topology_is_reported_without_perfdb_acceptance(base)
        test_phase454_rows_already_in_perfdb_are_idempotent(base)
