from __future__ import annotations

import csv
import sys
import types
from pathlib import Path

from collector.vllm import run_phase397q_mla


def test_phase397q_default_probe_targets_block64_heads8_b128_s9001() -> None:
    specs = run_phase397q_mla.build_phase397q_instrument_specs()
    assert specs == [
        run_phase397q_mla.Phase397QMLASpec(
            local_num_heads=8,
            kv_cache_dtype="float16",
            batch_size=128,
            target_seq_len=9001,
            block_size=64,
            randomize_blocks=True,
            max_num_splits=None,
        )
    ]
    assert specs[0].tp_size == 16
    assert specs[0].input_len == 9000


def test_phase397q_row_records_profile_and_remains_diagnostic_only() -> None:
    spec = run_phase397q_mla.Phase397QMLASpec()
    result = {
        "target_backend": "FlashAttnMLAImpl",
        "kernel_entrypoint": "forward_mqa",
        "kernel_source": "vllm_flash_attn_mla",
        "eager6_ms": 0.486,
        "eager200_p50_ms": 0.300,
        "graph_p50_ms": 0.280,
        "graph_capture": True,
        "phase397q_profile_path": "/tmp/profile.txt",
        "phase397q_profile_iters": 1,
        "phase397q_full_cudagraph_metadata": True,
        "scheduler_metadata_present": True,
        "scheduler_metadata_shape": "513",
        "decode_max_num_splits": 32,
        "called_attention_kernel": True,
        "called_model_forward": False,
    }
    row = run_phase397q_mla.build_row(spec, result)
    assert row["source"] == "phase397q_mla_split_instrument"
    assert row["block_size"] == "64"
    assert row["phase397q_profile_path"] == "/tmp/profile.txt"
    assert row["phase397q_profile_iters"] == "1"
    assert row["phase397q_full_cudagraph_metadata"] == "true"
    assert row["scheduler_metadata_present"] == "true"
    assert row["scheduler_metadata_shape"] == "513"
    assert row["decode_max_num_splits"] == "32"
    assert row["valid_for_default"] == "false"
    assert row["perf_database"] == "false"
    assert row["default_readiness"] == "No-Go"
    assert list(row) == run_phase397q_mla.PHASE397Q_COLUMNS


def test_phase397q_collect_spec_forwards_block64_and_profile_args(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_attention_torch(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return {
            "target_backend": "FlashAttnMLAImpl",
            "kernel_entrypoint": "forward_mqa",
            "kernel_source": "vllm_flash_attn_mla",
            "eager6_ms": 0.4,
            "eager200_p50_ms": 0.3,
            "graph_p50_ms": 0.2,
            "graph_capture": True,
            "phase397q_profile_path": kwargs["phase397q_profile_path"],
            "phase397q_profile_iters": kwargs["phase397q_profile_iters"],
            "phase397q_full_cudagraph_metadata": kwargs["phase397q_full_cudagraph_metadata"],
            "scheduler_metadata_present": True,
            "scheduler_metadata_shape": "513",
            "decode_max_num_splits": 32,
            "called_attention_kernel": True,
            "called_model_forward": False,
        }

    monkeypatch.setitem(
        sys.modules,
        "collect_mla",
        types.SimpleNamespace(run_attention_torch=fake_run_attention_torch),
    )
    result = run_phase397q_mla.collect_spec(
        run_phase397q_mla.Phase397QMLASpec(),
        device="cuda:7",
        warmup_iters=11,
        measure_iters=22,
        profile_path="/tmp/p.txt",
        profile_iters=3,
    )
    assert result["phase397q_profile_path"] == "/tmp/p.txt"
    assert result["phase397q_profile_iters"] == 3
    assert calls[0]["args"][9] == 64
    assert calls[0]["kwargs"]["device"] == "cuda:7"
    assert calls[0]["kwargs"]["phase397q_profile_path"] == "/tmp/p.txt"
    assert calls[0]["kwargs"]["phase397q_profile_iters"] == 3
    assert calls[0]["kwargs"]["phase397q_full_cudagraph_metadata"] is True
    assert calls[0]["kwargs"]["phase397q_max_num_splits"] is None


def test_phase397q_collect_spec_forwards_max_num_splits(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_attention_torch(*args, **kwargs):
        calls.append(kwargs)
        return {
            "target_backend": "FlashAttnMLAImpl",
            "kernel_entrypoint": "forward_mqa",
            "kernel_source": "vllm_flash_attn_mla",
            "eager6_ms": 0.4,
            "eager200_p50_ms": 0.3,
            "graph_p50_ms": 0.2,
            "graph_capture": True,
            "phase397q_profile_path": "",
            "phase397q_profile_iters": 0,
            "phase397q_full_cudagraph_metadata": True,
            "scheduler_metadata_present": True,
            "scheduler_metadata_shape": "257",
            "decode_max_num_splits": 1,
            "called_attention_kernel": True,
            "called_model_forward": False,
        }

    monkeypatch.setitem(
        sys.modules,
        "collect_mla",
        types.SimpleNamespace(run_attention_torch=fake_run_attention_torch),
    )
    spec = run_phase397q_mla.Phase397QMLASpec(max_num_splits=1)
    run_phase397q_mla.collect_spec(spec, device="cuda:0")
    assert calls[0]["phase397q_max_num_splits"] == 1


def test_phase397q_main_writes_single_instrument_row(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "phase397q.csv"
    profile = tmp_path / "profile.txt"

    monkeypatch.setattr(
        run_phase397q_mla,
        "collect_spec",
        lambda spec, device, warmup_iters=20, measure_iters=200, profile_path=None, profile_iters=1: {
            "target_backend": "FlashAttnMLAImpl",
            "kernel_entrypoint": "forward_mqa",
            "kernel_source": "vllm_flash_attn_mla",
            "eager6_ms": 0.4,
            "eager200_p50_ms": 0.3,
            "graph_p50_ms": 0.2,
            "graph_capture": True,
            "phase397q_profile_path": profile_path,
            "phase397q_profile_iters": profile_iters,
            "phase397q_full_cudagraph_metadata": True,
            "scheduler_metadata_present": True,
            "scheduler_metadata_shape": "513",
            "decode_max_num_splits": 32,
            "called_attention_kernel": True,
            "called_model_forward": False,
        },
    )

    assert run_phase397q_mla.main(["--out", str(output), "--profile-out", str(profile), "--device", "cuda:7"]) == 0
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["device"] == "cuda:7"
    assert rows[0]["block_size"] == "64"
    assert rows[0]["phase397q_profile_path"] == str(profile)
    assert rows[0]["scheduler_metadata_present"] == "true"
    assert rows[0]["decode_max_num_splits"] == "32"
