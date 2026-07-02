from __future__ import annotations

import csv
from pathlib import Path

from collector.vllm import run_phase397o_mla


def test_phase397o_shape_grid_has_48_rows() -> None:
    specs = run_phase397o_mla.build_phase397o_specs()
    assert len(specs) == 48
    assert {s.local_num_heads for s in specs} == {8, 16}
    assert {s.kv_cache_dtype for s in specs} == {"float16", "fp8"}
    assert {s.batch_size for s in specs} == {64, 128}
    assert {s.target_seq_len for s in specs} == {8192, 9001, 16384}
    assert {s.randomize_blocks for s in specs} == {False, True}


def test_phase397o_row_preserves_three_timing_modes() -> None:
    spec = run_phase397o_mla.Phase397OMLASpec(
        local_num_heads=8,
        kv_cache_dtype="float16",
        batch_size=128,
        target_seq_len=8192,
        randomize_blocks=True,
    )
    result = {
        "target_backend": "FlashAttnMLAImpl",
        "kernel_entrypoint": "forward_mqa",
        "kernel_source": "vllm_flash_attn_mla",
        "eager6_ms": 0.486,
        "eager200_p50_ms": 0.300,
        "graph_p50_ms": 0.280,
        "graph_capture": True,
        "called_attention_kernel": True,
        "called_model_forward": False,
    }
    row = run_phase397o_mla.build_row(spec, result)
    assert row["local_num_heads"] == "8"
    assert row["kv_cache_dtype"] == "float16"
    assert row["target_seq_len"] == "8192"
    assert row["input_len"] == "8191"
    assert row["randomize_blocks"] == "true"
    assert row["eager6_ms"] == "0.486000"
    assert row["eager200_p50_ms"] == "0.300000"
    assert row["graph_p50_ms"] == "0.280000"
    assert row["graph_capture"] == "true"
    assert row["valid_for_default"] == "false"
    assert list(row) == run_phase397o_mla.PHASE397O_COLUMNS


def test_phase397o_filter_selects_single_shape() -> None:
    specs = run_phase397o_mla.filter_specs(
        run_phase397o_mla.build_phase397o_specs(),
        local_heads=8,
        kv_cache_dtype="float16",
        batch_size=128,
        target_seq_len=16384,
        randomize_blocks=True,
    )
    assert specs == [
        run_phase397o_mla.Phase397OMLASpec(
            local_num_heads=8,
            kv_cache_dtype="float16",
            batch_size=128,
            target_seq_len=16384,
            randomize_blocks=True,
        )
    ]


def test_phase397o_main_writes_rows_with_stubbed_collector(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "phase397o.csv"

    specs = [
        run_phase397o_mla.Phase397OMLASpec(
            local_num_heads=8,
            kv_cache_dtype="float16",
            batch_size=64,
            target_seq_len=8192,
            randomize_blocks=False,
        )
    ]
    monkeypatch.setattr(run_phase397o_mla, "build_phase397o_specs", lambda: specs)
    monkeypatch.setattr(
        run_phase397o_mla,
        "collect_spec",
        lambda spec, device, warmup_iters=20, measure_iters=200: {
            "target_backend": "FlashAttnMLAImpl",
            "kernel_entrypoint": "forward_mqa",
            "kernel_source": "vllm_flash_attn_mla",
            "eager6_ms": 0.4,
            "eager200_p50_ms": 0.3,
            "graph_p50_ms": 0.2,
            "graph_capture": True,
            "called_attention_kernel": True,
            "called_model_forward": False,
        },
    )

    assert run_phase397o_mla.main(["--out", str(output), "--device", "cuda:7"]) == 0
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["device"] == "cuda:7"
    assert rows[0]["graph_p50_ms"] == "0.200000"
