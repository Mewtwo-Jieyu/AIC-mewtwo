import csv
from pathlib import Path

import pytest

from collector.vllm import collect_mla_state_smoke


def read_output(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_mla_state_smoke_writes_single_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "mla_state_smoke.csv"

    monkeypatch.setattr(
        collect_mla_state_smoke,
        "run_mla_state_smoke",
        lambda **_kwargs: collect_mla_state_smoke.MLAStateSmokeResult(
            target_backend="FlashAttnMLAImpl",
            kernel_entrypoint="forward_mqa",
            phase="mixed",
            topology_key="tp4dp2moetp1ep8",
            attention_actual_tokens=241,
            attention_max_query_len=16,
            num_tokens_padded=248,
            local_num_heads=16,
            local_num_kv_heads=16,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_head_dim=192,
            mla_head_size=576,
            v_head_dim=128,
            query_shape="241x16x576",
            kv_c_shape="241x512",
            k_pe_shape="241x1x64",
            block_table_shape="16x16384",
            kv_cache_shape="8192x16x576",
            metadata_type="FlashAttnMLAMetadata",
            metadata_num_actual_tokens=241,
            metadata_max_query_len=16,
            output_shape="241x16x512",
            called_attention_kernel=True,
            called_model_forward=False,
        ),
    )

    assert collect_mla_state_smoke.main(["--mode", "state-smoke", "--output", str(output)]) == 0

    rows = read_output(output)
    assert len(rows) == 1
    row = rows[0]
    assert row["smoke_name"] == "attention_kernel_mla_state_smoke"
    assert row["phase"] == "mixed"
    assert row["target_backend"] == "FlashAttnMLAImpl"
    assert row["kernel_entrypoint"] == "forward_mqa"
    assert row["attention_actual_tokens"] == "241"
    assert row["attention_max_query_len"] == "16"
    assert row["num_tokens_padded"] == "248"
    assert row["query_shape"] == "241x16x576"
    assert row["block_table_shape"] == "16x16384"
    assert row["metadata_num_actual_tokens"] == "241"
    assert row["output_shape"] == "241x16x512"
    assert row["called_attention_kernel"] == "true"
    assert row["called_model_forward"] == "false"
    assert row["timing"] == "none"
    assert row["valid_for_default"] == "false"
    assert list(row.keys()) == collect_mla_state_smoke.MLA_STATE_SMOKE_COLUMNS


def test_mla_timing_smoke_writes_single_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "mla_timing_smoke.csv"

    monkeypatch.setattr(
        collect_mla_state_smoke,
        "run_mla_timing_smoke",
        lambda **_kwargs: collect_mla_state_smoke.MLATimingSmokeResult(
            target_backend="FlashAttnMLAImpl",
            kernel_entrypoint="forward_mqa",
            phase="mixed",
            topology_key="tp4dp2moetp1ep8",
            attention_actual_tokens=241,
            attention_max_query_len=16,
            num_tokens_padded=248,
            query_shape="241x16x576",
            kv_cache_shape="16384x16x576",
            metadata_type="FlashAttnMLAMetadata",
            output_shape="241x16x512",
            warmup_iters=20,
            measure_iters=200,
            wall_ms_mean=0.4,
            wall_ms_p50=0.3,
            wall_ms_p99=0.9,
            cuda_event_ms_mean=0.2,
            cuda_event_ms_p50=0.2,
            cuda_event_ms_p99=0.5,
            called_attention_kernel=True,
            called_model_forward=False,
        ),
    )

    assert collect_mla_state_smoke.main(["--mode", "timing-smoke", "--output", str(output)]) == 0

    rows = read_output(output)
    assert len(rows) == 1
    row = rows[0]
    assert row["smoke_name"] == "mla_forward_mqa_timing_smoke"
    assert row["kernel_entrypoint"] == "forward_mqa"
    assert row["segment"] == "forward_mqa_only"
    assert row["warmup_iters"] == "20"
    assert row["measure_iters"] == "200"
    assert row["wall_ms_mean"] == "0.400000"
    assert row["cuda_event_ms_p99"] == "0.500000"
    assert row["called_attention_kernel"] == "true"
    assert row["called_model_forward"] == "false"
    assert row["valid_for_default"] == "false"
    assert list(row.keys()) == collect_mla_state_smoke.MLA_TIMING_SMOKE_COLUMNS


def test_mla_state_smoke_rejects_non_mixed_key_before_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_run(**_kwargs: object) -> collect_mla_state_smoke.MLAStateSmokeResult:
        raise AssertionError("invalid key must fail before state-smoke run")

    monkeypatch.setattr(collect_mla_state_smoke, "run_mla_state_smoke", fail_run)

    with pytest.raises(SystemExit, match="MLA state-smoke only supports"):
        collect_mla_state_smoke.main(
            [
                "--mode",
                "state-smoke",
                "--phase",
                "pure_decode",
                "--attention-actual-tokens",
                "16",
                "--attention-max-query-len",
                "1",
                "--output",
                str(tmp_path / "out.csv"),
            ]
        )


def test_mla_timing_smoke_rejects_non_mixed_key_before_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_run(**_kwargs: object) -> collect_mla_state_smoke.MLATimingSmokeResult:
        raise AssertionError("invalid key must fail before timing-smoke run")

    monkeypatch.setattr(collect_mla_state_smoke, "run_mla_timing_smoke", fail_run)

    with pytest.raises(SystemExit, match="MLA state-smoke only supports"):
        collect_mla_state_smoke.main(
            [
                "--mode",
                "timing-smoke",
                "--phase",
                "pure_decode",
                "--attention-actual-tokens",
                "16",
                "--attention-max-query-len",
                "1",
                "--output",
                str(tmp_path / "out.csv"),
            ]
        )


def test_mla_state_smoke_requires_output() -> None:
    with pytest.raises(SystemExit):
        collect_mla_state_smoke.main(["--mode", "state-smoke"])
