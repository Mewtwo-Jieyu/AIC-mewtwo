from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.analyze_vllm_scheduler_descriptor_phase62 import (
    dedupe_rows_by_alignment_dp,
    main,
    parse_rows,
)


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "source": "vllm_scheduler",
        "scenario": "10k2k_b32_bt8192",
        "alignment_key": "engine_dp:1:step:7",
        "alignment_key_type": "engine_core_dp_step",
        "engine_step_id": 7,
        "iteration": 7,
        "phase": "mixed",
        "rank": 4,
        "local_rank": 4,
        "dp_rank": 1,
        "tp_rank": 0,
        "ep_rank": 4,
        "scheduled_context_tokens": 225,
        "scheduled_decode_tokens": 16,
        "scheduled_total_tokens": 241,
        "scheduled_context_reqs": 15,
        "scheduled_decode_reqs": 1,
        "scheduled_total_reqs": 16,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 256,
        "forward_token_count": 248,
        "forward_regime": "NONE:248",
        "cudagraph_runtime_mode": "NONE",
        "topology_key": "tp4dp2moetp1ep8",
        "tp": 4,
        "dp": 2,
        "moe_tp": 1,
        "moe_ep": 8,
        "valid_for_default": False,
        "perf_database": False,
        "diagnostic_only": True,
    }
    payload.update(overrides)
    return payload


def _write_log(path: Path, payloads: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(
            "prefix AIC_SCHEDULER_ALIGNMENT_DESCRIPTOR_ROW "
            + json.dumps(payload, sort_keys=True)
            for payload in payloads
        )
        + "\n",
        encoding="utf-8",
    )


def test_parse_scheduler_alignment_marker(tmp_path: Path) -> None:
    log_path = tmp_path / "serve.log"
    _write_log(log_path, [_payload()])

    rows = parse_rows(log_path)

    assert rows[0]["alignment_key"] == "engine_dp:1:step:7"
    assert rows[0]["alignment_key_type"] == "engine_core_dp_step"
    assert rows[0]["engine_step_id"] == 7
    assert rows[0]["iteration"] == 7
    assert rows[0]["valid_for_default"] is False
    assert rows[0]["perf_database"] is False
    assert rows[0]["diagnostic_only"] is True


def test_missing_alignment_key_fails_fast(tmp_path: Path) -> None:
    log_path = tmp_path / "serve.log"
    payload = _payload()
    del payload["alignment_key"]
    _write_log(log_path, [payload])

    with pytest.raises(ValueError, match="alignment_key"):
        parse_rows(log_path)


def test_forbidden_field_fails_fast(tmp_path: Path) -> None:
    log_path = tmp_path / "serve.log"
    _write_log(log_path, [_payload(profiler_cuda_ms=1.0)])

    with pytest.raises(ValueError, match="forbidden field"):
        parse_rows(log_path)


def test_engine_step_alignment_must_match_iteration(tmp_path: Path) -> None:
    log_path = tmp_path / "serve.log"
    _write_log(log_path, [_payload(iteration=8)])

    with pytest.raises(ValueError, match="iteration must equal engine_step_id"):
        parse_rows(log_path)


def test_engine_step_alignment_key_must_match_dp_and_step(tmp_path: Path) -> None:
    log_path = tmp_path / "serve.log"
    _write_log(log_path, [_payload(alignment_key="engine_dp:0:step:7")])

    with pytest.raises(ValueError, match="alignment_key mismatch"):
        parse_rows(log_path)


def test_dedupe_by_alignment_dp_requires_tp_rank_consistency(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "serve.log"
    payloads = [
        _payload(rank=rank, local_rank=rank, tp_rank=rank - 4, ep_rank=rank)
        for rank in range(4, 8)
    ]
    _write_log(log_path, payloads)

    deduped = dedupe_rows_by_alignment_dp(parse_rows(log_path))

    assert len(deduped) == 1
    assert deduped[0]["alignment_key"] == "engine_dp:1:step:7"
    assert deduped[0]["dp_rank"] == 1
    assert "rank" not in deduped[0]

    bad_payloads = [
        _payload(rank=4, local_rank=4, tp_rank=0, ep_rank=4),
        _payload(
            rank=5,
            local_rank=5,
            tp_rank=1,
            ep_rank=5,
            scheduled_decode_tokens=17,
            scheduled_total_tokens=242,
            forward_token_count=248,
        ),
    ]
    _write_log(log_path, bad_payloads)

    with pytest.raises(ValueError, match="inconsistent TP rank rows"):
        dedupe_rows_by_alignment_dp(parse_rows(log_path))


def test_main_writes_raw_and_dedup_csv(tmp_path: Path) -> None:
    log_path = tmp_path / "serve.log"
    rows_out = tmp_path / "rows.csv"
    dedup_out = tmp_path / "dedup.csv"
    _write_log(log_path, [_payload()])

    assert main(
        [
            "--log",
            str(log_path),
            "--rows-out",
            str(rows_out),
            "--dedup-by-alignment-dp-out",
            str(dedup_out),
        ]
    ) == 0

    with rows_out.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with dedup_out.open(newline="", encoding="utf-8") as f:
        dedup_rows = list(csv.DictReader(f))
    assert rows[0]["alignment_key_type"] == "engine_core_dp_step"
    assert dedup_rows[0]["alignment_key"] == "engine_dp:1:step:7"
    assert "rank" not in dedup_rows[0]
