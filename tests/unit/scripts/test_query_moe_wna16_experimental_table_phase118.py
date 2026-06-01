from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.query_moe_wna16_experimental_table_phase118 import (
    QueryError,
    REQUIRED_QUERY_FIELDS,
    main,
    query_exact_key,
    read_key_json,
    read_table,
)


TABLE_FIELDS = [
    "backend",
    "vllm_version",
    "device",
    "model_family",
    "module",
    "experts_impl",
    "kernel",
    "dtype",
    "activation_dtype",
    "hidden",
    "intermediate",
    "global_experts",
    "local_experts",
    "topk",
    "topology",
    "config_sha256",
    "input_kind",
    "tokens_actual",
    "cuda_event_ms_mean",
    "wall_ms_mean",
    "run_count",
    "query_policy",
    "model_use",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]


def _table_row(tokens: int = 128) -> dict[str, str]:
    return {
        "backend": "vllm",
        "vllm_version": "0.19.0",
        "device": "NVIDIA_H200",
        "model_family": "KimiK25",
        "module": "DeepseekV2MoE",
        "experts_impl": "SharedFusedMoE",
        "kernel": "WNA16",
        "dtype": "int4_w4a16",
        "activation_dtype": "bfloat16",
        "hidden": "7168",
        "intermediate": "2048",
        "global_experts": "384",
        "local_experts": "48",
        "topk": "8",
        "topology": "tp4dp2ep8",
        "config_sha256": (
            "15c797eed1dd441e1d6d50fcf7584b2264d248d5ebc1144e1c95e6c4e885d853"
        ),
        "input_kind": "synthetic_random_hidden_states",
        "tokens_actual": str(tokens),
        "cuda_event_ms_mean": "1.04053000286",
        "wall_ms_mean": "1.0574859567",
        "run_count": "1",
        "query_policy": "exact_match_only",
        "model_use": "experimental_table_prototype_only",
        "diagnostic_only": "true",
        "valid_for_default": "false",
        "perf_database": "false",
    }


def _query_key(tokens: int = 128) -> dict[str, object]:
    row = _table_row(tokens)
    return {field: row[field] for field in REQUIRED_QUERY_FIELDS}


def _write_table(
    path: Path,
    rows: list[dict[str, str]],
    fields: list[str] | None = None,
) -> None:
    fieldnames = TABLE_FIELDS if fields is None else fields
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_key(path: Path, key: dict[str, object]) -> None:
    path.write_text(json.dumps(key, indent=2), encoding="utf-8")


def test_full_key_query_hits_and_cli_writes_single_diagnostic_row(tmp_path: Path) -> None:
    table_path = tmp_path / "table.csv"
    key_path = tmp_path / "key.json"
    out_path = tmp_path / "result.csv"
    _write_table(table_path, [_table_row(128), _table_row(248)])
    _write_key(key_path, _query_key(128))

    row = query_exact_key(read_table(table_path), read_key_json(key_path))

    assert row["tokens_actual"] == "128"
    assert row["cuda_event_ms_mean"] == "1.04053000286"
    assert row["model_use"] == "experimental_table_prototype_only"
    assert row["diagnostic_only"] == "true"
    assert row["valid_for_default"] == "false"
    assert row["perf_database"] == "false"
    assert (
        main(
            [
                "--table",
                str(table_path),
                "--key-json",
                str(key_path),
                "--out",
                str(out_path),
            ]
        )
        == 0
    )
    with out_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["tokens_actual"] == "128"


def test_missing_token_and_changed_key_fail_fast(tmp_path: Path) -> None:
    table_path = tmp_path / "table.csv"
    _write_table(table_path, [_table_row(128), _table_row(248)])
    rows = read_table(table_path)

    with pytest.raises(QueryError, match="no exact row"):
        query_exact_key(rows, _query_key(256))

    bad_key = _query_key(128)
    bad_key["config_sha256"] = "bad"
    with pytest.raises(QueryError, match="mismatched fields: config_sha256"):
        query_exact_key(rows, bad_key)


def test_missing_key_duplicate_row_forbidden_header_and_unsafe_row_fail_fast(
    tmp_path: Path,
) -> None:
    table_path = tmp_path / "table.csv"
    _write_table(table_path, [_table_row(128), _table_row(128)])
    with pytest.raises(QueryError, match="duplicate exact row"):
        query_exact_key(read_table(table_path), _query_key(128))

    missing_key = _query_key(128)
    del missing_key["topology"]
    with pytest.raises(QueryError, match="missing query fields: topology"):
        query_exact_key(read_table(table_path), missing_key)

    forbidden = tmp_path / "forbidden.csv"
    _write_table(
        forbidden,
        [{**_table_row(128), "latency_ms": "1.0"}],
        TABLE_FIELDS + ["latency_ms"],
    )
    with pytest.raises(QueryError, match="forbidden fields"):
        read_table(forbidden)

    unsafe = tmp_path / "unsafe.csv"
    _write_table(unsafe, [{**_table_row(128), "valid_for_default": "true"}])
    with pytest.raises(QueryError, match="valid_for_default must be false"):
        query_exact_key(read_table(unsafe), _query_key(128))
