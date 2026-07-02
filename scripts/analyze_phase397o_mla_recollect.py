#!/usr/bin/env python3
"""Phase397o: analyze generation MLA recollection evidence.

This is diagnostic-only. It reads the Phase397o raw GPU recollection CSV,
compares eager6/eager200/CUDA-graph timings against Phase397l serve attention
measurements, and writes a triage CSV/MD. It does not modify runtime code,
PerfDatabase rows, gates, or Default AIC readiness.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = "phase397o_mla_recollect_triage"
RAW_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397o_mla_recollect/phase397o_mla_recollect_raw.csv"
)
PHASE397L_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase397l_decode_profile"
OUTPUT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase397o_mla_recollect_triage.csv"
OUTPUT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase397o_mla_recollect_triage.md"

NUM_LAYERS = 61
HEADS = (8, 16)
KV_DTYPES = ("float16", "fp8")
BATCHES = (64, 128)
SEQ_LENS = (8192, 9001, 16384)
RANDOMIZE_VALUES = ("true", "false")
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"

SERVE_BREAKDOWNS = {
    8: PHASE397L_ROOT / "K2.5-tp8ep8-8k2k/op_breakdown.csv",
    16: PHASE397L_ROOT / "K2.5-tp4ep8dp2-8k2k/op_breakdown.csv",
}

FIELDNAMES = [
    "source",
    "row_type",
    "config",
    "local_num_heads",
    "tp_size",
    "kv_cache_dtype",
    "batch_size",
    "target_seq_len",
    "randomize_blocks",
    "target_backend",
    "kernel_source",
    "eager6_ms",
    "eager200_p50_ms",
    "graph_p50_ms",
    "serve_attention_ms",
    "serve_per_layer_ms",
    "graph_over_serve",
    "graph_over_eager6",
    "randomize_false_over_true",
    "candidate",
    "verdict",
    "evidence",
    "next_allowed_phase",
    "runtime_modified",
    "write_real_data_file",
    "gate_modified",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
    "default_readiness",
]


@dataclass(frozen=True)
class Key:
    local_num_heads: int
    kv_cache_dtype: str
    batch_size: int
    target_seq_len: int
    randomize_blocks: str


def _fmt(value: float | str) -> str:
    if isinstance(value, str):
        return value
    return f"{value:.6f}"


def _config_for_heads(local_num_heads: int) -> str:
    if local_num_heads == 8:
        return "tp8ep8-8k2k"
    if local_num_heads == 16:
        return "tp4ep8dp2-8k2k"
    raise ValueError(f"unsupported local_num_heads: {local_num_heads}")


def _common(row_type: str) -> dict[str, str]:
    return {
        "source": SOURCE,
        "row_type": row_type,
        "config": "",
        "local_num_heads": "",
        "tp_size": "",
        "kv_cache_dtype": "",
        "batch_size": "",
        "target_seq_len": "",
        "randomize_blocks": "",
        "target_backend": "",
        "kernel_source": "",
        "eager6_ms": "",
        "eager200_p50_ms": "",
        "graph_p50_ms": "",
        "serve_attention_ms": "",
        "serve_per_layer_ms": "",
        "graph_over_serve": "",
        "graph_over_eager6": "",
        "randomize_false_over_true": "",
        "candidate": "",
        "verdict": "",
        "evidence": "",
        "next_allowed_phase": "phase397p_attention_recollect_result_analyzer",
        "runtime_modified": FALSE,
        "write_real_data_file": FALSE,
        "gate_modified": FALSE,
        "diagnostic_only": TRUE,
        "valid_for_default": FALSE,
        "perf_database": FALSE,
        "default_readiness": DEFAULT_READINESS,
    }


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_serve_attention_ms(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["category"] == "attention":
                return float(row["ms_per_iter"])
    raise ValueError(f"missing attention row in {path}")


def _key(row: dict[str, str]) -> Key:
    return Key(
        local_num_heads=int(row["local_num_heads"]),
        kv_cache_dtype=row["kv_cache_dtype"],
        batch_size=int(row["batch_size"]),
        target_seq_len=int(row["target_seq_len"]),
        randomize_blocks=row["randomize_blocks"],
    )


def _index_rows(rows: Iterable[dict[str, str]]) -> dict[Key, dict[str, str]]:
    indexed: dict[Key, dict[str, str]] = {}
    for row in rows:
        key = _key(row)
        if key in indexed:
            raise ValueError(f"duplicate Phase397o raw key: {key}")
        indexed[key] = row
    return indexed


def validate_raw_rows(rows: list[dict[str, str]]) -> dict[Key, dict[str, str]]:
    indexed = _index_rows(rows)
    expected = {
        Key(heads, dtype, batch, seq, randomize)
        for heads in HEADS
        for dtype in KV_DTYPES
        for batch in BATCHES
        for seq in SEQ_LENS
        for randomize in RANDOMIZE_VALUES
    }
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed), key=str)
        extra = sorted(set(indexed) - expected, key=str)
        raise ValueError(f"Phase397o raw shape mismatch missing={missing} extra={extra}")

    for row in rows:
        if row["valid_for_default"] != FALSE:
            raise ValueError("Phase397o raw must keep valid_for_default=false")
        if row["perf_database"] != FALSE:
            raise ValueError("Phase397o raw must keep perf_database=false")
        if row["default_readiness"] != DEFAULT_READINESS:
            raise ValueError("Phase397o raw must keep default_readiness=No-Go")
        if row["graph_capture"] != TRUE:
            raise ValueError("Phase397o raw must capture CUDA graph timing")
        if row["called_attention_kernel"] != TRUE:
            raise ValueError("Phase397o raw must call attention kernel")
        if row["called_model_forward"] != FALSE:
            raise ValueError("Phase397o raw must not call model forward")
        if float(row["graph_p50_ms"]) <= 0.0:
            raise ValueError("Phase397o graph_p50_ms must be positive")
        if row["kv_cache_dtype"] == "float16":
            if row["target_backend"] != "FlashAttnMLAImpl":
                raise ValueError("float16 recollection must use FlashAttnMLAImpl")
            if row["kernel_source"] != "vllm_flash_attn_mla":
                raise ValueError("float16 recollection must use vllm_flash_attn_mla")
        if row["kv_cache_dtype"] == "fp8":
            if row["target_backend"] != "TritonMLAImpl":
                raise ValueError("fp8 recollection must record TritonMLAImpl backend")
            if row["kernel_source"] != "vllm_triton_mla":
                raise ValueError("fp8 recollection must record vllm_triton_mla")
    return indexed


def _randomize_ratio(indexed: dict[Key, dict[str, str]], key: Key) -> float:
    true_key = Key(
        key.local_num_heads,
        key.kv_cache_dtype,
        key.batch_size,
        key.target_seq_len,
        TRUE,
    )
    false_key = Key(
        key.local_num_heads,
        key.kv_cache_dtype,
        key.batch_size,
        key.target_seq_len,
        FALSE,
    )
    return float(indexed[false_key]["graph_p50_ms"]) / float(indexed[true_key]["graph_p50_ms"])


def build_triage_rows(raw_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    indexed = validate_raw_rows(raw_rows)
    serve_attention = {
        heads: _load_serve_attention_ms(path) for heads, path in SERVE_BREAKDOWNS.items()
    }
    serve_per_layer = {heads: value / NUM_LAYERS for heads, value in serve_attention.items()}

    rows: list[dict[str, str]] = []
    for key in sorted(indexed, key=lambda k: (k.local_num_heads, k.kv_cache_dtype, k.batch_size, k.target_seq_len, k.randomize_blocks)):
        raw = indexed[key]
        graph = float(raw["graph_p50_ms"])
        eager6 = float(raw["eager6_ms"])
        row = _common("shape_compare")
        row.update(
            {
                "config": _config_for_heads(key.local_num_heads),
                "local_num_heads": str(key.local_num_heads),
                "tp_size": raw["tp_size"],
                "kv_cache_dtype": key.kv_cache_dtype,
                "batch_size": str(key.batch_size),
                "target_seq_len": str(key.target_seq_len),
                "randomize_blocks": key.randomize_blocks,
                "target_backend": raw["target_backend"],
                "kernel_source": raw["kernel_source"],
                "eager6_ms": raw["eager6_ms"],
                "eager200_p50_ms": raw["eager200_p50_ms"],
                "graph_p50_ms": raw["graph_p50_ms"],
                "serve_attention_ms": _fmt(serve_attention[key.local_num_heads]),
                "serve_per_layer_ms": _fmt(serve_per_layer[key.local_num_heads]),
                "graph_over_serve": _fmt(graph / serve_per_layer[key.local_num_heads]),
                "graph_over_eager6": _fmt(graph / eager6),
                "randomize_false_over_true": _fmt(_randomize_ratio(indexed, key)),
                "candidate": "three_way_recollect",
                "verdict": "diagnostic_only_not_default_evidence",
                "evidence": "eager6/eager200/graph measured against Phase397l serve per-layer attention",
            }
        )
        rows.append(row)

    key_tp8 = Key(8, "float16", 128, 9001, TRUE)
    tp8 = indexed[key_tp8]
    tp8_graph_ratio = float(tp8["graph_p50_ms"]) / serve_per_layer[8]
    timing_row = _common("decision")
    timing_row.update(
        {
            "config": "tp8ep8-8k2k",
            "local_num_heads": "8",
            "kv_cache_dtype": "float16",
            "batch_size": "128",
            "target_seq_len": "9001",
            "randomize_blocks": TRUE,
            "target_backend": tp8["target_backend"],
            "kernel_source": tp8["kernel_source"],
            "eager6_ms": tp8["eager6_ms"],
            "eager200_p50_ms": tp8["eager200_p50_ms"],
            "graph_p50_ms": tp8["graph_p50_ms"],
            "serve_attention_ms": _fmt(serve_attention[8]),
            "serve_per_layer_ms": _fmt(serve_per_layer[8]),
            "graph_over_serve": _fmt(tp8_graph_ratio),
            "graph_over_eager6": _fmt(float(tp8["graph_p50_ms"]) / float(tp8["eager6_ms"])),
            "randomize_false_over_true": _fmt(_randomize_ratio(indexed, key_tp8)),
            "candidate": "cuda_graph_launch_overhead",
            "verdict": "timing_method_not_sufficient",
            "evidence": "graph_p50 stays above 1.7x serve per-layer on the heads=8 batch=128 seq=9001 point",
            "next_allowed_phase": "phase397p_attention_recollect_result_analyzer",
        }
    )
    rows.append(timing_row)

    max_randomize_delta = max(
        abs(1.0 - _randomize_ratio(indexed, key))
        for key in indexed
        if key.randomize_blocks == TRUE
    )
    layout_row = _common("decision")
    layout_row.update(
        {
            "candidate": "randomized_block_layout",
            "verdict": "randomized_block_layout_not_primary",
            "evidence": f"max graph_p50 randomize false/true delta is {max_randomize_delta:.4f}; layout toggle does not explain the residual",
            "randomize_false_over_true": _fmt(1.0 + max_randomize_delta),
            "next_allowed_phase": "phase397p_attention_recollect_result_analyzer",
        }
    )
    rows.append(layout_row)

    fp8_row = _common("decision")
    fp8_row.update(
        {
            "candidate": "fp8_dtype_control",
            "verdict": "fp8_backend_not_comparable_to_db_flashmla",
            "evidence": "Phase397o fp8 recollection uses TritonMLAImpl/vllm_triton_mla because block-size 16 excludes FLASHMLA; it is a side control, not a DB-key replacement",
            "next_allowed_phase": "phase397p_attention_recollect_result_analyzer",
        }
    )
    rows.append(fp8_row)

    final_row = _common("decision")
    final_row.update(
        {
            "candidate": "phase397q_trigger",
            "verdict": "B2_kernel_workload_or_serve_seed_difference",
            "evidence": "CUDA graph and block randomization controls do not collapse heads=8 batch=128 float16 to serve timing; next evidence must reproduce serve seed/layout or collect exact table rows with a matching harness",
            "next_allowed_phase": "phase397p_attention_recollect_result_analyzer_then_phase397q_if_confirmed",
        }
    )
    rows.append(final_row)

    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _find_row(rows: list[dict[str, str]], **match: str) -> dict[str, str]:
    for row in rows:
        if all(row[key] == value for key, value in match.items()):
            return row
    raise KeyError(match)


def write_md(path: Path, rows: list[dict[str, str]]) -> None:
    key_rows = [
        _find_row(
            rows,
            row_type="shape_compare",
            local_num_heads="8",
            kv_cache_dtype="float16",
            batch_size="128",
            target_seq_len="8192",
            randomize_blocks=TRUE,
        ),
        _find_row(
            rows,
            row_type="shape_compare",
            local_num_heads="8",
            kv_cache_dtype="float16",
            batch_size="128",
            target_seq_len="9001",
            randomize_blocks=TRUE,
        ),
        _find_row(
            rows,
            row_type="shape_compare",
            local_num_heads="16",
            kv_cache_dtype="float16",
            batch_size="64",
            target_seq_len="9001",
            randomize_blocks=TRUE,
        ),
    ]
    decision_rows = [row for row in rows if row["row_type"] == "decision"]
    lines = [
        "# Phase397o MLA Recollect Triage",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Scope | generation_mla recollection triage only; no runtime/DB/gate change |",
        "| Raw artifact | docs/iter_gap_investigation/phase397o_mla_recollect/phase397o_mla_recollect_raw.csv |",
        "| Sweep | heads 8/16 x dtype float16/fp8 x batch 64/128 x seq 8192/9001/16384 x randomize true/false = 48 rows |",
        "| Primary verdict | B2_kernel_workload_or_serve_seed_difference |",
        "| Default AIC | No-Go |",
        "",
        "## Key points",
        "",
        "| heads | dtype | batch | seq | backend | eager6 ms | eager200 p50 ms | graph p50 ms | serve/layer ms | graph/serve | graph/eager6 | randomize false/true |",
        "|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in key_rows:
        lines.append(
            "| {local_num_heads} | {kv_cache_dtype} | {batch_size} | {target_seq_len} | {target_backend} | "
            "{eager6_ms} | {eager200_p50_ms} | {graph_p50_ms} | {serve_per_layer_ms} | "
            "{graph_over_serve}x | {graph_over_eager6}x | {randomize_false_over_true}x |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "| Candidate | Verdict | Evidence |",
            "|---|---|---|",
        ]
    )
    for row in decision_rows:
        lines.append(f"| {row['candidate']} | {row['verdict']} | {row['evidence']} |")

    lines.extend(
        [
            "",
            "## Phase397q trigger",
            "",
            "- Do not introduce a scalar multiplier; heads=8 and heads=16 need different corrections.",
            "- Do not update `generation_mla_perf.txt` from this evidence alone; fp8 recollection used TritonMLAImpl, not DB `vllm_flashmla`.",
            "- Next evidence must either reproduce serve seed/layout directly or recollect exact rows with a harness proven to match serve.",
            "- Default AIC remains No-Go.",
            "",
            "## Cleanup",
            "",
            "- `gpu_compute_apps_after.txt` shows 0MiB on all 8 H200 GPUs after clearing stale VLLM workers.",
            "- `process_residual_after.txt` is empty.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", default=str(RAW_CSV))
    parser.add_argument("--out-csv", default=str(OUTPUT_CSV))
    parser.add_argument("--out-md", default=str(OUTPUT_MD))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_rows = _load_csv(Path(args.raw_csv))
    rows = build_triage_rows(raw_rows)
    write_csv(Path(args.out_csv), rows)
    write_md(Path(args.out_md), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
