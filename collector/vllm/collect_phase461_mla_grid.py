#!/usr/bin/env python3
"""Collect the Phase461 generation-MLA replacement grid."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path


BATCHES = (8, 16)
TARGET_SEQ_LENS = (16_384, 32_768, 65_536)
REPEATS = 3
GLOBAL_NUM_HEADS = 64
TP_SIZE = 8
LOCAL_NUM_HEADS = GLOBAL_NUM_HEADS // TP_SIZE
MAX_REPEAT_SPREAD = 1.15


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _select_row(path: Path, *, batch: int, target_seq_len: int) -> dict[str, str]:
    matches = [
        row
        for row in _read_rows(path)
        if row["op_name"] == "generation_mla"
        and row["kernel_source"] == "vllm_flash_attn_mla"
        and row["mla_dtype"] == "float16"
        and row["kv_cache_dtype"] == "float16"
        and int(row["num_heads"]) == LOCAL_NUM_HEADS
        and int(row["batch_size"]) == batch
        and int(row["tp_size"]) == TP_SIZE
        and int(row["isl"]) == 1
        and int(row["step"]) == target_seq_len - 1
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one MLA row in {path}, got {len(matches)}")
    return matches[0]


def collect(out_dir: Path, committed_root: Path, model_path: str) -> dict[str, object]:
    sys.path.insert(0, str(committed_root))
    from collector.vllm.collect_mla import run_attention_torch

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for repeat in range(1, REPEATS + 1):
        for batch in BATCHES:
            for target_seq_len in TARGET_SEQ_LENS:
                raw_path = raw_dir / f"mla_b{batch}_kv{target_seq_len}_r{repeat}.csv"
                run_attention_torch(
                    batch_size=batch,
                    input_len=target_seq_len - 1,
                    num_heads=GLOBAL_NUM_HEADS,
                    tp_size=TP_SIZE,
                    q_lora_rank=1536,
                    kv_lora_rank=512,
                    qk_rope_head_dim=64,
                    qk_nope_head_dim=128,
                    v_head_dim=128,
                    block_size=16,
                    model_name=model_path,
                    use_fp8_kv_cache=False,
                    is_context_phase=False,
                    perf_filename=str(raw_path),
                    device="cuda:0",
                    randomize_blocks=True,
                )
                row = _select_row(raw_path, batch=batch, target_seq_len=target_seq_len)
                rows.append(
                    {
                        "repeat": repeat,
                        "target_seq_len": target_seq_len,
                        **row,
                    }
                )

    expected = len(BATCHES) * len(TARGET_SEQ_LENS) * REPEATS
    if len(rows) != expected:
        raise RuntimeError(f"incomplete MLA grid: {len(rows)} != {expected}")

    point_summaries: list[dict[str, object]] = []
    for batch in BATCHES:
        for target_seq_len in TARGET_SEQ_LENS:
            values = [
                float(row["latency"])
                for row in rows
                if row["batch_size"] == str(batch)
                and row["target_seq_len"] == target_seq_len
            ]
            spread = max(values) / min(values)
            point_summaries.append(
                {
                    "batch_size": batch,
                    "target_seq_len": target_seq_len,
                    "samples": len(values),
                    "latency_median_ms": statistics.median(values),
                    "latency_min_ms": min(values),
                    "latency_max_ms": max(values),
                    "repeat_spread": spread,
                    "passed": len(values) == REPEATS and spread <= MAX_REPEAT_SPREAD,
                }
            )
    passed = all(bool(row["passed"]) for row in point_summaries)

    combined = out_dir / "phase461_mla_grid_raw.csv"
    fieldnames = ["repeat", "target_seq_len", *[key for key in rows[0] if key not in {"repeat", "target_seq_len"}]]
    with combined.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "phase": "phase461_cost_recollect",
        "measurement": "committed_collect_mla_kernel_microbench",
        "committed_collector_root": str(committed_root),
        "collector_commit": (committed_root / "collector_commit.txt").read_text(encoding="utf-8").strip(),
        "rows": len(rows),
        "expected_rows": expected,
        "max_repeat_spread": MAX_REPEAT_SPREAD,
        "points": point_summaries,
        "passed": passed,
    }
    (out_dir / "phase461_mla_grid_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError("MLA grid repeatability gate failed")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--committed-root",
        type=Path,
        default=Path(os.environ["COMMITTED_COLLECTOR_ROOT"]),
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "MODEL_PATH",
            "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/"
            "zskj-hub/models--moonshotai--Kimi-K2.5",
        ),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = collect(args.out_dir, args.committed_root, args.model_path)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
