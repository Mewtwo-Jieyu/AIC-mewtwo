#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase397o: recollect vLLM generation MLA timing points.

Runs a single-GPU sweep for Kimi-K2.5 decode MLA using the same fake MLA model
path as collect_mla.py. Each shape records three timing surfaces:
legacy eager6, eager200 p50, and CUDA graph p50. Writes diagnostic CSV only;
does not modify perf database rows or gates.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _path in (str(_REPO_ROOT), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


Q_LORA_RANK = 1536
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 128
V_HEAD_DIM = 128
BLOCK_SIZE = 16
GLOBAL_NUM_HEADS = 128
MODEL_NAME = "moonshotai/Kimi-K2.5"

LOCAL_HEADS = [8, 16]
KV_CACHE_DTYPES = ["float16", "fp8"]
BATCH_SIZES = [64, 128]
TARGET_SEQ_LENS = [8192, 9001, 16384]
RANDOMIZE_BLOCKS = [True, False]


PHASE397O_COLUMNS = [
    "source",
    "local_num_heads",
    "tp_size",
    "kv_cache_dtype",
    "batch_size",
    "target_seq_len",
    "input_len",
    "randomize_blocks",
    "target_backend",
    "kernel_entrypoint",
    "kernel_source",
    "eager6_ms",
    "eager200_mean_ms",
    "eager200_p50_ms",
    "eager200_p99_ms",
    "graph_mean_ms",
    "graph_p50_ms",
    "graph_p99_ms",
    "graph_capture",
    "warmup_iters",
    "measure_iters",
    "called_attention_kernel",
    "called_model_forward",
    "device",
    "valid_for_default",
    "perf_database",
    "default_readiness",
]


@dataclass(frozen=True)
class Phase397OMLASpec:
    local_num_heads: int
    kv_cache_dtype: str
    batch_size: int
    target_seq_len: int
    randomize_blocks: bool

    @property
    def tp_size(self) -> int:
        if GLOBAL_NUM_HEADS % self.local_num_heads != 0:
            raise ValueError(f"local_num_heads must divide {GLOBAL_NUM_HEADS}")
        return GLOBAL_NUM_HEADS // self.local_num_heads

    @property
    def input_len(self) -> int:
        return self.target_seq_len - 1

    @property
    def use_fp8_kv_cache(self) -> bool:
        if self.kv_cache_dtype == "float16":
            return False
        if self.kv_cache_dtype == "fp8":
            return True
        raise ValueError(f"unsupported kv_cache_dtype: {self.kv_cache_dtype}")


def _format_bool(value: bool) -> str:
    return str(value).lower()


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def build_phase397o_specs() -> list[Phase397OMLASpec]:
    specs = []
    for local_heads in LOCAL_HEADS:
        for kv_dtype in KV_CACHE_DTYPES:
            for batch_size in BATCH_SIZES:
                for target_seq_len in TARGET_SEQ_LENS:
                    for randomize_blocks in RANDOMIZE_BLOCKS:
                        specs.append(
                            Phase397OMLASpec(
                                local_num_heads=local_heads,
                                kv_cache_dtype=kv_dtype,
                                batch_size=batch_size,
                                target_seq_len=target_seq_len,
                                randomize_blocks=randomize_blocks,
                            )
                        )
    return specs


def filter_specs(
    specs: list[Phase397OMLASpec],
    local_heads: int | None = None,
    kv_cache_dtype: str | None = None,
    batch_size: int | None = None,
    target_seq_len: int | None = None,
    randomize_blocks: bool | None = None,
) -> list[Phase397OMLASpec]:
    filtered = []
    for spec in specs:
        if local_heads is not None and spec.local_num_heads != local_heads:
            continue
        if kv_cache_dtype is not None and spec.kv_cache_dtype != kv_cache_dtype:
            continue
        if batch_size is not None and spec.batch_size != batch_size:
            continue
        if target_seq_len is not None and spec.target_seq_len != target_seq_len:
            continue
        if randomize_blocks is not None and spec.randomize_blocks != randomize_blocks:
            continue
        filtered.append(spec)
    return filtered


def collect_spec(
    spec: Phase397OMLASpec,
    device: str,
    warmup_iters: int = 20,
    measure_iters: int = 200,
) -> dict[str, object]:
    from collect_mla import run_attention_torch

    return run_attention_torch(
        spec.batch_size,
        spec.input_len,
        GLOBAL_NUM_HEADS,
        spec.tp_size,
        Q_LORA_RANK,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        QK_NOPE_HEAD_DIM,
        V_HEAD_DIM,
        BLOCK_SIZE,
        MODEL_NAME,
        spec.use_fp8_kv_cache,
        False,
        "",
        device=device,
        randomize_blocks=spec.randomize_blocks,
        phase397o_timing=True,
        phase397o_warmup_iters=warmup_iters,
        phase397o_measure_iters=measure_iters,
    )


def build_row(
    spec: Phase397OMLASpec,
    result: dict[str, object],
    device: str = "cuda:0",
    warmup_iters: int = 20,
    measure_iters: int = 200,
) -> dict[str, str]:
    if result.get("called_model_forward") is True:
        raise SystemExit("Phase397o must not call model forward")
    if result.get("called_attention_kernel") is not True:
        raise SystemExit("Phase397o did not call attention kernel")
    row = {
        "source": "phase397o_mla_recollect",
        "local_num_heads": str(spec.local_num_heads),
        "tp_size": str(spec.tp_size),
        "kv_cache_dtype": spec.kv_cache_dtype,
        "batch_size": str(spec.batch_size),
        "target_seq_len": str(spec.target_seq_len),
        "input_len": str(spec.input_len),
        "randomize_blocks": _format_bool(spec.randomize_blocks),
        "target_backend": str(result["target_backend"]),
        "kernel_entrypoint": str(result["kernel_entrypoint"]),
        "kernel_source": str(result["kernel_source"]),
        "eager6_ms": _format_float(float(result["eager6_ms"])),
        "eager200_mean_ms": _format_float(float(result.get("eager200_mean_ms", result["eager200_p50_ms"]))),
        "eager200_p50_ms": _format_float(float(result["eager200_p50_ms"])),
        "eager200_p99_ms": _format_float(float(result.get("eager200_p99_ms", result["eager200_p50_ms"]))),
        "graph_mean_ms": _format_float(float(result.get("graph_mean_ms", result["graph_p50_ms"]))),
        "graph_p50_ms": _format_float(float(result["graph_p50_ms"])),
        "graph_p99_ms": _format_float(float(result.get("graph_p99_ms", result["graph_p50_ms"]))),
        "graph_capture": _format_bool(bool(result["graph_capture"])),
        "warmup_iters": str(warmup_iters),
        "measure_iters": str(measure_iters),
        "called_attention_kernel": _format_bool(bool(result["called_attention_kernel"])),
        "called_model_forward": _format_bool(bool(result["called_model_forward"])),
        "device": device,
        "valid_for_default": "false",
        "perf_database": "false",
        "default_readiness": "No-Go",
    }
    if list(row) != PHASE397O_COLUMNS:
        raise SystemExit("Phase397o row column order mismatch")
    return row


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PHASE397O_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/phase397o_mla_recollect.csv")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--measure-iters", type=int, default=200)
    parser.add_argument("--local-heads", type=int, choices=LOCAL_HEADS)
    parser.add_argument("--kv-cache-dtype", choices=KV_CACHE_DTYPES)
    parser.add_argument("--batch-size", type=int, choices=BATCH_SIZES)
    parser.add_argument("--target-seq-len", type=int, choices=TARGET_SEQ_LENS)
    parser.add_argument("--randomize-blocks", choices=["true", "false"])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.warmup_iters < 0:
        raise SystemExit("--warmup-iters must be non-negative")
    if args.measure_iters <= 0:
        raise SystemExit("--measure-iters must be positive")

    randomize_filter = None
    if args.randomize_blocks is not None:
        randomize_filter = args.randomize_blocks == "true"
    rows = []
    specs = filter_specs(
        build_phase397o_specs(),
        local_heads=args.local_heads,
        kv_cache_dtype=args.kv_cache_dtype,
        batch_size=args.batch_size,
        target_seq_len=args.target_seq_len,
        randomize_blocks=randomize_filter,
    )
    if not specs:
        raise SystemExit("Phase397o filter produced no shapes")
    for idx, spec in enumerate(specs, start=1):
        print(
            "[phase397o] "
            f"{idx}/{len(specs)} heads={spec.local_num_heads} dtype={spec.kv_cache_dtype} "
            f"batch={spec.batch_size} seq={spec.target_seq_len} randomize={spec.randomize_blocks}",
            flush=True,
        )
        result = collect_spec(
            spec,
            device=args.device,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
        )
        rows.append(
            build_row(
                spec,
                result,
                device=args.device,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
            )
        )
        write_rows(Path(args.out), rows)
    print(f"[phase397o] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
