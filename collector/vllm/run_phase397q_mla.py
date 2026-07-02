#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase397q: instrument Kimi-K2.5 generation MLA split-KV behavior.

This runner is diagnostic only. It keeps the Phase397o timing surfaces, switches
the target probe to block_size=64, and writes a torch profiler table so the H200
run can confirm whether forward_mqa emits FA3 split-KV scheduler/combine work.
"""

from __future__ import annotations

import argparse
import csv
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
GLOBAL_NUM_HEADS = 128
MODEL_NAME = "moonshotai/Kimi-K2.5"

DEFAULT_LOCAL_HEADS = 8
DEFAULT_KV_CACHE_DTYPE = "float16"
DEFAULT_BATCH_SIZE = 128
DEFAULT_TARGET_SEQ_LEN = 9001
DEFAULT_BLOCK_SIZE = 64
DEFAULT_RANDOMIZE_BLOCKS = True

LOCAL_HEADS = [8, 16]
KV_CACHE_DTYPES = ["float16", "fp8"]
TARGET_SEQ_LENS = [8192, 9001, 16384]


PHASE397Q_COLUMNS = [
    "source",
    "local_num_heads",
    "tp_size",
    "kv_cache_dtype",
    "batch_size",
    "target_seq_len",
    "input_len",
    "block_size",
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
    "phase397q_profile_path",
    "phase397q_profile_iters",
    "phase397q_full_cudagraph_metadata",
    "scheduler_metadata_present",
    "scheduler_metadata_shape",
    "decode_max_num_splits",
    "called_attention_kernel",
    "called_model_forward",
    "device",
    "valid_for_default",
    "perf_database",
    "default_readiness",
]


@dataclass(frozen=True)
class Phase397QMLASpec:
    local_num_heads: int = DEFAULT_LOCAL_HEADS
    kv_cache_dtype: str = DEFAULT_KV_CACHE_DTYPE
    batch_size: int = DEFAULT_BATCH_SIZE
    target_seq_len: int = DEFAULT_TARGET_SEQ_LEN
    block_size: int = DEFAULT_BLOCK_SIZE
    randomize_blocks: bool = DEFAULT_RANDOMIZE_BLOCKS
    max_num_splits: int | None = None

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


def build_phase397q_instrument_specs() -> list[Phase397QMLASpec]:
    return [Phase397QMLASpec()]


def collect_spec(
    spec: Phase397QMLASpec,
    device: str,
    warmup_iters: int = 20,
    measure_iters: int = 200,
    profile_path: str | None = None,
    profile_iters: int = 1,
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
        spec.block_size,
        MODEL_NAME,
        spec.use_fp8_kv_cache,
        False,
        "",
        device=device,
        randomize_blocks=spec.randomize_blocks,
        phase397o_timing=True,
        phase397o_warmup_iters=warmup_iters,
        phase397o_measure_iters=measure_iters,
        phase397q_profile_path=profile_path,
        phase397q_profile_iters=profile_iters,
        phase397q_full_cudagraph_metadata=True,
        phase397q_max_num_splits=spec.max_num_splits,
    )


def build_row(
    spec: Phase397QMLASpec,
    result: dict[str, object],
    device: str = "cuda:0",
    warmup_iters: int = 20,
    measure_iters: int = 200,
) -> dict[str, str]:
    if result.get("called_model_forward") is True:
        raise SystemExit("Phase397q must not call model forward")
    if result.get("called_attention_kernel") is not True:
        raise SystemExit("Phase397q did not call attention kernel")
    row = {
        "source": "phase397q_mla_split_instrument",
        "local_num_heads": str(spec.local_num_heads),
        "tp_size": str(spec.tp_size),
        "kv_cache_dtype": spec.kv_cache_dtype,
        "batch_size": str(spec.batch_size),
        "target_seq_len": str(spec.target_seq_len),
        "input_len": str(spec.input_len),
        "block_size": str(spec.block_size),
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
        "phase397q_profile_path": str(result.get("phase397q_profile_path", "")),
        "phase397q_profile_iters": str(result.get("phase397q_profile_iters", 0)),
        "phase397q_full_cudagraph_metadata": _format_bool(bool(result.get("phase397q_full_cudagraph_metadata"))),
        "scheduler_metadata_present": _format_bool(bool(result.get("scheduler_metadata_present"))),
        "scheduler_metadata_shape": str(result.get("scheduler_metadata_shape", "")),
        "decode_max_num_splits": str(result.get("decode_max_num_splits", 0)),
        "called_attention_kernel": _format_bool(bool(result["called_attention_kernel"])),
        "called_model_forward": _format_bool(bool(result["called_model_forward"])),
        "device": device,
        "valid_for_default": "false",
        "perf_database": "false",
        "default_readiness": "No-Go",
    }
    if list(row) != PHASE397Q_COLUMNS:
        raise SystemExit("Phase397q row column order mismatch")
    return row


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PHASE397Q_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/phase397q_mla_split_instrument.csv")
    parser.add_argument("--profile-out", default="/tmp/phase397q_mla_split_instrument_profiler.txt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--measure-iters", type=int, default=200)
    parser.add_argument("--profile-iters", type=int, default=1)
    parser.add_argument("--max-num-splits", type=int)
    parser.add_argument("--local-heads", type=int, choices=LOCAL_HEADS, default=DEFAULT_LOCAL_HEADS)
    parser.add_argument("--kv-cache-dtype", choices=KV_CACHE_DTYPES, default=DEFAULT_KV_CACHE_DTYPE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--target-seq-len", type=int, choices=TARGET_SEQ_LENS, default=DEFAULT_TARGET_SEQ_LEN)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--randomize-blocks", choices=["true", "false"], default="true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.warmup_iters < 0:
        raise SystemExit("--warmup-iters must be non-negative")
    if args.measure_iters <= 0:
        raise SystemExit("--measure-iters must be positive")
    if args.profile_iters <= 0:
        raise SystemExit("--profile-iters must be positive")
    if args.block_size <= 0:
        raise SystemExit("--block-size must be positive")

    spec = Phase397QMLASpec(
        local_num_heads=args.local_heads,
        kv_cache_dtype=args.kv_cache_dtype,
        batch_size=args.batch_size,
        target_seq_len=args.target_seq_len,
        block_size=args.block_size,
        randomize_blocks=args.randomize_blocks == "true",
        max_num_splits=args.max_num_splits,
    )
    print(
        "[phase397q] "
        f"heads={spec.local_num_heads} dtype={spec.kv_cache_dtype} "
        f"batch={spec.batch_size} seq={spec.target_seq_len} block={spec.block_size} "
        f"randomize={spec.randomize_blocks}",
        flush=True,
    )
    result = collect_spec(
        spec,
        device=args.device,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        profile_path=args.profile_out,
        profile_iters=args.profile_iters,
    )
    row = build_row(
        spec,
        result,
        device=args.device,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
    )
    write_rows(Path(args.out), [row])
    print(f"[phase397q] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
