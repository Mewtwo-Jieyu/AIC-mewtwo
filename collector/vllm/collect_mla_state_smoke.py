#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Phase 18 single-key MLA smoke wrapper.

This wrapper owns CLI/schema only. It imports collect_mla lazily in the real
smoke path so local dry tests do not require torch/vLLM/CUDA.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_TOPOLOGY_KEY = "tp4dp2moetp1ep8"
STATE_SMOKE_KEY = ("mixed", 241, 16, 248)

MLA_STATE_SMOKE_COLUMNS = [
    "smoke_name",
    "phase",
    "topology_key",
    "target_backend",
    "kernel_entrypoint",
    "attention_actual_tokens",
    "attention_max_query_len",
    "num_tokens_padded",
    "local_num_heads",
    "local_num_kv_heads",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_head_dim",
    "mla_head_size",
    "v_head_dim",
    "query_shape",
    "kv_c_shape",
    "k_pe_shape",
    "block_table_shape",
    "kv_cache_shape",
    "metadata_type",
    "metadata_num_actual_tokens",
    "metadata_max_query_len",
    "output_shape",
    "called_attention_kernel",
    "called_model_forward",
    "timing",
    "valid_for_default",
]

MLA_TIMING_SMOKE_COLUMNS = [
    "smoke_name",
    "phase",
    "topology_key",
    "target_backend",
    "kernel_entrypoint",
    "segment",
    "attention_actual_tokens",
    "attention_max_query_len",
    "num_tokens_padded",
    "query_shape",
    "kv_cache_shape",
    "metadata_type",
    "output_shape",
    "warmup_iters",
    "measure_iters",
    "wall_ms_mean",
    "wall_ms_p50",
    "wall_ms_p99",
    "cuda_event_ms_mean",
    "cuda_event_ms_p50",
    "cuda_event_ms_p99",
    "called_attention_kernel",
    "called_model_forward",
    "valid_for_default",
]


@dataclass(frozen=True)
class MLAStateSmokeResult:
    target_backend: str
    kernel_entrypoint: str
    phase: str
    topology_key: str
    attention_actual_tokens: int
    attention_max_query_len: int
    num_tokens_padded: int
    local_num_heads: int
    local_num_kv_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_head_dim: int
    mla_head_size: int
    v_head_dim: int
    query_shape: str
    kv_c_shape: str
    k_pe_shape: str
    block_table_shape: str
    kv_cache_shape: str
    metadata_type: str
    metadata_num_actual_tokens: int
    metadata_max_query_len: int
    output_shape: str
    called_attention_kernel: bool
    called_model_forward: bool


@dataclass(frozen=True)
class MLATimingSmokeResult:
    target_backend: str
    kernel_entrypoint: str
    phase: str
    topology_key: str
    attention_actual_tokens: int
    attention_max_query_len: int
    num_tokens_padded: int
    query_shape: str
    kv_cache_shape: str
    metadata_type: str
    output_shape: str
    warmup_iters: int
    measure_iters: int
    wall_ms_mean: float
    wall_ms_p50: float
    wall_ms_p99: float
    cuda_event_ms_mean: float
    cuda_event_ms_p50: float
    cuda_event_ms_p99: float
    called_attention_kernel: bool
    called_model_forward: bool


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["state-smoke", "timing-smoke"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", default="mixed")
    parser.add_argument("--topology-key", default=DEFAULT_TOPOLOGY_KEY)
    parser.add_argument("--attention-actual-tokens", type=int, default=241)
    parser.add_argument("--attention-max-query-len", type=int, default=16)
    parser.add_argument("--num-tokens-padded", type=int, default=248)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--measure-iters", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def validate_state_smoke_key(args: argparse.Namespace) -> None:
    key = (
        args.phase,
        args.attention_actual_tokens,
        args.attention_max_query_len,
        args.num_tokens_padded,
    )
    if key != STATE_SMOKE_KEY:
        raise SystemExit(
            "MLA state-smoke only supports "
            "phase=mixed, attention_actual_tokens=241, attention_max_query_len=16, num_tokens_padded=248"
        )
    if args.topology_key != DEFAULT_TOPOLOGY_KEY:
        raise SystemExit(f"MLA state-smoke only supports topology_key={DEFAULT_TOPOLOGY_KEY}")


def _format_bool(value: bool) -> str:
    return str(value).lower()


def build_state_smoke_row(result: MLAStateSmokeResult) -> dict[str, str]:
    if result.called_model_forward:
        raise SystemExit("MLA state-smoke must not call model forward")
    if not result.called_attention_kernel:
        raise SystemExit("MLA state-smoke did not call attention kernel")
    return {
        "smoke_name": "attention_kernel_mla_state_smoke",
        "phase": result.phase,
        "topology_key": result.topology_key,
        "target_backend": result.target_backend,
        "kernel_entrypoint": result.kernel_entrypoint,
        "attention_actual_tokens": str(result.attention_actual_tokens),
        "attention_max_query_len": str(result.attention_max_query_len),
        "num_tokens_padded": str(result.num_tokens_padded),
        "local_num_heads": str(result.local_num_heads),
        "local_num_kv_heads": str(result.local_num_kv_heads),
        "q_lora_rank": str(result.q_lora_rank),
        "kv_lora_rank": str(result.kv_lora_rank),
        "qk_head_dim": str(result.qk_head_dim),
        "mla_head_size": str(result.mla_head_size),
        "v_head_dim": str(result.v_head_dim),
        "query_shape": result.query_shape,
        "kv_c_shape": result.kv_c_shape,
        "k_pe_shape": result.k_pe_shape,
        "block_table_shape": result.block_table_shape,
        "kv_cache_shape": result.kv_cache_shape,
        "metadata_type": result.metadata_type,
        "metadata_num_actual_tokens": str(result.metadata_num_actual_tokens),
        "metadata_max_query_len": str(result.metadata_max_query_len),
        "output_shape": result.output_shape,
        "called_attention_kernel": _format_bool(result.called_attention_kernel),
        "called_model_forward": _format_bool(result.called_model_forward),
        "timing": "none",
        "valid_for_default": "false",
    }


def build_timing_smoke_row(result: MLATimingSmokeResult) -> dict[str, str]:
    if result.called_model_forward:
        raise SystemExit("MLA timing-smoke must not call model forward")
    if not result.called_attention_kernel:
        raise SystemExit("MLA timing-smoke did not call attention kernel")
    if result.kernel_entrypoint != "forward_mqa":
        raise SystemExit("MLA timing-smoke only supports forward_mqa")
    return {
        "smoke_name": "mla_forward_mqa_timing_smoke",
        "phase": result.phase,
        "topology_key": result.topology_key,
        "target_backend": result.target_backend,
        "kernel_entrypoint": result.kernel_entrypoint,
        "segment": "forward_mqa_only",
        "attention_actual_tokens": str(result.attention_actual_tokens),
        "attention_max_query_len": str(result.attention_max_query_len),
        "num_tokens_padded": str(result.num_tokens_padded),
        "query_shape": result.query_shape,
        "kv_cache_shape": result.kv_cache_shape,
        "metadata_type": result.metadata_type,
        "output_shape": result.output_shape,
        "warmup_iters": str(result.warmup_iters),
        "measure_iters": str(result.measure_iters),
        "wall_ms_mean": f"{result.wall_ms_mean:.6f}",
        "wall_ms_p50": f"{result.wall_ms_p50:.6f}",
        "wall_ms_p99": f"{result.wall_ms_p99:.6f}",
        "cuda_event_ms_mean": f"{result.cuda_event_ms_mean:.6f}",
        "cuda_event_ms_p50": f"{result.cuda_event_ms_p50:.6f}",
        "cuda_event_ms_p99": f"{result.cuda_event_ms_p99:.6f}",
        "called_attention_kernel": _format_bool(result.called_attention_kernel),
        "called_model_forward": _format_bool(result.called_model_forward),
        "valid_for_default": "false",
    }


def write_smoke_csv(path: Path, row: dict[str, str], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerow(row)


def run_mla_state_smoke(**kwargs: object) -> MLAStateSmokeResult:
    from collector.vllm import collect_mla

    result = collect_mla.run_mla_state_smoke(**kwargs)
    return MLAStateSmokeResult(**result)


def run_mla_timing_smoke(**kwargs: object) -> MLATimingSmokeResult:
    from collector.vllm import collect_mla

    result = collect_mla.run_mla_timing_smoke(**kwargs)
    return MLATimingSmokeResult(**result)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_state_smoke_key(args)
    common_kwargs = {
        "phase": args.phase,
        "topology_key": args.topology_key,
        "attention_actual_tokens": args.attention_actual_tokens,
        "attention_max_query_len": args.attention_max_query_len,
        "num_tokens_padded": args.num_tokens_padded,
        "device": args.device,
    }
    if args.mode == "state-smoke":
        state_result = run_mla_state_smoke(**common_kwargs)
        write_smoke_csv(Path(args.output), build_state_smoke_row(state_result), MLA_STATE_SMOKE_COLUMNS)
    else:
        timing_result = run_mla_timing_smoke(
            **common_kwargs,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
        )
        write_smoke_csv(Path(args.output), build_timing_smoke_row(timing_result), MLA_TIMING_SMOKE_COLUMNS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
