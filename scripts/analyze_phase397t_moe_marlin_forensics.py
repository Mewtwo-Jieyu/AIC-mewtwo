#!/usr/bin/env python3
"""Phase397t: MoE marlin serve-vs-microbench forensics.

This is report-only. It summarizes the serve trace marlin_moe_wna16 kernel,
contrasts it with the collector's direct fused_marlin_moe call shape, and ranks
mechanisms to test on H200 before any moe_perf rewrite.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397l_decode_profile/K2.5-tp8ep8-8k2k"
)
PROF_DIR = TRACE_DIR / "prof"
OP_BREAKDOWN = TRACE_DIR / "op_breakdown.csv"
COLLECT_MOE = REPO_ROOT / "collector/vllm/collect_moe.py"
OUTPUT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase397t_moe_marlin_forensics.csv"
OUTPUT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase397t_moe_marlin_forensics.md"

SOURCE = "phase397t_moe_marlin_forensics"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"
NUM_MOE_LAYERS = 60
SERVE_DECODE_STEPS = 26
PHASE397S_POWER_LAW_MS = 0.47062736511230463

FIELDNAMES = [
    "source",
    "row_type",
    "rank",
    "verdict",
    "evidence",
    "kernel_name",
    "serve_cuda_total_us",
    "serve_us_per_call",
    "serve_mean_us_per_call",
    "serve_spread_pct",
    "calls_per_rank",
    "ranks",
    "inferred_decode_steps",
    "serve_ms_per_layer",
    "phase397l_anchor_ms_per_layer",
    "phase397s_microbench_ms_per_layer",
    "microbench_over_serve_anchor",
    "collector_hidden_m_source",
    "collector_effective_m",
    "collector_uses_expert_map",
    "collector_direct_fused_marlin_moe",
    "vllm_batched_experts_path",
    "vllm_standard_experts_path",
    "vllm_align_ignore_invalid_experts",
    "vllm_block_size_candidates",
    "vllm_source_basis",
    "mechanism",
    "mechanism_rank",
    "prediction",
    "decision",
    "gpu_allowed_next",
    "write_moe_perf",
    "runtime_modified",
    "gate_modified",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
    "default_readiness",
    "next_allowed_phase",
]


@dataclass(frozen=True)
class ProfilerRow:
    rank: int
    name: str
    cuda_total_us: float
    cuda_time_avg_us: float
    calls: int


def _fmt(value: float | str) -> str:
    if isinstance(value, str):
        return value
    if math.isnan(value):
        return ""
    return f"{value:.6f}"


def _common(row_type: str, verdict: str = "", evidence: str = "") -> dict[str, str]:
    return {
        "source": SOURCE,
        "row_type": row_type,
        "rank": "",
        "verdict": verdict,
        "evidence": evidence,
        "kernel_name": "",
        "serve_cuda_total_us": "",
        "serve_us_per_call": "",
        "serve_mean_us_per_call": "",
        "serve_spread_pct": "",
        "calls_per_rank": "",
        "ranks": "",
        "inferred_decode_steps": "",
        "serve_ms_per_layer": "",
        "phase397l_anchor_ms_per_layer": "",
        "phase397s_microbench_ms_per_layer": "",
        "microbench_over_serve_anchor": "",
        "collector_hidden_m_source": "",
        "collector_effective_m": "",
        "collector_uses_expert_map": "",
        "collector_direct_fused_marlin_moe": "",
        "vllm_batched_experts_path": "",
        "vllm_standard_experts_path": "",
        "vllm_align_ignore_invalid_experts": "",
        "vllm_block_size_candidates": "",
        "vllm_source_basis": "",
        "mechanism": "",
        "mechanism_rank": "",
        "prediction": "",
        "decision": "",
        "gpu_allowed_next": FALSE,
        "write_moe_perf": FALSE,
        "runtime_modified": FALSE,
        "gate_modified": FALSE,
        "diagnostic_only": TRUE,
        "valid_for_default": FALSE,
        "perf_database": FALSE,
        "default_readiness": DEFAULT_READINESS,
        "next_allowed_phase": "phase397t_h200_marlin_mechanism_gate",
    }


def _parse_time_us(token: str) -> float:
    match = re.fullmatch(r"([0-9.]+)(us|ms|s)", token)
    if not match:
        raise ValueError(f"cannot parse profiler time token: {token}")
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "us":
        return value
    if unit == "ms":
        return value * 1000.0
    if unit == "s":
        return value * 1_000_000.0
    raise ValueError(token)


def _rank_from_path(path: Path) -> int:
    match = re.fullmatch(r"profiler_out_(\d+)\.txt", path.name)
    if not match:
        raise ValueError(f"unexpected profiler filename: {path}")
    return int(match.group(1))


def _parse_marlin_row(path: Path) -> ProfilerRow:
    for line in path.read_text(encoding="utf-8").splitlines():
        if "marlin_moe_wna16::Marlin" not in line:
            continue
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 11:
            raise ValueError(f"malformed marlin profiler row in {path}")
        return ProfilerRow(
            rank=_rank_from_path(path),
            name=parts[0],
            cuda_total_us=_parse_time_us(parts[8]),
            cuda_time_avg_us=_parse_time_us(parts[9]),
            calls=int(parts[10]),
        )
    raise ValueError(f"missing marlin profiler row in {path}")


def _load_op_breakdown(path: Path = OP_BREAKDOWN) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values: dict[str, float] = {}
    for row in rows:
        if row["ms_per_iter"]:
            values[row["category"]] = float(row["ms_per_iter"])
    required = {"moe_expert_gemm", "decode_bs"}
    missing = required - set(values)
    if missing:
        raise ValueError(f"missing op_breakdown rows: {sorted(missing)}")
    return values


def _collector_facts(path: Path = COLLECT_MOE) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    required = {
        "hidden_states[: tw.shape[0]]": "hidden_states[: tw.shape[0]]",
        "expert_map=expert_map": "expert_map=expert_map",
        "fused_marlin_moe(": "fused_marlin_moe",
    }
    missing = [needle for needle in required if needle not in text]
    if missing:
        raise ValueError(f"collector path changed; missing {missing}")
    return {
        "collector_hidden_m_source": "hidden_states[:tw.shape[0]]",
        "collector_effective_m": "128",
        "collector_uses_expert_map": TRUE,
        "collector_direct_fused_marlin_moe": TRUE,
    }


def _vllm_static_facts() -> dict[str, str]:
    return {
        "vllm_batched_experts_path": TRUE,
        "vllm_standard_experts_path": TRUE,
        "vllm_align_ignore_invalid_experts": TRUE,
        "vllm_block_size_candidates": "8/16/32/48/64",
        "vllm_source_basis": (
            "vLLM 0.19.0 fused_marlin_moe.py and "
            "compressed_tensors_moe.py read on H200 runtime"
        ),
    }


def build_rows(prof_dir: Path = PROF_DIR) -> list[dict[str, str]]:
    paths = sorted(prof_dir.glob("profiler_out_*.txt"))
    if len(paths) != 8:
        raise ValueError(f"expected 8 profiler_out files, got {len(paths)}")
    marlin_rows = [_parse_marlin_row(path) for path in paths]
    if len({row.calls for row in marlin_rows}) != 1:
        raise ValueError("marlin call count must be identical across ranks")

    op_breakdown = _load_op_breakdown()
    anchor_ms = op_breakdown["moe_expert_gemm"] / NUM_MOE_LAYERS
    serve_mean_us = statistics.mean(row.cuda_time_avg_us for row in marlin_rows)
    serve_spread_pct = (
        (max(row.cuda_time_avg_us for row in marlin_rows) - min(row.cuda_time_avg_us for row in marlin_rows))
        / serve_mean_us
        * 100.0
    )
    serve_ms_per_layer = serve_mean_us * 2.0 / 1000.0
    calls_per_rank = marlin_rows[0].calls
    inferred_steps = round(calls_per_rank / (NUM_MOE_LAYERS * 2))

    rows: list[dict[str, str]] = []
    for item in marlin_rows:
        row = _common(
            "serve_marlin_rank",
            "serve_marlin_rank_observed",
            "Phase397l profiler row for marlin_moe_wna16::Marlin.",
        )
        row.update(
            {
                "rank": str(item.rank),
                "kernel_name": item.name,
                "serve_cuda_total_us": _fmt(item.cuda_total_us),
                "serve_us_per_call": _fmt(item.cuda_time_avg_us),
                "calls_per_rank": str(item.calls),
            }
        )
        rows.append(row)

    row = _common(
        "serve_marlin_summary",
        "serve_marlin_anchor_confirmed",
        "Serve trace shows one marlin_moe_wna16 kernel family, about two GEMM calls per MoE layer per decode step.",
    )
    row.update(
        {
            "serve_mean_us_per_call": _fmt(serve_mean_us),
            "serve_spread_pct": _fmt(serve_spread_pct),
            "calls_per_rank": str(calls_per_rank),
            "ranks": str(len(marlin_rows)),
            "inferred_decode_steps": str(inferred_steps),
            "serve_ms_per_layer": _fmt(serve_ms_per_layer),
            "phase397l_anchor_ms_per_layer": _fmt(anchor_ms),
            "phase397s_microbench_ms_per_layer": _fmt(PHASE397S_POWER_LAW_MS),
            "microbench_over_serve_anchor": _fmt(PHASE397S_POWER_LAW_MS / anchor_ms),
        }
    )
    rows.append(row)

    row = _common(
        "collector_static_path",
        "collector_passes_full_m_with_expert_map",
        "Collector calls fused_marlin_moe directly with hidden_states sliced by topk row count and relies on expert_map to ignore non-local experts.",
    )
    row.update(_collector_facts())
    rows.append(row)

    row = _common(
        "vllm_static_path",
        "serve_path_has_standard_and_batched_experts",
        "vLLM 0.19.0 WNA16 Marlin can choose BatchedMarlinExperts for BatchedExperts activation format; fused_marlin_moe aligns with ignore_invalid_experts=True.",
    )
    row.update(_vllm_static_facts())
    rows.append(row)

    mechanisms = [
        (
            "mechanism_rank_1",
            "activation_format_or_effective_m_mismatch",
            "If serve uses BatchedExperts or smaller effective per-rank M, a collector variant that feeds the same effective M should approach the 0.1337 ms/layer anchor.",
            "test_on_h200",
        ),
        (
            "mechanism_rank_2",
            "marlin_block_size_or_template_selection",
            "If M and topk change the selected block_size/template, instrumentation should show a different block_size_m or Marlin template for serve-like inputs.",
            "test_on_h200",
        ),
        (
            "mechanism_rank_3",
            "route_distribution_only",
            "Phase397s already tested power_law, balanced, and power_law_eplb; all stayed far above the anchor.",
            "ruled_out_by_phase397s",
        ),
    ]
    for idx, (row_type, mechanism, prediction, decision) in enumerate(mechanisms, start=1):
        row = _common(
            row_type,
            "mechanism_ranked",
            "Mechanism ranking before H200 reproduction gate.",
        )
        row.update(
            {
                "mechanism": mechanism,
                "mechanism_rank": str(idx),
                "prediction": prediction,
                "decision": decision,
                "gpu_allowed_next": TRUE if decision == "test_on_h200" else FALSE,
            }
        )
        rows.append(row)

    row = _common(
        "decision",
        "h200_mechanism_gate_required_before_table_write",
        "Do not retire k or write moe_perf until a no-k structured collector variant reaches 0.1337 ms/layer +/-15%.",
    )
    row.update(
        {
            "mechanism": "activation_format_or_effective_m_mismatch",
            "decision": "run_phase397t_h200_gate",
            "gpu_allowed_next": TRUE,
        }
    )
    rows.append(row)

    validate_rows(rows)
    return rows


def validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("no Phase397t rows")
    if rows[-1]["row_type"] != "decision":
        raise ValueError("Phase397t final row must be decision")
    for row in rows:
        if row["write_moe_perf"] != FALSE:
            raise ValueError("Phase397t forensics requires write_moe_perf=false")
        if row["runtime_modified"] != FALSE:
            raise ValueError("Phase397t forensics requires runtime_modified=false")
        if row["gate_modified"] != FALSE:
            raise ValueError("Phase397t forensics requires gate_modified=false")
        if row["valid_for_default"] != FALSE:
            raise ValueError("Phase397t forensics requires valid_for_default=false")
        if row["perf_database"] != FALSE:
            raise ValueError("Phase397t forensics requires perf_database=false")
        if row["default_readiness"] != DEFAULT_READINESS:
            raise ValueError("Phase397t forensics requires default_readiness=No-Go")


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict[str, str]], path: Path) -> None:
    validate_rows(rows)
    summary = next(row for row in rows if row["row_type"] == "serve_marlin_summary")
    mechanisms = [row for row in rows if row["row_type"].startswith("mechanism_rank_")]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase397t MoE Marlin Forensics",
        "",
        "Phase397t is report-only until the H200 mechanism gate passes.",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Serve mean marlin us/call | {summary['serve_mean_us_per_call']} |",
        f"| Serve ms/layer from per-call | {summary['serve_ms_per_layer']} |",
        f"| Phase397l anchor ms/layer | {summary['phase397l_anchor_ms_per_layer']} |",
        f"| Phase397s microbench ms/layer | {summary['phase397s_microbench_ms_per_layer']} |",
        f"| Microbench / anchor | {summary['microbench_over_serve_anchor']}x |",
        "",
        "| Rank | Mechanism | Decision |",
        "|---:|---|---|",
    ]
    for row in mechanisms:
        lines.append(f"| {row['mechanism_rank']} | {row['mechanism']} | {row['decision']} |")
    lines.extend(
        [
            "",
            "Next gate: on H200, instrument actual M, padded rows, block_size_m/template, and per-call us.",
            "Only a no-k structured collector variant within 0.1337 ms/layer +/-15% may unlock table rewrites.",
            "",
            "Default AIC remains No-Go.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prof-dir", type=Path, default=PROF_DIR)
    parser.add_argument("--out-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--out-md", type=Path, default=OUTPUT_MD)
    args = parser.parse_args(argv)

    rows = build_rows(args.prof_dir)
    write_csv(rows, args.out_csv)
    write_md(rows, args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
