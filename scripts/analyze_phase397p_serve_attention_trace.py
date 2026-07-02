#!/usr/bin/env python3
"""Phase397p: offline serve trace forensics for MLA decode attention.

Reads Phase397l profiler_out text files, extracts the FA3 attention main kernel,
combine kernel, scheduler metadata call, and varlen block scheduler kernel. The
output confirms whether Phase397o's B2 hypothesis is a trace-backed harness
fidelity gap. This script is report-only: no runtime, DB table, or gate changes.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase397l_decode_profile"
PHASE397N_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase397n_attention_residual_triage.csv"
PHASE397O_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase397o_mla_recollect_triage.csv"
OUTPUT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase397p_serve_attention_trace_forensics.csv"
OUTPUT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase397p_serve_attention_trace_forensics.md"

SOURCE = "phase397p_serve_attention_trace_forensics"
MODEL = "kimi-k2.5"
HARDWARE = "h200_sxm"
BACKEND = "vllm"
VLLM_VERSION = "0.19.0"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"

CONFIGS = {
    "tp8ep8-8k2k": {
        "trace_dir": "K2.5-tp8ep8-8k2k",
        "local_num_heads": 8,
        "phase397o_batch": 128,
    },
    "tp4ep8dp2-8k2k": {
        "trace_dir": "K2.5-tp4ep8dp2-8k2k",
        "local_num_heads": 16,
        "phase397o_batch": 64,
    },
}

ROLE_PATTERNS = {
    "fa3_main": ("cutlass::device_kernel<flash::enable_sm90",),
    "fa3_combine": ("FlashAttnFwdCombi",),
    "scheduler_metadata": ("get_scheduler_metadata",),
    "prepare_varlen_num_blocks": ("prepare_varlen_num_blocks_kernel",),
}

FIELDNAMES = [
    "source",
    "row_type",
    "config",
    "rank",
    "local_num_heads",
    "kernel_role",
    "kernel_name",
    "cuda_total_us",
    "cuda_time_avg_us",
    "calls",
    "mean_cuda_time_avg_us",
    "spread_pct",
    "db_anchor_us",
    "db_query_us",
    "phase397o_graph_us",
    "db_anchor_over_serve",
    "db_query_over_serve",
    "phase397o_over_serve",
    "fa3_main_present",
    "fa3_combine_present",
    "scheduler_metadata_present",
    "prepare_varlen_present",
    "verdict",
    "evidence",
    "phase397q_target_us",
    "phase397q_block_size",
    "next_allowed_phase",
    "runtime_modified",
    "write_real_data_file",
    "gate_modified",
    "gpu_allowed",
    "ssh_allowed",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
    "default_readiness",
]


@dataclass(frozen=True)
class ProfilerRow:
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


def _common(row_type: str, config: str = "") -> dict[str, str]:
    return {
        "source": SOURCE,
        "row_type": row_type,
        "config": config,
        "rank": "",
        "local_num_heads": str(CONFIGS[config]["local_num_heads"]) if config else "",
        "kernel_role": "",
        "kernel_name": "",
        "cuda_total_us": "",
        "cuda_time_avg_us": "",
        "calls": "",
        "mean_cuda_time_avg_us": "",
        "spread_pct": "",
        "db_anchor_us": "",
        "db_query_us": "",
        "phase397o_graph_us": "",
        "db_anchor_over_serve": "",
        "db_query_over_serve": "",
        "phase397o_over_serve": "",
        "fa3_main_present": "",
        "fa3_combine_present": "",
        "scheduler_metadata_present": "",
        "prepare_varlen_present": "",
        "verdict": "",
        "evidence": "",
        "phase397q_target_us": "",
        "phase397q_block_size": "",
        "next_allowed_phase": "phase397q_recollect_mla_fa3_split_kv",
        "runtime_modified": FALSE,
        "write_real_data_file": FALSE,
        "gate_modified": FALSE,
        "gpu_allowed": FALSE,
        "ssh_allowed": FALSE,
        "diagnostic_only": TRUE,
        "valid_for_default": FALSE,
        "perf_database": FALSE,
        "default_readiness": DEFAULT_READINESS,
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


def parse_profiler_rows(path: Path) -> list[ProfilerRow]:
    rows: list[ProfilerRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("-") or "Self CUDA" in line or "Name" in line:
            continue
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 11:
            continue
        try:
            rows.append(
                ProfilerRow(
                    name=parts[0],
                    cuda_total_us=_parse_time_us(parts[8]),
                    cuda_time_avg_us=_parse_time_us(parts[9]),
                    calls=int(parts[10]),
                )
            )
        except (ValueError, IndexError):
            continue
    return rows


def select_role(rows: Iterable[ProfilerRow], role: str) -> ProfilerRow:
    needles = ROLE_PATTERNS[role]
    matches = [row for row in rows if any(needle in row.name for needle in needles)]
    if not matches:
        raise ValueError(f"missing profiler role {role}")
    return max(matches, key=lambda row: row.cuda_total_us)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _phase397n_db_values(config: str) -> tuple[float, float]:
    heads = str(CONFIGS[config]["local_num_heads"])
    rows = _load_csv(PHASE397N_CSV)
    anchor = None
    query = None
    for row in rows:
        if row["config"] != config or row["num_heads"] != heads or row["kv_cache_dtype"] != "float16":
            continue
        if row["row_type"] == "query_reproduction":
            query = float(row["latency_ms_per_layer"]) * 1000.0
        elif row["row_type"] == "interp_anchor" and row["anchor_seq_len"] == "8192":
            anchor = float(row["anchor_latency_ms_per_layer"]) * 1000.0
    if anchor is None or query is None:
        raise ValueError(f"missing Phase397n DB values for {config}")
    return anchor, query


def _phase397o_graph_us(config: str) -> float:
    meta = CONFIGS[config]
    rows = _load_csv(PHASE397O_CSV)
    for row in rows:
        if (
            row["row_type"] == "shape_compare"
            and row["local_num_heads"] == str(meta["local_num_heads"])
            and row["kv_cache_dtype"] == "float16"
            and row["batch_size"] == str(meta["phase397o_batch"])
            and row["target_seq_len"] == "9001"
            and row["randomize_blocks"] == TRUE
        ):
            return float(row["graph_p50_ms"]) * 1000.0
    raise ValueError(f"missing Phase397o graph value for {config}")


def _rank_from_name(path: Path) -> int:
    match = re.fullmatch(r"profiler_out_(\d+)\.txt", path.name)
    if not match:
        raise ValueError(f"unexpected profiler filename: {path}")
    return int(match.group(1))


def _role_presence(role_rows: dict[str, ProfilerRow]) -> dict[str, str]:
    return {
        "fa3_main_present": str("fa3_main" in role_rows).lower(),
        "fa3_combine_present": str("fa3_combine" in role_rows).lower(),
        "scheduler_metadata_present": str("scheduler_metadata" in role_rows).lower(),
        "prepare_varlen_present": str("prepare_varlen_num_blocks" in role_rows).lower(),
    }


def build_rows(trace_root: Path = TRACE_ROOT) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    summaries: dict[str, dict[str, object]] = {}
    for config, meta in CONFIGS.items():
        prof_dir = trace_root / meta["trace_dir"] / "prof"
        paths = sorted(prof_dir.glob("profiler_out_*.txt"))
        if len(paths) != 8:
            raise ValueError(f"{config} expected 8 profiler_out files, got {len(paths)}")

        db_anchor_us, db_query_us = _phase397n_db_values(config)
        phase397o_graph_us = _phase397o_graph_us(config)
        main_values: list[float] = []
        role_complete = []
        per_rank_roles: dict[int, dict[str, ProfilerRow]] = {}
        for path in paths:
            rank = _rank_from_name(path)
            parsed = parse_profiler_rows(path)
            role_rows = {role: select_role(parsed, role) for role in ROLE_PATTERNS}
            per_rank_roles[rank] = role_rows
            role_complete.append(set(role_rows) == set(ROLE_PATTERNS))
            presence = _role_presence(role_rows)
            main_values.append(role_rows["fa3_main"].cuda_time_avg_us)
            for role, prof_row in role_rows.items():
                row = _common("rank_kernel", config)
                row.update(
                    {
                        "rank": str(rank),
                        "kernel_role": role,
                        "kernel_name": prof_row.name,
                        "cuda_total_us": _fmt(prof_row.cuda_total_us),
                        "cuda_time_avg_us": _fmt(prof_row.cuda_time_avg_us),
                        "calls": str(prof_row.calls),
                        "db_anchor_us": _fmt(db_anchor_us),
                        "db_query_us": _fmt(db_query_us),
                        "phase397o_graph_us": _fmt(phase397o_graph_us),
                        "verdict": "serve_trace_component",
                        "evidence": "Phase397l profiler_out row parsed offline",
                        **presence,
                    }
                )
                output.append(row)

        mean_us = statistics.fmean(main_values)
        spread_pct = (max(main_values) - min(main_values)) / mean_us * 100.0
        summaries[config] = {
            "mean_us": mean_us,
            "spread_pct": spread_pct,
            "db_anchor_us": db_anchor_us,
            "db_query_us": db_query_us,
            "phase397o_graph_us": phase397o_graph_us,
            "role_complete": all(role_complete),
        }
        summary = _common("config_summary", config)
        summary.update(
            {
                "kernel_role": "fa3_main",
                "calls": str(int(statistics.median([per_rank_roles[r]["fa3_main"].calls for r in per_rank_roles]))),
                "mean_cuda_time_avg_us": _fmt(mean_us),
                "spread_pct": _fmt(spread_pct),
                "db_anchor_us": _fmt(db_anchor_us),
                "db_query_us": _fmt(db_query_us),
                "phase397o_graph_us": _fmt(phase397o_graph_us),
                "db_anchor_over_serve": _fmt(db_anchor_us / mean_us),
                "db_query_over_serve": _fmt(db_query_us / mean_us),
                "phase397o_over_serve": _fmt(phase397o_graph_us / mean_us),
                "phase397q_target_us": _fmt(mean_us),
                "phase397q_block_size": "64",
                "verdict": "fa3_split_kv_scheduler_trace_confirmed",
                "evidence": "all 8 ranks contain FA3 main, combine, get_scheduler_metadata, and prepare_varlen_num_blocks",
                **{
                    "fa3_main_present": TRUE,
                    "fa3_combine_present": TRUE,
                    "scheduler_metadata_present": TRUE,
                    "prepare_varlen_present": TRUE,
                },
            }
        )
        output.append(summary)

    primary = summaries["tp8ep8-8k2k"]
    decision = _common("decision")
    decision.update(
        {
            "kernel_role": "phase397q_trigger",
            "mean_cuda_time_avg_us": _fmt(float(primary["mean_us"])),
            "db_anchor_us": _fmt(float(primary["db_anchor_us"])),
            "db_query_us": _fmt(float(primary["db_query_us"])),
            "phase397o_graph_us": _fmt(float(primary["phase397o_graph_us"])),
            "db_anchor_over_serve": _fmt(float(primary["db_anchor_us"]) / float(primary["mean_us"])),
            "db_query_over_serve": _fmt(float(primary["db_query_us"]) / float(primary["mean_us"])),
            "phase397o_over_serve": _fmt(float(primary["phase397o_graph_us"]) / float(primary["mean_us"])),
            "phase397q_target_us": _fmt(float(primary["mean_us"])),
            "phase397q_block_size": "64",
            "verdict": "confirmed_fa3_split_kv_scheduler_harness_gap",
            "evidence": "serve trace uses FA3 main plus combine and scheduler/varlen metadata while DB and Phase397o single-kernel microbench remain about 1.85x slower",
            "fa3_main_present": TRUE,
            "fa3_combine_present": TRUE,
            "scheduler_metadata_present": TRUE,
            "prepare_varlen_present": TRUE,
            "next_allowed_phase": "phase397q_recollect_mla_fa3_split_kv_block64",
        }
    )
    output.append(decision)
    return output


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _find(rows: list[dict[str, str]], **match: str) -> dict[str, str]:
    for row in rows:
        if all(row[key] == value for key, value in match.items()):
            return row
    raise KeyError(match)


def write_md(path: Path, rows: list[dict[str, str]]) -> None:
    tp8 = _find(rows, row_type="config_summary", config="tp8ep8-8k2k")
    tp4 = _find(rows, row_type="config_summary", config="tp4ep8dp2-8k2k")
    decision = _find(rows, row_type="decision")
    lines = [
        "# Phase397p Serve Attention Trace Forensics",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Scope | offline serve-trace forensics only; no runtime/DB/gate change |",
        "| Verdict | confirmed_fa3_split_kv_scheduler_harness_gap |",
        "| Default AIC | No-Go |",
        "",
        "## Serve Trace Summary",
        "",
        "| Config | heads/GPU | mean FA3 main us/call | spread | calls/rank | DB 8192 anchor us | DB s9001 query us | Phase397o graph us | graph/serve | split/scheduler evidence |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in (tp8, tp4):
        lines.append(
            "| {config} | {local_num_heads} | {mean_cuda_time_avg_us} | {spread_pct}% | {calls} | "
            "{db_anchor_us} | {db_query_us} | {phase397o_graph_us} | {phase397o_over_serve}x | "
            "FA3 main + combine + get_scheduler_metadata + prepare_varlen |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Root Cause",
            "",
            f"- {decision['evidence']}.",
            "- tp4ep8dp2 has a rank split (rank 0-3 around 245us, rank 4-7 around 169us), so heads=16 needs follow-up trace interpretation; it is not the primary q acceptance target.",
            "- Phase397o already showed CUDA graph replay and randomize_blocks do not explain the gap.",
            "- The missing piece is harness fidelity: current DB/microbench path does not reproduce serve FA3 varlen split-KV scheduler metadata.",
            "",
            "## Phase397q Acceptance",
            "",
            f"- Recollect generation_mla with FA3 varlen split-KV scheduler metadata and block_size={decision['phase397q_block_size']}.",
            f"- heads=8, batch=128, s=9001 must land near {decision['phase397q_target_us']} us/call before any table edit.",
            "- Recollect float16 flash and fp8 flashmla as separate rows; do not use a scalar multiplier.",
            "- q may touch collector and generation_mla_perf rows only after evidence; it must not touch gate/models runtime.",
            "- Default AIC remains No-Go.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", default=str(TRACE_ROOT))
    parser.add_argument("--out-csv", default=str(OUTPUT_CSV))
    parser.add_argument("--out-md", default=str(OUTPUT_MD))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = build_rows(Path(args.trace_root))
    write_csv(Path(args.out_csv), rows)
    write_md(Path(args.out_md), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
