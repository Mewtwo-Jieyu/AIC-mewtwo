#!/usr/bin/env python3
"""Phase397n: offline triage for the vLLM generation MLA decode residual.

Report-only. This script reproduces the current cb_sim attention value through
PerfDatabase.query_generation_mla, inspects the interpolation anchors in
generation_mla_perf.txt, compares them with Phase397l profiler measurements,
and writes a Phase397o recollection spec. It does not modify runtime code, perf
tables, gates, or Default AIC readiness.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from aiconfigurator.sdk import common  # noqa: E402
from aiconfigurator.sdk.perf_database import PerfDatabase  # noqa: E402


SOURCE = "phase397n_attention_residual_triage"
SYSTEM = "h200_sxm"
BACKEND = "vllm"
VERSION = "0.19.0"
MODEL = "kimi-k2.5"
WORKLOAD = "decode_8k2k_b128"
NUM_LAYERS = 61
TARGET_SEQ_LEN = 9001
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"

GEN_MLA_TABLE = (
    SRC_ROOT
    / "aiconfigurator/systems/data/h200_sxm/vllm/0.19.0/generation_mla_perf.txt"
)
PHASE397L_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase397l_decode_profile"
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase397n_attention_residual_triage.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT / "docs/iter_gap_investigation/phase397n_attention_residual_triage.md"
)


FIELDNAMES = [
    "source",
    "row_type",
    "config",
    "model",
    "hardware",
    "backend",
    "vllm_version",
    "workload",
    "num_heads",
    "batch_size",
    "seq_len",
    "kv_cache_dtype",
    "kernel_source",
    "latency_ms_per_layer",
    "sim_total_ms",
    "measured_attention_ms",
    "measured_per_layer_ms",
    "sim_over_measured",
    "anchor_batch_size",
    "anchor_seq_len",
    "anchor_latency_ms_per_layer",
    "anchor_over_measured_per_layer",
    "candidate",
    "verdict",
    "evidence",
    "phase397o_recollect_item",
    "future_gpu_required",
    "next_allowed_phase",
    "runtime_modified",
    "write_real_data_file",
    "gpu_allowed",
    "ssh_allowed",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
    "default_readiness",
]


@dataclass(frozen=True)
class Case:
    config: str
    num_heads: int
    batch_size: int
    measured_path: Path


CASES = [
    Case(
        config="tp8ep8-8k2k",
        num_heads=8,
        batch_size=128,
        measured_path=PHASE397L_ROOT / "K2.5-tp8ep8-8k2k/op_breakdown.csv",
    ),
    Case(
        config="tp4ep8dp2-8k2k",
        num_heads=16,
        batch_size=69,
        measured_path=PHASE397L_ROOT / "K2.5-tp4ep8dp2-8k2k/op_breakdown.csv",
    ),
]


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _common(row_type: str, config: str = "") -> dict[str, str]:
    return {
        "source": SOURCE,
        "row_type": row_type,
        "config": config,
        "model": MODEL,
        "hardware": SYSTEM,
        "backend": BACKEND,
        "vllm_version": VERSION,
        "workload": WORKLOAD,
        "num_heads": "",
        "batch_size": "",
        "seq_len": "",
        "kv_cache_dtype": "",
        "kernel_source": "",
        "latency_ms_per_layer": "",
        "sim_total_ms": "",
        "measured_attention_ms": "",
        "measured_per_layer_ms": "",
        "sim_over_measured": "",
        "anchor_batch_size": "",
        "anchor_seq_len": "",
        "anchor_latency_ms_per_layer": "",
        "anchor_over_measured_per_layer": "",
        "candidate": "",
        "verdict": "",
        "evidence": "",
        "phase397o_recollect_item": "",
        "future_gpu_required": FALSE,
        "next_allowed_phase": "phase397o_recollect_generation_mla_microbench",
        "runtime_modified": FALSE,
        "write_real_data_file": FALSE,
        "gpu_allowed": FALSE,
        "ssh_allowed": FALSE,
        "diagnostic_only": TRUE,
        "valid_for_default": FALSE,
        "perf_database": FALSE,
        "default_readiness": DEFAULT_READINESS,
    }


def _kv_mode(name: str) -> common.KVCacheQuantMode:
    return common.KVCacheQuantMode[name]


def _kernel_source(kv_dtype: str) -> str:
    if kv_dtype == "fp8":
        return "vllm_flashmla"
    return "vllm_flash_attn_mla"


def _load_measured_attention(path: Path) -> tuple[float, int]:
    if not path.exists():
        raise FileNotFoundError(path)
    measured = None
    decode_bs = None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["category"] == "attention":
                measured = float(row["ms_per_iter"])
            elif row["category"] == "decode_bs":
                decode_bs = int(float(row["ms_per_iter"]))
    if measured is None or decode_bs is None:
        raise ValueError(f"missing attention/decode_bs in {path}")
    return measured, decode_bs


def _load_generation_rows() -> list[dict[str, str]]:
    with GEN_MLA_TABLE.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _actual_seq_len(row: dict[str, str]) -> int:
    return int(row["isl"]) + int(row["step"])


def _bounds(value: int, values: list[int]) -> tuple[int, int]:
    uniq = sorted(set(values))
    if value in uniq:
        return value, value
    lefts = [v for v in uniq if v < value]
    rights = [v for v in uniq if v > value]
    if not lefts or not rights:
        raise ValueError(f"value {value} outside interpolation bounds {uniq[:3]}..{uniq[-3:]}")
    return max(lefts), min(rights)


def _anchor_rows(
    table_rows: list[dict[str, str]],
    case: Case,
    kv_dtype: str,
) -> list[dict[str, str]]:
    filtered = [
        row
        for row in table_rows
        if row["op_name"] == "generation_mla"
        and row["kv_cache_dtype"] == kv_dtype
        and int(row["num_heads"]) == case.num_heads
        and row["kernel_source"] == _kernel_source(kv_dtype)
    ]
    batch_left, batch_right = _bounds(
        case.batch_size, [int(row["batch_size"]) for row in filtered]
    )
    seq_left, seq_right = _bounds(
        TARGET_SEQ_LEN, [_actual_seq_len(row) for row in filtered]
    )
    anchors = []
    for row in filtered:
        batch = int(row["batch_size"])
        seq = _actual_seq_len(row)
        if batch in {batch_left, batch_right} and seq in {seq_left, seq_right}:
            anchors.append(row)
    return sorted(anchors, key=lambda r: (int(r["batch_size"]), _actual_seq_len(r)))


def _query_rows(
    db: PerfDatabase,
    table_rows: list[dict[str, str]],
    case: Case,
) -> list[dict[str, str]]:
    measured_ms, measured_bs = _load_measured_attention(case.measured_path)
    if measured_bs != case.batch_size:
        raise ValueError(
            f"{case.config} measured decode_bs {measured_bs} != expected {case.batch_size}"
        )
    measured_per_layer = measured_ms / NUM_LAYERS
    rows: list[dict[str, str]] = []

    for kv_dtype in ("float16", "fp8"):
        latency = float(
            db.query_generation_mla(
                case.batch_size,
                TARGET_SEQ_LEN,
                case.num_heads,
                _kv_mode(kv_dtype),
            )
        )
        total = latency * NUM_LAYERS
        row = _common("query_reproduction", case.config)
        row.update(
            {
                "num_heads": str(case.num_heads),
                "batch_size": str(case.batch_size),
                "seq_len": str(TARGET_SEQ_LEN),
                "kv_cache_dtype": kv_dtype,
                "kernel_source": _kernel_source(kv_dtype),
                "latency_ms_per_layer": _fmt(latency),
                "sim_total_ms": _fmt(total),
                "measured_attention_ms": _fmt(measured_ms),
                "measured_per_layer_ms": _fmt(measured_per_layer),
                "sim_over_measured": _fmt(total / measured_ms),
                "verdict": (
                    "dtype_not_primary_float16_matches_bf16_runtime"
                    if kv_dtype == "float16"
                    else "fp8_lower_but_still_overpredicts_tp8"
                ),
            }
        )
        rows.append(row)

        for anchor in _anchor_rows(table_rows, case, kv_dtype):
            anchor_latency = float(anchor["latency"])
            anchor_row = _common("interp_anchor", case.config)
            anchor_row.update(
                {
                    "num_heads": str(case.num_heads),
                    "batch_size": str(case.batch_size),
                    "seq_len": str(TARGET_SEQ_LEN),
                    "kv_cache_dtype": kv_dtype,
                    "kernel_source": anchor["kernel_source"],
                    "measured_attention_ms": _fmt(measured_ms),
                    "measured_per_layer_ms": _fmt(measured_per_layer),
                    "anchor_batch_size": anchor["batch_size"],
                    "anchor_seq_len": str(_actual_seq_len(anchor)),
                    "anchor_latency_ms_per_layer": _fmt(anchor_latency),
                    "anchor_over_measured_per_layer": _fmt(
                        anchor_latency / measured_per_layer
                    ),
                    "verdict": "anchor_value_high" if kv_dtype == "float16" else "dtype_control_anchor",
                }
            )
            rows.append(anchor_row)

    return rows


def _candidate_rows(query_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    tp8_f16 = next(
        r
        for r in query_rows
        if r["row_type"] == "query_reproduction"
        and r["config"] == "tp8ep8-8k2k"
        and r["kv_cache_dtype"] == "float16"
    )
    tp8_fp8 = next(
        r
        for r in query_rows
        if r["row_type"] == "query_reproduction"
        and r["config"] == "tp8ep8-8k2k"
        and r["kv_cache_dtype"] == "fp8"
    )
    tp8_anchor_8192 = next(
        r
        for r in query_rows
        if r["row_type"] == "interp_anchor"
        and r["config"] == "tp8ep8-8k2k"
        and r["kv_cache_dtype"] == "float16"
        and r["anchor_batch_size"] == "128"
        and r["anchor_seq_len"] == "8192"
    )

    candidates = [
        (
            "layer_mtp_count",
            "eliminated_offline",
            "Kimi-K2.5 nextn=0 gives _mtp_scale_factor=1; GenerationMLA is charged as 61 layers, matching Phase397l.",
        ),
        (
            "dtype_key",
            "not_primary",
            (
                "Runtime kernel is bf16/FlashAttnFwdSm90, mapping to the float16 row. "
                f"fp8 still predicts {tp8_fp8['sim_total_ms']}ms vs measured {tp8_fp8['measured_attention_ms']}ms "
                f"({tp8_fp8['sim_over_measured']}x)."
            ),
        ),
        (
            "interp_overshoot",
            "not_primary",
            (
                "Interpolation to s=9001 stays between 8192 and 16384 anchors. "
                f"The 8192 float16 anchor is already {tp8_anchor_8192['anchor_latency_ms_per_layer']}ms/layer "
                f"= {tp8_anchor_8192['anchor_over_measured_per_layer']}x measured per-layer."
            ),
        ),
        (
            "anchor_microbench_high",
            "primary_hypothesis",
            (
                f"Current query gives {tp8_f16['latency_ms_per_layer']}ms/layer and "
                f"{tp8_f16['sim_total_ms']}ms total vs {tp8_f16['measured_attention_ms']}ms measured "
                f"({tp8_f16['sim_over_measured']}x). The nearest low anchor itself is high."
            ),
        ),
        (
            "collector_workload_method",
            "needs_phase397o",
            (
                "collector/vllm/collect_mla.py times eager forward_mqa with CUDA events and no graph replay "
                "(lines 303-317); paged/varlen layout and real serve kernel path need GPU recollection."
            ),
        ),
    ]
    rows = []
    for candidate, verdict, evidence in candidates:
        row = _common("hypothesis_rank", "tp8ep8-8k2k")
        row.update(
            {
                "kv_cache_dtype": "float16",
                "candidate": candidate,
                "verdict": verdict,
                "evidence": evidence,
            }
        )
        rows.append(row)
    return rows


def _phase397o_rows() -> list[dict[str, str]]:
    items = [
        (
            "cuda_graph_vs_eager_float16_heads8_b128_s9001",
            "num_heads=8,batch=128,kv_dtype=float16,seq_len around 8192/9001/16384; compare eager DB row vs CUDA-graph replay vs Phase397l serve.",
        ),
        (
            "cuda_graph_vs_eager_fp8_heads8_b128_s9001",
            "num_heads=8,batch=128,kv_dtype=fp8,seq_len around 8192/9001/16384; dtype control only, not expected primary fix.",
        ),
        (
            "cuda_graph_vs_eager_float16_heads16_b128_s9001",
            "num_heads=16,batch=128,kv_dtype=float16,seq_len around 8192/9001/16384; tp4 head-count control.",
        ),
        (
            "cuda_graph_vs_eager_fp8_heads16_b128_s9001",
            "num_heads=16,batch=128,kv_dtype=fp8,seq_len around 8192/9001/16384; tp4 head-count dtype control.",
        ),
        (
            "paged_varlen_layout_match",
            "capture attn_metadata/kv-cache layout path close to real decode; classify timing-method vs kernel/workload mismatch.",
        ),
    ]
    rows = []
    for item, evidence in items:
        row = _common("phase397o_recollect_spec", "")
        row.update(
            {
                "phase397o_recollect_item": item,
                "future_gpu_required": TRUE,
                "verdict": "recollect_microbench_required",
                "evidence": evidence,
            }
        )
        rows.append(row)
    return rows


def analyze_phase397n() -> list[dict[str, str]]:
    db = PerfDatabase(
        system=SYSTEM,
        backend=BACKEND,
        version=VERSION,
        systems_root=str(SRC_ROOT / "aiconfigurator/systems"),
    )
    table_rows = _load_generation_rows()
    rows: list[dict[str, str]] = []
    for case in CASES:
        rows.extend(_query_rows(db, table_rows, case))
    rows.extend(_candidate_rows(rows))
    rows.extend(_phase397o_rows())
    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Phase397n produced no rows")
    for row in rows:
        label = f"{row['row_type']}/{row['config'] or '-'}"
        if set(row) != set(FIELDNAMES):
            raise ValueError(f"{label} fields mismatch")
        for guard in (
            "runtime_modified",
            "write_real_data_file",
            "gpu_allowed",
            "ssh_allowed",
            "valid_for_default",
            "perf_database",
        ):
            if row[guard] != FALSE:
                raise ValueError(f"{label} {guard} must be false")
        if row["diagnostic_only"] != TRUE:
            raise ValueError(f"{label} diagnostic_only must be true")
        if row["default_readiness"] != DEFAULT_READINESS:
            raise ValueError(f"{label} default_readiness must be No-Go")

    queries = {
        (r["config"], r["kv_cache_dtype"]): r
        for r in rows
        if r["row_type"] == "query_reproduction"
    }
    required = {
        ("tp8ep8-8k2k", "float16"),
        ("tp8ep8-8k2k", "fp8"),
        ("tp4ep8dp2-8k2k", "float16"),
        ("tp4ep8dp2-8k2k", "fp8"),
    }
    if set(queries) != required:
        raise ValueError(f"unexpected query rows: {sorted(queries)}")
    if abs(float(queries[("tp8ep8-8k2k", "float16")]["sim_total_ms"]) - 31.891) > 0.001:
        raise ValueError("tp8 float16 total no longer reproduces 31.891ms")
    if float(queries[("tp8ep8-8k2k", "fp8")]["sim_over_measured"]) < 1.4:
        raise ValueError("fp8 row should still overpredict enough to rule out dtype as primary")

    candidates = {r["candidate"]: r["verdict"] for r in rows if r["row_type"] == "hypothesis_rank"}
    if candidates.get("anchor_microbench_high") != "primary_hypothesis":
        raise ValueError("primary hypothesis must be anchor_microbench_high")
    if candidates.get("collector_workload_method") != "needs_phase397o":
        raise ValueError("collector workload/method must route to Phase397o")


def write_phase397n_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase397n_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)

    def query(config: str, dtype: str) -> dict[str, str]:
        return next(
            r
            for r in rows
            if r["row_type"] == "query_reproduction"
            and r["config"] == config
            and r["kv_cache_dtype"] == dtype
        )

    q_tp8_f16 = query("tp8ep8-8k2k", "float16")
    q_tp8_fp8 = query("tp8ep8-8k2k", "fp8")
    q_tp4_f16 = query("tp4ep8dp2-8k2k", "float16")
    q_tp4_fp8 = query("tp4ep8dp2-8k2k", "fp8")

    lines = [
        "# Phase397n Attention Residual Triage",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Scope | report-only offline triage for vLLM generation MLA decode |",
        "| Workload | Kimi-K2.5 / h200_sxm / vLLM 0.19.0 / decode 8k2k |",
        "| Primary residual | tp8 attention cb_sim 31.891ms vs measured 16.898ms = 1.887x |",
        "| Verdict | primary hypothesis is generation_mla microbench anchor/workload mismatch, not dtype or layer count |",
        "| Phase397o | recollect generation_mla with CUDA graph replay and paged/varlen-like decode layout |",
        "| Default AIC | No-Go |",
        "",
        "## Query reproduction",
        "",
        "| Config | KV dtype | heads/GPU | batch | seq | DB ms/layer | x61 ms | measured ms | sim/measured |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| tp8ep8-8k2k | float16 | {q_tp8_f16['num_heads']} | {q_tp8_f16['batch_size']} | "
            f"{q_tp8_f16['seq_len']} | {q_tp8_f16['latency_ms_per_layer']} | "
            f"{q_tp8_f16['sim_total_ms']} | {q_tp8_f16['measured_attention_ms']} | "
            f"{q_tp8_f16['sim_over_measured']}x |"
        ),
        (
            f"| tp8ep8-8k2k | fp8 | {q_tp8_fp8['num_heads']} | {q_tp8_fp8['batch_size']} | "
            f"{q_tp8_fp8['seq_len']} | {q_tp8_fp8['latency_ms_per_layer']} | "
            f"{q_tp8_fp8['sim_total_ms']} | {q_tp8_fp8['measured_attention_ms']} | "
            f"{q_tp8_fp8['sim_over_measured']}x |"
        ),
        (
            f"| tp4ep8dp2-8k2k | float16 | {q_tp4_f16['num_heads']} | {q_tp4_f16['batch_size']} | "
            f"{q_tp4_f16['seq_len']} | {q_tp4_f16['latency_ms_per_layer']} | "
            f"{q_tp4_f16['sim_total_ms']} | {q_tp4_f16['measured_attention_ms']} | "
            f"{q_tp4_f16['sim_over_measured']}x |"
        ),
        (
            f"| tp4ep8dp2-8k2k | fp8 | {q_tp4_fp8['num_heads']} | {q_tp4_fp8['batch_size']} | "
            f"{q_tp4_fp8['seq_len']} | {q_tp4_fp8['latency_ms_per_layer']} | "
            f"{q_tp4_fp8['sim_total_ms']} | {q_tp4_fp8['measured_attention_ms']} | "
            f"{q_tp4_fp8['sim_over_measured']}x |"
        ),
        "",
        "## Candidate triage",
        "",
        "| Candidate | Offline verdict | Evidence |",
        "|---|---|---|",
    ]
    for row in [r for r in rows if r["row_type"] == "hypothesis_rank"]:
        lines.append(f"| {row['candidate']} | {row['verdict']} | {row['evidence']} |")
    lines += [
        "",
        "## Phase397o recollection spec",
        "",
        "| Item | Evidence target |",
        "|---|---|",
    ]
    for row in [r for r in rows if r["row_type"] == "phase397o_recollect_spec"]:
        lines.append(f"| {row['phase397o_recollect_item']} | {row['evidence']} |")
    lines += [
        "",
        "## Guardrails",
        "",
        "- Phase397n does not modify `models.py`, `operations.py`, `perf_database.py`, perf tables, or gates.",
        "- No new scale factor is introduced; MoE `k=0.312` is not applied to attention.",
        "- CSV contains the interpolation anchors; `generation_mla_perf.txt` is not changed.",
        "- Phase397o may use GPU for recollection, but this phase does not use SSH/GPU.",
        "- Default AIC remains No-Go.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase397n()
    write_phase397n_csv(args.output_csv, rows)
    write_phase397n_md(args.output_md, rows)


if __name__ == "__main__":
    main()
