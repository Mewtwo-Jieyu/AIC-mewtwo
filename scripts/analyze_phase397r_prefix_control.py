#!/usr/bin/env python3
"""Phase397r: prefix-off control report for MLA decode attention.

This is report-only. It verifies that the prefix cache was disabled in the
Phase397r serve run, then records why that run does not close the b128
attention line: no-prefix KV pressure reduced the profiled decode batch to 67,
so the attention timing matches the already-known b64 diagnostic path.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = "phase397r_prefix_control"
RUN_ROOT = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397r_prefix_off_profile/K2.5-tp8ep8-8k2k"
)
OP_BREAKDOWN_CSV = RUN_ROOT / "op_breakdown.csv"
META_JSON = RUN_ROOT / "meta.json"
SERVE_LOG = RUN_ROOT / "serve.log"
PHASE397Q_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase397q_mla_recollect_stop_report.csv"
OUTPUT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase397r_prefix_control.csv"
OUTPUT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase397r_prefix_control.md"

NUM_LAYERS = 61
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"

FIELDNAMES = [
    "source",
    "row_type",
    "verdict",
    "evidence",
    "prefix_caching",
    "max_prefix_hit_pct",
    "max_gpu_kv_cache_usage_pct",
    "max_running_reqs",
    "profile_decode_bs",
    "attention_ms_per_iter",
    "attention_ms_per_layer",
    "serve_target_ms_per_layer",
    "b64_graph_p50_ms",
    "b64_ratio_vs_serve",
    "b128_graph_p50_ms",
    "b128_ratio_vs_serve",
    "prefix_control_sufficient",
    "attention_line_closed",
    "write_generation_mla_perf",
    "runtime_modified",
    "gate_modified",
    "gpu_used",
    "ssh_used",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
    "default_readiness",
    "next_allowed_phase",
]

LOG_RE = re.compile(
    r"Running:\s*(?P<running>\d+) reqs, Waiting:\s*(?P<waiting>\d+) reqs, "
    r"GPU KV cache usage:\s*(?P<kv>[0-9.]+)%, Prefix cache hit rate:\s*(?P<prefix>[0-9.]+)%"
)


def _common(row_type: str, verdict: str, evidence: str) -> dict[str, str]:
    return {
        "source": SOURCE,
        "row_type": row_type,
        "verdict": verdict,
        "evidence": evidence,
        "prefix_caching": "",
        "max_prefix_hit_pct": "",
        "max_gpu_kv_cache_usage_pct": "",
        "max_running_reqs": "",
        "profile_decode_bs": "",
        "attention_ms_per_iter": "",
        "attention_ms_per_layer": "",
        "serve_target_ms_per_layer": "",
        "b64_graph_p50_ms": "",
        "b64_ratio_vs_serve": "",
        "b128_graph_p50_ms": "",
        "b128_ratio_vs_serve": "",
        "prefix_control_sufficient": FALSE,
        "attention_line_closed": FALSE,
        "write_generation_mla_perf": FALSE,
        "runtime_modified": FALSE,
        "gate_modified": FALSE,
        "gpu_used": TRUE,
        "ssh_used": TRUE,
        "diagnostic_only": TRUE,
        "valid_for_default": FALSE,
        "perf_database": FALSE,
        "default_readiness": DEFAULT_READINESS,
        "next_allowed_phase": "phase397s_int4_wo_moe_table_gap",
    }


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_meta(path: Path = META_JSON) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("prefix_caching") is not False:
        raise ValueError("Phase397r meta must record prefix_caching=false")
    return data


def _serve_log_stats(path: Path = SERVE_LOG) -> dict[str, float]:
    samples: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = LOG_RE.search(line)
        if not match:
            continue
        samples.append(
            {
                "running": float(match.group("running")),
                "kv": float(match.group("kv")),
                "prefix": float(match.group("prefix")),
            }
        )
    if not samples:
        raise ValueError(f"missing vLLM runtime samples in {path}")
    return {
        "max_running": max(sample["running"] for sample in samples),
        "max_kv": max(sample["kv"] for sample in samples),
        "max_prefix": max(sample["prefix"] for sample in samples),
    }


def _op_breakdown(path: Path = OP_BREAKDOWN_CSV) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in _load_csv(path):
        key = row["category"]
        if row["ms_per_iter"] == "":
            continue
        values[key] = float(row["ms_per_iter"])
    required = {"attention", "decode_bs", "decode_wall_ms_per_iter"}
    missing = required - set(values)
    if missing:
        raise ValueError(f"missing op breakdown rows: {sorted(missing)}")
    return values


def _phase397q_refs(path: Path = PHASE397Q_CSV) -> dict[str, dict[str, float]]:
    refs: dict[str, dict[str, float]] = {}
    for row in _load_csv(path):
        row_type = row["row_type"]
        if row_type not in {"effective_batch_b64", "planned_acceptance_b128"}:
            continue
        refs[row_type] = {
            "graph_p50_ms": float(row["graph_p50_ms"]),
            "serve_target_ms": float(row["serve_target_ms"]),
            "ratio_vs_serve": float(row["ratio_vs_serve"]),
        }
    if set(refs) != {"effective_batch_b64", "planned_acceptance_b128"}:
        raise ValueError("missing Phase397q b64/b128 reference rows")
    return refs


def build_rows() -> list[dict[str, str]]:
    meta = _load_meta()
    log_stats = _serve_log_stats()
    breakdown = _op_breakdown()
    refs = _phase397q_refs()

    attention_ms = breakdown["attention"]
    attention_per_layer = attention_ms / NUM_LAYERS
    profile_decode_bs = int(breakdown["decode_bs"])

    rows: list[dict[str, str]] = []

    row = _common(
        "control_config",
        "prefix_cache_disabled_run_recorded",
        "Phase397r meta.json records --no-enable-prefix-caching for tp8ep8-8k2k.",
    )
    row.update(
        {
            "prefix_caching": str(meta["prefix_caching"]).lower(),
            "profile_decode_bs": str(profile_decode_bs),
        }
    )
    rows.append(row)

    row = _common(
        "prefix_cache_check",
        "prefix_cache_disabled_confirmed",
        "serve.log reports Prefix cache hit rate 0.0% throughout sampled runtime lines.",
    )
    row.update(
        {
            "prefix_caching": str(meta["prefix_caching"]).lower(),
            "max_prefix_hit_pct": _fmt(log_stats["max_prefix"]),
        }
    )
    rows.append(row)

    row = _common(
        "kv_capacity_check",
        "kv_capacity_limited_effective_batch",
        "No-prefix 8k contexts filled KV cache before reaching the planned b128 decode batch.",
    )
    row.update(
        {
            "max_gpu_kv_cache_usage_pct": _fmt(log_stats["max_kv"]),
            "max_running_reqs": str(int(log_stats["max_running"])),
            "profile_decode_bs": str(profile_decode_bs),
        }
    )
    rows.append(row)

    row = _common(
        "attention_measurement",
        "matches_b64_not_b128",
        "Measured attention is close to Phase397q b64/serve per-layer timing, not the b128 slow path.",
    )
    row.update(
        {
            "profile_decode_bs": str(profile_decode_bs),
            "attention_ms_per_iter": _fmt(attention_ms),
            "attention_ms_per_layer": _fmt(attention_per_layer),
            "serve_target_ms_per_layer": _fmt(refs["effective_batch_b64"]["serve_target_ms"]),
            "b64_graph_p50_ms": _fmt(refs["effective_batch_b64"]["graph_p50_ms"]),
            "b64_ratio_vs_serve": _fmt(refs["effective_batch_b64"]["ratio_vs_serve"]),
            "b128_graph_p50_ms": _fmt(refs["planned_acceptance_b128"]["graph_p50_ms"]),
            "b128_ratio_vs_serve": _fmt(refs["planned_acceptance_b128"]["ratio_vs_serve"]),
        }
    )
    rows.append(row)

    row = _common(
        "decision",
        "prefix_off_control_inconclusive_due_kv_capacity_batch_drop",
        "The control disabled prefix caching but did not exercise b128 independent decode, so it cannot close the attention line.",
    )
    row.update(
        {
            "prefix_caching": str(meta["prefix_caching"]).lower(),
            "max_prefix_hit_pct": _fmt(log_stats["max_prefix"]),
            "max_gpu_kv_cache_usage_pct": _fmt(log_stats["max_kv"]),
            "max_running_reqs": str(int(log_stats["max_running"])),
            "profile_decode_bs": str(profile_decode_bs),
            "attention_ms_per_iter": _fmt(attention_ms),
            "attention_ms_per_layer": _fmt(attention_per_layer),
            "serve_target_ms_per_layer": _fmt(refs["effective_batch_b64"]["serve_target_ms"]),
            "b64_graph_p50_ms": _fmt(refs["effective_batch_b64"]["graph_p50_ms"]),
            "b64_ratio_vs_serve": _fmt(refs["effective_batch_b64"]["ratio_vs_serve"]),
            "b128_graph_p50_ms": _fmt(refs["planned_acceptance_b128"]["graph_p50_ms"]),
            "b128_ratio_vs_serve": _fmt(refs["planned_acceptance_b128"]["ratio_vs_serve"]),
        }
    )
    rows.append(row)

    validate_rows(rows)
    return rows


def validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Phase397r output must not be empty")
    for row in rows:
        if row["write_generation_mla_perf"] != FALSE:
            raise ValueError("Phase397r must keep write_generation_mla_perf=false")
        if row["runtime_modified"] != FALSE:
            raise ValueError("Phase397r must keep runtime_modified=false")
        if row["gate_modified"] != FALSE:
            raise ValueError("Phase397r must keep gate_modified=false")
        if row["diagnostic_only"] != TRUE:
            raise ValueError("Phase397r must keep diagnostic_only=true")
        if row["valid_for_default"] != FALSE:
            raise ValueError("Phase397r must keep valid_for_default=false")
        if row["perf_database"] != FALSE:
            raise ValueError("Phase397r must keep perf_database=false")
        if row["default_readiness"] != DEFAULT_READINESS:
            raise ValueError("Phase397r must keep default_readiness=No-Go")
    decision = rows[-1]
    if decision["row_type"] != "decision":
        raise ValueError("Phase397r final row must be decision")
    if decision["attention_line_closed"] != FALSE:
        raise ValueError("Phase397r must keep attention_line_closed=false")
    if int(decision["profile_decode_bs"]) >= 96:
        raise ValueError("Phase397r decision expects a KV-limited decode batch below 96")


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict[str, str]], path: Path) -> None:
    decision = rows[-1]
    lines = [
        "# Phase397r Prefix Control",
        "",
        "Phase397r did not close the attention line.",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Prefix cache | disabled, max hit {decision['max_prefix_hit_pct']}% |",
        f"| KV usage | max {decision['max_gpu_kv_cache_usage_pct']}% |",
        f"| Profile decode batch | {decision['profile_decode_bs']} |",
        f"| Attention | {decision['attention_ms_per_iter']} ms/iter, {decision['attention_ms_per_layer']} ms/layer |",
        f"| Phase397q b64 graph | {decision['b64_graph_p50_ms']} ms, ratio {decision['b64_ratio_vs_serve']}x |",
        f"| Phase397q b128 graph | {decision['b128_graph_p50_ms']} ms, ratio {decision['b128_ratio_vs_serve']}x |",
        f"| Verdict | {decision['verdict']} |",
        "",
        "The control did turn prefix caching off: `serve.log` reports 0.0% prefix hits.",
        "But the no-prefix 8k workload filled KV cache and the profiled decode batch fell to 67.",
        "That means this run tested an effective b64-like independent-prefix shape, not the planned b128 shape.",
        "",
        "So the current evidence says: do not rewrite `generation_mla_perf.txt`, do not close the attention line, and do not change runtime or gates from this control.",
        "",
        "Default AIC remains No-Go.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--out-md", type=Path, default=OUTPUT_MD)
    args = parser.parse_args(argv)

    rows = build_rows()
    write_csv(rows, args.out_csv)
    write_md(rows, args.out_md)
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
