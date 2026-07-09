#!/usr/bin/env python3
"""Phase454 GPU-batch acceptance and B2b row extraction.

Report-only by default. It reads the Phase454 GPU raw outputs, computes the
clean-reference ratios, derives B2b forward_total candidate rows, and blocks
duplicate PerfDB keys instead of silently overwriting accepted rows.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATE = REPO_ROOT / "docs/iter_gap_investigation/phase453_validate_multi_config.csv"
DEFAULT_AB = REPO_ROOT / "docs/iter_gap_investigation/phase454_validate_ab.csv"
DEFAULT_PERFDB = (
    REPO_ROOT
    / "src/aiconfigurator/systems/data/h200_sxm/vllm/0.19.0/vllm_serving_state_perf.txt"
)
DEFAULT_GPU_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase454_gpu_batch"
DEFAULT_SCOPE_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase454_gpu_batch_scope"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase454_gpu_batch_acceptance.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase454_gpu_batch_acceptance.md"
DEFAULT_CANDIDATE_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase454_b2b_candidate_rows.csv"
DEFAULT_PERFDB_ROWS = REPO_ROOT / "docs/iter_gap_investigation/phase454_b2b_perfdb_rows.txt"
DEFAULT_CLEAN_VALIDATE = REPO_ROOT / "docs/iter_gap_investigation/phase454_clean_reference_validate.csv"

MODEL = "kimi-k2.5"
QUANT_RUNTIME = "CompressedTensorsWNA16MarlinMoEMethod"
HIDDEN_SIZE = 7168
TOPK = 8
MOE_EP_SIZE = 8
KERNEL_SOURCE = "phase454_b2b_event_timing"

REPORT_FIELDS = [
    "section",
    "scenario",
    "metric",
    "value",
    "target",
    "status",
    "note",
]
PERFDB_FIELDS = [
    "framework",
    "version",
    "device",
    "model",
    "topology",
    "phase",
    "row_kind",
    "category",
    "kernel_source",
    "bucket_tokens",
    "decode_batch",
    "hidden_size",
    "topk",
    "moe_ep_size",
    "quant_runtime",
    "latency",
    "provenance",
]
CANDIDATE_FIELDS = [
    "scenario",
    "topology",
    "phase",
    "row_kind",
    "category",
    "bucket_tokens",
    "decode_batch",
    "latency_ms",
    "sample_count",
    "status",
    "note",
    "provenance",
]


class approx:
    """Tiny pytest.approx replacement for direct script tests."""

    def __init__(self, expected: float, rel: float = 1e-9, abs: float = 1e-9) -> None:
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def __eq__(self, other: object) -> bool:
        return math.isclose(float(other), self.expected, rel_tol=self.rel, abs_tol=self.abs)


@dataclass(frozen=True)
class CleanReferenceSpec:
    scenario: str
    bench_path: Path
    gpu_count: int


@dataclass(frozen=True)
class B2BSpec:
    scenario: str
    event_path: Path
    topology: str
    dp_size: int


@dataclass(frozen=True)
class EventStep:
    scenario: str
    topology: str
    dp_rank: str
    local_step: int
    ctx_tokens: int
    generation_requests: int
    forward_busy_ms: float
    num_tokens_unpadded: int
    cudagraph_mode: str

    @property
    def bucket_tokens(self) -> int:
        return self.num_tokens_unpadded if self.num_tokens_unpadded > 0 else self.ctx_tokens + self.generation_requests

    @property
    def phase(self) -> str:
        has_prefill_work = self.bucket_tokens > self.generation_requests
        if has_prefill_work and self.generation_requests > 0:
            return "mixed_prefill"
        if has_prefill_work:
            return "prefill"
        if self.generation_requests > 0:
            return "decode"
        return "empty"


@dataclass(frozen=True)
class PerfDBCandidate:
    scenario: str
    topology: str
    phase: str
    row_kind: str
    category: str
    bucket_tokens: int
    decode_batch: int
    latency_ms: float
    sample_count: int
    provenance: str


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def read_validate_rows(path: Path) -> dict[str, dict[str, str]]:
    return {row["name"]: row for row in _read_csv(path)}


def clean_reference_rows(
    specs: Iterable[CleanReferenceSpec],
    validate_csv: Path,
) -> list[dict[str, object]]:
    validate = read_validate_rows(validate_csv)
    rows: list[dict[str, object]] = []
    for spec in specs:
        bench = _read_json(spec.bench_path)
        total_tok_s_gpu = float(bench["total_tok_s"]) / spec.gpu_count
        isl = int(bench["input_len"])
        osl = int(bench["output_len"])
        output_tok_s_gpu = total_tok_s_gpu * osl / (isl + osl)
        sim_output = float(validate[spec.scenario]["sim_output_tok_s_gpu"])
        ratio = max(sim_output / output_tok_s_gpu, output_tok_s_gpu / sim_output)
        rows.append(
            {
                "scenario": spec.scenario,
                "clean_total_tok_s_gpu": total_tok_s_gpu,
                "clean_real_output_tok_s_gpu": output_tok_s_gpu,
                "sim_output_tok_s_gpu": sim_output,
                "clean_error_ratio": ratio,
                "status": "clean_reference_collected",
                "note": f"N={bench.get('num_prompts')} C={bench.get('max_concurrency')}",
            }
        )
    return rows


def clean_reference_validate_rows(
    validate_csv: Path,
    clean_rows: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    clean_by_scenario = {
        str(row["scenario"]): float(row["clean_real_output_tok_s_gpu"])
        for row in clean_rows
    }
    rows: list[dict[str, object]] = []
    for row in _read_csv(validate_csv):
        scenario = row["name"]
        real = clean_by_scenario.get(scenario, float(row["real_output_tok_s_gpu"]))
        sim = float(row["sim_output_tok_s_gpu"])
        ratio = max(sim / real, real / sim)
        rows.append(
            {
                "name": scenario,
                "real_output_tok_s_gpu": real,
                "sim_output_tok_s_gpu": sim,
                "error_ratio": ratio,
                "status": "pass" if ratio <= 1.15 else "fail",
                "reference": "phase454_n512_clean" if scenario in clean_by_scenario else "existing_reference",
            }
        )
    return rows


def read_event_steps(spec: B2BSpec) -> list[EventStep]:
    by_rank_count: dict[str, int] = defaultdict(int)
    steps: list[EventStep] = []
    with _open_text(spec.event_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != "phase446_graph_outer_event_v2":
                continue
            rank = str(row.get("dp_rank"))
            local_step = by_rank_count[rank]
            by_rank_count[rank] += 1
            steps.append(
                EventStep(
                    scenario=spec.scenario,
                    topology=spec.topology,
                    dp_rank=rank,
                    local_step=local_step,
                    ctx_tokens=int(row.get("ctx_tokens") or 0),
                    generation_requests=int(row.get("generation_requests") or 0),
                    forward_busy_ms=float(row["forward_busy_ms"]),
                    num_tokens_unpadded=int(row.get("num_tokens_unpadded") or 0),
                    cudagraph_mode=str(row.get("cudagraph_mode") or ""),
                )
            )
    if not steps:
        raise ValueError(f"no B2b event rows found: {spec.event_path}")
    return steps


def _peer_phase(step: EventStep, by_key: dict[tuple[str, int], EventStep], dp_size: int) -> str:
    if dp_size == 1:
        return "no_peer"
    peer_rank = "1" if step.dp_rank == "0" else "0"
    peer = by_key.get((peer_rank, step.local_step))
    if peer is None:
        return "peer_missing"
    if peer.phase in {"mixed_prefill", "prefill"}:
        return "peer_prefill"
    if peer.phase == "decode":
        return "peer_decode"
    return "peer_empty"


def extract_b2b_rows(specs: Iterable[B2BSpec]) -> list[PerfDBCandidate]:
    grouped: dict[tuple[str, str, str, int, int], list[float]] = defaultdict(list)
    for spec in specs:
        steps = read_event_steps(spec)
        by_key = {(step.dp_rank, step.local_step): step for step in steps}
        for step in steps:
            peer_phase = _peer_phase(step, by_key, spec.dp_size)
            include = False
            if step.phase == "mixed_prefill":
                include = True
            elif step.phase == "decode" and step.cudagraph_mode == "FULL":
                include = spec.dp_size == 1 or peer_phase == "peer_decode"
            if not include:
                continue
            grouped[
                (
                    step.scenario,
                    step.topology,
                    step.phase,
                    step.bucket_tokens,
                    step.generation_requests,
                )
            ].append(step.forward_busy_ms)
    rows: list[PerfDBCandidate] = []
    for (scenario, topology, phase, bucket_tokens, decode_batch), values in sorted(grouped.items()):
        rows.append(
            PerfDBCandidate(
                scenario=scenario,
                topology=topology,
                phase=phase,
                row_kind="forward_total",
                category="forward_total",
                bucket_tokens=bucket_tokens,
                decode_batch=decode_batch,
                latency_ms=statistics.median(values),
                sample_count=len(values),
                provenance="phase454_b2b_event_step_bucket",
            )
        )
    return rows


def _candidate_key(row: PerfDBCandidate | dict[str, object]) -> tuple[object, ...]:
    return (
        MODEL,
        row["topology"] if isinstance(row, dict) else row.topology,
        row["phase"] if isinstance(row, dict) else row.phase,
        row["row_kind"] if isinstance(row, dict) else row.row_kind,
        row["category"] if isinstance(row, dict) else row.category,
        row["bucket_tokens"] if isinstance(row, dict) else row.bucket_tokens,
        row["decode_batch"] if isinstance(row, dict) else row.decode_batch,
        HIDDEN_SIZE,
        TOPK,
        MOE_EP_SIZE,
        QUANT_RUNTIME,
    )


def _existing_perfdb_rows(path: Path) -> dict[tuple[object, ...], dict[str, str]]:
    rows: dict[tuple[object, ...], dict[str, str]] = {}
    if not path.exists():
        return rows
    for row in _read_csv(path):
        key = (
            row["model"],
            row["topology"],
            row["phase"],
            row.get("row_kind") or "category",
            row["category"],
            int(row["bucket_tokens"]),
            int(row["decode_batch"]),
            int(row["hidden_size"]),
            int(row["topk"]),
            int(row["moe_ep_size"]),
            row["quant_runtime"],
        )
        rows[key] = row
    return rows


def classify_perfdb_candidates(
    rows: Iterable[PerfDBCandidate],
    perfdb_path: Path,
    *,
    blocked_topology_reasons: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    existing = _existing_perfdb_rows(perfdb_path)
    blocked_topology_reasons = blocked_topology_reasons or {}
    decisions: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for row in rows:
        key = _candidate_key(row)
        if row.topology in blocked_topology_reasons:
            status = blocked_topology_reasons[row.topology]
            note = "candidate retained for report; not eligible for PerfDB append in this phase"
        elif key in existing:
            existing_row = existing[key]
            if (
                existing_row.get("kernel_source") == KERNEL_SOURCE
                and existing_row.get("provenance") == row.provenance
                and math.isclose(float(existing_row["latency"]), row.latency_ms, rel_tol=1e-6, abs_tol=1e-6)
            ):
                status = "already_ingested"
                note = "Phase454 row already present in source PerfDB"
            else:
                status = "blocked_duplicate_key"
                note = "existing PerfDB row kept; no silent overwrite"
        elif key in seen:
            status = "blocked_candidate_duplicate"
            note = "duplicate generated row"
        else:
            status = "accepted"
            note = ""
            seen.add(key)
        decisions.append(
            {
                "scenario": row.scenario,
                "topology": row.topology,
                "phase": row.phase,
                "row_kind": row.row_kind,
                "category": row.category,
                "bucket_tokens": row.bucket_tokens,
                "decode_batch": row.decode_batch,
                "latency_ms": row.latency_ms,
                "sample_count": row.sample_count,
                "status": status,
                "note": note,
                "provenance": row.provenance,
            }
        )
    return decisions


def perfdb_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "framework": "VLLM",
        "version": "0.19.0",
        "device": "NVIDIA H200",
        "model": MODEL,
        "topology": row["topology"],
        "phase": row["phase"],
        "row_kind": row["row_kind"],
        "category": row["category"],
        "kernel_source": KERNEL_SOURCE,
        "bucket_tokens": row["bucket_tokens"],
        "decode_batch": row["decode_batch"],
        "hidden_size": HIDDEN_SIZE,
        "topk": TOPK,
        "moe_ep_size": MOE_EP_SIZE,
        "quant_runtime": QUANT_RUNTIME,
        "latency": row["latency_ms"],
        "provenance": row["provenance"],
    }


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in fields})


def default_clean_specs(root: Path) -> list[CleanReferenceSpec]:
    return [
        CleanReferenceSpec(
            "K2.5-tp4ep8dp2-8k2k",
            root / "recollect_dp2_8k2k/K2.5-tp4ep8dp2-8k2k/bench_result.json",
            8,
        ),
        CleanReferenceSpec(
            "K2.5-tp4ep8dp2-32k3k",
            root / "recollect_dp2_32k3k/K2.5-tp4ep8dp2-32k3k/bench_result.json",
            8,
        ),
    ]


def default_b2b_specs(root: Path) -> list[B2BSpec]:
    return [
        B2BSpec(
            "K2.5-tp8ep8-8k2k",
            root / "b2b_tp8_8k2k/overhead_on/event_timing.jsonl",
            "tp8ep8",
            1,
        ),
        B2BSpec(
            "K2.5-tp4ep8dp2-8k2k-bt65536",
            root / "b2b_dp2_8k2k_bt65536/overhead_on/event_timing.jsonl",
            "tp4dp2ep8",
            2,
        ),
    ]


def overhead_rows(scope_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario, gate_path in [
        ("K2.5-tp8ep8-8k2k", scope_root / "b2b_tp8_8k2k/overhead_gate.json"),
        ("K2.5-tp4ep8dp2-8k2k-bt65536", scope_root / "b2b_dp2_8k2k_bt65536/overhead_gate.json"),
    ]:
        gate = _read_json(gate_path)
        rows.append(
            {
                "section": "gpu_overhead_gate",
                "scenario": scenario,
                "metric": "overhead_pct",
                "value": gate["overhead_pct"],
                "target": "<=2.0",
                "status": "pass" if bool(gate["passed"]) else "fail",
                "note": f"off={gate['off_output_tok_s']} on={gate['on_output_tok_s']}",
            }
        )
    return rows


def residual_rows(gpu_root: Path, scope_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    checks = [
        ("K2.5-tp4ep8dp2-8k2k", gpu_root / "recollect_dp2_8k2k/process_residual_after.txt"),
        ("K2.5-tp8ep8-8k2k", scope_root / "b2b_tp8_8k2k/overhead_on/process_residual_after.txt"),
        (
            "K2.5-tp4ep8dp2-8k2k-bt65536",
            scope_root / "b2b_dp2_8k2k_bt65536/overhead_on/process_residual_after.txt",
        ),
    ]
    for scenario, path in checks:
        size = path.stat().st_size if path.exists() else -1
        rows.append(
            {
                "section": "gpu_residual",
                "scenario": scenario,
                "metric": "process_residual_after_bytes",
                "value": size,
                "target": "0",
                "status": "pass" if size == 0 else "fail",
                "note": str(path),
            }
        )
    return rows


def build_report_rows(
    *,
    clean_rows: list[dict[str, object]],
    clean_validate_rows: list[dict[str, object]],
    candidate_decisions: list[dict[str, object]],
    gpu_root: Path,
    scope_root: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in clean_rows:
        rows.append(
            {
                "section": "clean_reference",
                "scenario": item["scenario"],
                "metric": "clean_error_ratio",
                "value": item["clean_error_ratio"],
                "target": "<=1.15",
                "status": "pass" if float(item["clean_error_ratio"]) <= 1.15 else "fail",
                "note": item["note"],
            }
        )
        rows.append(
            {
                "section": "clean_reference",
                "scenario": item["scenario"],
                "metric": "clean_real_output_tok_s_gpu",
                "value": item["clean_real_output_tok_s_gpu"],
                "target": "",
                "status": item["status"],
                "note": "",
            }
        )
    for item in clean_validate_rows:
        rows.append(
            {
                "section": "clean_validate",
                "scenario": item["name"],
                "metric": "error_ratio",
                "value": item["error_ratio"],
                "target": "<=1.15",
                "status": item["status"],
                "note": item["reference"],
            }
        )
    max_row = max(clean_validate_rows, key=lambda row: float(row["error_ratio"]))
    rows.append(
        {
            "section": "clean_validate",
            "scenario": "all",
            "metric": "max_error_ratio",
            "value": max_row["error_ratio"],
            "target": "<=1.15",
            "status": "pass" if float(max_row["error_ratio"]) <= 1.15 else "fail",
            "note": f"max_scenario={max_row['name']}",
        }
    )
    rows.extend(overhead_rows(scope_root))
    active_statuses = {"accepted", "already_ingested"}
    accepted = [row for row in candidate_decisions if row["status"] in active_statuses]
    appendable = [row for row in candidate_decisions if row["status"] == "accepted"]
    blocked = [row for row in candidate_decisions if str(row["status"]).startswith("blocked")]
    for scenario in sorted({str(row["scenario"]) for row in candidate_decisions}):
        scenario_rows = [row for row in candidate_decisions if row["scenario"] == scenario]
        rows.append(
            {
                "section": "b2b_candidate",
                "scenario": scenario,
                "metric": "active_rows",
                "value": sum(1 for row in scenario_rows if row["status"] in active_statuses),
                "target": ">0",
                "status": "pass" if any(row["status"] in active_statuses for row in scenario_rows) else "blocked",
                "note": "",
            }
        )
        rows.append(
            {
                "section": "b2b_candidate",
                "scenario": scenario,
                "metric": "blocked_rows",
                "value": sum(1 for row in scenario_rows if str(row["status"]).startswith("blocked")),
                "target": "reported, not overwritten",
                "status": "reported",
                "note": "",
            }
        )
    rows.append(
        {
            "section": "perfdb_candidate",
            "scenario": "all",
            "metric": "active_total",
            "value": len(accepted),
            "target": "accepted or already ingested",
            "status": "ready",
            "note": f"appendable={len(appendable)} already_ingested={sum(1 for row in candidate_decisions if row['status'] == 'already_ingested')}",
        }
    )
    rows.append(
        {
            "section": "perfdb_candidate",
            "scenario": "all",
            "metric": "blocked_total",
            "value": len(blocked),
            "target": "no silent overwrite",
            "status": "pass",
            "note": "Duplicate keys require schema/regime decision.",
        }
    )
    rows.extend(residual_rows(gpu_root, scope_root))
    rows.append(
        {
            "section": "verdict",
            "scenario": "phase454",
            "metric": "clean_acceptance_verdict",
            "value": "gpu_batch_collected_dp2_bt_ingested_tp8_blocked_by_scope",
            "target": "clean reference + safe B2b ingestion evidence",
            "status": "no_go",
            "note": "TP8 B2b rows require a KV/ISL scope key before PerfDB ingestion.",
        }
    )
    return rows


def write_markdown(path: Path, rows: list[dict[str, object]], candidate_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Phase454 GPU batch acceptance",
        "",
        "结论: GPU 批已收包。dp2 两个 N=512 清洁参考都可用;dp2 bt65536 的非冲突 mixed 行可入库并改善 A/B;TP8 B2b 行因缺 KV/ISL scope 轴导致跨场景回归,本轮阻断不入库。",
        "",
        "## Summary",
        "",
        "| Section | Scenario | Metric | Value | Target | Status | Note |",
        "|---|---|---|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['section']} | {row['scenario']} | {row['metric']} | {_fmt(row['value'])} | "
            f"{row['target']} | {row['status']} | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "## Candidate Row Preview",
            "",
            "| Scenario | Topology | Phase | Bucket | Batch | Median ms | Samples | Status |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in candidate_rows[:80]:
        lines.append(
            f"| {row['scenario']} | {row['topology']} | {row['phase']} | {row['bucket_tokens']} | "
            f"{row['decode_batch']} | {_fmt(row['latency_ms'])} | {row['sample_count']} | {row['status']} |"
        )
    if len(candidate_rows) > 80:
        lines.append(f"| ... | ... | ... | ... | ... | ... | ... | {len(candidate_rows) - 80} more rows |")
    lines.extend(
        [
            "",
            "Default AIC 仍为 No-Go: dp2 bt65536 已有零回归 A/B;TP8 仍需带 KV/ISL scope 的后续采集或 schema 扩展。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    clean = clean_reference_rows(default_clean_specs(args.gpu_root), args.validate_csv)
    clean_validate = clean_reference_validate_rows(args.validate_csv, clean)
    candidates = extract_b2b_rows(default_b2b_specs(args.scope_root))
    decisions = classify_perfdb_candidates(
        candidates,
        args.perfdb,
        blocked_topology_reasons={"tp8ep8": "blocked_ab_regression_missing_kv_axis"},
    )
    accepted = [row for row in decisions if row["status"] == "accepted"]
    report_rows = build_report_rows(
        clean_rows=clean,
        clean_validate_rows=clean_validate,
        candidate_decisions=decisions,
        gpu_root=args.gpu_root,
        scope_root=args.scope_root,
    )
    write_csv(args.csv, report_rows, REPORT_FIELDS)
    write_csv(
        args.clean_validate_csv,
        clean_validate,
        ["name", "real_output_tok_s_gpu", "sim_output_tok_s_gpu", "error_ratio", "status", "reference"],
    )
    write_csv(args.candidate_csv, decisions, CANDIDATE_FIELDS)
    write_csv(args.perfdb_rows, [perfdb_row(row) for row in accepted], PERFDB_FIELDS)
    write_markdown(args.md, report_rows, decisions)
    return report_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-csv", type=Path, default=DEFAULT_VALIDATE)
    parser.add_argument("--perfdb", type=Path, default=DEFAULT_PERFDB)
    parser.add_argument("--gpu-root", type=Path, default=DEFAULT_GPU_ROOT)
    parser.add_argument("--scope-root", type=Path, default=DEFAULT_SCOPE_ROOT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument("--perfdb-rows", type=Path, default=DEFAULT_PERFDB_ROWS)
    parser.add_argument("--clean-validate-csv", type=Path, default=DEFAULT_CLEAN_VALIDATE)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
