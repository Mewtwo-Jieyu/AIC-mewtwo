#!/usr/bin/env python3
"""Phase462 Step1: report-only dynamics-family triage."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import scripts.analyze_phase459_residual_triage as phase459
import scripts.validate_cb_simulator as validate
from aiconfigurator.sdk.backends.cb_simulator import CBSimulator
from aiconfigurator.sdk.backends.cb_simulator.datatypes import RequestState
from aiconfigurator.sdk.backends.cb_simulator.iteration_latency import IterationLatencyCalculator
from aiconfigurator.sdk.backends.cb_simulator.scheduler import CBScheduler
from aiconfigurator.sdk.perf_database import PerfDatabase


TP8_BT = "K2.5-tp8ep8-8k2k-bt65536"
DP2_BT = "K2.5-tp4ep8dp2-8k2k-bt65536"
DP2_32K = "K2.5-tp4ep8dp2-32k3k"
NUM_REQUESTS = 512
COMMON_ROOT_THRESHOLD = 0.70
TP8_BT_RESIDUAL_FRACTION = 0.235
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase462_dynamics_triage.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase462_dynamics_triage.md"
ANOMALY_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase461_perfdb_anomaly_scan.csv"
CELL_TRIAGE_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase461_step4_cell_triage.csv"
MAX_BT_SCOPE_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase461_max_bt_scope.csv"
SCOPE_REPLAY_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase461_step2_scope_replay.csv"
PERFDB_ROOT = REPO_ROOT / "src/aiconfigurator/systems/data/h200_sxm/vllm/0.19.0"

REAL_METRICS = {
    "K2.5-tp8ep8-8k2k": REPO_ROOT
    / "docs/iter_gap_investigation/phase458_n512_unify/recollect_tp8_8k2k"
    / "K2.5-tp8ep8-8k2k/metrics.jsonl.gz",
    "K2.5-tp8ep8-32k3k": REPO_ROOT
    / "docs/iter_gap_investigation/phase458_n512_unify/recollect_tp8_32k3k"
    / "K2.5-tp8ep8-32k3k/metrics.jsonl.gz",
    "K2.5-tp4ep8dp2-8k2k": REPO_ROOT
    / "docs/iter_gap_investigation/phase454_gpu_batch/recollect_dp2_8k2k"
    / "K2.5-tp4ep8dp2-8k2k/metrics.jsonl",
    DP2_32K: REPO_ROOT
    / "docs/iter_gap_investigation/phase454_gpu_batch/recollect_dp2_32k3k"
    / "K2.5-tp4ep8dp2-32k3k/metrics.jsonl",
    TP8_BT: REPO_ROOT
    / "docs/iter_gap_investigation/phase458_n512_unify/recollect_tp8_8k2k_bt65536"
    / f"{TP8_BT}/metrics.jsonl.gz",
    DP2_BT: REPO_ROOT
    / "docs/iter_gap_investigation/phase458_n512_unify/recollect_dp2_8k2k_bt65536"
    / f"{DP2_BT}/metrics.jsonl.gz",
}

CSV_FIELDS = [
    "section",
    "scenario",
    "metric",
    "value",
    "target",
    "status",
    "source",
    "note",
]


@dataclass(frozen=True)
class AttributedStep:
    bucket_tokens: int
    decode_batch: int
    wall_ms: float
    prefill_tokens: int
    recompute_tokens: int


@dataclass(frozen=True)
class MlaCandidate:
    filename: str
    line: int
    heads: int
    batch: int
    seq: int
    deviation_ratio: float
    measured_ms: float = 0.0
    expected_ms: float = 0.0


def summarize_recompute_correlation(
    steps: Iterable[AttributedStep],
    *,
    sim_only_cells: set[tuple[int, int]],
    common_root_threshold: float,
) -> dict[str, float | str]:
    steps = list(steps)
    total_wall = sum(step.wall_ms for step in steps)
    sim_only = [
        step for step in steps if (step.bucket_tokens, step.decode_batch) in sim_only_cells
    ]
    sim_only_wall = sum(step.wall_ms for step in sim_only)
    associated_wall = sum(step.wall_ms for step in sim_only if step.recompute_tokens > 0)
    sim_only_prefill = sum(step.prefill_tokens for step in sim_only)
    sim_only_recompute = sum(step.recompute_tokens for step in sim_only)
    association = associated_wall / sim_only_wall if sim_only_wall else 0.0
    return {
        "mixed_wall_ms": total_wall,
        "sim_only_wall_ms": sim_only_wall,
        "sim_only_wall_share": sim_only_wall / total_wall if total_wall else 0.0,
        "recompute_associated_sim_only_wall_ms": associated_wall,
        "recompute_associated_sim_only_wall_share": association,
        "recompute_token_share_in_sim_only": (
            sim_only_recompute / sim_only_prefill if sim_only_prefill else 0.0
        ),
        "verdict": (
            "preemption_recompute_common_root_supported"
            if association >= common_root_threshold
            else "giant_bucket_packing_has_independent_root"
        ),
    }


def _nearest_pair(value: int, values: Iterable[int]) -> tuple[int, int]:
    ordered = sorted(set(values))
    if len(ordered) < 2 or value < ordered[0] or value > ordered[-1]:
        raise ValueError(f"value {value} outside interpolation grid {ordered}")
    for index, item in enumerate(ordered):
        if value >= item and index != len(ordered) - 1:
            continue
        return ordered[index - 1], item
    raise ValueError(f"cannot bracket {value} in {ordered}")


def interpolation_stencil(
    data: dict[int, dict[int, dict[int, float]]],
    *,
    x: int,
    y: int,
    z: int,
) -> set[tuple[int, int, int]]:
    stencil: set[tuple[int, int, int]] = set()
    x_pair = _nearest_pair(x, data)
    for x_axis in x_pair:
        y_pair = _nearest_pair(y, data[x_axis])
        for y_axis in y_pair:
            z_pair = _nearest_pair(z, data[x_axis][y_axis])
            stencil.update((x_axis, y_axis, z_axis) for z_axis in z_pair)
    return stencil


def preemption_comparison(
    *,
    scenario: str,
    real_preemptions: float,
    sim_preemptions: float,
    requests: int,
) -> dict[str, float | str]:
    ratio = sim_preemptions / real_preemptions if real_preemptions else math.inf
    if ratio > 1.5:
        status = "sim_over_preempts"
    elif ratio < 2.0 / 3.0:
        status = "sim_under_preempts"
    else:
        status = "same_order"
    return {
        "scenario": scenario,
        "real_preemptions": real_preemptions,
        "sim_preemptions": sim_preemptions,
        "real_preemptions_per_request": real_preemptions / requests,
        "sim_preemptions_per_request": sim_preemptions / requests,
        "sim_over_real": ratio,
        "status": status,
    }


def candidate_excess_bound(
    *,
    measured_ms: float,
    expected_ms: float,
    layers: int,
    influence_count: int,
    reference_wall_ms: float,
    observed_residual_fraction: float,
) -> dict[str, float | bool]:
    excess = max(0.0, measured_ms - expected_ms) * layers * influence_count
    share = excess / reference_wall_ms if reference_wall_ms else math.inf
    return {
        "excess_wall_ms": excess,
        "wall_share": share,
        "can_explain_residual": share >= observed_residual_fraction,
    }


def _point(name: str):
    return next(point for point in validate.MULTI_CONFIG_DATA if point.name == name)


def load_sim_only_cells(path: Path, scenario: str) -> set[tuple[int, int]]:
    with path.open(newline="", encoding="utf-8") as source:
        return {
            (int(row["bucket_tokens"]), int(row["decode_batch"]))
            for row in csv.DictReader(source)
            if row["section"] == "cell"
            and row["scenario"] == scenario
            and row["classification"] == "sim_only_for_phase458_reference"
        }


def run_recompute_attribution(
    name: str,
    *,
    num_requests: int | None = None,
) -> dict[str, object]:
    point = _point(name)
    model, database, backend = validate._load_model_and_db(
        tp=point.tp,
        dp=point.dp,
        moe_tp=point.moe_tp,
        moe_ep=point.moe_ep,
    )
    config = validate._make_multi_config_cb_config(
        point,
        overlap_factor=0.0,
        ep8_per_iteration_overhead_ms=validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
    )
    if num_requests is not None:
        config = replace(
            config,
            num_requests=num_requests,
            warmup_requests=point.batch_size,
        )
    pending: deque[dict[str, int]] = deque()
    steps: list[AttributedStep] = []
    preemptions = 0
    original_schedule = CBScheduler.schedule
    original_preempt = CBScheduler._preempt
    original_compute = IterationLatencyCalculator.compute

    def wrapped_schedule(self, waiting, running):
        result = original_schedule(self, waiting, running)
        if not result.is_empty:
            pending.append(
                {
                    "prefill_tokens": result.total_prefill_tokens,
                    "prefill_batch": len(result.prefill_reqs),
                    "decode_batch": len(result.decode_reqs),
                    "recompute_tokens": sum(
                        result.prefill_tokens[req.request_id]
                        for req in result.prefill_reqs
                        if req.num_preemptions > 0
                        and req.state in {RequestState.PREEMPTED, RequestState.PREFILLING}
                    ),
                }
            )
        return result

    def wrapped_preempt(self, victim, waiting, running, result, preempted_ids):
        nonlocal preemptions
        preemptions += 1
        return original_preempt(self, victim, waiting, running, result, preempted_ids)

    def wrapped_compute(
        self,
        prefill_tokens: int,
        prefill_batch_size: int,
        prefill_seq_len: int,
        decode_batch_size: int,
        decode_avg_kv_len: int,
    ) -> float:
        wall_ms = original_compute(
            self,
            prefill_tokens,
            prefill_batch_size,
            prefill_seq_len,
            decode_batch_size,
            decode_avg_kv_len,
        )
        if pending:
            current = pending.popleft()
            signature = (
                current["prefill_tokens"],
                current["prefill_batch"],
                current["decode_batch"],
            )
            observed = (prefill_tokens, prefill_batch_size, decode_batch_size)
            if signature != observed:
                raise RuntimeError(
                    f"schedule/latency signature mismatch: {signature=} {observed=}"
                )
            if prefill_tokens > 0 and decode_batch_size > 0:
                steps.append(
                    AttributedStep(
                        bucket_tokens=prefill_tokens + decode_batch_size,
                        decode_batch=decode_batch_size,
                        wall_ms=wall_ms,
                        prefill_tokens=prefill_tokens,
                        recompute_tokens=current["recompute_tokens"],
                    )
                )
        return wall_ms

    with (
        patch.object(CBScheduler, "schedule", wrapped_schedule),
        patch.object(CBScheduler, "_preempt", wrapped_preempt),
        patch.object(IterationLatencyCalculator, "compute", wrapped_compute),
    ):
        sim = CBSimulator(backend, model, database, config)
        sim.run(
            isl=point.isl,
            osl=point.osl,
            concurrency=math.ceil(point.batch_size / point.dp),
            num_gpus=point.tp,
        )
    if pending:
        raise RuntimeError(f"{len(pending)} schedule rows were not paired with latency calls")
    return {
        "steps": steps,
        "preemptions": preemptions,
        "num_requests": config.num_requests,
    }


def _load_mla_grid(filename: str) -> dict[int, dict[int, dict[int, float]]]:
    data: dict[int, dict[int, dict[int, float]]] = {}
    with (PERFDB_ROOT / filename).open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["mla_dtype"] != "float16" or row["kv_cache_dtype"] != "float16":
                continue
            heads = int(row["num_heads"])
            batch = int(row["batch_size"])
            if filename == "context_mla_perf.txt":
                y_axis, z_axis = int(row["isl"]), batch
            else:
                y_axis, z_axis = batch, int(row["isl"]) + int(row["step"])
            data.setdefault(heads, {}).setdefault(y_axis, {})[z_axis] = float(row["latency"])
    return data


def load_active_mla_candidates(
    path: Path,
    *,
    perfdb_root: Path = PERFDB_ROOT,
) -> list[MlaCandidate]:
    candidates: dict[tuple[str, int], MlaCandidate] = {}
    with path.open(newline="", encoding="utf-8") as source:
        anomaly_rows = [
            row
            for row in csv.DictReader(source)
            if row["active_six_scope"] == "true"
            and row["file"] in {"context_mla_perf.txt", "generation_mla_perf.txt"}
        ]
    source_rows: dict[str, dict[int, dict[str, str]]] = {}
    for filename in {row["file"] for row in anomaly_rows}:
        with (perfdb_root / filename).open(newline="", encoding="utf-8") as source:
            source_rows[filename] = {
                line: row for line, row in enumerate(csv.DictReader(source), start=2)
            }
    for row in anomaly_rows:
        filename = row["file"]
        line = int(row["line"])
        measured = source_rows[filename][line]
        if (
            measured["mla_dtype"] != "float16"
            or measured["kv_cache_dtype"] != "float16"
        ):
            continue
        seq = int(measured["isl"])
        if filename == "generation_mla_perf.txt":
            seq += int(measured["step"])
        candidate = MlaCandidate(
            filename=filename,
            line=line,
            heads=int(measured["num_heads"]),
            batch=int(measured["batch_size"]),
            seq=seq,
            deviation_ratio=float(row["deviation_ratio"]),
            measured_ms=float(measured["latency"]),
            expected_ms=float(row["expected_latency_ms"]),
        )
        key = (candidate.filename, candidate.line)
        if key not in candidates or candidate.deviation_ratio > candidates[key].deviation_ratio:
            candidates[key] = candidate
    return sorted(candidates.values(), key=lambda item: (item.filename, item.line))


def serving_state_candidate_dispositions(
    anomaly_path: Path = ANOMALY_CSV,
    scope_path: Path = MAX_BT_SCOPE_CSV,
    replay_path: Path = SCOPE_REPLAY_CSV,
) -> list[dict[str, object]]:
    with (PERFDB_ROOT / "vllm_serving_state_perf.txt").open(
        newline="", encoding="utf-8"
    ) as source:
        measured_rows = {
            line: row for line, row in enumerate(csv.DictReader(source), start=2)
        }
    with anomaly_path.open(newline="", encoding="utf-8") as source:
        anomaly_rows: dict[tuple[int, int], dict[str, str]] = {}
        for row in csv.DictReader(source):
            if (
                row["active_six_scope"] != "true"
                or row["file"] != "vllm_serving_state_perf.txt"
            ):
                continue
            measured = measured_rows[int(row["line"])]
            key = (int(measured["bucket_tokens"]), int(measured["decode_batch"]))
            if key not in anomaly_rows or float(row["deviation_ratio"]) > float(
                anomaly_rows[key]["deviation_ratio"]
            ):
                anomaly_rows[key] = row
    with scope_path.open(newline="", encoding="utf-8") as source:
        scope_rows = {
            (int(row["bucket_tokens"]), int(row["decode_batch"])): row
            for row in csv.DictReader(source)
            if row["section"] == "spike_reachability"
        }
    with replay_path.open(newline="", encoding="utf-8") as source:
        replay_rows = [
            row
            for row in csv.DictReader(source)
            if row["section"] == "spike_crosscheck" and row["metric"] == "busy_over_wall"
        ]
    result: list[dict[str, object]] = []
    for key, anomaly in sorted(anomaly_rows.items()):
        scope = scope_rows[key]
        if scope["status"] == "not_reached_current_six":
            disposition = "not_reached_current_six"
            status = "pass"
        else:
            crosscheck = next(row for row in replay_rows if f"cell={key[0]}/{key[1]}" in row["note"])
            disposition = "reachable_real_execution_state"
            status = "pass" if crosscheck["status"] == "pass" else "fail"
        result.append(
            {
                "file": anomaly["file"],
                "line": int(anomaly["line"]),
                "bucket_tokens": key[0],
                "decode_batch": key[1],
                "deviation_ratio": float(anomaly["deviation_ratio"]),
                "disposition": disposition,
                "status": status,
            }
        )
    return result


def run_profile_with_mla_queries(name: str) -> tuple[dict[str, object], dict[str, list[tuple[int, int, int]]]]:
    queries: dict[str, list[tuple[int, int, int]]] = {"context_mla_perf.txt": [], "generation_mla_perf.txt": []}
    original_context = PerfDatabase.query_context_mla
    original_generation = PerfDatabase.query_generation_mla

    def wrapped_context(self, b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode, database_mode=None):
        queries["context_mla_perf.txt"].append((int(num_heads), int(s + prefix), int(b)))
        return original_context(
            self, b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode, database_mode
        )

    def wrapped_generation(self, b, s, num_heads, kvcache_quant_mode, database_mode=None):
        queries["generation_mla_perf.txt"].append((int(num_heads), int(b), int(s)))
        return original_generation(self, b, s, num_heads, kvcache_quant_mode, database_mode)

    with (
        patch.object(PerfDatabase, "query_context_mla", wrapped_context),
        patch.object(PerfDatabase, "query_generation_mla", wrapped_generation),
    ):
        profile = phase459.run_sim_profile(name)
    return profile, queries


def candidate_reachability(
    candidates: Iterable[MlaCandidate],
    queries_by_scenario: dict[str, dict[str, list[tuple[int, int, int]]]],
) -> list[dict[str, object]]:
    grids = {
        filename: _load_mla_grid(filename)
        for filename in ("context_mla_perf.txt", "generation_mla_perf.txt")
    }
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        coordinate = (
            (candidate.heads, candidate.seq, candidate.batch)
            if candidate.filename == "context_mla_perf.txt"
            else (candidate.heads, candidate.batch, candidate.seq)
        )
        for scenario, by_file in queries_by_scenario.items():
            influence = 0
            for query in by_file[candidate.filename]:
                try:
                    stencil = interpolation_stencil(
                        grids[candidate.filename], x=query[0], y=query[1], z=query[2]
                    )
                except ValueError:
                    continue
                influence += int(coordinate in stencil)
            rows.append(
                {
                    "candidate": candidate,
                    "scenario": scenario,
                    "query_count": len(by_file[candidate.filename]),
                    "influence_count": influence,
                    "reachable": influence > 0,
                }
            )
    return rows


def _add(
    rows: list[dict[str, object]],
    section: str,
    scenario: str,
    metric: str,
    value: object,
    *,
    target: str = "",
    status: str = "",
    source: str = "",
    note: str = "",
) -> None:
    rows.append(
        {
            "section": section,
            "scenario": scenario,
            "metric": metric,
            "value": value,
            "target": target,
            "status": status,
            "source": source,
            "note": note,
        }
    )


def build_report_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    attribution = run_recompute_attribution(TP8_BT)
    sim_only_cells = load_sim_only_cells(CELL_TRIAGE_CSV, TP8_BT)
    common_root = summarize_recompute_correlation(
        attribution["steps"],
        sim_only_cells=sim_only_cells,
        common_root_threshold=COMMON_ROOT_THRESHOLD,
    )
    _add(
        rows,
        "common_root",
        TP8_BT,
        "num_requests",
        attribution["num_requests"],
        status="protocol_match",
        source="phase461_step4_cell_triage.py",
        note="matches the simulation protocol that produced the sim-only cell list",
    )
    _add(
        rows,
        "common_root",
        TP8_BT,
        "sim_preemptions",
        attribution["preemptions"],
        status="informational",
        source="phase461 Step4a-2 protocol replay",
        note="same run used for recompute-token attribution",
    )
    for metric, value in common_root.items():
        _add(
            rows,
            "common_root",
            TP8_BT,
            metric,
            value,
            target=(f">={COMMON_ROOT_THRESHOLD}" if metric == "recompute_associated_sim_only_wall_share" else ""),
            status=(
                "pass"
                if metric == "verdict" and value == "preemption_recompute_common_root_supported"
                else "fail"
                if metric == "verdict"
                else "informational"
            ),
            source="phase461 Step4a-2 protocol replay + phase461_step4_cell_triage.csv",
            note="request num_preemptions marks recompute prefill tokens; no runtime mutation",
        )

    query_sets: dict[str, dict[str, list[tuple[int, int, int]]]] = {}
    for scenario in REAL_METRICS:
        if scenario in {TP8_BT, DP2_32K, DP2_BT}:
            profile, queries = run_profile_with_mla_queries(scenario)
            query_sets[scenario] = queries
        else:
            profile = phase459.run_sim_profile(scenario)
        real = phase459.parse_metrics_profile(REAL_METRICS[scenario])
        comparison = preemption_comparison(
            scenario=scenario,
            real_preemptions=real.preemptions,
            sim_preemptions=float(profile["preemptions"]),
            requests=NUM_REQUESTS,
        )
        for metric in (
            "real_preemptions",
            "sim_preemptions",
            "real_preemptions_per_request",
            "sim_preemptions_per_request",
            "sim_over_real",
        ):
            _add(
                rows,
                "preemption_panorama",
                scenario,
                metric,
                comparison[metric],
                status=str(comparison["status"]) if metric == "sim_over_real" else "informational",
                source=str(REAL_METRICS[scenario].relative_to(REPO_ROOT)),
                note="real=first/last metrics counter delta; sim=current N=512 diagnostic run",
            )

    candidates = load_active_mla_candidates(ANOMALY_CSV)
    reachability = candidate_reachability(candidates, query_sets)
    for item in reachability:
        candidate = item["candidate"]
        assert isinstance(candidate, MlaCandidate)
        bound: dict[str, float | bool] | None = None
        if item["reachable"] and item["scenario"] == TP8_BT:
            bound = candidate_excess_bound(
                measured_ms=candidate.measured_ms,
                expected_ms=candidate.expected_ms,
                layers=61,
                influence_count=int(item["influence_count"]),
                reference_wall_ms=float(common_root["mixed_wall_ms"]),
                observed_residual_fraction=TP8_BT_RESIDUAL_FRACTION,
            )
        status = "pass"
        if candidate.filename == "generation_mla_perf.txt" and candidate.line == 2521:
            status = "resolved_phase461_recollect"
        elif item["reachable"]:
            status = (
                "bounded_non_primary"
                if bound is not None and not bool(bound["can_explain_residual"])
                else "fail"
            )
        bound_note = ""
        if bound is not None:
            bound_note = (
                f";excess_upper_ms={float(bound['excess_wall_ms']):.6f}"
                f";mixed_wall_share={float(bound['wall_share']):.9f}"
                f";observed_residual={TP8_BT_RESIDUAL_FRACTION:.3f}"
            )
        _add(
            rows,
            "mla_candidate_reachability",
            str(item["scenario"]),
            f"{candidate.filename}:{candidate.line}",
            int(item["influence_count"]),
            target="0 reachable anomaly queries",
            status=status,
            source="phase461_perfdb_anomaly_scan.csv + runtime MLA query stencil",
            note=(
                f"heads={candidate.heads};batch={candidate.batch};seq={candidate.seq};"
                f"deviation={candidate.deviation_ratio:.6f};queries={item['query_count']}"
                f"{bound_note}"
            ),
        )

    serving_candidates = serving_state_candidate_dispositions()
    for candidate in serving_candidates:
        _add(
            rows,
            "serving_candidate_disposition",
            "all",
            f"{candidate['file']}:{candidate['line']}",
            candidate["disposition"],
            target="not a corrupt cost row",
            status=str(candidate["status"]),
            source="phase461_max_bt_scope.csv + phase461_step2_scope_replay.csv",
            note=(
                f"bucket={candidate['bucket_tokens']};decode={candidate['decode_batch']};"
                f"deviation={candidate['deviation_ratio']:.6f}"
            ),
        )
    unique_mla = {(candidate.filename, candidate.line) for candidate in candidates}
    active_candidate_count = len(unique_mla) + len(serving_candidates)
    unresolved = sum(
        int(row["status"] == "fail")
        for row in rows
        if row["section"] in {"mla_candidate_reachability", "serving_candidate_disposition"}
    )
    _add(
        rows,
        "decision",
        "all",
        "active_anomaly_candidates_audited",
        active_candidate_count,
        target="16",
        status="pass" if active_candidate_count == 16 else "fail",
        note="unique PerfDB file+line candidates from the Phase461 active-six snapshot",
    )
    _add(
        rows,
        "decision",
        "all",
        "unresolved_anomaly_candidate_pairs",
        unresolved,
        target="0",
        status="pass" if unresolved == 0 else "fail",
        note="reachable Phase461-recollected MLA row is resolved; serving spikes use prior mechanism evidence",
    )
    _add(
        rows,
        "decision",
        "all",
        "step2_route",
        (
            "preemption_first_divergence"
            if common_root["verdict"] == "preemption_recompute_common_root_supported"
            else "chunk_packing_semantics_first"
        ),
        status="report_only",
        note="runtime/PerfDB/gate unchanged; user confirmation required before Step2",
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _value(rows: list[dict[str, object]], section: str, scenario: str, metric: str):
    return next(
        row["value"]
        for row in rows
        if row["section"] == section and row["scenario"] == scenario and row["metric"] == metric
    )


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    common_verdict = _value(rows, "common_root", TP8_BT, "verdict")
    association = float(
        _value(rows, "common_root", TP8_BT, "recompute_associated_sim_only_wall_share")
    )
    sim_only_share = float(_value(rows, "common_root", TP8_BT, "sim_only_wall_share"))
    preemption_rows = [
        row for row in rows if row["section"] == "preemption_panorama" and row["metric"] in {
            "real_preemptions", "sim_preemptions", "sim_over_real"
        }
    ]
    by_scenario: dict[str, dict[str, object]] = {}
    for row in preemption_rows:
        by_scenario.setdefault(str(row["scenario"]), {})[str(row["metric"])] = row["value"]
    reachable_rows = [
        row
        for row in rows
        if row["section"] == "mla_candidate_reachability" and row["status"] == "fail"
    ]
    bounded_rows = [
        row
        for row in rows
        if row["section"] == "mla_candidate_reachability"
        and row["status"] == "bounded_non_primary"
    ]
    lines = [
        "# Phase462 Step 1: dynamics-family triage",
        "",
        "## Verdict",
        "",
        (
            f"Common-root verdict: `{common_verdict}`. Sim-only giant-bucket mixed steps account for "
            f"{sim_only_share:.1%} of simulated mixed wall; {association:.1%} of that wall occurs in "
            "steps carrying recompute prefill tokens."
        ),
        "",
        f"Unresolved MLA anomaly/scenario pairs: {len(reachable_rows)}. "
        f"Active anomaly candidates audited: {_value(rows, 'decision', 'all', 'active_anomaly_candidates_audited')}/16.",
        "",
        "This phase is report-only: runtime, PerfDB, validation gate, and Default AIC were not changed.",
        "",
        "## Preemption panorama",
        "",
        "| Scenario | Real | Sim | Sim/real |",
        "|---|---:|---:|---:|",
    ]
    for scenario, values in by_scenario.items():
        ratio = float(values["sim_over_real"])
        ratio_text = "inf" if math.isinf(ratio) else f"{ratio:.2f}x"
        lines.append(
            f"| {scenario} | {float(values['real_preemptions']):.0f} | "
            f"{float(values['sim_preemptions']):.0f} | {ratio_text} |"
        )
    lines.extend(
        [
            "",
            "## PerfDB anomaly audit",
            "",
            "A candidate is reachable only when its exact row coordinate is part of the runtime "
            "three-dimensional interpolation stencil. Static shape-range overlap is not counted.",
            "",
        ]
    )
    if reachable_rows:
        lines.extend(["| Scenario | Candidate | Influence calls |", "|---|---|---:|"])
        for row in reachable_rows:
            lines.append(f"| {row['scenario']} | {row['metric']} | {row['value']} |")
    else:
        lines.append("No MLA anomaly candidate remains unresolved in the three failing scenarios.")
    if bounded_rows:
        lines.extend(
            [
                "",
                "Reachable but bounded below the observed residual:",
                "",
                "| Scenario | Candidate | Influence calls | Bound |",
                "|---|---|---:|---|",
            ]
        )
        for row in bounded_rows:
            lines.append(
                f"| {row['scenario']} | {row['metric']} | {row['value']} | {row['note']} |"
            )
    lines.extend(
        [
            "",
            "## Next gate",
            "",
            f"Step 2 route: `{_value(rows, 'decision', 'all', 'step2_route')}`. "
            "No fix is authorized by this report alone; first-divergence evidence and vLLM 0.19 "
            "source agreement remain mandatory.",
            "",
            "`diagnostic_only=true`, `valid_for_default=false`, `perf_database=false`, Default AIC No-Go.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    rows = build_report_rows()
    write_csv(args.csv, rows)
    write_markdown(args.md, rows)


if __name__ == "__main__":
    main()
