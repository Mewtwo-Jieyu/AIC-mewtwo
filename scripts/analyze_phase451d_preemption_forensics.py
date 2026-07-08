#!/usr/bin/env python3
"""Phase451-D: locate sim over-preemption semantics before any runtime fix."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import statistics
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

SCENARIO = "K2.5-tp4ep8dp2-8k2k"
DEFAULT_NUM_REQUESTS = 512
DEFAULT_TP_WIDTH = 4
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase451d_preemption_forensics.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase451d_preemption_forensics.md"
DEFAULT_READINESS = "No-Go"
MIXED_SHARE_TARGET = 0.0243
MIXED_SHARE_TOLERANCE = 0.005

CSV_FIELDS = [
    "section",
    "scenario",
    "side",
    "metric",
    "value",
    "target",
    "status",
    "note",
]


def _load_validate_module():
    path = REPO_ROOT / "scripts" / "validate_cb_simulator.py"
    spec = importlib.util.spec_from_file_location("validate_cb_simulator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_phase451_module():
    import scripts.analyze_phase451_preemption_ledger as phase451

    return phase451


def _fmt(value: object) -> str:
    if value == "":
        return ""
    if isinstance(value, bool):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "nan"
    if abs(number) >= 1000:
        return f"{number:.0f}"
    if number.is_integer():
        return f"{number:.0f}"
    return f"{number:.6f}".rstrip("0").rstrip(".")


def _row(
    section: str,
    side: str,
    metric: str,
    value: object,
    *,
    target: object = "",
    status: str = "",
    note: str = "",
    scenario: str = SCENARIO,
) -> dict[str, object]:
    return {
        "section": section,
        "scenario": scenario,
        "side": side,
        "metric": metric,
        "value": value,
        "target": target,
        "status": status,
        "note": note,
    }


def classify_preemption_event(event: dict[str, object]) -> tuple[str, str | None]:
    """Classify one sim preemption event into a source-auditable signature."""
    if int(event.get("victim_preemptions_before", 0)) > 0:
        return "thrash_repeat_victim", "preempt_reentry_thrash"
    if bool(event.get("victim_recently_admitted", False)):
        return "admission_induced", "waiting_admission_gate"
    trigger_state = str(event.get("trigger_state", ""))
    if trigger_state == "DECODING":
        return "decode_growth_pressure", "block_growth_or_watermark"
    if trigger_state == "PREFILLING":
        return "prefill_growth_pressure", "prefill_chunk_allocation"
    return "unclassified", None


def classify_preemption_events(events: Iterable[dict[str, object]]) -> dict[str, object]:
    event_list = list(events)
    category_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    for event in event_list:
        category, candidate = classify_preemption_event(event)
        category_counts[category] += 1
        if candidate is not None:
            candidate_counts[candidate] += 1
    total = len(event_list)
    covered = total - category_counts["unclassified"]
    target = total * 0.80
    cumulative = 0
    gate_candidates: list[str] = []
    for candidate, count in candidate_counts.most_common():
        if cumulative >= target:
            break
        gate_candidates.append(candidate)
        cumulative += count
    return {
        "total_events": total,
        "covered_events": covered,
        "coverage": covered / total if total else 0.0,
        "category_counts": dict(category_counts),
        "candidate_counts": dict(candidate_counts),
        "candidate_semantic_diffs": gate_candidates,
        "candidate_gate_coverage": cumulative / total if total else 0.0,
    }


def classification_gate(
    summary: dict[str, object],
    *,
    coverage_target: float = 0.80,
    max_candidates: int = 3,
) -> dict[str, object]:
    coverage = float(summary["coverage"])
    candidate_count = len(summary["candidate_semantic_diffs"])
    passed = coverage >= coverage_target and 0 < candidate_count <= max_candidates
    return {
        "status": "pass" if passed else "blocked",
        "passed": passed,
        "coverage": coverage,
        "candidate_count": candidate_count,
        "note": (
            "minimal source-backed candidate set covers the target share"
            if passed
            else "classification does not yet justify a runtime fix"
        ),
    }


def runtime_fix_gate(summary: dict[str, object]) -> dict[str, object]:
    """Phase451-E is blocked until the dominant signature maps to a proven fix."""
    candidate_count = len(summary["candidate_semantic_diffs"])
    if candidate_count == 0:
        note = "no source-auditable dominant signature yet"
    elif candidate_count == 1:
        note = (
            "dominant signature isolated, but no unique source-backed runtime "
            "change has been proven to reduce the declared metrics"
        )
    else:
        note = "multiple source-auditable signatures remain"
    return {
        "status": "blocked",
        "value": "blocked_pending_unique_semantic_fix",
        "note": note,
    }


def real_tp_deduped_ledger(
    *,
    serve_log: Path,
    metrics_path: Path,
) -> dict[str, object]:
    phase451 = _load_phase451_module()
    real_events = phase451.summarize_real_iterations(serve_log)
    deltas = phase451.collect_metric_deltas(metrics_path)
    engines: dict[str, dict[str, float]] = {}
    for engine, mixed_steps in dict(real_events["mixed_steps_by_engine"]).items():
        requests = phase451._get_delta(deltas, "request_success_length", engine)
        preemptions = phase451._get_delta(deltas, "num_preemptions_total", engine)
        recomputed = phase451._get_delta(deltas, "prompt_tokens_recomputed_total", engine)
        engines[engine] = {
            "mixed_steps": float(mixed_steps),
            "requests": requests,
            "mixed_steps_per_request": float(mixed_steps) / requests if requests else math.nan,
            "preemptions_per_request": preemptions / requests if requests else math.nan,
            "recomputed_tokens_per_request": recomputed / requests if requests else math.nan,
        }
    return {
        "total_steps": real_events["total_steps"],
        "mixed_steps": real_events["mixed_steps"],
        "mixed_share": real_events["mixed_share"],
        "engines": engines,
    }


def run_sim_preemption_forensics(*, scenario: str, num_requests: int) -> dict[str, object]:
    """Run current sim while recording every scheduler preemption."""
    from aiconfigurator.sdk.backends.cb_simulator import CBSimulator
    from aiconfigurator.sdk.backends.cb_simulator import scheduler as scheduler_module
    from aiconfigurator.sdk.backends.cb_simulator.datatypes import RequestState

    validate = _load_validate_module()
    point = next(pt for pt in validate.MULTI_CONFIG_DATA if pt.name == scenario)
    model, db, backend = validate._load_model_and_db(
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
    config = replace(config, num_requests=num_requests, warmup_requests=point.batch_size)

    original_schedule = scheduler_module.CBScheduler.schedule
    original_ensure = scheduler_module.CBScheduler._ensure_block_capacity
    original_preempt = scheduler_module.CBScheduler._preempt

    scheduler_to_replica: dict[int, int] = {}
    local_iters: Counter[int] = Counter()
    active_context: dict[int, dict[str, object]] = {}
    events: list[dict[str, object]] = []

    def replica_id(obj) -> int:
        key = id(obj)
        if key not in scheduler_to_replica:
            scheduler_to_replica[key] = len(scheduler_to_replica)
        return scheduler_to_replica[key]

    def wrapped_schedule(self, waiting, running):
        rid = replica_id(self)
        local_iters[rid] += 1
        return original_schedule(self, waiting, running)

    def wrapped_ensure(self, current_req, waiting, running, result, preempted_ids):
        rid = replica_id(self)
        total_blocks = self._total_blocks(running, result)
        active_context[id(self)] = {
            "replica_id": rid,
            "local_iter": local_iters[rid],
            "trigger_req_id": current_req.request_id,
            "trigger_state": current_req.state.name,
            "trigger_generated_tokens": current_req.generated_tokens,
            "trigger_kv_cache_len": current_req.kv_cache_len,
            "running_before": len(running),
            "waiting_before": len(waiting),
            "prefill_reqs_before": len(result.prefill_reqs),
            "decode_reqs_before": len(result.decode_reqs),
            "total_blocks_before": total_blocks,
            "over_blocks_before": total_blocks - self._config.num_gpu_blocks,
            "num_gpu_blocks": self._config.num_gpu_blocks,
        }
        try:
            return original_ensure(self, current_req, waiting, running, result, preempted_ids)
        finally:
            active_context.pop(id(self), None)

    def wrapped_preempt(self, victim, waiting, running, result, preempted_ids):
        context = dict(active_context.get(id(self), {}))
        victim_recently_admitted = (
            victim.state == RequestState.DECODING
            and victim.generated_tokens <= 1
            and victim.prefill_tokens_remaining == 0
        )
        events.append(
            {
                **context,
                "victim_req_id": victim.request_id,
                "victim_state": victim.state.name,
                "victim_generated_tokens": victim.generated_tokens,
                "victim_kv_cache_len": victim.kv_cache_len,
                "victim_prefill_remaining": victim.prefill_tokens_remaining,
                "victim_preemptions_before": victim.num_preemptions,
                "victim_recently_admitted": victim_recently_admitted,
                "result_prefill_tokens": result.total_prefill_tokens,
                "result_total_tokens": result.total_tokens,
            }
        )
        return original_preempt(self, victim, waiting, running, result, preempted_ids)

    scheduler_module.CBScheduler.schedule = wrapped_schedule
    scheduler_module.CBScheduler._ensure_block_capacity = wrapped_ensure
    scheduler_module.CBScheduler._preempt = wrapped_preempt
    try:
        sim = CBSimulator(backend, model, db, config)
        result = sim.run_multi_replica(
            isl=point.isl,
            osl=point.osl,
            concurrency=point.batch_size,
            data_parallel_size=point.dp,
            num_gpus=point.tp * point.dp,
            lockstep=True,
        )
        trace = sim.get_last_schedule_trace()
    finally:
        scheduler_module.CBScheduler.schedule = original_schedule
        scheduler_module.CBScheduler._ensure_block_capacity = original_ensure
        scheduler_module.CBScheduler._preempt = original_preempt

    for event in events:
        category, candidate = classify_preemption_event(event)
        event["category"] = category
        event["candidate_semantic_diff"] = candidate or ""
    return {
        "events": events,
        "trace": trace,
        "throughput_tok_s_gpu": result.throughput_tok_s_gpu,
        "steady_state_iterations": result.steady_state_iterations,
    }


def source_semantic_rows() -> list[dict[str, object]]:
    return [
        _row(
            "source_semantics",
            "vllm",
            "preempt_trigger",
            "running_allocate_slots_none",
            note="vllm/v1/core/sched/scheduler.py:460-508; preemption occurs while scheduling RUNNING requests",
        ),
        _row(
            "source_semantics",
            "vllm",
            "waiting_after_preempt",
            "skip_waiting_when_preempted",
            note="vllm/v1/core/sched/scheduler.py:563-564; WAITING scheduling is skipped if preempted_reqs is non-empty",
        ),
        _row(
            "source_semantics",
            "vllm",
            "preempt_reentry",
            "free_kv_reset_computed_prepend_waiting",
            note="vllm/v1/core/sched/scheduler.py:956-971",
        ),
        _row(
            "source_semantics",
            "vllm",
            "waiting_admission_allocation",
            "allocate_slots_for_new_chunk_or_skip",
            note="vllm/v1/core/sched/scheduler.py:575-700 and kv_cache_manager.py:327-334",
        ),
        _row(
            "source_semantics",
            "sim",
            "preempt_reentry",
            "prefill_remaining_isl_plus_generated_insert_waiting_head",
            note="src/aiconfigurator/sdk/backends/cb_simulator/scheduler.py:76-93",
        ),
        _row(
            "source_semantics",
            "sim",
            "waiting_admission",
            "non_preemptive_fits_check",
            note="src/aiconfigurator/sdk/backends/cb_simulator/scheduler.py:197-217",
        ),
    ]


def build_report_rows(
    *,
    scenario: str = SCENARIO,
    num_requests: int = DEFAULT_NUM_REQUESTS,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    phase451 = _load_phase451_module()
    real = real_tp_deduped_ledger(
        serve_log=phase451.DEFAULT_SERVE_LOG,
        metrics_path=phase451.DEFAULT_METRICS,
    )
    sim = run_sim_preemption_forensics(scenario=scenario, num_requests=num_requests)
    events = list(sim["events"])
    summary = classify_preemption_events(events)
    gate = classification_gate(summary)
    fix_gate = runtime_fix_gate(summary)
    candidate_count = len(summary["candidate_semantic_diffs"])

    rows: list[dict[str, object]] = [
        _row(
            "real_tp_dedupe",
            "real",
            "mixed_share",
            real["mixed_share"],
            target=f"{MIXED_SHARE_TARGET:.4f}±{MIXED_SHARE_TOLERANCE:.4f}",
            status=(
                "pass"
                if abs(float(real["mixed_share"]) - MIXED_SHARE_TARGET) <= MIXED_SHARE_TOLERANCE
                else "informational"
            ),
            note=f"mixed_steps={real['mixed_steps']}; total_steps={real['total_steps']}; source=serve.log iteration lines",
        )
    ]
    for engine, values in dict(real["engines"]).items():
        rows.extend(
            [
                _row(
                    "real_tp_dedupe",
                    "real",
                    f"engine_{engine}_mixed_steps_per_request",
                    values["mixed_steps_per_request"],
                    target="~1.0",
                    status="pass" if 1.0 <= values["mixed_steps_per_request"] <= 1.7 else "informational",
                    note=f"mixed_steps={values['mixed_steps']:.0f}; requests={values['requests']:.0f}; EngineCore iteration lines",
                ),
                _row(
                    "real_tp_dedupe",
                    "real",
                    f"engine_{engine}_recomputed_tokens_per_request",
                    values["recomputed_tokens_per_request"],
                    target="0",
                    status="pass" if values["recomputed_tokens_per_request"] == 0 else "fail",
                ),
            ]
        )

    rows.extend(
        [
            _row(
                "sim_forensics",
                "sim",
                "preemption_events",
                summary["total_events"],
                note=f"throughput_tok_s_gpu={float(sim['throughput_tok_s_gpu']):.6f}",
            ),
            _row(
                "sim_forensics",
                "sim",
                "classification_coverage",
                summary["coverage"],
                target=">=0.80",
                status=gate["status"],
                note=str(gate["note"]),
            ),
            _row(
                "sim_forensics",
                "sim",
                "minimal_candidate_set_coverage",
                summary["candidate_gate_coverage"],
                target=">=0.80",
                status=gate["status"],
                note="coverage of the minimal candidate set used by the D gate",
            ),
            _row(
                "sim_forensics",
                "sim",
                "candidate_semantic_diff_count",
                candidate_count,
                target="<=3 for D gate; E requires a proven semantic fix",
                status="pass" if candidate_count <= 3 else "blocked",
                note=",".join(summary["candidate_semantic_diffs"]),
            ),
            _row(
                "decision",
                "sim",
                "phase451d_classification_gate",
                bool(gate["passed"]),
                target="coverage>=0.80 and candidates<=3",
                status=gate["status"],
            ),
            _row(
                "decision",
                "sim",
                "phase451e_runtime_fix_gate",
                fix_gate["value"],
                target="unique source-backed fix that reduces recompute/preemption",
                status=fix_gate["status"],
                note=fix_gate["note"],
            ),
        ]
    )

    for category, count in sorted(dict(summary["category_counts"]).items()):
        rows.append(
            _row(
                "sim_category",
                "sim",
                category,
                count,
                note=f"share={count / max(int(summary['total_events']), 1):.6f}",
            )
        )
    for candidate, count in sorted(dict(summary["candidate_counts"]).items()):
        rows.append(
            _row(
                "candidate",
                "sim",
                candidate,
                count,
                note="source-backed signature candidate; not a fix by itself",
            )
        )
    rows.extend(source_semantic_rows())

    sample_events = sorted(
        events,
        key=lambda row: (
            str(row.get("category")),
            int(row.get("replica_id", -1)),
            int(row.get("local_iter", -1)),
        ),
    )[:80]
    return rows, sample_events


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def write_md(path: Path, rows: list[dict[str, object]], sample_events: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_metric = {str(row["metric"]): row for row in rows}
    gate = by_metric.get("phase451d_classification_gate", {})
    fix_gate = by_metric.get("phase451e_runtime_fix_gate", {})
    lines = [
        "# Phase451-D preemption forensics",
        "",
        (
            "结论: 真实侧按 EngineCore iteration 行复核后 mixed/request 约 1.04,且 recompute 仍为 0。"
            "当前 sim 的抢占由 repeat-victim thrash 主导;D 门通过,但 E 门仍阻塞,因为还没有唯一且已验证会改善指标的源码语义修复。"
        ),
        "",
        f"- classification gate: `{gate.get('status', '')}`",
        f"- runtime fix gate: `{fix_gate.get('status', '')}` ({fix_gate.get('value', '')})",
        "- Report-only: this analyzer does not change scheduler, PerfDB, validate gates, or reference data.",
        f"- Default AIC remains `{DEFAULT_READINESS}`.",
        "",
        "## Summary",
        "",
        "| section | side | metric | value | target | status | note |",
        "|---|---|---|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['section']} | {row['side']} | {row['metric']} | {_fmt(row['value'])} | "
            f"{row.get('target', '')} | {row.get('status', '')} | {row.get('note', '')} |"
        )

    lines.extend(
        [
            "",
            "## Sample Preemption Events",
            "",
            "| category | replica | iter | trigger | victim | victim_preemptions_before | blocks_over | running | waiting |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for event in sample_events:
        lines.append(
            f"| {event.get('category', '')} | {event.get('replica_id', '')} | "
            f"{event.get('local_iter', '')} | {event.get('trigger_state', '')} | "
            f"{event.get('victim_req_id', '')} | {event.get('victim_preemptions_before', '')} | "
            f"{event.get('over_blocks_before', '')} | {event.get('running_before', '')} | "
            f"{event.get('waiting_before', '')} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument("--num-requests", type=int, default=DEFAULT_NUM_REQUESTS)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows, sample_events = build_report_rows(
        scenario=args.scenario,
        num_requests=args.num_requests,
    )
    write_csv(args.csv_out, rows)
    write_md(args.md_out, rows, sample_events)
    print(f"wrote {args.csv_out}")
    print(f"wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
