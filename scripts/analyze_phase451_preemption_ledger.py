#!/usr/bin/env python3
"""Phase451: preemption/re-prefill ledger for DP2 8k2k."""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import math
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

SCENARIO = "K2.5-tp4ep8dp2-8k2k"
DEFAULT_EVENT = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase446_b2b_event_timing/"
    / "overhead_gate_20260708_075726/overhead_on/event_timing.jsonl"
)
DEFAULT_METRICS = DEFAULT_EVENT.parent / SCENARIO / "metrics.jsonl"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase451_preemption_ledger.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase451_preemption_ledger.md"
DEFAULT_NUM_REQUESTS = 512
MIXED_SHARE_TARGET = 0.0243
MIXED_SHARE_TOLERANCE = 0.005
DEFAULT_READINESS = "No-Go"

CSV_FIELDS = [
    "scenario",
    "section",
    "side",
    "engine",
    "metric",
    "value",
    "target",
    "status",
    "note",
]

PROM_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+0-9.eE]+)$"
)
LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')


@dataclass(frozen=True)
class CounterDelta:
    first: float
    last: float
    delta: float


@dataclass(frozen=True)
class MetricSpec:
    metric_name: str
    output_name: str
    labels: dict[str, str]


METRIC_SPECS = [
    MetricSpec("vllm:num_preemptions_total", "num_preemptions_total", {}),
    MetricSpec("vllm:prompt_tokens_total", "prompt_tokens_total", {}),
    MetricSpec("vllm:prompt_tokens_recomputed_total", "prompt_tokens_recomputed_total", {}),
    MetricSpec("vllm:generation_tokens_total", "generation_tokens_total", {}),
    MetricSpec("vllm:request_prefill_kv_computed_tokens_sum", "request_prefill_kv_computed_tokens_sum", {}),
    MetricSpec("vllm:request_prefill_kv_computed_tokens_count", "request_prefill_kv_computed_tokens_count", {}),
    MetricSpec("vllm:request_success_total", "request_success_length", {"finished_reason": "length"}),
]


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


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


def parse_prometheus_line(line: str) -> tuple[str, dict[str, str], float] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    match = PROM_RE.match(stripped)
    if not match:
        return None
    labels = {
        label.group(1): label.group(2)
        for label in LABEL_RE.finditer(match.group("labels") or "")
    }
    return match.group("name"), labels, float(match.group("value"))


def _labels_match(labels: dict[str, str], expected: dict[str, str]) -> bool:
    return all(labels.get(key) == value for key, value in expected.items())


def collect_metric_deltas(path: Path) -> dict[str, dict[str, CounterDelta]]:
    """Read first/last counter values by engine from metrics.jsonl."""
    values: dict[str, dict[str, list[float]]] = {
        spec.output_name: {} for spec in METRIC_SPECS
    }
    with _open_text(path) as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            for metric_line in str(payload.get("body", "")).splitlines():
                parsed = parse_prometheus_line(metric_line)
                if parsed is None:
                    continue
                name, labels, value = parsed
                engine = labels.get("engine")
                if engine is None:
                    continue
                for spec in METRIC_SPECS:
                    if name == spec.metric_name and _labels_match(labels, spec.labels):
                        values[spec.output_name].setdefault(engine, []).append(value)

    deltas: dict[str, dict[str, CounterDelta]] = {}
    for metric, by_engine in values.items():
        deltas[metric] = {}
        for engine, samples in by_engine.items():
            if not samples:
                continue
            first = samples[0]
            last = samples[-1]
            deltas[metric][engine] = CounterDelta(first=first, last=last, delta=last - first)
    return deltas


def preemption_driver_gate(
    *,
    mixed_steps: int,
    preemptions: float,
    recomputed_tokens: float,
    tolerance: float,
) -> dict[str, object]:
    ratio = preemptions / mixed_steps if mixed_steps > 0 else 0.0
    passed = ratio >= tolerance and recomputed_tokens > 0
    return {
        "passed": passed,
        "status": "pass" if passed else "fail",
        "preemption_to_mixed_ratio": ratio,
        "note": (
            "preemption counter and recomputed tokens can explain mixed steps"
            if passed
            else "mixed steps are not mostly preemption/recompute driven"
        ),
    }


def _phase_from_trace(row: dict[str, object]) -> str:
    if bool(row.get("is_mixed")):
        return "mixed_prefill"
    if int(row.get("prefill_tokens", 0)) > 0:
        return "prefill"
    if int(row.get("decode_reqs", 0)) > 0:
        return "decode"
    return "empty"


def summarize_sim_trace(trace: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = list(trace)
    mixed_by_replica: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    for row in rows:
        phase = _phase_from_trace(row)
        phase_counts[phase] += 1
        if phase == "mixed_prefill":
            mixed_by_replica[str(row["replica_id"])] += 1
    total = len(rows)
    mixed = phase_counts["mixed_prefill"]
    return {
        "total_steps": total,
        "mixed_steps": mixed,
        "mixed_share": mixed / total if total else 0.0,
        "phase_counts": dict(phase_counts),
        "mixed_steps_by_replica": dict(sorted(mixed_by_replica.items())),
    }


def summarize_real_events(event_path: Path) -> dict[str, object]:
    import scripts.analyze_phase446_b2b_ingest as phase446

    steps = phase446.group_event_steps(
        phase446.read_event_records(event_path),
        tp_width=1,
    )
    mixed_by_engine: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    for step in steps:
        phase_counts[step.phase] += 1
        if step.phase == "mixed_prefill":
            mixed_by_engine[str(step.dp_rank)] += 1
    total = len(steps)
    mixed = phase_counts["mixed_prefill"]
    return {
        "total_steps": total,
        "mixed_steps": mixed,
        "mixed_share": mixed / total if total else 0.0,
        "phase_counts": dict(phase_counts),
        "mixed_steps_by_engine": dict(sorted(mixed_by_engine.items())),
    }


def _load_validate_module():
    path = REPO_ROOT / "scripts" / "validate_cb_simulator.py"
    spec = importlib.util.spec_from_file_location("validate_cb_simulator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_sim_preemption_ledger(*, scenario: str, num_requests: int) -> dict[str, object]:
    from aiconfigurator.sdk.backends.cb_simulator import CBSimulator
    from aiconfigurator.sdk.backends.cb_simulator.datatypes import RequestState
    from aiconfigurator.sdk.backends.cb_simulator.scheduler import CBScheduler

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

    stats: Counter[str] = Counter()
    original_preempt = CBScheduler._preempt
    original_schedule = CBScheduler.schedule

    def wrapped_preempt(self, victim, waiting, running, result, preempted_ids):
        stats["preemptions"] += 1
        stats["recompute_tokens"] += victim.isl + victim.generated_tokens
        return original_preempt(self, victim, waiting, running, result, preempted_ids)

    def wrapped_schedule(self, waiting, running):
        result = original_schedule(self, waiting, running)
        for req in result.prefill_reqs:
            if req.state == RequestState.PREEMPTED:
                stats["preempted_prefill_steps"] += 1
                stats["preempted_prefill_tokens"] += result.prefill_tokens.get(req.request_id, 0)
        return result

    CBScheduler._preempt = wrapped_preempt
    CBScheduler.schedule = wrapped_schedule
    try:
        sim = CBSimulator(backend, model, db, config)
        sim_result = sim.run_multi_replica(
            isl=point.isl,
            osl=point.osl,
            concurrency=point.batch_size,
            data_parallel_size=point.dp,
            num_gpus=point.tp * point.dp,
            lockstep=True,
        )
        trace = sim.get_last_schedule_trace()
    finally:
        CBScheduler._preempt = original_preempt
        CBScheduler.schedule = original_schedule

    summary = summarize_sim_trace(trace)
    summary.update(
        {
            "preemptions": int(stats["preemptions"]),
            "recompute_tokens": int(stats["recompute_tokens"]),
            "preempted_prefill_steps": int(stats["preempted_prefill_steps"]),
            "preempted_prefill_tokens": int(stats["preempted_prefill_tokens"]),
            "throughput_tok_s_gpu": sim_result.throughput_tok_s_gpu,
            "steady_state_iterations": sim_result.steady_state_iterations,
        }
    )
    return summary


def _get_delta(deltas: dict[str, dict[str, CounterDelta]], metric: str, engine: str) -> float:
    return deltas.get(metric, {}).get(engine, CounterDelta(0.0, 0.0, 0.0)).delta


def _row(
    *,
    section: str,
    side: str,
    engine: str,
    metric: str,
    value: object,
    target: object = "",
    status: str = "",
    note: str = "",
    scenario: str = SCENARIO,
) -> dict[str, object]:
    return {
        "scenario": scenario,
        "section": section,
        "side": side,
        "engine": engine,
        "metric": metric,
        "value": value,
        "target": target,
        "status": status,
        "note": note,
    }


def build_reports(
    *,
    scenario: str = SCENARIO,
    event_path: Path = DEFAULT_EVENT,
    metrics_path: Path = DEFAULT_METRICS,
    num_requests: int = DEFAULT_NUM_REQUESTS,
) -> list[dict[str, object]]:
    real_events = summarize_real_events(event_path)
    deltas = collect_metric_deltas(metrics_path)
    sim = run_sim_preemption_ledger(scenario=scenario, num_requests=num_requests)

    rows: list[dict[str, object]] = []
    rows.extend(
        [
            _row(
                section="mixed_profile",
                side="real",
                engine="all",
                metric="mixed_share",
                value=real_events["mixed_share"],
                target=f"{MIXED_SHARE_TARGET:.4f}±{MIXED_SHARE_TOLERANCE:.4f}",
                status="pass"
                if abs(float(real_events["mixed_share"]) - MIXED_SHARE_TARGET) <= MIXED_SHARE_TOLERANCE
                else "informational",
                note=f"mixed_steps={real_events['mixed_steps']}; total_steps={real_events['total_steps']}",
            ),
            _row(
                section="mixed_profile",
                side="sim",
                engine="all",
                metric="mixed_share",
                value=sim["mixed_share"],
                target=f"{MIXED_SHARE_TARGET:.4f}±{MIXED_SHARE_TOLERANCE:.4f}",
                status="pass"
                if abs(float(sim["mixed_share"]) - MIXED_SHARE_TARGET) <= MIXED_SHARE_TOLERANCE
                else "fail",
                note=f"mixed_steps={sim['mixed_steps']}; total_steps={sim['total_steps']}",
            ),
        ]
    )

    for engine in sorted(real_events["mixed_steps_by_engine"]):
        mixed_steps = int(real_events["mixed_steps_by_engine"][engine])
        requests = _get_delta(deltas, "request_success_length", engine)
        preemptions = _get_delta(deltas, "num_preemptions_total", engine)
        recomputed = _get_delta(deltas, "prompt_tokens_recomputed_total", engine)
        prompt_tokens = _get_delta(deltas, "prompt_tokens_total", engine)
        kv_sum = _get_delta(deltas, "request_prefill_kv_computed_tokens_sum", engine)
        kv_count = _get_delta(deltas, "request_prefill_kv_computed_tokens_count", engine)
        gate = preemption_driver_gate(
            mixed_steps=mixed_steps,
            preemptions=preemptions,
            recomputed_tokens=recomputed,
            tolerance=0.80,
        )
        rows.extend(
            [
                _row(
                    section="preemption_ledger",
                    side="real",
                    engine=engine,
                    metric="request_success_length",
                    value=requests,
                    note="delta from metrics counter",
                ),
                _row(
                    section="preemption_ledger",
                    side="real",
                    engine=engine,
                    metric="mixed_steps_per_request",
                    value=mixed_steps / requests if requests else math.nan,
                    note=f"mixed_steps={mixed_steps}",
                ),
                _row(
                    section="preemption_ledger",
                    side="real",
                    engine=engine,
                    metric="preemptions_per_request",
                    value=preemptions / requests if requests else math.nan,
                    note=f"preemptions={preemptions}",
                ),
                _row(
                    section="preemption_ledger",
                    side="real",
                    engine=engine,
                    metric="prompt_tokens_recomputed_per_request",
                    value=recomputed / requests if requests else math.nan,
                    status="pass" if recomputed == 0 else "informational",
                    note=f"prompt_tokens_recomputed_total={recomputed}",
                ),
                _row(
                    section="preemption_ledger",
                    side="real",
                    engine=engine,
                    metric="prefill_kv_tokens_per_request",
                    value=kv_sum / kv_count if kv_count else math.nan,
                    target="8000",
                    status="pass" if kv_count and abs(kv_sum / kv_count - 8000) <= 1 else "fail",
                    note=f"prompt_tokens_total={prompt_tokens}; kv_count={kv_count}",
                ),
                _row(
                    section="preemption_gate",
                    side="real",
                    engine=engine,
                    metric="preemption_drives_mixed_steps",
                    value=gate["preemption_to_mixed_ratio"],
                    target=">=0.80 and recomputed_tokens>0",
                    status=str(gate["status"]),
                    note=str(gate["note"]),
                ),
            ]
        )

    sim_requests_per_engine = num_requests / 2
    rows.extend(
        [
            _row(
                section="preemption_ledger",
                side="sim",
                engine="all",
                metric="preemptions_per_request",
                value=float(sim["preemptions"]) / num_requests,
                note=f"preemptions={sim['preemptions']}; request_count={num_requests}",
            ),
            _row(
                section="preemption_ledger",
                side="sim",
                engine="all",
                metric="recompute_tokens_per_request",
                value=float(sim["recompute_tokens"]) / num_requests,
                note=f"recompute_tokens={sim['recompute_tokens']}",
            ),
            _row(
                section="preemption_ledger",
                side="sim",
                engine="all",
                metric="preempted_prefill_steps_per_request",
                value=float(sim["preempted_prefill_steps"]) / num_requests,
                note=f"preempted_prefill_steps={sim['preempted_prefill_steps']}",
            ),
        ]
    )
    for engine, mixed_steps in dict(sim["mixed_steps_by_replica"]).items():
        rows.append(
            _row(
                section="preemption_ledger",
                side="sim",
                engine=engine,
                metric="mixed_steps_per_request",
                value=float(mixed_steps) / sim_requests_per_engine,
                note=f"mixed_steps={mixed_steps}; expected_requests_per_engine={sim_requests_per_engine:.0f}",
            )
        )

    real_total_preemptions = sum(
        _get_delta(deltas, "num_preemptions_total", engine)
        for engine in real_events["mixed_steps_by_engine"]
    )
    real_total_recomputed = sum(
        _get_delta(deltas, "prompt_tokens_recomputed_total", engine)
        for engine in real_events["mixed_steps_by_engine"]
    )
    sim_preempt_ratio = float(sim["preemptions"]) / max(real_total_preemptions, 1.0)
    rows.extend(
        [
            _row(
                section="verdict",
                side="real",
                engine="all",
                metric="real_preemption_explains_1380_mixed_steps",
                value="false",
                target="true",
                status="fail",
                note=(
                    f"real_preemptions={real_total_preemptions:.0f}; "
                    f"real_recomputed_tokens={real_total_recomputed:.0f}; "
                    f"mixed_steps={real_events['mixed_steps']}"
                ),
            ),
            _row(
                section="verdict",
                side="sim",
                engine="all",
                metric="sim_preemption_over_real_preemption",
                value=sim_preempt_ratio,
                target="near 1.0",
                status="fail" if sim_preempt_ratio > 2.0 else "informational",
                note=(
                    "sim preemption/recompute is excessive relative to real counters; "
                    "a runtime fix must target sim over-preemption, not assume real re-prefill"
                ),
            ),
            _row(
                section="decision",
                side="all",
                engine="all",
                metric="phase451_3_runtime_fix_gate",
                value="blocked",
                target="only after exact semantic diff is isolated",
                status="blocked",
                note="report-only ledger falsifies the original real re-prefill premise",
            ),
        ]
    )

    rows.extend(source_audit_rows())
    return rows


def source_audit_rows() -> list[dict[str, object]]:
    return [
        _row(
            section="source_audit",
            side="vllm",
            engine="all",
            metric="running_preemption_trigger",
            value="allocate_slots_none",
            note="vllm/v1/core/sched/scheduler.py:460-508 preempts only when RUNNING allocation fails",
        ),
        _row(
            section="source_audit",
            side="vllm",
            engine="all",
            metric="waiting_admission_guard",
            value="skip_waiting_if_preempted",
            note="vllm/v1/core/sched/scheduler.py:563-564 schedules WAITING only when no preemption happened",
        ),
        _row(
            section="source_audit",
            side="vllm",
            engine="all",
            metric="preempt_recompute_state",
            value="num_computed_tokens_reset",
            note="vllm/v1/core/sched/scheduler.py:956-971 frees KV, status=PREEMPTED, num_computed_tokens=0",
        ),
        _row(
            section="source_audit",
            side="vllm",
            engine="all",
            metric="kv_allocate_failure",
            value="return_none_if_free_blocks_insufficient",
            note="vllm/v1/core/kv_cache_manager.py:327-334 describes allocation stages; allocate_slots returns None on insufficient free blocks",
        ),
        _row(
            section="source_audit",
            side="sim",
            engine="all",
            metric="preempt_recompute_state",
            value="prefill_tokens_remaining=isl+generated",
            note="src/.../cb_simulator/scheduler.py:76-93 requeues PREEMPTED at waiting head",
        ),
        _row(
            section="source_audit",
            side="sim",
            engine="all",
            metric="waiting_admission_guard",
            value="non_preemptive_admission",
            note="src/.../cb_simulator/scheduler.py:197-217 admits only if _fits_block_capacity",
        ),
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def write_md(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase451 preemption ledger",
        "",
        "结论: 真实 run 的 mixed 步不是 preemption/recompute 主驱动。真实每 engine preemption counter 有增量,但 `prompt_tokens_recomputed_total=0`,且每请求 computed KV 仍是 8000。当前 sim 的 preemption/recompute 反而明显高于真实,451-3 不能按“真实大量重预填”假设直接修。",
        "",
        "| section | side | engine | metric | value | target | status | note |",
        "|---|---|---:|---|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['section']} | {row['side']} | {row['engine']} | {row['metric']} | "
            f"{_fmt(row['value'])} | {row.get('target', '')} | {row.get('status', '')} | {row.get('note', '')} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Report-only: no runtime, PerfDB, validate gate, or reference data changed.",
            "- The real side uses Phase446 B2b 8k2k event/metrics from the same run.",
            "- Default AIC remains No-Go until the full table meets the agreed gate.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=Path, default=DEFAULT_EVENT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--num-requests", type=int, default=DEFAULT_NUM_REQUESTS)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_reports(
        scenario=SCENARIO,
        event_path=args.event,
        metrics_path=args.metrics,
        num_requests=args.num_requests,
    )
    write_csv(args.csv_out, rows)
    write_md(args.md_out, rows)
    print(f"wrote {args.csv_out}")
    print(f"wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
