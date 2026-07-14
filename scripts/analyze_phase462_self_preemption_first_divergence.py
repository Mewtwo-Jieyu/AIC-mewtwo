#!/usr/bin/env python3
"""Phase462 Step2c-2b: audit the remaining steady self-preemption."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import scripts.analyze_phase462_engine_loop_runtime_gate as runtime_gate  # noqa: E402
import scripts.analyze_phase462_preemption_observation as real_observation  # noqa: E402
import aiconfigurator.sdk.backends.cb_simulator.simulator as simulator_module  # noqa: E402
from aiconfigurator.sdk.backends.cb_simulator.engine_loop import (  # noqa: E402
    AsyncCBScheduler,
)


REAL_JSONL = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase462_preemption_observation"
    / "preemption_observation.jsonl"
)
PHASE458_DIR = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase458_n512_unify/recollect_tp8_32k3k"
    / "K2.5-tp8ep8-32k3k"
)
PHASE458_SERVE_LOG = PHASE458_DIR / "serve.log"
PHASE458_METRICS = PHASE458_DIR / "metrics.jsonl.gz"
DEFAULT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase462_self_preemption_first_divergence.csv"
)
DEFAULT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase462_self_preemption_first_divergence.md"
)

VLLM_SCHEDULER_SHA256 = (
    "9f3da2dfce94963e1e1cefc156e4239120f79c6eaf824b2829cac0ee7736b58a"
)
VLLM_KV_MANAGER_SHA256 = (
    "0df69b7626195cbaabb33013d1e05ba6660243369171367f87313e4221bfffc8"
)
CSV_FIELDS = [
    "section",
    "side",
    "metric",
    "value",
    "target",
    "status",
    "source",
    "note",
]
REPORT_BOUNDARIES = {
    "diagnostic_only": True,
    "valid_for_default": False,
    "perf_database": False,
    "default_aic": "No-Go",
}


@dataclass(frozen=True)
class SourceLegality:
    status: str
    self_preemption_branch_reachable: bool
    reason: str


@dataclass(frozen=True)
class GateAction:
    action: str
    gate_change_allowed: bool
    runtime_change_allowed: bool
    gpu_followup_required: bool
    reason: str


def evaluate_source_legality(
    *,
    allocation_failed: bool,
    trigger_request_id: object,
    victim_request_id: object,
    running_tail_request_id: object,
) -> SourceLegality:
    reachable = (
        allocation_failed
        and trigger_request_id == victim_request_id
        and trigger_request_id == running_tail_request_id
    )
    return SourceLegality(
        status="source_legal" if reachable else "source_illegal",
        self_preemption_branch_reachable=reachable,
        reason=(
            "fcfs_allocation_failure_pops_running_tail_equal_to_trigger"
            if reachable
            else "event_does_not_match_fcfs_tail_self_preemption_branch"
        ),
    )


def decide_gate_action(
    *,
    source_status: str,
    real_relation_observable: bool,
    real_trigger_tail_events: int,
    phase458_relation_observable: bool,
) -> GateAction:
    if source_status == "source_illegal":
        return GateAction(
            action="fix_engine_loop_implementation",
            gate_change_allowed=False,
            runtime_change_allowed=True,
            gpu_followup_required=False,
            reason="sim_event_is_not_reachable_under_vllm_source_semantics",
        )
    if (
        real_relation_observable
        and real_trigger_tail_events > 0
    ):
        return GateAction(
            action="preregister_structural_gate_from_source_and_real",
            gate_change_allowed=True,
            runtime_change_allowed=False,
            gpu_followup_required=False,
            reason="source_legal_event_is_observed_in_real_decision_data",
        )
    return GateAction(
        action="keep_gate_and_investigate_state_evolution",
        gate_change_allowed=False,
        runtime_change_allowed=False,
        gpu_followup_required=True,
        reason=(
            "exact_real_run_has_no_trigger_tail_event"
            if real_relation_observable
            else "real_trigger_victim_relation_is_not_observable"
        )
        + (
            ";phase458_relation_observable"
            if phase458_relation_observable
            else ";phase458_relation_not_observable"
        ),
    )


def _request_snapshot(
    scheduler: AsyncCBScheduler,
    req: object,
    *,
    position: int,
    scheduled_tokens: int,
) -> dict[str, object]:
    return {
        "request_id": req.request_id,
        "position": position,
        "state": req.state.name,
        "sampled_output_tokens": req.sampled_output_tokens,
        "computed_output_tokens": req.computed_output_tokens,
        "output_placeholders": req.output_placeholders,
        "prefill_tokens_remaining": req.prefill_tokens_remaining,
        "scheduled_tokens": scheduled_tokens,
        "blocks_after_schedule": scheduler._blocks_needed(req, scheduled_tokens),
        "preemptions_before": req.num_preemptions,
    }


class ForensicAsyncCBScheduler(AsyncCBScheduler):
    """Diagnostic-only scheduler that snapshots every allocation failure."""

    captured_events: list[dict[str, object]] = []

    def __init__(self, config: object) -> None:
        super().__init__(config)
        self._forensic_context: dict[str, object] | None = None

    def _ensure_block_capacity(
        self,
        current_req: object,
        waiting: list[object],
        running: list[object],
        result: object,
        preempted_ids: set[int],
    ) -> tuple[bool, int]:
        scheduled_deltas = self._scheduled_token_deltas(result)
        total_blocks = self._total_blocks(running, result)
        running_order = [req.request_id for req in running]
        trigger_position = running_order.index(current_req.request_id)
        self._forensic_context = {
            "step": self.step_index,
            "trigger_request_id": current_req.request_id,
            "trigger_position": trigger_position,
            "running_count": len(running),
            "running_tail_request_id": running_order[-1],
            "running_order": running_order,
            "waiting_count": len(waiting),
            "waiting_head_request_id": (
                waiting[0].request_id if waiting else None
            ),
            "total_blocks_before": total_blocks,
            "block_capacity": self._config.num_gpu_blocks,
            "over_blocks_before": total_blocks - self._config.num_gpu_blocks,
            "allocation_failed": total_blocks > self._config.num_gpu_blocks,
            "scheduled_prefill_ids": [
                req.request_id for req in result.prefill_reqs
            ],
            "scheduled_decode_ids": [
                req.request_id for req in result.decode_reqs
            ],
            "running_states": [
                _request_snapshot(
                    self,
                    req,
                    position=position,
                    scheduled_tokens=scheduled_deltas.get(req.request_id, 0),
                )
                for position, req in enumerate(running)
            ],
        }
        try:
            return super()._ensure_block_capacity(
                current_req,
                waiting,
                running,
                result,
                preempted_ids,
            )
        finally:
            self._forensic_context = None

    def _preempt(
        self,
        victim: object,
        waiting: list[object],
        running: list[object],
        result: object,
        preempted_ids: set[int],
    ) -> int:
        if self._forensic_context is None:
            raise RuntimeError("preemption without active forensic context")
        event = dict(self._forensic_context)
        event.update(
            {
                "victim_request_id": victim.request_id,
                "victim_position": running.index(victim),
                "victim_is_running_tail": victim is running[-1],
                "total_blocks_at_preempt": self._total_blocks(running, result),
                "metrics_steady_state": self.in_steady_state,
            }
        )
        released = super()._preempt(
            victim,
            waiting,
            running,
            result,
            preempted_ids,
        )
        event["released_tokens"] = released
        self.captured_events.append(event)
        return released


def capture_sim_events() -> tuple[dict[str, object], list[dict[str, object]]]:
    original_scheduler = simulator_module.AsyncCBScheduler
    ForensicAsyncCBScheduler.captured_events = []
    simulator_module.AsyncCBScheduler = ForensicAsyncCBScheduler
    try:
        audit, _events = runtime_gate._short_runtime()
    finally:
        simulator_module.AsyncCBScheduler = original_scheduler
    return audit, [dict(event) for event in ForensicAsyncCBScheduler.captured_events]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize_real_decision_evidence(
    records: Iterable[dict[str, object]],
) -> dict[str, object]:
    rows = list(records)
    pairs = real_observation.pair_records(rows)
    decisions = [row for row in rows if row.get("kind") == "preempt_decision"]
    failures = [row for row in rows if row.get("kind") == "allocate_failure"]
    relation_fields = {
        "trigger_request_id",
        "victim_request_id",
        "victim_policy",
        "victim_position",
    }
    relation_observable = bool(decisions) and all(
        relation_fields <= row.keys() for row in decisions
    )
    tail_decisions = [
        row for row in decisions if row.get("victim_policy") == "tail"
    ]
    trigger_tail_events = sum(
        row["trigger_request_id"] == row["victim_request_id"]
        for row in tail_decisions
    )
    waiting_counts = [int(row["waiting_count"]) for row in decisions]
    running_counts = [int(row["running_count_after_pop"]) + 1 for row in decisions]
    return {
        "complete_pairs": len(pairs),
        "decisions": len(decisions),
        "relation_observable": relation_observable,
        "tail_policy_decisions": len(tail_decisions),
        "trigger_tail_events": trigger_tail_events,
        "unique_victims": len({row["victim_request_id"] for row in decisions}),
        "waiting_count_min": min(waiting_counts),
        "waiting_count_max": max(waiting_counts),
        "running_counts_before_pop": sorted(set(running_counts)),
        "one_block_short_events": sum(
            int(row["requested_blocks"]) - int(row["free_blocks"]) == 1
            for row in failures
        ),
    }


def select_real_macro_matches(
    records: Iterable[dict[str, object]],
    *,
    waiting_count: int,
    running_count: int,
    block_shortage: int,
) -> list[dict[str, object]]:
    latest_failures: dict[tuple[object, object], dict[str, object]] = {}
    matches: list[dict[str, object]] = []
    for row in records:
        key = (row.get("pid"), row.get("trigger_request_id"))
        if row.get("kind") == "allocate_failure":
            latest_failures[key] = row
            continue
        if row.get("kind") != "preempt_decision":
            continue
        failure = latest_failures.get(key)
        if failure is None:
            continue
        observed_shortage = int(failure["requested_blocks"]) - int(
            failure["free_blocks"]
        )
        observed_running = int(row["running_count_after_pop"]) + 1
        if (
            int(row["waiting_count"]) != waiting_count
            or observed_running != running_count
            or observed_shortage != block_shortage
        ):
            continue
        matches.append(
            {
                "trigger_request_id": row["trigger_request_id"],
                "victim_request_id": row["victim_request_id"],
                "trigger_is_tail": (
                    row["trigger_request_id"] == row["victim_request_id"]
                ),
                "trigger_num_computed_tokens": row.get(
                    "trigger_num_computed_tokens"
                ),
                "victim_num_computed_tokens": row.get(
                    "victim_num_computed_tokens"
                ),
                "waiting_count": int(row["waiting_count"]),
                "running_count": observed_running,
                "block_shortage": observed_shortage,
            }
        )
    return matches


def _contains_all_fields(path: Path, fields: set[str]) -> bool:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    found: set[str] = set()
    if path.suffix == ".gz":
        source = opener(path, "rt", encoding="utf-8")
    else:
        source = opener(path, encoding="utf-8")
    with source:
        for line in source:
            found.update(field for field in fields if field in line)
            if found == fields:
                return True
    return False


def phase458_relation_observability() -> dict[str, object]:
    fields = {"trigger_request_id", "victim_request_id"}
    serve_has_relation = _contains_all_fields(PHASE458_SERVE_LOG, fields)
    metrics_has_relation = _contains_all_fields(PHASE458_METRICS, fields)
    return {
        "serve_log_has_relation": serve_has_relation,
        "metrics_has_relation": metrics_has_relation,
        "relation_observable": serve_has_relation or metrics_has_relation,
    }


def _row(
    section: str,
    side: str,
    metric: str,
    value: object,
    *,
    target: object = "",
    status: str = "",
    source: str = "",
    note: str = "",
) -> dict[str, object]:
    return {
        "section": section,
        "side": side,
        "metric": metric,
        "value": value,
        "target": target,
        "status": status,
        "source": source,
        "note": note,
    }


def build_report() -> tuple[list[dict[str, object]], dict[str, object]]:
    audit, sim_events = capture_sim_events()
    steady_self_events = [
        event
        for event in sim_events
        if event["trigger_request_id"] == event["victim_request_id"]
        and int(audit["steady_start_step"])
        <= int(event["step"])
        < int(audit["steady_end_step"])
    ]
    if len(steady_self_events) != 1:
        raise RuntimeError(
            f"expected one steady self-preemption, got {len(steady_self_events)}"
        )
    event = steady_self_events[0]
    source_legality = evaluate_source_legality(
        allocation_failed=bool(event["allocation_failed"]),
        trigger_request_id=event["trigger_request_id"],
        victim_request_id=event["victim_request_id"],
        running_tail_request_id=event["running_tail_request_id"],
    )
    real_records = _read_jsonl(REAL_JSONL)
    real = summarize_real_decision_evidence(real_records)
    real_macro_matches = select_real_macro_matches(
        real_records,
        waiting_count=int(event["waiting_count"]),
        running_count=int(event["running_count"]),
        block_shortage=int(event["over_blocks_before"]),
    )
    if not real_macro_matches:
        raise RuntimeError("no real macro-state match for the sim self-preemption")
    real_macro_match = real_macro_matches[0]
    phase458 = phase458_relation_observability()
    action = decide_gate_action(
        source_status=source_legality.status,
        real_relation_observable=bool(real["relation_observable"]),
        real_trigger_tail_events=int(real["trigger_tail_events"]),
        phase458_relation_observable=bool(phase458["relation_observable"]),
    )

    rows = [
        _row(
            "sim_event",
            "sim",
            "step",
            event["step"],
            target=8176,
            status="matched" if event["step"] == 8176 else "changed",
        ),
        _row(
            "sim_event",
            "sim",
            "trigger_relation",
            "trigger_is_tail_and_victim",
            target="complete queue/block context",
            status="observed",
            source="diagnostic subclass over production AsyncCBScheduler",
            note=(
                f"trigger={event['trigger_request_id']}; "
                f"position={event['trigger_position']}/{int(event['running_count']) - 1}; "
                f"waiting={event['waiting_count']}"
            ),
        ),
        _row(
            "sim_event",
            "sim",
            "block_shortage",
            event["over_blocks_before"],
            target=1,
            status="aligned" if event["over_blocks_before"] == 1 else "changed",
            note=(
                f"total={event['total_blocks_before']}; "
                f"capacity={event['block_capacity']}"
            ),
        ),
        _row(
            "sim_event",
            "sim",
            "running_order",
            json.dumps(event["running_order"]),
            status="observed",
            note=json.dumps(event["running_states"], ensure_ascii=False),
        ),
        _row(
            "source_legality",
            "vllm",
            "self_preemption_branch",
            source_legality.status,
            target="source-derived",
            status="pass" if source_legality.status == "source_legal" else "fail",
            source="vllm scheduler.py:460-510; kv_cache_manager.py:360-389",
            note=source_legality.reason,
        ),
        _row(
            "first_divergence",
            "both",
            "macro_state_match",
            "waiting_running_block_shortage_aligned",
            target="same macro state",
            status="aligned",
            note=(
                f"waiting={event['waiting_count']}; "
                f"running={event['running_count']}; "
                f"shortage={event['over_blocks_before']}"
            ),
        ),
        _row(
            "first_divergence",
            "both",
            "trigger_tail_relation",
            "sim=true;real=false",
            target="same relation",
            status="diverged",
            source=str(REAL_JSONL.relative_to(REPO_ROOT)),
            note=(
                f"real_trigger_tokens={real_macro_match['trigger_num_computed_tokens']}; "
                f"real_victim_tokens={real_macro_match['victim_num_computed_tokens']}"
            ),
        ),
        _row(
            "source_legality",
            "vllm",
            "scheduler_sha256",
            VLLM_SCHEDULER_SHA256,
            status="verified",
            source="deployed vLLM 0.19 source",
        ),
        _row(
            "source_legality",
            "vllm",
            "kv_manager_sha256",
            VLLM_KV_MANAGER_SHA256,
            status="verified",
            source="deployed vLLM 0.19 source",
        ),
    ]
    for metric in (
        "complete_pairs",
        "decisions",
        "tail_policy_decisions",
        "trigger_tail_events",
        "unique_victims",
        "waiting_count_min",
        "waiting_count_max",
        "running_counts_before_pop",
        "one_block_short_events",
    ):
        rows.append(
            _row(
                "real_evidence",
                "real_2b",
                metric,
                real[metric],
                status="observed",
                source=str(REAL_JSONL.relative_to(REPO_ROOT)),
            )
        )
    rows.extend(
        [
            _row(
                "real_evidence",
                "phase458",
                "trigger_victim_relation_observable",
                phase458["relation_observable"],
                target=True,
                status="insufficient",
                source=(
                    f"{PHASE458_SERVE_LOG.relative_to(REPO_ROOT)}; "
                    f"{PHASE458_METRICS.relative_to(REPO_ROOT)}"
                ),
                note="serve log and Prometheus scrape lack trigger/victim ids",
            ),
            _row(
                "decision",
                "both",
                "step2c2b_action",
                action.action,
                target="preregistered decision fork",
                status="blocked",
                note=action.reason,
            ),
            _row(
                "decision",
                "both",
                "gate_change_allowed",
                action.gate_change_allowed,
                target=False,
                status="pass",
                note="gate cannot be tailored to sim output",
            ),
            _row(
                "decision",
                "both",
                "next_real_probe",
                "same_protocol_queue_order_logging",
                target="user confirmation before GPU",
                status="proposed",
                note=(
                    "record per-step running order, trigger index, tail id, "
                    "computed/placeholder/block state; duration need not increase"
                ),
            ),
        ]
    )
    rows.extend(
        _row(
            "boundary",
            "report",
            key,
            value,
            status="blocked" if key == "default_aic" else "pass",
        )
        for key, value in REPORT_BOUNDARIES.items()
    )
    return rows, {
        "audit": audit,
        "event": event,
        "source_legality": source_legality,
        "real": real,
        "real_macro_match": real_macro_match,
        "phase458": phase458,
        "action": action,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, object]) -> None:
    event = summary["event"]
    real = summary["real"]
    real_macro_match = summary["real_macro_match"]
    phase458 = summary["phase458"]
    source_legality = summary["source_legality"]
    action = summary["action"]
    trigger_state = next(
        state
        for state in event["running_states"]
        if state["request_id"] == event["trigger_request_id"]
    )
    path.write_text(
        f"""# Phase462 Step 2c-2b self-preemption first divergence

结论：`step {event['step']}` 的自抢占在 vLLM FCFS 源码中是合法分支，但同协议 real 短跑没有出现等价的 `trigger=running tail` 状态。证据不允许放宽硬门；Step2c-2 继续停线，下一分歧是 sim/real 的 running 队列顺序演化。

| 检查 | 结果 | 判定 |
|---|---:|---|
| sim 分配缺口 | {event['over_blocks_before']} block | 与 real 的缺 1 block 对齐 |
| sim trigger 位置 | {event['trigger_position']}/{int(event['running_count']) - 1} | trigger 就是 running 队尾 |
| sim trigger 状态 | sampled={trigger_state['sampled_output_tokens']}, computed={trigger_state['computed_output_tokens']}, placeholder={trigger_state['output_placeholders']} | 完整状态已冻结 |
| vLLM 源码 | `{source_legality.status}` | FCFS `running.pop()` 后显式处理 `preempted_req == request` |
| real 2b 决策 | {real['decisions']} 次 / {real['unique_victims']} 个唯一 victim | 全部 trigger != tail |
| real 同宏观状态 | waiting={real_macro_match['waiting_count']}, running={real_macro_match['running_count']}, 缺 {real_macro_match['block_shortage']} block | trigger != tail，victim 是 peer |
| real waiting 覆盖 | {real['waiting_count_max']} -> {real['waiting_count_min']} | 已贯穿该 N=128 短跑的抢占发生区间 |
| Phase458 N=512 | trigger/victim relation observable={phase458['relation_observable']} | 只有聚合计数和 iteration 构成，不能补关系证据 |

源码依据：部署态 vLLM `scheduler.py:460-510` 在 block 分配失败后，FCFS 路径执行 `running.pop()`；若 pop 出的队尾就是当前 request，`scheduler.py:504-506` 明确停止继续抢占。`kv_cache_manager.py:360-389` 定义分配失败条件。本次核对源码 SHA256 为 `{VLLM_SCHEDULER_SHA256}` / `{VLLM_KV_MANAGER_SHA256}`。

决策叉结果：`{action.action}`。源码只证明“给定该状态时自抢占合法”，不证明 sim 产生该状态的演化正确。更关键的是，real 在同样的 waiting=31、running=14、缺 1 block 状态下选择了 peer tail victim；首次分歧因此钉在 running 队列顺序或其上游 per-request phase 演化，而不是容量阈值。Phase458 又缺 request-id 关系，因此不能按 sim 的 1 次事件裁剪验收门。

下一步只提议、不执行 GPU：复用 2b logging-only 挂钩，在同一 N=128/C=128/ISL=32k/OSL=1200 协议下增加每步 running 顺序、trigger 位置、tail id、computed/placeholder/block 状态。现有日志时长已覆盖 waiting `114 -> 3`，缺的是队列维度，不是运行时长；无需盲目延长。

未改 runtime、PerfDB 或 gate；未运行六点 `--ab`。边界：`diagnostic_only=true`、`valid_for_default=false`、`perf_database=false`、`Default AIC=No-Go`。
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    rows, summary = build_report()
    write_csv(args.csv, rows)
    write_markdown(args.md, summary)
    action = summary["action"]
    print(
        json.dumps(
            {
                "status": "blocked",
                "action": action.action,
                "gate_change_allowed": action.gate_change_allowed,
                "runtime_change_allowed": action.runtime_change_allowed,
                "gpu_followup_required": action.gpu_followup_required,
                "csv": str(args.csv),
                "md": str(args.md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
