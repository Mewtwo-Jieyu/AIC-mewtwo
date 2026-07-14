#!/usr/bin/env python3
"""Phase462 Step2c-2 runtime integration gates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import scripts.analyze_phase462_engine_loop_state_machine as oracle  # noqa: E402
import scripts.validate_cb_simulator as validate  # noqa: E402
from aiconfigurator.sdk.backends.cb_simulator.simulator import CBSimulator  # noqa: E402


SCENARIO = "K2.5-tp8ep8-32k3k"
SHORT_OSL = 1200
SHORT_REQUESTS = 128
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase462_engine_loop_runtime_gate.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase462_engine_loop_runtime_gate.md"
EXPOSURE_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase462_exposure_sensitivity.csv"
CSV_FIELDS = ["section", "metric", "value", "target", "status", "note"]

REPORT_BOUNDARIES = {
    "diagnostic_only": True,
    "valid_for_default": False,
    "perf_database": False,
    "default_aic": "No-Go",
}


@dataclass(frozen=True)
class RuntimeGateVerdict:
    oracle_passed: bool
    steady_passed: bool
    default_enable_allowed: bool
    status: str
    reason: str


def evaluate_runtime_gate(
    *,
    drain_match: float,
    first_schedule_match: float,
    first_16_match: float,
    preemptions: int,
    steady_self_preemptions: int,
    repeat_victim_events: int,
) -> RuntimeGateVerdict:
    oracle_passed = (
        drain_match == 1.0
        and first_schedule_match >= 0.9375
        and first_16_match == 1.0
    )
    steady_passed = (
        abs(preemptions - 10) <= 2
        and steady_self_preemptions == 0
        and repeat_victim_events == 0
    )
    if not oracle_passed:
        reason = "oracle_regression_anchor_failed"
    elif not steady_passed:
        reason = "steady_state_signature_failed"
    else:
        reason = "all_runtime_gates_passed"
    return RuntimeGateVerdict(
        oracle_passed=oracle_passed,
        steady_passed=steady_passed,
        default_enable_allowed=oracle_passed and steady_passed,
        status="pass" if oracle_passed and steady_passed else "blocked",
        reason=reason,
    )


def _oracle_metrics() -> dict[str, float]:
    rows, _summary = oracle.build_report()
    mode = "target_run_drain_oracle_plus_future_oracle"
    values = {
        row["metric"]: float(row["value"])
        for row in rows
        if row["section"] == "prototype" and row["mode"] == mode
    }
    return {
        "drain_step_match": values["drain_step_match"],
        "first_schedule_step_match": values["first_schedule_step_match"],
        "first_16_match": values["first_16_match"],
    }


def _short_runtime() -> tuple[dict[str, object], list[dict[str, int | bool]]]:
    point = next(item for item in validate.MULTI_CONFIG_DATA if item.name == SCENARIO)
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
    config = replace(
        config,
        num_requests=SHORT_REQUESTS,
        warmup_requests=0,
        engine_loop_enabled=True,
    )
    sim = CBSimulator(backend, model, database, config)
    sim.run(
        isl=point.isl,
        osl=SHORT_OSL,
        concurrency=point.batch_size,
        prefix=0,
        num_gpus=point.tp,
    )
    return sim.get_last_engine_loop_audit(), sim.get_last_preemption_events()


def _exposure_evidence() -> str:
    with EXPOSURE_CSV.open(newline="", encoding="utf-8") as source:
        row = next(
            item
            for item in csv.DictReader(source)
            if item["section"] == "self_preemption_attribution"
            and item["scenario"] == "tp8_32k3k_short"
        )
    return f"value={row['value']}; {row['note']}"


def _row(
    section: str,
    metric: str,
    value: object,
    *,
    target: object = "",
    status: str = "",
    note: str = "",
) -> dict[str, object]:
    return {
        "section": section,
        "metric": metric,
        "value": value,
        "target": target,
        "status": status,
        "note": note,
    }


def build_report() -> tuple[list[dict[str, object]], dict[str, object]]:
    anchor = _oracle_metrics()
    audit, events = _short_runtime()
    verdict = evaluate_runtime_gate(
        drain_match=anchor["drain_step_match"],
        first_schedule_match=anchor["first_schedule_step_match"],
        first_16_match=anchor["first_16_match"],
        preemptions=int(audit["preemptions"]),
        steady_self_preemptions=int(audit["steady_self_preemptions"]),
        repeat_victim_events=int(audit["repeat_victim_events"]),
    )
    self_events = [
        {
            **event,
            "lifecycle_steady_state": (
                int(audit["steady_start_step"])
                <= int(event["step"])
                < int(audit["steady_end_step"])
            ),
        }
        for event in events
        if event["trigger_request_id"] == event["victim_request_id"]
    ]
    rows = [
        _row(
            "oracle",
            "drain_step_match",
            anchor["drain_step_match"],
            target="1.0",
            status="pass" if anchor["drain_step_match"] == 1.0 else "fail",
        ),
        _row(
            "oracle",
            "first_schedule_step_match",
            anchor["first_schedule_step_match"],
            target=">=0.9375",
            status="pass" if anchor["first_schedule_step_match"] >= 0.9375 else "fail",
        ),
        _row(
            "oracle",
            "first_16_match",
            anchor["first_16_match"],
            target="1.0",
            status="pass" if anchor["first_16_match"] == 1.0 else "fail",
        ),
        _row(
            "steady",
            "preemptions",
            audit["preemptions"],
            target="10+/-2",
            status="pass" if abs(int(audit["preemptions"]) - 10) <= 2 else "fail",
        ),
        _row(
            "steady",
            "steady_self_preemptions",
            audit["steady_self_preemptions"],
            target="0",
            status="pass" if audit["steady_self_preemptions"] == 0 else "fail",
            note=json.dumps(self_events, ensure_ascii=False),
        ),
        _row(
            "steady",
            "repeat_victim_events",
            audit["repeat_victim_events"],
            target="0",
            status="pass" if audit["repeat_victim_events"] == 0 else "fail",
        ),
        _row("runtime", "sim_runtime_seconds", audit["sim_runtime_seconds"], status="measured"),
        _row(
            "preregistered_evidence",
            "phase462_2a3f_self_preemption",
            _exposure_evidence(),
            target="steady=0",
            status="contradicts_gate",
        ),
        _row("gate", "step2c2", verdict.status, target="pass", status=verdict.status, note=verdict.reason),
        _row(
            "gate",
            "default_engine_loop",
            False,
            target=False,
            status="pass",
            note="engine_loop_enabled remains false by default",
        ),
        _row(
            "boundary",
            "multi_replica_engine_loop",
            "deferred_fail_closed",
            target="Phase462 Step 3",
            status="pass",
            note="enabled DP calls raise instead of silently using the legacy loop",
        ),
    ]
    rows.extend(
        _row("boundary", key, value, status="blocked" if key == "default_aic" else "pass")
        for key, value in REPORT_BOUNDARIES.items()
    )
    return rows, {
        "anchor": anchor,
        "audit": audit,
        "self_events": self_events,
        "verdict": verdict,
        "exposure_evidence": _exposure_evidence(),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=CSV_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, object]) -> None:
    anchor = summary["anchor"]
    audit = summary["audit"]
    verdict = summary["verdict"]
    drain_status = "PASS" if anchor["drain_step_match"] == 1.0 else "FAIL"
    first_schedule_status = (
        "PASS" if anchor["first_schedule_step_match"] >= 0.9375 else "FAIL"
    )
    first_16_status = "PASS" if anchor["first_16_match"] == 1.0 else "FAIL"
    preemption_status = (
        "PASS" if abs(int(audit["preemptions"]) - 10) <= 2 else "FAIL"
    )
    steady_self_status = (
        "PASS" if int(audit["steady_self_preemptions"]) == 0 else "FAIL"
    )
    repeat_status = (
        "PASS" if int(audit["repeat_victim_events"]) == 0 else "FAIL"
    )
    path.write_text(
        f"""# Phase462 Step 2c-2 runtime gate

结论：oracle 回归锚通过，短跑稳态硬门失败；停在六点 `--ab` 前，新引擎环不进入默认路径。

| 门 | 结果 | 目标 | 判定 |
|---|---:|---:|---|
| oracle drain | {anchor['drain_step_match']:.2%} | 100% | {drain_status} |
| oracle 首调度步 | {anchor['first_schedule_step_match']:.2%} | >=93.75% | {first_schedule_status} |
| oracle 前 16 步 | {anchor['first_16_match']:.2%} | 100% | {first_16_status} |
| 短跑抢占 | {audit['preemptions']} | 10+/-2 | {preemption_status} |
| 短跑稳态自抢占 | {audit['steady_self_preemptions']} | 0 | {steady_self_status} |
| 短跑重复 victim | {audit['repeat_victim_events']} | 0 | {repeat_status} |
| sim 实际运行时长 | {float(audit['sim_runtime_seconds']):.3f}s | 记录 | measured |

稳态窗口为 `[{audit['steady_start_step']}, {audit['steady_end_step']})`，模式 `{audit['steady_window_mode']}`。自抢占事件：

```json
{json.dumps(summary['self_events'], ensure_ascii=False, indent=2)}
```

2a-3f 已登记的 32k 证据同样包含稳态自抢占：`{summary['exposure_evidence']}`。因此本次失败不是 arrival 接线回归，也不能通过改窗口、经验阈值或随机去同步修补。

`step2c2={verdict.status}`，原因 `{verdict.reason}`。`engine_loop_enabled` 默认仍为 false；
DP 新路径在 Step 3 前显式拒绝，不静默切旧环。未运行六点 `--ab`，未进入归档行、DP 或 dp2-32k3k 级联。

边界：`diagnostic_only=true`、`valid_for_default=false`、`perf_database=false`、`Default AIC=No-Go`。
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
    verdict = summary["verdict"]
    print(
        json.dumps(
            {
                "status": verdict.status,
                "reason": verdict.reason,
                "default_enable_allowed": verdict.default_enable_allowed,
                "csv": str(args.csv),
                "md": str(args.md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if verdict.default_enable_allowed else 1


if __name__ == "__main__":
    raise SystemExit(main())
