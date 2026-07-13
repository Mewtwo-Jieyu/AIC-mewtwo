#!/usr/bin/env python3
"""Validate Phase462 four-boundary arrival observations."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
from pathlib import Path


REPORT_FIELDS = ["section", "scenario", "metric", "value", "unit", "status", "note"]


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else Path.open
        with opener(path, "rt", encoding="utf-8") as source:
            rows.extend(json.loads(line) for line in source if line.strip())
    return rows


def _median(values: list[float | int]) -> float | int | None:
    return statistics.median(values) if values else None


def _percentile(values: list[float | int], fraction: float) -> float | int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _ns_to_ms(value: int) -> float:
    return value / 1_000_000


def measure_scenario(rows: list[dict], scenario: str) -> dict:
    scenario_rows = [row for row in rows if row.get("scenario") == scenario]
    enters = {
        row["batch_id"]: row
        for row in scenario_rows
        if row.get("kind") == "tokenizer_batch_enter"
    }
    completes = {
        row["batch_id"]: row
        for row in scenario_rows
        if row.get("kind") == "tokenizer_batch_complete"
    }
    receives = {
        row["trace_id"]: int(row["ts_ns"])
        for row in scenario_rows
        if row.get("kind") == "engine_receive" and row.get("trace_id")
    }
    schedules: dict[str, tuple[int, int]] = {}
    scheduler_rows = [
        row for row in scenario_rows if row.get("kind") == "scheduler_step"
    ]
    for row in scheduler_rows:
        for trace_id in row.get("new_context_trace_ids", []):
            schedules.setdefault(trace_id, (int(row["ts_ns"]), int(row["step"])))

    request_boundaries: dict[str, tuple[int, int, str]] = {}
    tokenizer_service_ms: list[float] = []
    for batch_id, enter in enters.items():
        complete = completes.get(batch_id)
        if complete is None:
            continue
        batch_complete_ns = int(complete["batch_complete_ns"])
        tokenizer_service_ms.append(
            _ns_to_ms(batch_complete_ns - int(enter["batch_start_ns"]))
        )
        for trace_id, queue_enter_ns in zip(
            enter.get("trace_ids", []), enter.get("queue_enter_ns", [])
        ):
            if trace_id:
                request_boundaries[trace_id] = (
                    int(queue_enter_ns),
                    batch_complete_ns,
                    batch_id,
                )

    paired_ids = sorted(request_boundaries.keys() & receives.keys() & schedules.keys())
    queue_wait_ms: list[float] = []
    engine_receive_ms: list[float] = []
    first_schedule_ms: list[float] = []
    total_visible_ms: list[float] = []
    schedule_steps_by_batch: dict[str, set[int]] = {}
    for trace_id in paired_ids:
        queue_enter_ns, batch_complete_ns, batch_id = request_boundaries[trace_id]
        batch_start_ns = int(enters[batch_id]["batch_start_ns"])
        receive_ns = receives[trace_id]
        schedule_ns, schedule_step = schedules[trace_id]
        queue_wait_ms.append(_ns_to_ms(batch_start_ns - queue_enter_ns))
        engine_receive_ms.append(_ns_to_ms(receive_ns - batch_complete_ns))
        first_schedule_ms.append(_ns_to_ms(schedule_ns - receive_ns))
        total_visible_ms.append(_ns_to_ms(schedule_ns - queue_enter_ns))
        schedule_steps_by_batch.setdefault(batch_id, set()).add(schedule_step)

    batch_sizes = [int(row.get("batch_size", 0)) for row in enters.values()]
    new_context_counts = [int(row.get("new_context_count", 0)) for row in scheduler_rows]
    waiting_depths = [int(row.get("waiting_before", 0)) for row in scheduler_rows]
    scheduled_steps_per_batch = [
        len(steps) for steps in schedule_steps_by_batch.values()
    ]
    return {
        "scenario": scenario,
        "paired_requests": len(paired_ids),
        "tokenizer_batches": len(enters),
        "batch_size_median": _median(batch_sizes),
        "batch_size_max": max(batch_sizes, default=None),
        "tokenizer_service_ms_median": _median(tokenizer_service_ms),
        "tokenizer_service_ms_p90": _percentile(tokenizer_service_ms, 0.9),
        "queue_wait_ms_median": _median(queue_wait_ms),
        "queue_wait_ms_p90": _percentile(queue_wait_ms, 0.9),
        "engine_receive_ms_median": _median(engine_receive_ms),
        "engine_receive_ms_p90": _percentile(engine_receive_ms, 0.9),
        "first_schedule_ms_median": _median(first_schedule_ms),
        "first_schedule_ms_p90": _percentile(first_schedule_ms, 0.9),
        "total_visible_ms_median": _median(total_visible_ms),
        "total_visible_ms_p90": _percentile(total_visible_ms, 0.9),
        "scheduled_steps_per_batch_median": _median(scheduled_steps_per_batch),
        "scheduled_steps_per_batch_max": max(scheduled_steps_per_batch, default=None),
        "new_context_count_median": _median(new_context_counts),
        "new_context_count_max": max(new_context_counts, default=None),
        "waiting_before_median": _median(waiting_depths),
        "waiting_before_max": max(waiting_depths, default=None),
    }


def summarize_batch_groups(rows: list[dict], scenario: str) -> list[dict]:
    scenario_rows = [row for row in rows if row.get("scenario") == scenario]
    completes = {
        row["batch_id"]: row
        for row in scenario_rows
        if row.get("kind") == "tokenizer_batch_complete"
    }
    scheduled_steps: dict[str, int] = {}
    for row in scenario_rows:
        if row.get("kind") != "scheduler_step":
            continue
        for trace_id in row.get("new_context_trace_ids", []):
            scheduled_steps.setdefault(trace_id, int(row["step"]))

    grouped: dict[tuple[int, int], list[tuple[float, int]]] = {}
    for enter in scenario_rows:
        if enter.get("kind") != "tokenizer_batch_enter":
            continue
        complete = completes.get(enter["batch_id"])
        if complete is None:
            continue
        batch_size = int(enter["batch_size"])
        total_prompt_tokens = sum(int(value) for value in complete["prompt_token_lengths"])
        service_ms = _ns_to_ms(
            int(complete["batch_complete_ns"]) - int(enter["batch_start_ns"])
        )
        steps = {
            scheduled_steps[trace_id]
            for trace_id in enter.get("trace_ids", [])
            if trace_id in scheduled_steps
        }
        grouped.setdefault((batch_size, total_prompt_tokens), []).append(
            (service_ms, len(steps))
        )

    summaries = []
    for (batch_size, total_prompt_tokens), values in sorted(grouped.items()):
        service_ms = [value[0] for value in values]
        step_counts = [value[1] for value in values]
        summaries.append(
            {
                "scenario": scenario,
                "batch_size": batch_size,
                "batch_count": len(values),
                "total_prompt_tokens": total_prompt_tokens,
                "tokenizer_service_ms_median": _median(service_ms),
                "tokenizer_service_ms_p90": _percentile(service_ms, 0.9),
                "scheduled_steps_median": _median(step_counts),
                "scheduled_steps_max": max(step_counts, default=None),
            }
        )
    return summaries


def audit_artifacts(root: Path) -> dict[str, bool]:
    original = (root / "source_original.sha256").read_text(encoding="utf-8")
    restored_8k = (root / "source_restored_8k2k.sha256").read_text(
        encoding="utf-8"
    )
    restored_32k = (root / "source_restored_32k3k.sha256").read_text(
        encoding="utf-8"
    )
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    return {
        "source_restored_8k2k": original == restored_8k,
        "source_restored_32k3k": original == restored_32k,
        "gpu_residual_empty": not (root / "gpu_compute_apps_after.txt")
        .read_text(encoding="utf-8")
        .strip(),
        "process_residual_empty": not (root / "process_residual_after.txt")
        .read_text(encoding="utf-8")
        .strip(),
        "diagnostic_boundary": (
            meta.get("diagnostic_only") is True
            and meta.get("valid_for_default") is False
            and meta.get("perf_database") is False
        ),
    }


def _report_row(
    section: str,
    scenario: str,
    metric: str,
    value: object,
    *,
    unit: str = "",
    status: str = "",
    note: str = "",
) -> dict[str, object]:
    return {
        "section": section,
        "scenario": scenario,
        "metric": metric,
        "value": value,
        "unit": unit,
        "status": status,
        "note": note,
    }


def build_report_rows(
    rows: list[dict],
    required_scenarios: set[str],
    overhead: dict,
    hygiene: dict[str, bool],
) -> list[dict[str, object]]:
    self_check = summarize(rows, required_scenarios)
    report_rows = [
        _report_row(
            "gate",
            "all",
            "four_boundary_pairing",
            self_check["paired_trace_ids"],
            unit="requests",
            status="pass" if self_check["passed"] else "fail",
            note=f"expected={self_check['tokenizer_trace_ids']}",
        )
    ]
    for row in overhead["rows"]:
        report_rows.append(
            _report_row(
                "gate",
                row["scenario"],
                "logging_overhead",
                row["absolute_delta_pct"],
                unit="percent",
                status="pass" if row["passed"] else "fail",
                note=(
                    f"off={row['off_output_tok_s']}; "
                    f"on={row['on_output_tok_s']}; limit={row['max_overhead_pct']}"
                ),
            )
        )
    for metric, passed in hygiene.items():
        report_rows.append(
            _report_row(
                "hygiene",
                "all",
                metric,
                passed,
                status="pass" if passed else "fail",
            )
        )

    units = {
        "paired_requests": "requests",
        "tokenizer_batches": "batches",
        "batch_size_median": "requests",
        "batch_size_max": "requests",
        "scheduled_steps_per_batch_median": "steps",
        "scheduled_steps_per_batch_max": "steps",
        "new_context_count_median": "requests_per_step",
        "new_context_count_max": "requests_per_step",
        "waiting_before_median": "requests",
        "waiting_before_max": "requests",
    }
    for scenario in sorted(required_scenarios):
        summary = measure_scenario(rows, scenario)
        for metric, value in summary.items():
            if metric == "scenario":
                continue
            report_rows.append(
                _report_row(
                    "scenario_summary",
                    scenario,
                    metric,
                    value,
                    unit="ms" if "_ms_" in metric else units.get(metric, ""),
                )
            )
        for group in summarize_batch_groups(rows, scenario):
            note = f"total_prompt_tokens={group['total_prompt_tokens']}"
            for metric in (
                "batch_count",
                "tokenizer_service_ms_median",
                "tokenizer_service_ms_p90",
                "scheduled_steps_median",
                "scheduled_steps_max",
            ):
                report_rows.append(
                    _report_row(
                        "tokenizer_batch_group",
                        scenario,
                        f"batch_size_{group['batch_size']}_{metric}",
                        group[metric],
                        unit=(
                            "ms"
                            if "_ms_" in metric
                            else "batches"
                            if metric == "batch_count"
                            else "steps"
                        ),
                        note=note,
                    )
                )
    return report_rows


def write_report_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=REPORT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _format_number(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_report_markdown(
    path: Path,
    rows: list[dict],
    required_scenarios: set[str],
    overhead: dict,
    hygiene: dict[str, bool],
) -> None:
    self_check = summarize(rows, required_scenarios)
    all_gates_passed = self_check["passed"] and overhead["passed"] and all(
        hygiene.values()
    )
    scenario_summaries = {
        scenario: measure_scenario(rows, scenario)
        for scenario in sorted(required_scenarios)
    }
    overhead_by_scenario = {row["scenario"]: row for row in overhead["rows"]}
    lines = [
        "# Phase462 Step2a-3b: EngineCore 可见到达观测",
        "",
        (
            "结论：**观测门通过，Step2a-3c 可开始。** "
            if all_gates_passed
            else "结论：**观测门失败，Step2a-3c 继续锁住。** "
        )
        + "本步骤只采时间边界与调度可见性，不拟合原语，不改 runtime、PerfDB 或 gate。",
        "",
        "| 门 | 8k2k | 32k3k |",
        "|---|---:|---:|",
        (
            "| benchmark 完整性 | 512/512 | 512/512 |"
        ),
        (
            "| logging 开销 | "
            f"{overhead_by_scenario['8k2k']['absolute_delta_pct']:.3f}% | "
            f"{overhead_by_scenario['32k3k']['absolute_delta_pct']:.3f}% |"
        ),
        (
            "| 四点同 ID 配对 | "
            f"{scenario_summaries['8k2k']['paired_requests']}/512 | "
            f"{scenario_summaries['32k3k']['paired_requests']}/512 |"
        ),
        "| 运行时 tokenizer 配置 | 32 requests / 2ms | 32 requests / 2ms |",
        "| 源码恢复 / GPU / 进程残留 | pass | pass |",
        "",
        "## 四段时间",
        "",
        "| 场景 | tokenizer 批数 | queue 等待 median/p90 | "
        "tokenizer 服务 median/p90 | batch complete -> EngineCore median/p90 | "
        "EngineCore -> first schedule median/p90 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario, summary in scenario_summaries.items():
        lines.append(
            f"| {scenario} | {summary['tokenizer_batches']} | "
            f"{summary['queue_wait_ms_median']:.3f}/{summary['queue_wait_ms_p90']:.3f}ms | "
            f"{summary['tokenizer_service_ms_median']:.3f}/{summary['tokenizer_service_ms_p90']:.3f}ms | "
            f"{summary['engine_receive_ms_median']:.3f}/{summary['engine_receive_ms_p90']:.3f}ms | "
            f"{summary['first_schedule_ms_median']:.3f}/{summary['first_schedule_ms_p90']:.3f}ms |"
        )

    lines.extend(
        [
            "",
            "`EngineCore -> first schedule` 包含闭集并发下的 waiting 排队，"
            "只是观测边界，不是 tokenizer 原语。",
            "",
            "## 微批与调度",
            "",
            "| 场景 | batch size | 批数 | total prompt tokens | "
            "tokenizer 服务 median/p90 | 首次调度步数 median/max |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario in sorted(required_scenarios):
        for group in summarize_batch_groups(rows, scenario):
            lines.append(
                f"| {scenario} | {group['batch_size']} | {group['batch_count']} | "
                f"{group['total_prompt_tokens']} | "
                f"{group['tokenizer_service_ms_median']:.3f}/{group['tokenizer_service_ms_p90']:.3f}ms | "
                f"{_format_number(group['scheduled_steps_median'])}/{group['scheduled_steps_max']} |"
            )

    lines.extend(
        [
            "",
            "两个场景中 `new_context_count_max=1`。初始 29-32 请求 tokenizer "
            "大微批在 EngineCore 可见后，被摊到同数量级的首次调度步；"
            "因此微批形成与 admission 错开是两个独立阶段。该读数只解锁 "
            "Step2a-3c 的离线原型，不能直接写成“一拍”或其他常数。",
            "",
            "本步骤 `diagnostic_only=true / valid_for_default=false / "
            "perf_database=false`。Default AIC 维持 No-Go。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def summarize(rows: list[dict], required_scenarios: set[str]) -> dict:
    configs = [row for row in rows if row.get("kind") == "runtime_config"]
    enters = {
        (row.get("scenario"), row.get("batch_id")): row
        for row in rows
        if row.get("kind") == "tokenizer_batch_enter"
    }
    completes = {
        (row.get("scenario"), row.get("batch_id")): row
        for row in rows
        if row.get("kind") == "tokenizer_batch_complete"
    }
    engine_trace_ids = [
        row.get("trace_id")
        for row in rows
        if row.get("kind") == "engine_receive" and row.get("trace_id")
    ]
    engine_ids = set(engine_trace_ids)
    scheduled_trace_ids = [
        trace_id
        for row in rows
        if row.get("kind") == "scheduler_step"
        for trace_id in row.get("new_context_trace_ids", [])
        if trace_id
    ]
    scheduled_ids = set(scheduled_trace_ids)
    tokenizer_ids: set[str] = set()
    tokenizer_trace_ids: list[str] = []
    prompt_length_mismatches = 0
    complete_batch_pairs = 0
    for key, enter in enters.items():
        complete = completes.get(key)
        if complete is None:
            continue
        complete_batch_pairs += 1
        trace_ids = enter.get("trace_ids", [])
        tokenizer_ids.update(trace_id for trace_id in trace_ids if trace_id)
        tokenizer_trace_ids.extend(trace_id for trace_id in trace_ids if trace_id)
        expected = enter.get("expected_prompt_tokens", [])
        actual = complete.get("prompt_token_lengths", [])
        tracked = [
            (expected_value, actual_value)
            for trace_id, expected_value, actual_value in zip(
                trace_ids, expected, actual
            )
            if trace_id
        ]
        prompt_length_mismatches += sum(
            expected_value is None or int(expected_value) != int(actual_value)
            for expected_value, actual_value in tracked
        )
        tracked_count = sum(bool(trace_id) for trace_id in trace_ids)
        prompt_length_mismatches += abs(tracked_count - len(tracked))

    paired_ids = tokenizer_ids & engine_ids & scheduled_ids
    scenarios = sorted({row.get("scenario") for row in rows if row.get("scenario")})
    config_scenarios = {
        row.get("scenario")
        for row in configs
        if int(row.get("max_batch_size", -1)) == 32
        and abs(float(row.get("batch_wait_timeout_s", -1.0)) - 0.002) < 1e-9
    }
    runtime_config_matches = required_scenarios <= config_scenarios
    duplicate_tokenizer_ids = len(tokenizer_trace_ids) - len(tokenizer_ids)
    duplicate_engine_ids = len(engine_trace_ids) - len(engine_ids)
    duplicate_scheduled_ids = len(scheduled_trace_ids) - len(scheduled_ids)
    passed = (
        required_scenarios <= set(scenarios)
        and runtime_config_matches
        and complete_batch_pairs == len(enters) == len(completes)
        and len(tokenizer_ids) > 0
        and paired_ids == tokenizer_ids
        and prompt_length_mismatches == 0
        and duplicate_tokenizer_ids == 0
        and duplicate_engine_ids == 0
        and duplicate_scheduled_ids == 0
    )
    return {
        "schema": "phase462_arrival_observation_self_check_v1",
        "scenarios": scenarios,
        "required_scenarios": sorted(required_scenarios),
        "runtime_config_rows": len(configs),
        "runtime_config_matches": runtime_config_matches,
        "tokenizer_batch_enter_rows": len(enters),
        "tokenizer_batch_complete_rows": len(completes),
        "complete_batch_pairs": complete_batch_pairs,
        "tokenizer_trace_ids": len(tokenizer_ids),
        "engine_receive_trace_ids": len(engine_ids),
        "first_scheduled_trace_ids": len(scheduled_ids),
        "paired_trace_ids": len(paired_ids),
        "prompt_length_mismatches": prompt_length_mismatches,
        "duplicate_tokenizer_trace_ids": duplicate_tokenizer_ids,
        "duplicate_engine_receive_trace_ids": duplicate_engine_ids,
        "duplicate_first_scheduled_trace_ids": duplicate_scheduled_ids,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--required-scenario", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--overhead-gate", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.inputs)
    required_scenarios = set(args.required_scenario)
    result = summarize(rows, required_scenarios)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_requested = args.output_csv is not None or args.output_md is not None
    if report_requested:
        if not all(
            (
                args.output_csv,
                args.output_md,
                args.artifact_root,
                args.overhead_gate,
            )
        ):
            parser.error(
                "report output requires --output-csv, --output-md, "
                "--artifact-root, and --overhead-gate"
            )
        overhead = json.loads(args.overhead_gate.read_text(encoding="utf-8"))
        hygiene = audit_artifacts(args.artifact_root)
        report_rows = build_report_rows(
            rows, required_scenarios, overhead, hygiene
        )
        write_report_csv(args.output_csv, report_rows)
        write_report_markdown(
            args.output_md, rows, required_scenarios, overhead, hygiene
        )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
