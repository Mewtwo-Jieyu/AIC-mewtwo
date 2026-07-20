#!/usr/bin/env python3
"""Summarize Phase466 v3 rank-local iteration timing diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "phase466_rank_timing_report_v1"
FORMAL_SCENARIOS = {
    "K2.5-tp4ep8dp2-32k3k": 2,
    "K2.5-tp8ep8-8k2k-bt65536": 1,
    "K2.5-tp4ep8dp2-8k2k-bt65536": 2,
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty_csv:{path}")
    return rows


def _sum(rows: list[dict[str, str]], key: str, cast: Any = int) -> int | float:
    return sum(cast(row[key]) for row in rows)


def _window_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    elapsed_ms = float(_sum(rows, "iteration_elapsed_ms", float))
    prefill_tokens = int(_sum(rows, "scheduled_prefill_tokens"))
    decode_tokens = int(_sum(rows, "scheduled_decode_tokens"))
    scheduled_tokens = prefill_tokens + decode_tokens
    if scheduled_tokens <= 0 or not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
        raise ValueError("rank_window_has_no_timed_progress")
    return {
        "progress_window_id": int(rows[0]["progress_window_id"]),
        "progress_start_tokens": min(int(row["progress_start_tokens"]) for row in rows),
        "progress_end_tokens": max(int(row["progress_end_tokens"]) for row in rows),
        "iteration_count": len(rows),
        "elapsed_ms": elapsed_ms,
        "scheduled_tokens": scheduled_tokens,
        "elapsed_ms_per_scheduled_token": elapsed_ms / scheduled_tokens,
        "scheduled_prefill_tokens": prefill_tokens,
        "scheduled_decode_tokens": decode_tokens,
        "prefill_request_count": int(_sum(rows, "prefill_request_count")),
        "decode_request_count": int(_sum(rows, "decode_request_count")),
    }


def summarize_scenario(root: Path, scenario: str, *, expected_dp: int) -> dict[str, Any]:
    run_dir = root / "formal" / scenario
    probe = _read_json(run_dir / "probe_summary.json")
    if probe.get("status") != "ARTIFACT_VALID":
        raise ValueError(f"formal_artifact_not_valid:{scenario}")
    rows = _read_csv(run_dir / "iteration_rows.csv")
    rank_rows = _read_csv(run_dir / "rank_summary.csv")
    preemptions = {int(row["rank_id"]): float(row["preemptions"]) for row in rank_rows}
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["rank_id"])].append(row)
    expected_ranks = list(range(expected_dp))
    if sorted(grouped) != expected_ranks or sorted(preemptions) != expected_ranks:
        raise ValueError(f"rank_timing_rank_set_mismatch:{scenario}")
    ranks: list[dict[str, Any]] = []
    for rank in expected_ranks:
        ordered = sorted(grouped[rank], key=lambda row: int(row["iteration_seq"]))
        work_rows = [
            row
            for row in ordered
            if int(row["scheduled_prefill_tokens"])
            + int(row["scheduled_decode_tokens"])
            > 0
        ]
        if not work_rows:
            raise ValueError(f"rank_timing_has_no_work_iterations:{scenario}:{rank}")
        by_window: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in work_rows:
            by_window[int(row["progress_window_id"])].append(row)
        windows = [_window_summary(by_window[key]) for key in sorted(by_window)]
        elapsed_ms = sum(window["elapsed_ms"] for window in windows)
        scheduled_tokens = sum(window["scheduled_tokens"] for window in windows)
        ranks.append(
            {
                "rank_id": rank,
                "progress_start_tokens": windows[0]["progress_start_tokens"],
                "progress_end_tokens": windows[-1]["progress_end_tokens"],
                "elapsed_ms": elapsed_ms,
                "scheduled_tokens": scheduled_tokens,
                "elapsed_ms_per_scheduled_token": elapsed_ms / scheduled_tokens,
                "scheduled_prefill_tokens": sum(
                    window["scheduled_prefill_tokens"] for window in windows
                ),
                "scheduled_decode_tokens": sum(
                    window["scheduled_decode_tokens"] for window in windows
                ),
                "prefill_request_count": sum(
                    window["prefill_request_count"] for window in windows
                ),
                "decode_request_count": sum(
                    window["decode_request_count"] for window in windows
                ),
                "preemptions": preemptions[rank],
                "progress_windows": windows,
            }
        )
    comparison = None
    if expected_dp == 2:
        elapsed_per_token = [rank["elapsed_ms_per_scheduled_token"] for rank in ranks]
        comparison = {
            "rank0_over_rank1_elapsed_per_token": elapsed_per_token[0]
            / elapsed_per_token[1],
            "max_over_min_elapsed_per_token": max(elapsed_per_token)
            / min(elapsed_per_token),
            "progress_end_token_delta": ranks[0]["progress_end_tokens"]
            - ranks[1]["progress_end_tokens"],
        }
    return {
        "scenario": scenario,
        "topology_role": "single_rank_control" if expected_dp == 1 else "dp_rank_comparison",
        "rank_count": expected_dp,
        "ranks": ranks,
        "rank_comparison": comparison,
    }


def analyze(artifact_root: Path) -> dict[str, Any]:
    scenarios = [
        summarize_scenario(artifact_root, scenario, expected_dp=expected_dp)
        for scenario, expected_dp in FORMAL_SCENARIOS.items()
    ]
    return {
        "schema": SCHEMA,
        "status": "DIAGNOSTIC_COMPLETE",
        "scenarios": scenarios,
        "route_selection_executed": False,
        "simulator_rank_rows_generated": False,
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
        "default_readiness": "No-Go",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.artifact_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
