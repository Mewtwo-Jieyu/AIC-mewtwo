#!/usr/bin/env python3
"""Phase400: summarize prefix-off clean recollect evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

SOURCE = "phase400_clean_recollect"
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase400_clean_recollect"
PHASE398_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase398_kv_capacity_fix.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase400_clean_recollect.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase400_clean_recollect.md"
DEFAULT_READINESS = "No-Go"

SCENARIOS = (
    "K2.5-tp8ep8-8k2k",
    "K2.5-tp8ep8-32k3k",
    "K2.5-tp4ep8dp2-8k2k",
    "K2.5-tp4ep8dp2-32k3k",
)

CSV_FIELDS = [
    "source",
    "scenario",
    "tp",
    "dp",
    "ep",
    "isl",
    "osl",
    "max_num_batched_tokens",
    "max_model_len",
    "max_num_seqs",
    "world_size",
    "ok_requests",
    "failed_requests",
    "use_prompt_token_ids",
    "wall_s",
    "request_rate_rps",
    "output_tok_s_global",
    "output_tok_s_gpu",
    "mean_latency_ms",
    "p50_latency_ms",
    "p99_latency_ms",
    "serve_enable_prefix_caching",
    "serve_prefix_hit_rate_max_pct",
    "serve_running_reqs_max",
    "serve_running_reqs_mean",
    "serve_running_reqs_samples",
    "serve_gpu_kv_usage_max_pct",
    "stats_status",
    "gpu_kv_cache_tokens",
    "num_gpu_blocks",
    "block_size",
    "override_num_gpu_blocks",
    "override_line_count",
    "kv_cache_line_numbers",
    "override_line_numbers",
    "phase397z_sim_output_tok_s_gpu",
    "phase397z_vs_phase400_error_ratio",
    "phase397z_vs_phase400_direction",
    "phase398_capacity_sim_output_tok_s_gpu",
    "phase398_capacity_vs_phase400_error_ratio",
    "phase398_capacity_vs_phase400_direction",
    "gpu_compute_apps_before_empty",
    "gpu_compute_apps_after_empty",
    "process_residual_after_empty",
    "collection_status",
    "classification",
    "next_phase_target",
    "gpu_allowed",
    "ssh_allowed",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
    "default_readiness",
]


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6f}"
    if isinstance(value, (tuple, list)):
        return ";".join(str(item) for item in value)
    return str(value)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_phase398_rows(path: Path = PHASE398_CSV) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        return {row["scenario"]: row for row in csv.DictReader(f)}


def _line_numbers(lines: list[str], pattern: str) -> tuple[int, ...]:
    regex = re.compile(pattern)
    return tuple(idx for idx, line in enumerate(lines, start=1) if regex.search(line))


def _unique_ints(lines: list[str], pattern: str) -> tuple[int, ...]:
    regex = re.compile(pattern)
    values: list[int] = []
    for line in lines:
        match = regex.search(line)
        if match:
            value = int(match.group(1).replace(",", ""))
            if value not in values:
                values.append(value)
    return tuple(values)


def _parse_stats(lines: list[str]) -> dict[str, object]:
    stats_re = re.compile(
        r"Engine (?P<engine>\d+): Avg prompt throughput: (?P<prompt>[0-9.]+) tokens/s, "
        r"Avg generation throughput: (?P<generation>[0-9.]+) tokens/s, "
        r"Running: (?P<running>\d+) reqs, Waiting: (?P<waiting>\d+) reqs, "
        r"GPU KV cache usage: (?P<kv_usage>[0-9.]+)%, "
        r"Prefix cache hit rate: (?P<prefix>[0-9.]+)%"
    )
    running: list[int] = []
    prefix_rates: list[float] = []
    kv_usage: list[float] = []
    for line in lines:
        match = stats_re.search(line)
        if not match:
            continue
        running.append(int(match.group("running")))
        prefix_rates.append(float(match.group("prefix")))
        kv_usage.append(float(match.group("kv_usage")))
    return {
        "running_max": max(running) if running else None,
        "running_mean": sum(running) / len(running) if running else None,
        "running_samples": len(running),
        "prefix_max": max(prefix_rates) if prefix_rates else None,
        "kv_usage_max": max(kv_usage) if kv_usage else None,
    }


def _parse_prefix_flag(lines: list[str]) -> bool:
    joined = "\n".join(lines)
    if "'enable_prefix_caching': False" in joined or "enable_prefix_caching=False" in joined:
        return False
    if "'enable_prefix_caching': True" in joined or "enable_prefix_caching=True" in joined:
        return True
    raise ValueError("serve log is missing enable_prefix_caching")


def _ratio_and_direction(sim_value: float, real_value: float) -> tuple[float, str]:
    if sim_value <= 0 or real_value <= 0:
        return math.inf, "invalid"
    ratio = max(sim_value, real_value) / min(sim_value, real_value)
    direction = "sim_over_predicts_throughput" if sim_value > real_value else "sim_under_predicts_throughput"
    return ratio, direction


def _base_row() -> dict[str, object]:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "source": SOURCE,
            "gpu_allowed": True,
            "ssh_allowed": True,
            "diagnostic_only": True,
            "valid_for_default": False,
            "perf_database": False,
            "default_readiness": DEFAULT_READINESS,
            "next_phase_target": "phase401_capacity_and_preemption_fix",
        }
    )
    return row


def _classify(scenario: str, stats_status: str, phase398_direction: str) -> str:
    if stats_status == "missing_dp_stats":
        return "clean_prefix_off_dp_throughput_collected_but_running_stats_missing"
    if phase398_direction == "sim_under_predicts_throughput":
        return "clean_prefix_off_capacity_binding_observed_naive_capacity_over_restricts"
    return "clean_prefix_off_capacity_binding_observed_sim_still_over_predicts"


def build_phase400_rows(
    raw_root: Path = RAW_ROOT,
    phase398_csv: Path = PHASE398_CSV,
) -> list[dict[str, str]]:
    phase398 = _read_phase398_rows(phase398_csv)
    gpu_before_empty = (raw_root / "gpu_compute_apps_before.txt").read_text(encoding="utf-8") == ""
    gpu_after_empty = (raw_root / "gpu_compute_apps_after.txt").read_text(encoding="utf-8") == ""
    process_after_empty = (raw_root / "process_residual_after.txt").read_text(encoding="utf-8") == ""

    rows: list[dict[str, str]] = []
    for scenario in SCENARIOS:
        scenario_dir = raw_root / scenario
        meta = _read_json(scenario_dir / "meta.json")
        bench = _read_json(scenario_dir / "bench_result.json")
        lines = (scenario_dir / "serve.log").read_text(encoding="utf-8", errors="replace").splitlines()
        stats = _parse_stats(lines)

        kv_tokens = _unique_ints(lines, r"GPU KV cache size: ([0-9,]+) tokens")
        overrides = _unique_ints(lines, r"num_gpu_blocks_override=(\d+)")
        if len(kv_tokens) != 1:
            raise ValueError(f"{scenario}: expected one unique GPU KV cache size")
        if len(overrides) != 1:
            raise ValueError(f"{scenario}: expected one unique override observation")

        world_size = int(meta["world_size"])
        real_output_gpu = float(bench["output_tok_s"]) / world_size
        phase398_row = phase398[scenario]
        phase397z_sim = float(phase398_row["phase397z_sim_output_tok_s_gpu"])
        phase398_sim = float(phase398_row["phase398_validation_sim_output_tok_s_gpu"])
        phase397z_ratio, phase397z_direction = _ratio_and_direction(phase397z_sim, real_output_gpu)
        phase398_ratio, phase398_direction = _ratio_and_direction(phase398_sim, real_output_gpu)

        dp = int(meta["dp"])
        stats_status = "stats_present" if stats["running_samples"] else "missing_dp_stats"
        if dp == 1 and stats_status != "stats_present":
            raise ValueError(f"{scenario}: TP8 scenario must contain server stats")
        if dp > 1 and stats_status != "missing_dp_stats":
            raise ValueError(f"{scenario}: DP scenario stats semantics changed; audit parser")

        row = _base_row()
        row.update(
            {
                "scenario": scenario,
                "tp": int(meta["tp"]),
                "dp": dp,
                "ep": int(meta["ep"]),
                "isl": int(meta["isl"]),
                "osl": int(meta["osl"]),
                "max_num_batched_tokens": int(meta["max_num_batched_tokens"]),
                "max_model_len": int(meta["max_model_len"]),
                "max_num_seqs": int(meta["max_num_seqs"]),
                "world_size": world_size,
                "ok_requests": int(bench["ok_requests"]),
                "failed_requests": int(bench["failed_requests"]),
                "use_prompt_token_ids": bool(bench["use_prompt_token_ids"]),
                "wall_s": float(bench["wall_s"]),
                "request_rate_rps": float(bench["request_rate_rps"]),
                "output_tok_s_global": float(bench["output_tok_s"]),
                "output_tok_s_gpu": real_output_gpu,
                "mean_latency_ms": float(bench["mean_latency_ms"]),
                "p50_latency_ms": float(bench["p50_latency_ms"]),
                "p99_latency_ms": float(bench["p99_latency_ms"]),
                "serve_enable_prefix_caching": _parse_prefix_flag(lines),
                "serve_prefix_hit_rate_max_pct": stats["prefix_max"],
                "serve_running_reqs_max": stats["running_max"],
                "serve_running_reqs_mean": stats["running_mean"],
                "serve_running_reqs_samples": stats["running_samples"],
                "serve_gpu_kv_usage_max_pct": stats["kv_usage_max"],
                "stats_status": stats_status,
                "gpu_kv_cache_tokens": kv_tokens[0],
                "num_gpu_blocks": kv_tokens[0] // 16,
                "block_size": 16,
                "override_num_gpu_blocks": overrides[0],
                "override_line_count": len(_line_numbers(lines, r"num_gpu_blocks_override=")),
                "kv_cache_line_numbers": _line_numbers(lines, r"GPU KV cache size:"),
                "override_line_numbers": _line_numbers(lines, r"num_gpu_blocks_override="),
                "phase397z_sim_output_tok_s_gpu": phase397z_sim,
                "phase397z_vs_phase400_error_ratio": phase397z_ratio,
                "phase397z_vs_phase400_direction": phase397z_direction,
                "phase398_capacity_sim_output_tok_s_gpu": phase398_sim,
                "phase398_capacity_vs_phase400_error_ratio": phase398_ratio,
                "phase398_capacity_vs_phase400_direction": phase398_direction,
                "gpu_compute_apps_before_empty": gpu_before_empty,
                "gpu_compute_apps_after_empty": gpu_after_empty,
                "process_residual_after_empty": process_after_empty,
                "collection_status": "passed_prefix_off_clean_recollect",
                "classification": _classify(scenario, stats_status, phase398_direction),
            }
        )
        rows.append({field: _fmt(row[field]) for field in CSV_FIELDS})

    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 4:
        raise ValueError("Phase400 must contain exactly 4 scenario rows")
    if {row["scenario"] for row in rows} != set(SCENARIOS):
        raise ValueError("Phase400 scenario set mismatch")
    for row in rows:
        for field in ("diagnostic_only", "gpu_allowed", "ssh_allowed"):
            if row[field] != "true":
                raise ValueError(field)
        for field in ("valid_for_default", "perf_database"):
            if row[field] != "false":
                raise ValueError(field)
        if row["default_readiness"] != DEFAULT_READINESS:
            raise ValueError("default_readiness")
        if row["serve_enable_prefix_caching"] != "false":
            raise ValueError("serve_enable_prefix_caching")
        if row["ok_requests"] != "128" or row["failed_requests"] != "0":
            raise ValueError("benchmark requests")
        if row["gpu_compute_apps_after_empty"] != "true":
            raise ValueError("gpu_compute_apps_after_empty")
        if row["process_residual_after_empty"] != "true":
            raise ValueError("process_residual_after_empty")


def write_phase400_csv(path: Path = DEFAULT_CSV, rows: list[dict[str, str]] | None = None) -> None:
    rows = build_phase400_rows() if rows is None else rows
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_phase400_md(rows: list[dict[str, str]]) -> str:
    scenario_lines = []
    for row in rows:
        running = row["serve_running_reqs_max"] or "missing"
        scenario_lines.append(
            "| {scenario} | {output_tok_s_gpu} | {serve_running_reqs_max} | "
            "{gpu_kv_cache_tokens} | {serve_prefix_hit_rate_max_pct} | "
            "{phase397z_vs_phase400_error_ratio} | {phase398_capacity_vs_phase400_error_ratio} |".format(
                **{**row, "serve_running_reqs_max": running}
            )
        )

    return "\n".join(
        [
            "# Phase400 clean recollect",
            "",
            "Phase400 recollected four 0.19-real 8-card scenarios with prefix caching disabled. "
            "All four benchmark runs completed 128/128 requests and the cleanup evidence is empty.",
            "",
            "| scenario | clean output tok/s/gpu | running max | GPU KV tokens | prefix hit max % | no-capacity sim ratio | naive-capacity sim ratio |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *scenario_lines,
            "",
            "Key findings:",
            "",
            "- `prefix hit is 0.0%` for TP8 logs; DP2 logs do not emit periodic `Running:` stats in this collection, so their running batch is recorded as missing instead of inferred.",
            "- The command did not pass `--num-gpu-blocks-override`, but vLLM still logged worker-level `num_gpu_blocks_override=512` in every scenario. Phase401 must treat this as an observed vLLM semantic, not as a CLI knob used by the recollect.",
            "- Compared with the clean prefix-off real baseline, the no-capacity simulator path still over-predicts throughput, while the Phase398 naive capacity path under-predicts long-context throughput. Capacity alone is not the final fix; preemption and capacity semantics need a targeted Phase401 change.",
            "- No simulator runtime, PerfDatabase data, or acceptance gate changed in Phase400. Default AIC remains No-Go.",
            "",
            "Next: Phase401 should use these clean rows to fix capacity/preemption semantics, then rerun the 4-scenario gate.",
            "",
        ]
    )


def write_phase400_md(path: Path = DEFAULT_MD, rows: list[dict[str, str]] | None = None) -> None:
    rows = build_phase400_rows() if rows is None else rows
    path.write_text(render_phase400_md(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    rows = build_phase400_rows()
    write_phase400_csv(args.csv, rows)
    write_phase400_md(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
