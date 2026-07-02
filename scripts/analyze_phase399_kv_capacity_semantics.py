#!/usr/bin/env python3
"""Phase399: forensic attribution for KV capacity semantics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aiconfigurator.sdk.backends.cb_simulator.datatypes import (  # noqa: E402
    CBSimConfig,
    Request,
    RequestState,
)
from aiconfigurator.sdk.backends.cb_simulator.scheduler import CBScheduler  # noqa: E402

SOURCE = "phase399_kv_capacity_semantics"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase399_kv_capacity_semantics.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase399_kv_capacity_semantics.md"
PHASE397K_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase397k_measured_0190"
PHASE397Z_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase397z_effective_batch_attribution.csv"
PHASE398_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase398_kv_capacity_fix.csv"
DEFAULT_READINESS = "No-Go"
VLLM_SOURCE_URL = (
    "https://raw.githubusercontent.com/vllm-project/vllm/v0.19.0/"
    "vllm/v1/core/kv_cache_utils.py"
)

CSV_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "tier",
    "tp",
    "dp",
    "ep",
    "isl",
    "osl",
    "batch_size",
    "max_num_batched_tokens",
    "world_size",
    "bench_request_rate_rps",
    "bench_mean_latency_ms",
    "little_law_in_system_global",
    "little_law_in_system_per_engine",
    "bench_output_tok_s_global",
    "bench_output_tok_s_gpu",
    "real_decode_iter_ms",
    "real_effective_decode_batch_per_engine",
    "real_effective_decode_batch_global",
    "kv_cache_tokens",
    "num_gpu_blocks",
    "block_size",
    "full_sequence_capacity_per_engine",
    "full_sequence_capacity_global",
    "capacity_vs_little_decode_ratio",
    "phase398_validation_sim_output_tok_s_gpu",
    "phase398_validation_error_ratio",
    "phase398_validation_avg_decode_reqs_per_iter",
    "phase398_budget_sim_output_tok_s_gpu",
    "phase398_budget_error_ratio",
    "phase398_budget_avg_decode_reqs_per_iter",
    "serve_max_model_len",
    "serve_gpu_memory_utilization",
    "serve_max_num_seqs",
    "serve_max_num_batched_tokens",
    "serve_enable_prefix_caching",
    "serve_prefix_hit_rate_max_pct",
    "serve_running_reqs_max",
    "override_num_gpu_blocks",
    "serve_log_line_numbers",
    "override_line_numbers",
    "max_concurrency_for_logged_model_len",
    "preempt_probe_running",
    "preempt_probe_preemptions",
    "preempt_probe_recompute_tokens",
    "config_cleanliness",
    "classification",
    "clean_gate_candidate",
    "source_line_range",
    "source_url",
    "source_summary",
    "next_gate_recommendation",
    "gpu_allowed",
    "ssh_allowed",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
    "default_readiness",
]


def _fmt(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        if math.isnan(value):
            return ""
        return f"{value:.6f}"
    if isinstance(value, (tuple, list)):
        return ";".join(str(item) for item in value)
    return str(value)


def _base_row(row_type: str) -> dict[str, object]:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "source": SOURCE,
            "row_type": row_type,
            "gpu_allowed": False,
            "ssh_allowed": False,
            "diagnostic_only": True,
            "valid_for_default": False,
            "perf_database": False,
            "default_readiness": DEFAULT_READINESS,
        }
    )
    return row


def _read_csv_by_scenario(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        return {
            row["scenario"]: row
            for row in csv.DictReader(f)
            if row.get("scenario")
        }


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_first_int(text: str, pattern: str, default: int | None = None) -> int | None:
    match = re.search(pattern, text)
    if match is None:
        return default
    return int(match.group(1).replace(",", ""))


def _extract_first_float(text: str, pattern: str, default: float | None = None) -> float | None:
    match = re.search(pattern, text)
    if match is None:
        return default
    return float(match.group(1))


def _extract_first_bool(text: str, pattern: str, default: bool | None = None) -> bool | None:
    match = re.search(pattern, text)
    if match is None:
        return default
    return match.group(1) == "True"


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


def _unique_floats(lines: list[str], pattern: str) -> tuple[float, ...]:
    regex = re.compile(pattern)
    values: list[float] = []
    for line in lines:
        match = regex.search(line)
        if match:
            value = float(match.group(1))
            if value not in values:
                values.append(value)
    return tuple(values)


def _parse_serve_log(scenario_dir: Path) -> dict[str, object]:
    path = scenario_dir / "serve.log"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    joined = "\n".join(lines)
    non_default = next((line for line in lines if "non-default args:" in line), "")

    kv_tokens = _unique_ints(lines, r"GPU KV cache size: ([0-9,]+) tokens")
    override_values = _unique_ints(lines, r"num_gpu_blocks_override=(\d+)")
    max_concurrency = _unique_floats(
        lines,
        r"Maximum concurrency for [0-9,]+ tokens per request: ([0-9.]+)x",
    )
    prefix_rates = _unique_floats(lines, r"Prefix cache hit rate: ([0-9.]+)%")
    running_values = _unique_ints(lines, r"Running: (\d+) reqs")

    gpu_util = _extract_first_float(non_default, r"'gpu_memory_utilization': ([0-9.]+)", 0.8)
    max_num_seqs = _extract_first_int(non_default, r"'max_num_seqs': (\d+)", 256)
    max_bt = _extract_first_int(non_default, r"'max_num_batched_tokens': (\d+)", None)
    max_model_len = _extract_first_int(non_default, r"'max_model_len': (\d+)", None)
    if max_model_len is None:
        max_model_len = _extract_first_int(joined, r"max_seq_len=(\d+)", None)
    enable_prefix = _extract_first_bool(
        non_default,
        r"'enable_prefix_caching': (True|False)",
        None,
    )

    return {
        "kv_cache_tokens": kv_tokens[0] if kv_tokens else 0,
        "kv_cache_token_values": kv_tokens,
        "serve_log_line_numbers": _line_numbers(lines, r"GPU KV cache size:"),
        "override_num_gpu_blocks": override_values[0] if override_values else 0,
        "override_line_numbers": _line_numbers(lines, r"num_gpu_blocks_override="),
        "max_concurrency_for_logged_model_len": max_concurrency[0] if max_concurrency else 0.0,
        "serve_max_model_len": max_model_len or 0,
        "serve_gpu_memory_utilization": gpu_util or 0.0,
        "serve_max_num_seqs": max_num_seqs or 0,
        "serve_max_num_batched_tokens": max_bt or 0,
        "serve_enable_prefix_caching": bool(enable_prefix),
        "serve_prefix_hit_rate_max_pct": max(prefix_rates) if prefix_rates else 0.0,
        "serve_running_reqs_max": max(running_values) if running_values else 0,
    }


def _phase397z_anchor_rows() -> dict[str, dict[str, str]]:
    rows = _read_csv_by_scenario(PHASE397Z_CSV)
    missing = {
        "K2.5-tp8ep8-8k2k",
        "K2.5-tp4ep8dp2-8k2k",
    } - set(rows)
    if missing:
        raise ValueError(f"missing Phase397z anchor rows: {sorted(missing)}")
    return rows


def _topology_anchor_ms(scenario: str, phase397z: dict[str, dict[str, str]]) -> float:
    if "tp4ep8dp2" in scenario:
        return float(phase397z["K2.5-tp4ep8dp2-8k2k"]["real_decode_iter_ms"])
    return float(phase397z["K2.5-tp8ep8-8k2k"]["real_decode_iter_ms"])


def _clean_config(meta: dict[str, object], serve: dict[str, object]) -> bool:
    return (
        int(serve["serve_max_model_len"]) == 262_144
        and float(serve["serve_gpu_memory_utilization"]) == 0.8
        and int(serve["serve_max_num_seqs"]) == 256
        and int(serve["override_num_gpu_blocks"]) == 512
        and int(serve["serve_max_num_batched_tokens"]) == int(meta["max_num_batched_tokens"])
    )


def _preempt_probe(
    *,
    isl: int,
    osl: int,
    running_count: int,
    num_gpu_blocks: int,
    block_size: int,
) -> dict[str, int]:
    running: list[Request] = []
    for request_id in range(max(running_count, 0)):
        request = Request(request_id=request_id, isl=isl, osl=osl, arrival_time_ms=0.0)
        request.state = RequestState.DECODING
        request.prefill_tokens_remaining = 0
        running.append(request)

    scheduler = CBScheduler(
        CBSimConfig(
            max_num_batched_tokens=max(running_count, 1),
            num_gpu_blocks=num_gpu_blocks,
            block_size=block_size,
        )
    )
    stats = {"preemptions": 0, "recompute_tokens": 0}
    original_preempt = scheduler._preempt

    def wrapped_preempt(victim, waiting, running_reqs, result, preempted_ids):
        stats["preemptions"] += 1
        stats["recompute_tokens"] += victim.isl + victim.generated_tokens
        return original_preempt(victim, waiting, running_reqs, result, preempted_ids)

    scheduler._preempt = wrapped_preempt  # type: ignore[method-assign]
    scheduler.schedule(waiting=[], running=running)
    return stats


def _classify(
    *,
    scenario: str,
    clean: bool,
    capacity_ratio: float,
    phase398_error: float,
    phase398_sim: float,
    bench_output_gpu: float,
) -> str:
    if "bt65536" in scenario:
        return "dirty_bt65536_capacity_conflict"
    if not clean:
        return "config_drift"
    if capacity_ratio < 0.8 and phase398_error > 5.0:
        return "preempt_thrash_artifact"
    if phase398_sim > bench_output_gpu * 1.5:
        return "capacity_not_binding_prefill_or_scheduler_gap"
    return "kv_capacity_semantics_incomplete"


def _scenario_rows() -> list[dict[str, object]]:
    phase397z = _phase397z_anchor_rows()
    phase398 = _read_csv_by_scenario(PHASE398_CSV)
    rows: list[dict[str, object]] = []
    for scenario_dir in sorted(PHASE397K_ROOT.iterdir()):
        if not scenario_dir.is_dir():
            continue
        bench_path = scenario_dir / "bench_result.json"
        meta_path = scenario_dir / "meta.json"
        serve_path = scenario_dir / "serve.log"
        if not (bench_path.exists() and meta_path.exists() and serve_path.exists()):
            continue

        meta = _read_json(meta_path)
        bench = _read_json(bench_path)
        serve = _parse_serve_log(scenario_dir)
        scenario = str(meta["name"])
        p398 = phase398[scenario]

        dp = int(meta["dp"])
        tp = int(meta["tp"])
        world_size = int(meta["world_size"])
        request_rate = float(bench["request_rate_rps"])
        mean_latency_ms = float(bench["mean_latency_ms"])
        in_system_global = request_rate * mean_latency_ms / 1000.0
        in_system_per_engine = in_system_global / max(dp, 1)
        output_tok_s_global = float(bench["output_tok_s"])
        output_tok_s_gpu = output_tok_s_global / max(world_size, 1)
        real_decode_iter_ms = _topology_anchor_ms(scenario, phase397z)
        real_decode_per_engine = output_tok_s_gpu * tp * real_decode_iter_ms / 1000.0
        real_decode_global = real_decode_per_engine * dp

        kv_tokens = int(serve["kv_cache_tokens"])
        block_size = int(p398["block_size"])
        num_gpu_blocks = int(p398["num_gpu_blocks"])
        full_sequence_blocks = math.ceil((int(meta["isl"]) + int(meta["osl"])) / block_size)
        full_capacity_per_engine = num_gpu_blocks / full_sequence_blocks
        full_capacity_global = full_capacity_per_engine * dp
        capacity_ratio = full_capacity_global / real_decode_global if real_decode_global > 0 else 0.0
        phase398_sim = float(p398["phase398_validation_sim_output_tok_s_gpu"])
        phase398_error = float(p398["phase398_validation_error_ratio"])
        clean = _clean_config(meta, serve) and "bt65536" not in scenario
        probe_running = max(1, round(real_decode_per_engine))
        probe = _preempt_probe(
            isl=int(meta["isl"]),
            osl=int(meta["osl"]),
            running_count=probe_running,
            num_gpu_blocks=num_gpu_blocks,
            block_size=block_size,
        )

        row = _base_row("scenario")
        row.update(
            {
                "scenario": scenario,
                "tier": "clean_gate_candidate" if clean else "reference_only",
                "tp": tp,
                "dp": dp,
                "ep": int(meta["ep"]),
                "isl": int(meta["isl"]),
                "osl": int(meta["osl"]),
                "batch_size": int(meta["batch_size"]),
                "max_num_batched_tokens": int(meta["max_num_batched_tokens"]),
                "world_size": world_size,
                "bench_request_rate_rps": request_rate,
                "bench_mean_latency_ms": mean_latency_ms,
                "little_law_in_system_global": in_system_global,
                "little_law_in_system_per_engine": in_system_per_engine,
                "bench_output_tok_s_global": output_tok_s_global,
                "bench_output_tok_s_gpu": output_tok_s_gpu,
                "real_decode_iter_ms": real_decode_iter_ms,
                "real_effective_decode_batch_per_engine": real_decode_per_engine,
                "real_effective_decode_batch_global": real_decode_global,
                "kv_cache_tokens": kv_tokens,
                "num_gpu_blocks": num_gpu_blocks,
                "block_size": block_size,
                "full_sequence_capacity_per_engine": full_capacity_per_engine,
                "full_sequence_capacity_global": full_capacity_global,
                "capacity_vs_little_decode_ratio": capacity_ratio,
                "phase398_validation_sim_output_tok_s_gpu": phase398_sim,
                "phase398_validation_error_ratio": phase398_error,
                "phase398_validation_avg_decode_reqs_per_iter": float(
                    p398["phase398_validation_avg_decode_reqs_per_iter"]
                ),
                "phase398_budget_sim_output_tok_s_gpu": float(
                    p398["phase398_budget_sim_output_tok_s_gpu"]
                ),
                "phase398_budget_error_ratio": (
                    float(p398["phase398_budget_error_ratio"])
                    if p398["phase398_budget_error_ratio"] != "inf"
                    else float("inf")
                ),
                "phase398_budget_avg_decode_reqs_per_iter": float(
                    p398["phase398_budget_avg_decode_reqs_per_iter"]
                ),
                "serve_max_model_len": int(serve["serve_max_model_len"]),
                "serve_gpu_memory_utilization": float(serve["serve_gpu_memory_utilization"]),
                "serve_max_num_seqs": int(serve["serve_max_num_seqs"]),
                "serve_max_num_batched_tokens": int(serve["serve_max_num_batched_tokens"]),
                "serve_enable_prefix_caching": bool(serve["serve_enable_prefix_caching"]),
                "serve_prefix_hit_rate_max_pct": float(serve["serve_prefix_hit_rate_max_pct"]),
                "serve_running_reqs_max": int(serve["serve_running_reqs_max"]),
                "override_num_gpu_blocks": int(serve["override_num_gpu_blocks"]),
                "serve_log_line_numbers": serve["serve_log_line_numbers"],
                "override_line_numbers": serve["override_line_numbers"],
                "max_concurrency_for_logged_model_len": float(
                    serve["max_concurrency_for_logged_model_len"]
                ),
                "preempt_probe_running": probe_running,
                "preempt_probe_preemptions": probe["preemptions"],
                "preempt_probe_recompute_tokens": probe["recompute_tokens"],
                "config_cleanliness": "clean" if clean else "reference_dirty",
                "classification": _classify(
                    scenario=scenario,
                    clean=clean,
                    capacity_ratio=capacity_ratio,
                    phase398_error=phase398_error,
                    phase398_sim=phase398_sim,
                    bench_output_gpu=output_tok_s_gpu,
                ),
                "clean_gate_candidate": clean,
                "next_gate_recommendation": (
                    "keep_in_four_clean_gate"
                    if clean
                    else "demote_to_reference_until_recollected"
                ),
            }
        )
        rows.append(row)
    return rows


def _source_fact_rows() -> list[dict[str, object]]:
    override = _base_row("vllm_source_override")
    override.update(
        {
            "classification": "override_replaces_num_blocks",
            "source_line_range": "823-836",
            "source_url": VLLM_SOURCE_URL,
            "source_summary": (
                "may_override_num_blocks replaces computed num_blocks when "
                "num_gpu_blocks_override is set"
            ),
            "next_gate_recommendation": "model override and logged capacity together",
        }
    )
    gpu_size = _base_row("vllm_source_gpu_kv_size")
    gpu_size.update(
        {
            "classification": "logged_token_capacity_after_override",
            "source_line_range": "1303-1319",
            "source_url": VLLM_SOURCE_URL,
            "source_summary": (
                "GPU KV cache size logs kv_cache_config.num_blocks divided by groups "
                "times minimum block size"
            ),
            "next_gate_recommendation": "do not treat token capacity as per-request ownership under prefix sharing",
        }
    )
    return [override, gpu_size]


def build_phase399_rows() -> list[dict[str, object]]:
    return _source_fact_rows() + _scenario_rows()


def _validate_row(row: dict[str, object]) -> None:
    if row.get("valid_for_default") is not False:
        raise ValueError("valid_for_default must stay false")
    if row.get("perf_database") is not False:
        raise ValueError("perf_database must stay false")
    if row.get("default_readiness") != DEFAULT_READINESS:
        raise ValueError("default_readiness must stay No-Go")


def write_phase399_csv(path: Path, rows: list[dict[str, object]]) -> None:
    for row in rows:
        _validate_row(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in CSV_FIELDS})


def render_phase399_md(rows: list[dict[str, object]]) -> str:
    scenarios = [row for row in rows if row["row_type"] == "scenario"]
    clean = [row for row in scenarios if row["config_cleanliness"] == "clean"]
    dirty = [row for row in scenarios if row["config_cleanliness"] != "clean"]
    worst = max(
        scenarios,
        key=lambda row: float(row["phase398_validation_error_ratio"]),
    )
    lines = [
        "# Phase399 KV Capacity Semantics",
        "",
        "## Verdict",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Scope | Offline forensics only |",
        "| GPU/SSH | Not used |",
        "| vLLM source | v0.19.0 `kv_cache_utils.py` official tag |",
        "| Main finding | `GPU KV cache size` is real logged token capacity after override, but cb_sim cannot use it as full per-request KV ownership under prefix sharing |",
        "| Phase398 failure | Naive capacity binding caused severe under-prediction, worst "
        f"{float(worst['phase398_validation_error_ratio']):.2f}x on {worst['scenario']} |",
        "| Gate recommendation | gate shrink to four clean scenarios; demote bt65536 to reference until recollected |",
        "| Default | Default AIC remains No-Go |",
        "",
        "## Source Semantics",
        "",
        "| Fact | Lines | Meaning |",
        "|---|---:|---|",
        "| override | 823-836 | `num_gpu_blocks_override` replaces computed `num_blocks` |",
        "| GPU KV size | 1303-1319 | logged token capacity is derived from `kv_cache_config.num_blocks`, group count, and block size |",
        "",
        "## Scenario Classification",
        "",
        "| Scenario | Little in-system | Decode batch global | Full-seq capacity global | Phase398 error | Class |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in scenarios:
        lines.append(
            "| {name} | {little:.2f} | {decode:.2f} | {capacity:.2f} | {err:.2f}x | {cls} |".format(
                name=row["scenario"],
                little=float(row["little_law_in_system_global"]),
                decode=float(row["real_effective_decode_batch_global"]),
                capacity=float(row["full_sequence_capacity_global"]),
                err=float(row["phase398_validation_error_ratio"]),
                cls=row["classification"],
            )
        )
    lines.extend(
        [
            "",
            "## Clean Gate",
            "",
            f"Clean candidates: {', '.join(str(row['scenario']) for row in clean)}.",
            f"Reference-only: {', '.join(str(row['scenario']) for row in dirty)}.",
            "",
            "The two bt65536 rows are not acceptable gate rows here: one has a different max model length / memory utilization / max seqs, and both are retry-path budget variants. They can stay as diagnostic references only.",
            "",
            "## Next",
            "",
            "Phase400 should not promote the Phase398 wiring directly. It should first model prefix-shared KV ownership and vLLM preemption/recompute behavior, then re-run only the four clean scenarios.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_phase399_md(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase399_md(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_phase399_rows()
    write_phase399_csv(args.csv, rows)
    write_phase399_md(args.md, rows)


if __name__ == "__main__":
    main()
