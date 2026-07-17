#!/usr/bin/env python3
"""Export Phase466 CB-simulator iterations with the real-probe alignment contract."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable


PROGRESS_WINDOW_TOKENS = 65536
SIMULATOR_BACKEND = "vllm"
SIMULATOR_SYSTEM = "h200_sxm"
SIMULATOR_DB_VERSION = "0.19.0"


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot_load_script:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure_diagnose(diagnose: Any) -> None:
    if diagnose.BACKEND != SIMULATOR_BACKEND:
        raise ValueError(f"unexpected_simulator_backend:{diagnose.BACKEND}")
    if diagnose.SYSTEM != SIMULATOR_SYSTEM:
        raise ValueError(f"unexpected_simulator_system:{diagnose.SYSTEM}")
    diagnose.DB_VERSION = SIMULATOR_DB_VERSION


def validate_scenario_exportability(run: dict[str, Any]) -> None:
    if int(run["dp"]) > 1 and int(run["max_num_batched_tokens"]) != int(
        run["input_len"]
    ):
        raise ValueError(
            "unsupported_dp_trace_backend_legacy_single_replica:"
            f"{run['scenario']}"
        )


def normalize_rows(
    rows: Iterable[Any],
    *,
    run_id: str,
    workload_cohort_digest: str,
    expected_num_prompts: int,
) -> list[dict[str, Any]]:
    if not run_id:
        raise ValueError("missing_simulator_run_id")
    if not re.fullmatch(r"[0-9a-f]{64}", workload_cohort_digest):
        raise ValueError("invalid_workload_cohort_digest")
    normalized: list[dict[str, Any]] = []
    cumulative_tokens = 0
    previous_end_ms = 0.0
    for row in rows:
        iteration = int(row.iter_index)
        elapsed_ms = float(row.iter_lat_ms)
        end_ms = float(row.clock_ms)
        start_ms = end_ms - elapsed_ms
        if (
            not math.isfinite(elapsed_ms)
            or elapsed_ms <= 0
            or not math.isclose(start_ms, previous_end_ms, abs_tol=1e-6)
        ):
            raise ValueError(f"simulator_clock_mismatch:{iteration}")
        prefill_tokens = int(row.prefill_tokens)
        decode_tokens = int(row.decode_batch_size)
        scheduled_tokens = prefill_tokens + decode_tokens
        if scheduled_tokens <= 0:
            raise ValueError(f"simulator_empty_iteration:{iteration}")
        progress_start = cumulative_tokens
        cumulative_tokens += scheduled_tokens
        normalized.append(
            {
                "run_id": run_id,
                "source": "sim",
                "rank_id": 0,
                "rank_scope": "dp_rank",
                "workload_cohort_digest": workload_cohort_digest,
                "iteration_seq": iteration,
                "progress_start_tokens": progress_start,
                "progress_end_tokens": cumulative_tokens,
                "prefill_request_count": int(row.prefill_requests),
                "scheduled_prefill_tokens": prefill_tokens,
                "decode_request_count": int(row.decode_batch_size),
                "scheduled_decode_tokens": decode_tokens,
                "iteration_elapsed_ms": elapsed_ms,
                "iteration_start_offset_ms": start_ms,
                "iteration_end_offset_ms": end_ms,
                "cumulative_scheduled_tokens": cumulative_tokens,
                "progress_window_id": max(
                    0, (cumulative_tokens - 1) // PROGRESS_WINDOW_TOKENS
                ),
                "running_count": int(row.running_requests),
                "waiting_count": int(row.waiting_requests),
                "completed_request_count": int(row.completed_requests),
                "sim_predicted_iteration_ms": elapsed_ms,
                "sim_component_cost_ms": elapsed_ms,
                "sim_context_non_attention_ms": float(row.ctx_non_attn_ms),
                "sim_context_attention_ms": float(row.ctx_attn_ms),
                "sim_generation_non_attention_ms": float(row.gen_non_attn_ms),
                "sim_generation_attention_ms": float(row.gen_attn_ms),
            }
        )
        previous_end_ms = end_ms
    if not normalized:
        raise ValueError("missing_simulator_iteration_rows")
    if int(normalized[-1]["completed_request_count"]) != expected_num_prompts:
        raise ValueError(
            "simulator_incomplete_requests:"
            f"{normalized[-1]['completed_request_count']}!={expected_num_prompts}"
        )
    return normalized


def normalize_multi_replica_trace(
    trace: Iterable[dict[str, Any]],
    *,
    run_id: str,
    workload_cohort_digest: str,
    expected_dp: int,
) -> list[dict[str, Any]]:
    if not run_id:
        raise ValueError("missing_simulator_run_id")
    if not re.fullmatch(r"[0-9a-f]{64}", workload_cohort_digest):
        raise ValueError("invalid_workload_cohort_digest")
    cumulative = {rank: 0 for rank in range(expected_dp)}
    previous_end = {rank: 0.0 for rank in range(expected_dp)}
    previous_iter = {rank: 0 for rank in range(expected_dp)}
    normalized: list[dict[str, Any]] = []
    for raw in trace:
        rank = int(raw["replica_id"])
        if rank not in cumulative:
            raise ValueError(f"simulator_rank_out_of_range:{rank}")
        iteration = int(raw["local_iter"])
        if iteration != previous_iter[rank] + 1:
            raise ValueError(f"simulator_iteration_gap:{rank}:{iteration}")
        start_ms = float(raw["start_ms"])
        end_ms = float(raw["end_ms"])
        elapsed_ms = end_ms - start_ms
        if (
            not math.isfinite(elapsed_ms)
            or elapsed_ms <= 0
            or not math.isclose(start_ms, previous_end[rank], abs_tol=1e-6)
        ):
            raise ValueError(f"simulator_clock_mismatch:{rank}:{iteration}")
        prefill_tokens = int(raw["prefill_tokens"])
        decode_tokens = int(raw["total_tokens"]) - prefill_tokens
        if decode_tokens != int(raw["decode_reqs"]):
            raise ValueError(f"simulator_decode_token_mismatch:{rank}:{iteration}")
        scheduled_tokens = prefill_tokens + decode_tokens
        if scheduled_tokens <= 0:
            raise ValueError(f"simulator_empty_iteration:{rank}:{iteration}")
        progress_start = cumulative[rank]
        cumulative[rank] += scheduled_tokens
        normalized.append(
            {
                "run_id": run_id,
                "source": "sim",
                "rank_id": rank,
                "rank_scope": "dp_rank",
                "workload_cohort_digest": workload_cohort_digest,
                "iteration_seq": iteration,
                "progress_start_tokens": progress_start,
                "progress_end_tokens": cumulative[rank],
                "prefill_request_count": int(raw["prefill_reqs"]),
                "scheduled_prefill_tokens": prefill_tokens,
                "decode_request_count": int(raw["decode_reqs"]),
                "scheduled_decode_tokens": decode_tokens,
                "iteration_elapsed_ms": elapsed_ms,
                "iteration_start_offset_ms": start_ms,
                "iteration_end_offset_ms": end_ms,
                "cumulative_scheduled_tokens": cumulative[rank],
                "progress_window_id": max(
                    0, (cumulative[rank] - 1) // PROGRESS_WINDOW_TOKENS
                ),
                "sim_predicted_iteration_ms": elapsed_ms,
                "sim_component_cost_ms": elapsed_ms,
            }
        )
        previous_iter[rank] = iteration
        previous_end[rank] = end_ms
    if not normalized:
        raise ValueError("missing_simulator_iteration_rows")
    actual_ranks = sorted({int(row["rank_id"]) for row in normalized})
    if actual_ranks != list(range(expected_dp)):
        raise ValueError(f"simulator_rank_set_mismatch:{actual_ranks}")
    return normalized


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _collect_scenario(contract: Any, diagnose: Any, run: dict[str, Any]) -> list[dict[str, Any]]:
    parser = diagnose._build_parser()
    args = parser.parse_args(
        [
            "--isl",
            str(run["input_len"]),
            "--osl",
            str(run["output_len"]),
            "--concurrency",
            str(run["concurrency"]),
            "--tp",
            str(run["tp"]),
            "--dp",
            str(run["dp"]),
            "--moe-tp",
            "1",
            "--moe-ep",
            str(run["ep"]),
            "--max-num-batched-tokens",
            str(run["max_num_batched_tokens"]),
            "--max-num-seqs",
            "256",
            "--num-requests",
            str(run["num_prompts"]),
            "--warmup-requests",
            "128",
        ]
    )
    digest = contract.workload_digest(
        contract.SCENARIOS[run["scenario"]],
        num_prompts=run["num_prompts"],
        concurrency=run["concurrency"],
    )
    if int(run["dp"]) == 1:
        rows = diagnose.collect_cb_iteration_trace(args)
        return normalize_rows(
            rows,
            run_id=f"sim-{run['scenario']}",
            workload_cohort_digest=digest,
            expected_num_prompts=int(run["num_prompts"]),
        )

    model, database, backend = diagnose._load_model_and_db(
        tp=args.tp,
        dp=args.dp,
        moe_tp=args.moe_tp,
        moe_ep=args.moe_ep,
    )
    config = diagnose._make_cb_config(args)
    simulator = diagnose.CBSimulator(backend, model, database, config)
    result = simulator.run_multi_replica(
        isl=args.isl,
        osl=args.osl,
        concurrency=args.concurrency,
        data_parallel_size=args.dp,
        prefix=args.prefix,
        num_gpus=args.tp * args.dp,
        lockstep=True,
    )
    expected_steady = config.num_requests - config.warmup_requests
    if result.steady_state_requests != expected_steady:
        raise ValueError(
            "simulator_incomplete_requests:"
            f"{result.steady_state_requests}!={expected_steady}"
        )
    return normalize_multi_replica_trace(
        simulator.get_last_schedule_trace(),
        run_id=f"sim-{run['scenario']}",
        workload_cohort_digest=digest,
        expected_dp=args.dp,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    contract = _load_script(
        "phase466_probe_contract",
        root / "scripts" / "analyze_phase466_low_overhead_probe.py",
    )
    diagnose = _load_script(
        "phase466_cb_diagnose",
        root / "scripts" / "diagnose_cb_iter_latency.py",
    )
    configure_diagnose(diagnose)
    plan = contract.build_run_plan()
    formal = [run for run in plan["runs"] if run["stage"] == "formal"]
    for run in formal:
        validate_scenario_exportability(run)
    manifest: list[dict[str, Any]] = []
    for run in formal:
        rows = _collect_scenario(contract, diagnose, run)
        output = args.output_dir / run["scenario"] / "iteration_rows.csv"
        _write_csv(output, rows)
        manifest.append(
            {
                "scenario": run["scenario"],
                "run_id": rows[0]["run_id"],
                "workload_cohort_digest": rows[0]["workload_cohort_digest"],
                "row_count": len(rows),
                "path": str(output),
                "iteration_rows_identity": contract.iteration_csv_identity(output),
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": contract.SCHEMA,
                "source": "sim",
                "rank_scope": "dp_rank",
                "backend": SIMULATOR_BACKEND,
                "system": SIMULATOR_SYSTEM,
                "database_version": SIMULATOR_DB_VERSION,
                "scenarios": manifest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
