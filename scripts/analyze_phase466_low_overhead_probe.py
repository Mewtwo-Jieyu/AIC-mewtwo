#!/usr/bin/env python3
"""Build and validate the Phase466 stock-vLLM aggregate probe plan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shlex
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


BASE_COMMIT = "822ae421a0b79eb0a69e8f59a2c4d09cba327eaa"
SCHEMA = "phase466_rank_timing_v4"
RANK_LOG_IDENTITY_SCHEMA = "phase466_rank_log_identity_v1"
PROBE_FLAG = "--enable-logging-iteration-details"
MAX_OVERHEAD_PCT = 2.0
OVERHEAD_PAIR_COUNT = 6
EQUIVALENCE_RATIO_BOUNDS = (0.98, 1.02)
T_CRITICAL_90_DF5 = 2.0150483733330233
PROGRESS_WINDOW_TOKENS = 65536
MODEL_PATH = (
    "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/"
    "zskj-hub/models--moonshotai--Kimi-K2.5"
)
SERVED_MODEL_NAME = "kimi-k2.5"
OVERHEAD_SCENARIO = "K2.5-tp4ep8dp2-8k2k-bt65536"
FORMAL_SCENARIOS = (
    "K2.5-tp4ep8dp2-32k3k",
    "K2.5-tp8ep8-8k2k-bt65536",
    "K2.5-tp4ep8dp2-8k2k-bt65536",
)
SOURCE_FILES = {
    "/usr/local/lib/python3.12/dist-packages/vllm/logger.py": (
        "233720900b1207434e4824bbddcf86dba0dc8557a0bfe78a639545462a9e85b5"
    ),
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py": (
        "896730e749cbcabb487ce50c703974594d197fc31a1d3b26fe096197d142d2d5"
    ),
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py": (
        "9f3da2dfce94963e1e1cefc156e4239120f79c6eaf824b2829cac0ee7736b58a"
    ),
}

ITERATION_RE = re.compile(
    r"^Iteration\((?P<iteration>\d+)\):\s+"
    r"(?P<context_requests>\d+)\s+context requests,\s+"
    r"(?P<context_tokens>\d+)\s+context tokens,\s+"
    r"(?P<generation_requests>\d+)\s+generation requests,\s+"
    r"(?P<generation_tokens>\d+)\s+generation tokens,\s+"
    r"iteration elapsed time:\s+(?P<elapsed_ms>[0-9.]+)\s+ms$"
)
PROM_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+0-9.eE]+)$"
)
LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')


@dataclass(frozen=True)
class Scenario:
    name: str
    tp: int
    dp: int
    ep: int
    input_len: int
    output_len: int
    max_num_batched_tokens: int
    max_model_len: int


@dataclass(frozen=True)
class IterationRow:
    rank_id: int
    rank_scope: str
    run_id: str
    source: str
    workload_cohort_digest: str
    iteration_seq: int
    progress_start_tokens: int
    progress_end_tokens: int
    prefill_request_count: int
    scheduled_prefill_tokens: int
    decode_request_count: int
    scheduled_decode_tokens: int
    iteration_elapsed_ms: float
    iteration_start_offset_ms: float
    iteration_end_offset_ms: float
    cumulative_scheduled_tokens: int
    progress_window_id: int


@dataclass(frozen=True)
class RankLogRecord:
    rank: int
    pid: int
    process_name: str
    match: re.Match[str]


SCENARIOS = {
    "K2.5-tp4ep8dp2-32k3k": Scenario(
        "K2.5-tp4ep8dp2-32k3k", 4, 2, 8, 32000, 3000, 32000, 262144
    ),
    "K2.5-tp8ep8-8k2k-bt65536": Scenario(
        "K2.5-tp8ep8-8k2k-bt65536", 8, 1, 8, 8000, 2000, 65536, 262144
    ),
    "K2.5-tp4ep8dp2-8k2k-bt65536": Scenario(
        "K2.5-tp4ep8dp2-8k2k-bt65536", 4, 2, 8, 8000, 2000, 65536, 131072
    ),
}


def workload_digest(
    scenario: Scenario, *, num_prompts: int, concurrency: int
) -> str:
    payload = {
        "model": SERVED_MODEL_NAME,
        "scenario": scenario.name,
        "num_prompts": num_prompts,
        "concurrency": concurrency,
        "input_len": scenario.input_len,
        "output_len": scenario.output_len,
        "tp": scenario.tp,
        "dp": scenario.dp,
        "ep": scenario.ep,
        "max_num_batched_tokens": scenario.max_num_batched_tokens,
        "max_model_len": scenario.max_model_len,
        "prompt_variant_mode": "fixed",
        "stream": True,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _serve_argv(scenario: Scenario, *, port: int, probe_enabled: bool) -> list[str]:
    argv = [
        "python3",
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        MODEL_PATH,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(scenario.tp),
        "--data-parallel-size",
        str(scenario.dp),
        "--enable-expert-parallel",
        "--max-model-len",
        str(scenario.max_model_len),
        "--gpu-memory-utilization",
        "0.80",
        "--max-num-batched-tokens",
        str(scenario.max_num_batched_tokens),
        "--max-num-seqs",
        "256",
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--cudagraph-metrics",
        "--mm-encoder-tp-mode",
        "data",
        "--skip-mm-profiling",
        "--trust-remote-code",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "kimi_k2",
        "--reasoning-parser",
        "kimi_k2",
    ]
    if probe_enabled:
        argv.append(PROBE_FLAG)
    return argv


def _benchmark_argv(
    scenario: Scenario,
    *,
    port: int,
    num_prompts: int,
    concurrency: int,
    artifact_dir: str,
) -> list[str]:
    return [
        "python3",
        "scripts/run_openai_fixed_shape_benchmark.py",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        SERVED_MODEL_NAME,
        "--tokenizer",
        MODEL_PATH,
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(concurrency),
        "--input-len",
        str(scenario.input_len),
        "--output-len",
        str(scenario.output_len),
        "--warmup-requests",
        "0",
        "--timeout-s",
        "21600",
        "--skip-prompt-token-id-probe",
        "--stream",
        "--result-json",
        f"{artifact_dir}/bench_result.json",
        "--records-jsonl",
        f"{artifact_dir}/bench_records.jsonl",
    ]


def _run_spec(
    *,
    run_id: str,
    stage: str,
    scenario: Scenario,
    probe_enabled: bool,
    num_prompts: int,
    port: int,
    pair_id: str | None = None,
    order_index: int | None = None,
) -> dict[str, Any]:
    if stage == "overhead":
        if pair_id is None or order_index is None:
            raise ValueError("overhead_run_requires_pair_identity")
        artifact_dir = f"overhead/{pair_id}/{'on' if probe_enabled else 'off'}"
    else:
        artifact_dir = f"formal/{scenario.name}"
    serve_argv = _serve_argv(scenario, port=port, probe_enabled=probe_enabled)
    benchmark_argv = _benchmark_argv(
        scenario,
        port=port,
        num_prompts=num_prompts,
        concurrency=128,
        artifact_dir=artifact_dir,
    )
    workload_cohort_digest = workload_digest(
        scenario, num_prompts=num_prompts, concurrency=128
    )
    spec = {
        "id": run_id,
        "stage": stage,
        "scenario": scenario.name,
        "probe_enabled": probe_enabled,
        "probe_mode": "on" if probe_enabled else "off",
        "source": "real",
        "workload_cohort_digest": workload_cohort_digest,
        "num_prompts": num_prompts,
        "concurrency": 128,
        "warmup_num_prompts": 128,
        "warmup_concurrency": 128,
        "measurement_requires_warmup_cutoff": probe_enabled,
        "pair_id": pair_id,
        "order_index": order_index,
        "tp": scenario.tp,
        "dp": scenario.dp,
        "ep": scenario.ep,
        "input_len": scenario.input_len,
        "output_len": scenario.output_len,
        "max_num_batched_tokens": scenario.max_num_batched_tokens,
        "artifact_dir": artifact_dir,
        "requires_overhead_gate": stage == "formal",
        "serve_argv": serve_argv,
        "serve_command": shlex.join(serve_argv),
        "benchmark_argv": benchmark_argv,
        "benchmark_command": shlex.join(benchmark_argv),
    }
    warmup_argv = _benchmark_argv(
        scenario,
        port=port,
        num_prompts=128,
        concurrency=128,
        artifact_dir=f"{artifact_dir}/warmup",
    )
    spec["warmup_benchmark_argv"] = warmup_argv
    spec["warmup_benchmark_command"] = shlex.join(warmup_argv)
    spec["run_contract_digest"] = run_contract_digest(spec)
    return spec


def run_contract_digest(spec: dict[str, Any]) -> str:
    payload = {
        key: spec.get(key)
        for key in (
            "id",
            "stage",
            "scenario",
            "probe_mode",
            "source",
            "workload_cohort_digest",
            "num_prompts",
            "concurrency",
            "warmup_num_prompts",
            "warmup_concurrency",
            "pair_id",
            "order_index",
            "tp",
            "dp",
            "ep",
            "input_len",
            "output_len",
            "max_num_batched_tokens",
        )
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def expected_run_meta(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        key: spec[key]
        for key in (
            "id",
            "stage",
            "scenario",
            "probe_mode",
            "source",
            "workload_cohort_digest",
            "run_contract_digest",
            "num_prompts",
            "concurrency",
            "warmup_num_prompts",
            "warmup_concurrency",
            "pair_id",
            "order_index",
            "tp",
            "dp",
            "ep",
            "input_len",
            "output_len",
            "max_num_batched_tokens",
        )
    }


def build_run_plan() -> dict[str, Any]:
    """Return the preregistered command and artifact contract without executing it."""
    overhead = SCENARIOS[OVERHEAD_SCENARIO]
    runs: list[dict[str, Any]] = []
    port = 21460
    for pair_index in range(1, OVERHEAD_PAIR_COUNT + 1):
        pair_id = f"pair-{pair_index:02d}"
        modes = (False, True) if pair_index % 2 else (True, False)
        for order_index, probe_enabled in enumerate(modes, start=1):
            mode = "on" if probe_enabled else "off"
            runs.append(
                _run_spec(
                    run_id=f"overhead-{pair_id}-{mode}",
                    stage="overhead",
                    scenario=overhead,
                    probe_enabled=probe_enabled,
                    num_prompts=128,
                    port=port,
                    pair_id=pair_id,
                    order_index=order_index,
                )
            )
            port += 1
    for index, name in enumerate(FORMAL_SCENARIOS, start=0):
        runs.append(
            _run_spec(
                run_id=f"formal-{name}",
                stage="formal",
                scenario=SCENARIOS[name],
                probe_enabled=True,
                num_prompts=512,
                port=port + index,
            )
        )
    return {
        "schema": SCHEMA,
        "phase": "phase466",
        "base_commit": BASE_COMMIT,
        "probe_default": "off",
        "implementation": "rank_local_logging_handler_no_vllm_source_patch",
        "comparison_semantics": {
            "tp8_vs_tp4dp2": "topology_control_not_pure_dp_control",
        },
        "runtime_source_patch": False,
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
        "default_readiness": "No-Go",
        "environment": {
            "VLLM_ENABLE_CUDA_COMPATIBILITY": "1",
            "VLLM_LOGGING_LEVEL": "INFO",
            "VLLM_LOGGING_CONFIG_PATH": "scripts/phase466_rank_local_logging.json",
            "PYTHONPATH": "<workdir>/scripts",
            "VLLM_NUM_GPU_BLOCKS_OVERRIDE": "unset",
            "NUM_GPU_BLOCKS_OVERRIDE": "unset",
        },
        "source_files": dict(SOURCE_FILES),
        "source_capture_argv": ["sha256sum", *SOURCE_FILES],
        "environment_capture": [
            "required 40-character source commit from coordinator",
            'python3 -c "import vllm; print(vllm.__version__)"',
            "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits",
        ],
        "residue_checks": {
            "gpu": (
                "nvidia-smi --query-compute-apps=pid,process_name,used_memory "
                "--format=csv,noheader"
            ),
            "process": (
                "pgrep -f 'vllm.entrypoints.cli.main serve|ray::|raylet|"
                "gcs_server|VLLM::APIServer|VLLM::EngineCore'"
            ),
        },
        "sampling": {
            "iteration_rank": "rank-local stock-vLLM timing per engine iteration",
            "preemptions": "Prometheus num_preemptions_total before/after per engine",
            "throughput": "one fixed-shape benchmark result per run",
            "gpu_health": "before/after compute-app residue snapshots",
            "per_request_events": "forbidden",
            "alignment": (
                "same workload_cohort_digest; rank-local elapsed offsets; "
                "cumulative scheduled-token progress windows"
            ),
        },
        "overhead_gate": {
            "scenario": OVERHEAD_SCENARIO,
            "num_prompts": 128,
            "concurrency": 128,
            "pair_count": OVERHEAD_PAIR_COUNT,
            "pair_order": [
                "off,on",
                "on,off",
                "off,on",
                "on,off",
                "off,on",
                "on,off",
            ],
            "equivalence_ratio_bounds": list(EQUIVALENCE_RATIO_BOUNDS),
            "confidence_interval": 0.90,
            "method": "paired_log_ratio_tost",
            "stop_before_n512_unless_pass": True,
        },
        "runs": runs,
        "artifact_contract": [
            "manifest.json",
            "environment.json",
            "source.sha256",
            "tooling.sha256",
            "status.json",
            "phase466_result.json",
            "gpu_compute_apps_before.txt",
            "process_residue_before.txt",
            "overhead/<pair-id>/<mode>/meta.json",
            "overhead/<pair-id>/<mode>/source.sha256",
            "overhead/<pair-id>/<mode>/warmup/bench_result.json",
            "overhead/<pair-id>/<mode>/bench_result.json",
            "overhead/<pair-id>/<mode>/metrics_before.prom",
            "overhead/<pair-id>/<mode>/metrics_after.prom",
            "overhead/<pair-id>/<mode>/serve.log.gz",
            "overhead/<pair-id>/<mode>/rank_logs/rank-<id>.jsonl",
            "overhead/<pair-id>/<mode>/gpu_compute_apps_after.txt",
            "overhead/<pair-id>/<mode>/process_residue_after.txt",
            "overhead/<pair-id>/pair_result.json",
            "overhead/overhead_gate.json",
            "formal/<scenario>/meta.json",
            "formal/<scenario>/source.sha256",
            "formal/<scenario>/bench_result.json",
            "formal/<scenario>/metrics_before.prom",
            "formal/<scenario>/metrics_after.prom",
            "formal/<scenario>/serve.log.gz",
            "formal/<scenario>/rank_logs/rank-<id>.jsonl",
            "formal/<scenario>/iteration_rows.csv",
            "formal/<scenario>/rank_summary.csv",
            "formal/<scenario>/prompt_identity.json",
            "formal/<scenario>/probe_summary.json",
            "formal/<scenario>/gpu_compute_apps_after.txt",
            "formal/<scenario>/process_residue_after.txt",
            "rank_timing_report.json",
        ],
        "stop_rules": [
            "preflight_gpu_or_process_residue",
            "vllm_version_not_0.19.0",
            "vllm_source_hash_mismatch",
            "model_snapshot_or_tokenizer_identity_mismatch",
            "prompt_cohort_sha256_missing_or_mismatch",
            "source_hash_changed_between_off_and_on",
            "benchmark_failed_or_token_counts_inexact",
            "service_exited_or_restarted",
            "missing_or_duplicate_iteration_rank_rows",
            "missing_run_source_or_workload_cohort_digest",
            "prometheus_counter_missing_or_reset",
            "overhead_gate_not_pass_stop_before_n512",
            "postflight_gpu_or_process_residue",
            "scenario_outside_three_formal_targets",
        ],
    }


def empty_rank_log_identity() -> dict[str, Any]:
    return {"schema": RANK_LOG_IDENTITY_SCHEMA, "files": {}}


def _load_rank_logs(
    rank_log_dir: Path, *, expected_ranks: set[int]
) -> tuple[dict[int, list[RankLogRecord]], dict[str, Any]]:
    if not expected_ranks or expected_ranks != set(range(len(expected_ranks))):
        raise ValueError(f"invalid_expected_rank_set:{sorted(expected_ranks)}")
    if not rank_log_dir.is_dir():
        raise ValueError(f"missing_rank_log_dir:{rank_log_dir}")
    expected_files = {f"rank-{rank}.jsonl" for rank in expected_ranks}
    actual_files = {path.name for path in rank_log_dir.iterdir()}
    if actual_files != expected_files:
        raise ValueError(
            f"rank_log_file_set_mismatch:{sorted(actual_files)}!={sorted(expected_files)}"
        )

    by_rank: dict[int, list[RankLogRecord]] = {}
    files_identity: dict[str, dict[str, Any]] = {}
    expected_dp_process_names = len(expected_ranks) > 1
    for rank in sorted(expected_ranks):
        filename = f"rank-{rank}.jsonl"
        path = rank_log_dir / filename
        content = path.read_bytes()
        try:
            lines = content.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ValueError(f"rank_log_not_utf8:{filename}") from exc
        if not lines:
            raise ValueError(f"empty_rank_log:{filename}")
        expected_process_name = (
            f"EngineCore_DP{rank}" if expected_dp_process_names else "EngineCore"
        )
        records: list[RankLogRecord] = []
        pids: set[int] = set()
        iterations: list[int] = []
        for line_number, line in enumerate(lines, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid_rank_log_json:{filename}:{line_number}") from exc
            if not isinstance(value, dict) or set(value) != {
                "rank",
                "pid",
                "process_name",
                "message",
            }:
                raise ValueError(f"invalid_rank_log_record:{filename}:{line_number}")
            record_rank = value["rank"]
            pid = value["pid"]
            process_name = value["process_name"]
            message = value["message"]
            if isinstance(record_rank, bool) or record_rank != rank:
                raise ValueError(f"rank_log_record_rank_mismatch:{filename}:{line_number}")
            if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
                raise ValueError(f"invalid_rank_log_pid:{filename}:{line_number}")
            if process_name != expected_process_name:
                raise ValueError(f"rank_log_process_name_mismatch:{filename}:{line_number}")
            if not isinstance(message, str):
                raise ValueError(f"invalid_rank_log_message:{filename}:{line_number}")
            match = ITERATION_RE.fullmatch(message)
            if match is None:
                raise ValueError(f"invalid_rank_iteration_message:{filename}:{line_number}")
            elapsed_ms = float(match.group("elapsed_ms"))
            if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
                raise ValueError(f"invalid_iteration_elapsed:{filename}:{line_number}")
            pids.add(pid)
            iterations.append(int(match.group("iteration")))
            records.append(RankLogRecord(rank, pid, process_name, match))
        if len(pids) != 1:
            raise ValueError(f"rank_log_pid_drift:{filename}")
        if iterations != list(range(iterations[0], iterations[-1] + 1)):
            raise ValueError(f"rank_iteration_not_contiguous:{filename}")
        by_rank[rank] = records
        files_identity[filename] = {
            "rank": rank,
            "pid": next(iter(pids)),
            "process_name": expected_process_name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "line_count": len(lines),
            "first_iteration_seq": iterations[0],
            "last_iteration_seq": iterations[-1],
        }
    return by_rank, {
        "schema": RANK_LOG_IDENTITY_SCHEMA,
        "files": files_identity,
    }


def rank_log_identity(
    rank_log_dir: Path, *, expected_ranks: set[int]
) -> dict[str, Any]:
    _, identity = _load_rank_logs(rank_log_dir, expected_ranks=expected_ranks)
    return identity


def last_iteration_by_rank(
    rank_log_dir: Path, *, expected_ranks: set[int]
) -> dict[int, int]:
    _, identity = _load_rank_logs(rank_log_dir, expected_ranks=expected_ranks)
    return {
        int(row["rank"]): int(row["last_iteration_seq"])
        for row in identity["files"].values()
    }


def parse_iteration_rows(
    rank_log_dir: Path,
    *,
    expected_ranks: set[int],
    run_id: str = "",
    source: str = "",
    workload_cohort_digest: str = "",
    after_iteration_by_rank: dict[int, int] | None = None,
    through_iteration_by_rank: dict[int, int] | None = None,
) -> list[IterationRow]:
    by_rank, _ = _load_rank_logs(rank_log_dir, expected_ranks=expected_ranks)
    if after_iteration_by_rank is not None and set(after_iteration_by_rank) != expected_ranks:
        raise ValueError("warmup_cutoff_rank_set_mismatch")
    if through_iteration_by_rank is not None:
        if after_iteration_by_rank is None:
            raise ValueError("measurement_end_requires_warmup_cutoff")
        if set(through_iteration_by_rank) != expected_ranks:
            raise ValueError("measurement_rank_set_mismatch")

    rows: list[IterationRow] = []
    elapsed_by_rank: dict[int, float] = defaultdict(float)
    tokens_by_rank: dict[int, int] = defaultdict(int)
    for rank, records in sorted(by_rank.items()):
        start = after_iteration_by_rank[rank] + 1 if after_iteration_by_rank else None
        end = through_iteration_by_rank[rank] if through_iteration_by_rank else None
        selected = [
            record
            for record in records
            if (start is None or int(record.match.group("iteration")) >= start)
            and (end is None or int(record.match.group("iteration")) <= end)
        ]
        if not selected:
            raise ValueError(f"missing_iteration_rank_rows:{rank}")
        selected_iterations = [
            int(record.match.group("iteration")) for record in selected
        ]
        expected_start = start if start is not None else selected_iterations[0]
        expected_end = end if end is not None else selected_iterations[-1]
        if selected_iterations != list(range(expected_start, expected_end + 1)):
            raise ValueError(f"measurement_iteration_gap:{rank}")
        for record in selected:
            match = record.match
            iteration_seq = int(match.group("iteration"))
            elapsed_ms = float(match.group("elapsed_ms"))
            scheduled_tokens = int(match.group("context_tokens")) + int(
                match.group("generation_tokens")
            )
            start_offset_ms = elapsed_by_rank[rank]
            end_offset_ms = start_offset_ms + elapsed_ms
            progress_start_tokens = tokens_by_rank[rank]
            cumulative_scheduled_tokens = progress_start_tokens + scheduled_tokens
            rows.append(
                IterationRow(
                    rank_id=rank,
                    rank_scope="dp_rank",
                    run_id=run_id,
                    source=source,
                    workload_cohort_digest=workload_cohort_digest,
                    iteration_seq=iteration_seq,
                    progress_start_tokens=progress_start_tokens,
                    progress_end_tokens=cumulative_scheduled_tokens,
                    prefill_request_count=int(match.group("context_requests")),
                    scheduled_prefill_tokens=int(match.group("context_tokens")),
                    decode_request_count=int(match.group("generation_requests")),
                    scheduled_decode_tokens=int(match.group("generation_tokens")),
                    iteration_elapsed_ms=elapsed_ms,
                    iteration_start_offset_ms=start_offset_ms,
                    iteration_end_offset_ms=end_offset_ms,
                    cumulative_scheduled_tokens=cumulative_scheduled_tokens,
                    progress_window_id=max(
                        0,
                        (cumulative_scheduled_tokens - 1)
                        // PROGRESS_WINDOW_TOKENS,
                    ),
                )
            )
            elapsed_by_rank[rank] = end_offset_ms
            tokens_by_rank[rank] = cumulative_scheduled_tokens
    if not rows:
        raise ValueError("missing_iteration_rank_rows")
    return rows


def _preemption_values(text: str) -> dict[int, float]:
    values: dict[int, float] = {}
    for line in text.splitlines():
        match = PROM_RE.match(line.strip())
        if not match or match.group("name") != "vllm:num_preemptions_total":
            continue
        labels = dict(LABEL_RE.findall(match.group("labels") or ""))
        rank = int(labels.get("engine", "0"))
        if rank in values:
            raise ValueError(f"duplicate_preemption_series:{rank}")
        value = float(match.group("value"))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid_preemption_counter:{rank}:{value}")
        values[rank] = value
    if not values:
        raise ValueError("missing_preemption_counter")
    return values


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def iteration_csv_identity(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty_iteration_csv:{path}")
    run_ids = {row.get("run_id", "") for row in rows}
    sources = {row.get("source", "") for row in rows}
    digests = {row.get("workload_cohort_digest", "") for row in rows}
    if len(run_ids) != 1 or "" in run_ids:
        raise ValueError(f"iteration_csv_run_id_mismatch:{path}")
    if len(sources) != 1 or "" in sources:
        raise ValueError(f"iteration_csv_source_mismatch:{path}")
    if len(digests) != 1 or not re.fullmatch(r"[0-9a-f]{64}", next(iter(digests))):
        raise ValueError(f"iteration_csv_workload_digest_mismatch:{path}")
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["rank_id"])].append(row)
    rank_bounds: dict[str, dict[str, Any]] = {}
    for rank, rank_rows in sorted(grouped.items()):
        ordered = sorted(rank_rows, key=lambda row: int(row["iteration_seq"]))
        first = ordered[0]
        last = ordered[-1]
        rank_bounds[str(rank)] = {
            "row_count": len(ordered),
            "first_iteration_seq": int(first["iteration_seq"]),
            "last_iteration_seq": int(last["iteration_seq"]),
            "first_progress_start_tokens": int(first["progress_start_tokens"]),
            "last_progress_end_tokens": int(last["progress_end_tokens"]),
            "first_start_offset_ms": float(first["iteration_start_offset_ms"]),
            "last_end_offset_ms": float(last["iteration_end_offset_ms"]),
        }
    return {
        "sha256": hashlib.sha256(content).hexdigest(),
        "row_count": len(rows),
        "run_id": next(iter(run_ids)),
        "source": next(iter(sources)),
        "workload_cohort_digest": next(iter(digests)),
        "rank_bounds": rank_bounds,
    }


def summarize_ranks(
    rows: list[IterationRow],
    *,
    metrics_before: str,
    metrics_after: str,
) -> list[dict[str, Any]]:
    run_ids = {row.run_id for row in rows}
    sources = {row.source for row in rows}
    workload_digests = {row.workload_cohort_digest for row in rows}
    rank_scopes = {row.rank_scope for row in rows}
    if len(run_ids) != 1 or "" in run_ids or len(sources) != 1:
        raise ValueError("missing_run_id_or_source")
    run_id = next(iter(run_ids))
    source = next(iter(sources))
    if source not in {"real", "sim"}:
        raise ValueError("missing_run_id_or_source")
    if len(workload_digests) != 1:
        raise ValueError("invalid_workload_cohort_digest")
    if rank_scopes != {"dp_rank"}:
        raise ValueError("invalid_real_rank_scope")
    workload_cohort_digest = next(iter(workload_digests))
    if not re.fullmatch(r"[0-9a-f]{64}", workload_cohort_digest):
        raise ValueError("invalid_workload_cohort_digest")
    before = _preemption_values(metrics_before)
    after = _preemption_values(metrics_after)
    grouped: dict[int, list[IterationRow]] = defaultdict(list)
    for row in rows:
        grouped[row.rank_id].append(row)
    summaries: list[dict[str, Any]] = []
    for rank, rank_rows in sorted(grouped.items()):
        if rank not in before or rank not in after:
            raise ValueError(f"missing_preemption_rank:{rank}")
        preemptions = after[rank] - before[rank]
        if preemptions < 0:
            raise ValueError(f"preemption_counter_reset:{rank}")
        context_tokens = sum(row.scheduled_prefill_tokens for row in rank_rows)
        generation_tokens = sum(row.scheduled_decode_tokens for row in rank_rows)
        scheduled_tokens = context_tokens + generation_tokens
        if scheduled_tokens <= 0:
            raise ValueError(f"rank_has_no_scheduled_tokens:{rank}")
        elapsed = [row.iteration_elapsed_ms for row in rank_rows]
        elapsed_sum = sum(elapsed)
        summaries.append(
            {
                "run_id": run_id,
                "source": source,
                "workload_cohort_digest": workload_cohort_digest,
                "rank_id": rank,
                "rank_scope": "dp_rank",
                "iteration_count": len(rank_rows),
                "context_only_iteration_count": sum(
                    row.scheduled_prefill_tokens > 0
                    and row.scheduled_decode_tokens == 0
                    for row in rank_rows
                ),
                "generation_only_iteration_count": sum(
                    row.scheduled_prefill_tokens == 0
                    and row.scheduled_decode_tokens > 0
                    for row in rank_rows
                ),
                "mixed_iteration_count": sum(
                    row.scheduled_prefill_tokens > 0
                    and row.scheduled_decode_tokens > 0
                    for row in rank_rows
                ),
                "context_requests_total": sum(
                    row.prefill_request_count for row in rank_rows
                ),
                "context_tokens_total": context_tokens,
                "generation_requests_total": sum(
                    row.decode_request_count for row in rank_rows
                ),
                "generation_tokens_total": generation_tokens,
                "scheduled_tokens_total": scheduled_tokens,
                "elapsed_ms_sum": elapsed_sum,
                "iteration_start_offset_ms": rank_rows[
                    0
                ].iteration_start_offset_ms,
                "iteration_end_offset_ms": rank_rows[-1].iteration_end_offset_ms,
                "progress_start_tokens": rank_rows[0].progress_start_tokens,
                "progress_end_tokens": rank_rows[-1].progress_end_tokens,
                "cumulative_scheduled_tokens": rank_rows[
                    -1
                ].cumulative_scheduled_tokens,
                "progress_window_count": len(
                    {row.progress_window_id for row in rank_rows}
                ),
                "elapsed_ms_mean": statistics.mean(elapsed),
                "elapsed_ms_p50": _percentile(elapsed, 50.0),
                "elapsed_ms_p90": _percentile(elapsed, 90.0),
                "elapsed_ms_p99": _percentile(elapsed, 99.0),
                "elapsed_ms_per_scheduled_token": elapsed_sum / scheduled_tokens,
                "preemptions": preemptions,
            }
        )
    return summaries


def evaluate_paired_overhead_gate(
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_pairs = {f"pair-{index:02d}" for index in range(1, OVERHEAD_PAIR_COUNT + 1)}
    pair_ids = [str(pair.get("pair_id", "")) for pair in pairs]
    if len(pair_ids) != OVERHEAD_PAIR_COUNT or set(pair_ids) != expected_pairs:
        raise ValueError(f"overhead_pair_set_mismatch:{sorted(pair_ids)}")

    normalized: list[dict[str, Any]] = []
    log_ratios: list[float] = []
    for pair in sorted(pairs, key=lambda row: str(row["pair_id"])):
        off = float(pair["off_output_tok_s"])
        on = float(pair["on_output_tok_s"])
        if not math.isfinite(off) or not math.isfinite(on) or off <= 0 or on <= 0:
            raise ValueError(f"overhead_pair_invalid_throughput:{pair['pair_id']}")
        ratio = on / off
        log_ratio = math.log(ratio)
        log_ratios.append(log_ratio)
        normalized.append(
            {
                "pair_id": pair["pair_id"],
                "off_output_tok_s": off,
                "on_output_tok_s": on,
                "ratio": ratio,
                "absolute_delta_pct": abs(ratio - 1.0) * 100.0,
            }
        )

    mean_log_ratio = statistics.mean(log_ratios)
    sample_stddev = statistics.stdev(log_ratios)
    margin = T_CRITICAL_90_DF5 * sample_stddev / math.sqrt(OVERHEAD_PAIR_COUNT)
    ci_lower = mean_log_ratio - margin
    ci_upper = mean_log_ratio + margin
    lower_bound = math.log(EQUIVALENCE_RATIO_BOUNDS[0])
    upper_bound = math.log(EQUIVALENCE_RATIO_BOUNDS[1])
    if ci_lower >= lower_bound and ci_upper <= upper_bound:
        status = "PASS"
    elif ci_upper < lower_bound or ci_lower > upper_bound:
        status = "FAIL"
    else:
        status = "INCONCLUSIVE"

    return {
        "schema": SCHEMA,
        "gate": "dp2_bt65536_stock_iteration_detail_paired_equivalence",
        "method": "paired_log_ratio_tost",
        "pair_count": OVERHEAD_PAIR_COUNT,
        "pairs": normalized,
        "geometric_mean_ratio": math.exp(mean_log_ratio),
        "confidence_interval_ratio": [math.exp(ci_lower), math.exp(ci_upper)],
        "equivalence_ratio_bounds": list(EQUIVALENCE_RATIO_BOUNDS),
        "status": status,
        "passed": status == "PASS",
        "formal_collection_allowed": status == "PASS",
    }


def require_formal_collection(gate: dict[str, Any]) -> None:
    pairs = gate.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("overhead_gate_failed_stop_before_n512")
    try:
        recomputed = evaluate_paired_overhead_gate(pairs)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("overhead_gate_failed_stop_before_n512") from exc
    if gate != recomputed or recomputed.get("status") != "PASS":
        raise ValueError("overhead_gate_failed_stop_before_n512")


def overhead_gate_digest(gate: dict[str, Any]) -> str:
    require_formal_collection(gate)
    payload = json.dumps(gate, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def _load_source_hashes(path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = raw.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"invalid_source_hash_line:{raw}")
        digest, source_path = parts
        source_path = source_path.lstrip("*")
        if source_path in hashes:
            raise ValueError(f"duplicate_source_hash_path:{source_path}")
        hashes[source_path] = digest
    return hashes


def _validate_residue(root: Path) -> None:
    checks = {
        "gpu_compute_apps_after.txt": "nonempty_gpu_residue",
        "process_residue_after.txt": "nonempty_process_residue",
    }
    for filename, error in checks.items():
        path = root / filename
        if not path.exists():
            raise ValueError(f"missing_residue_check:{path}")
        if path.read_text(encoding="utf-8").strip():
            raise ValueError(f"{error}:{path}")


def _validate_overhead_benchmark(result: dict[str, Any], *, root: Path) -> float:
    expected = {
        "ok_requests": 128,
        "failed_requests": 0,
        "total_prompt_tokens": 128 * 8000,
        "total_completion_tokens": 128 * 2000,
    }
    for key, wanted in expected.items():
        if result.get(key) != wanted:
            raise ValueError(f"overhead_benchmark_mismatch:{root}:{key}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(result.get("prompt_cohort_sha256", ""))):
        raise ValueError(f"overhead_prompt_cohort_sha256_missing:{root}")
    throughput = float(result["output_tok_s"])
    if not math.isfinite(throughput) or throughput <= 0:
        raise ValueError(f"overhead_benchmark_invalid_throughput:{root}")
    return throughput


def _validate_exact_run_meta(
    meta: dict[str, Any],
    spec: dict[str, Any],
    *,
    expected_gate_digest: str | None = None,
) -> None:
    expected = expected_run_meta(spec)
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"run_meta_mismatch:{key}")
    if meta.get("vllm_version") != "0.19.0":
        raise ValueError("run_meta_mismatch:vllm_version")
    if meta.get("model_identity_schema") not in {
        "phase466_snapshot_model_identity_v1",
        "phase466_flat_model_fingerprint_v1",
    }:
        raise ValueError("run_meta_mismatch:model_identity_schema")
    if not re.fullmatch(
        r"[0-9a-f]{64}", str(meta.get("model_identity_sha256", ""))
    ):
        raise ValueError("run_meta_mismatch:model_identity_sha256")
    cutoffs = meta.get("measurement_start_after_iteration")
    if not isinstance(cutoffs, dict):
        raise ValueError("run_meta_mismatch:measurement_start_after_iteration")
    for rank, value in cutoffs.items():
        if not str(rank).isdigit() or not isinstance(value, int) or value < -1:
            raise ValueError("run_meta_mismatch:measurement_start_after_iteration")
    ends = meta.get("measurement_end_at_iteration")
    if not isinstance(ends, dict) or set(ends) != set(cutoffs):
        raise ValueError("run_meta_mismatch:measurement_end_at_iteration")
    for rank, value in ends.items():
        if not str(rank).isdigit() or not isinstance(value, int):
            raise ValueError("run_meta_mismatch:measurement_end_at_iteration")
        if value < int(cutoffs[rank]):
            raise ValueError("run_meta_mismatch:measurement_end_at_iteration")
    rank_logs_identity = meta.get("rank_logs_identity")
    if not isinstance(rank_logs_identity, dict) or rank_logs_identity.get(
        "schema"
    ) != RANK_LOG_IDENTITY_SCHEMA or not isinstance(
        rank_logs_identity.get("files"), dict
    ):
        raise ValueError("run_meta_mismatch:rank_logs_identity")
    if spec["probe_enabled"]:
        expected_ranks = {str(rank) for rank in range(int(spec["dp"]))}
        if set(cutoffs) != expected_ranks or not rank_logs_identity["files"]:
            raise ValueError("run_meta_mismatch:rank_logs_identity")
    elif cutoffs or ends or rank_logs_identity != empty_rank_log_identity():
        raise ValueError("run_meta_mismatch:rank_logs_identity")
    manifest_digest = meta.get("execution_manifest_sha256")
    if not isinstance(manifest_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", manifest_digest
    ):
        raise ValueError("run_meta_mismatch:execution_manifest_sha256")
    if expected_gate_digest is not None:
        if meta.get("overhead_gate_sha256") != expected_gate_digest:
            raise ValueError("run_meta_mismatch:overhead_gate_sha256")
    for key in ("warmup_prompt_cohort_sha256", "prompt_cohort_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(meta.get(key, ""))):
            raise ValueError(f"run_meta_mismatch:{key}")


def _validate_source_and_cleanup(root: Path) -> None:
    hashes = _load_source_hashes(root / "source.sha256")
    if hashes != SOURCE_FILES:
        raise ValueError("vllm_source_hash_mismatch")
    _validate_residue(root)


def _validate_off_rank_logs(root: Path) -> None:
    rank_log_dir = root / "rank_logs"
    if not rank_log_dir.is_dir():
        raise ValueError(f"missing_rank_log_dir:{rank_log_dir}")
    if any(rank_log_dir.iterdir()):
        raise ValueError("off_rank_logs_present")


def _validated_rank_rows(
    root: Path, *, meta: dict[str, Any], spec: dict[str, Any]
) -> tuple[list[IterationRow], dict[str, Any]]:
    expected_ranks = set(range(int(spec["dp"])))
    rank_log_dir = root / "rank_logs"
    identity = rank_log_identity(rank_log_dir, expected_ranks=expected_ranks)
    if identity != meta["rank_logs_identity"]:
        raise ValueError("rank_logs_identity_mismatch")
    cutoffs = {
        int(rank): int(value)
        for rank, value in meta["measurement_start_after_iteration"].items()
    }
    ends = {
        int(rank): int(value)
        for rank, value in meta["measurement_end_at_iteration"].items()
    }
    rows = parse_iteration_rows(
        rank_log_dir,
        expected_ranks=expected_ranks,
        run_id=str(meta["id"]),
        source=str(meta["source"]),
        workload_cohort_digest=str(meta["workload_cohort_digest"]),
        after_iteration_by_rank=cutoffs,
        through_iteration_by_rank=ends,
    )
    return rows, identity


def validate_overhead_pair_artifacts(
    off_dir: Path,
    on_dir: Path,
    *,
    off_spec: dict[str, Any],
    on_spec: dict[str, Any],
) -> dict[str, Any]:
    if off_spec.get("pair_id") != on_spec.get("pair_id"):
        raise ValueError("overhead_pair_identity_mismatch")
    if off_spec.get("probe_mode") != "off" or on_spec.get("probe_mode") != "on":
        raise ValueError("overhead_probe_mode_mismatch")
    if {off_spec.get("order_index"), on_spec.get("order_index")} != {1, 2}:
        raise ValueError("overhead_pair_order_mismatch")

    off_meta = _load_json(off_dir / "meta.json")
    on_meta = _load_json(on_dir / "meta.json")
    _validate_exact_run_meta(off_meta, off_spec)
    _validate_exact_run_meta(on_meta, on_spec)
    _validate_source_and_cleanup(off_dir)
    _validate_source_and_cleanup(on_dir)
    _validate_off_rank_logs(off_dir)

    off_benchmark = _load_json(off_dir / "bench_result.json")
    on_benchmark = _load_json(on_dir / "bench_result.json")
    off_throughput = _validate_overhead_benchmark(off_benchmark, root=off_dir)
    on_throughput = _validate_overhead_benchmark(on_benchmark, root=on_dir)
    if off_benchmark["prompt_cohort_sha256"] != off_meta["prompt_cohort_sha256"]:
        raise ValueError("prompt_cohort_sha256_mismatch:off")
    if on_benchmark["prompt_cohort_sha256"] != on_meta["prompt_cohort_sha256"]:
        raise ValueError("prompt_cohort_sha256_mismatch:on")
    if off_benchmark["prompt_cohort_sha256"] != on_benchmark["prompt_cohort_sha256"]:
        raise ValueError("overhead_prompt_cohort_sha256_drift")

    metrics_before = on_dir / "metrics_before.prom"
    metrics_after = on_dir / "metrics_after.prom"
    if not metrics_before.exists() or not metrics_after.exists():
        raise ValueError(f"missing_probe_metrics:{on_dir}")
    rows, rank_logs_identity = _validated_rank_rows(
        on_dir, meta=on_meta, spec=on_spec
    )
    rank_summary = summarize_ranks(
        rows,
        metrics_before=metrics_before.read_text(encoding="utf-8"),
        metrics_after=metrics_after.read_text(encoding="utf-8"),
    )
    if [row["rank_id"] for row in rank_summary] != [0, 1]:
        raise ValueError("overhead_probe_missing_dp_rank")
    return {
        "schema": SCHEMA,
        "pair_id": off_spec["pair_id"],
        "off_output_tok_s": off_throughput,
        "on_output_tok_s": on_throughput,
        "rank_summary": rank_summary,
        "rank_logs_identity": rank_logs_identity,
        "source_hash_match": True,
        "cleanup": True,
    }


def validate_overhead_root(root: Path) -> dict[str, Any]:
    runs = {run["id"]: run for run in build_run_plan()["runs"]}
    pair_results: list[dict[str, Any]] = []
    for pair_index in range(1, OVERHEAD_PAIR_COUNT + 1):
        pair_id = f"pair-{pair_index:02d}"
        off_id = f"overhead-{pair_id}-off"
        on_id = f"overhead-{pair_id}-on"
        off_dir = root / str(runs[off_id]["artifact_dir"])
        on_dir = root / str(runs[on_id]["artifact_dir"])
        if not off_dir.is_dir() or not on_dir.is_dir():
            raise ValueError(f"missing_overhead_pair_artifact:{pair_id}")
        pair_results.append(
            validate_overhead_pair_artifacts(
                off_dir,
                on_dir,
                off_spec=runs[off_id],
                on_spec=runs[on_id],
            )
        )
    return evaluate_paired_overhead_gate(pair_results)


def formal_spec(scenario: str) -> dict[str, Any]:
    matches = [
        run
        for run in build_run_plan()["runs"]
        if run["stage"] == "formal" and run["scenario"] == scenario
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown_formal_scenario:{scenario}")
    return matches[0]


def _validate_formal_benchmark(
    result: dict[str, Any], *, spec: dict[str, Any]
) -> float:
    expected = {
        "ok_requests": int(spec["num_prompts"]),
        "failed_requests": 0,
        "total_prompt_tokens": int(spec["num_prompts"]) * int(spec["input_len"]),
        "total_completion_tokens": int(spec["num_prompts"])
        * int(spec["output_len"]),
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"formal_benchmark_mismatch:{key}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(result.get("prompt_cohort_sha256", ""))):
        raise ValueError("formal_prompt_cohort_sha256_missing")
    throughput = float(result.get("output_tok_s", 0.0))
    if not math.isfinite(throughput) or throughput <= 0:
        raise ValueError("formal_benchmark_invalid_throughput")
    return throughput


def validate_formal_artifacts(
    root: Path,
    *,
    spec: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    gate_digest = overhead_gate_digest(gate)
    if spec.get("stage") != "formal" or spec.get("scenario") not in FORMAL_SCENARIOS:
        raise ValueError("scenario_outside_three_formal_targets")
    meta = _load_json(root / "meta.json")
    _validate_exact_run_meta(meta, spec, expected_gate_digest=gate_digest)
    _validate_source_and_cleanup(root)
    benchmark = _load_json(root / "bench_result.json")
    throughput = _validate_formal_benchmark(benchmark, spec=spec)
    if benchmark["prompt_cohort_sha256"] != meta["prompt_cohort_sha256"]:
        raise ValueError("formal_prompt_cohort_sha256_mismatch")

    metrics_before = root / "metrics_before.prom"
    metrics_after = root / "metrics_after.prom"
    if not metrics_before.exists() or not metrics_after.exists():
        raise ValueError(f"missing_probe_metrics:{root}")
    rows, rank_logs_identity = _validated_rank_rows(root, meta=meta, spec=spec)
    rank_summary = summarize_ranks(
        rows,
        metrics_before=metrics_before.read_text(encoding="utf-8"),
        metrics_after=metrics_after.read_text(encoding="utf-8"),
    )
    expected_ranks = list(range(int(spec["dp"])))
    actual_ranks = [row["rank_id"] for row in rank_summary]
    if actual_ranks != expected_ranks:
        raise ValueError(f"formal_rank_set_mismatch:{actual_ranks}!={expected_ranks}")
    return {
        "schema": SCHEMA,
        "status": "ARTIFACT_VALID",
        "scenario": spec["scenario"],
        "output_tok_s": throughput,
        "iteration_rows": [asdict(row) for row in rows],
        "rank_summary": rank_summary,
        "rank_logs_identity": rank_logs_identity,
        "source_hash_match": True,
        "cleanup": True,
        "overhead_gate_sha256": gate_digest,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot_write_empty_rank_summary")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="write the preregistered run plan")
    plan.add_argument("--output-json", type=Path, required=True)

    summarize = subparsers.add_parser(
        "summarize", help="parse stock iteration logs into rank aggregates"
    )
    summarize.add_argument("--rank-log-dir", type=Path, required=True)
    summarize.add_argument("--expected-rank-count", type=int, required=True)
    summarize.add_argument("--metrics-before", type=Path, required=True)
    summarize.add_argument("--metrics-after", type=Path, required=True)
    summarize.add_argument("--run-id", required=True)
    summarize.add_argument("--source", choices=("real", "sim"), required=True)
    summarize.add_argument("--workload-cohort-digest", required=True)
    summarize.add_argument("--output-iteration-csv", type=Path, required=True)
    summarize.add_argument("--output-json", type=Path, required=True)
    summarize.add_argument("--output-csv", type=Path, required=True)

    gate = subparsers.add_parser("gate", help="validate all six overhead pairs")
    gate.add_argument("--artifact-root", type=Path, required=True)
    gate.add_argument("--output-json", type=Path, required=True)

    formal = subparsers.add_parser("formal", help="validate one preregistered formal run")
    formal.add_argument("--run-dir", type=Path, required=True)
    formal.add_argument("--scenario", choices=FORMAL_SCENARIOS, required=True)
    formal.add_argument("--artifact-root", type=Path, required=True)
    formal.add_argument("--output-json", type=Path, required=True)
    formal.add_argument("--output-iteration-csv", type=Path, required=True)
    formal.add_argument("--output-rank-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "plan":
        _write_json(args.output_json, build_run_plan())
        return 0
    if args.command == "summarize":
        rows = parse_iteration_rows(
            args.rank_log_dir,
            expected_ranks=set(range(args.expected_rank_count)),
            run_id=args.run_id,
            source=args.source,
            workload_cohort_digest=args.workload_cohort_digest,
        )
        summaries = summarize_ranks(
            rows,
            metrics_before=args.metrics_before.read_text(encoding="utf-8"),
            metrics_after=args.metrics_after.read_text(encoding="utf-8"),
        )
        _write_json(
            args.output_json,
            {
                "schema": SCHEMA,
                "rank_count": len(summaries),
                "ranks": summaries,
                "request_identity_collected": False,
            },
        )
        _write_csv(args.output_iteration_csv, [asdict(row) for row in rows])
        _write_csv(args.output_csv, summaries)
        return 0
    if args.command == "gate":
        gate = validate_overhead_root(args.artifact_root)
        _write_json(args.output_json, gate)
        return 0 if gate["status"] == "PASS" else 2
    result = validate_formal_artifacts(
        args.run_dir,
        spec=formal_spec(args.scenario),
        gate=validate_overhead_root(args.artifact_root),
    )
    _write_json(
        args.output_json,
        {key: value for key, value in result.items() if key != "iteration_rows"},
    )
    _write_csv(args.output_iteration_csv, result["iteration_rows"])
    _write_csv(args.output_rank_csv, result["rank_summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
