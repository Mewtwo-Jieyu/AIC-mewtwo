#!/usr/bin/env python3
"""Build and validate the Phase466 stock-vLLM aggregate probe plan."""

from __future__ import annotations

import argparse
import csv
import gzip
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


BASE_COMMIT = "9ba5ad93ca8805a1eb427593ced8cee01aea6804"
SCHEMA = "phase466_low_overhead_probe_v1"
PROBE_FLAG = "--enable-logging-iteration-details"
MAX_OVERHEAD_PCT = 2.0
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
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py": (
        "896730e749cbcabb487ce50c703974594d197fc31a1d3b26fe096197d142d2d5"
    ),
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py": (
        "9f3da2dfce94963e1e1cefc156e4239120f79c6eaf824b2829cac0ee7736b58a"
    ),
}

ITERATION_RE = re.compile(
    r"EngineCore(?:_DP(?P<rank>\d+))?.*?"
    r"Iteration\((?P<iteration>\d+)\):\s+"
    r"(?P<context_requests>\d+)\s+context requests,\s+"
    r"(?P<context_tokens>\d+)\s+context tokens,\s+"
    r"(?P<generation_requests>\d+)\s+generation requests,\s+"
    r"(?P<generation_tokens>\d+)\s+generation tokens,\s+"
    r"iteration elapsed time:\s+(?P<elapsed_ms>[0-9.]+)\s+ms"
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
        "scenario": scenario.name,
        "num_prompts": num_prompts,
        "concurrency": concurrency,
        "input_len": scenario.input_len,
        "output_len": scenario.output_len,
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
) -> dict[str, Any]:
    artifact_dir = (
        f"overhead/{'on' if probe_enabled else 'off'}"
        if stage == "overhead"
        else f"formal/{scenario.name}"
    )
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
    return {
        "id": run_id,
        "stage": stage,
        "scenario": scenario.name,
        "probe_enabled": probe_enabled,
        "probe_mode": "on" if probe_enabled else "off",
        "source": "real",
        "workload_cohort_digest": workload_cohort_digest,
        "num_prompts": num_prompts,
        "concurrency": 128,
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


def build_run_plan() -> dict[str, Any]:
    """Return the preregistered command and artifact contract without executing it."""
    overhead = SCENARIOS[OVERHEAD_SCENARIO]
    runs = [
        _run_spec(
            run_id="overhead-off",
            stage="overhead",
            scenario=overhead,
            probe_enabled=False,
            num_prompts=128,
            port=21460,
        ),
        _run_spec(
            run_id="overhead-on",
            stage="overhead",
            scenario=overhead,
            probe_enabled=True,
            num_prompts=128,
            port=21461,
        ),
    ]
    for index, name in enumerate(FORMAL_SCENARIOS, start=0):
        runs.append(
            _run_spec(
                run_id=f"formal-{name}",
                stage="formal",
                scenario=SCENARIOS[name],
                probe_enabled=True,
                num_prompts=512,
                port=21462 + index,
            )
        )
    return {
        "schema": SCHEMA,
        "phase": "phase466",
        "base_commit": BASE_COMMIT,
        "probe_default": "off",
        "implementation": "stock_vllm_iteration_details_no_source_patch",
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
            "VLLM_NUM_GPU_BLOCKS_OVERRIDE": "unset",
            "NUM_GPU_BLOCKS_OVERRIDE": "unset",
        },
        "source_files": dict(SOURCE_FILES),
        "source_capture_argv": ["sha256sum", *SOURCE_FILES],
        "environment_capture": [
            "git rev-parse HEAD",
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
            "iteration_rank": "one aggregate stock-vLLM line per engine iteration",
            "preemptions": "Prometheus num_preemptions_total before/after per engine",
            "throughput": "one fixed-shape benchmark result per run",
            "gpu_health": "nvidia-smi every 2 seconds; not a model attribution field",
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
            "max_absolute_delta_pct": MAX_OVERHEAD_PCT,
            "stop_before_n512_on_failure": True,
        },
        "runs": runs,
        "artifact_contract": [
            "manifest.json",
            "environment.json",
            "source.sha256",
            "gpu_compute_apps_before.txt",
            "process_residue_before.txt",
            "overhead/off/meta.json",
            "overhead/off/source.sha256",
            "overhead/off/bench_result.json",
            "overhead/off/metrics_before.prom",
            "overhead/off/metrics_after.prom",
            "overhead/off/serve.log.gz",
            "overhead/off/gpu_compute_apps_after.txt",
            "overhead/off/process_residue_after.txt",
            "overhead/on/meta.json",
            "overhead/on/source.sha256",
            "overhead/on/bench_result.json",
            "overhead/on/metrics_before.prom",
            "overhead/on/metrics_after.prom",
            "overhead/on/serve.log.gz",
            "overhead/on/iteration_rows.csv",
            "overhead/on/gpu_compute_apps_after.txt",
            "overhead/on/process_residue_after.txt",
            "overhead/overhead_gate.json",
            "formal/<scenario>/meta.json",
            "formal/<scenario>/source.sha256",
            "formal/<scenario>/bench_result.json",
            "formal/<scenario>/metrics_before.prom",
            "formal/<scenario>/metrics_after.prom",
            "formal/<scenario>/serve.log.gz",
            "formal/<scenario>/iteration_rows.csv",
            "formal/<scenario>/rank_summary.csv",
            "formal/<scenario>/probe_summary.json",
            "formal/<scenario>/gpu_compute_apps_after.txt",
            "formal/<scenario>/process_residue_after.txt",
        ],
        "stop_rules": [
            "preflight_gpu_or_process_residue",
            "vllm_version_not_0.19.0",
            "vllm_source_hash_mismatch",
            "source_hash_changed_between_off_and_on",
            "benchmark_failed_or_token_counts_inexact",
            "service_exited_or_restarted",
            "missing_or_duplicate_iteration_rank_rows",
            "missing_run_source_or_workload_cohort_digest",
            "prometheus_counter_missing_or_reset",
            "overhead_absolute_delta_above_2pct_stop_before_n512",
            "postflight_gpu_or_process_residue",
            "scenario_outside_three_formal_targets",
        ],
    }


def parse_iteration_rows(
    text: str,
    *,
    run_id: str = "",
    source: str = "",
    workload_cohort_digest: str = "",
) -> list[IterationRow]:
    rows: list[IterationRow] = []
    seen: set[tuple[int, int]] = set()
    elapsed_by_rank: dict[int, float] = defaultdict(float)
    tokens_by_rank: dict[int, int] = defaultdict(int)
    for line in text.splitlines():
        match = ITERATION_RE.search(line)
        if not match:
            continue
        rank = int(match.group("rank") or 0)
        elapsed_ms = float(match.group("elapsed_ms"))
        scheduled_tokens = int(match.group("context_tokens")) + int(
            match.group("generation_tokens")
        )
        start_offset_ms = elapsed_by_rank[rank]
        end_offset_ms = start_offset_ms + elapsed_ms
        progress_start_tokens = tokens_by_rank[rank]
        cumulative_scheduled_tokens = progress_start_tokens + scheduled_tokens
        row = IterationRow(
            rank_id=rank,
            run_id=run_id,
            source=source,
            workload_cohort_digest=workload_cohort_digest,
            iteration_seq=int(match.group("iteration")),
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
                0, (cumulative_scheduled_tokens - 1) // PROGRESS_WINDOW_TOKENS
            ),
        )
        key = (row.rank_id, row.iteration_seq)
        if key in seen:
            raise ValueError(f"duplicate_rank_iteration:{key}")
        if not math.isfinite(row.iteration_elapsed_ms) or row.iteration_elapsed_ms <= 0:
            raise ValueError(
                f"invalid_iteration_elapsed:{key}:{row.iteration_elapsed_ms}"
            )
        seen.add(key)
        rows.append(row)
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


def summarize_ranks(
    rows: list[IterationRow],
    *,
    metrics_before: str,
    metrics_after: str,
) -> list[dict[str, Any]]:
    run_ids = {row.run_id for row in rows}
    sources = {row.source for row in rows}
    workload_digests = {row.workload_cohort_digest for row in rows}
    if len(run_ids) != 1 or "" in run_ids or len(sources) != 1:
        raise ValueError("missing_run_id_or_source")
    run_id = next(iter(run_ids))
    source = next(iter(sources))
    if source not in {"real", "sim"}:
        raise ValueError("missing_run_id_or_source")
    if len(workload_digests) != 1:
        raise ValueError("invalid_workload_cohort_digest")
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


def evaluate_overhead_gate(
    *,
    off_output_tok_s: float,
    on_output_tok_s: float,
) -> dict[str, float | bool | str]:
    if (
        not math.isfinite(off_output_tok_s)
        or not math.isfinite(on_output_tok_s)
        or off_output_tok_s <= 0
        or on_output_tok_s <= 0
    ):
        raise ValueError("overhead_gate_nonpositive_or_nonfinite_throughput")
    absolute_delta_pct = abs(on_output_tok_s / off_output_tok_s - 1.0) * 100.0
    return {
        "schema": SCHEMA,
        "gate": "dp2_bt65536_stock_iteration_detail_overhead",
        "off_output_tok_s": off_output_tok_s,
        "on_output_tok_s": on_output_tok_s,
        "absolute_delta_pct": absolute_delta_pct,
        "max_absolute_delta_pct": MAX_OVERHEAD_PCT,
        "passed": absolute_delta_pct <= MAX_OVERHEAD_PCT,
    }


def require_formal_collection(gate: dict[str, Any]) -> None:
    if gate.get("passed") is not True:
        raise ValueError("overhead_gate_failed_stop_before_n512")


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
    throughput = float(result["output_tok_s"])
    if not math.isfinite(throughput) or throughput <= 0:
        raise ValueError(f"overhead_benchmark_invalid_throughput:{root}")
    return throughput


def validate_overhead_artifacts(off_dir: Path, on_dir: Path) -> dict[str, Any]:
    off_meta = _load_json(off_dir / "meta.json")
    on_meta = _load_json(on_dir / "meta.json")
    expected_meta = {
        "scenario": OVERHEAD_SCENARIO,
        "num_prompts": 128,
        "concurrency": 128,
        "vllm_version": "0.19.0",
        "source": "real",
    }
    for key, expected in expected_meta.items():
        if off_meta.get(key) != expected or on_meta.get(key) != expected:
            raise ValueError(f"overhead_meta_mismatch:{key}")
    if off_meta.get("probe_mode") != "off" or on_meta.get("probe_mode") != "on":
        raise ValueError("overhead_probe_mode_mismatch")
    for meta in (off_meta, on_meta):
        if not meta.get("run_id"):
            raise ValueError("missing_run_id_or_source")
        if not re.fullmatch(r"[0-9a-f]{64}", str(meta.get("workload_cohort_digest", ""))):
            raise ValueError("invalid_workload_cohort_digest")
    if off_meta["workload_cohort_digest"] != on_meta["workload_cohort_digest"]:
        raise ValueError("workload_cohort_digest_mismatch")

    off_hashes = _load_source_hashes(off_dir / "source.sha256")
    on_hashes = _load_source_hashes(on_dir / "source.sha256")
    if off_hashes != SOURCE_FILES or on_hashes != SOURCE_FILES:
        raise ValueError("vllm_source_hash_mismatch")
    if off_hashes != on_hashes:
        raise ValueError("source_hash_changed_between_off_and_on")
    _validate_residue(off_dir)
    _validate_residue(on_dir)

    off_bench = _load_json(off_dir / "bench_result.json")
    on_bench = _load_json(on_dir / "bench_result.json")
    off_throughput = _validate_overhead_benchmark(off_bench, root=off_dir)
    on_throughput = _validate_overhead_benchmark(on_bench, root=on_dir)

    serve_log = on_dir / "serve.log.gz"
    if not serve_log.exists():
        serve_log = on_dir / "serve.log"
    if not serve_log.exists():
        raise ValueError(f"missing_probe_serve_log:{on_dir}")
    metrics_before = on_dir / "metrics_before.prom"
    metrics_after = on_dir / "metrics_after.prom"
    if not metrics_before.exists() or not metrics_after.exists():
        raise ValueError(f"missing_probe_metrics:{on_dir}")
    rank_summary = summarize_ranks(
        parse_iteration_rows(
            _read_text(serve_log),
            run_id=str(on_meta["run_id"]),
            source=str(on_meta["source"]),
            workload_cohort_digest=str(on_meta["workload_cohort_digest"]),
        ),
        metrics_before=metrics_before.read_text(encoding="utf-8"),
        metrics_after=metrics_after.read_text(encoding="utf-8"),
    )
    if [row["rank_id"] for row in rank_summary] != [0, 1]:
        raise ValueError("overhead_probe_missing_dp_rank")

    gate = evaluate_overhead_gate(
        off_output_tok_s=off_throughput,
        on_output_tok_s=on_throughput,
    )
    gate["source_hash_match"] = True
    gate["cleanup"] = True
    gate["formal_collection_allowed"] = gate["passed"]
    gate["rank_summary"] = rank_summary
    return gate


def _read_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as stream:
            return stream.read()
    return path.read_text(encoding="utf-8", errors="replace")


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
    summarize.add_argument("--serve-log", type=Path, required=True)
    summarize.add_argument("--metrics-before", type=Path, required=True)
    summarize.add_argument("--metrics-after", type=Path, required=True)
    summarize.add_argument("--run-id", required=True)
    summarize.add_argument("--source", choices=("real", "sim"), required=True)
    summarize.add_argument("--workload-cohort-digest", required=True)
    summarize.add_argument("--output-iteration-csv", type=Path, required=True)
    summarize.add_argument("--output-json", type=Path, required=True)
    summarize.add_argument("--output-csv", type=Path, required=True)

    gate = subparsers.add_parser("gate", help="validate off/on artifact directories")
    gate.add_argument("--off-dir", type=Path, required=True)
    gate.add_argument("--on-dir", type=Path, required=True)
    gate.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "plan":
        _write_json(args.output_json, build_run_plan())
        return 0
    if args.command == "summarize":
        rows = parse_iteration_rows(
            _read_text(args.serve_log),
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
    gate = validate_overhead_artifacts(args.off_dir, args.on_dir)
    _write_json(args.output_json, gate)
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
