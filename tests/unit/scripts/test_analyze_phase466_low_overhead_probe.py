from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase466_low_overhead_probe.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_phase466_low_overhead_probe", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _iteration_message(
    iteration: int,
    *,
    context_requests: int,
    context_tokens: int,
    generation_requests: int,
    generation_tokens: int,
    elapsed_ms: float,
) -> str:
    return (
        f"Iteration({iteration}): {context_requests} context requests, "
        f"{context_tokens} context tokens, {generation_requests} generation requests, "
        f"{generation_tokens} generation tokens, iteration elapsed time: "
        f"{elapsed_ms} ms"
    )


def _rank_records() -> dict[int, list[dict[str, object]]]:
    return {
        0: [
            {
                "rank": 0,
                "pid": 10,
                "process_name": "EngineCore_DP0",
                "message": _iteration_message(
                    0,
                    context_requests=1,
                    context_tokens=8000,
                    generation_requests=0,
                    generation_tokens=0,
                    elapsed_ms=100.0,
                ),
            },
            {
                "rank": 0,
                "pid": 10,
                "process_name": "EngineCore_DP0",
                "message": _iteration_message(
                    1,
                    context_requests=0,
                    context_tokens=0,
                    generation_requests=8,
                    generation_tokens=8,
                    elapsed_ms=20.0,
                ),
            },
        ],
        1: [
            {
                "rank": 1,
                "pid": 11,
                "process_name": "EngineCore_DP1",
                "message": _iteration_message(
                    0,
                    context_requests=1,
                    context_tokens=7900,
                    generation_requests=4,
                    generation_tokens=4,
                    elapsed_ms=120.0,
                ),
            },
            {
                "rank": 1,
                "pid": 11,
                "process_name": "EngineCore_DP1",
                "message": _iteration_message(
                    1,
                    context_requests=0,
                    context_tokens=0,
                    generation_requests=8,
                    generation_tokens=8,
                    elapsed_ms=22.0,
                ),
            },
        ],
    }


def _write_rank_logs(
    root: Path, records: dict[int, list[dict[str, object]]] | None = None
) -> Path:
    rank_dir = root / "rank_logs"
    rank_dir.mkdir(parents=True)
    for rank, rows in (records or _rank_records()).items():
        (rank_dir / f"rank-{rank}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    return rank_dir


def _preemption_metrics(*, dp0: int, dp1: int) -> str:
    return "\n".join(
        [
            f'vllm:num_preemptions_total{{engine="0",model_name="kimi-k2.5"}} {dp0}',
            f'vllm:num_preemptions_total{{engine="1",model_name="kimi-k2.5"}} {dp1}',
        ]
    ) + "\n"


def _cohort_digest() -> str:
    return "a" * 64


def _identity_meta(prompt_digest: str = "d" * 64) -> dict[str, object]:
    return {
        "model_identity_schema": "phase466_flat_model_fingerprint_v1",
        "model_identity_sha256": "b" * 64,
        "warmup_prompt_cohort_sha256": "c" * 64,
        "prompt_cohort_sha256": prompt_digest,
    }


def _source_hash_text() -> str:
    return (
        "233720900b1207434e4824bbddcf86dba0dc8557a0bfe78a639545462a9e85b5  "
        "/usr/local/lib/python3.12/dist-packages/vllm/logger.py\n"
        "896730e749cbcabb487ce50c703974594d197fc31a1d3b26fe096197d142d2d5  "
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py\n"
        "9f3da2dfce94963e1e1cefc156e4239120f79c6eaf824b2829cac0ee7736b58a  "
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py\n"
    )


def _empty_rank_log_identity() -> dict[str, object]:
    return {"schema": "phase466_rank_log_identity_v1", "files": {}}


def _pass_gate(phase466):
    return phase466.evaluate_paired_overhead_gate(
        [
            {
                "pair_id": f"pair-{index:02d}",
                "off_output_tok_s": 100.0,
                "on_output_tok_s": 100.0,
            }
            for index in range(1, 7)
        ]
    )


def test_plan_is_default_off_and_gates_formal_collection() -> None:
    phase466 = _load_module()

    plan = phase466.build_run_plan()

    assert plan["probe_default"] == "off"
    assert plan["implementation"] == "rank_local_logging_handler_no_vllm_source_patch"
    assert plan["comparison_semantics"]["tp8_vs_tp4dp2"] == (
        "topology_control_not_pure_dp_control"
    )
    assert plan["overhead_gate"]["scenario"] == "K2.5-tp4ep8dp2-8k2k-bt65536"
    assert plan["overhead_gate"]["pair_count"] == 6
    assert plan["overhead_gate"]["equivalence_ratio_bounds"] == [0.98, 1.02]

    runs = {run["id"]: run for run in plan["runs"]}
    assert "--enable-logging-iteration-details" not in runs["overhead-pair-01-off"]["serve_argv"]
    assert "--enable-logging-iteration-details" in runs["overhead-pair-01-on"]["serve_argv"]
    assert runs["overhead-pair-01-off"]["num_prompts"] == 128
    assert runs["overhead-pair-01-on"]["num_prompts"] == 128
    assert all(run["source"] == "real" for run in runs.values())
    formal = [run for run in plan["runs"] if run["stage"] == "formal"]
    assert {run["scenario"] for run in formal} == {
        "K2.5-tp4ep8dp2-32k3k",
        "K2.5-tp8ep8-8k2k-bt65536",
        "K2.5-tp4ep8dp2-8k2k-bt65536",
    }
    assert all(run["num_prompts"] == 512 for run in formal)
    assert all(run["concurrency"] == 128 for run in formal)
    assert all(run["requires_overhead_gate"] for run in formal)
    assert all("--enable-logging-iteration-details" in run["serve_argv"] for run in formal)
    assert all(run["source"] == "real" for run in plan["runs"])
    assert all(len(run["workload_cohort_digest"]) == 64 for run in plan["runs"])


def test_parser_summarizes_iteration_rank_without_request_identity(
    tmp_path: Path,
) -> None:
    phase466 = _load_module()

    rows = phase466.parse_iteration_rows(
        _write_rank_logs(tmp_path),
        expected_ranks={0, 1},
        run_id="unit-real",
        source="real",
        workload_cohort_digest=_cohort_digest(),
    )
    summaries = phase466.summarize_ranks(
        rows,
        metrics_before=_preemption_metrics(dp0=3, dp1=5),
        metrics_after=_preemption_metrics(dp0=4, dp1=7),
    )

    assert set(rows[0].__dataclass_fields__) == {
        "rank_id",
        "rank_scope",
        "run_id",
        "source",
        "workload_cohort_digest",
        "iteration_seq",
        "progress_start_tokens",
        "progress_end_tokens",
        "prefill_request_count",
        "scheduled_prefill_tokens",
        "decode_request_count",
        "scheduled_decode_tokens",
        "iteration_elapsed_ms",
        "iteration_start_offset_ms",
        "iteration_end_offset_ms",
        "cumulative_scheduled_tokens",
        "progress_window_id",
    }
    assert rows[0].progress_start_tokens == 0
    assert rows[0].rank_scope == "dp_rank"
    assert rows[0].progress_end_tokens == 8000
    assert rows[1].progress_start_tokens == 8000
    assert rows[2].progress_start_tokens == 0
    by_rank = {row["rank_id"]: row for row in summaries}
    assert by_rank[0]["run_id"] == "unit-real"
    assert by_rank[0]["source"] == "real"
    assert by_rank[0]["workload_cohort_digest"] == _cohort_digest()
    assert by_rank[0]["progress_end_tokens"] == 8008
    assert by_rank[0]["iteration_count"] == 2
    assert by_rank[0]["context_tokens_total"] == 8000
    assert by_rank[0]["generation_tokens_total"] == 8
    assert by_rank[0]["mixed_iteration_count"] == 0
    assert by_rank[0]["preemptions"] == 1
    assert by_rank[1]["mixed_iteration_count"] == 1
    assert by_rank[1]["preemptions"] == 2
    assert by_rank[1]["elapsed_ms_sum"] == pytest.approx(142.0)
    assert by_rank[1]["run_id"] == "unit-real"
    assert by_rank[1]["source"] == "real"
    assert by_rank[1]["workload_cohort_digest"] == _cohort_digest()
    assert by_rank[1]["iteration_start_offset_ms"] == pytest.approx(0.0)
    assert by_rank[1]["iteration_end_offset_ms"] == pytest.approx(142.0)
    assert by_rank[1]["cumulative_scheduled_tokens"] == 7912
    assert by_rank[1]["elapsed_ms_per_scheduled_token"] == pytest.approx(
        142.0 / (7900 + 4 + 8)
    )


def test_parser_rejects_duplicate_rank_iteration(tmp_path: Path) -> None:
    phase466 = _load_module()
    records = _rank_records()
    records[0].append(dict(records[0][0]))

    with pytest.raises(ValueError, match="rank_iteration_not_contiguous"):
        phase466.parse_iteration_rows(
            _write_rank_logs(tmp_path, records),
            expected_ranks={0, 1},
            run_id="unit-real",
            source="real",
            workload_cohort_digest=_cohort_digest(),
        )


def test_parser_rejects_missing_or_extra_rank_file(tmp_path: Path) -> None:
    phase466 = _load_module()
    rank_dir = _write_rank_logs(tmp_path)
    (rank_dir / "rank-1.jsonl").unlink()

    with pytest.raises(ValueError, match="rank_log_file_set_mismatch"):
        phase466.parse_iteration_rows(
            rank_dir,
            expected_ranks={0, 1},
            run_id="unit-real",
            source="real",
            workload_cohort_digest=_cohort_digest(),
        )

    (rank_dir / "rank-1.jsonl").write_text("", encoding="utf-8")
    (rank_dir / "rank-2.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="rank_log_file_set_mismatch"):
        phase466.parse_iteration_rows(
            rank_dir,
            expected_ranks={0, 1},
            run_id="unit-real",
            source="real",
            workload_cohort_digest=_cohort_digest(),
        )


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda records: records[0][0].update(process_name="EngineCore_DP1"),
            "rank_log_process_name_mismatch",
        ),
        (
            lambda records: records[0][1].update(pid=99),
            "rank_log_pid_drift",
        ),
        (
            lambda records: records[0][1].update(
                message=str(records[0][1]["message"]).replace(
                    "Iteration(1)", "Iteration(2)"
                )
            ),
            "rank_iteration_not_contiguous",
        ),
    ],
)
def test_parser_rejects_rank_identity_or_sequence_drift(
    tmp_path: Path, mutate, error: str
) -> None:
    phase466 = _load_module()
    records = _rank_records()
    mutate(records)

    with pytest.raises(ValueError, match=error):
        phase466.parse_iteration_rows(
            _write_rank_logs(tmp_path, records),
            expected_ranks={0, 1},
            run_id="unit-real",
            source="real",
            workload_cohort_digest=_cohort_digest(),
        )


def test_single_dp_requires_engine_core_without_dp_suffix(tmp_path: Path) -> None:
    phase466 = _load_module()
    records = {0: _rank_records()[0]}

    with pytest.raises(ValueError, match="rank_log_process_name_mismatch"):
        phase466.parse_iteration_rows(
            _write_rank_logs(tmp_path, records),
            expected_ranks={0},
            run_id="unit-real",
            source="real",
            workload_cohort_digest=_cohort_digest(),
        )

    for row in records[0]:
        row["process_name"] = "EngineCore"
    rows = phase466.parse_iteration_rows(
        _write_rank_logs(tmp_path / "single", records),
        expected_ranks={0},
        run_id="unit-real",
        source="real",
        workload_cohort_digest=_cohort_digest(),
    )
    assert {row.rank_id for row in rows} == {0}


def test_plan_records_source_environment_and_artifact_contract() -> None:
    phase466 = _load_module()
    plan = phase466.build_run_plan()

    assert plan["environment"]["VLLM_ENABLE_CUDA_COMPATIBILITY"] == "1"
    assert plan["source_files"] == {
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
    assert "overhead/<pair-id>/<mode>/meta.json" in plan["artifact_contract"]
    assert "overhead/<pair-id>/<mode>/serve.log.gz" in plan["artifact_contract"]
    assert "overhead/<pair-id>/<mode>/rank_logs/rank-<id>.jsonl" in plan[
        "artifact_contract"
    ]
    assert "overhead/overhead_gate.json" in plan["artifact_contract"]
    assert "tooling.sha256" in plan["artifact_contract"]
    assert "status.json" in plan["artifact_contract"]
    assert "phase466_result.json" in plan["artifact_contract"]
    assert "formal/<scenario>/rank_summary.csv" in plan["artifact_contract"]
    assert "formal/<scenario>/iteration_rows.csv" in plan["artifact_contract"]
    assert plan["stop_rules"][0] == "preflight_gpu_or_process_residue"
    assert "cumulative scheduled-token progress windows" in plan["sampling"]["alignment"]
    assert plan["sampling"]["gpu_health"] == "before/after compute-app residue snapshots"


def test_summary_rejects_missing_alignment_identity(tmp_path: Path) -> None:
    phase466 = _load_module()
    rows = phase466.parse_iteration_rows(
        _write_rank_logs(tmp_path), expected_ranks={0, 1}
    )

    with pytest.raises(ValueError, match="missing_run_id_or_source"):
        phase466.summarize_ranks(
            rows,
            metrics_before=_preemption_metrics(dp0=0, dp1=0),
            metrics_after=_preemption_metrics(dp0=1, dp1=1),
        )


def test_v3_plan_uses_six_counterbalanced_pairs_and_exact_warmup() -> None:
    phase466 = _load_module()

    plan = phase466.build_run_plan()
    overhead = [run for run in plan["runs"] if run["stage"] == "overhead"]

    assert plan["schema"] == "phase466_rank_timing_v4"
    assert len(overhead) == 12
    assert [run["probe_mode"] for run in overhead] == [
        "off",
        "on",
        "on",
        "off",
        "off",
        "on",
        "on",
        "off",
        "off",
        "on",
        "on",
        "off",
    ]
    assert {run["pair_id"] for run in overhead} == {
        "pair-01",
        "pair-02",
        "pair-03",
        "pair-04",
        "pair-05",
        "pair-06",
    }
    assert all(run["warmup_num_prompts"] == 128 for run in overhead)
    assert all(run["warmup_concurrency"] == 128 for run in overhead)
    assert all(
        run["measurement_requires_warmup_cutoff"] == run["probe_enabled"]
        for run in overhead
    )


def test_paired_equivalence_gate_has_pass_fail_and_inconclusive_states() -> None:
    phase466 = _load_module()

    passed = phase466.evaluate_paired_overhead_gate(
        [
            {"pair_id": f"pair-{index:02d}", "off_output_tok_s": 100.0, "on_output_tok_s": 100.0}
            for index in range(1, 7)
        ]
    )
    assert passed["status"] == "PASS"
    assert passed["formal_collection_allowed"] is True

    failed = phase466.evaluate_paired_overhead_gate(
        [
            {"pair_id": f"pair-{index:02d}", "off_output_tok_s": 100.0, "on_output_tok_s": 95.0}
            for index in range(1, 7)
        ]
    )
    assert failed["status"] == "FAIL"
    assert failed["formal_collection_allowed"] is False

    inconclusive = phase466.evaluate_paired_overhead_gate(
        [
            {"pair_id": "pair-01", "off_output_tok_s": 100.0, "on_output_tok_s": 97.0},
            {"pair_id": "pair-02", "off_output_tok_s": 100.0, "on_output_tok_s": 103.0},
            {"pair_id": "pair-03", "off_output_tok_s": 100.0, "on_output_tok_s": 97.0},
            {"pair_id": "pair-04", "off_output_tok_s": 100.0, "on_output_tok_s": 103.0},
            {"pair_id": "pair-05", "off_output_tok_s": 100.0, "on_output_tok_s": 97.0},
            {"pair_id": "pair-06", "off_output_tok_s": 100.0, "on_output_tok_s": 103.0},
        ]
    )
    assert inconclusive["status"] == "INCONCLUSIVE"
    assert inconclusive["formal_collection_allowed"] is False


def test_paired_gate_rejects_missing_or_duplicate_preregistered_pair() -> None:
    phase466 = _load_module()
    rows = [
        {"pair_id": f"pair-{index:02d}", "off_output_tok_s": 100.0, "on_output_tok_s": 100.0}
        for index in range(1, 7)
    ]

    with pytest.raises(ValueError, match="overhead_pair_set_mismatch"):
        phase466.evaluate_paired_overhead_gate(rows[:-1])

    rows[-1]["pair_id"] = "pair-05"
    with pytest.raises(ValueError, match="overhead_pair_set_mismatch"):
        phase466.evaluate_paired_overhead_gate(rows)


def test_overhead_root_requires_and_validates_all_six_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    phase466 = _load_module()

    with pytest.raises(ValueError, match="missing_overhead_pair_artifact:pair-01"):
        phase466.validate_overhead_root(tmp_path)

    validated: list[str] = []
    for pair_index in range(1, 7):
        pair_id = f"pair-{pair_index:02d}"
        (tmp_path / "overhead" / pair_id / "off").mkdir(parents=True)
        (tmp_path / "overhead" / pair_id / "on").mkdir(parents=True)

    def validate_pair(off_dir, on_dir, *, off_spec, on_spec):
        assert off_dir.name == "off"
        assert on_dir.name == "on"
        assert off_spec["pair_id"] == on_spec["pair_id"]
        validated.append(off_spec["pair_id"])
        return {
            "pair_id": off_spec["pair_id"],
            "off_output_tok_s": 100.0,
            "on_output_tok_s": 100.0,
        }

    monkeypatch.setattr(phase466, "validate_overhead_pair_artifacts", validate_pair)

    gate = phase466.validate_overhead_root(tmp_path)

    assert gate["status"] == "PASS"
    assert validated == [f"pair-{index:02d}" for index in range(1, 7)]


def test_parser_applies_rank_local_warmup_cutoff(tmp_path: Path) -> None:
    phase466 = _load_module()

    rows = phase466.parse_iteration_rows(
        _write_rank_logs(tmp_path),
        expected_ranks={0, 1},
        run_id="unit-real",
        source="real",
        workload_cohort_digest=_cohort_digest(),
        after_iteration_by_rank={0: 0, 1: 0},
    )

    assert [(row.rank_id, row.iteration_seq) for row in rows] == [(0, 1), (1, 1)]
    assert all(row.iteration_start_offset_ms == 0.0 for row in rows)


def test_parser_retains_zero_token_iteration_wall_time(tmp_path: Path) -> None:
    phase466 = _load_module()
    records = {
        0: [
            {
                "rank": 0,
                "pid": 10,
                "process_name": "EngineCore",
                "message": _iteration_message(
                    1,
                    context_requests=0,
                    context_tokens=0,
                    generation_requests=0,
                    generation_tokens=0,
                    elapsed_ms=7.0,
                ),
            },
            {
                "rank": 0,
                "pid": 10,
                "process_name": "EngineCore",
                "message": _iteration_message(
                    2,
                    context_requests=0,
                    context_tokens=0,
                    generation_requests=8,
                    generation_tokens=8,
                    elapsed_ms=3.0,
                ),
            },
        ]
    }

    rows = phase466.parse_iteration_rows(
        _write_rank_logs(tmp_path, records),
        expected_ranks={0},
        run_id="unit-real",
        source="real",
        workload_cohort_digest=_cohort_digest(),
        after_iteration_by_rank={0: 0},
        through_iteration_by_rank={0: 2},
    )

    assert len(rows) == 2
    assert rows[0].scheduled_prefill_tokens == 0
    assert rows[0].scheduled_decode_tokens == 0
    assert rows[0].iteration_end_offset_ms == 7.0
    assert rows[1].iteration_start_offset_ms == 7.0


def test_iteration_csv_identity_binds_content_and_rank_bounds(tmp_path: Path) -> None:
    phase466 = _load_module()
    path = tmp_path / "iteration_rows.csv"
    rows = [
        {
            "run_id": "real-test",
            "source": "real",
            "rank_id": rank,
            "rank_scope": "dp_rank",
            "workload_cohort_digest": _cohort_digest(),
            "iteration_seq": iteration,
            "progress_start_tokens": 0,
            "progress_end_tokens": 8,
            "iteration_start_offset_ms": 0.0,
            "iteration_end_offset_ms": 1.0,
        }
        for rank, iteration in ((0, 4), (1, 7))
    ]
    phase466._write_csv(path, rows)

    identity = phase466.iteration_csv_identity(path)

    assert identity["row_count"] == 2
    assert identity["run_id"] == "real-test"
    assert identity["workload_cohort_digest"] == _cohort_digest()
    assert identity["rank_bounds"]["0"]["first_iteration_seq"] == 4
    assert identity["rank_bounds"]["1"]["last_iteration_seq"] == 7
    assert len(identity["sha256"]) == 64


def test_overhead_meta_requires_exact_run_identity_and_recomputed_digest(tmp_path: Path) -> None:
    phase466 = _load_module()
    off = tmp_path / "off"
    on = tmp_path / "on"
    off.mkdir()
    on.mkdir()
    plan = phase466.build_run_plan()
    runs = {run["id"]: run for run in plan["runs"]}
    off_spec = runs["overhead-pair-01-off"]
    on_spec = runs["overhead-pair-01-on"]

    for root, spec, throughput in ((off, off_spec, 100.0), (on, on_spec, 100.0)):
        rank_dir = root / "rank_logs"
        if spec["probe_enabled"]:
            _write_rank_logs(root)
            rank_identity = phase466.rank_log_identity(rank_dir, expected_ranks={0, 1})
            cutoffs = {"0": 0, "1": 0}
            ends = {"0": 1, "1": 1}
        else:
            rank_dir.mkdir()
            rank_identity = _empty_rank_log_identity()
            cutoffs = {}
            ends = {}
        meta = phase466.expected_run_meta(spec)
        meta.update(
            {
                "vllm_version": "0.19.0",
                "execution_manifest_sha256": "c" * 64,
                "measurement_start_after_iteration": cutoffs,
                "measurement_end_at_iteration": ends,
                "rank_logs_identity": rank_identity,
                **_identity_meta(),
            }
        )
        (root / "meta.json").write_text(json.dumps(meta) + "\n")
        (root / "bench_result.json").write_text(
            json.dumps(
                {
                    "output_tok_s": throughput,
                    "ok_requests": 128,
                    "failed_requests": 0,
                    "total_prompt_tokens": 1024000,
                    "total_completion_tokens": 256000,
                    "prompt_cohort_sha256": "d" * 64,
                }
            )
            + "\n"
        )
        (root / "source.sha256").write_text(_source_hash_text())
        (root / "gpu_compute_apps_after.txt").write_text("")
        (root / "process_residue_after.txt").write_text("")
    (on / "serve.log").write_text(
        "stdout may contain duplicated or prefixless iteration text\n"
    )
    (on / "metrics_before.prom").write_text(_preemption_metrics(dp0=0, dp1=0))
    (on / "metrics_after.prom").write_text(_preemption_metrics(dp0=1, dp1=2))

    result = phase466.validate_overhead_pair_artifacts(off, on, off_spec=off_spec, on_spec=on_spec)
    assert result["pair_id"] == "pair-01"

    rank_zero = on / "rank_logs" / "rank-0.jsonl"
    original_rank_zero = rank_zero.read_text()
    rank_zero.write_text(original_rank_zero.replace("100.0 ms", "101.0 ms"))
    with pytest.raises(ValueError, match="rank_logs_identity_mismatch"):
        phase466.validate_overhead_pair_artifacts(
            off, on, off_spec=off_spec, on_spec=on_spec
        )
    rank_zero.write_text(original_rank_zero)

    (off / "rank_logs" / "rank-0.jsonl").write_text(original_rank_zero)
    with pytest.raises(ValueError, match="off_rank_logs_present"):
        phase466.validate_overhead_pair_artifacts(
            off, on, off_spec=off_spec, on_spec=on_spec
        )
    (off / "rank_logs" / "rank-0.jsonl").unlink()

    bad_meta = json.loads((on / "meta.json").read_text())
    bad_meta["workload_cohort_digest"] = "b" * 64
    (on / "meta.json").write_text(json.dumps(bad_meta) + "\n")
    with pytest.raises(ValueError, match="run_meta_mismatch:workload_cohort_digest"):
        phase466.validate_overhead_pair_artifacts(off, on, off_spec=off_spec, on_spec=on_spec)


def test_formal_collection_requires_complete_passed_v3_gate() -> None:
    phase466 = _load_module()

    phase466.require_formal_collection(_pass_gate(phase466))
    for gate in (
        {"schema": "phase466_stock_probe_v2", "status": "PASS", "pair_count": 6},
        phase466.evaluate_paired_overhead_gate(
            [
                {
                    "pair_id": f"pair-{index:02d}",
                    "off_output_tok_s": 100.0,
                    "on_output_tok_s": 95.0,
                }
                for index in range(1, 7)
            ]
        ),
    ):
        with pytest.raises(ValueError, match="overhead_gate_failed_stop_before_n512"):
            phase466.require_formal_collection(gate)


def test_formal_artifact_validation_enforces_gate_meta_tokens_and_rank_set(tmp_path: Path) -> None:
    phase466 = _load_module()
    plan = phase466.build_run_plan()
    spec = next(
        run
        for run in plan["runs"]
        if run["id"] == "formal-K2.5-tp4ep8dp2-8k2k-bt65536"
    )
    root = tmp_path / "formal"
    root.mkdir()
    gate = _pass_gate(phase466)
    meta = phase466.expected_run_meta(spec)
    meta.update(
        {
            "vllm_version": "0.19.0",
            "execution_manifest_sha256": "c" * 64,
            "measurement_start_after_iteration": {"0": 0, "1": 0},
            "measurement_end_at_iteration": {"0": 1, "1": 1},
            "overhead_gate_sha256": phase466.overhead_gate_digest(gate),
            **_identity_meta(),
        }
    )
    rank_dir = _write_rank_logs(root)
    meta["rank_logs_identity"] = phase466.rank_log_identity(
        rank_dir, expected_ranks={0, 1}
    )
    (root / "meta.json").write_text(json.dumps(meta) + "\n")
    (root / "bench_result.json").write_text(
        json.dumps(
            {
                "output_tok_s": 100.0,
                "ok_requests": 512,
                "failed_requests": 0,
                "total_prompt_tokens": 512 * 8000,
                "total_completion_tokens": 512 * 2000,
                "prompt_cohort_sha256": "d" * 64,
            }
        )
        + "\n"
    )
    (root / "source.sha256").write_text(_source_hash_text())
    (root / "gpu_compute_apps_after.txt").write_text("")
    (root / "process_residue_after.txt").write_text("")
    (root / "serve.log").write_text(
        "(EngineCore_DP0) (EngineCore_DP1) duplicated stdout is ignored\n"
    )
    (root / "metrics_before.prom").write_text(_preemption_metrics(dp0=0, dp1=0))
    (root / "metrics_after.prom").write_text(_preemption_metrics(dp0=1, dp1=2))

    result = phase466.validate_formal_artifacts(root, spec=spec, gate=gate)
    assert result["status"] == "ARTIFACT_VALID"
    assert [row["rank_id"] for row in result["rank_summary"]] == [0, 1]

    bad_gate = dict(gate)
    bad_gate["status"] = "INCONCLUSIVE"
    with pytest.raises(ValueError, match="overhead_gate_failed_stop_before_n512"):
        phase466.validate_formal_artifacts(root, spec=spec, gate=bad_gate)

    bench = json.loads((root / "bench_result.json").read_text())
    bench["total_completion_tokens"] -= 1
    (root / "bench_result.json").write_text(json.dumps(bench) + "\n")
    with pytest.raises(ValueError, match="formal_benchmark_mismatch:total_completion_tokens"):
        phase466.validate_formal_artifacts(root, spec=spec, gate=gate)
