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


def _iteration_log() -> str:
    return "\n".join(
        [
            "(EngineCore_DP0 pid=10) INFO [core.py:359] Iteration(0): "
            "1 context requests, 8000 context tokens, 0 generation requests, "
            "0 generation tokens, iteration elapsed time: 100.0 ms",
            "(EngineCore_DP0 pid=10) INFO [core.py:359] Iteration(1): "
            "0 context requests, 0 context tokens, 8 generation requests, "
            "8 generation tokens, iteration elapsed time: 20.0 ms",
            "(EngineCore_DP1 pid=11) INFO [core.py:359] Iteration(0): "
            "1 context requests, 7900 context tokens, 4 generation requests, "
            "4 generation tokens, iteration elapsed time: 120.0 ms",
            "(EngineCore_DP1 pid=11) INFO [core.py:359] Iteration(1): "
            "0 context requests, 0 context tokens, 8 generation requests, "
            "8 generation tokens, iteration elapsed time: 22.0 ms",
        ]
    ) + "\n"


def _preemption_metrics(*, dp0: int, dp1: int) -> str:
    return "\n".join(
        [
            f'vllm:num_preemptions_total{{engine="0",model_name="kimi-k2.5"}} {dp0}',
            f'vllm:num_preemptions_total{{engine="1",model_name="kimi-k2.5"}} {dp1}',
        ]
    ) + "\n"


def _cohort_digest() -> str:
    return "a" * 64


def test_plan_is_default_off_and_gates_formal_collection() -> None:
    phase466 = _load_module()

    plan = phase466.build_run_plan()

    assert plan["probe_default"] == "off"
    assert plan["implementation"] == "stock_vllm_iteration_details_no_source_patch"
    assert plan["comparison_semantics"]["tp8_vs_tp4dp2"] == (
        "topology_control_not_pure_dp_control"
    )
    assert plan["overhead_gate"] == {
        "scenario": "K2.5-tp4ep8dp2-8k2k-bt65536",
        "num_prompts": 128,
        "concurrency": 128,
        "max_absolute_delta_pct": 2.0,
        "stop_before_n512_on_failure": True,
    }

    runs = {run["id"]: run for run in plan["runs"]}
    assert "--enable-logging-iteration-details" not in runs["overhead-off"]["serve_argv"]
    assert "--enable-logging-iteration-details" in runs["overhead-on"]["serve_argv"]
    assert runs["overhead-off"]["num_prompts"] == 128
    assert runs["overhead-on"]["num_prompts"] == 128
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


def test_parser_summarizes_iteration_rank_without_request_identity() -> None:
    phase466 = _load_module()

    rows = phase466.parse_iteration_rows(
        _iteration_log(),
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


def test_parser_rejects_duplicate_rank_iteration() -> None:
    phase466 = _load_module()
    duplicated = _iteration_log() + _iteration_log().splitlines()[0] + "\n"

    with pytest.raises(ValueError, match="duplicate_rank_iteration"):
        phase466.parse_iteration_rows(
            duplicated,
            run_id="unit-real",
            source="real",
            workload_cohort_digest=_cohort_digest(),
        )


def test_overhead_gate_is_symmetric_and_blocks_formal_runs() -> None:
    phase466 = _load_module()

    passed = phase466.evaluate_overhead_gate(
        off_output_tok_s=100.0,
        on_output_tok_s=98.5,
    )
    assert passed["absolute_delta_pct"] == pytest.approx(1.5)
    assert passed["passed"] is True
    phase466.require_formal_collection(passed)

    failed = phase466.evaluate_overhead_gate(
        off_output_tok_s=100.0,
        on_output_tok_s=102.1,
    )
    assert failed["absolute_delta_pct"] == pytest.approx(2.1)
    assert failed["passed"] is False
    with pytest.raises(ValueError, match="overhead_gate_failed_stop_before_n512"):
        phase466.require_formal_collection(failed)


def test_artifact_validation_requires_hash_match_and_empty_residue(tmp_path: Path) -> None:
    phase466 = _load_module()
    off = tmp_path / "off"
    on = tmp_path / "on"
    off.mkdir()
    on.mkdir()
    for root, mode, throughput in ((off, "off", 100.0), (on, "on", 99.0)):
        (root / "meta.json").write_text(
            json.dumps(
                {
                    "run_id": f"overhead-{mode}",
                    "scenario": "K2.5-tp4ep8dp2-8k2k-bt65536",
                    "probe_mode": mode,
                    "source": "real",
                    "num_prompts": 128,
                    "concurrency": 128,
                    "vllm_version": "0.19.0",
                    "workload_cohort_digest": phase466.workload_digest(
                        phase466.SCENARIOS[
                            "K2.5-tp4ep8dp2-8k2k-bt65536"
                        ],
                        num_prompts=128,
                        concurrency=128,
                    ),
                }
            )
            + "\n"
        )
        (root / "bench_result.json").write_text(
            json.dumps(
                {
                    "output_tok_s": throughput,
                    "ok_requests": 128,
                    "failed_requests": 0,
                    "total_prompt_tokens": 1024000,
                    "total_completion_tokens": 256000,
                }
            )
            + "\n"
        )
        (root / "source.sha256").write_text(
            "896730e749cbcabb487ce50c703974594d197fc31a1d3b26fe096197d142d2d5  "
            "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py\n"
            "9f3da2dfce94963e1e1cefc156e4239120f79c6eaf824b2829cac0ee7736b58a  "
            "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py\n"
        )
        (root / "gpu_compute_apps_after.txt").write_text("")
        (root / "process_residue_after.txt").write_text("")
    (on / "serve.log").write_text(_iteration_log())
    (on / "metrics_before.prom").write_text(_preemption_metrics(dp0=0, dp1=0))
    (on / "metrics_after.prom").write_text(_preemption_metrics(dp0=1, dp1=2))

    gate = phase466.validate_overhead_artifacts(off, on)
    assert gate["passed"] is True
    assert gate["source_hash_match"] is True
    assert [row["rank_id"] for row in gate["rank_summary"]] == [0, 1]

    (on / "process_residue_after.txt").write_text("123 vllm\n")
    with pytest.raises(ValueError, match="nonempty_process_residue"):
        phase466.validate_overhead_artifacts(off, on)

    (on / "process_residue_after.txt").write_text("")
    (on / "serve.log").unlink()
    with pytest.raises(ValueError, match="missing_probe_serve_log"):
        phase466.validate_overhead_artifacts(off, on)


def test_plan_records_source_environment_and_artifact_contract() -> None:
    phase466 = _load_module()
    plan = phase466.build_run_plan()

    assert plan["environment"]["VLLM_ENABLE_CUDA_COMPATIBILITY"] == "1"
    assert plan["source_files"] == {
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py": (
            "896730e749cbcabb487ce50c703974594d197fc31a1d3b26fe096197d142d2d5"
        ),
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py": (
            "9f3da2dfce94963e1e1cefc156e4239120f79c6eaf824b2829cac0ee7736b58a"
        ),
    }
    assert "overhead/off/meta.json" in plan["artifact_contract"]
    assert "overhead/on/serve.log.gz" in plan["artifact_contract"]
    assert "overhead/overhead_gate.json" in plan["artifact_contract"]
    assert "formal/<scenario>/rank_summary.csv" in plan["artifact_contract"]
    assert "formal/<scenario>/iteration_rows.csv" in plan["artifact_contract"]
    assert plan["stop_rules"][0] == "preflight_gpu_or_process_residue"
    assert "cumulative scheduled-token progress windows" in plan["sampling"]["alignment"]


def test_summary_rejects_missing_alignment_identity() -> None:
    phase466 = _load_module()
    rows = phase466.parse_iteration_rows(_iteration_log())

    with pytest.raises(ValueError, match="missing_run_id_or_source"):
        phase466.summarize_ranks(
            rows,
            metrics_before=_preemption_metrics(dp0=0, dp1=0),
            metrics_after=_preemption_metrics(dp0=1, dp1=1),
        )
