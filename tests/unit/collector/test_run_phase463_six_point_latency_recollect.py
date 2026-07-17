from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "collector/vllm/run_phase463_six_point_latency_recollect.sh"


def test_runner_locks_the_canary_six_formal_points_and_c64_diagnostic() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert '"K2.5-tp8ep8-8k2k-canary-nonstream canary_nonstream 0 8 1 8 8000 2000 8000 262144 128"' in text
    assert '"K2.5-tp8ep8-8k2k formal 1 8 1 8 8000 2000 8000 262144 128"' in text
    assert '"K2.5-tp8ep8-32k3k formal 1 8 1 8 32000 3000 32000 262144 128"' in text
    assert '"K2.5-tp4ep8dp2-8k2k formal 1 4 2 8 8000 2000 8000 131072 128"' in text
    assert '"K2.5-tp4ep8dp2-32k3k formal 1 4 2 8 32000 3000 32000 262144 128"' in text
    assert '"K2.5-tp8ep8-8k2k-bt65536 formal 1 8 1 8 8000 2000 65536 262144 128"' in text
    assert '"K2.5-tp4ep8dp2-8k2k-bt65536 formal 1 4 2 8 8000 2000 65536 131072 128"' in text
    assert '"K2.5-tp4ep8dp2-32k3k-c64-diagnostic diagnostic 1 4 2 8 32000 3000 32000 262144 64"' in text


def test_runner_collects_exact_metrics_stream_records_and_gpu_telemetry() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert "metrics_before.prom" in text
    assert "metrics_after.prom" in text
    assert "metrics.jsonl" in text
    assert "gpu.csv" in text
    assert "--stream" in text
    assert "--skip-prompt-token-id-probe" in text
    assert "--query-gpu=timestamp,index,utilization.gpu,power.draw,memory.used" in text
    assert 'SAMPLE_INTERVAL_S="${SAMPLE_INTERVAL_S:-2}"' in text


def test_runner_fails_each_point_through_the_phase463_analyzer_and_canary_gate() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert "analyze_phase463_six_point_latency_recollect.py" in text
    assert "--scenario-dir" in text
    assert "stream_canary_throughput_delta" in text
    assert "POINT_FAILED" in text
    assert "set -Eeuo pipefail" in text
    assert "trap point_failed ERR" in text
    assert 'if ! run_point "${idx}" "${spec}"' not in text
    assert "wait_for_gpu_drain" in text
    assert "process_residue_after" in text
    assert "FORCE" not in text
    assert "warmup_requests_must_be_zero" in text


def test_runner_is_measurement_only() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert "diagnostic_only" in text
    assert "valid_for_default" in text
    assert "perf_database" in text
    assert "default_readiness" in text
    assert "git commit" not in text
    assert "git push" not in text
