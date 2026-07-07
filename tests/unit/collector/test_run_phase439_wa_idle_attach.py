from __future__ import annotations

from pathlib import Path


RUNNER = Path("collector/vllm/run_phase439_wa_idle_attach.sh")


def test_phase439_runner_uses_idle_attach_and_smoke_gate() -> None:
    text = RUNNER.read_text()

    assert "PHASE439_GATE1_SMOKE" in text
    assert "PHASE439_GATE2_WA" in text
    assert "profile_start_idle" in text
    assert "start_bench" in text
    assert "waiting_for_steady_running" in text
    assert "steady_record_seconds" in text
    assert "TRACE_FLUSH_WAIT_SECONDS=\"${TRACE_FLUSH_WAIT_SECONDS:-90}\"" in text


def test_phase439_runner_exports_required_gpu_environment() -> None:
    text = RUNNER.read_text()

    assert "VLLM_ENABLE_CUDA_COMPATIBILITY" in text
    assert "/usr/local/cuda-12.9/compat" in text
    assert "/usr/local/nvidia/bin" in text
    assert "/nccl/lib" in text
