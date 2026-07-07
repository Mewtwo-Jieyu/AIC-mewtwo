from __future__ import annotations

from pathlib import Path


def test_phase442_b2_runner_sets_required_hardware_env_and_patch_lifecycle():
    script = Path(__file__).resolve().parents[3] / "collector" / "vllm" / "run_phase442_b2_event_timing.sh"
    text = script.read_text(encoding="utf-8")

    assert "VLLM_ENABLE_CUDA_COMPATIBILITY" in text
    assert "/usr/local/cuda-12.9/compat" in text
    assert "/nccl/lib" in text
    assert "phase442_event_timing_patch.py" in text
    assert "--apply" in text
    assert "--restore" in text
    assert "AIC_PHASE442_EVENT_JSONL" in text


def test_phase442_b2_runner_has_overhead_gate_without_cupti_profiler():
    script = Path(__file__).resolve().parents[3] / "collector" / "vllm" / "run_phase442_b2_event_timing.sh"
    text = script.read_text(encoding="utf-8")

    assert "OVERHEAD_MAX_PCT" in text
    assert "output_tok_s" in text
    assert "overhead_pct" in text
    assert "start_profile" not in text
    assert "VLLM_TORCH_PROFILER_DIR" not in text
