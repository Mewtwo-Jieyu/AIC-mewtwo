from __future__ import annotations

from pathlib import Path


def _script_text() -> str:
    script = Path(__file__).resolve().parents[3] / "collector" / "vllm" / "run_phase446_b2b_event_timing.sh"
    return script.read_text(encoding="utf-8")


def test_phase446_b2b_runner_sets_required_hardware_env_and_patch_lifecycle():
    text = _script_text()

    assert "VLLM_ENABLE_CUDA_COMPATIBILITY" in text
    assert "/usr/local/cuda-12.9/compat" in text
    assert "/nccl/lib" in text
    assert "phase446_b2b_event_timing_patch.py" in text
    assert "--apply" in text
    assert "--restore" in text
    assert "AIC_PHASE446_EVENT_JSONL" in text
    assert "AIC_PHASE446_FLUSH_INTERVAL" in text
    assert "PRE_STOP_HOOK_SCRIPT" in text
    assert "USR1" in text


def test_phase446_b2b_runner_has_overhead_and_two_collection_modes_without_cupti():
    text = _script_text()

    assert "OVERHEAD_MAX_PCT" in text
    assert "output_tok_s" in text
    assert "overhead_pct" in text
    assert "event_on_8k" in text
    assert "event_on_32k" in text
    assert "start_profile" not in text
    assert "VLLM_TORCH_PROFILER_DIR" not in text


if __name__ == "__main__":
    test_phase446_b2b_runner_sets_required_hardware_env_and_patch_lifecycle()
    test_phase446_b2b_runner_has_overhead_and_two_collection_modes_without_cupti()
