from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "analyze_phase447_sequence_fingerprint.py"
    spec = importlib.util.spec_from_file_location("analyze_phase447_sequence_fingerprint", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase447_counts_phase_pairs_and_mixed_runs() -> None:
    phase447 = _load_module()
    phase446 = phase447.phase446
    steps = [
        phase446.EngineStep("0", 0, 0, 42, 40.0, num_tokens_unpadded=42, cudagraph_mode="FULL"),
        phase446.EngineStep("1", 0, 0, 42, 41.0, num_tokens_unpadded=42, cudagraph_mode="FULL"),
        phase446.EngineStep("0", 1, 0, 42, 1072.0, generation_tokens=42, num_tokens_unpadded=42, cudagraph_mode="FULL"),
        phase446.EngineStep("1", 1, 0, 42, 1131.0, generation_tokens=8000, num_tokens_unpadded=8000),
        phase446.EngineStep("0", 2, 0, 43, 1130.0, generation_tokens=8000, num_tokens_unpadded=8000),
        phase446.EngineStep("1", 2, 0, 42, 1070.0, generation_tokens=42, num_tokens_unpadded=42, cudagraph_mode="FULL"),
        phase446.EngineStep("0", 3, 0, 42, 40.0, num_tokens_unpadded=42, cudagraph_mode="FULL"),
        phase446.EngineStep("1", 3, 0, 42, 41.0, num_tokens_unpadded=42, cudagraph_mode="FULL"),
    ]

    fingerprint = phase447.build_fingerprint(steps, scenario="unit")

    joint = {
        (row["phase_pair"], row["count"])
        for row in fingerprint.phase_joint_rows
    }
    assert ("decode+decode", 2) in joint
    assert ("decode+mixed_prefill", 1) in joint
    assert ("mixed_prefill+decode", 1) in joint
    assert fingerprint.wave_rows[0]["wave_mixed_steps"] == 2
    assert fingerprint.engine_summary_rows[0]["decode_run_p50"] == 1
    assert fingerprint.mixed_distribution_rows[0]["decode_batch"] == 43


if __name__ == "__main__":
    test_phase447_counts_phase_pairs_and_mixed_runs()
