from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "analyze_phase447_sim_fingerprint_gate.py"
    spec = importlib.util.spec_from_file_location("analyze_phase447_sim_fingerprint_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase447_gate_fails_when_sim_mixed_batch_outside_real_band() -> None:
    phase447 = _load_module()
    verdict = phase447.evaluate_mixed_batch_gate(
        sim_batches=[62, 62, 63],
        real_low=34,
        real_high=52,
    )

    assert not verdict.passed
    assert verdict.sim_p50 == 62
    assert verdict.reason == "sim_mixed_decode_batch_outside_real_band"


if __name__ == "__main__":
    test_phase447_gate_fails_when_sim_mixed_batch_outside_real_band()
