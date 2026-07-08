from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_validate_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "validate_cb_simulator.py"
    spec = importlib.util.spec_from_file_location("validate_cb_simulator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase449_dp2_8k2k_uses_clean_log_kv_capacity() -> None:
    validate = _load_validate_module()
    capacity = validate.PHASE397K_KV_CAPACITY_BY_SCENARIO["K2.5-tp4ep8dp2-8k2k"]

    assert capacity.kv_cache_tokens == 458_128
    assert capacity.num_gpu_blocks == 28_633
    assert "phase446_b2b_event_timing" in capacity.serve_log


if __name__ == "__main__":
    test_phase449_dp2_8k2k_uses_clean_log_kv_capacity()
