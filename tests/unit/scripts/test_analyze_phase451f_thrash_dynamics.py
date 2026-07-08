from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "analyze_phase451f_thrash_dynamics.py"
    spec = importlib.util.spec_from_file_location("analyze_phase451f_thrash_dynamics", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_hypothesis_gate_selects_wave_overshoot_when_events_cluster_near_mixed() -> None:
    phase451f = _load_module()

    verdict = phase451f.decide_hypothesis(
        near_mixed_share=0.91,
        interval_cv=1.8,
        counterfactual_preemptions_per_request=0.0,
        block_deficit_detected=False,
    )

    assert verdict["hypothesis"] == "H2_wave_overshoot"
    assert verdict["fix_gate"] == "blocked_pending_wave_model"


def test_hypothesis_gate_selects_block_deficit_when_staggered_still_preempts() -> None:
    phase451f = _load_module()

    verdict = phase451f.decide_hypothesis(
        near_mixed_share=0.20,
        interval_cv=0.15,
        counterfactual_preemptions_per_request=0.8,
        block_deficit_detected=True,
    )

    assert verdict["hypothesis"] == "H1_block_deficit"
    assert verdict["fix_gate"] == "ready_for_block_accounting_fix"


def test_distance_to_nearest_mixed_uses_same_replica_iterations() -> None:
    phase451f = _load_module()

    distances = phase451f.distances_to_nearest_mixed(
        events=[
            {"replica_id": 0, "local_iter": 11},
            {"replica_id": 1, "local_iter": 99},
        ],
        trace=[
            {"replica_id": 0, "local_iter": 10, "is_mixed": True},
            {"replica_id": 1, "local_iter": 105, "is_mixed": True},
        ],
    )

    assert distances == [1, 6]


if __name__ == "__main__":
    test_hypothesis_gate_selects_wave_overshoot_when_events_cluster_near_mixed()
    test_hypothesis_gate_selects_block_deficit_when_staggered_still_preempts()
    test_distance_to_nearest_mixed_uses_same_replica_iterations()
