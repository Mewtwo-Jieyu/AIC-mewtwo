from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "analyze_phase451d_preemption_forensics.py"
    spec = importlib.util.spec_from_file_location("analyze_phase451d_preemption_forensics", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_classification_gate_requires_covered_events_and_small_candidate_set() -> None:
    phase451d = _load_module()
    events = [
        {"trigger_state": "DECODING", "victim_preemptions_before": 0, "victim_recently_admitted": False},
        {"trigger_state": "DECODING", "victim_preemptions_before": 1, "victim_recently_admitted": False},
        {"trigger_state": "DECODING", "victim_preemptions_before": 2, "victim_recently_admitted": False},
        {"trigger_state": "PREFILLING", "victim_preemptions_before": 0, "victim_recently_admitted": True},
        {"trigger_state": "UNKNOWN", "victim_preemptions_before": 0, "victim_recently_admitted": False},
    ]

    summary = phase451d.classify_preemption_events(events)
    gate = phase451d.classification_gate(summary, coverage_target=0.80, max_candidates=3)

    assert summary["total_events"] == 5
    assert summary["covered_events"] == 4
    assert summary["coverage"] == 0.8
    assert summary["category_counts"]["thrash_repeat_victim"] == 2
    assert summary["category_counts"]["admission_induced"] == 1
    assert summary["category_counts"]["decode_growth_pressure"] == 1
    assert gate["status"] == "pass"
    assert "preempt_reentry_thrash" in summary["candidate_semantic_diffs"]
    assert phase451d.runtime_fix_gate(summary)["status"] == "blocked"
    assert (
        phase451d.runtime_fix_gate(summary)["value"]
        == "blocked_pending_unique_semantic_fix"
    )


def test_classification_gate_blocks_uncovered_noise() -> None:
    phase451d = _load_module()

    summary = phase451d.classify_preemption_events([
        {"trigger_state": "UNKNOWN", "victim_preemptions_before": 0, "victim_recently_admitted": False}
    ])
    gate = phase451d.classification_gate(summary, coverage_target=0.80, max_candidates=3)

    assert summary["coverage"] == 0.0
    assert gate["status"] == "blocked"


if __name__ == "__main__":
    test_classification_gate_requires_covered_events_and_small_candidate_set()
    test_classification_gate_blocks_uncovered_noise()
